"""Expose models from your Devin subscription to Hermes Agent via ACP.

Drives the official Devin CLI (``devin acp``) over stdio using a Devin-specific
ACP client subclass that approves tool permissions (Devin's ACP server provides
its own native tools — exec, read, edit — and asks permission before executing
them; the base ``CopilotACPClient`` cancels all permission requests, which was
correct for Copilot's tool-less ACP server but prevents Devin from running any
tool).

Session model: unlike the base class (which spawns a fresh CLI process and a
fresh ``session/new`` per request), this plugin keeps ONE ``devin acp`` process +
ACP session per Hermes conversation in a module-level registry keyed by the
Hermes session id. Devin already tracks conversation history in its own
session store (``$XDG_DATA_HOME/devin/cli/``), so each turn sends only the message
delta instead of the whole transcript — no per-turn process/MCP startup, no
session-per-request spam in ``devin``'s resume list, and Devin-side prompt
caching actually gets a warm prefix. If the Hermes-side history diverges from
what Devin saw (compaction, edits) the Devin session is rebuilt from the full
transcript; if the process died (idle reaper, crash, Hermes restart) the Devin
session is resumed via ``session/load`` so the conversation still continues in
the same Devin session row. Auth stays inside the Devin CLI (``devin auth
login``); when the ACP server demands an explicit ``authenticate`` (nested-ACP
environments where ``ACP_BACKEND`` is set) the plugin authenticates with the
CLI's own stored API key and never logs it.

This is a standalone plugin (installed via ``hermes plugins install paxytools/hermes-devin-acp``
into the Hermes home's ``plugins/`` directory) or pip-installed via the ``hermes_agent.plugins`` entry
point. It targets the ``process_command`` / ``process_args`` ProviderProfile contract and passes
the launch command/args explicitly via ``create_client()``, so no env-var setup is required.
"""

from __future__ import annotations

import atexit
import hashlib
import itertools
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

_log = logging.getLogger("plugins.devin_acp")

# ``devin models list`` prints model rows indented two spaces:
#   claude-opus-5-medium    Claude Opus 5 Medium  [1M context, $5 / 1M Input · ...]
_MODEL_LINE = re.compile(r"^\s{2}(\S+)\s{2,}\S")

_ACP_INITIALIZE_PARAMS = {
    "protocolVersion": 1,
    "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": True}},
    "clientInfo": {"name": "hermes-agent", "title": "Hermes Agent", "version": "0.0.0"},
}

# Preamble for a FRESH Devin session (replaces the base class's Copilot-oriented
# preamble, which instructs the model to emit <tool_call>{...}</tool_call> JSON
# blocks — wrong for Devin, which has native tools).
_DEVIN_PROMPT_PREAMBLE = (
    "You are being used as the active ACP agent backend for Hermes.",
    "Use your own tools and ACP capabilities to complete tasks.",
    "If no tool is needed, answer normally.",
)

# Env vars that mark "we are inside an ACP host". When ACP_BACKEND is set (e.g.
# Hermes launched from a terminal inside Windsurf/Devin), `devin acp` refuses
# session/new until the host calls `authenticate`; stripping it restores the
# normal local-credential path, with `authenticate` as a fallback.
_SPAWN_ENV_STRIP = {"ACP_BACKEND"}

# Idle `devin acp` processes are terminated after this many seconds. The Devin
# session row survives; the next prompt resumes it via session/load.
_IDLE_TTL_SECONDS = float(os.environ.get("HERMES_DEVIN_ACP_IDLE_SECONDS", "1800"))
_REAPER_INTERVAL_SECONDS = 60.0
_STATE_FILENAME = "devin_acp_sessions.json"
_AUTH_REQUIRED_RE = re.compile(r"not authenticated|not authenticated|authenticate", re.IGNORECASE)

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool", "context": "Context"}


class _AcpError(RuntimeError):
    """ACP JSON-RPC error carrying the server-provided code."""

    def __init__(self, message: str, *, code: Any = None):
        super().__init__(message)
        self.code = code


def _devin_credentials_api_key() -> str:
    """Read the Devin CLI's own API key from its credentials file.

    Only consulted when ``session/new`` fails demanding an explicit
    ``authenticate`` call (the nested-ACP path the CLI itself prescribes:
    "Call the `authenticate` ACP method with `meta.api_key` set to the user's
    API key"). The value is never logged or persisted by this plugin.
    """
    try:
        data_home = os.environ.get("XDG_DATA_HOME", "").strip() or os.path.join(
            os.path.expanduser("~"), ".local", "share")
        text = (Path(data_home) / "devin" / "credentials.toml").read_text(encoding="utf-8")
        match = re.search(r'windsurf_api_key\s*=\s*"([^"]+)"', text)
        return match.group(1) if match else ""
    except Exception:
        return ""


def _message_hash(message: dict) -> str:
    try:
        blob = json.dumps(message, sort_keys=True, default=str, ensure_ascii=True)
    except Exception:
        blob = repr(message)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _state_path() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / _STATE_FILENAME
    except Exception:
        return Path(os.path.expanduser("~")) / ".hermes" / _STATE_FILENAME


def _load_state() -> dict:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def _format_transcript(messages: list[dict], *, fresh: bool) -> str:
    """Render messages as a prompt. ``fresh`` adds the preamble (new Devin
    session); otherwise this is the delta for a session that already has the
    prefix in its own history."""
    from agent.copilot_acp_client import _render_message_content

    blocks: list[str] = []
    for message in (m for m in messages if isinstance(m, dict)):
        role = str(message.get("role") or "unknown").strip().lower()
        if rendered := _render_message_content(message.get("content")):
            blocks.append(f"{_ROLE_LABELS.get(role, 'Context')}:\n{rendered}")
    if not blocks:
        return ""
    transcript = "\n\n".join(blocks)
    if fresh:
        sections = [*_DEVIN_PROMPT_PREAMBLE, "Conversation transcript:\n\n" + transcript]
    else:
        sections = ["Conversation update:\n\n" + transcript]
    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


class _DevinSession:
    """One long-lived ``devin acp`` process + ACP session, shared by per-request clients.

    Lifecycle: ``ensure_ready`` spawns the process (initialize + lazy
    authenticate), ``open_session`` runs ``session/new`` — or ``session/load``
    to resume the persisted Devin session when the sent-prefix still matches —
    and ``prompt`` runs ``session/prompt``. ``sent_hashes`` is the fingerprint
    of the message prefix Devin has definitely seen; ``None`` means "unknown,
    resync with a fresh session on the next call".
    """

    def __init__(self, key: str | None, command: str, args: list, cwd: str):
        self.key = key
        self.command, self.args, self.cwd = command, list(args), cwd
        self.proc: subprocess.Popen | None = None
        self.inbox: queue.Queue = queue.Queue()
        self.stderr_tail: deque = deque(maxlen=40)
        self.request_ids = itertools.count(1)
        self.session_id = ""
        self.sent_hashes: list[str] | None = None
        self.applied_model: str | None = None
        self.last_used = time.monotonic()
        self.in_flight = False
        self.cancelled = threading.Event()
        self.lock = threading.Lock()        # serializes prompts on this session
        self._write_lock = threading.Lock()  # keeps stdin writes atomic across threads
        self._authenticated = False
        self._session_result: dict = {}

    # ---------- process plumbing ----------

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @staticmethod
    def _pump(stream, sink) -> None:
        for line in stream or ():
            sink(line)

    @staticmethod
    def _decode(line: str) -> dict:
        try:
            return json.loads(line)
        except Exception:
            return {"raw": line.rstrip("\n")}

    def _spawn(self) -> None:
        from agent.copilot_acp_client import _build_subprocess_env
        from hermes_cli._subprocess_compat import windows_hide_flags

        env = _build_subprocess_env()
        for key in _SPAWN_ENV_STRIP:
            env.pop(key, None)
        self.proc = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            cwd=self.cwd, env=env, creationflags=windows_hide_flags(),
        )
        self.stderr_tail.clear()
        threading.Thread(
            target=self._pump,
            args=(self.proc.stdout, lambda line: self.inbox.put(self._decode(line))),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._pump,
            args=(self.proc.stderr, lambda line: self.stderr_tail.append(line.rstrip("\n"))),
            daemon=True,
        ).start()

    def terminate(self, delete_session: bool = False) -> None:
        if delete_session and self.session_id and self.alive():
            # Ephemeral sessions (auxiliary calls) leave a permanent row in
            # Devin's session store otherwise — delete it before dying.
            try:
                self._request("session/delete", {"sessionId": self.session_id},
                              deadline=time.monotonic() + 2.0)
            except Exception:
                pass
        proc, self.proc = self.proc, None
        self.session_id = ""
        self.sent_hashes = None
        self.applied_model = None
        self._session_result = {}
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def cancel(self) -> None:
        """Abort the in-flight prompt: ``session/cancel`` + mark Devin-side state
        unknown so the next call resyncs from the full transcript."""
        self.cancelled.set()
        self.sent_hashes = None
        try:
            if self.alive() and self.session_id:
                self._write({"jsonrpc": "2.0", "method": "session/cancel",
                             "params": {"sessionId": self.session_id}})
        except Exception:
            pass

    # ---------- JSON-RPC ----------

    def _write(self, payload: dict) -> None:
        with self._write_lock:
            if self.proc is not None and self.proc.stdin is not None:
                self.proc.stdin.write(json.dumps(payload) + "\n")
                self.proc.stdin.flush()

    def _request(self, method: str, params: dict, *, deadline: float, dispatch=None,
                 interrupted=None) -> Any:
        request_id = next(self.request_ids)
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while time.monotonic() < deadline and self.alive() and not self.cancelled.is_set():
            if interrupted is not None and interrupted():
                # The host's abort path only force-closes TCP sockets — an ACP
                # client has none, so close() never runs mid-request. Poll the
                # agent's interrupt flag instead; terminate() cleans up after.
                self.cancelled.set()
                raise InterruptedError("Devin ACP prompt cancelled")
            try:
                msg = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if dispatch is not None and dispatch(msg):
                continue
            if msg.get("id") != request_id:
                continue
            if "error" in msg:
                err = msg.get("error") or {}
                raise _AcpError(str(err.get("message") or err), code=err.get("code"))
            return msg.get("result")
        if self.cancelled.is_set():
            raise InterruptedError("Devin ACP prompt cancelled")
        stderr_text = "\n".join(self.stderr_tail).strip()
        if not self.alive() and stderr_text:
            raise _AcpError(f"Devin ACP process exited early: {stderr_text}")
        if not self.alive():
            raise _AcpError("Devin ACP process exited unexpectedly.")
        raise TimeoutError(f"Timed out waiting for Devin ACP response to {method}.")

    def _drain_stale(self) -> None:
        """Drop notifications that arrived between prompts (MCP banners, replay
        tails from session/load). Server *requests* only occur mid-prompt, so
        nothing here needs an answer."""
        while True:
            try:
                self.inbox.get_nowait()
            except queue.Empty:
                return

    # ---------- session lifecycle ----------

    def _authenticate(self, *, deadline: float, dispatch) -> bool:
        api_key = _devin_credentials_api_key()
        if not api_key or self._authenticated:
            return False
        try:
            self._request(
                "authenticate",
                {"methodId": "devin-browser", "_meta": {"api_key": api_key}},
                deadline=deadline, dispatch=dispatch)
            self._authenticated = True
            return True
        except Exception as exc:
            _log.warning("Devin ACP authenticate failed: %s", exc)
            return False

    def ensure_ready(self, *, deadline: float, dispatch) -> None:
        """Process alive + initialized. Does NOT open an ACP session."""
        if self.alive():
            return
        self.cancelled.clear()
        self._spawn()
        self._authenticated = False
        self._request("initialize", _ACP_INITIALIZE_PARAMS, deadline=deadline, dispatch=dispatch)
        self.session_id = ""
        self.sent_hashes = None
        self.applied_model = None
        self._session_result = {}

    def _open_with_auth_retry(self, method: str, params: dict, *, deadline: float, dispatch) -> dict:
        try:
            return self._request(method, params, deadline=deadline, dispatch=dispatch) or {}
        except _AcpError as exc:
            if not _AUTH_REQUIRED_RE.search(str(exc)):
                raise
            if not self._authenticate(deadline=deadline, dispatch=dispatch):
                raise
            return self._request(method, params, deadline=deadline, dispatch=dispatch) or {}

    def open_session(self, *, model: str | None, hashes: list[str], deadline: float,
                     dispatch, persisted: dict | None) -> None:
        """Open (or resume) the Devin session and select ``model``.

        ``session/load`` is attempted when the persisted Devin session's sent
        prefix still matches the current message prefix — i.e. the process died
        but the conversation continued unchanged. Otherwise a fresh
        ``session/new`` starts a new Devin session.
        """
        persisted = persisted or {}
        p_hashes = persisted.get("hashes") or []
        p_sid = str(persisted.get("devin_sid") or "")
        can_resume = (
            bool(p_sid) and persisted.get("cwd") == self.cwd
            and len(hashes) >= len(p_hashes) and hashes[:len(p_hashes)] == p_hashes
        )
        session: dict = {}
        if self.session_id:
            # History diverged from a live session — close the stale Devin
            # session before opening a fresh one (keeps devin's own store tidy).
            try:
                self._write({"jsonrpc": "2.0", "method": "session/close",
                             "params": {"sessionId": self.session_id}})
            except Exception:
                pass
            self.session_id = ""
            self.sent_hashes = None
            self.applied_model = None
        if can_resume:
            try:
                session = self._open_with_auth_retry(
                    "session/load",
                    {"sessionId": p_sid, "cwd": self.cwd, "mcpServers": []},
                    deadline=deadline, dispatch=dispatch)
                self.session_id = p_sid
                self.sent_hashes = list(p_hashes)
                _log.info("Devin ACP resumed session %s for %s", p_sid, self.key)
            except Exception as exc:
                _log.info("Devin ACP session/load %s failed (%s); starting fresh.", p_sid, exc)
                session = {}
        if not self.session_id:
            session = self._open_with_auth_retry(
                "session/new", {"cwd": self.cwd, "mcpServers": []},
                deadline=deadline, dispatch=dispatch)
            self.session_id = str(session.get("sessionId") or "").strip()
            if not self.session_id:
                raise _AcpError("Devin ACP did not return a sessionId.")
            self.sent_hashes = []
        self._session_result = session
        self._apply_model(session, model, deadline=deadline, dispatch=dispatch)

    def _apply_model(self, session: dict, model: str | None, *, deadline: float, dispatch) -> None:
        requested = str(model or "").strip()
        if not requested or requested == "devin-acp" or requested == self.applied_model:
            return
        try:
            from agent.copilot_acp_client import _model_selection_request

            if (selection := _model_selection_request(session, requested)) is not None:
                self._request(selection[0], selection[1], deadline=deadline, dispatch=dispatch)
                self.applied_model = requested
            else:
                _log.warning("Devin ACP does not offer model %r; using the session default.", requested)
        except Exception as exc:
            _log.warning("Devin ACP model selection for %r failed; continuing: %s", requested, exc)

    def prompt(self, prompt_text: str, *, deadline: float, dispatch, interrupted=None) -> None:
        """Send ``session/prompt``; the dispatch callable consumes updates and
        collects text/reasoning parts until the response arrives."""
        self._drain_stale()
        self.cancelled.clear()
        self.in_flight = True
        try:
            self._request(
                "session/prompt",
                {"sessionId": self.session_id,
                 "prompt": [{"type": "text", "text": prompt_text}]},
                deadline=deadline, dispatch=dispatch, interrupted=interrupted)
        finally:
            self.in_flight = False
            self.last_used = time.monotonic()


# ---------------- session registry ----------------
#
# Per-request clients are created and closed on every API call, so the durable
# anchor is the Hermes session id: each conversation owns one _DevinSession.
# `key=None` sessions are ephemeral (auxiliary calls with no conversation).

_SESSIONS: dict[str, _DevinSession] = {}
_SESSIONS_LOCK = threading.Lock()
_PERSISTED: dict | None = None


def _persisted_state() -> dict:
    global _PERSISTED
    with _SESSIONS_LOCK:
        if _PERSISTED is None:
            raw = _load_state()
            # Prune stale entries (>30 days).
            cutoff = time.time() - 30 * 86400
            _PERSISTED = {
                k: v for k, v in raw.items()
                if isinstance(v, dict) and float(v.get("updated") or 0) > cutoff
            }
        return _PERSISTED


def _persist_entry(sess: _DevinSession) -> None:
    if not sess.key or not sess.session_id or not sess.sent_hashes:
        return
    state = _persisted_state()
    # serve and gateway are separate processes each caching _PERSISTED; merge
    # the on-disk state first so a stale cache can't drop the other's entry.
    for k, v in _load_state().items():
        if isinstance(v, dict):
            state.setdefault(k, v)
    state[sess.key] = {
        "devin_sid": sess.session_id,
        "hashes": list(sess.sent_hashes),
        "cwd": sess.cwd,
        "updated": time.time(),
    }
    _save_state(state)


def _session_for(key: str, command: str, args: list, cwd: str) -> _DevinSession:
    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(key)
        if sess is not None and (sess.command, sess.args, sess.cwd) != (command, list(args), cwd):
            sess.terminate()
            sess = None
        if sess is None:
            sess = _DevinSession(key, command, args, cwd)
            _SESSIONS[key] = sess
        return sess


def _idle_reaper() -> None:
    while True:
        time.sleep(_REAPER_INTERVAL_SECONDS)
        try:
            with _SESSIONS_LOCK:
                sessions = list(_SESSIONS.values())
            for sess in sessions:
                if (sess.alive() and not sess.in_flight
                        and time.monotonic() - sess.last_used > _IDLE_TTL_SECONDS):
                    _log.info("Devin ACP reaping idle session for %s", sess.key)
                    sess.terminate()
        except Exception:
            pass


threading.Thread(target=_idle_reaper, daemon=True, name="devin-acp-reaper").start()


def _atexit_cleanup() -> None:
    with _SESSIONS_LOCK:
        sessions = list(_SESSIONS.values())
    for sess in sessions:
        try:
            sess.terminate()
        except Exception:
            pass


atexit.register(_atexit_cleanup)


class DevinACPClient(Any):  # type: ignore[misc]
    """Devin-specific ACP stdio client.

    Subclasses ``CopilotACPClient`` and overrides ``_handle_server_message`` to
    approve ``session/request_permission`` requests instead of cancelling them.
    Devin's ACP server provides native tools (exec, read, edit) and asks
    permission before executing; the base class cancels all permission requests
    (correct for Copilot's tool-less ACP server, but prevents Devin from doing
    any real work). This subclass approves them so the model can use its tools.
    """

    def __new__(cls, **kwargs: Any) -> Any:
        from agent.copilot_acp_client import CopilotACPClient

        class _DevinClient(CopilotACPClient):
            def __init__(self, **kw):
                super().__init__(**kw)
                self._devin_tool_seq = 0
                self._devin_active_tools: dict = {}
                self._devin_think_buffer = ""
                self._devin_session = None

            def _devin_agent(self):
                """Resolve the current turn's agent via the runtime session ContextVar.

                The ContextVar is set on the turn thread (``_run_prompt_submit``'s
                ``run()`` function) and propagated via ``copy_context()`` to worker
                threads. Title generation does NOT set it, so background calls are
                naturally excluded. Returns ``None`` when no main turn is active.
                """
                try:
                    from tui_gateway.server import _current_runtime_session_record
                    session = _current_runtime_session_record.get()
                    if session is None:
                        return None
                    return session.get("agent")
                except Exception:
                    return None

            def _devin_sid(self):
                """Resolve the current turn's UI session ID via the runtime session ContextVar."""
                try:
                    from tui_gateway.server import _current_runtime_session_record, _sessions
                    session = _current_runtime_session_record.get()
                    if session is None:
                        return ""
                    for sid, sess in _sessions.items():
                        if sess is session:
                            return sid
                    return ""
                except Exception:
                    return ""

            def _devin_persist(self, agent, content, display_kind, *,
                              role="assistant", tool_calls=None,
                              tool_name=None, tool_call_id=None,
                              reasoning=None):
                """Write a display-only DB row using the proper append_message API,
                then deactivate it (active=0, compacted=1) so it shows in the
                display projection but isn't fed to the model.

                append_message handles write guards, counter bumps, and encoding;
                the follow-up UPDATE just flips active/compacted on our row.
                """
                try:
                    db = getattr(agent, "_session_db", None)
                    sid = getattr(agent, "session_id", None)
                    if db is None or not sid:
                        return
                    # Parse tool_calls JSON to the list format append_message expects.
                    tc_list = None
                    if tool_calls:
                        import json as _json
                        try:
                            tc_list = _json.loads(tool_calls)
                        except Exception:
                            tc_list = None
                    row_id = db.append_message(
                        sid, role, content=content or None,
                        tool_calls=tc_list, tool_name=tool_name,
                        tool_call_id=tool_call_id, reasoning=reasoning,
                        display_kind=display_kind)
                    # Deactivate: active=0, compacted=1 → display-only, not fed to model.
                    if row_id:
                        db._execute_write(
                            lambda conn: conn.execute(
                                "UPDATE messages SET active = 0, compacted = 1 WHERE id = ?",
                                (row_id,)))
                except Exception:
                    pass

            def _devin_flush_thinking(self, agent):
                """Flush accumulated thinking as its own persisted DB row, then clear the buffer.

                Writing to the ``reasoning`` column (not ``content``) makes the frontend
                render it as a thought/reasoning block, not a plain text bubble.
                """
                think_text = getattr(self, "_devin_think_buffer", "") or ""
                if think_text:
                    self._devin_persist(agent, "", "devin_thinking", reasoning=think_text)
                    self._devin_think_buffer = ""

            def _devin_session_key(self, agent):
                """Stable registry key for the shared Devin session, or None.

                Desktop/TUI turns resolve the agent via the runtime-session
                ContextVar; messaging-gateway/cron turns expose the durable
                HERMES_SESSION_ID ContextVar (propagated to request workers via
                copy_context). Auxiliary calls (title gen, probes) have neither
                → ephemeral, spawn-per-call behavior.
                """
                try:
                    from agent.auxiliary_client import _RELAY_AUX_CALL_CONTEXT

                    if _RELAY_AUX_CALL_CONTEXT.get() is not None:
                        # Auxiliary LLM call (approval guardian, title gen,
                        # compression) inside a turn. It must NOT share the
                        # conversation's Devin session: the guardian runs inside
                        # the permission dispatch — mid-prompt on the SAME thread
                        # — so the shared session's lock is already held by us
                        # (hard self-deadlock), and an aux transcript would churn
                        # the Devin session's hash prefix anyway.
                        return None
                except Exception:
                    pass
                if threading.current_thread().name == "bg-review":
                    # Background memory/skill review fork. On the gateway path its
                    # __init__ publishes a fresh throwaway HERMES_SESSION_ID (a new
                    # key → new Devin session per review); on the desktop path it
                    # resolves the PARENT agent → its transcript would delta into
                    # the live conversation's Devin session. Either way: ephemeral.
                    return None
                try:
                    from agent.delegation_context import is_delegated_child_context

                    if is_delegated_child_context():
                        # Delegated subagent: isolated one-shot work; keying to the
                        # parent session would splice its transcript into the
                        # parent's Devin history.
                        return None
                except Exception:
                    pass
                sid = str(getattr(agent, "session_id", "") or "") if agent is not None else ""
                if sid:
                    return sid
                try:
                    from gateway.session_context import get_session_env

                    return (get_session_env("HERMES_SESSION_ID", "")
                            or get_session_env("HERMES_SESSION_KEY", "")
                            or None)
                except Exception:
                    return None

            def _devin_dispatch(self, sess, text_parts, reasoning_parts, *, record_updates):
                """Bind this client's message handler for a _DevinSession request loop."""
                def _dispatch(msg):
                    return self._handle_server_message(
                        msg, process=sess.proc, cwd=sess.cwd,
                        text_parts=text_parts, reasoning_parts=reasoning_parts,
                        allow_file_requests=True, record_updates=record_updates,
                        writer=sess._write)
                return _dispatch

            def _handle_server_message(self, msg, *, process, cwd, text_parts, reasoning_parts,
                                       allow_file_requests=True, record_updates=True, writer=None):
                method = msg.get("method", "")
                if method == "session/update" and not record_updates:
                    # session/load replays history as session/update notifications;
                    # they were persisted when they first streamed — swallow them so
                    # they don't duplicate display rows or pollute text_parts.
                    return True
                if method and msg.get("id") is None and method != "session/update":
                    # A notification (no id): Devin emits _cognition.ai/* chatter
                    # (output banners, mcp/serversChanged). The base class would
                    # answer with a spurious {"id": null} error — swallow instead.
                    return True
                if method == "session/request_permission":
                    import json
                    message_id = msg.get("id")
                    # ACP spec: AllowedOutcome has outcome="selected" + optionId.
                    # The base class sends {"outcome": {"outcome": "cancelled"}} (DeniedOutcome).
                    # We approve so Devin can run its native tools (exec, read, edit).
                    params = msg.get("params") or {}
                    options = params.get("options") or []
                    # Extract the actual command from Devin's permission request.
                    # Devin sends the real command in toolCall._meta.cognition.ai/editableCommand.
                    tool_call = params.get("toolCall") or {}
                    meta = tool_call.get("_meta") or {}
                    perm_desc = (
                        meta.get("cognition.ai/editableCommand")
                        or params.get("description")
                        or ""
                    )
                    if not perm_desc:
                        for opt in options:
                            if isinstance(opt, dict) and opt.get("name"):
                                perm_desc = opt["name"]
                                break
                        if not perm_desc:
                            perm_desc = "Devin requests permission to run a tool"
                    # option_id stays None when the gate denies/fails → we send reject_once.
                    # It's only set to "allow_once" when the gate explicitly approves.
                    option_id = None
                    in_aux_call = False
                    try:
                        from agent.auxiliary_client import _RELAY_AUX_CALL_CONTEXT
                        in_aux_call = _RELAY_AUX_CALL_CONTEXT.get() is not None
                    except Exception:
                        pass
                    if in_aux_call:
                        # Permission request inside an auxiliary (ephemeral) session —
                        # e.g. the smart-approval guardian's own Devin call. Routing it
                        # back through the gate would invoke the guardian recursively
                        # (one process per level). Auxiliary tasks are Q&A-only; fail
                        # closed so Devin proceeds without tools.
                        response = {"jsonrpc": "2.0", "id": message_id,
                                    "result": {"outcome": {"outcome": "cancelled"}}}
                        if writer is not None:
                            writer(response)
                        elif process.stdin is not None:
                            process.stdin.write(json.dumps(response) + "\n")
                            process.stdin.flush()
                        return True
                    # Pass through Hermes' built-in approval gate. The ACP handler thread
                    # isn't recognized as interactive by default, so we set the interactive
                    # contextvar first when a user is watching (main turn active). Then
                    # check_all_command_guards handles everything: dangerous detection,
                    # CLI prompt, gateway push (Telegram/Discord), TUI/desktop clarify,
                    # and cron/kanban/unattended config. Safe commands pass through in
                    # ALL contexts; only dangerous ones get gated.
                    try:
                        from tools.approval_context import (
                            set_hermes_interactive_context, reset_hermes_interactive_context,
                        )
                        from tools.approval import check_all_command_guards
                        # Mark interactive only when a user is watching (main turn active).
                        # Background tasks (title gen, cron, kanban) stay non-interactive
                        # so the gate applies cron_mode/single_query_mode/unattended_mode.
                        _interactive = self._devin_agent() is not None
                        _token = set_hermes_interactive_context(_interactive)
                        try:
                            result = check_all_command_guards(perm_desc, env_type="local")
                        finally:
                            reset_hermes_interactive_context(_token)
                        if result.get("approved"):
                            option_id = "allow_once"
                    except Exception:
                        # Fail closed: do NOT approve on exception.
                        option_id = None
                    # Only verify the option if the gate approved. If the gate denied or
                    # threw, option_id stays None and we select reject_once to tell Devin
                    # to stop retrying (cancelled alone causes Devin to retry the command).
                    if option_id is not None:
                        offered = {opt.get("optionId", "") for opt in options if isinstance(opt, dict)}
                        if offered and option_id not in offered:
                            for fallback in ("allow_once", "allow_session", "allow_always"):
                                if fallback in offered:
                                    option_id = fallback
                                    break
                            else:
                                # allow_once not offered → deny rather than grant permanent.
                                option_id = None
                    if option_id:
                        response = {"jsonrpc": "2.0", "id": message_id,
                                    "result": {"outcome": {"outcome": "selected", "optionId": option_id}}}
                    else:
                        # Gate denied or threw: prefer reject_once (tells Devin to stop),
                        # fall back to cancelled if reject_once isn't offered.
                        offered = {opt.get("optionId", "") for opt in options if isinstance(opt, dict)}
                        if "reject_once" in offered:
                            response = {"jsonrpc": "2.0", "id": message_id,
                                        "result": {"outcome": {"outcome": "selected", "optionId": "reject_once"}}}
                        else:
                            response = {"jsonrpc": "2.0", "id": message_id,
                                        "result": {"outcome": {"outcome": "cancelled"}}}
                    if writer is not None:
                        writer(response)
                    elif process.stdin is not None:
                        process.stdin.write(json.dumps(response) + "\n")
                        process.stdin.flush()
                    return True
                if method == "session/update":
                    update = (msg.get("params") or {}).get("update") or {}
                    utype = update.get("sessionUpdate", "")
                    agent = self._devin_agent()
                    if agent is not None:
                        if utype == "agent_thought_chunk":
                            # Stream thinking live via the agent's reasoning-delta
                            # callback (same path codex uses). Also accumulate so we
                            # can persist it as its own DB row when the next phase starts.
                            content = update.get("content") or {}
                            text = content.get("text", "") if isinstance(content, dict) else ""
                            if text:
                                self._devin_think_buffer = (getattr(self, "_devin_think_buffer", "") or "") + text
                                try:
                                    agent._fire_reasoning_delta(text)
                                except Exception:
                                    pass
                        elif utype == "agent_message_chunk":
                            # The assistant's final response text is arriving.
                            # Flush accumulated thinking as its own persisted DB row.
                            self._devin_flush_thinking(agent)
                        elif utype == "tool_call":
                            # Flush any pending thinking first, then persist the tool call.
                            self._devin_flush_thinking(agent)
                            try:
                                tool_call_id = update.get("toolCallId", "")
                                title = update.get("title", "Devin tool")
                                kind = update.get("kind", "")
                                raw_input = {}
                                content_list = update.get("content") or []
                                if isinstance(content_list, list) and content_list:
                                    for item in content_list:
                                        if isinstance(item, dict):
                                            inner = item.get("content", {})
                                            if isinstance(inner, dict):
                                                if inner.get("type") == "resource":
                                                    res = inner.get("resource", {})
                                                    cmd_text = res.get("text", "")
                                                    if cmd_text:
                                                        raw_input = {"command": cmd_text} if kind == "execute" else {"content": cmd_text}
                                                        break
                                                elif inner.get("type") == "text":
                                                    txt = inner.get("text", "")
                                                    if txt:
                                                        raw_input = {"command": txt} if kind == "execute" else {"content": txt}
                                                        break
                                tool_name = {"read": "read_file", "execute": "terminal",
                                             "edit": "write_file"}.get(kind, kind or "devin_tool")
                                self._devin_tool_seq += 1
                                tc_id = tool_call_id or f"devin:{self._devin_tool_seq}"
                                self._devin_active_tools[tc_id] = {
                                    "name": tool_name, "args": raw_input, "title": title}
                                # Persist the tool call as a structured DB row with tool_calls JSON
                                # so the frontend renders it as a proper tool card.
                                import json as _json
                                tc_json = [{"id": tc_id, "type": "function",
                                            "function": {"name": tool_name,
                                                         "arguments": _json.dumps(raw_input)}}]
                                self._devin_persist(agent, "", "devin_tool_call",
                                                    tool_calls=_json.dumps(tc_json))
                                try:
                                    agent.tool_start_callback(tc_id, tool_name, raw_input)
                                except Exception:
                                    pass
                            except Exception as e:
                                _log.error("devin_tool: tool_call handler failed: %s", e, exc_info=True)
                        elif utype == "tool_call_update":
                            status = update.get("status", "")
                            tool_call_id = update.get("toolCallId", "")
                            if status in ("completed", "failed"):
                                try:
                                    tc_id = tool_call_id or f"devin:{self._devin_tool_seq}"
                                    tool_info = self._devin_active_tools.pop(tc_id, {})
                                    tool_name = tool_info.get("name", "devin_tool")
                                    tool_args = tool_info.get("args", {})
                                    content = update.get("content") or []
                                    result_text = ""
                                    if isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict):
                                                inner = item.get("content", {})
                                                if isinstance(inner, dict):
                                                    result_text = inner.get("text", "")
                                                    if result_text:
                                                        break
                                    if status == "failed":
                                        result = {"error": result_text or "Tool execution failed"}
                                    else:
                                        result = {"result": result_text or "OK"}
                                    # Persist the tool result as a tool-role DB row so the
                                    # frontend matches it to the tool-call card.
                                    self._devin_persist(agent, result_text or "OK", "devin_tool_result",
                                                        role="tool", tool_name=tool_name,
                                                        tool_call_id=tc_id)
                                    try:
                                        agent.tool_complete_callback(tc_id, tool_name, tool_args, result)
                                    except Exception:
                                        pass
                                except Exception as e:
                                    _log.error("devin_tool: tool_call_update handler failed: %s", e, exc_info=True)
                return super()._handle_server_message(
                    msg, process=process, cwd=cwd, text_parts=text_parts,
                    reasoning_parts=reasoning_parts, allow_file_requests=allow_file_requests)

            def _create_chat_completion(self, *, model=None, messages=None, timeout=None,
                                         tools=None, tool_choice=None, stream=False, **_):
                # Devin's ACP server provides its OWN native tools (exec, read, edit).
                # The base class injects Hermes' tool schemas into the prompt text and
                # instructs the model to emit tool calls as {...} JSON blocks -- which
                # conflicts with Devin's native tool-call mechanism and confuses the
                # model. We strip Hermes' tools so the model uses Devin's native tools
                # and returns results as text.
                from agent.copilot_acp_client import _effective_timeout
                from agent.acp_openai_bridge import completion_to_stream_chunks as _completion_to_stream_chunks
                from types import SimpleNamespace
                self._devin_tool_seq = 0
                self._devin_active_tools = {}
                messages = messages or []
                timeout_seconds = _effective_timeout(timeout)
                agent = self._devin_agent()
                response_text, _reasoning = self._devin_complete(
                    messages, model=model, timeout_seconds=timeout_seconds, agent=agent)
                # Flush any remaining thinking as its own persisted DB row.
                if agent is not None:
                    self._devin_flush_thinking(agent)
                # Don't pass reasoning back — we already persisted it as its own
                # row above. Returning it here would duplicate it on the final
                # answer bubble (the conversation loop writes it to msg["reasoning"]).
                message = SimpleNamespace(
                    content=response_text, tool_calls=[], reasoning=None,
                    reasoning_content=None, reasoning_details=None)
                completion = SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason="stop")],
                    usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0,
                                          total_tokens=0,
                                          prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
                    model=model or "devin-acp")
                return _completion_to_stream_chunks(completion) if stream else completion

            def _devin_complete(self, messages, *, model, timeout_seconds, agent):
                """Run one prompt over the shared per-conversation Devin session.

                Sends only the message DELTA when Devin's recorded prefix still
                matches (hashes), else resyncs from the full transcript. Returns
                ``(text_parts_joined, reasoning_parts_joined)``.
                """
                key = self._devin_session_key(agent)
                deadline = time.monotonic() + timeout_seconds
                if key is None:
                    # No stable conversation anchor (auxiliary call) — ephemeral
                    # session, spawn-prompt-kill like the base class.
                    return self._devin_ephemeral(messages, model=model, deadline=deadline)
                sess = _session_for(key, self._acp_command, self._acp_args, self._acp_cwd)
                self._devin_session = sess
                if getattr(sess, "owner_tid", None) == threading.get_ident():
                    # Reentrant call on the thread already holding sess.lock —
                    # a non-reentrant acquire here would self-deadlock (e.g. an
                    # unmarked nested LLM call inside permission dispatch).
                    return self._devin_ephemeral(messages, model=model, deadline=deadline)
                hashes = [_message_hash(m) for m in messages]
                def _prefix_match(sent):
                    return (sent is not None and len(hashes) >= len(sent)
                            and hashes[:len(sent)] == sent)

                with sess.lock:
                    sess.owner_tid = threading.get_ident()
                    try:
                        setup_dispatch = self._devin_dispatch(sess, None, None, record_updates=False)
                        sess.ensure_ready(deadline=deadline, dispatch=setup_dispatch)
                        sent = sess.sent_hashes
                        if not (sess.session_id and _prefix_match(sent)):
                            # Unknown or diverged history — (re)open: session/load
                            # resumes the persisted session when its sent prefix still
                            # matches, else session/new starts a fresh Devin session.
                            sess.open_session(model=model, hashes=hashes, deadline=deadline,
                                              dispatch=setup_dispatch,
                                              persisted=_persisted_state().get(key))
                            sent = sess.sent_hashes
                        if sent and _prefix_match(sent):
                            # Devin has this exact prefix — send only the delta.
                            # Assistant messages in the tail are Devin's own prior
                            # replies (already in its history); forward only new
                            # user/tool/context messages.
                            delta = [m for m in messages[len(sent):]
                                     if not (isinstance(m, dict) and m.get("role") == "assistant")]
                            prompt_text = _format_transcript(delta, fresh=False)
                        else:
                            # Fresh Devin session (or reopened but empty) — full transcript.
                            prompt_text = _format_transcript(messages, fresh=True)
                        if not prompt_text:
                            prompt_text = "Continue the conversation from the latest user request."
                        # Model may have changed mid-conversation on the same session.
                        sess._apply_model(sess._session_result, model,
                                          deadline=deadline, dispatch=setup_dispatch)
                        text_parts: list = []
                        reasoning_parts: list = []
                        interrupted = (lambda: bool(getattr(agent, "_interrupt_requested", False))
                                       if agent is not None else None)
                        try:
                            sess.prompt(
                                prompt_text, deadline=deadline,
                                dispatch=self._devin_dispatch(
                                    sess, text_parts, reasoning_parts, record_updates=True),
                                interrupted=interrupted)
                        except Exception:
                            # Outcome unknown — Devin may have consumed the prompt. Drop the
                            # session so the next call resyncs from the full transcript.
                            sess.terminate()
                            raise
                        sess.sent_hashes = hashes
                        _persist_entry(sess)
                    finally:
                        sess.owner_tid = None
                return "".join(text_parts), "".join(reasoning_parts)

            def _devin_ephemeral(self, messages, *, model, deadline):
                sess = _DevinSession(None, self._acp_command, self._acp_args, self._acp_cwd)
                self._devin_session = sess
                try:
                    setup_dispatch = self._devin_dispatch(sess, None, None, record_updates=False)
                    sess.ensure_ready(deadline=deadline, dispatch=setup_dispatch)
                    sess.open_session(model=model, hashes=[], deadline=deadline,
                                      dispatch=setup_dispatch, persisted=None)
                    prompt_text = _format_transcript(messages, fresh=True) or (
                        "Continue the conversation from the latest user request.")
                    text_parts: list = []
                    reasoning_parts: list = []
                    interrupted = (lambda: bool(getattr(agent, "_interrupt_requested", False))
                                   if (agent := self._devin_agent()) is not None else None)
                    sess.prompt(
                        prompt_text, deadline=deadline,
                        dispatch=self._devin_dispatch(
                            sess, text_parts, reasoning_parts, record_updates=True),
                        interrupted=interrupted)
                    return "".join(text_parts), "".join(reasoning_parts)
                finally:
                    sess.terminate(delete_session=True)

            def close(self) -> None:
                # Per-request clients are evicted after every call; the shared
                # _DevinSession must survive that. Only an in-flight prompt is
                # cancelled (the real abort path), never a healthy idle session.
                self.is_closed = True
                sess = self._devin_session
                if sess is not None and sess.in_flight:
                    sess.cancel()
                try:
                    super().close()  # cleans up _active_process probes (list_models)
                except Exception:
                    pass

        return _DevinClient(**kwargs)


class DevinACPProfile(ProviderProfile):
    """Devin CLI models using your authenticated subscription, over ACP stdio."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Build the Devin ACP stdio shim, passing the launch command/args explicitly.

        ``client_kwargs`` (from the core) carries ``api_key``/`base_url``/etc.
        but NOT ``command``/``args`` -- those are resolved from env vars inside
        ``CopilotACPClient`` by default. We inject them from the profile fields
        so no ``HERMES_*`` env var is needed for a standard install.
        """
        client_kwargs.setdefault("acp_command", self.process_command)
        client_kwargs.setdefault("acp_args", list(self.process_args))
        # Use the user's home directory as the ACP session cwd so Devin's native
        # tools (read, exec, edit) can access the Hermes home dir and other
        # project dirs. The base class defaults to os.getcwd(), which is too
        # narrow when Hermes runs from its install tree.
        client_kwargs.setdefault("acp_cwd", os.path.expanduser("~"))
        return DevinACPClient(**client_kwargs)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Discover the model catalog from the authenticated Devin CLI."""
        command = shutil.which(self.process_command)
        if not command:
            return None
        try:
            completed = subprocess.run(
                [command, "models", "list"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        models: list[str] = []
        for line in completed.stdout.splitlines():
            match = _MODEL_LINE.match(line)
            if match:
                models.append(match.group(1))
        return list(dict.fromkeys(models)) or None


devin_acp = DevinACPProfile(
    name="devin-acp",
    aliases=("devin", "devin-subscription"),
    display_name="Devin ACP",
    description="Devin CLI models using your authenticated subscription",
    signup_url="https://app.devin.ai/",
    api_mode="chat_completions",
    env_vars=(),  # Auth lives inside the Devin CLI, not in env vars.
    base_url="acp://devin",
    auth_type="external_process",
    supports_health_check=False,
    process_command="devin",
    process_args=("acp",),
    process_command_env_vars=("HERMES_DEVIN_ACP_COMMAND",),
    process_args_env_var="HERMES_DEVIN_ACP_ARGS",
    fallback_models=("opus", "sonnet", "gpt", "gemini", "swe"),
)

register_provider(devin_acp)


# Register a dedicated catalog fetcher so the /model picker populates Devin models.
# The generic ``_profile_live_catalog`` path in hermes_cli/models.py requires
# ``auth_type == "api_key"`` and skips ``external_process`` providers, so without
# this entry the picker shows zero models even though ``fetch_models()`` works.
# This is the same runtime-registry-extension pattern as ``register_provider``
# (appending to a module-level dict at import time); it does NOT edit core files.
def _devin_catalog(normalized: str, force_refresh: bool) -> list[str] | None:
    return devin_acp.fetch_models() or list(devin_acp.fallback_models) or None


try:
    from hermes_cli.models import _PROVIDER_CATALOG_FETCHERS

    _PROVIDER_CATALOG_FETCHERS["devin-acp"] = _devin_catalog
except Exception:
    pass  # hermes_cli.models not yet importable in this context; non-fatal.

# The canonical provider list (models_catalog_static.CANONICAL_PROVIDERS) feeds the
# /model picker's provider menu. Its auto-extension loop skips external_process
# providers ("Non-api-key flows need bespoke picker UX"), so an ACP provider like
# devin-acp never appears unless explicitly appended -- copilot-acp only shows because
# it is hardcoded in the list. We append devin-acp at import time (same runtime-
# mutation pattern as the catalog fetcher above); this does NOT edit core files.
try:
    from hermes_cli.models_catalog_static import CANONICAL_PROVIDERS, ProviderEntry

    if not any(p.slug == "devin-acp" for p in CANONICAL_PROVIDERS):
        CANONICAL_PROVIDERS.append(ProviderEntry(
            "devin-acp", "Devin ACP",
            "Devin ACP (Spawns devin acp --stdio, uses your Devin CLI login)",
        ))
except Exception:
    pass  # non-fatal; /model <id> --provider devin-acp still works without this.

# The /model picker's credential check has an external_process branch ONLY in the overlay
# section (_overlay_has_creds), not in the canonical-rows section. copilot-acp appears in the
# picker because it has a HERMES_OVERLAYS entry; without one, devin-acp falls through to
# _lap_canonical_rows which has no external_process branch and filters it out as "no creds".
# We add the overlay at import time (same runtime-mutation pattern as the registrations
# above); this does NOT edit core files.
try:
    from hermes_cli.providers import HERMES_OVERLAYS, HermesOverlay

    if "devin-acp" not in HERMES_OVERLAYS:
        HERMES_OVERLAYS["devin-acp"] = HermesOverlay(
            auth_type="external_process", base_url_override="acp://devin")
except Exception:
    pass  # non-fatal; the provider still works via --provider devin-acp.

# The dashboard Accounts tab renders a sign-in command per provider. For providers not in the
# curated _OAUTH_PROVIDER_CATALOG it auto-generates ``hermes auth add <slug>`` and then
# displays "Sign in with Devin CLI". For external_process providers we provide a
# human-readable sign-in hint instead of the auto-generated ``hermes auth add``
# command, since auth happens inside the Devin CLI.
try:
    from hermes_cli.providers import _OAUTH_PROVIDER_CATALOG

    if "devin-acp" not in _OAUTH_PROVIDER_CATALOG:
        _OAUTH_PROVIDER_CATALOG["devin-acp"] = {
            "signin_hint": "Run `devin auth login` in your terminal to authenticate.",
            "signin_url": "https://app.devin.ai/",
        }
except Exception:
    pass  # non-fatal; the dashboard still works without this entry.
