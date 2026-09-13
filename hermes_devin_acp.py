"""Devin subscription provider for Hermes Agent via ACP.

Drives the official Devin CLI (``devin acp``) over stdio using a Devin-specific
ACP client subclass that approves tool permissions (Devin's ACP server provides
its own native tools — exec, read, edit — and asks permission before executing
them; the base ``CopilotACPClient`` cancels all permission requests, which was
correct for Copilot's tool-less ACP server but prevents Devin from running any
tool). Auth stays inside the Devin CLI (``devin auth login``); this plugin never
reads credentials.

This is a standalone plugin (installed via ``hermes plugins install paxytools/hermes-devin-acp``
into ``~/.hermes/plugins/devin-acp/``) or pip-installed via the ``hermes_agent.plugins`` entry
point. It targets the ``process_command`` / ``process_args`` ProviderProfile contract and passes
the launch command/args explicitly via ``create_client()``, so no env-var setup is required.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

_log = logging.getLogger("plugins.devin_acp")

# ``devin models list`` prints model rows indented two spaces:
#   claude-opus-5-medium    Claude Opus 5 Medium  [1M context, $5 / 1M Input · ...]
_MODEL_LINE = re.compile(r"^\s{2}(\S+)\s{2,}\S")


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

            def _handle_server_message(self, msg, *, process, cwd, text_parts, reasoning_parts):
                method = msg.get("method", "")
                if method == "session/request_permission":
                    import json
                    message_id = msg.get("id")
                    # ACP spec: AllowedOutcome has outcome="selected" + optionId.
                    # The base class sends {"outcome": {"outcome": "cancelled"}} (DeniedOutcome).
                    # We approve so Devin can run its native tools (exec, read, edit).
                    params = msg.get("params") or {}
                    options = params.get("options") or []
                    # Build a description of what Devin wants to do for the approval prompt.
                    perm_desc = params.get("description") or ""
                    if not perm_desc:
                        for opt in options:
                            if isinstance(opt, dict) and opt.get("name"):
                                perm_desc = opt["name"]
                                break
                        if not perm_desc:
                            perm_desc = "Devin requests permission to run a tool"
                    option_id = ""
                    # 1. Try the terminal tool's approval callback (set in CLI mode).
                    try:
                        from tools.terminal_tool import _get_approval_callback
                        cb = _get_approval_callback()
                        if cb is not None:
                            choice = cb("", perm_desc, allow_permanent=True)
                            _choice_map = {
                                "once": "allow_once",
                                "session": "allow_session",
                                "always": "allow_always",
                            }
                            if choice in _choice_map:
                                option_id = _choice_map[choice]
                    except Exception:
                        pass
                    # 2. If no CLI callback, try the gateway's _block mechanism (desktop/TUI).
                    if not option_id:
                        sid = self._devin_sid()
                        if sid:
                            try:
                                from tui_gateway.server import _block
                                choices = ["Allow once", "Allow for session", "Always allow", "Deny"]
                                answer = _block("clarify.request", sid, {
                                    "question": perm_desc,
                                    "choices": choices,
                                }, timeout=300)
                                _gateway_map = {
                                    "Allow once": "allow_once",
                                    "Allow for session": "allow_session",
                                    "Always allow": "allow_always",
                                    "Deny": "",
                                }
                                option_id = _gateway_map.get(answer, "")
                            except Exception:
                                pass
                    # 3. Fallback: auto-approve (no callback available, e.g. cron/title gen).
                    if not option_id:
                        option_id = "allow_always"
                    # Verify the chosen option is actually offered by Devin.
                    offered = {opt.get("optionId", "") for opt in options if isinstance(opt, dict)}
                    if offered and option_id not in offered:
                        for fallback in ("allow_always", "allow_session", "allow_once"):
                            if fallback in offered:
                                option_id = fallback
                                break
                    if option_id:
                        response = {"jsonrpc": "2.0", "id": message_id,
                                    "result": {"outcome": {"outcome": "selected", "optionId": option_id}}}
                    else:
                        response = {"jsonrpc": "2.0", "id": message_id,
                                    "result": {"outcome": {"outcome": "cancelled"}}}
                    if process.stdin is not None:
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
                    reasoning_parts=reasoning_parts)

            def _create_chat_completion(self, *, model=None, messages=None, timeout=None,
                                         tools=None, tool_choice=None, stream=False, **_):
                # Devin's ACP server provides its OWN native tools (exec, read, edit).
                # The base class injects Hermes' tool schemas into the prompt text and
                # instructs the model to emit tool calls as {...} JSON blocks -- which
                # conflicts with Devin's native tool-call mechanism and confuses the
                # model. We strip Hermes' tools so the model uses Devin's native tools
                # and returns results as text.
                from agent.copilot_acp_client import _format_messages_as_prompt, _effective_timeout
                from agent.acp_openai_bridge import completion_to_stream_chunks as _completion_to_stream_chunks
                from types import SimpleNamespace
                self._devin_tool_seq = 0
                self._devin_active_tools = {}
                prompt_text = _format_messages_as_prompt(
                    messages or [], model=model, tools=None, tool_choice=None)
                response_text, reasoning = self._run_prompt(
                    prompt_text, timeout_seconds=_effective_timeout(timeout), model=model)
                agent = self._devin_agent()
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
        # tools (read, exec, edit) can access files under ~/.hermes and other
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
    display_name="Devin Subscription",
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
            "devin-acp", "Devin Subscription",
            "Devin Subscription (Spawns devin acp --stdio, uses your Devin CLI login)",
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
