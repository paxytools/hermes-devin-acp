"""Model-catalog tests: configOptions extraction, JSON parse, launch resolution.

Also covers the proactive ``authenticate`` (mandatory on current Devin builds)
and ``fetch_models`` end-to-end through tests/devin_stub.py — the ACP
configOptions path is authoritative, ``devin models list`` is the fallback.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hermes_stubs  # noqa: E402  (installs stubs on import)
from hermes_stubs import DEVIN_STUB, load_plugin  # noqa: E402

m = load_plugin()


class ModelCatalogTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = self._tmp.name
        self.store = os.path.join(tmp, "store.json")
        self.addCleanup(self._tmp.cleanup)
        os.environ["HERMES_HOME"] = tmp
        os.environ["DEVIN_STUB_STORE"] = self.store
        self.addCleanup(os.environ.pop, "DEVIN_STUB_STORE", None)
        self._orig_reaper = m._REAPER_INTERVAL_SECONDS
        m._REAPER_INTERVAL_SECONDS = 3600
        self._env_restore = {k: os.environ.get(k) for k in
                             ("HERMES_DEVIN_ACP_COMMAND", "HERMES_DEVIN_ACP_ARGS")}
        with m._SESSIONS_LOCK:
            m._SESSIONS.clear()

    def tearDown(self):
        with m._SESSIONS_LOCK:
            sessions = list(m._SESSIONS.values())
            m._SESSIONS.clear()
        for s in sessions:
            with contextlib.suppress(Exception):
                s.terminate()
        m._REAPER_INTERVAL_SECONDS = self._orig_reaper
        m._PERSISTED.clear()
        for k, v in self._env_restore.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ------------------------------ helpers ---------------------------------

    def _store(self):
        try:
            with open(self.store) as f:
                return json.load(f)
        except Exception:
            return {"next": 1, "sessions": {}, "deleted": [], "calls": []}

    def _devin_shim(self) -> str:
        """Executable ``devin`` shim that execs the stub under the interpreter —
        lets ``shutil.which``/``subprocess.run`` treat it like the real CLI."""
        shim = pathlib.Path(self._tmp.name) / "devin"
        shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{DEVIN_STUB}" "$@"\n')
        shim.chmod(0o755)
        return str(shim)

    # --------------------------- pure extractors -----------------------------

    def test_config_option_model_ids(self):
        session = {"configOptions": [
            {"id": "mode", "category": "mode",
             "options": [{"value": "accept-edits"}]},
            {"id": "model", "category": "model",
             "options": [{"value": "a"}, {"value": "b"}, {"value": "a"},
                         {"value": "off", "_meta": {"copilotEnablement": "disabled"}},
                         {"value": ""}, "junk", {}]},
        ]}
        self.assertEqual(m._config_option_model_ids(session), ["a", "b"])

    def test_config_option_model_ids_by_id_without_category(self):
        session = {"configOptions": [
            {"id": "model", "options": [{"value": "x"}]}]}
        self.assertEqual(m._config_option_model_ids(session), ["x"])

    def test_config_option_model_ids_absent(self):
        self.assertIsNone(m._config_option_model_ids({}))
        self.assertIsNone(m._config_option_model_ids({"configOptions": None}))
        self.assertIsNone(m._config_option_model_ids({"configOptions": [
            {"id": "mode", "category": "mode", "options": [{"value": "ask"}]}]}))

    def test_models_from_json_shapes(self):
        self.assertEqual(m._models_from_json('["a","b"]'), ["a", "b"])
        self.assertEqual(m._models_from_json(
            '{"models": [{"id": "a"}, {"slug": "b"}, {"value": "c"}]}'),
            ["a", "b", "c"])
        self.assertEqual(m._models_from_json(
            '{"data": [{"modelId": "m1"}, {"model_id": "m2"}]}'), ["m1", "m2"])

    def test_models_from_json_junk(self):
        self.assertEqual(m._models_from_json(""), [])
        self.assertEqual(m._models_from_json("not json"), [])
        self.assertEqual(m._models_from_json("{}"), [])
        self.assertEqual(m._models_from_json('[{"other": 1}, 5]'), [])

    # --------------------------- launch resolution ---------------------------

    def test_resolved_launch_defaults(self):
        for k in self._env_restore:
            os.environ.pop(k, None)
        command, args = m.devin_acp._resolved_launch()
        self.assertEqual(command, "devin")
        self.assertEqual(args, ["acp"])

    def test_resolved_launch_env_overrides(self):
        shim = self._devin_shim()
        os.environ["HERMES_DEVIN_ACP_COMMAND"] = shim
        os.environ["HERMES_DEVIN_ACP_ARGS"] = "acp --model opus"
        command, args = m.devin_acp._resolved_launch()
        self.assertEqual(command, shim)
        self.assertEqual(args, ["acp", "--model", "opus"])

    # --------------------------- authenticate order --------------------------

    def test_authenticate_precedes_session_open(self):
        """Current Devin requires authenticate before session/new; ensure_ready
        must send it proactively (patch the key reader so this holds even on a
        box with no credentials.toml)."""
        orig = m._devin_credentials_api_key
        m._devin_credentials_api_key = lambda: "test-key"
        self.addCleanup(setattr, m, "_devin_credentials_api_key", orig)
        sess = m._DevinSession("k", sys.executable, [str(DEVIN_STUB), self.store], "/tmp")
        try:
            sess.ensure_ready(deadline=time.monotonic() + 15, dispatch=None)
            self.assertTrue(sess._authenticated)
            sess.open_session(model=None, hashes=[], deadline=time.monotonic() + 15,
                              dispatch=None, persisted=None)
        finally:
            sess.terminate(delete_session=True)
        calls = self._store().get("calls", [])
        self.assertIn("authenticate", calls)
        self.assertLess(calls.index("authenticate"), calls.index("session/new"))

    def test_no_key_skips_authenticate(self):
        orig = m._devin_credentials_api_key
        m._devin_credentials_api_key = lambda: ""
        self.addCleanup(setattr, m, "_devin_credentials_api_key", orig)
        sess = m._DevinSession("k", sys.executable, [str(DEVIN_STUB), self.store], "/tmp")
        try:
            sess.ensure_ready(deadline=time.monotonic() + 15, dispatch=None)
            self.assertFalse(sess._authenticated)
        finally:
            sess.terminate()
        self.assertNotIn("authenticate", self._store().get("calls", []))

    # ------------------------------ fetch_models -----------------------------

    def test_fetch_models_via_acp_config_options(self):
        """fetch_models prefers the authenticated session's model configOptions:
        ordered values, disabled entries skipped, deduped — and the probe's
        session row is deleted on the way out."""
        os.environ["HERMES_DEVIN_ACP_COMMAND"] = self._devin_shim()
        models = m.devin_acp.fetch_models(timeout=20.0)
        self.assertEqual(models, ["swe-2-high", "claude-opus-5-5-medium"])
        st = self._store()
        self.assertEqual(st.get("sessions"), {})
        self.assertTrue(st.get("deleted"))

    def test_models_via_cli_text_table(self):
        """The CLI fallback parses the text table when --format json output
        isn't JSON (stub always prints text rows)."""
        models = m.devin_acp._models_via_cli(
            self._devin_shim(), time.monotonic() + 15)
        self.assertEqual(models, ["swe", "swe-2"])

    def test_fetch_models_missing_binary(self):
        os.environ["HERMES_DEVIN_ACP_COMMAND"] = "/nonexistent/devin"
        self.assertIsNone(m.devin_acp.fetch_models(timeout=5.0))

    # --------------------------- config preservation -------------------------

    def test_model_switch_restores_thought_level(self):
        """Devin resets per-model options (thought_level high → medium) on a
        model switch; the plugin must restore the pre-switch values."""
        stub = sys.modules["agent.copilot_acp_client"]
        orig = stub._model_selection_request
        stub._model_selection_request = lambda session, requested: (
            "session/set_config_option",
            {"sessionId": session["sessionId"], "configId": "model", "value": requested})
        self.addCleanup(setattr, stub, "_model_selection_request", orig)
        sess = m._DevinSession("k", sys.executable, [str(DEVIN_STUB), self.store], "/tmp")
        try:
            sess.ensure_ready(deadline=time.monotonic() + 15, dispatch=None)
            sess.open_session(model=None, hashes=[], deadline=time.monotonic() + 15,
                              dispatch=None, persisted=None)
            sess._apply_model(sess._session_result, "claude-opus-5-5-medium",
                              deadline=time.monotonic() + 15, dispatch=None)
            cfg = self._store()["sessions"][sess.session_id]["config"]
            self.assertEqual(cfg["model"], "claude-opus-5-5-medium")
            self.assertEqual(cfg["thought_level"], "high")  # restored, not "medium"
        finally:
            sess.terminate(delete_session=True)
        self.assertEqual(self._store()["calls"].count("session/set_config_option"), 2)


if __name__ == "__main__":
    unittest.main()
