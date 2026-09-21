"""Session-layer tests: reuse, delta prompts, tombstones, sweep, divergence.

Drives the real plugin through tests/devin_stub.py — a shared-state ACP
stub — so no real ``devin`` CLI or Hermes checkout is needed. unittest-style
so it runs under both ``python -m unittest`` and ``pytest``.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hermes_stubs  # noqa: E402  (installs stubs on import)
from hermes_stubs import DEVIN_STUB, load_plugin  # noqa: E402

m = load_plugin()


class SessionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = self._tmp.name
        self.store = os.path.join(tmp, "store.json")
        self.addCleanup(self._tmp.cleanup)
        # Plugin state files (session map + pending-delete) -> tmp dir.
        os.environ["HERMES_HOME"] = tmp
        self._orig_state_path = m._state_path
        m._state_path = lambda: pathlib.Path(tmp) / "devin_acp_sessions.json"
        self._orig_reaper = m._REAPER_INTERVAL_SECONDS
        m._REAPER_INTERVAL_SECONDS = 3600  # keep the reaper out of tests
        m._PERSISTED.clear()
        hermes_stubs.SESSION_ENV.clear()
        hermes_stubs.OWNED_KANBAN = ""
        hermes_stubs.DELEGATED_CHILD = False
        with m._SESSIONS_LOCK:
            m._SESSIONS.clear()

    def tearDown(self):
        with m._SESSIONS_LOCK:
            sessions = list(m._SESSIONS.values())
            m._SESSIONS.clear()
        for s in sessions:
            with contextlib.suppress(Exception):
                s.terminate()
        m._state_path = self._orig_state_path
        m._REAPER_INTERVAL_SECONDS = self._orig_reaper
        m._PERSISTED.clear()

    # ------------------------------ helpers ---------------------------------

    def _store(self):
        try:
            with open(self.store) as f:
                return json.load(f)
        except Exception:
            return {"next": 1, "sessions": {}, "deleted": [], "fail_delete": []}

    def _spawn(self, key="k", cwd="/tmp", **attrs):
        sess = m._DevinSession(key, sys.executable, [str(DEVIN_STUB), self.store], cwd)
        for k, v in attrs.items():
            setattr(sess, k, v)
        sess.ensure_ready(deadline=time.monotonic() + 15, dispatch=None)
        return sess

    def _open(self, sess, hashes=None, persisted=None):
        sess.open_session(model=None, hashes=hashes or [],
                          deadline=time.monotonic() + 15,
                          dispatch=lambda msg: False, persisted=persisted)

    def _prompt(self, sess, text="hi"):
        sess.prompt(text, deadline=time.monotonic() + 15,
                    dispatch=lambda msg: False)

    def _pending(self):
        return m._load_pending_delete()

    def _age_all(self):
        entries = self._pending()
        for e in entries:
            e["marked"] = time.time() - m._IDLE_TTL_SECONDS - 5
        m._save_pending_delete(entries)

    def _client(self):
        return m.DevinACPClient(acp_command=sys.executable,
                                acp_args=[str(DEVIN_STUB), self.store], acp_cwd="/tmp")

    # --------------------------- tombstone file ------------------------------

    def test_mark_unmark_roundtrip(self):
        m._mark_pending_delete("s1", "/tmp")
        self.assertEqual([e["sid"] for e in self._pending()], ["s1"])
        m._unmark_pending_delete("s1")
        self.assertEqual(self._pending(), [])

    def test_mark_dedup_preserves_timestamp(self):
        m._mark_pending_delete("s1", "/tmp")
        first = self._pending()[0]["marked"]
        time.sleep(0.01)
        m._mark_pending_delete("s1", "/tmp")
        self.assertEqual(self._pending()[0]["marked"], first)

    def test_mark_stale_preages_and_updates_existing(self):
        m._mark_pending_delete("s1", "/tmp")
        m._mark_pending_delete("s1", "/tmp", stale=True)
        entry = self._pending()[0]
        self.assertLess(entry["marked"], time.time() - m._IDLE_TTL_SECONDS)

    # --------------------------- open / terminate ----------------------------

    def test_ephemeral_cleanup_tombstoned_at_open(self):
        sess = self._spawn(ephemeral_cleanup=True)
        self._open(sess)
        self.assertTrue(any(e["sid"] == sess.session_id for e in self._pending()))
        sess.terminate()

    def test_normal_session_not_tombstoned(self):
        sess = self._spawn()
        self._open(sess)
        self.assertEqual(self._pending(), [])
        sess.terminate()

    def test_terminate_deletes_row_and_clears_tombstone(self):
        sess = self._spawn(ephemeral_cleanup=True)
        self._open(sess)
        self._prompt(sess)
        sid = sess.session_id
        sess.terminate()
        self.assertNotIn(sid, self._store()["sessions"])
        self.assertFalse(any(e["sid"] == sid for e in self._pending()))

    def test_normal_terminate_keeps_row(self):
        """A real conversation's row must survive terminate — it is resumable."""
        sess = self._spawn()
        self._open(sess)
        self._prompt(sess)
        sid = sess.session_id
        sess.terminate()
        self.assertIn(sid, self._store()["sessions"])

    def test_dead_proc_terminate_leaves_stale_tombstone(self):
        sess = self._spawn()
        self._open(sess)
        self._prompt(sess)
        sid = sess.session_id
        sess.proc.kill()
        sess.proc.wait()
        sess.terminate(delete_session=True)
        entry = next(e for e in self._pending() if e["sid"] == sid)
        self.assertLess(entry["marked"], time.time() - m._IDLE_TTL_SECONDS)

    # ------------------------------- sweep -----------------------------------

    def test_sweep_removes_orphan_via_donor(self):
        victim = self._spawn(ephemeral_cleanup=True)
        self._open(victim)
        self._prompt(victim)
        sid = victim.session_id
        victim.proc.kill()                  # crash: no terminate, tombstone stays
        victim.proc = None
        self._age_all()
        donor = self._spawn(key="donor")
        m._SESSIONS["donor"] = donor
        m._sweep_pending_deletes()
        self.assertNotIn(sid, self._store()["sessions"])
        self.assertFalse(any(e["sid"] == sid for e in self._pending()))

    def test_sweep_skips_fresh_tombstone(self):
        victim = self._spawn(ephemeral_cleanup=True)
        self._open(victim)
        self._prompt(victim)
        sid = victim.session_id
        victim.proc.kill()
        victim.proc = None                  # fresh tombstone (< TTL): not eligible
        donor = self._spawn(key="donor")
        m._SESSIONS["donor"] = donor
        m._sweep_pending_deletes()
        self.assertIn(sid, self._store()["sessions"])

    def test_sweep_skips_live_owned_session(self):
        sess = self._spawn()
        self._open(sess)
        sid = sess.session_id
        m._mark_pending_delete(sid, "/tmp", stale=True)
        m._SESSIONS["k"] = sess
        donor = self._spawn(key="donor")
        m._SESSIONS["donor"] = donor
        m._sweep_pending_deletes()
        self.assertIn(sid, self._store()["sessions"])

    def test_sweep_clears_not_found(self):
        m._mark_pending_delete("ghost-sid", "/tmp", stale=True)
        donor = self._spawn(key="donor")
        m._SESSIONS["donor"] = donor
        m._sweep_pending_deletes()
        self.assertEqual(self._pending(), [])

    def test_sweep_keeps_tombstone_on_failure(self):
        st = self._store()
        st["sessions"]["stuck-sid"] = {"cwd": "/tmp", "prompts": 0}
        st["fail_delete"] = ["stuck-sid"]
        with open(self.store, "w") as f:
            json.dump(st, f)
        m._mark_pending_delete("stuck-sid", "/tmp", stale=True)
        donor = self._spawn(key="donor")
        m._SESSIONS["donor"] = donor
        m._sweep_pending_deletes()
        self.assertTrue(any(e["sid"] == "stuck-sid" for e in self._pending()))

    def test_sweep_drops_expired_without_delete(self):
        st = self._store()
        st["sessions"]["old-sid"] = {"cwd": "/tmp", "prompts": 0}
        with open(self.store, "w") as f:
            json.dump(st, f)
        m._save_pending_delete([{"sid": "old-sid", "cwd": "/tmp",
                                 "marked": time.time() - 8 * 86400}])
        donor = self._spawn(key="donor")
        m._SESSIONS["donor"] = donor
        m._sweep_pending_deletes()
        self.assertEqual(self._pending(), [])
        self.assertIn("old-sid", self._store()["sessions"])  # dropped, not deleted

    def test_sweep_drops_corrupt_entry(self):
        m._save_pending_delete([{"sid": "bad", "marked": "garbage"}])
        donor = self._spawn(key="donor")
        m._SESSIONS["donor"] = donor
        m._sweep_pending_deletes()
        self.assertEqual(self._pending(), [])

    def test_sweep_no_donor_is_noop(self):
        m._mark_pending_delete("s1", "/tmp", stale=True)
        m._sweep_pending_deletes()          # no live sessions -> nothing happens
        self.assertTrue(any(e["sid"] == "s1" for e in self._pending()))

    def test_sweep_skips_live_ephemeral(self):
        """key=None sessions aren't in _SESSIONS but are still live — the
        sweep must not delete their rows once the open tombstone ages out."""
        sess = self._spawn(ephemeral_cleanup=True)
        self._open(sess)
        self._prompt(sess)
        sid = sess.session_id
        m._EPHEMERAL.add(sess)              # simulate a long-running aux call
        try:
            self._age_all()                 # tombstone past TTL -> eligible
            donor = self._spawn(key="donor")
            m._SESSIONS["donor"] = donor
            m._sweep_pending_deletes()
            self.assertIn(sid, self._store()["sessions"])
        finally:
            m._EPHEMERAL.discard(sess)
            sess.terminate()

    # ------------------------- divergence / resume ---------------------------

    def test_divergence_supersede_deletes_old(self):
        sess = self._spawn()
        self._open(sess)
        self._prompt(sess)
        old_sid = sess.session_id
        self._open(sess, hashes=["diverged-hash"])  # sent prefix no longer matches
        self.assertNotEqual(sess.session_id, old_sid)
        self.assertNotIn(old_sid, self._store()["sessions"])
        self.assertFalse(any(e["sid"] == old_sid for e in self._pending()))
        sess.terminate()

    def test_resume_via_load_unmarks_tombstone(self):
        first = self._spawn()
        self._open(first)
        sid = first.session_id
        first.proc.kill()                   # process dies; map + row survive
        first.proc = None
        m._mark_pending_delete(sid, "/tmp", stale=True)
        second = self._spawn(key="k2")
        self._open(second, hashes=["h1"],
                   persisted={"devin_sid": sid, "hashes": ["h1"], "cwd": "/tmp"})
        self.assertEqual(second.session_id, sid)
        self.assertFalse(any(e["sid"] == sid for e in self._pending()))
        m._SESSIONS["k2"] = second
        m._sweep_pending_deletes()
        self.assertIn(sid, self._store()["sessions"])  # resumed session survives
        second.terminate()

    def test_failed_load_supersedes_persisted(self):
        sess = self._spawn()
        self._open(sess, hashes=["h1"],
                   persisted={"devin_sid": "deleted-elsewhere",
                              "hashes": ["h1"], "cwd": "/tmp"})
        self.assertTrue(sess.session_id)
        self.assertNotEqual(sess.session_id, "deleted-elsewhere")
        self.assertIn(sess.session_id, self._store()["sessions"])
        sess.terminate()

    def test_diverged_persisted_sid_deleted(self):
        first = self._spawn()
        self._open(first)
        old_sid = first.session_id
        first.proc.kill()
        first.proc = None
        second = self._spawn(key="k2")
        self._open(second, hashes=["new-h1"],   # persisted prefix diverged
                   persisted={"devin_sid": old_sid,
                              "hashes": ["old-h1", "old-h2"], "cwd": "/tmp"})
        self.assertNotIn(old_sid, self._store()["sessions"])
        second.terminate()

    def test_map_pointing_at_diverged_live_session_not_reloaded(self):
        """p_sid == the session we just diverged from: delete it, never reload."""
        sess = self._spawn()
        self._open(sess)
        self._prompt(sess)
        old_sid = sess.session_id
        self._open(sess, hashes=["new"],
                   persisted={"devin_sid": old_sid, "hashes": ["h1"],
                              "cwd": "/tmp"})
        self.assertNotEqual(sess.session_id, old_sid)
        self.assertNotIn(old_sid, self._store()["sessions"])
        sess.terminate()

    # ----------------------------- persistence -------------------------------

    def test_persist_entry_skips_ephemeral(self):
        sess = self._spawn(key="ck", ephemeral_cleanup=True)
        self._open(sess)
        self._prompt(sess)
        sess.sent_hashes = ["h1"]
        m._persist_entry(sess)
        self.assertIsNone(m._persisted_state().get("ck"))
        sess.terminate()

    def test_persist_entry_writes_normal(self):
        sess = self._spawn(key="ck")
        self._open(sess)
        sess.sent_hashes = ["h1"]
        m._persist_entry(sess)
        entry = m._persisted_state().get("ck")
        self.assertTrue(entry)
        self.assertEqual(entry["devin_sid"], sess.session_id)
        sess.terminate()

    def test_persisted_state_prunes_old_entries(self):
        m._save_state({"ancient": {"devin_sid": "x", "updated": 1},
                       "recent": {"devin_sid": "y", "updated": time.time()}})
        m._PERSISTED.clear()
        state = m._persisted_state()
        self.assertIn("recent", state)
        self.assertNotIn("ancient", state)

    def test_persisted_state_survives_corrupt_updated(self):
        """One malformed 'updated' must not raise through the cache forever."""
        m._save_state({"broken": {"devin_sid": "x", "updated": "garbage"},
                       "recent": {"devin_sid": "y", "updated": time.time()}})
        m._PERSISTED.clear()
        state = m._persisted_state()
        self.assertIn("recent", state)
        self.assertNotIn("broken", state)  # corrupt -> epoch -> pruned

    def test_pending_delete_normalizes_nonstring_sid(self):
        """A corrupt numeric sid still matches mark/unmark/sweep comparisons."""
        m._save_pending_delete([{"sid": 123, "cwd": "/tmp",
                                 "marked": time.time()}])
        self.assertEqual(self._pending()[0]["sid"], "123")
        m._unmark_pending_delete("123")
        self.assertEqual(self._pending(), [])

    def test_state_files_migrate_to_plugin_data(self):
        """Legacy <home>/devin_acp_*.json files move into plugin-data once."""
        patched = m._state_path
        m._state_path = self._orig_state_path  # exercise the real path
        try:
            root = pathlib.Path(os.environ["HERMES_HOME"])
            (root / "devin_acp_sessions.json").write_text(json.dumps({"k": {}}))
            (root / "devin_acp_pending_delete.json").write_text(
                json.dumps([{"sid": "s1", "cwd": "/tmp", "marked": 1}]))
            new_dir = root / "plugin-data" / "devin-acp"
            self.assertEqual(m._state_path(),
                             new_dir / "devin_acp_sessions.json")
            self.assertEqual(m._pending_delete_path(),
                             new_dir / "devin_acp_pending_delete.json")
            self.assertTrue((new_dir / "devin_acp_sessions.json").exists())
            self.assertTrue((new_dir / "devin_acp_pending_delete.json").exists())
            self.assertFalse((root / "devin_acp_sessions.json").exists())
            self.assertFalse((root / "devin_acp_pending_delete.json").exists())
            # A re-created legacy file must not clobber the migrated one.
            (root / "devin_acp_sessions.json").write_text(json.dumps({"stale": {}}))
            m._state_path()
            self.assertEqual(
                json.loads((new_dir / "devin_acp_sessions.json").read_text()),
                {"k": {}})
            self.assertEqual(self._pending()[0]["sid"], "s1")
        finally:
            m._state_path = patched


class ClientEndToEndTest(unittest.TestCase):
    """Drives _devin_complete — the real request path — through the stub server."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = self._tmp.name
        self.store = os.path.join(tmp, "store.json")
        self.addCleanup(self._tmp.cleanup)
        os.environ["HERMES_HOME"] = tmp
        self._orig_state_path = m._state_path
        m._state_path = lambda: pathlib.Path(tmp) / "devin_acp_sessions.json"
        self._orig_reaper = m._REAPER_INTERVAL_SECONDS
        m._REAPER_INTERVAL_SECONDS = 3600
        m._PERSISTED.clear()
        hermes_stubs.SESSION_ENV.clear()
        hermes_stubs.OWNED_KANBAN = ""
        hermes_stubs.DELEGATED_CHILD = False
        with m._SESSIONS_LOCK:
            m._SESSIONS.clear()

    def tearDown(self):
        with m._SESSIONS_LOCK:
            sessions = list(m._SESSIONS.values())
            m._SESSIONS.clear()
        for s in sessions:
            with contextlib.suppress(Exception):
                s.terminate()
        m._state_path = self._orig_state_path
        m._REAPER_INTERVAL_SECONDS = self._orig_reaper
        m._PERSISTED.clear()

    def _store(self):
        try:
            with open(self.store) as f:
                return json.load(f)
        except Exception:
            return {"next": 1, "sessions": {}, "deleted": [], "fail_delete": []}

    def _client(self):
        return m.DevinACPClient(acp_command=sys.executable,
                                acp_args=[str(DEVIN_STUB), self.store], acp_cwd="/tmp")

    def _msgs(self, *texts):
        return [{"role": "user", "content": t} for t in texts]

    def test_shared_session_and_delta_prompt(self):
        hermes_stubs.SESSION_ENV["HERMES_SESSION_ID"] = "conv-1"
        client = self._client()
        client._devin_complete(self._msgs("first"), model=None,
                               timeout_seconds=30, agent=None)
        client2 = self._client()   # per-request client, same conversation key
        client2._devin_complete(self._msgs("first", "second"), model=None,
                                timeout_seconds=30, agent=None)
        sessions = self._store()["sessions"]
        self.assertEqual(len(sessions), 1)          # one Devin session total
        texts = next(iter(sessions.values()))["texts"]
        self.assertEqual(len(texts), 2)
        self.assertIn("Conversation transcript", texts[0])   # full on first call
        self.assertIn("Conversation update", texts[1])       # delta on second
        self.assertNotIn("first", texts[1])                  # delta has no prefix

    def test_persistent_mapping_written(self):
        hermes_stubs.SESSION_ENV["HERMES_SESSION_ID"] = "conv-1"
        self._client()._devin_complete(self._msgs("hi"), model=None,
                                       timeout_seconds=30, agent=None)
        entry = m._persisted_state()["conv-1"]
        self.assertIn(entry["devin_sid"], self._store()["sessions"])

    def test_cron_context_flags_one_shot(self):
        hermes_stubs.SESSION_ENV["HERMES_SESSION_ID"] = "cron_abc_20260919_190017"
        hermes_stubs.SESSION_ENV["HERMES_CRON_SESSION"] = "1"
        self._client()._devin_complete(self._msgs("hi"), model=None,
                                       timeout_seconds=30, agent=None)
        sess = m._SESSIONS["cron_abc_20260919_190017"]
        self.assertTrue(sess.ephemeral_cleanup)
        self.assertIsNone(m._persisted_state().get("cron_abc_20260919_190017"))
        sid = sess.session_id
        sess.terminate()
        self.assertNotIn(sid, self._store()["sessions"])

    def test_kanban_context_flags_one_shot(self):
        hermes_stubs.SESSION_ENV["HERMES_SESSION_ID"] = "kanban-t1"
        hermes_stubs.OWNED_KANBAN = "t1"
        self._client()._devin_complete(self._msgs("hi"), model=None,
                                       timeout_seconds=30, agent=None)
        self.assertTrue(m._SESSIONS["kanban-t1"].ephemeral_cleanup)
        self.assertIsNone(m._persisted_state().get("kanban-t1"))

    def test_normal_conversation_not_flagged(self):
        hermes_stubs.SESSION_ENV["HERMES_SESSION_ID"] = "conv-2"
        self._client()._devin_complete(self._msgs("hi"), model=None,
                                       timeout_seconds=30, agent=None)
        self.assertFalse(m._SESSIONS["conv-2"].ephemeral_cleanup)

    def test_delegated_child_uses_ephemeral(self):
        hermes_stubs.SESSION_ENV["HERMES_SESSION_ID"] = "parent-conv"
        hermes_stubs.DELEGATED_CHILD = True
        self._client()._devin_complete(self._msgs("hi"), model=None,
                                       timeout_seconds=30, agent=None)
        # No keyed session created; the ephemeral session deleted its own row.
        self.assertNotIn("parent-conv", m._SESSIONS)
        self.assertEqual(m._load_pending_delete(), [])
        self.assertEqual(len(self._store()["sessions"]), 0)

    def test_process_death_resumes_via_load(self):
        """Process dies between turns; the persisted map resumes the same sid."""
        hermes_stubs.SESSION_ENV["HERMES_SESSION_ID"] = "conv-3"
        self._client()._devin_complete(self._msgs("hi"), model=None,
                                       timeout_seconds=30, agent=None)
        sess = m._SESSIONS["conv-3"]
        sid = sess.session_id
        sess.proc.kill()                    # simulate idle-reaper/crash
        sess.proc = None
        self._client()._devin_complete(self._msgs("hi", "more"), model=None,
                                       timeout_seconds=30, agent=None)
        self.assertEqual(sess.session_id, sid)          # resumed, not recreated
        texts = self._store()["sessions"][sid]["texts"]
        self.assertIn("Conversation update", texts[-1]) # delta on resumed session


if __name__ == "__main__":
    unittest.main()
