"""Fake ``devin`` CLI / ACP server for tests.

JSON-RPC over stdio. Session state lives in a shared JSON file (argv[1]) so
every spawned process sees the same store — like the real Devin CLI's
sessions.db — which is what makes cross-process effects testable (a donor
process deleting a crashed owner's session, resume after process death).

Also doubles as the ``devin models list`` CLI: ``devin_stub.py models``
prints catalog rows in the format the plugin's parser expects.

State file shape::

    {"next": 1, "sessions": {sid: {"cwd": ..., "prompts": n, "texts": [...]}},
     "deleted": [sid...], "fail_delete": [sid...]}

``fail_delete`` lists sids whose delete must fail (simulates a locked/live row).
"""

from __future__ import annotations

import json
import os
import sys

# `devin models list` — two-space indent, model id, 2+ spaces, display name.
if len(sys.argv) > 1 and sys.argv[1] == "models":
    print("  swe        SWE")
    print("  swe-2      SWE-2")
    sys.exit(0)

# `devin acp` (via a PATH shim) or `devin_stub.py <store.json>` directly.
if len(sys.argv) > 1 and sys.argv[1] == "acp":
    STATE = os.environ["DEVIN_STUB_STORE"]
else:
    STATE = sys.argv[1]


def load():
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return {"next": 1, "sessions": {}, "deleted": [], "fail_delete": []}


def save(st):
    with open(STATE, "w") as f:
        json.dump(st, f)


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue
    mid, method = msg.get("id"), msg.get("method")
    params = msg.get("params") or {}
    st = load()
    resp, err = {}, None
    if method == "initialize":
        resp = {"protocolVersion": 1, "agentCapabilities": {}}
    elif method == "session/new":
        sid = "stub-%d" % st["next"]
        st["next"] += 1
        st["sessions"][sid] = {"cwd": params.get("cwd"), "prompts": 0, "texts": []}
        save(st)
        resp = {"sessionId": sid}
    elif method == "session/load":
        sid = params.get("sessionId")
        if sid in st["sessions"]:
            resp = {"sessionId": sid}
        else:
            err = {"code": -32602, "message": "session %s not found in store" % sid}
    elif method == "session/prompt":
        sid = params.get("sessionId")
        if sid in st["sessions"]:
            st["sessions"][sid]["prompts"] += 1
            text = "".join(p.get("text", "") for p in params.get("prompt") or []
                           if isinstance(p, dict))
            st["sessions"][sid]["texts"].append(text)
            save(st)
            resp = {"stopReason": "end_turn"}
        else:
            err = {"code": -32602, "message": "session %s not found in store" % sid}
    elif method == "session/delete":
        sid = params.get("sessionId")
        if sid in st.get("fail_delete", []):
            err = {"code": -32000,
                   "message": "ACP session/delete: failed to delete session: locked"}
        elif sid in st["sessions"]:
            del st["sessions"][sid]
            st.setdefault("deleted", []).append(sid)
            save(st)
        else:
            err = {"code": -32602, "message": "session %s not found in store" % sid}
    elif method == "authenticate":
        resp = {}
    # session/close and other notifications carry no id -> no response needed.
    if mid is None:
        continue
    out = {"jsonrpc": "2.0", "id": mid}
    if err is not None:
        out["error"] = err
    else:
        out["result"] = resp
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
