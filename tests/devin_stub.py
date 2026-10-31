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
    st.setdefault("calls", []).append(method)
    save(st)
    resp, err = {}, None
    if method == "initialize":
        resp = {"protocolVersion": 1, "agentCapabilities": {}}
    elif method == "session/new":
        sid = "stub-%d" % st["next"]
        st["next"] += 1
        st["sessions"][sid] = {"cwd": params.get("cwd"), "prompts": 0, "texts": [],
                               "config": {"model": "swe-2-high", "thought_level": "high"}}
        save(st)
        resp = {"sessionId": sid, "configOptions": [
            {"id": "mode", "category": "mode", "type": "select",
             "currentValue": "accept-edits",
             "options": [{"value": "accept-edits"}, {"value": "bypass"}]},
            {"id": "model", "category": "model", "type": "select",
             "currentValue": "swe-2-high",
             "options": [{"value": "swe-2-high"},
                         {"value": "claude-opus-5-5-medium"},
                         {"value": "claude-opus-5-5-medium"},
                         {"value": "disabled-model",
                          "_meta": {"copilotEnablement": "disabled"}}]},
            {"id": "thought_level", "category": "thought_level", "type": "select",
             "currentValue": "high",
             "options": [{"value": "low"}, {"value": "medium"}, {"value": "high"}]},
        ]}
    elif method == "session/set_config_option":
        sid = params.get("sessionId")
        if sid not in st["sessions"]:
            err = {"code": -32602, "message": "session %s not found in store" % sid}
        else:
            cfg = st["sessions"][sid].setdefault(
                "config", {"model": "swe-2-high", "thought_level": "high"})
            cfg[params.get("configId")] = params.get("value")
            if params.get("configId") == "model":
                # Devin resets per-model options to the new model's defaults.
                cfg["thought_level"] = "medium"
            save(st)
            resp = {"configOptions": [
                {"id": "model", "category": "model", "type": "select",
                 "currentValue": cfg["model"],
                 "options": [{"value": "swe-2-high"}, {"value": "claude-opus-5-5-medium"}]},
                {"id": "thought_level", "category": "thought_level", "type": "select",
                 "currentValue": cfg["thought_level"],
                 "options": [{"value": "low"}, {"value": "medium"}, {"value": "high"}]},
            ]}
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
