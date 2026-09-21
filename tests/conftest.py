"""Test-suite guards.

Importing `hermes_stubs` installs the stub Hermes packages before any test
imports the plugin, so no test can resolve real Devin credentials or spawn a
real `devin` CLI — tests/devin_stub.py is the only backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hermes_stubs import DEVIN_STUB, PLUGIN, load_plugin  # noqa: E402,F401

assert DEVIN_STUB.exists(), f"Devin CLI stub missing: {DEVIN_STUB}"
assert PLUGIN.exists(), f"plugin missing: {PLUGIN}"
