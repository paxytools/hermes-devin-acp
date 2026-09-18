"""Thin wrapper for `hermes plugins install` discovery.

The real plugin lives in ``hermes_devin_acp.py`` so pip can install it as a
single-module package via the ``hermes_agent.plugins`` entry point. When
Hermes clones this repo into its plugins directory, discovery
imports this ``__init__.py`` via ``spec_from_file_location`` — sibling
imports don't resolve in that path, so we load the file explicitly.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_devin_acp_loaded", str(Path(__file__).parent / "hermes_devin_acp.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update(_mod.__dict__)
