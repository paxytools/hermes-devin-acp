"""Shared plugin loader for the test suite.

Installs stub ``providers`` / ``hermes_cli`` / ``hermes_constants`` /
``agent`` / ``gateway`` / ``tools`` / ``tui_gateway`` packages *before* the
plugin is imported, so:

* the plugin is exercised as it really ships (a standalone file), and
* no real Hermes checkout, credentials, or ``devin`` CLI is ever needed —
  tests/devin_stub.py is the only backend.
"""

from __future__ import annotations

import contextvars
import importlib.util
import os
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "hermes_devin_acp.py"
DEVIN_STUB = ROOT / "tests" / "devin_stub.py"

_CACHED: types.ModuleType | None = None

# Per-test controllable context, surfaced through the stub functions below.
SESSION_ENV: dict[str, str] = {}          # gateway.session_context.get_session_env
OWNED_KANBAN: str = ""                    # agent.delegation_context.owned_kanban_task
DELEGATED_CHILD: bool = False             # agent.delegation_context.is_delegated_child_context


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def install_stubs() -> None:
    if sys.modules.get("providers") is not None and getattr(
            sys.modules["providers"], "_is_test_stub", False):
        return

    registered: list = []
    providers = _module("providers", register_provider=registered.append,
                        _registered=registered, _is_test_stub=True)

    class ProviderProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    _module("providers.base", ProviderProfile=ProviderProfile)
    providers.base = sys.modules["providers.base"]

    _module("hermes_constants",
            get_hermes_home=lambda: os.environ["HERMES_HOME"],
            get_default_hermes_root=lambda: os.environ["HERMES_HOME"])

    plugins = _module("plugins")
    _module("plugins.plugin_storage",
            plugin_data_dir=lambda name: pathlib.Path(
                os.environ["HERMES_HOME"]) / "plugin-data" / name)
    plugins.plugin_storage = sys.modules["plugins.plugin_storage"]

    class CopilotACPClient:
        """Minimal stand-in: stores the launch config the plugin reuses."""

        def __init__(self, **kw):
            self._acp_command = kw.get("acp_command", "devin")
            self._acp_args = list(kw.get("acp_args") or ["acp"])
            self._acp_cwd = kw.get("acp_cwd", os.getcwd())
            self.is_closed = False

        def _handle_server_message(self, msg, *, process, cwd, text_parts,
                                   reasoning_parts, allow_file_requests=True):
            return False

        def close(self):
            self.is_closed = True

    def _render_message_content(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(p.get("text", "") for p in content
                             if isinstance(p, dict) and p.get("type") == "text")
        return ""

    agent = _module("agent")
    _module("agent.copilot_acp_client",
            CopilotACPClient=CopilotACPClient,
            _render_message_content=_render_message_content,
            _build_subprocess_env=lambda: dict(os.environ),
            _model_selection_request=lambda session, requested: None,
            _effective_timeout=lambda t: t or 300)
    _module("agent.acp_openai_bridge",
            completion_to_stream_chunks=lambda completion: completion)
    _module("agent.auxiliary_client",
            _RELAY_AUX_CALL_CONTEXT=contextvars.ContextVar("aux", default=None))
    _module("agent.delegation_context",
            owned_kanban_task=lambda: OWNED_KANBAN,
            is_delegated_child_context=lambda: DELEGATED_CHILD)

    _module("gateway")
    _module("gateway.session_context",
            get_session_env=lambda name, default=None: SESSION_ENV.get(name, default))

    _module("tui_gateway")
    _module("tui_gateway.server",
            _current_runtime_session_record=contextvars.ContextVar("rt", default=None),
            _sessions={})

    _module("tools")
    _module("tools.approval", check_all_command_guards=lambda *a, **k: {"approved": True})
    _module("tools.approval_context",
            set_hermes_interactive_context=lambda v: None,
            reset_hermes_interactive_context=lambda t: None)

    hermes_cli = _module("hermes_cli")
    _module("hermes_cli._subprocess_compat", windows_hide_flags=lambda: 0)
    _module("hermes_cli.models", _PROVIDER_CATALOG_FETCHERS={})

    class ProviderEntry:
        def __init__(self, slug, name, description):
            self.slug, self.name, self.description = slug, name, description

    _module("hermes_cli.models_catalog_static",
            CANONICAL_PROVIDERS=[], ProviderEntry=ProviderEntry)

    class HermesOverlay:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    _module("hermes_cli.providers",
            HERMES_OVERLAYS={}, HermesOverlay=HermesOverlay,
            _OAUTH_PROVIDER_CATALOG={})


def load_plugin() -> types.ModuleType:
    """Import the plugin against the stubs; cached so tests share one module."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    install_stubs()
    spec = importlib.util.spec_from_file_location("devin_acp_test_plugin", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    _CACHED = module
    return module


install_stubs()
