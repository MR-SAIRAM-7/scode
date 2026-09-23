"""Model providers.

Submodules are imported lazily so that `scode.config` can use the profile
registry without pulling in the HTTP clients (which import config back).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AssistantMessage": "base",
    "Provider": "base",
    "StreamDone": "base",
    "TextDelta": "base",
    "ToolCall": "base",
    "ToolCallStarted": "base",
    "OpenAICompatibleProvider": "openai_compatible",
    "ToolsNotSupported": "openai_compatible",
    "ProviderProfile": "profiles",
    "BUILTIN_PROFILES": "profiles",
    "resolve_profile": "profiles",
    "get_provider": "registry",
    "list_catalog": "registry",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'scode.providers' has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)
