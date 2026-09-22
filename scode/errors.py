"""Exception hierarchy for scode."""

from __future__ import annotations


class ScodeError(Exception):
    """Base class for all expected, user-facing failures."""


class ConfigError(ScodeError):
    """Invalid or missing configuration."""


class ProviderError(ScodeError):
    """The model provider returned an error or an unusable response."""


class AuthError(ProviderError):
    """Missing or rejected API credentials."""


class ToolError(ScodeError):
    """A tool could not complete. The message is fed back to the model."""


class PermissionDenied(ScodeError):
    """The user declined a tool invocation."""


class Interrupted(ScodeError):
    """The user interrupted the current turn (Ctrl+C / Esc)."""
