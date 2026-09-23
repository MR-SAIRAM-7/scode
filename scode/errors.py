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


class ContextOverflow(ProviderError):
    """The conversation no longer fits the model's context window."""


def is_context_overflow(detail: str) -> bool:
    """Recognise the many ways providers say the prompt is too long."""
    text = detail.lower()
    return any(
        marker in text
        for marker in (
            "maximum context length",
            "context length",
            "context_length_exceeded",
            "context window",
            "prompt is too long",
            "input is too long",
            "too many tokens",
            "reduce the length of the messages",
            "exceeds the model's maximum",
        )
    )
