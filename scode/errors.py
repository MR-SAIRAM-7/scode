"""Exception hierarchy for scode."""

from __future__ import annotations

import re


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


class TransientProviderError(ProviderError):
    """A failure that should clear on its own: overload, rate limit, a dropped stream.

    The agent loop retries these with backoff instead of ending the turn.
    """


_TRANSIENT = re.compile(
    r"overload|temporar|try again|rate.?limit|too many requests|capacity|server is busy|"
    r"timed? ?out|timeout|service unavailable|internal server error|bad gateway|"
    r"connection (?:reset|aborted|closed|broken)"
)
TRANSIENT_CODES = frozenset({"429", "500", "502", "503", "504", "529", "overloaded", "overloaded_error",
                             "rate_limit_exceeded", "rate_limit_error", "server_error", "api_error"})


def is_transient(detail: str, code: object = None) -> bool:
    """Recognise errors that are worth retrying after a pause."""
    if code is not None and str(code).lower() in TRANSIENT_CODES:
        return True
    return bool(_TRANSIENT.search(detail.lower()))


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
