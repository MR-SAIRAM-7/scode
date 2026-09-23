"""Provider-facing data types and the canonical conversation format.

Conversations are stored in one provider-neutral shape (OpenAI-style role
dicts) and each wire adapter converts on the way out:

    {"role": "system", "content": str}                 # only at index 0
    {"role": "user", "content": str | [part, ...]}     # part: text or image
    {"role": "assistant", "content": str, "tool_calls": [...],
     "provider_state": {...}}                          # opaque per-provider data
    {"role": "tool", "tool_call_id": str, "name": str, "content": str,
     "is_error": bool}

`provider_state` carries what a provider needs replayed verbatim on the next
request, such as Anthropic thinking blocks with their signatures or
OpenRouter's reasoning_details. Adapters ignore state that isn't theirs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

# Normalised stop reasons, whatever the wire calls them.
STOP_END = "end_turn"
STOP_TOOL_USE = "tool_use"
STOP_MAX_TOKENS = "max_tokens"
STOP_REFUSAL = "refusal"
STOP_PAUSE = "pause_turn"

_OPENAI_STOP_MAP = {
    "stop": STOP_END,
    "tool_calls": STOP_TOOL_USE,
    "function_call": STOP_TOOL_USE,
    "length": STOP_MAX_TOKENS,
    "content_filter": STOP_REFUSAL,
}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    # Set when the arguments could not be parsed into a JSON object.
    parse_error: str | None = None

    @staticmethod
    def from_raw(call_id: str, name: str, raw: str) -> ToolCall:
        raw = raw or ""
        if not raw.strip():
            return ToolCall(id=call_id, name=name, arguments={}, raw_arguments=raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ToolCall(
                id=call_id,
                name=name,
                arguments={},
                raw_arguments=raw,
                parse_error=f"arguments were not valid JSON: {exc}",
            )
        if not isinstance(parsed, dict):
            return ToolCall(
                id=call_id,
                name=name,
                arguments={},
                raw_arguments=raw,
                parse_error="arguments must be a JSON object",
            )
        return ToolCall(id=call_id, name=name, arguments=parsed, raw_arguments=raw)

    @staticmethod
    def from_input(call_id: str, name: str, value: Any) -> ToolCall:
        """Build from an already-parsed input (the Anthropic SDK hands us dicts)."""
        raw = json.dumps(value, ensure_ascii=False) if value is not None else "{}"
        if not isinstance(value, dict):
            return ToolCall(
                id=call_id,
                name=name,
                raw_arguments=raw,
                parse_error="arguments must be a JSON object",
            )
        return ToolCall(id=call_id, name=name, arguments=value, raw_arguments=raw)


@dataclass
class AssistantMessage:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # The wire's own finish reason, kept for display and debugging.
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    # Prompt tokens served from the provider's cache, and written to it.
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    # Cost reported by the provider itself (OpenRouter does), in USD.
    reported_cost: float | None = None
    refusal_category: str | None = None
    provider_state: dict[str, Any] = field(default_factory=dict)
    model: str = ""

    @property
    def stop_reason(self) -> str:
        reason = self.finish_reason or ""
        if reason in {STOP_END, STOP_TOOL_USE, STOP_MAX_TOKENS, STOP_REFUSAL, STOP_PAUSE}:
            return reason
        if reason == "model_context_window_exceeded":
            return STOP_MAX_TOKENS
        if reason in _OPENAI_STOP_MAP:
            return _OPENAI_STOP_MAP[reason]
        return STOP_TOOL_USE if self.tool_calls else STOP_END

    def to_message(self) -> dict[str, Any]:
        """Serialise into the canonical assistant message."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
                }
                for call in self.tool_calls
            ]
        if self.provider_state:
            message["provider_state"] = self.provider_state
        return message


# ------------------------------------------------------------ stream events

@dataclass
class TextDelta:
    text: str


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class ToolCallStarted:
    name: str


@dataclass
class StreamDone:
    message: AssistantMessage


StreamEvent = TextDelta | ReasoningDelta | ToolCallStarted | StreamDone


class Provider(Protocol):
    name: str
    model: str
    supports_tools: bool

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[StreamEvent]:
        """Yield incremental events, ending with exactly one StreamDone."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AssistantMessage:
        """Return the full assistant message in one shot."""

    def list_models(self) -> list[str]:
        """Model ids the endpoint offers."""


# ------------------------------------------------------------ content parts

def text_of(content: Any) -> str:
    """Plain text of a canonical content value (string or part list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return "" if content is None else str(content)


def image_part(media_type: str, data_b64: str) -> dict[str, Any]:
    return {"type": "image", "media_type": media_type, "data": data_b64}
