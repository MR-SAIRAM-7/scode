"""Provider-facing data types."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    # Set when raw_arguments could not be parsed as JSON.
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


@dataclass
class AssistantMessage:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def to_message(self) -> dict[str, Any]:
        """Serialise back into an OpenAI-style assistant message."""
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
