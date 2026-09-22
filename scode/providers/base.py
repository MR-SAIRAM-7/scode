from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol


class ChatProvider(Protocol):
    name: str

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        max_output_tokens: int,
        temperature: float,
        seed: int,
        reasoning_effort: str,
    ) -> dict[str, Any]:
        """Generate a chat completion response."""

    def stream_text(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        max_output_tokens: int,
        temperature: float,
        seed: int,
        reasoning_effort: str,
    ) -> Iterator[str]:
        """Generate streamed text chunks."""
