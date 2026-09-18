from __future__ import annotations

from typing import Any, Protocol


class ChatProvider(Protocol):
    name: str

    def complete(self, *, messages: list[dict[str, Any]], model: str, max_output_tokens: int) -> dict[str, Any]:
        """Generate a chat completion response."""
