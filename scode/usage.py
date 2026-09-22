"""Token accounting for a session."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    """Rough char-per-token estimate, used before the API reports real counts."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    requests: int = 0
    tool_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)
    api_seconds: float = 0.0

    # Tokens currently held in the live conversation, set by the agent loop.
    context_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def wall_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def record(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        seconds: float = 0.0,
    ) -> None:
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)
        self.cached_tokens += max(0, cached_tokens)
        self.api_seconds += max(0.0, seconds)
        self.requests += 1

    def merge(self, other: Usage) -> None:
        """Fold a subagent's usage into this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        self.requests += other.requests
        self.tool_calls += other.tool_calls
        self.api_seconds += other.api_seconds

    def context_percent(self, window: int) -> float:
        if window <= 0:
            return 0.0
        return min(100.0, 100.0 * self.context_tokens / window)


def format_tokens(count: int) -> str:
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1_000:.1f}k".replace(".0k", "k")
    return f"{count / 1_000_000:.2f}M"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"
