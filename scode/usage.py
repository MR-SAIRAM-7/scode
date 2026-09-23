"""Token and cost accounting for a session."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def estimate_tokens(text: str) -> int:
    """Rough char-per-token estimate, used before the API reports real counts."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def request_cost(
    info: Any,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """USD for one request from catalog prices (per million tokens), or None."""
    if info is None or not getattr(info, "priced", False):
        return None
    cache_read = info.cache_read_cost if info.cache_read_cost is not None else info.input_cost
    cache_write = info.cache_write_cost if info.cache_write_cost is not None else info.input_cost
    total = (
        input_tokens * info.input_cost
        + output_tokens * info.output_cost
        + cache_read_tokens * cache_read
        + cache_write_tokens * cache_write
    )
    return total / 1_000_000


@dataclass
class Usage:
    # input_tokens excludes cached tokens on every provider.
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    # False once any request could not be priced, so totals are shown as partial.
    fully_priced: bool = True
    started_at: float = field(default_factory=time.monotonic)
    api_seconds: float = 0.0

    # Tokens currently held in the live conversation, set by the agent loop.
    context_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cached_tokens + self.cache_write_tokens + self.output_tokens

    @property
    def wall_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def cache_hit_rate(self) -> float:
        prompt = self.input_tokens + self.cached_tokens + self.cache_write_tokens
        return 100.0 * self.cached_tokens / prompt if prompt else 0.0

    def record(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
        seconds: float = 0.0,
        cost: float | None = None,
    ) -> None:
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)
        self.cached_tokens += max(0, cached_tokens)
        self.cache_write_tokens += max(0, cache_write_tokens)
        self.api_seconds += max(0.0, seconds)
        self.requests += 1
        if cost is None:
            self.fully_priced = False
        else:
            self.cost_usd += max(0.0, cost)

    def merge(self, other: Usage) -> None:
        """Fold a subagent's usage into this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.requests += other.requests
        self.tool_calls += other.tool_calls
        self.api_seconds += other.api_seconds
        self.cost_usd += other.cost_usd
        self.fully_priced = self.fully_priced and other.fully_priced

    def context_percent(self, window: int) -> float:
        if window <= 0:
            return 0.0
        return min(100.0, 100.0 * self.context_tokens / window)

    def cost_label(self) -> str:
        if self.requests and not self.cost_usd and not self.fully_priced:
            return "unknown"
        label = format_cost(self.cost_usd)
        return label if self.fully_priced else f">={label}"


def format_cost(usd: float) -> str:
    if usd == 0:
        return "$0.00"
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.2f}"


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
