"""Context compaction: summarise the transcript so a long session can continue.

Shape: summarise, then restart from [system, summary + the user's latest
request]. Nothing from the old transcript is replayed. Keeping recent turns
verbatim looks attractive but breaks on models whose thinking is bound to the
exact history that produced it, and costs the same cache reset anyway.

The summary is requested by appending one message to the existing
conversation, so the whole history is served from the provider's prompt cache
rather than re-sent at full price to a second model. When the conversation is
already too large for that, a rendered transcript goes to the small model.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..config import Settings
from ..errors import ProviderError
from ..providers.base import text_of
from .prompts import COMPACT_PROMPT, reminder

SUMMARY_MAX_TOKENS = 16_000
INTERRUPT_NOTE = "[The user interrupted. Stop, and wait for their next message.]"


def _is_scaffolding(content: str) -> bool:
    """User-role messages scode writes itself, as opposed to the human's."""
    stripped = content.lstrip()
    return (
        stripped.startswith("<system-reminder>")
        or stripped.startswith("<tool_result")
        or stripped.startswith("<session_summary>")
        or stripped.startswith("[The user ran this command")
        or stripped == INTERRUPT_NOTE
    )


def last_user_request(messages: list[dict[str, Any]]) -> str:
    """The most recent message the human actually typed."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = text_of(message.get("content"))
        if text.strip() and not _is_scaffolding(text):
            return text.strip()
    return ""


def summarize_in_place(
    provider: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    instructions: str = "",
    on_usage: Callable[[Any], None] | None = None,
) -> str | None:
    """Ask the working model for a summary, reusing its cached prefix."""
    ask = COMPACT_PROMPT
    if instructions.strip():
        ask += f"\n\nThe user asked you to focus on: {instructions.strip()}"
    ask += "\n\nDo not call any tools. Reply with the summary only."
    request = [*messages, {"role": "user", "content": reminder(ask)}]
    try:
        result = provider.complete(request, tools=tools, max_output_tokens=SUMMARY_MAX_TOKENS)
    except ProviderError:
        return None
    if on_usage:
        on_usage(result)
    text = (result.content or "").strip()
    return text or None


def summarize_transcript(
    provider: Any,
    messages: list[dict[str, Any]],
    *,
    settings: Settings,
    instructions: str = "",
    on_usage: Callable[[Any], None] | None = None,
) -> str:
    """Fallback: send a rendered transcript to the small model."""
    ask = COMPACT_PROMPT
    if instructions.strip():
        ask += f"\n\nThe user asked you to focus on: {instructions.strip()}"
    request = [
        {
            "role": "system",
            "content": "You summarise engineering sessions precisely and without flourish.",
        },
        {"role": "user", "content": f"{ask}\n\n<transcript>\n{render_transcript(messages)}\n</transcript>"},
    ]
    try:
        result = provider.complete(
            request,
            model=settings.small_model or provider.model,
            max_output_tokens=SUMMARY_MAX_TOKENS,
        )
    except ProviderError:
        return _fallback_summary(messages)
    if on_usage:
        on_usage(result)
    return result.content.strip() or _fallback_summary(messages)


def _fallback_summary(messages: list[dict[str, Any]]) -> str:
    lines = ["(An automatic summary was unavailable; these were the user's requests.)"]
    requests = [
        text_of(m.get("content")).strip()
        for m in messages
        if m.get("role") == "user" and not _is_scaffolding(text_of(m.get("content")))
    ]
    for text in requests[-6:]:
        lines.append(f"- {text[:300]}")
    return "\n".join(lines)


def render_transcript(messages: list[dict[str, Any]], *, limit: int = 120_000) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        content = text_of(message.get("content"))
        if role == "assistant" and message.get("tool_calls"):
            calls = ", ".join(
                f"{c['function']['name']}({_short(c['function'].get('arguments', ''))})"
                for c in message["tool_calls"]
            )
            content = f"{content}\n[called: {calls}]".strip()
        if role == "tool":
            content = f"[result of {message.get('name', 'tool')}]\n{_short(content, 1200)}"
        parts.append(f"<{role}>\n{content}\n</{role}>")

    joined = "\n\n".join(parts)
    if len(joined) > limit:
        joined = (
            joined[: limit // 3]
            + "\n\n[...middle of the session omitted...]\n\n"
            + joined[-(2 * limit) // 3 :]
        )
    return joined


def _short(text: str, limit: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def build_carrier(summary: str, request: str, *, mid_turn: bool) -> dict[str, Any]:
    """The single user message a compacted conversation restarts from."""
    parts = [
        "<session_summary>",
        "Earlier turns were compacted to save context. This summary is the full record of them:",
        "",
        summary.strip(),
        "</session_summary>",
    ]
    if request:
        parts += ["", "The user's most recent request, verbatim:", "<request>", request, "</request>"]
    parts += [
        "",
        "Continue the work from where the summary leaves off."
        if mid_turn
        else "Wait for the user's next message.",
    ]
    return {"role": "user", "content": "\n".join(parts)}


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    from ..usage import estimate_tokens

    return sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)
