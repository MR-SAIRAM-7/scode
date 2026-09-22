"""Context compaction: summarise the transcript so a long session can continue."""

from __future__ import annotations

import json
from typing import Any

from ..config import Settings
from ..errors import ProviderError
from ..providers.openai_compatible import OpenAICompatibleProvider
from .prompts import COMPACT_PROMPT

# How many trailing messages to keep verbatim, at minimum.
MIN_TAIL = 2
MAX_TAIL = 8


def _last_safe_split(messages: list[dict[str, Any]]) -> int:
    """Index of the newest user message that is safe to cut at.

    Cutting before an assistant message that carries tool_calls would orphan the
    matching tool results, which the API rejects, so only user turns qualify.
    """
    candidates = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and index > 0
    ]
    if not candidates:
        return len(messages)

    for index in reversed(candidates):
        tail_length = len(messages) - index
        if MIN_TAIL <= tail_length <= MAX_TAIL:
            return index
    return candidates[-1]


def summarize(
    provider: OpenAICompatibleProvider,
    messages: list[dict[str, Any]],
    *,
    settings: Settings,
    instructions: str = "",
) -> str:
    transcript = render_transcript(messages)
    ask = COMPACT_PROMPT
    if instructions.strip():
        ask += f"\n\nThe user asked you to focus on: {instructions.strip()}"

    request = [
        {
            "role": "system",
            "content": "You summarise engineering sessions precisely and without flourish.",
        },
        {"role": "user", "content": f"{ask}\n\n<transcript>\n{transcript}\n</transcript>"},
    ]
    try:
        result = provider.complete(
            request,
            model=settings.small_model or provider.model,
            max_output_tokens=min(4000, settings.max_output_tokens),
            temperature=0.2,
        )
    except ProviderError:
        # A failed summary must not lose the session; fall back to a stub.
        return _fallback_summary(messages)
    return result.content.strip() or _fallback_summary(messages)


def _fallback_summary(messages: list[dict[str, Any]]) -> str:
    users = [m for m in messages if m.get("role") == "user"]
    lines = ["(Automatic summary unavailable; keeping the request history.)"]
    for message in users[-6:]:
        content = str(message.get("content") or "")
        lines.append(f"- User asked: {content[:300]}")
    return "\n".join(lines)


def render_transcript(messages: list[dict[str, Any]], *, limit: int = 120_000) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if role == "assistant" and message.get("tool_calls"):
            calls = ", ".join(
                f"{c['function']['name']}({_short(c['function'].get('arguments', ''))})"
                for c in message["tool_calls"]
            )
            content = f"{content}\n[called: {calls}]".strip()
        if role == "tool":
            content = f"[result of {message.get('name', 'tool')}]\n{_short(str(content), 1200)}"
        parts.append(f"<{role}>\n{content}\n</{role}>")

    joined = "\n\n".join(parts)
    if len(joined) > limit:
        joined = joined[: limit // 3] + "\n\n[...middle of the session omitted...]\n\n" + joined[-(2 * limit) // 3 :]
    return joined


def _short(text: str, limit: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def compact_messages(
    provider: OpenAICompatibleProvider,
    messages: list[dict[str, Any]],
    *,
    settings: Settings,
    instructions: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Return (summary, messages_to_keep) excluding the system message."""
    body = [m for m in messages if m.get("role") != "system"]
    if len(body) <= MIN_TAIL:
        return "", body

    split = _last_safe_split(body)
    head, tail = body[:split], body[split:]
    if not head:
        return "", body

    summary = summarize(provider, head, settings=settings, instructions=instructions)
    carrier = {
        "role": "user",
        "content": (
            "<session_summary>\n"
            "Earlier turns were compacted to save context. Here is what happened:\n\n"
            f"{summary}\n"
            "</session_summary>\n\n"
            "Continue from here."
        ),
    }
    return summary, [carrier, *tail]


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    from ..usage import estimate_tokens

    return sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)
