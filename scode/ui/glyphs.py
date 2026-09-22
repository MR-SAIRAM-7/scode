"""Glyphs that degrade to ASCII when the terminal cannot encode them."""

from __future__ import annotations

import os
import sys
from functools import lru_cache

_UNICODE = {
    "bullet": "●",
    "arrow": "⏺",
    "branch": "⎿",
    "check": "✔",
    "cross": "✘",
    "warn": "⚠",
    "todo_pending": "☐",
    "todo_active": "▸",
    "todo_done": "☑",
    "spinner": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
    "bar_full": "█",
    "bar_empty": "░",
    "prompt": "›",
    "ellipsis": "…",
}

_ASCII = {
    "bullet": "*",
    "arrow": ">",
    "branch": "\\-",
    "check": "+",
    "cross": "x",
    "warn": "!",
    "todo_pending": "[ ]",
    "todo_active": "[>]",
    "todo_done": "[x]",
    "spinner": "|/-\\",
    "bar_full": "#",
    "bar_empty": ".",
    "prompt": ">",
    "ellipsis": "...",
}


@lru_cache(maxsize=1)
def supports_unicode() -> bool:
    """True when stdout can actually encode the fancy glyphs."""
    if os.getenv("SCODE_ASCII"):
        return False
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    if not encoding:
        return False
    if "utf" in encoding:
        return True
    try:
        "●⏺⎿☑".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def g(name: str) -> str:
    table = _UNICODE if supports_unicode() else _ASCII
    return table.get(name, _ASCII.get(name, ""))


def enable_utf8_stdout() -> None:
    """Best-effort switch of the console to UTF-8 so box glyphs survive."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    supports_unicode.cache_clear()
