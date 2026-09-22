"""Colour themes."""

from __future__ import annotations

from rich.theme import Theme

DARK = Theme(
    {
        "scode.brand": "bold #76b900",       # NVIDIA green
        "scode.accent": "#76b900",
        "scode.user": "bold #7aa2f7",
        "scode.assistant": "default",
        "scode.tool": "#c0a3ff",
        "scode.tool.result": "dim",
        "scode.muted": "dim",
        "scode.error": "bold red",
        "scode.warn": "yellow",
        "scode.success": "green",
        "scode.diff.add": "green",
        "scode.diff.del": "red",
        "scode.diff.meta": "cyan",
        "scode.diff.hunk": "dim cyan",
        "scode.prompt": "bold #76b900",
        "scode.thinking": "dim italic",
        "scode.badge": "black on #76b900",
    }
)

LIGHT = Theme(
    {
        "scode.brand": "bold #4a7c00",
        "scode.accent": "#4a7c00",
        "scode.user": "bold #2b5fd9",
        "scode.assistant": "default",
        "scode.tool": "#6b3fc0",
        "scode.tool.result": "dim",
        "scode.muted": "dim",
        "scode.error": "bold red",
        "scode.warn": "dark_orange",
        "scode.success": "dark_green",
        "scode.diff.add": "dark_green",
        "scode.diff.del": "red",
        "scode.diff.meta": "blue",
        "scode.diff.hunk": "dim blue",
        "scode.prompt": "bold #4a7c00",
        "scode.thinking": "dim italic",
        "scode.badge": "white on #4a7c00",
    }
)

THEMES = {"dark": DARK, "light": LIGHT}


def get_theme(name: str) -> Theme:
    return THEMES.get(name.lower(), DARK)
