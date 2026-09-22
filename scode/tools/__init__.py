"""Built-in tool registry."""

from __future__ import annotations

from .base import TodoItem, Tool, ToolContext, ToolRegistry, ToolResult
from .fetch import WebFetchTool
from .file_tools import EditTool, MultiEditTool, ReadTool, WriteTool, unified_diff
from .search_tools import GlobTool, GrepTool, LSTool
from .shell import BashTool, detect_shell
from .task import TaskTool
from .workflow import ExitPlanModeTool, TodoWriteTool, render_todos

# Tools a read-only subagent (explore/plan) is allowed to use.
READ_ONLY_TOOLS = ("Read", "Glob", "Grep", "LS", "TodoWrite")


def build_registry(*, include_task: bool = True, read_only: bool = False) -> ToolRegistry:
    tools: list[Tool] = [
        ReadTool(),
        GlobTool(),
        GrepTool(),
        LSTool(),
        TodoWriteTool(),
    ]
    if not read_only:
        tools += [
            WriteTool(),
            EditTool(),
            MultiEditTool(),
            BashTool(),
            WebFetchTool(),
            ExitPlanModeTool(),
        ]
    if include_task and not read_only:
        tools.append(TaskTool())
    return ToolRegistry(tools)


__all__ = [
    "BashTool",
    "EditTool",
    "ExitPlanModeTool",
    "GlobTool",
    "GrepTool",
    "LSTool",
    "MultiEditTool",
    "READ_ONLY_TOOLS",
    "ReadTool",
    "TaskTool",
    "TodoItem",
    "TodoWriteTool",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "WebFetchTool",
    "WriteTool",
    "build_registry",
    "detect_shell",
    "render_todos",
    "unified_diff",
]
