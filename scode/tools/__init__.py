"""Built-in tool registry."""

from __future__ import annotations

from collections.abc import Iterable

from .background import BashOutputTool, KillShellTool
from .base import Checkpoint, TodoItem, Tool, ToolContext, ToolRegistry, ToolResult
from .fetch import WebFetchTool
from .file_tools import EditTool, MultiEditTool, ReadTool, WriteTool, unified_diff
from .search_tools import GlobTool, GrepTool, LSTool
from .shell import BashTool, detect_shell
from .task import TaskTool
from .workflow import AskUserQuestionTool, ExitPlanModeTool, TodoWriteTool, render_todos

# Tools a read-only subagent (explore/plan) is allowed to use.
READ_ONLY_TOOLS = ("Read", "Glob", "Grep", "LS", "TodoWrite")
# Pure reads: safe to run concurrently when the model batches them.
PARALLEL_SAFE = frozenset({"Read", "Glob", "Grep", "LS", "BashOutput"})


def build_registry(
    *,
    include_task: bool = True,
    read_only: bool = False,
    custom_agents: dict[str, str] | None = None,
    extra: Iterable[Tool] = (),
    only: Iterable[str] | None = None,
    interactive: bool = True,
) -> ToolRegistry:
    """Assemble the tool set.

    `extra` adds MCP tools; `only` restricts to a named subset (custom agents'
    `tools:` list). AskUserQuestion is left out where nobody could answer.
    """
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
            BashOutputTool(),
            KillShellTool(),
            WebFetchTool(),
            ExitPlanModeTool(),
        ]
        if interactive:
            tools.append(AskUserQuestionTool())
    if include_task and not read_only:
        tools.append(TaskTool(custom_agents))
    tools += list(extra)

    if only is not None:
        allowed = set(only)
        tools = [t for t in tools if t.name in allowed or _server_allowed(t.name, allowed)]
    return ToolRegistry(tools)


def _server_allowed(name: str, allowed: set[str]) -> bool:
    """`mcp__server` in a tools list admits every tool from that server."""
    return name.startswith("mcp__") and any(
        a.startswith("mcp__") and a.count("__") == 1 and name.startswith(a + "__") for a in allowed
    )


__all__ = [
    "PARALLEL_SAFE",
    "READ_ONLY_TOOLS",
    "AskUserQuestionTool",
    "BashOutputTool",
    "BashTool",
    "Checkpoint",
    "EditTool",
    "ExitPlanModeTool",
    "GlobTool",
    "GrepTool",
    "KillShellTool",
    "LSTool",
    "MultiEditTool",
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
