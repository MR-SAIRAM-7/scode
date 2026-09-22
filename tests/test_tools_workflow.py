from __future__ import annotations

import pytest

from scode.errors import ToolError
from scode.tools import ToolContext
from scode.tools.fetch import html_to_text
from scode.tools.task import TaskTool
from scode.tools.workflow import ExitPlanModeTool, TodoWriteTool

# --------------------------------------------------------------- TodoWrite

def test_todos_are_stored(ctx: ToolContext) -> None:
    result = TodoWriteTool().run(
        {
            "todos": [
                {"content": "Read the code", "status": "completed", "activeForm": "Reading"},
                {"content": "Write the fix", "status": "in_progress", "activeForm": "Writing"},
                {"content": "Run tests", "status": "pending", "activeForm": "Testing"},
            ]
        },
        ctx,
    )
    assert len(ctx.todos) == 3
    assert "1/3 complete" in result.output
    assert result.display == "Writing"


def test_only_one_task_may_be_active(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="Only one task"):
        TodoWriteTool().run(
            {
                "todos": [
                    {"content": "a", "status": "in_progress"},
                    {"content": "b", "status": "in_progress"},
                ]
            },
            ctx,
        )


def test_bad_status_is_rejected(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="status must be one of"):
        TodoWriteTool().run({"todos": [{"content": "a", "status": "doing"}]}, ctx)


def test_empty_content_is_rejected(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="content is required"):
        TodoWriteTool().run({"todos": [{"content": "  ", "status": "pending"}]}, ctx)


def test_todos_must_be_a_list(ctx: ToolContext) -> None:
    with pytest.raises(ToolError):
        TodoWriteTool().run({"todos": "nope"}, ctx)


def test_todo_list_replaces_the_previous_one(ctx: ToolContext) -> None:
    TodoWriteTool().run({"todos": [{"content": "a", "status": "pending"}]}, ctx)
    TodoWriteTool().run({"todos": [{"content": "b", "status": "completed"}]}, ctx)
    assert [t.content for t in ctx.todos] == ["b"]


# ------------------------------------------------------------ ExitPlanMode

def test_exit_plan_records_the_plan(ctx: ToolContext) -> None:
    result = ExitPlanModeTool().run({"plan": "1. Do the thing"}, ctx)
    assert ctx.plan_submitted == "1. Do the thing"
    assert "approved" in result.output


def test_exit_plan_rejects_an_empty_plan(ctx: ToolContext) -> None:
    with pytest.raises(ToolError):
        ExitPlanModeTool().run({"plan": "   "}, ctx)


def test_exit_plan_permission_shows_the_plan(ctx: ToolContext) -> None:
    request = ExitPlanModeTool().permission_request({"plan": "step one"}, ctx)
    assert request.detail == "step one"
    assert request.mutating is False


# -------------------------------------------------------------------- Task

def test_task_needs_a_spawner(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="not available"):
        TaskTool().run({"description": "x", "prompt": "find things"}, ctx)


def test_task_rejects_an_unknown_type(ctx: ToolContext) -> None:
    ctx.spawn_subagent = lambda **kwargs: "report"
    with pytest.raises(ToolError, match="Unknown subagent_type"):
        TaskTool().run(
            {"description": "x", "prompt": "p", "subagent_type": "wizard"}, ctx
        )


def test_task_passes_read_only_for_explore(ctx: ToolContext) -> None:
    seen: dict = {}

    def spawn(**kwargs):
        seen.update(kwargs)
        return "found it in src/main.py:12"

    ctx.spawn_subagent = spawn
    result = TaskTool().run(
        {"description": "find auth", "prompt": "locate auth", "subagent_type": "explore"}, ctx
    )
    assert seen["read_only"] is True
    assert seen["subagent_type"] == "explore"
    assert "src/main.py:12" in result.output


def test_task_defaults_to_general_purpose(ctx: ToolContext) -> None:
    seen: dict = {}
    ctx.spawn_subagent = lambda **kwargs: seen.update(kwargs) or "done"
    TaskTool().run({"description": "x", "prompt": "p"}, ctx)
    assert seen["subagent_type"] == "general-purpose"
    assert seen["read_only"] is False


# ---------------------------------------------------------------- WebFetch

def test_html_to_text_extracts_readable_content() -> None:
    html = """
    <html><head><title>Docs</title><style>body{color:red}</style></head>
    <body><nav>skip me</nav><h1>Title</h1><p>First paragraph.</p>
    <script>evil()</script><p>Second paragraph.</p></body></html>
    """
    title, text = html_to_text(html)
    assert title == "Docs"
    assert "First paragraph." in text
    assert "Second paragraph." in text
    assert "evil()" not in text
    assert "color:red" not in text


def test_html_to_text_survives_broken_markup() -> None:
    _, text = html_to_text("<p>unclosed <b>bold")
    assert "unclosed" in text
