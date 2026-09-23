"""TodoWrite and ExitPlanMode: the tools that structure a turn."""

from __future__ import annotations

from typing import Any

from ..errors import ToolError
from ..permissions import PermissionRequest
from ..ui.glyphs import g
from .base import TodoItem, Tool, ToolContext, ToolResult

STATUSES = ("pending", "in_progress", "completed")


def _mark(status: str) -> str:
    return {
        "pending": g("todo_pending"),
        "in_progress": g("todo_active"),
        "completed": g("todo_done"),
    }.get(status, g("todo_pending"))


class TodoWriteTool(Tool):
    name = "TodoWrite"
    description = (
        "Create and update the task list for multi-step work. Send the full list "
        "every time. Keep exactly one task in_progress, and mark it completed as "
        "soon as it is done. Skip this tool for single-step requests."
    )
    mutating = False
    verb = "Planning"
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Imperative form, e.g. 'Add retry logic'",
                        },
                        "status": {"type": "string", "enum": list(STATUSES)},
                        "activeForm": {
                            "type": "string",
                            "description": "Present continuous, e.g. 'Adding retry logic'",
                        },
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"TodoWrite({len(args.get('todos') or [])} items)"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        raw = args.get("todos")
        if not isinstance(raw, list):
            raise ToolError("todos must be an array")

        items: list[TodoItem] = []
        for index, entry in enumerate(raw, start=1):
            if not isinstance(entry, dict):
                raise ToolError(f"todos[{index}] must be an object")
            content = str(entry.get("content") or "").strip()
            if not content:
                raise ToolError(f"todos[{index}].content is required")
            status = str(entry.get("status") or "pending")
            if status not in STATUSES:
                raise ToolError(
                    f"todos[{index}].status must be one of: {', '.join(STATUSES)}"
                )
            items.append(
                TodoItem(
                    content=content,
                    status=status,
                    active_form=str(entry.get("activeForm") or content),
                )
            )

        active = [i for i in items if i.status == "in_progress"]
        if len(active) > 1:
            raise ToolError(
                "Only one task may be in_progress at a time; "
                f"{len(active)} were marked in_progress."
            )

        ctx.todos[:] = items
        done = sum(1 for i in items if i.status == "completed")
        listing = render_todos(items)
        current = active[0].active_form if active else None
        return ToolResult(
            output=f"Task list updated ({done}/{len(items)} complete).\n{listing}",
            display=current or f"Todos: {done}/{len(items)} complete",
            detail=listing,
            metadata={"todos": [i.__dict__ for i in items]},
        )


def render_todos(items: list[TodoItem]) -> str:
    return "\n".join(f"{_mark(i.status)} {i.content}" for i in items)


class ExitPlanModeTool(Tool):
    name = "ExitPlanMode"
    description = (
        "Present a finished implementation plan and ask the user to approve "
        "leaving plan mode. Only call this after the research is done, and only "
        "for work that will change files."
    )
    mutating = False
    verb = "Planning"
    parameters = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "The plan, in markdown, concise and ordered",
            }
        },
        "required": ["plan"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "ExitPlanMode"

    def permission_request(self, args: dict[str, Any], ctx: ToolContext) -> PermissionRequest:
        return PermissionRequest(
            tool=self.name,
            specifier="",
            title="Ready to code?",
            detail=str(args.get("plan", "")),
            mutating=False,
            suggestions=(),
        )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        plan = str(args["plan"]).strip()
        if not plan:
            raise ToolError("plan is empty")
        ctx.plan_submitted = plan
        return ToolResult(
            output="The user approved the plan. Plan mode is off; start implementing.",
            display="Plan approved",
            detail=plan,
            metadata={"plan": plan},
        )


class AskUserQuestionTool(Tool):
    name = "AskUserQuestion"
    description = (
        "Ask the user one multiple-choice question when a decision is genuinely theirs: "
        "a destructive step, an ambiguous requirement where the options lead to different "
        "work, or a preference you can't infer. Offer 2-6 concrete options; the user can "
        "also type their own answer. Don't use it for things you can decide or look up."
    )
    mutating = False
    verb = "Asking"
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question, ending with '?'"},
            "options": {
                "type": "array",
                "description": "2-6 choices",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Short choice text"},
                        "description": {"type": "string", "description": "What choosing it means"},
                    },
                    "required": ["label"],
                },
            },
            "multiSelect": {"type": "boolean", "description": "Allow more than one choice"},
        },
        "required": ["question", "options"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        question = str(args.get("question", ""))
        return f"AskUserQuestion({question[:60]}{'...' if len(question) > 60 else ''})"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        question = str(args["question"]).strip()
        raw_options = args.get("options")
        if not isinstance(raw_options, list) or not raw_options:
            raise ToolError("options must be a non-empty array")
        options = []
        for entry in raw_options[:6]:
            if isinstance(entry, dict) and str(entry.get("label", "")).strip():
                options.append(
                    {
                        "label": str(entry["label"]).strip(),
                        "description": str(entry.get("description", "")).strip(),
                    }
                )
            elif isinstance(entry, str) and entry.strip():
                options.append({"label": entry.strip(), "description": ""})
        if len(options) < 2:
            raise ToolError("give at least two options")

        if ctx.ask_user is None:
            return ToolResult(
                output=(
                    "Nobody is available to answer (non-interactive session). Proceed with "
                    "your best judgment and state the assumption you made."
                ),
                display="no user available to answer",
            )
        answers = ctx.ask_user(question, options, bool(args.get("multiSelect")))
        if not answers:
            return ToolResult(
                output="The user dismissed the question without answering. Ask what they want instead.",
                display="dismissed",
            )
        joined = "; ".join(answers)
        return ToolResult(output=f"The user answered: {joined}", display=f"answered: {joined}")
