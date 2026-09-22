"""The Task tool: run a focused subagent with its own context."""

from __future__ import annotations

from typing import Any

from ..errors import ToolError
from .base import Tool, ToolContext, ToolResult

SUBAGENT_TYPES: dict[str, str] = {
    "general-purpose": (
        "Handles open-ended, multi-step work with the full tool set."
    ),
    "explore": (
        "Read-only searcher. Use it to locate code across many files when you "
        "only need the conclusion, not every file's contents."
    ),
    "plan": (
        "Read-only architect. Returns an implementation plan and the files it "
        "would touch."
    ),
}

READ_ONLY_TYPES = {"explore", "plan"}


class TaskTool(Tool):
    name = "Task"
    verb = "Delegating"
    mutating = False
    description = (
        "Launch a subagent with its own context window for work that would "
        "otherwise flood this conversation — broad codebase searches, or an "
        "independent chunk of a large job. The subagent reports back once and "
        "cannot ask follow-up questions, so give it a complete brief. Types: "
        + "; ".join(f"{name} — {desc}" for name, desc in SUBAGENT_TYPES.items())
    )
    parameters = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "3-5 word label for the task",
            },
            "prompt": {
                "type": "string",
                "description": "The full, self-contained task for the subagent",
            },
            "subagent_type": {
                "type": "string",
                "enum": list(SUBAGENT_TYPES),
                "description": "Which subagent to run (default: general-purpose)",
            },
        },
        "required": ["description", "prompt"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        label = str(args.get("description") or "subagent")
        kind = str(args.get("subagent_type") or "general-purpose")
        return f"Task({kind}: {label})"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        if ctx.spawn_subagent is None:
            raise ToolError("Subagents are not available in this context")

        kind = str(args.get("subagent_type") or "general-purpose")
        if kind not in SUBAGENT_TYPES:
            raise ToolError(
                f"Unknown subagent_type {kind!r}. Choose from: {', '.join(SUBAGENT_TYPES)}"
            )
        prompt = str(args["prompt"]).strip()
        if not prompt:
            raise ToolError("prompt is empty")

        label = str(args.get("description") or kind)
        ctx.note(f"Subagent ({kind}): {label}")

        report = ctx.spawn_subagent(
            prompt=prompt,
            subagent_type=kind,
            read_only=kind in READ_ONLY_TYPES,
            label=label,
        )
        return ToolResult(
            output=report,
            display=f"Subagent finished: {label}",
            metadata={"subagent_type": kind},
        )
