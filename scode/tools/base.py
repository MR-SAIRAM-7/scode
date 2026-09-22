"""Tool contract shared by every built-in tool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..constants import MAX_TOOL_OUTPUT_CHARS
from ..errors import ToolError
from ..permissions import PermissionEngine, PermissionRequest, tool_specifier


@dataclass
class TodoItem:
    content: str
    status: str = "pending"  # pending | in_progress | completed
    active_form: str = ""


@dataclass
class ToolContext:
    """Everything a tool is allowed to touch."""

    workspace: Path
    settings: Settings
    permissions: PermissionEngine
    # file path -> mtime at the time it was last read by the agent
    read_files: dict[str, float] = field(default_factory=dict)
    todos: list[TodoItem] = field(default_factory=list)
    # Set by the REPL so tools can stream progress lines to the user.
    emit: Callable[[str], None] | None = None
    # Provided by the agent loop so the Task tool can spawn subagents.
    spawn_subagent: Callable[..., str] | None = None
    # Flipped by ExitPlanMode.
    plan_submitted: str | None = None
    cancelled: Callable[[], bool] = lambda: False

    def note(self, line: str) -> None:
        if self.emit:
            self.emit(line)

    def resolve(self, raw_path: str) -> Path:
        """Turn a user/model supplied path into an absolute path."""
        if not raw_path or not str(raw_path).strip():
            raise ToolError("A file path is required")
        candidate = Path(str(raw_path).strip()).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        try:
            return candidate.resolve()
        except OSError as exc:
            raise ToolError(f"Invalid path {raw_path!r}: {exc}") from exc

    def inside_workspace(self, path: Path) -> bool:
        try:
            path.relative_to(self.workspace)
        except ValueError:
            return False
        return True

    def display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(path)


@dataclass
class ToolResult:
    """What a tool hands back.

    `output` is fed to the model; `display` is the one-line summary shown to
    the user; `detail` is optional rich content (a diff, a file listing).
    """

    output: str
    display: str = ""
    detail: str = ""
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def truncated_output(self, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
        if len(self.output) <= limit:
            return self.output
        head = self.output[: limit // 2]
        tail = self.output[-limit // 2 :]
        dropped = len(self.output) - limit
        return f"{head}\n\n... [{dropped} characters truncated] ...\n\n{tail}"


class Tool(ABC):
    name: str = ""
    description: str = ""
    # JSON Schema for the arguments object.
    parameters: dict[str, Any] = {}
    # False for tools that only read.
    mutating: bool = True
    # Shown while the tool runs.
    verb: str = "Running"

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ...

    # ------------------------------------------------------------ metadata
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """One-line description of a pending call, e.g. `Read(src/app.py)`."""
        specifier = tool_specifier(self.name, args)
        return f"{self.name}({specifier})" if specifier else self.name

    def permission_request(self, args: dict[str, Any], ctx: ToolContext) -> PermissionRequest:
        specifier = tool_specifier(self.name, args)
        return PermissionRequest(
            tool=self.name,
            specifier=specifier,
            title=self.summarize_call(args, ctx),
            mutating=self.mutating,
            suggestions=(f"{self.name}({specifier})", self.name) if specifier else (self.name,),
        )

    def validate(self, args: dict[str, Any]) -> None:
        """Check required properties are present before running."""
        required = (self.parameters or {}).get("required") or []
        missing = [key for key in required if args.get(key) in (None, "")]
        if missing:
            raise ToolError(
                f"{self.name} is missing required argument(s): {', '.join(missing)}"
            )


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Tools need a name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in self.names()]

    def schemas(self, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
        return [
            tool.schema() for tool in self.all() if not exclude or tool.name not in exclude
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
