from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]


def build_tools(workspace: Path) -> dict[str, Tool]:
    workspace = workspace.resolve()

    def _safe_path(relative_path: str) -> Path:
        target = (workspace / relative_path).resolve()
        if workspace not in target.parents and target != workspace:
            raise ValueError("Path escapes workspace")
        return target

    def list_files(path: str = ".") -> list[str]:
        root = _safe_path(path)
        if not root.exists() or not root.is_dir():
            raise ValueError("Directory does not exist")
        return sorted(p.name for p in root.iterdir())

    def read_file(path: str) -> str:
        target = _safe_path(path)
        return target.read_text(encoding="utf-8")

    def write_file(path: str, content: str) -> str:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {path}"

    def run_command(command: str) -> str:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        output = completed.stdout + completed.stderr
        return output[-20_000:]

    return {
        "list_files": Tool("list_files", "List files in a directory", list_files),
        "read_file": Tool("read_file", "Read a file", read_file),
        "write_file": Tool("write_file", "Write a file", write_file),
        "run_command": Tool("run_command", "Run a shell command", run_command),
    }
