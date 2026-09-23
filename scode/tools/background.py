"""Background shells: long-running commands the agent can check on later."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any

from ..errors import ToolError
from .base import Tool, ToolContext, ToolResult

MAX_BUFFERED_LINES = 5_000


class BackgroundShell:
    def __init__(self, shell_id: str, command: str, process: subprocess.Popen) -> None:
        self.id = shell_id
        self.command = command
        self.process = process
        self.started = time.time()
        self._lines: deque[str] = deque(maxlen=MAX_BUFFERED_LINES)
        self._total = 0
        self._read_upto = 0
        self._lock = threading.Lock()
        self.killed = False
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                threading.Thread(target=self._pump, args=(stream,), daemon=True).start()

    def _pump(self, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                with self._lock:
                    self._lines.append(line.rstrip("\n"))
                    self._total += 1
        except (OSError, ValueError):
            pass

    @property
    def status(self) -> str:
        code = self.process.poll()
        if self.killed:
            return "killed"
        return "running" if code is None else f"exited ({code})"

    def new_output(self, pattern: str | None = None) -> tuple[list[str], int]:
        """Lines produced since the last read, and how many were dropped."""
        with self._lock:
            first_kept = self._total - len(self._lines)
            start = max(self._read_upto, first_kept)
            dropped = start - self._read_upto
            lines = list(self._lines)[start - first_kept :]
            self._read_upto = self._total
        if pattern:
            try:
                regex = re.compile(pattern)
            except re.error as exc:
                raise ToolError(f"Invalid filter regex {pattern!r}: {exc}") from exc
            lines = [line for line in lines if regex.search(line)]
        return lines, dropped

    def kill(self) -> None:
        if self.process.poll() is not None:
            return
        self.killed = True
        try:
            if sys.platform == "win32":
                # terminate() would only stop the shell, leaving its children running.
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


def start_background(ctx: ToolContext, command: str, spec: Any, env: dict[str, str]) -> BackgroundShell:
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(
        [spec.executable, *spec.args, command],
        cwd=str(ctx.workspace),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        **kwargs,
    )
    shell_id = f"bash_{len(ctx.shells) + 1}"
    shell = BackgroundShell(shell_id, command, process)
    ctx.shells[shell_id] = shell
    return shell


def kill_all(ctx: ToolContext) -> int:
    count = 0
    for shell in list(ctx.shells.values()):
        if shell.status == "running":
            shell.kill()
            count += 1
    return count


class BashOutputTool(Tool):
    name = "BashOutput"
    description = (
        "Read new output from a background shell started with Bash(run_in_background=true). "
        "Returns only lines produced since the last read, plus the shell's status."
    )
    mutating = False
    verb = "Reading"
    parameters = {
        "type": "object",
        "properties": {
            "bash_id": {"type": "string", "description": "The id Bash returned, e.g. bash_1"},
            "filter": {"type": "string", "description": "Only return lines matching this regex"},
        },
        "required": ["bash_id"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"BashOutput({args.get('bash_id', '')})"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        shell = ctx.shells.get(str(args["bash_id"]))
        if shell is None:
            known = ", ".join(ctx.shells) or "none"
            raise ToolError(f"No background shell {args['bash_id']!r}. Running: {known}")
        lines, dropped = shell.new_output(args.get("filter") or None)
        header = f"[{shell.id}: {shell.status}]"
        if dropped:
            header += f" [{dropped} older lines were dropped]"
        body = "\n".join(lines) if lines else "(no new output)"
        return ToolResult(
            output=f"{header}\n{body}",
            display=f"{shell.id} {shell.status}, {len(lines)} new lines",
        )


class KillShellTool(Tool):
    name = "KillShell"
    description = "Stop a background shell (and everything it started) by its id."
    mutating = False
    verb = "Stopping"
    parameters = {
        "type": "object",
        "properties": {"shell_id": {"type": "string", "description": "e.g. bash_1"}},
        "required": ["shell_id"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"KillShell({args.get('shell_id', '')})"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        shell = ctx.shells.get(str(args["shell_id"]))
        if shell is None:
            raise ToolError(f"No background shell {args['shell_id']!r}")
        shell.kill()
        return ToolResult(output=f"{shell.id} stopped ({shell.status}).", display=f"Stopped {shell.id}")
