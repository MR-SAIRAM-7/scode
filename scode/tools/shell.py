"""The Bash tool: run shell commands in the workspace."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from ..constants import BASH_DEFAULT_TIMEOUT, BASH_MAX_TIMEOUT
from ..errors import ToolError
from ..permissions import PermissionRequest, bash_specifier, suggest_bash_rules
from .base import Tool, ToolContext, ToolResult

# Commands that destroy an entire filesystem or disk are refused outright.
CATASTROPHIC = (
    re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rf][a-zA-Z]*\s+/(\s|$)"),
    re.compile(r"\brm\s+-rf\s+(~|\$HOME|/\*)(\s|$)"),
    re.compile(r"\bmkfs(\.|\s)"),
    re.compile(r"\bdd\s+[^|]*\bof=/dev/(sd|nvme|hd)"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:"),  # fork bomb
    re.compile(r"\bformat\s+[a-zA-Z]:", re.IGNORECASE),
    re.compile(r"Remove-Item\s+.*\b[a-zA-Z]:\\\s*(-Recurse|$)", re.IGNORECASE),
)

# Shown with a warning banner in the approval prompt.
RISKY = (
    re.compile(r"\brm\s+-[a-zA-Z]*r"),
    re.compile(r"\bgit\s+(push|reset\s+--hard|clean\s+-[a-z]*f|checkout\s+--)"),
    re.compile(r"\b(sudo|doas)\b"),
    re.compile(r"\bcurl\b[^|]*\|\s*(ba)?sh"),
    re.compile(r"\bnpm\s+publish\b|\bpip\s+install\b.*--break-system-packages"),
    re.compile(r"\bdrop\s+(table|database)\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b|\breboot\b"),
)


@dataclass(frozen=True)
class ShellSpec:
    executable: str
    args: tuple[str, ...]
    label: str


def detect_shell() -> ShellSpec:
    """Pick a shell: SCODE_SHELL, then bash, then the platform default."""
    override = os.getenv("SCODE_SHELL")
    if override:
        resolved = shutil.which(override) or override
        name = os.path.basename(resolved).lower()
        if "powershell" in name or name.startswith("pwsh"):
            return ShellSpec(resolved, ("-NoProfile", "-NonInteractive", "-Command"), "powershell")
        if name.startswith("cmd"):
            return ShellSpec(resolved, ("/d", "/c"), "cmd")
        return ShellSpec(resolved, ("-lc",), name or "shell")

    bash = shutil.which("bash")
    if bash:
        return ShellSpec(bash, ("-lc",), "bash")

    if sys.platform == "win32":
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if pwsh:
            return ShellSpec(pwsh, ("-NoProfile", "-NonInteractive", "-Command"), "powershell")
        return ShellSpec(os.environ.get("COMSPEC", "cmd.exe"), ("/d", "/c"), "cmd")

    return ShellSpec(shutil.which("sh") or "/bin/sh", ("-c",), "sh")


class BashTool(Tool):
    name = "Bash"
    verb = "Running"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run"},
            "description": {
                "type": "string",
                "description": "5-10 word description shown to the user",
            },
            "timeout": {
                "type": "integer",
                "description": f"Timeout in seconds (default {BASH_DEFAULT_TIMEOUT}, max {BASH_MAX_TIMEOUT})",
            },
        },
        "required": ["command"],
    }

    def __init__(self) -> None:
        self.shell = detect_shell()
        self.description = (
            f"Run a shell command in the workspace using {self.shell.label}. "
            "Use it for builds, tests, git and any CLI work. Prefer Read/Glob/Grep "
            "over cat/find/grep. Quote paths that contain spaces."
        )

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        command = bash_specifier(str(args.get("command", "")))
        if len(command) > 70:
            command = command[:67] + "..."
        return f"Bash({command})"

    def permission_request(self, args: dict[str, Any], ctx: ToolContext) -> PermissionRequest:
        command = str(args.get("command", ""))
        note = str(args.get("description") or "").strip()
        warning = ""
        if any(pattern.search(command) for pattern in RISKY):
            warning = "⚠ This command can change state outside the workspace.\n\n"
        detail = warning + command
        if note:
            detail = f"{warning}{note}\n\n{command}"
        return PermissionRequest(
            tool=self.name,
            specifier=bash_specifier(command),
            title=note or self.summarize_call(args, ctx),
            detail=detail,
            mutating=True,
            suggestions=suggest_bash_rules(command) + ("Bash",),
        )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        command = str(args["command"]).strip()
        if not command:
            raise ToolError("command is empty")

        for pattern in CATASTROPHIC:
            if pattern.search(command):
                raise ToolError(
                    "Refused: this command would destroy a filesystem or disk. "
                    "Run it yourself if you truly intend it."
                )

        timeout = int(args.get("timeout") or BASH_DEFAULT_TIMEOUT)
        timeout = max(1, min(timeout, BASH_MAX_TIMEOUT))

        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env["SCODE"] = "1"
        # Stop child CLIs from trying to open pagers or editors.
        env["PAGER"] = "cat"
        env["GIT_PAGER"] = "cat"

        try:
            completed = subprocess.run(
                [self.shell.executable, *self.shell.args, command],
                cwd=str(ctx.workspace),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                env=env,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            partial = _combine(
                exc.stdout if isinstance(exc.stdout, str) else "",
                exc.stderr if isinstance(exc.stderr, str) else "",
            )
            raise ToolError(
                f"Command timed out after {timeout}s. Partial output:\n{partial[-4000:]}"
            ) from exc
        except FileNotFoundError as exc:
            raise ToolError(
                f"Shell {self.shell.executable!r} is not available: {exc}"
            ) from exc
        except OSError as exc:
            raise ToolError(f"Could not run the command: {exc}") from exc

        output = _combine(completed.stdout, completed.stderr).rstrip()
        code = completed.returncode

        if not output:
            output = "(no output)"
        body = output if code == 0 else f"{output}\n\n[exit code {code}]"

        return ToolResult(
            output=body,
            display=_preview(output, code),
            is_error=code != 0,
            metadata={"exit_code": code},
        )


def _preview(output: str, code: int) -> str:
    """One line summarising what the command printed."""
    lines = [line for line in output.splitlines() if line.strip()]
    first = lines[0] if lines else "(no output)"
    if len(first) > 80:
        first = first[:77] + "..."
    if len(lines) > 1:
        first += f"  (+{len(lines) - 1} more lines)"
    if code != 0:
        first += f"  [exit {code}]"
    return first


def _combine(stdout: str | None, stderr: str | None) -> str:
    parts = [part for part in (stdout or "", stderr or "") if part]
    return "\n".join(parts)
