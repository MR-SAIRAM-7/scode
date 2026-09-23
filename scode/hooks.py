"""User-configured hooks, compatible with Claude Code's hook protocol.

Configured under "hooks" in settings.json:

    "hooks": {
      "PreToolUse": [
        {"matcher": "Bash|Write", "hooks": [{"type": "command", "command": "./check.sh"}]}
      ]
    }

Each hook command receives a JSON object on stdin describing the event.
Exit code 0 means proceed; exit code 2 blocks, and stderr is shown to the
model (tool events, Stop) or the user (UserPromptSubmit). Any other exit code
is a non-blocking error shown to the user. A hook may instead print a JSON
object: {"decision": "block", "reason": "..."}, {"continue": false},
or {"hookSpecificOutput": {"permissionDecision": "allow" | "deny" | "ask",
"additionalContext": "..."}}.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "UserPromptSubmit",
    "Stop",
    "SubagentStop",
    "SessionStart",
    "SessionEnd",
    "PreCompact",
)
# Events whose matcher is compared against the tool name.
TOOL_EVENTS = frozenset({"PreToolUse", "PostToolUse"})
DEFAULT_TIMEOUT = 60


@dataclass
class HookOutcome:
    blocked: bool = False
    # Why it blocked: fed to the model for tool and Stop events.
    reason: str = ""
    # PreToolUse only: "allow" skips the permission prompt, "deny" refuses.
    permission: str | None = None
    # Extra context to hand the model (UserPromptSubmit, SessionStart).
    context: list[str] = field(default_factory=list)
    # Notes for the user: non-blocking failures and the like.
    notices: list[str] = field(default_factory=list)
    # {"continue": false}: stop the whole turn.
    halt: bool = False

    def merge(self, other: HookOutcome) -> None:
        self.blocked = self.blocked or other.blocked
        if other.reason:
            self.reason = "\n".join(filter(None, [self.reason, other.reason]))
        # A deny from any hook wins over an allow from another.
        if other.permission == "deny" or (other.permission and self.permission is None):
            self.permission = other.permission
        self.context.extend(other.context)
        self.notices.extend(other.notices)
        self.halt = self.halt or other.halt


@dataclass(frozen=True)
class _Hook:
    command: str
    timeout: int = DEFAULT_TIMEOUT


def _matches(matcher: str, subject: str) -> bool:
    if not matcher or matcher == "*":
        return True
    try:
        return re.fullmatch(matcher, subject) is not None
    except re.error:
        return matcher == subject


class HookRunner:
    def __init__(
        self,
        config: dict[str, Any] | None,
        workspace: Path,
        *,
        session_id: str = "",
        transcript_path: str = "",
    ) -> None:
        self.workspace = workspace
        self.session_id = session_id
        self.transcript_path = transcript_path
        self._table: dict[str, list[tuple[str, _Hook]]] = {}
        self.errors: list[str] = []
        self._load(config or {})

    def _load(self, config: dict[str, Any]) -> None:
        for event, groups in config.items():
            if event not in EVENTS:
                self.errors.append(f"Unknown hook event {event!r} (known: {', '.join(EVENTS)})")
                continue
            if not isinstance(groups, list):
                self.errors.append(f"hooks.{event} must be a list")
                continue
            for group in groups:
                if not isinstance(group, dict):
                    continue
                matcher = str(group.get("matcher") or "")
                for hook in group.get("hooks") or []:
                    if not isinstance(hook, dict) or hook.get("type", "command") != "command":
                        continue
                    command = str(hook.get("command") or "").strip()
                    if not command:
                        continue
                    timeout = int(hook.get("timeout") or DEFAULT_TIMEOUT)
                    self._table.setdefault(event, []).append((matcher, _Hook(command, timeout)))

    def has(self, event: str) -> bool:
        return bool(self._table.get(event))

    def describe(self) -> list[tuple[str, str, str]]:
        return [
            (event, matcher or "*", hook.command)
            for event, entries in sorted(self._table.items())
            for matcher, hook in entries
        ]

    def run(self, event: str, payload: dict[str, Any], *, subject: str = "") -> HookOutcome:
        outcome = HookOutcome()
        for matcher, hook in self._table.get(event, []):
            if event in TOOL_EVENTS and not _matches(matcher, subject):
                continue
            outcome.merge(self._run_one(event, hook, payload))
        return outcome

    def _run_one(self, event: str, hook: _Hook, payload: dict[str, Any]) -> HookOutcome:
        from .tools.shell import detect_shell

        body = {
            "session_id": self.session_id,
            "transcript_path": self.transcript_path,
            "cwd": str(self.workspace),
            "hook_event_name": event,
            **payload,
        }
        env = dict(os.environ)
        env["SCODE_PROJECT_DIR"] = str(self.workspace)
        env["CLAUDE_PROJECT_DIR"] = str(self.workspace)  # scripts written for Claude Code

        shell = detect_shell()
        try:
            completed = subprocess.run(
                [shell.executable, *shell.args, hook.command],
                input=json.dumps(body),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(self.workspace),
                env=env,
                timeout=hook.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return HookOutcome(notices=[f"{event} hook timed out after {hook.timeout}s: {hook.command}"])
        except OSError as exc:
            return HookOutcome(notices=[f"{event} hook could not run ({exc}): {hook.command}"])

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()

        if completed.returncode == 2:
            return HookOutcome(blocked=True, reason=stderr or f"Blocked by a {event} hook.")
        if completed.returncode != 0:
            note = stderr or f"exit code {completed.returncode}"
            return HookOutcome(notices=[f"{event} hook failed: {note}"])

        return self._parse_stdout(event, stdout)

    @staticmethod
    def _parse_stdout(event: str, stdout: str) -> HookOutcome:
        outcome = HookOutcome()
        if not stdout:
            return outcome
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            # Plain text on success: context for prompt-shaped events.
            if event in {"UserPromptSubmit", "SessionStart"}:
                outcome.context.append(stdout)
            return outcome

        if data.get("continue") is False:
            outcome.halt = True
            outcome.reason = str(data.get("stopReason") or "Stopped by a hook.")

        decision = data.get("decision")
        if decision == "block":
            outcome.blocked = True
            outcome.reason = str(data.get("reason") or f"Blocked by a {event} hook.")
        elif decision == "approve" and event == "PreToolUse":
            outcome.permission = "allow"

        specific = data.get("hookSpecificOutput")
        if isinstance(specific, dict):
            permission = specific.get("permissionDecision")
            if permission in {"allow", "deny", "ask"}:
                outcome.permission = permission
                if permission == "deny":
                    outcome.blocked = True
                    outcome.reason = str(
                        specific.get("permissionDecisionReason") or "Denied by a PreToolUse hook."
                    )
            extra = specific.get("additionalContext")
            if isinstance(extra, str) and extra.strip():
                outcome.context.append(extra.strip())
        return outcome


def empty_runner(workspace: Path) -> HookRunner:
    return HookRunner({}, workspace)
