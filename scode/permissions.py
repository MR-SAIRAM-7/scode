"""Permission engine: decides whether a tool call runs, asks, or is refused.

Rule syntax mirrors Claude Code:

    Bash                    every Bash call
    Bash(git status:*)      any command starting with "git status"
    Bash(ls *)              glob match against the command
    Edit(src/**)            glob match against the path the tool touches
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import Settings


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class Rule:
    tool: str
    pattern: str | None = None

    @staticmethod
    def parse(raw: str) -> Rule:
        text = raw.strip()
        if text.endswith(")") and "(" in text:
            tool, _, rest = text.partition("(")
            return Rule(tool.strip(), rest[:-1].strip() or None)
        return Rule(text, None)

    def matches(self, tool: str, specifier: str) -> bool:
        if self.tool != tool and self.tool != "*":
            return False
        if self.pattern is None:
            return True
        pattern = self.pattern
        if pattern.endswith(":*"):
            return specifier.startswith(pattern[:-2])
        return fnmatch.fnmatch(specifier, pattern) or specifier == pattern

    def render(self) -> str:
        return f"{self.tool}({self.pattern})" if self.pattern else self.tool


@dataclass
class PermissionRequest:
    tool: str
    specifier: str
    title: str
    detail: str = ""
    mutating: bool = True
    # Rules offered as "don't ask again" choices, broadest last.
    suggestions: tuple[str, ...] = ()


# Answer returned by the interactive prompt.
class Answer(str, Enum):
    ONCE = "once"
    ALWAYS = "always"
    NO = "no"


Asker = Callable[[PermissionRequest], tuple[Answer, str | None]]


@dataclass
class PermissionEngine:
    settings: Settings
    asker: Asker | None = None
    session_allow: list[Rule] = field(default_factory=list)
    session_deny: list[Rule] = field(default_factory=list)
    mode_override: str | None = None

    # Tools that never mutate anything and never leave the machine.
    READ_ONLY = frozenset({"Read", "Glob", "Grep", "LS", "TodoWrite", "NotebookRead"})
    # Tools blocked outright while planning.
    MUTATING = frozenset({"Write", "Edit", "MultiEdit", "Bash", "WebFetch"})

    @property
    def mode(self) -> str:
        return self.mode_override or self.settings.permission_mode

    def set_mode(self, mode: str) -> None:
        self.mode_override = mode

    def _configured(self) -> tuple[list[Rule], list[Rule]]:
        allow = [Rule.parse(r) for r in self.settings.allowed_tools] + self.session_allow
        deny = [Rule.parse(r) for r in self.settings.denied_tools] + self.session_deny
        return allow, deny

    def check(self, request: PermissionRequest) -> Decision:
        allow, deny = self._configured()

        for rule in deny:
            if rule.matches(request.tool, request.specifier):
                return Decision.DENY

        if self.mode == "bypassPermissions":
            return Decision.ALLOW

        if request.tool in self.READ_ONLY or not request.mutating:
            return Decision.ALLOW

        if self.mode == "plan":
            return Decision.DENY

        for rule in allow:
            if rule.matches(request.tool, request.specifier):
                return Decision.ALLOW

        if self.mode == "acceptEdits" and request.tool in {"Write", "Edit", "MultiEdit"}:
            return Decision.ALLOW

        return Decision.ASK

    def authorize(self, request: PermissionRequest) -> tuple[bool, str]:
        """Resolve a request, prompting the user when the rules do not decide it.

        Returns (allowed, reason).
        """
        decision = self.check(request)
        if decision is Decision.ALLOW:
            return True, "allowed"
        if decision is Decision.DENY:
            if self.mode == "plan":
                return False, (
                    "Blocked by plan mode: scode is only reading right now. "
                    "Present the plan and call ExitPlanMode for approval before changing anything."
                )
            return False, "Blocked by a deny rule in your settings."

        if self.asker is None:
            return False, (
                f"{request.tool} is disabled in this non-interactive session, because "
                "there is nobody to approve it. Do not try this call again. Finish with "
                "what you already have, and say plainly what you could not do."
            )

        answer, rule_text = self.asker(request)
        if answer is Answer.NO:
            return False, "The user declined this action."
        if answer is Answer.ALWAYS and rule_text:
            self.remember(rule_text)
        return True, "approved by user"

    def remember(self, rule_text: str, *, deny: bool = False, persist: bool = True) -> None:
        rule = Rule.parse(rule_text)
        (self.session_deny if deny else self.session_allow).append(rule)
        if persist:
            from .config import save_project_permission

            try:
                save_project_permission(self.settings.workspace, rule.render(), deny=deny)
            except OSError:
                pass  # An unwritable project dir must not break the session.

    def describe(self) -> str:
        allow, deny = self._configured()
        parts = [f"mode: {self.mode}"]
        if allow:
            parts.append("allow: " + ", ".join(r.render() for r in allow))
        if deny:
            parts.append("deny: " + ", ".join(r.render() for r in deny))
        return "\n".join(parts)


def bash_specifier(command: str) -> str:
    """Normalise a shell command for rule matching."""
    return " ".join(command.strip().split())


def suggest_bash_rules(command: str) -> tuple[str, ...]:
    """Offer progressively broader allow-rules for a shell command."""
    parts = bash_specifier(command).split()
    if not parts:
        return ()
    head = parts[0]
    suggestions = [f"Bash({head}:*)"]
    if len(parts) > 1 and not parts[1].startswith("-"):
        suggestions.insert(0, f"Bash({head} {parts[1]}:*)")
    return tuple(suggestions)


def tool_specifier(tool: str, tool_input: dict[str, Any]) -> str:
    """The string a rule pattern is matched against for a given tool call."""
    if tool == "Bash":
        return bash_specifier(str(tool_input.get("command", "")))
    for key in ("file_path", "path", "pattern", "url"):
        if key in tool_input and tool_input[key]:
            return str(tool_input[key])
    return ""
