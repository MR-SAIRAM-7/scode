from __future__ import annotations

from pathlib import Path

import pytest

from scode.config import Settings, load_settings
from scode.permissions import (
    Answer,
    Decision,
    PermissionEngine,
    PermissionRequest,
    Rule,
    bash_specifier,
    suggest_bash_rules,
    tool_specifier,
)


def request(tool: str = "Bash", specifier: str = "git status", mutating: bool = True) -> PermissionRequest:
    return PermissionRequest(tool=tool, specifier=specifier, title=tool, mutating=mutating)


def engine(settings: Settings, mode: str = "default", **kwargs) -> PermissionEngine:
    eng = PermissionEngine(settings, **kwargs)
    eng.set_mode(mode)
    return eng


# ------------------------------------------------------------------- rules

@pytest.mark.parametrize(
    "raw, tool, pattern",
    [
        ("Bash", "Bash", None),
        ("Bash(git status:*)", "Bash", "git status:*"),
        ("Edit(src/**)", "Edit", "src/**"),
        ("  Read( *.py )  ", "Read", "*.py"),
    ],
)
def test_rule_parse(raw: str, tool: str, pattern: str | None) -> None:
    rule = Rule.parse(raw)
    assert rule.tool == tool
    assert rule.pattern == pattern


def test_bare_rule_matches_any_specifier() -> None:
    assert Rule.parse("Bash").matches("Bash", "anything at all")
    assert not Rule.parse("Bash").matches("Edit", "anything")


def test_prefix_rule() -> None:
    rule = Rule.parse("Bash(git status:*)")
    assert rule.matches("Bash", "git status --short")
    assert not rule.matches("Bash", "git push origin main")


def test_glob_rule() -> None:
    rule = Rule.parse("Edit(src/**)")
    assert rule.matches("Edit", "src/deep/file.py")
    assert not rule.matches("Edit", "tests/file.py")


def test_wildcard_tool() -> None:
    assert Rule.parse("*").matches("Bash", "rm -rf build")


def test_rule_render_roundtrip() -> None:
    assert Rule.parse("Bash(ls:*)").render() == "Bash(ls:*)"
    assert Rule.parse("Read").render() == "Read"


# ------------------------------------------------------------------ modes

def test_read_only_tools_are_always_allowed(settings: Settings) -> None:
    eng = engine(settings)
    assert eng.check(request("Read", "app.py", mutating=False)) is Decision.ALLOW


def test_default_mode_asks_for_bash(settings: Settings) -> None:
    assert engine(settings).check(request()) is Decision.ASK


def test_accept_edits_allows_edits_but_asks_for_bash(settings: Settings) -> None:
    eng = engine(settings, "acceptEdits")
    assert eng.check(request("Edit", "app.py")) is Decision.ALLOW
    assert eng.check(request("Bash", "npm test")) is Decision.ASK


def test_plan_mode_denies_every_mutation(settings: Settings) -> None:
    eng = engine(settings, "plan")
    assert eng.check(request("Edit", "app.py")) is Decision.DENY
    assert eng.check(request("Bash", "ls")) is Decision.DENY
    assert eng.check(request("Read", "app.py", mutating=False)) is Decision.ALLOW


def test_bypass_allows_everything(settings: Settings) -> None:
    assert engine(settings, "bypassPermissions").check(request()) is Decision.ALLOW


def test_deny_rules_beat_bypass(workspace: Path) -> None:
    settings = load_settings(workspace, {"denied_tools": ("Bash(rm:*)",)})
    eng = engine(settings, "bypassPermissions")
    assert eng.check(request("Bash", "rm -rf build")) is Decision.DENY
    assert eng.check(request("Bash", "ls")) is Decision.ALLOW


def test_allow_rules_skip_the_prompt(workspace: Path) -> None:
    settings = load_settings(workspace, {"allowed_tools": ("Bash(git status:*)",)})
    eng = engine(settings)
    assert eng.check(request("Bash", "git status --short")) is Decision.ALLOW
    assert eng.check(request("Bash", "git push")) is Decision.ASK


# -------------------------------------------------------------- authorize

def test_authorize_without_an_asker_denies(settings: Settings) -> None:
    allowed, reason = engine(settings).authorize(request())
    assert allowed is False
    assert "non-interactive" in reason


def test_authorize_asks_and_accepts_once(settings: Settings) -> None:
    calls: list[PermissionRequest] = []

    def asker(req: PermissionRequest) -> tuple[Answer, str | None]:
        calls.append(req)
        return Answer.ONCE, None

    eng = engine(settings, asker=asker)
    assert eng.authorize(request())[0] is True
    assert len(calls) == 1
    # "Once" must not persist a rule, so the next call asks again.
    assert eng.check(request()) is Decision.ASK


def test_authorize_remembers_always(settings: Settings) -> None:
    eng = engine(settings, asker=lambda req: (Answer.ALWAYS, "Bash(git status:*)"))
    assert eng.authorize(request())[0] is True
    assert eng.check(request("Bash", "git status --short")) is Decision.ALLOW
    # And it was written to the project's local settings.
    local = settings.workspace / ".scode" / "settings.local.json"
    assert local.is_file()
    assert "Bash(git status:*)" in local.read_text(encoding="utf-8")


def test_authorize_declines(settings: Settings) -> None:
    eng = engine(settings, asker=lambda req: (Answer.NO, None))
    allowed, reason = eng.authorize(request())
    assert allowed is False
    assert "declined" in reason


def test_plan_mode_denial_explains_itself(settings: Settings) -> None:
    allowed, reason = engine(settings, "plan").authorize(request("Edit", "a.py"))
    assert allowed is False
    assert "ExitPlanMode" in reason


def test_session_rules_are_not_persisted_when_asked(settings: Settings) -> None:
    eng = engine(settings)
    eng.remember("Bash(ls:*)", persist=False)
    assert eng.check(request("Bash", "ls -la")) is Decision.ALLOW
    assert not (settings.workspace / ".scode" / "settings.local.json").exists()


# ------------------------------------------------------------- specifiers

def test_bash_specifier_normalises_whitespace() -> None:
    assert bash_specifier("  git   status  --short ") == "git status --short"


def test_tool_specifier_picks_the_right_field() -> None:
    assert tool_specifier("Bash", {"command": "ls  -la"}) == "ls -la"
    assert tool_specifier("Edit", {"file_path": "src/a.py"}) == "src/a.py"
    assert tool_specifier("Grep", {"pattern": "TODO"}) == "TODO"
    assert tool_specifier("WebFetch", {"url": "https://x.dev"}) == "https://x.dev"
    assert tool_specifier("Unknown", {}) == ""


def test_suggest_bash_rules_narrow_then_broad() -> None:
    assert suggest_bash_rules("git status --short") == ("Bash(git status:*)", "Bash(git:*)")
    assert suggest_bash_rules("ls -la") == ("Bash(ls:*)",)
    assert suggest_bash_rules("") == ()


def test_describe_lists_rules(workspace: Path) -> None:
    settings = load_settings(workspace, {"allowed_tools": ("Read",), "denied_tools": ("Bash",)})
    text = engine(settings).describe()
    assert "mode: default" in text
    assert "allow: Read" in text
    assert "deny: Bash" in text
