from __future__ import annotations

import sys

import pytest

from scode.errors import ToolError
from scode.tools import ToolContext
from scode.tools.shell import BashTool, detect_shell


def test_detect_shell_returns_something_runnable() -> None:
    spec = detect_shell()
    assert spec.executable
    assert spec.args


def test_shell_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCODE_SHELL", "powershell")
    spec = detect_shell()
    assert spec.label == "powershell"
    assert "-Command" in spec.args


def test_echo_runs(ctx: ToolContext) -> None:
    result = BashTool().run({"command": "echo hello"}, ctx)
    assert "hello" in result.output
    assert result.is_error is False


def test_runs_in_the_workspace(ctx: ToolContext) -> None:
    tool = BashTool()
    command = "cd" if tool.shell.label == "cmd" else "pwd"
    result = tool.run({"command": command}, ctx)
    assert ctx.workspace.name in result.output


def test_nonzero_exit_is_reported(ctx: ToolContext) -> None:
    result = BashTool().run({"command": "exit 3"}, ctx)
    assert result.is_error is True
    assert "[exit code 3]" in result.output
    assert result.metadata["exit_code"] == 3


def test_stderr_is_captured(ctx: ToolContext) -> None:
    result = BashTool().run({"command": "echo oops 1>&2"}, ctx)
    assert "oops" in result.output


def test_empty_command_is_rejected(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="empty"):
        BashTool().run({"command": "   "}, ctx)


def test_timeout(ctx: ToolContext) -> None:
    tool = BashTool()
    sleep = "Start-Sleep -Seconds 5" if tool.shell.label == "powershell" else "sleep 5"
    with pytest.raises(ToolError, match="timed out"):
        tool.run({"command": sleep, "timeout": 1}, ctx)


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf ~",
        "mkfs.ext4 /dev/sda1",
        ":(){ :|:& };:",
        "format C:",
    ],
)
def test_catastrophic_commands_are_refused(ctx: ToolContext, command: str) -> None:
    with pytest.raises(ToolError, match="Refused"):
        BashTool().run({"command": command}, ctx)


def test_ordinary_rm_is_allowed(ctx: ToolContext) -> None:
    (ctx.workspace / "junk.txt").write_text("x", encoding="utf-8")
    tool = BashTool()
    if tool.shell.label in {"bash", "sh"}:
        tool.run({"command": "rm -rf junk.txt"}, ctx)
        assert not (ctx.workspace / "junk.txt").exists()


def test_risky_commands_get_a_warning(ctx: ToolContext) -> None:
    request = BashTool().permission_request({"command": "git push origin main"}, ctx)
    assert "can change state outside the workspace" in request.detail


def test_permission_request_suggests_rules(ctx: ToolContext) -> None:
    request = BashTool().permission_request({"command": "npm run test -- --watch"}, ctx)
    assert request.suggestions[0] == "Bash(npm run:*)"
    assert request.suggestions[-1] == "Bash"


def test_description_is_used_as_the_title(ctx: ToolContext) -> None:
    request = BashTool().permission_request(
        {"command": "pytest -q", "description": "Run the test suite"}, ctx
    )
    assert request.title == "Run the test suite"


def test_long_commands_are_shortened_in_the_summary(ctx: ToolContext) -> None:
    command = "echo " + "x" * 200
    summary = BashTool().summarize_call({"command": command}, ctx)
    assert summary.endswith("...)")
    assert len(summary) < 90


def test_pager_is_disabled(ctx: ToolContext) -> None:
    tool = BashTool()
    if tool.shell.label in {"bash", "sh"}:
        result = tool.run({"command": "echo $GIT_PAGER"}, ctx)
        assert "cat" in result.output


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific refusal")
def test_windows_recursive_delete_is_refused(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="Refused"):
        BashTool().run({"command": "Remove-Item C:\\ -Recurse"}, ctx)
