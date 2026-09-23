"""Hooks: the Claude Code protocol (stdin JSON, exit 2 blocks, JSON decisions)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import sent_text, text_turn, tool_turn

from scode.hooks import HookRunner

PYTHON = Path(sys.executable).as_posix()


def script(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / f"{name}.py"
    path.write_text("import json, sys\npayload = json.load(sys.stdin)\n" + body, encoding="utf-8")
    return f'"{PYTHON}" "{path.as_posix()}"'


def hooks(event: str, command: str, matcher: str = "") -> dict:
    return {event: [{"matcher": matcher, "hooks": [{"type": "command", "command": command}]}]}


# ------------------------------------------------------------------ runner

def test_exit_two_blocks_with_stderr_as_the_reason(tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "block", "sys.stderr.write('no rm allowed'); sys.exit(2)")
    outcome = HookRunner(hooks("PreToolUse", command), workspace).run(
        "PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "rm x"}}, subject="Bash")
    assert outcome.blocked and outcome.reason == "no rm allowed"


def test_payload_describes_the_event(tmp_path: Path, workspace: Path) -> None:
    record = tmp_path / "seen.json"
    command = script(tmp_path, "record", f"open(r'{record.as_posix()}', 'w').write(json.dumps(payload))")
    runner = HookRunner(hooks("PreToolUse", command), workspace, session_id="s1", transcript_path="t.jsonl")
    runner.run("PreToolUse", {"tool_name": "Edit", "tool_input": {"file_path": "a.py"}}, subject="Edit")
    seen = json.loads(record.read_text(encoding="utf-8"))
    assert seen["hook_event_name"] == "PreToolUse"
    assert seen["tool_name"] == "Edit" and seen["tool_input"] == {"file_path": "a.py"}
    assert seen["session_id"] == "s1" and seen["cwd"] == str(workspace)


def test_matchers_are_regexes_over_the_tool_name(tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "block", "sys.exit(2)")
    runner = HookRunner(hooks("PreToolUse", command, matcher="Edit|Write"), workspace)
    assert runner.run("PreToolUse", {}, subject="Write").blocked
    assert not runner.run("PreToolUse", {}, subject="Bash").blocked
    assert not runner.run("PreToolUse", {}, subject="WriteMore").blocked  # full match only


def test_json_permission_decisions(tmp_path: Path, workspace: Path) -> None:
    allow = script(tmp_path, "allow", "print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'allow'}}))")
    deny = script(tmp_path, "deny", "print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'deny', "
                                    "'permissionDecisionReason': 'policy'}}))")
    assert HookRunner(hooks("PreToolUse", allow), workspace).run("PreToolUse", {}, subject="X").permission == "allow"
    denied = HookRunner(hooks("PreToolUse", deny), workspace).run("PreToolUse", {}, subject="X")
    assert denied.blocked and denied.permission == "deny" and denied.reason == "policy"


def test_other_exit_codes_are_non_blocking_notices(tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "fail", "sys.stderr.write('lint crashed'); sys.exit(1)")
    outcome = HookRunner(hooks("PostToolUse", command), workspace).run("PostToolUse", {}, subject="Edit")
    assert not outcome.blocked and "lint crashed" in outcome.notices[0]


def test_plain_stdout_is_context_for_prompt_hooks(tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "ctx", "print('Current sprint: payments')")
    outcome = HookRunner(hooks("UserPromptSubmit", command), workspace).run("UserPromptSubmit", {"prompt": "hi"})
    assert outcome.context == ["Current sprint: payments"]


def test_continue_false_halts(tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "halt", "print(json.dumps({'continue': False, 'stopReason': 'budget spent'}))")
    outcome = HookRunner(hooks("PostToolUse", command), workspace).run("PostToolUse", {}, subject="Bash")
    assert outcome.halt and outcome.reason == "budget spent"


def test_timeouts_are_notices(tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "slow", "import time; time.sleep(5)")
    config = {"PreToolUse": [{"hooks": [{"type": "command", "command": command, "timeout": 1}]}]}
    outcome = HookRunner(config, workspace).run("PreToolUse", {}, subject="X")
    assert not outcome.blocked and "timed out" in outcome.notices[0]


def test_unknown_events_are_reported(workspace: Path) -> None:
    runner = HookRunner({"BeforeLunch": []}, workspace)
    assert runner.errors and "BeforeLunch" in runner.errors[0]


def test_describe_lists_hooks(workspace: Path) -> None:
    runner = HookRunner(hooks("Stop", "echo done", matcher=""), workspace)
    assert runner.describe() == [("Stop", "*", "echo done")]


# ------------------------------------------------------------- in the loop

def test_pre_tool_use_block_stops_the_call(make_agent, tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "block", "sys.stderr.write('Bash is disabled here'); sys.exit(2)")
    runner = HookRunner(hooks("PreToolUse", command, matcher="Bash"), workspace)
    agent, _ = make_agent([tool_turn("Bash", {"command": "echo ran > ran.txt"}), text_turn("ok")], hooks=runner)
    agent.run("go")
    assert not (workspace / "ran.txt").exists()
    result = next(m for m in agent.messages if m.get("role") == "tool")
    assert "Bash is disabled here" in result["content"] and result["is_error"]


def test_pre_tool_use_allow_skips_the_prompt(make_agent, tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "allow", "print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'allow'}}))")
    runner = HookRunner(hooks("PreToolUse", command), workspace)
    # default mode with no asker would normally deny Bash
    agent, _ = make_agent([tool_turn("Bash", {"command": "echo approved-by-hook"}), text_turn("ok")],
                          permission_mode="default", hooks=runner)
    agent.run("go")
    assert "approved-by-hook" in next(m for m in agent.messages if m.get("role") == "tool")["content"]


def test_post_tool_use_feedback_reaches_the_model(make_agent, tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "lint", "sys.stderr.write('lint: missing docstring'); sys.exit(2)")
    runner = HookRunner(hooks("PostToolUse", command, matcher="Edit|Write"), workspace)
    agent, _ = make_agent([tool_turn("Write", {"file_path": "new.py", "content": "x = 1\n"}), text_turn("ok")],
                          hooks=runner)
    agent.run("go")
    result = next(m for m in agent.messages if m.get("role") == "tool")
    assert "[PostToolUse hook] lint: missing docstring" in result["content"]
    assert (workspace / "new.py").exists()  # the write itself happened


def test_user_prompt_submit_can_block(make_agent, tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "guard", "sys.stderr.write('no secrets in prompts'); sys.exit(2)")
    runner = HookRunner(hooks("UserPromptSubmit", command), workspace)
    agent, provider = make_agent([text_turn("never")], hooks=runner)
    result = agent.run("my password is hunter2")
    assert result.reason == "blocked" and not provider.requests


def test_user_prompt_submit_context_is_delivered(make_agent, tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "ctx", "print('Deploy freeze until Friday.')")
    runner = HookRunner(hooks("UserPromptSubmit", command), workspace)
    agent, provider = make_agent([text_turn("noted")], hooks=runner)
    agent.run("ship it")
    assert "Deploy freeze until Friday." in sent_text(provider.requests[0])


def test_stop_hook_sends_the_model_back_once(make_agent, tmp_path: Path, workspace: Path) -> None:
    marker = tmp_path / "stopped-once"
    command = script(
        tmp_path, "stop",
        f"import os\nm = r'{marker.as_posix()}'\n"
        "if not os.path.exists(m):\n    open(m, 'w').close()\n    sys.stderr.write('run the tests first'); sys.exit(2)\n",
    )
    runner = HookRunner(hooks("Stop", command), workspace)
    agent, provider = make_agent([text_turn("done early"), text_turn("tests pass, done")], hooks=runner)
    result = agent.run("fix it")
    assert result.text == "tests pass, done"
    assert "run the tests first" in sent_text(provider.requests[1])


def test_stop_hooks_cannot_loop_forever(make_agent, tmp_path: Path, workspace: Path) -> None:
    command = script(tmp_path, "nag", "sys.stderr.write('again'); sys.exit(2)")
    runner = HookRunner(hooks("Stop", command), workspace)
    agent, provider = make_agent([text_turn(f"attempt {i}") for i in range(10)], hooks=runner)
    agent.run("go")
    assert len(provider.requests) == 4  # the first try plus three hook-driven retries


@pytest.mark.parametrize("event", ["SessionStart", "SessionEnd", "PreCompact", "SubagentStop"])
def test_lifecycle_events_are_accepted(event: str, workspace: Path) -> None:
    assert HookRunner(hooks(event, "echo hi"), workspace).has(event)
