from __future__ import annotations

from pathlib import Path

from conftest import text_turn, tool_turn

from scode.providers.base import AssistantMessage, ToolCall


def _tool_results(agent) -> list[str]:
    """Text-protocol tool results, which arrive as user messages."""
    return [
        str(m["content"])
        for m in agent.messages
        if m.get("role") == "user" and "<tool_result" in str(m.get("content", ""))
    ]


# ------------------------------------------------------------- simple turns

def test_plain_answer_finishes_in_one_step(make_agent) -> None:
    agent, provider = make_agent([text_turn("The answer is 4.")])
    result = agent.run("what is 2+2?")

    assert result.text == "The answer is 4."
    assert result.reason == "done"
    assert result.steps == 1
    assert len(provider.requests) == 1


def test_system_prompt_is_first(make_agent) -> None:
    agent, provider = make_agent([text_turn("hi")])
    agent.run("hello")
    sent = provider.requests[0]
    assert sent[0]["role"] == "system"
    assert "scode" in sent[0]["content"]
    assert sent[1] == {"role": "user", "content": "hello"}


def test_tool_schemas_are_sent(make_agent) -> None:
    agent, provider = make_agent([text_turn("hi")])
    agent.run("hello")
    names = {t["function"]["name"] for t in provider.tools_seen[0]}
    assert {"Read", "Write", "Edit", "Bash", "Glob", "Grep", "TodoWrite"} <= names


# ---------------------------------------------------------------- tool use

def test_tool_call_then_answer(make_agent, workspace: Path) -> None:
    agent, provider = make_agent([
        tool_turn("Read", {"file_path": "app.py"}),
        text_turn("It defines add()."),
    ])
    result = agent.run("what does app.py do?")

    assert result.text == "It defines add()."
    assert result.steps == 2
    assert agent.usage.tool_calls == 1

    tool_message = next(m for m in agent.messages if m.get("role") == "tool")
    assert tool_message["tool_call_id"] == "call_1"
    assert "def add(a, b):" in tool_message["content"]


def test_edit_flows_through_the_loop(make_agent, workspace: Path) -> None:
    agent, _ = make_agent([
        tool_turn("Read", {"file_path": "app.py"}, call_id="c1"),
        tool_turn("Edit", {"file_path": "app.py", "old_string": "a + b", "new_string": "a * b"}, call_id="c2"),
        text_turn("Changed addition to multiplication."),
    ])
    result = agent.run("make it multiply")

    assert result.reason == "done"
    assert (workspace / "app.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a * b\n"


def test_parallel_tool_calls_all_run(make_agent, workspace: Path) -> None:
    turn = AssistantMessage(tool_calls=[
        ToolCall(id="c1", name="Read", arguments={"file_path": "app.py"}, raw_arguments="{}"),
        ToolCall(id="c2", name="LS", arguments={}, raw_arguments="{}"),
    ])
    agent, _ = make_agent([turn, text_turn("done")])
    agent.run("look around")

    tool_ids = [m["tool_call_id"] for m in agent.messages if m.get("role") == "tool"]
    assert tool_ids == ["c1", "c2"]


def test_unknown_tool_is_reported_back(make_agent) -> None:
    agent, _ = make_agent([tool_turn("Teleport", {}), text_turn("ok")])
    agent.run("go")

    tool_message = next(m for m in agent.messages if m.get("role") == "tool")
    assert "no tool named 'Teleport'" in tool_message["content"]
    assert "Available tools:" in tool_message["content"]


def test_tool_errors_are_fed_back_not_raised(make_agent) -> None:
    agent, _ = make_agent([
        tool_turn("Read", {"file_path": "missing.py"}),
        text_turn("That file does not exist."),
    ])
    result = agent.run("read missing.py")

    assert result.reason == "done"
    tool_message = next(m for m in agent.messages if m.get("role") == "tool")
    assert "does not exist" in tool_message["content"]


def test_malformed_arguments_are_reported_back(make_agent) -> None:
    bad = AssistantMessage(tool_calls=[ToolCall.from_raw("c1", "Read", "{broken")])
    agent, _ = make_agent([bad, text_turn("retrying")])
    agent.run("go")

    tool_message = next(m for m in agent.messages if m.get("role") == "tool")
    assert "valid JSON" in tool_message["content"]


def test_a_crashing_tool_does_not_kill_the_turn(make_agent, monkeypatch) -> None:
    from scode.tools.search_tools import LSTool

    def boom(self, args, ctx):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(LSTool, "run", boom)
    agent, _ = make_agent([tool_turn("LS", {}), text_turn("recovered")])
    result = agent.run("list")

    assert result.reason == "done"
    tool_message = next(m for m in agent.messages if m.get("role") == "tool")
    assert "RuntimeError: kaboom" in tool_message["content"]


# -------------------------------------------------------------- permissions

def test_denied_tool_call_is_explained_to_the_model(make_agent) -> None:
    agent, _ = make_agent(
        [tool_turn("Bash", {"command": "rm -rf build"}), text_turn("understood")],
        permission_mode="default",  # no asker is wired, so this denies
    )
    agent.run("clean the build")

    tool_message = next(m for m in agent.messages if m.get("role") == "tool")
    assert "was not run" in tool_message["content"]


def test_plan_mode_blocks_edits(make_agent, workspace: Path) -> None:
    agent, _ = make_agent(
        [tool_turn("Write", {"file_path": "new.txt", "content": "x"}), text_turn("ok")],
        permission_mode="plan",
    )
    agent.run("create a file")

    assert not (workspace / "new.txt").exists()
    tool_message = next(m for m in agent.messages if m.get("role") == "tool")
    assert "plan mode" in tool_message["content"].lower()


def test_plan_mode_allows_reads(make_agent) -> None:
    agent, _ = make_agent(
        [tool_turn("Read", {"file_path": "app.py"}), text_turn("Here is the plan.")],
        permission_mode="plan",
    )
    result = agent.run("plan a change")
    assert result.text == "Here is the plan."


def test_allow_rule_skips_the_prompt(make_agent) -> None:
    agent, _ = make_agent(
        [tool_turn("Bash", {"command": "echo hi"}), text_turn("done")],
        permission_mode="default",
        allowed_tools=("Bash(echo:*)",),
    )
    agent.run("say hi")

    tool_message = next(m for m in agent.messages if m.get("role") == "tool")
    assert "hi" in tool_message["content"]


# ------------------------------------------------------------- plan approval

def test_exit_plan_mode_switches_mode_when_approved(make_agent) -> None:
    agent, _ = make_agent(
        [tool_turn("ExitPlanMode", {"plan": "1. do it"}), text_turn("doing it")],
        permission_mode="plan",
    )
    agent.plan_approver = lambda plan: True
    result = agent.run("plan it")

    assert agent.ctx.permissions.mode == "acceptEdits"
    assert result.text == "doing it"


def test_exit_plan_mode_stops_when_declined(make_agent) -> None:
    agent, _ = make_agent([tool_turn("ExitPlanMode", {"plan": "1. do it"})], permission_mode="plan")
    agent.plan_approver = lambda plan: False
    result = agent.run("plan it")

    assert agent.ctx.permissions.mode == "plan"
    assert result.reason == "done"
    tool_message = next(m for m in agent.messages if m.get("role") == "tool")
    assert "did not approve" in tool_message["content"]


def test_exit_plan_without_an_approver_stops(make_agent) -> None:
    agent, _ = make_agent([tool_turn("ExitPlanMode", {"plan": "x"})], permission_mode="plan")
    result = agent.run("plan")
    assert result.reason == "done"


# ------------------------------------------------------------- text protocol

def test_text_protocol_parses_tool_use_blocks(make_agent) -> None:
    block = 'Let me look.\n<tool_use>\n{"name": "Read", "input": {"file_path": "app.py"}}\n</tool_use>'
    agent, provider = make_agent(
        [text_turn(block), text_turn("It adds numbers.")],
        native_tools=False,
    )
    result = agent.run("what does it do?")

    assert result.text == "It adds numbers."
    assert provider.tools_seen[0] is None  # no schemas sent in text mode

    results = _tool_results(agent)
    assert results and "def add(a, b):" in results[0]


def test_text_protocol_strips_the_block_from_the_reply(make_agent) -> None:
    block = 'Checking.\n<tool_use>\n{"name": "LS", "input": {}}\n</tool_use>'
    agent, _ = make_agent([text_turn(block), text_turn("done")], native_tools=False)
    agent.run("look")

    assistant = next(m for m in agent.messages if m.get("role") == "assistant")
    assert "<tool_use>" not in assistant["content"]
    assert assistant["content"] == "Checking."


def test_text_protocol_reports_bad_json(make_agent) -> None:
    agent, _ = make_agent(
        [text_turn("<tool_use>\n{not json}\n</tool_use>"), text_turn("sorry")],
        native_tools=False,
    )
    agent.run("go")

    results = _tool_results(agent)
    assert results and "could not parse" in results[0]


def test_text_protocol_system_prompt_describes_tools(make_agent) -> None:
    agent, provider = make_agent([text_turn("hi")], native_tools=False)
    agent.run("hello")
    system = provider.requests[0][0]["content"]
    assert "<tool_use>" in system
    assert "## Read" in system


def test_falls_back_when_the_endpoint_rejects_tools(make_agent) -> None:
    from scode.providers.openai_compatible import ToolsNotSupported

    agent, provider = make_agent([text_turn("hello without tools")], stream=True)
    original = provider.stream
    calls = {"n": 0}

    def flaky(messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ToolsNotSupported("nope")
        return original(messages, **kwargs)

    provider.stream = flaky  # type: ignore[assignment]
    result = agent.run("hi")

    assert result.text == "hello without tools"
    assert agent._native_tools is False


# ------------------------------------------------------------------- limits

def test_max_steps_stops_the_loop(make_agent) -> None:
    turns = [tool_turn("LS", {}, call_id=f"c{i}") for i in range(10)]
    agent, _ = make_agent(turns, max_steps=3)
    result = agent.run("loop forever")

    assert result.reason == "max_steps"
    assert result.steps == 3


def test_length_finish_reason_warns(make_agent) -> None:
    truncated = AssistantMessage(content="half an ans", finish_reason="length")
    agent, _ = make_agent([truncated])
    result = agent.run("write a novel")
    assert result.reason == "done"
    assert result.text == "half an ans"


def test_provider_errors_end_the_turn_cleanly(make_agent) -> None:
    from scode.errors import ProviderError

    agent, provider = make_agent([text_turn("never reached")], stream=True)

    def boom(messages, **kwargs):
        raise ProviderError("provider exploded")

    provider.stream = boom  # type: ignore[assignment]
    result = agent.run("hi")

    assert result.reason == "error"
    assert "exploded" in (result.error or "")


def test_interrupt_is_recorded(make_agent) -> None:
    from scode.errors import Interrupted

    agent, provider = make_agent([text_turn("x")], stream=True)

    def boom(messages, **kwargs):
        raise Interrupted("stop")

    provider.stream = boom  # type: ignore[assignment]
    result = agent.run("hi")

    assert result.reason == "interrupted"
    assert "interrupted" in agent.messages[-1]["content"].lower()


# -------------------------------------------------------------- bookkeeping

def test_context_tokens_are_tracked(make_agent) -> None:
    agent, _ = make_agent([text_turn("ok")])
    agent.run("hello")
    assert agent.usage.context_tokens > 0


def test_on_message_receives_every_message(make_agent) -> None:
    agent, _ = make_agent([tool_turn("LS", {}), text_turn("done")])
    seen: list[dict] = []
    agent.on_message = seen.append
    agent.run("list files")

    roles = [m["role"] for m in seen]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_refresh_system_prompt_replaces_in_place(make_agent) -> None:
    agent, _ = make_agent([text_turn("ok")])
    before = agent.messages[0]["content"]
    agent.ctx.permissions.set_mode("plan")
    agent.refresh_system_prompt()

    assert agent.messages[0]["role"] == "system"
    assert agent.messages[0]["content"] != before
    assert "Plan mode is ON" in agent.messages[0]["content"]


def test_project_memory_is_included(make_agent, workspace: Path) -> None:
    (workspace / "SCODE.md").write_text("Always use tabs.\n", encoding="utf-8")
    agent, provider = make_agent([text_turn("ok")])
    agent.refresh_system_prompt()
    agent.run("hi")

    assert "Always use tabs." in provider.requests[0][0]["content"]
