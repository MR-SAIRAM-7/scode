"""The loop behaviours added for production use: reminders, parallel reads,
interrupt safety, refusals, truncated tool calls, costs."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from conftest import multi_tool_turn, sent_text, text_turn, tool_turn

from scode.providers.base import AssistantMessage, ToolCall
from scode.providers.catalog import Catalog, set_catalog


def tool_messages(agent) -> list[dict]:
    return [m for m in agent.messages if m.get("role") == "tool"]


# ---------------------------------------------------------------- reminders

def test_system_prompt_is_frozen_across_mode_changes(make_agent) -> None:
    agent, provider = make_agent([text_turn("one"), text_turn("two")])
    agent.run("first")
    agent.set_mode("plan")
    agent.run("second")
    assert provider.requests[0][0] == provider.requests[1][0]


def test_mode_change_reaches_the_model_as_a_reminder(make_agent) -> None:
    agent, provider = make_agent([text_turn("one"), text_turn("two")])
    agent.run("first")
    agent.set_mode("plan")
    agent.run("second")
    later = sent_text(provider.requests[1])
    assert "<system-reminder>" in later and "Plan mode is ON" in later


def test_leaving_plan_mode_is_announced(make_agent) -> None:
    agent, provider = make_agent([text_turn("ok")], permission_mode="plan")
    agent.set_mode("acceptEdits")
    agent.run("go")
    assert "Plan mode is OFF" in sent_text(provider.requests[0])


def test_setting_the_same_mode_adds_nothing(make_agent) -> None:
    agent, provider = make_agent([text_turn("ok")])
    agent.set_mode(agent.ctx.permissions.mode)
    agent.run("go")
    assert "<system-reminder>" not in sent_text(provider.requests[0])


def test_history_is_append_only_between_requests(make_agent) -> None:
    """Earlier messages must be byte-identical in the next request (caching, thinking)."""
    agent, provider = make_agent([tool_turn("LS", {}), text_turn("done"), text_turn("again")])
    agent.run("one")
    agent.set_mode("acceptEdits")
    agent.run("two")
    first, second, third = provider.requests
    assert second[: len(first)] == first
    assert third[: len(second)] == second


def test_reminders_are_delivered_once(make_agent) -> None:
    agent, provider = make_agent([text_turn("a"), text_turn("b")])
    agent.remind("note to self")
    agent.run("first")
    agent.run("second")
    assert sent_text(provider.requests[1]).count("note to self") == 1


# ------------------------------------------------------------ parallel reads

def test_adjacent_reads_run_concurrently(make_agent, workspace: Path, monkeypatch) -> None:
    from scode.tools.file_tools import ReadTool

    (workspace / "b.py").write_text("B\n", encoding="utf-8")
    active = {"now": 0, "peak": 0}
    lock = threading.Lock()
    original = ReadTool.run

    def slow_run(self, args, ctx):
        with lock:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        time.sleep(0.15)
        try:
            return original(self, args, ctx)
        finally:
            with lock:
                active["now"] -= 1

    monkeypatch.setattr(ReadTool, "run", slow_run)
    agent, _ = make_agent([
        multi_tool_turn(("Read", {"file_path": "app.py"}), ("Read", {"file_path": "b.py"})),
        text_turn("done"),
    ])
    agent.run("read both")
    assert active["peak"] == 2


def test_parallel_results_keep_the_call_order(make_agent, workspace: Path) -> None:
    (workspace / "b.py").write_text("SECOND FILE\n", encoding="utf-8")
    agent, _ = make_agent([
        multi_tool_turn(("Read", {"file_path": "app.py"}), ("Glob", {"pattern": "*.py"}),
                        ("Read", {"file_path": "b.py"})),
        text_turn("done"),
    ])
    agent.run("go")
    results = tool_messages(agent)
    assert [m["tool_call_id"] for m in results] == ["c0", "c1", "c2"]
    assert "def add" in results[0]["content"] and "SECOND FILE" in results[2]["content"]


def test_writes_are_not_parallelised(make_agent, workspace: Path) -> None:
    agent, _ = make_agent([
        multi_tool_turn(("Read", {"file_path": "app.py"}),
                        ("Edit", {"file_path": "app.py", "old_string": "a + b", "new_string": "a - b"}),
                        ("Read", {"file_path": "app.py"})),
        text_turn("done"),
    ])
    agent.run("go")
    # The second Read runs after the Edit and sees its effect.
    assert "a - b" in tool_messages(agent)[2]["content"]


# ---------------------------------------------------------- interrupt safety

def test_interrupt_mid_batch_still_answers_every_call(make_agent, monkeypatch) -> None:
    """A tool call without a result makes the next request fail on every provider."""
    from scode.tools.search_tools import LSTool

    def interrupted(self, args, ctx):
        raise KeyboardInterrupt

    monkeypatch.setattr(LSTool, "run", interrupted)
    agent, _ = make_agent([multi_tool_turn(("Bash", {"command": "echo hi"}), ("LS", {}),
                                           ("Bash", {"command": "echo later"}))])
    result = agent.run("go")

    assert result.reason == "interrupted"
    ids = [m["tool_call_id"] for m in tool_messages(agent)]
    assert ids == ["c0", "c1", "c2"]
    assert "Interrupted" in tool_messages(agent)[2]["content"]
    assert agent.messages[-1]["content"].startswith("[The user interrupted")


# ------------------------------------------------------------ stop reasons

def test_refusal_ends_the_turn_without_appending(make_agent) -> None:
    refused = AssistantMessage(content="", finish_reason="refusal", refusal_category="cyber")
    agent, _ = make_agent([refused])
    result = agent.run("something")
    assert result.reason == "refused"
    assert not any(m.get("role") == "assistant" for m in agent.messages)


def test_truncated_tool_call_is_not_run(make_agent, workspace: Path) -> None:
    truncated = AssistantMessage(
        tool_calls=[ToolCall(id="t1", name="Write", arguments={"file_path": "half.py", "content": "def f("},
                             raw_arguments="{}")],
        finish_reason="length",
    )
    agent, provider = make_agent([truncated, text_turn("smaller steps now")])
    result = agent.run("write a big file")

    assert not (workspace / "half.py").exists()
    assert result.text == "smaller steps now"
    assert "output token limit" in sent_text(provider.requests[1])


def test_repeated_truncation_gives_up(make_agent) -> None:
    def cut() -> AssistantMessage:
        return AssistantMessage(
            tool_calls=[ToolCall(id="t", name="LS", arguments={}, raw_arguments="{}")],
            finish_reason="max_tokens",
        )

    agent, _ = make_agent([cut(), cut(), cut()])
    assert agent.run("go").reason == "error"


def test_pause_turn_resumes_without_a_user_message(make_agent) -> None:
    paused = AssistantMessage(content="working...", finish_reason="pause_turn")
    agent, provider = make_agent([paused, text_turn("finished")])
    assert agent.run("go").text == "finished"
    assert provider.requests[1][-1]["role"] == "assistant"


# --------------------------------------------------------------- accounting

def test_cost_comes_from_catalog_prices(make_agent, settings) -> None:
    set_catalog(Catalog({"nvidia": {"models": {"fake/model": {
        "cost": {"input": 2.0, "output": 10.0, "cache_read": 0.2}}}}}))
    reply = AssistantMessage(content="hi", input_tokens=1_000_000, output_tokens=100_000, cached_tokens=500_000)
    agent, _ = make_agent([reply])
    agent.run("hello")
    # 1M * $2 + 0.1M * $10 + 0.5M * $0.20 = 2 + 1 + 0.1
    assert abs(agent.usage.cost_usd - 3.1) < 1e-9
    assert agent.usage.fully_priced


def test_provider_reported_cost_wins(make_agent) -> None:
    reply = AssistantMessage(content="hi", input_tokens=10, output_tokens=5, reported_cost=0.0123)
    agent, _ = make_agent([reply])
    agent.run("hello")
    assert agent.usage.cost_usd == 0.0123


def test_unpriced_models_are_marked(make_agent) -> None:
    agent, _ = make_agent([text_turn("hi")])
    agent.run("hello")
    assert not agent.usage.fully_priced
    assert agent.usage.cost_label() == "unknown"


def test_context_uses_real_token_counts(make_agent) -> None:
    reply = AssistantMessage(content="hi", input_tokens=4000, cached_tokens=6000, output_tokens=50)
    agent, _ = make_agent([reply])
    agent.run("hello")
    assert agent.usage.context_tokens >= 10_050
