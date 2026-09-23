from __future__ import annotations

import base64
from pathlib import Path

from conftest import FakeProvider, text_turn, tool_turn

from scode.agent.compact import (
    build_carrier,
    last_user_request,
    render_transcript,
    summarize_in_place,
    summarize_transcript,
)
from scode.agent.context import (
    append_memory,
    environment_block,
    expand_file_mentions,
    load_project_memory,
    memory_files,
)
from scode.config import Settings, with_overrides
from scode.errors import ContextOverflow, ProviderError


def conversation() -> list[dict]:
    return [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "Read", "content": "file text"},
        {"role": "assistant", "content": "here is what I found"},
        {"role": "user", "content": "second request"},
        {"role": "assistant", "content": "working on it"},
    ]


# ------------------------------------------------------------- summarising

def test_in_place_summary_reuses_the_conversation(settings: Settings) -> None:
    """The summary request is the live conversation plus one message: a cache hit."""
    provider = FakeProvider([text_turn("SUMMARY")])
    messages = conversation()
    summary = summarize_in_place(provider, messages, tools=[{"x": 1}])

    assert summary == "SUMMARY"
    sent = provider.requests[0]
    assert sent[: len(messages)] == messages  # identical prefix
    assert "Do not call any tools" in sent[-1]["content"]
    assert provider.tools_seen[0] == [{"x": 1}]  # same tools, so the cache still matches


def test_in_place_summary_gives_up_on_provider_errors() -> None:
    class Broken(FakeProvider):
        def complete(self, *args, **kwargs):
            raise ProviderError("down")

    assert summarize_in_place(Broken(), conversation(), tools=None) is None


def test_transcript_summary_uses_the_small_model(settings: Settings) -> None:
    local = with_overrides(settings, small_model="cheap/model")
    provider = FakeProvider([text_turn("summary")])
    summarize_transcript(provider, conversation()[1:], settings=local)
    assert provider.models_seen == ["cheap/model"]


def test_transcript_summary_falls_back_to_the_request_list(settings: Settings) -> None:
    class Broken(FakeProvider):
        def complete(self, *args, **kwargs):
            raise ProviderError("down")

    summary = summarize_transcript(Broken(), conversation()[1:], settings=settings)
    assert "first request" in summary and "second request" in summary


def test_focus_instructions_reach_the_summariser() -> None:
    provider = FakeProvider([text_turn("s")])
    summarize_in_place(provider, conversation(), tools=None, instructions="the retry bug")
    assert "the retry bug" in provider.requests[0][-1]["content"]


def test_usage_callback_sees_the_summary_request() -> None:
    seen = []
    summarize_in_place(FakeProvider([text_turn("s")]), conversation(), tools=None, on_usage=seen.append)
    assert len(seen) == 1


def test_render_transcript_labels_tool_calls() -> None:
    text = render_transcript(conversation())
    assert "[called: Read({})]" in text
    assert "[result of Read]" in text
    assert "SYSTEM" not in text


# ---------------------------------------------------------------- carrier

def test_last_user_request_skips_scaffolding() -> None:
    messages = conversation() + [
        {"role": "user", "content": "<system-reminder>\nPlan mode is ON\n</system-reminder>"},
        {"role": "user", "content": "[The user interrupted. Stop, and wait for their next message.]"},
    ]
    assert last_user_request(messages) == "second request"


def test_carrier_mid_turn_asks_to_continue() -> None:
    carrier = build_carrier("did things", "fix the bug", mid_turn=True)
    assert carrier["role"] == "user"
    assert "<session_summary>" in carrier["content"]
    assert "fix the bug" in carrier["content"]
    assert "Continue the work" in carrier["content"]


def test_carrier_between_turns_waits() -> None:
    assert "Wait for the user's next message" in build_carrier("x", "", mid_turn=False)["content"]


# ------------------------------------------------------------ in the loop

def test_compaction_replays_nothing_from_the_old_transcript(make_agent) -> None:
    agent, provider = make_agent([text_turn("SUMMARY OF WORK")])
    agent.messages.extend(conversation()[1:])
    agent.compact()

    assert len(agent.messages) == 2
    assert agent.messages[0]["role"] == "system"
    carrier = agent.messages[1]["content"]
    assert "SUMMARY OF WORK" in carrier
    assert "second request" in carrier
    # No tool results or assistant turns survive to be replayed.
    assert all(m["role"] in {"system", "user"} for m in agent.messages)


def test_compaction_notifies_the_transcript(make_agent) -> None:
    agent, _ = make_agent([text_turn("SUMMARY")])
    carriers = []
    agent.on_compact = carriers.append
    agent.messages.extend(conversation()[1:])
    agent.compact()
    assert carriers and "SUMMARY" in carriers[0]["content"]


def test_agent_auto_compacts_when_the_window_fills(make_agent, monkeypatch) -> None:
    agent, provider = make_agent([text_turn("summary of earlier work"), text_turn("ok")])
    monkeypatch.setattr(type(agent.settings), "context_window", lambda self: 10)
    agent.messages.extend([
        {"role": "user", "content": "old request " * 50},
        {"role": "assistant", "content": "old reply " * 50},
    ])
    agent.run("go")

    joined = "".join(str(m.get("content", "")) for m in agent.messages)
    assert "<session_summary>" in joined
    assert "old reply" not in joined


def test_context_overflow_compacts_and_retries(make_agent) -> None:
    agent, provider = make_agent([
        ContextOverflow("too long"),      # the real request overflows
        text_turn("rendered summary"),     # the transcript summary (in-place is skipped)
        text_turn("answer after compaction"),
    ])
    agent.messages.extend(conversation()[1:])
    result = agent.run("next thing")

    assert result.text == "answer after compaction"
    assert result.reason == "done"
    assert "<session_summary>" in agent.messages[1]["content"]


def test_a_second_overflow_is_reported(make_agent) -> None:
    agent, _ = make_agent([ContextOverflow("too long"), text_turn("summary"), ContextOverflow("still")])
    agent.messages.extend(conversation()[1:])
    result = agent.run("next")
    assert result.reason == "error"


# ---------------------------------------------------------------- context

def test_environment_block_reports_the_basics(workspace: Path) -> None:
    text = environment_block(workspace, model="a/b", permission_mode="plan")
    assert str(workspace) in text
    assert "Model: a/b" in text
    assert "Permission mode: plan" in text
    assert "app.py" in text


def test_memory_file_is_loaded(workspace: Path) -> None:
    (workspace / "SCODE.md").write_text("Use tabs.", encoding="utf-8")
    assert "Use tabs." in load_project_memory(workspace)


def test_claude_md_is_honoured(workspace: Path) -> None:
    (workspace / "CLAUDE.md").write_text("Legacy instructions.", encoding="utf-8")
    assert "Legacy instructions." in load_project_memory(workspace)


def test_scode_md_wins_over_claude_md(workspace: Path) -> None:
    (workspace / "SCODE.md").write_text("Primary.", encoding="utf-8")
    (workspace / "CLAUDE.md").write_text("Secondary.", encoding="utf-8")
    memory = load_project_memory(workspace)
    assert "Primary." in memory
    assert "Secondary." not in memory


def test_user_memory_is_included(workspace: Path, isolated_home: Path) -> None:
    (isolated_home / "SCODE.md").write_text("Global rule.", encoding="utf-8")
    assert "Global rule." in load_project_memory(workspace)


def test_no_memory_is_empty(workspace: Path) -> None:
    assert load_project_memory(workspace) == ""
    assert memory_files(workspace) == []


def test_append_memory_creates_and_appends(workspace: Path) -> None:
    path = append_memory(workspace, "Prefer pytest")
    append_memory(workspace, "Run make lint")
    text = path.read_text(encoding="utf-8")
    assert path.name == "SCODE.md"
    assert "- Prefer pytest" in text and "- Run make lint" in text


# ------------------------------------------------------------- @ mentions

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def test_mention_inlines_a_file(workspace: Path) -> None:
    text, attached = expand_file_mentions("explain @app.py please", workspace)
    assert attached == ["app.py"]
    assert '<attached path="app.py">' in text
    assert "def add(a, b):" in text


def test_multiple_mentions(workspace: Path) -> None:
    _, attached = expand_file_mentions("@app.py and @README.md", workspace)
    assert sorted(attached) == ["README.md", "app.py"]


def test_unknown_mention_is_left_alone(workspace: Path) -> None:
    text, attached = expand_file_mentions("check @nope.py", workspace)
    assert attached == [] and text == "check @nope.py"


def test_email_like_text_is_not_a_mention(workspace: Path) -> None:
    _, attached = expand_file_mentions("mail me at me@example.com", workspace)
    assert attached == []


def test_mention_of_a_directory_is_ignored(workspace: Path) -> None:
    (workspace / "src").mkdir()
    _, attached = expand_file_mentions("look at @src", workspace)
    assert attached == []


def test_image_mentions_become_image_parts(workspace: Path) -> None:
    (workspace / "shot.png").write_bytes(PNG)
    content, attached = expand_file_mentions("what's in @shot.png and @app.py", workspace)
    assert attached == ["shot.png", "app.py"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text" and "def add" in content[0]["text"]
    assert content[1] == {"type": "image", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}


def test_images_are_skipped_for_text_only_models(workspace: Path) -> None:
    (workspace / "shot.png").write_bytes(PNG)
    content, attached = expand_file_mentions("see @shot.png", workspace, allow_images=False)
    assert content == "see @shot.png" and attached == []


def test_oversized_images_are_skipped(workspace: Path, monkeypatch) -> None:
    monkeypatch.setattr("scode.agent.context.MAX_IMAGE_BYTES", 10)
    (workspace / "big.png").write_bytes(PNG)
    _, attached = expand_file_mentions("see @big.png", workspace)
    assert attached == []


def test_image_messages_reach_the_provider(make_agent, workspace: Path) -> None:
    (workspace / "shot.png").write_bytes(PNG)
    agent, provider = make_agent([text_turn("a pixel")])
    content, _ = expand_file_mentions("describe @shot.png", workspace)
    agent.run(content)
    user = next(m for m in provider.requests[0] if m["role"] == "user")
    assert any(part.get("type") == "image" for part in user["content"])


def test_tool_turn_helper_is_usable(make_agent) -> None:
    agent, _ = make_agent([tool_turn("LS", {}), text_turn("done")])
    assert agent.run("list").text == "done"
