from __future__ import annotations

from pathlib import Path

from conftest import FakeProvider, text_turn

from scode.agent.compact import _last_safe_split, compact_messages, render_transcript
from scode.agent.context import (
    append_memory,
    environment_block,
    expand_file_mentions,
    load_project_memory,
    memory_files,
)
from scode.config import Settings

# ------------------------------------------------------------- compaction

def conversation() -> list[dict]:
    return [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "Read", "content": "file text"},
        {"role": "assistant", "content": "here is what I found"},
        {"role": "user", "content": "second request"},
        {"role": "assistant", "content": "working on it"},
    ]


def test_split_never_orphans_a_tool_result() -> None:
    messages = conversation()
    split = _last_safe_split(messages)
    assert messages[split]["role"] == "user"
    # Nothing after the split may be a tool message without its assistant call.
    tail = messages[split:]
    assert tail[0]["role"] == "user"


def test_compact_replaces_the_head_with_a_summary(settings: Settings) -> None:
    provider = FakeProvider([text_turn("Summary: the user asked for X; app.py was read.")])
    summary, kept = compact_messages(provider, conversation(), settings=settings)

    assert "the user asked for X" in summary
    assert kept[0]["role"] == "user"
    assert "<session_summary>" in kept[0]["content"]
    assert kept[-1]["content"] == "working on it"
    assert len(kept) < len(conversation())


def test_compact_uses_the_small_model(settings: Settings) -> None:
    from scode.config import with_overrides

    local = with_overrides(settings, small_model="cheap/model")

    captured: dict = {}

    class Recorder(FakeProvider):
        def complete(self, messages, *, tools=None, model=None, **kwargs):
            captured["model"] = model
            return text_turn("summary")

    compact_messages(Recorder(), conversation(), settings=local)
    assert captured["model"] == "cheap/model"


def test_compact_survives_a_provider_failure(settings: Settings) -> None:
    from scode.errors import ProviderError

    class Broken(FakeProvider):
        def complete(self, messages, **kwargs):
            raise ProviderError("down")

    summary, kept = compact_messages(Broken(), conversation(), settings=settings)
    assert "Automatic summary unavailable" in summary
    assert kept


def test_compact_skips_a_short_conversation(settings: Settings) -> None:
    short = [{"role": "user", "content": "hi"}]
    summary, kept = compact_messages(FakeProvider(), short, settings=settings)
    assert summary == ""
    assert kept == short


def test_compact_passes_focus_instructions(settings: Settings) -> None:
    captured: dict = {}

    class Recorder(FakeProvider):
        def complete(self, messages, **kwargs):
            captured["prompt"] = messages[-1]["content"]
            return text_turn("s")

    compact_messages(Recorder(), conversation(), settings=settings, instructions="the retry bug")
    assert "the retry bug" in captured["prompt"]


def test_render_transcript_labels_tool_calls() -> None:
    text = render_transcript(conversation())
    assert "[called: Read({})]" in text
    assert "[result of Read]" in text
    assert "<system>" not in text


def test_agent_auto_compacts_when_the_window_fills(make_agent, monkeypatch) -> None:
    agent, provider = make_agent([text_turn("ok")])
    provider.turns.insert(0, text_turn("summary of earlier work"))

    # Pretend the window is tiny so the threshold trips immediately.
    monkeypatch.setattr(type(agent.settings), "context_window", lambda self: 10)
    agent.messages.extend([
        {"role": "user", "content": "old request " * 50},
        {"role": "assistant", "content": "old reply " * 50},
        {"role": "user", "content": "recent request"},
    ])
    agent.run("go")

    joined = "".join(str(m.get("content", "")) for m in agent.messages)
    assert "<session_summary>" in joined


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
    assert path.name == "SCODE.md"
    append_memory(workspace, "Run make lint")
    text = path.read_text(encoding="utf-8")
    assert "- Prefer pytest" in text
    assert "- Run make lint" in text


# ------------------------------------------------------------- @ mentions

def test_mention_inlines_a_file(workspace: Path) -> None:
    text, attached = expand_file_mentions("explain @app.py please", workspace)
    assert attached == ["app.py"]
    assert '<attached path="app.py">' in text
    assert "def add(a, b):" in text


def test_multiple_mentions(workspace: Path) -> None:
    text, attached = expand_file_mentions("@app.py and @README.md", workspace)
    assert sorted(attached) == ["README.md", "app.py"]


def test_unknown_mention_is_left_alone(workspace: Path) -> None:
    text, attached = expand_file_mentions("check @nope.py", workspace)
    assert attached == []
    assert text == "check @nope.py"


def test_email_like_text_is_not_a_mention(workspace: Path) -> None:
    text, attached = expand_file_mentions("mail me at me@example.com", workspace)
    assert attached == []


def test_mention_of_a_directory_is_ignored(workspace: Path) -> None:
    (workspace / "src").mkdir()
    _, attached = expand_file_mentions("look at @src", workspace)
    assert attached == []
