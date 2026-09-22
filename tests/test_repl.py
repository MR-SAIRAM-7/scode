from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeProvider, text_turn, tool_turn

from scode.config import load_settings
from scode.permissions import Answer, PermissionRequest
from scode.ui.console import UI


@pytest.fixture
def repl(workspace: Path, monkeypatch: pytest.MonkeyPatch):
    """A real Repl with the interactive input layer stubbed out."""
    from scode import repl as repl_module

    monkeypatch.setattr(repl_module, "InputSession", lambda *a, **k: None)

    provider = FakeProvider([])
    monkeypatch.setattr(repl_module, "get_provider", lambda settings: provider)

    settings = load_settings(
        workspace,
        {"api_key": "nvapi-test", "stream": False, "permission_mode": "bypassPermissions"},
    )
    instance = repl_module.Repl(settings, UI(quiet=True))
    instance.provider = provider
    instance.agent.provider = provider  # type: ignore[assignment]
    return instance


def script(repl, turns) -> FakeProvider:
    repl.provider.turns = list(turns)
    return repl.provider


# ------------------------------------------------------------------ routing

def test_plain_text_goes_to_the_model(repl) -> None:
    provider = script(repl, [text_turn("hello back")])
    assert repl._dispatch("hello") is False
    assert provider.requests


def test_slash_command_is_dispatched(repl) -> None:
    assert repl._dispatch("/exit") is True


def test_unknown_slash_command_does_not_exit(repl) -> None:
    assert repl._dispatch("/nonsense") is False


def test_hash_saves_a_note(repl, workspace: Path) -> None:
    repl._dispatch("#always run make lint")
    assert "always run make lint" in (workspace / "SCODE.md").read_text(encoding="utf-8")


def test_empty_hash_is_ignored(repl, workspace: Path) -> None:
    repl._dispatch("#   ")
    assert not (workspace / "SCODE.md").exists()


def test_bang_runs_a_shell_command(repl) -> None:
    repl._dispatch("!echo from-the-shell")
    last = repl.agent.messages[-1]
    assert last["role"] == "user"
    assert "ran this command themselves" in last["content"]
    assert "from-the-shell" in last["content"]


def test_bang_with_no_command_is_ignored(repl) -> None:
    before = len(repl.agent.messages)
    repl._dispatch("!   ")
    assert len(repl.agent.messages) == before


def test_command_prompts_are_sent_to_the_model(repl) -> None:
    provider = script(repl, [text_turn("wrote it")])
    repl._dispatch("/init")
    assert provider.requests
    assert "SCODE.md" in provider.requests[0][-1]["content"]


def test_mentions_are_expanded_before_sending(repl) -> None:
    provider = script(repl, [text_turn("ok")])
    repl._dispatch("explain @app.py")
    assert "def add(a, b):" in provider.requests[0][-1]["content"]


def test_sending_without_a_key_is_refused(repl) -> None:
    from scode.config import with_overrides

    repl.settings = with_overrides(repl.settings, api_key="")
    provider = script(repl, [text_turn("should not happen")])
    repl._dispatch("hello")
    assert not provider.requests


# -------------------------------------------------------------- permissions

def test_permission_answer_once(repl, monkeypatch) -> None:
    monkeypatch.setattr("scode.repl.choose", lambda *a, **k: 0)
    answer, rule = repl._ask_permission(
        PermissionRequest(tool="Bash", specifier="ls", title="Bash(ls)", suggestions=("Bash(ls:*)",))
    )
    assert answer is Answer.ONCE
    assert rule is None


def test_permission_answer_always(repl, monkeypatch) -> None:
    monkeypatch.setattr("scode.repl.choose", lambda *a, **k: 1)
    answer, rule = repl._ask_permission(
        PermissionRequest(tool="Bash", specifier="ls", title="Bash(ls)", suggestions=("Bash(ls:*)",))
    )
    assert answer is Answer.ALWAYS
    assert rule == "Bash(ls:*)"


def test_permission_answer_no(repl, monkeypatch) -> None:
    monkeypatch.setattr("scode.repl.choose", lambda *a, **k: 2)
    answer, _ = repl._ask_permission(
        PermissionRequest(tool="Bash", specifier="ls", title="Bash(ls)", suggestions=("Bash(ls:*)",))
    )
    assert answer is Answer.NO


def test_permission_without_suggestions_has_two_options(repl, monkeypatch) -> None:
    seen: dict = {}

    def fake_choose(ui, title, options, **kwargs):
        seen["options"] = options
        return 1

    monkeypatch.setattr("scode.repl.choose", fake_choose)
    answer, _ = repl._ask_permission(
        PermissionRequest(tool="Write", specifier="", title="Write", suggestions=())
    )
    assert len(seen["options"]) == 2
    assert answer is Answer.NO


def test_plan_approval_is_wired(repl, monkeypatch) -> None:
    monkeypatch.setattr("scode.repl.choose", lambda *a, **k: 0)
    assert repl._approve_plan("1. do it") is True
    monkeypatch.setattr("scode.repl.choose", lambda *a, **k: 1)
    assert repl._approve_plan("1. do it") is False


def test_the_agent_uses_the_repl_asker(repl, monkeypatch) -> None:
    repl.permissions.set_mode("default")
    monkeypatch.setattr("scode.repl.choose", lambda *a, **k: 0)
    script(repl, [tool_turn("Bash", {"command": "echo approved"}), text_turn("done")])

    repl._dispatch("run echo")
    tool_message = next(m for m in repl.agent.messages if m.get("role") == "tool")
    assert "approved" in tool_message["content"]


# -------------------------------------------------------------- subagents

def test_subagent_runs_and_reports(repl) -> None:
    script(repl, [
        tool_turn("Task", {"description": "find add", "prompt": "where is add?", "subagent_type": "explore"}),
        text_turn("main agent done"),
        # consumed by the subagent:
        text_turn("add() lives in app.py:1"),
    ])
    # The subagent pulls from the same scripted queue, so order matters:
    repl.provider.turns = [
        tool_turn("Task", {"description": "find add", "prompt": "where is add?", "subagent_type": "explore"}),
        text_turn("add() lives in app.py:1"),
        text_turn("The main agent is done."),
    ]
    repl._dispatch("find add")

    tool_message = next(m for m in repl.agent.messages if m.get("role") == "tool")
    assert "app.py:1" in tool_message["content"]


def test_subagent_usage_is_merged(repl) -> None:
    repl.provider.turns = [
        tool_turn("Task", {"description": "x", "prompt": "p"}),
        text_turn("subagent report"),
        text_turn("done"),
    ]
    repl._dispatch("delegate")
    assert repl.usage.requests >= 3


def test_read_only_subagent_has_no_write_tools(repl) -> None:
    captured: dict = {}

    original = repl._spawn_subagent

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    repl.agent.ctx.spawn_subagent = spy
    repl.provider.turns = [
        tool_turn("Task", {"description": "x", "prompt": "p", "subagent_type": "plan"}),
        text_turn("plan report"),
        text_turn("done"),
    ]
    repl._dispatch("plan it")
    assert captured["read_only"] is True


# ----------------------------------------------------------- state changes

def test_set_model_updates_everything(repl) -> None:
    repl.set_model("z-ai/glm-5.3")
    assert repl.settings.model == "z-ai/glm-5.3"
    assert repl.provider.model == "z-ai/glm-5.3"


def test_set_mode_refreshes_the_system_prompt(repl) -> None:
    repl.set_mode("plan")
    assert "Plan mode is ON" in repl.agent.messages[0]["content"]


def test_reset_conversation_starts_a_new_session(repl) -> None:
    old_id = repl.store.id
    repl._dispatch("hello")
    repl.reset_conversation()
    assert repl.store.id != old_id
    assert len(repl.agent.messages) == 1  # system prompt only


def test_resume_restores_messages(repl, workspace: Path) -> None:
    from scode.session.store import SessionStore

    with SessionStore(workspace) as store:
        store.append({"role": "user", "content": "earlier question"})
        store.append({"role": "assistant", "content": "earlier answer"})
        session_id = store.id

    repl.resume_session(session_id)
    contents = [m.get("content") for m in repl.agent.messages]
    assert "earlier question" in contents
    assert "earlier answer" in contents
    assert repl.agent.messages[0]["role"] == "system"


def test_toolbar_is_renderable(repl) -> None:
    text = repl._toolbar()
    assert repl.settings.model in text
    assert "ctx" in text


def test_set_theme_swaps_the_ui(repl) -> None:
    repl.set_theme("light")
    assert repl.settings.theme == "light"
    assert repl.agent.ui is repl.ui
