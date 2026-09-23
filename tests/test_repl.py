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

    # One fake stands in for every provider the session builds, tracking the model.
    provider = FakeProvider([])

    def build(settings, *args, **kwargs):
        provider.model = kwargs.get("model") or settings.model
        return provider

    monkeypatch.setattr("scode.runtime.build_provider", build)

    settings = load_settings(
        workspace,
        {"api_key": "nvapi-test", "stream": False, "permission_mode": "bypassPermissions"},
    )
    instance = repl_module.Repl(settings, UI(quiet=True))
    yield instance
    instance.close()


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

    original = repl.spawn_subagent

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


def test_set_mode_leaves_the_system_prompt_alone(repl) -> None:
    """Mode changes are reminders: editing the prompt would reset the provider cache."""
    before = repl.agent.messages[0]["content"]
    provider = script(repl, [text_turn("ok")])
    repl.set_mode("plan")
    repl._dispatch("plan the change")
    assert repl.agent.messages[0]["content"] == before
    assert any("Plan mode is ON" in str(m.get("content")) for m in provider.requests[0])


# ------------------------------------------------------------ new behaviour

def test_shift_tab_cycles_modes(repl) -> None:
    repl.permissions.set_mode("default")
    seen = []
    for _ in range(4):
        repl._cycle_mode()
        seen.append(repl.permissions.mode)
    assert seen == ["acceptEdits", "plan", "default", "acceptEdits"]


def test_hash_notes_reach_the_model_this_session(repl) -> None:
    provider = script(repl, [text_turn("ok")])
    repl._dispatch("#use tabs for indentation")
    repl._dispatch("format the file")
    assert any("use tabs for indentation" in str(m.get("content")) for m in provider.requests[0])


def test_custom_commands_dispatch(repl, workspace: Path) -> None:
    folder = workspace / ".scode" / "commands"
    folder.mkdir(parents=True)
    (folder / "explain.md").write_text("Explain $ARGUMENTS in one paragraph.", encoding="utf-8")
    from scode.extensions import load_commands

    repl.custom_commands = load_commands(workspace)
    provider = script(repl, [text_turn("explained")])
    repl._dispatch("/explain the agent loop")
    assert provider.requests[0][-1]["content"] == "Explain the agent loop in one paragraph."


def test_builtin_commands_win_over_custom_ones(repl, workspace: Path) -> None:
    from scode.extensions import CustomCommand

    repl.custom_commands = {"exit": CustomCommand("exit", "never", workspace / "x.md")}
    assert repl._dispatch("/exit") is True


def test_set_provider_switches_everything(repl) -> None:
    repl.set_provider("openrouter")
    assert repl.settings.provider == "openrouter"
    assert repl.settings.model == "anthropic/claude-opus-5"
    assert repl.provider.model == "anthropic/claude-opus-5"
    assert repl.agent.settings.provider == "openrouter"


def test_provider_colon_model_through_set_model(repl) -> None:
    repl.set_model("anthropic:sonnet")
    assert (repl.settings.provider, repl.settings.model) == ("anthropic", "claude-sonnet-5")


def test_history_survives_a_provider_switch(repl, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    provider = script(repl, [text_turn("first answer"), text_turn("second answer")])
    repl._dispatch("hello")
    repl.set_provider("openrouter")
    repl._dispatch("still there?")
    assert any(m.get("content") == "first answer" for m in provider.requests[1])


def test_ask_user_question_uses_the_chooser(repl, monkeypatch) -> None:
    monkeypatch.setattr("scode.repl.choose", lambda *a, **k: 1)
    answer = repl._ask_user("Which DB?", [{"label": "Postgres"}, {"label": "SQLite"}], False)
    assert answer == ["SQLite"]


def test_ask_user_question_free_text(repl, monkeypatch) -> None:
    monkeypatch.setattr("scode.repl.choose", lambda *a, **k: 2)  # "Something else"
    monkeypatch.setattr("scode.repl.read_line", lambda prompt: "MySQL, actually")
    assert repl._ask_user("Which DB?", [{"label": "Postgres"}, {"label": "SQLite"}], False) == ["MySQL, actually"]


def test_custom_agent_runs_with_its_own_prompt_and_tools(repl, workspace: Path) -> None:
    folder = workspace / ".scode" / "agents"
    folder.mkdir(parents=True)
    (folder / "auditor.md").write_text(
        "---\ndescription: Audits code\ntools: Read, Grep\n---\nYou audit code for security bugs.",
        encoding="utf-8",
    )
    from scode.extensions import load_agents

    repl.custom_agents = load_agents(workspace)
    provider = script(repl, [text_turn("no issues found")])
    report = repl.spawn_subagent(prompt="audit app.py", subagent_type="auditor", read_only=False, label="audit")

    assert report == "no issues found"
    system = provider.requests[0][0]["content"]
    assert "You audit code for security bugs." in system
    tools = {t["function"]["name"] for t in provider.tools_seen[0]}
    assert tools == {"Grep", "Read"}


@pytest.mark.parametrize("spec", ["inherit", "small", "nvidia/nemotron-3-ultra-550b-a55b"])
def test_custom_agent_model_field(repl, workspace: Path, spec: str) -> None:
    folder = workspace / ".scode" / "agents"
    folder.mkdir(parents=True)
    (folder / "helper.md").write_text(f"---\nmodel: {spec}\n---\nYou help.", encoding="utf-8")
    from scode.extensions import load_agents

    repl.custom_agents = load_agents(workspace)
    provider = script(repl, [text_turn("ok")])
    expected = {
        "inherit": repl.settings.model,
        "small": repl.settings.small_model,
    }.get(spec, spec)
    assert repl.spawn_subagent(prompt="x", subagent_type="helper", read_only=True, label="h") == "ok"
    assert provider.model == expected


def test_rewind_restores_files(repl, workspace: Path) -> None:
    script(repl, [
        tool_turn("Read", {"file_path": "app.py"}, call_id="r"),
        tool_turn("Edit", {"file_path": "app.py", "old_string": "a + b", "new_string": "a * b"}, call_id="e"),
        tool_turn("Write", {"file_path": "new.py", "content": "x = 1\n"}, call_id="w"),
        text_turn("done"),
    ])
    repl._dispatch("change things")
    assert "a * b" in (workspace / "app.py").read_text(encoding="utf-8")

    from scode.commands import resolve

    resolve("rewind").handler(repl, "")
    assert "a + b" in (workspace / "app.py").read_text(encoding="utf-8")
    assert not (workspace / "new.py").exists()
    assert not repl.agent.ctx.checkpoints


def test_close_stops_background_shells(repl, workspace: Path) -> None:
    import sys

    from scode.tools.shell import BashTool

    python = Path(sys.executable).as_posix()
    BashTool().run({"command": f'"{python}" -c "import time; time.sleep(30)"', "run_in_background": True},
                   repl.agent.ctx)
    shell = repl.agent.ctx.shells["bash_1"]
    repl.close()
    assert shell.status in {"killed"} or shell.status.startswith("exited")


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


def test_switching_to_a_keyless_provider_explains_how_to_fix_it(repl, capsys) -> None:
    repl.ui.quiet = False
    repl.set_provider("openrouter")
    repl._dispatch("hello")
    out = capsys.readouterr().out
    assert "OPENROUTER_API_KEY" in out and "/login openrouter" in out
    assert "openrouter.ai/keys" in out
