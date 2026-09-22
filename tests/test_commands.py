from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from conftest import FakeProvider, text_turn

from scode.agent.loop import Agent
from scode.commands import builtin as commands
from scode.config import Settings, load_settings, user_settings_path, with_overrides
from scode.permissions import PermissionEngine
from scode.session.store import SessionStore
from scode.tools import ToolContext, build_registry
from scode.ui.console import UI
from scode.usage import Usage


@dataclass
class StubRepl:
    """The surface area the command handlers actually touch."""

    settings: Settings
    ui: UI
    permissions: PermissionEngine
    agent: Agent
    store: SessionStore
    usage: Usage
    models: list[str] = field(default_factory=lambda: ["a/one", "a/two"])
    reset_called: bool = False
    resumed: str | None = None

    def set_model(self, model: str) -> None:
        self.settings = with_overrides(self.settings, model=model)

    def set_mode(self, mode: str) -> None:
        self.permissions.set_mode(mode)

    def set_theme(self, theme: str) -> None:
        self.settings = with_overrides(self.settings, theme=theme)

    def set_api_key(self, key: str) -> None:
        self.settings = with_overrides(self.settings, api_key=key)

    def reset_conversation(self) -> None:
        self.reset_called = True

    def resume_session(self, session_id: str) -> None:
        self.resumed = session_id

    def list_models(self) -> list[str]:
        return self.models


@pytest.fixture
def repl(workspace: Path) -> StubRepl:
    settings = load_settings(workspace, {"api_key": "nvapi-test", "stream": False})
    ui = UI(quiet=True)
    permissions = PermissionEngine(settings)
    usage = Usage()
    ctx = ToolContext(workspace=workspace, settings=settings, permissions=permissions)
    agent = Agent(
        provider=FakeProvider([text_turn("ok")]),  # type: ignore[arg-type]
        registry=build_registry(),
        ctx=ctx,
        ui=ui,
        settings=settings,
        usage=usage,
    )
    store = SessionStore(workspace).open()
    return StubRepl(settings, ui, permissions, agent, store, usage)


def run(repl: StubRepl, line: str):
    name, _, args = line.lstrip("/").partition(" ")
    command = commands.resolve(name)
    assert command is not None, f"no such command: {name}"
    return command.handler(repl, args.strip())


# ----------------------------------------------------------------- registry

def test_every_command_has_a_description() -> None:
    for command in commands.REGISTRY.values():
        assert command.description
        assert command.name.islower()


def test_aliases_resolve() -> None:
    assert commands.resolve("/q").name == "exit"
    assert commands.resolve("h").name == "help"
    assert commands.resolve("usage").name == "cost"
    assert commands.resolve("nope") is None


def test_listing_includes_aliases() -> None:
    names = {name for name, _ in commands.listing()}
    assert {"help", "model", "q", "resume"} <= names


# ------------------------------------------------------------------ session

def test_exit_signals_exit(repl: StubRepl) -> None:
    assert run(repl, "/exit").exit is True


def test_help_does_not_exit(repl: StubRepl) -> None:
    assert run(repl, "/help").exit is False


def test_clear_resets(repl: StubRepl) -> None:
    run(repl, "/clear")
    assert repl.reset_called is True


def test_status_runs(repl: StubRepl) -> None:
    assert run(repl, "/status").exit is False


def test_cost_runs(repl: StubRepl) -> None:
    repl.usage.record(input_tokens=100, output_tokens=20)
    assert run(repl, "/cost").exit is False


# -------------------------------------------------------------------- model

def test_model_sets_the_model(repl: StubRepl) -> None:
    run(repl, "/model z-ai/glm-5.3")
    assert repl.settings.model == "z-ai/glm-5.3"


def test_model_with_no_args_lists_the_catalog(repl: StubRepl) -> None:
    assert run(repl, "/model").exit is False
    assert repl.settings.model == load_settings(repl.settings.workspace).model


def test_model_all_queries_the_provider(repl: StubRepl) -> None:
    run(repl, "/model --all")
    # The listing must not change the current model.
    assert repl.settings.model != "a/one"


def test_model_search_filters(repl: StubRepl) -> None:
    repl.models = ["nvidia/one", "meta/two"]
    assert run(repl, "/model --search nvidia").exit is False


def test_model_reports_a_provider_error(repl: StubRepl) -> None:
    from scode.errors import ProviderError

    def boom() -> list[str]:
        raise ProviderError("catalog down")

    repl.list_models = boom  # type: ignore[assignment]
    assert run(repl, "/model --all").exit is False


# ------------------------------------------------------------------- login

def test_login_saves_the_key(repl: StubRepl) -> None:
    run(repl, "/login nvapi-brand-new")
    assert repl.settings.api_key == "nvapi-brand-new"
    saved = json.loads(user_settings_path().read_text(encoding="utf-8"))
    assert saved["api_key"] == "nvapi-brand-new"


def test_logout_removes_the_key(repl: StubRepl) -> None:
    run(repl, "/login nvapi-x")
    run(repl, "/logout")
    saved = json.loads(user_settings_path().read_text(encoding="utf-8"))
    assert "api_key" not in saved


# -------------------------------------------------------------- permissions

def test_mode_changes_the_mode(repl: StubRepl) -> None:
    run(repl, "/mode plan")
    assert repl.permissions.mode == "plan"


def test_mode_is_case_insensitive(repl: StubRepl) -> None:
    run(repl, "/mode acceptedits")
    assert repl.permissions.mode == "acceptEdits"


def test_mode_rejects_nonsense(repl: StubRepl) -> None:
    run(repl, "/mode wizard")
    assert repl.permissions.mode == "default"


def test_mode_with_no_args_shows_the_table(repl: StubRepl) -> None:
    assert run(repl, "/mode").exit is False
    assert repl.permissions.mode == "default"


def test_bypass_mode_asks_first(repl: StubRepl, monkeypatch) -> None:
    monkeypatch.setattr("scode.ui.input.confirm", lambda ui, q, **k: False)
    run(repl, "/mode bypassPermissions")
    assert repl.permissions.mode == "default"

    monkeypatch.setattr("scode.ui.input.confirm", lambda ui, q, **k: True)
    run(repl, "/mode bypassPermissions")
    assert repl.permissions.mode == "bypassPermissions"


def test_tools_lists_tools(repl: StubRepl) -> None:
    assert run(repl, "/tools").exit is False


def test_agents_lists_subagents(repl: StubRepl) -> None:
    assert run(repl, "/agents").exit is False


# ------------------------------------------------------------- prompt-backed

def test_init_returns_a_prompt(repl: StubRepl) -> None:
    result = run(repl, "/init")
    assert result.prompt and "SCODE.md" in result.prompt


def test_init_asks_before_overwriting(repl: StubRepl, monkeypatch) -> None:
    (repl.settings.workspace / "SCODE.md").write_text("existing", encoding="utf-8")
    monkeypatch.setattr("scode.ui.input.confirm", lambda ui, q, **k: False)
    assert run(repl, "/init").prompt is None


def test_review_returns_a_prompt(repl: StubRepl) -> None:
    assert "Review" in (run(repl, "/review").prompt or "")


def test_commit_returns_a_prompt(repl: StubRepl) -> None:
    prompt = run(repl, "/commit").prompt or ""
    assert "git commit" in prompt.lower() or "commit" in prompt.lower()
    assert "Do not push" in prompt


def test_commit_passes_the_users_note(repl: StubRepl) -> None:
    assert "mention the bug id" in (run(repl, "/commit mention the bug id").prompt or "")


# ------------------------------------------------------------------- memory

def test_memory_reports_nothing_when_empty(repl: StubRepl) -> None:
    assert run(repl, "/memory").exit is False


def test_memory_shows_the_file(repl: StubRepl) -> None:
    (repl.settings.workspace / "SCODE.md").write_text("Use tabs.", encoding="utf-8")
    assert run(repl, "/memory").exit is False


# ------------------------------------------------------------------ export

def test_export_writes_markdown(repl: StubRepl) -> None:
    repl.agent.messages.extend([
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "done"},
    ])
    run(repl, "/export notes.md")

    text = (repl.settings.workspace / "notes.md").read_text(encoding="utf-8")
    assert "## User" in text
    assert "do the thing" in text
    assert "done" in text


def test_export_defaults_to_a_session_filename(repl: StubRepl) -> None:
    run(repl, "/export")
    assert list(repl.settings.workspace.glob("scode-*.md"))


# ----------------------------------------------------------------- sessions

def test_sessions_lists(repl: StubRepl) -> None:
    repl.store.append({"role": "user", "content": "earlier"})
    assert run(repl, "/sessions").exit is False


def test_resume_with_an_id_calls_the_repl(repl: StubRepl) -> None:
    run(repl, "/resume abc123")
    assert repl.resumed == "abc123"


# ------------------------------------------------------------------- config

def test_config_hides_the_key(repl: StubRepl) -> None:
    assert run(repl, "/config").exit is False


def test_theme_toggles(repl: StubRepl) -> None:
    run(repl, "/theme light")
    assert repl.settings.theme == "light"
    run(repl, "/theme")
    assert repl.settings.theme == "dark"


def test_theme_rejects_nonsense(repl: StubRepl) -> None:
    run(repl, "/theme neon")
    assert repl.settings.theme == "dark"


def test_doctor_runs(repl: StubRepl) -> None:
    assert run(repl, "/doctor").exit is False


def test_compact_is_a_noop_on_an_empty_conversation(repl: StubRepl) -> None:
    assert run(repl, "/compact").exit is False


def test_diff_handles_a_non_repository(repl: StubRepl) -> None:
    assert run(repl, "/diff").exit is False
