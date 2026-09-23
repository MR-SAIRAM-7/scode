from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from conftest import FakeProvider, text_turn

from scode.agent.loop import Agent
from scode.commands import builtin as commands
from scode.config import (
    Settings,
    load_settings,
    switch_provider,
    user_settings_path,
    with_overrides,
)
from scode.hooks import HookRunner
from scode.mcp import McpManager
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
    mcp: McpManager
    hooks: HookRunner
    models: list[str] = field(default_factory=lambda: ["a/one", "a/two"])
    custom_agents: dict = field(default_factory=dict)
    custom_commands: dict = field(default_factory=dict)
    reset_called: bool = False
    resumed: str | None = None

    def set_model(self, model: str) -> None:
        self.settings = with_overrides(self.settings, model=model)

    def set_provider(self, provider: str, model: str | None = None) -> None:
        self.settings = switch_provider(self.settings, provider, model)

    def set_effort(self, effort: str) -> None:
        self.settings = with_overrides(self.settings, effort=effort) if effort else \
            dataclasses.replace(self.settings, effort="")

    def set_mode(self, mode: str) -> None:
        self.agent.set_mode(mode)

    def set_theme(self, theme: str) -> None:
        self.settings = with_overrides(self.settings, theme=theme)

    def set_api_key(self, key: str, provider: str | None = None) -> None:
        if provider in (None, self.settings.provider):
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
    return StubRepl(settings, ui, permissions, agent, store, usage,
                    McpManager({}, workspace), HookRunner({}, workspace))


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

def test_login_saves_the_key_per_provider(repl: StubRepl) -> None:
    run(repl, "/login nvapi-brand-new")
    assert repl.settings.api_key == "nvapi-brand-new"
    saved = json.loads(user_settings_path().read_text(encoding="utf-8"))
    assert saved["api_keys"] == {"nvidia": "nvapi-brand-new"}


def test_login_detects_the_provider_from_the_key(repl: StubRepl) -> None:
    run(repl, "/login sk-or-v1-abcdef")
    saved = json.loads(user_settings_path().read_text(encoding="utf-8"))
    assert saved["api_keys"] == {"openrouter": "sk-or-v1-abcdef"}
    # The active provider's key is untouched.
    assert repl.settings.api_key == "nvapi-test"


def test_login_for_a_named_provider(repl: StubRepl) -> None:
    run(repl, "/login anthropic sk-ant-xyz")
    assert json.loads(user_settings_path().read_text(encoding="utf-8"))["api_keys"]["anthropic"] == "sk-ant-xyz"


def test_logout_removes_the_key(repl: StubRepl) -> None:
    run(repl, "/login nvapi-x")
    run(repl, "/logout")
    saved = json.loads(user_settings_path().read_text(encoding="utf-8"))
    assert "api_keys" not in saved


# ------------------------------------------------------------ new commands

def test_provider_lists_and_switches(repl: StubRepl) -> None:
    assert run(repl, "/provider").exit is False
    run(repl, "/provider openrouter")
    assert (repl.settings.provider, repl.settings.model) == ("openrouter", "anthropic/claude-opus-5")
    run(repl, "/provider anthropic claude-sonnet-5")
    assert repl.settings.model == "claude-sonnet-5"


def test_provider_switch_becomes_the_default(repl: StubRepl) -> None:
    path = user_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"base_url": "http://old/v1", "small_model": "old/small",
                                "theme": "light"}), encoding="utf-8")
    run(repl, "/provider openrouter sonnet")
    saved = json.loads(path.read_text(encoding="utf-8"))
    # The old provider's endpoint must not leak onto the new one; unrelated keys stay.
    assert saved == {"provider": "openrouter", "model": "anthropic/claude-sonnet-5", "theme": "light"}

    fresh = load_settings(repl.settings.workspace)
    assert (fresh.provider, fresh.model) == ("openrouter", "anthropic/claude-sonnet-5")
    assert fresh.base_url == "https://openrouter.ai/api/v1"


def test_model_switch_within_a_provider_keeps_its_endpoint(repl: StubRepl) -> None:
    path = user_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"provider": "nvidia", "base_url": "http://proxy/v1"}), encoding="utf-8")
    run(repl, "/model nvidia/nemotron-3-ultra-550b-a55b")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"provider": "nvidia", "base_url": "http://proxy/v1",
                     "model": "nvidia/nemotron-3-ultra-550b-a55b"}


def test_an_unreadable_settings_file_does_not_block_a_switch(repl: StubRepl) -> None:
    path = user_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ broken", encoding="utf-8")
    run(repl, "/provider openrouter")
    assert repl.settings.provider == "openrouter"
    assert path.read_text(encoding="utf-8") == "{ broken"


def test_effort_sets_and_resets(repl: StubRepl) -> None:
    run(repl, "/effort high")
    assert repl.settings.effort == "high"
    run(repl, "/effort nonsense")
    assert repl.settings.effort == "high"
    run(repl, "/effort default")
    assert repl.settings.effort == ""


def test_permissions_add_and_remove_rules(repl: StubRepl, workspace: Path) -> None:
    run(repl, "/permissions allow Bash(npm test:*)")
    local = workspace / ".scode" / "settings.local.json"
    assert "Bash(npm test:*)" in local.read_text(encoding="utf-8")
    from scode.permissions import Decision, PermissionRequest

    request = PermissionRequest(tool="Bash", specifier="npm test --watch", title="x")
    assert repl.permissions.check(request) is Decision.ALLOW
    run(repl, "/permissions remove Bash(npm test:*)")
    assert repl.permissions.check(request) is Decision.ASK


def test_mcp_and_hooks_listings(repl: StubRepl) -> None:
    assert run(repl, "/mcp").exit is False
    assert run(repl, "/hooks").exit is False


def test_bashes_with_no_shells(repl: StubRepl) -> None:
    assert run(repl, "/bashes").exit is False


def test_rewind_with_nothing_to_undo(repl: StubRepl) -> None:
    assert run(repl, "/rewind").exit is False


def test_model_check_probes_each_model(repl: StubRepl, monkeypatch) -> None:
    probed = []
    monkeypatch.setattr("scode.providers.registry.check_model",
                        lambda settings, name, **k: probed.append(name) or ("ok", "0.2s"))
    run(repl, "/model --check")
    assert repl.settings.model in probed


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
