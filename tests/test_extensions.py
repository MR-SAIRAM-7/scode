"""Custom slash commands, custom subagents, background shells, profiles, catalog."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from scode.errors import ConfigError
from scode.extensions import load_agents, load_commands, parse_frontmatter
from scode.providers.catalog import Catalog, get_catalog, set_catalog
from scode.providers.profiles import (
    BUILTIN_PROFILES,
    canonical_name,
    known_provider_names,
    resolve_profile,
    split_model_spec,
)

# ----------------------------------------------------------------- commands


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_frontmatter_is_parsed() -> None:
    meta, body = parse_frontmatter("---\ndescription: Review it\nmodel: \"sonnet\"\n---\nDo the review.\n")
    assert meta == {"description": "Review it", "model": "sonnet"}
    assert body.strip() == "Do the review."


def test_files_without_frontmatter_are_all_body() -> None:
    assert parse_frontmatter("plain text") == ({}, "plain text")


def test_commands_load_with_namespaces(workspace: Path) -> None:
    write(workspace / ".scode/commands/review.md", "---\ndescription: Review the diff\n---\nReview $ARGUMENTS")
    write(workspace / ".scode/commands/frontend/component.md", "Build a component named $1 in $2.")
    commands = load_commands(workspace)
    assert set(commands) == {"review", "frontend:component"}
    assert commands["review"].description == "Review the diff"


def test_arguments_are_substituted() -> None:
    from scode.extensions import CustomCommand

    review = CustomCommand("review", "Review $ARGUMENTS carefully.", Path("x"))
    assert review.render("the auth module") == "Review the auth module carefully."
    component = CustomCommand("c", "Make $1 in $2.", Path("x"))
    assert component.render("Button src/ui") == "Make Button in src/ui."
    bare = CustomCommand("fix", "Fix the failing tests.", Path("x"))
    assert bare.render("in api/") == "Fix the failing tests.\n\nin api/"


def test_project_commands_beat_claude_code_ones(workspace: Path) -> None:
    write(workspace / ".claude/commands/deploy.md", "claude version")
    write(workspace / ".scode/commands/deploy.md", "scode version")
    assert load_commands(workspace)["deploy"].body == "scode version"


def test_claude_code_commands_are_picked_up(workspace: Path) -> None:
    write(workspace / ".claude/commands/triage.md", "Triage the open issues.")
    assert "triage" in load_commands(workspace)


# ------------------------------------------------------------------- agents

def test_agents_load_with_tools_and_model(workspace: Path) -> None:
    write(workspace / ".scode/agents/reviewer.md",
          "---\nname: code-reviewer\ndescription: Reviews diffs\ntools: Read, Grep, Glob\nmodel: sonnet\n---\n"
          "You review code for bugs.")
    agents = load_agents(workspace)
    reviewer = agents["code-reviewer"]
    assert reviewer.tools == ("Read", "Grep", "Glob")
    assert reviewer.model == "sonnet"
    assert reviewer.prompt == "You review code for bugs."


def test_inherit_model_means_none(workspace: Path) -> None:
    write(workspace / ".scode/agents/a.md", "---\nmodel: inherit\n---\nprompt")
    assert load_agents(workspace)["a"].model == ""


def test_empty_agent_files_are_skipped(workspace: Path) -> None:
    write(workspace / ".scode/agents/empty.md", "---\ndescription: nothing\n---\n")
    assert "empty" not in load_agents(workspace)


def test_task_tool_offers_custom_agents() -> None:
    from scode.tools.task import TaskTool

    tool = TaskTool({"code-reviewer": "Reviews diffs"})
    assert "code-reviewer" in tool.parameters["properties"]["subagent_type"]["enum"]
    assert "Reviews diffs" in tool.description


def test_registry_only_filter(workspace: Path) -> None:
    from scode.tools import build_registry

    registry = build_registry(only=["Read", "Grep"])
    assert registry.names() == ["Grep", "Read"]


# ---------------------------------------------------------- background bash

def test_background_shell_lifecycle(ctx) -> None:
    from scode.tools.background import BashOutputTool, KillShellTool, kill_all
    from scode.tools.shell import BashTool

    python = Path(sys.executable).as_posix()
    command = f'"{python}" -c "import time; print(\'ready\', flush=True); time.sleep(30)"'
    started = BashTool().run({"command": command, "run_in_background": True}, ctx)
    shell_id = started.metadata["shell_id"]
    assert shell_id == "bash_1"

    output = BashOutputTool()
    deadline = time.time() + 15
    text = ""
    while time.time() < deadline and "ready" not in text:
        text += output.run({"bash_id": shell_id}, ctx).output
        time.sleep(0.2)
    assert "ready" in text and "running" in text

    # Only new lines come back on the next read.
    assert "(no new output)" in output.run({"bash_id": shell_id}, ctx).output

    KillShellTool().run({"shell_id": shell_id}, ctx)
    assert ctx.shells[shell_id].status == "killed"
    assert kill_all(ctx) == 0


def test_bash_output_rejects_unknown_ids(ctx) -> None:
    from scode.errors import ToolError
    from scode.tools.background import BashOutputTool

    with pytest.raises(ToolError, match="No background shell"):
        BashOutputTool().run({"bash_id": "bash_9"}, ctx)


def test_background_output_filter(ctx) -> None:
    from scode.tools.background import BashOutputTool
    from scode.tools.shell import BashTool

    python = Path(sys.executable).as_posix()
    command = f'"{python}" -c "print(\'ok 1\'); print(\'ERROR 2\'); print(\'ok 3\')"'
    shell_id = BashTool().run({"command": command, "run_in_background": True}, ctx).metadata["shell_id"]
    ctx.shells[shell_id].process.wait(timeout=15)
    time.sleep(0.3)
    output = BashOutputTool().run({"bash_id": shell_id, "filter": "ERROR"}, ctx).output
    assert "ERROR 2" in output and "ok 1" not in output


# ----------------------------------------------------------------- profiles

def test_every_builtin_profile_is_complete() -> None:
    for name, profile in BUILTIN_PROFILES.items():
        assert profile.name == name
        assert profile.label
        if name != "custom":
            assert profile.base_url.startswith(("http://", "https://")), name
        if profile.key_required:
            assert profile.api_key_env, name


@pytest.mark.parametrize("alias, name", [
    ("claude", "anthropic"), ("open-router", "openrouter"), ("omni", "omniroute"),
    ("google", "gemini"), ("NIM", "nvidia"), ("grok", "xai"),
])
def test_provider_aliases(alias: str, name: str) -> None:
    assert canonical_name(alias) == name


def test_split_model_spec() -> None:
    assert split_model_spec("anthropic:claude-opus-5") == ("anthropic", "claude-opus-5")
    assert split_model_spec("openrouter:qwen/qwen3-coder:free") == ("openrouter", "qwen/qwen3-coder:free")
    assert split_model_spec("qwen/qwen3-coder:free") == (None, "qwen/qwen3-coder:free")
    assert split_model_spec("mylab:coder", {"mylab": {}}) == ("mylab", "coder")


def test_known_provider_names_include_custom() -> None:
    assert "mylab" in known_provider_names({"mylab": {"base_url": "x"}})


def test_profile_override_type_errors() -> None:
    with pytest.raises(ConfigError, match="must be an object"):
        resolve_profile("nvidia", {"nvidia": {"headers": ["not", "a", "dict"]}})


def test_unknown_providers_resolve_from_the_catalog() -> None:
    set_catalog(Catalog({"acme": {
        "name": "Acme AI", "api": "https://api.acme.ai/v1", "env": ["ACME_API_KEY"],
        "npm": "@ai-sdk/openai-compatible",
        "models": {
            "acme-old": {"tool_call": True, "release_date": "2025-01-01"},
            "acme-new": {"tool_call": True, "release_date": "2026-09-01"},
            "acme-chat": {"tool_call": False, "release_date": "2026-09-10"},
        },
    }}))
    profile = resolve_profile("acme")
    assert profile.base_url == "https://api.acme.ai/v1"
    assert profile.api_key_env == ("ACME_API_KEY",)
    assert profile.default_model == "acme-new"  # newest model that can call tools


def test_non_openai_catalog_providers_are_not_guessed() -> None:
    set_catalog(Catalog({"weird": {"api": "", "npm": "@ai-sdk/weird", "models": {}}}))
    with pytest.raises(ConfigError, match="Unknown provider"):
        resolve_profile("weird")


# ------------------------------------------------------------------ catalog

def test_catalog_parses_models() -> None:
    catalog = Catalog({"p": {"models": {"m": {
        "name": "M", "limit": {"context": 1000, "output": 100}, "tool_call": True, "reasoning": True,
        "modalities": {"input": ["text", "image"]}, "cost": {"input": 1, "output": 2, "cache_read": 0.1},
    }}}})
    info = catalog.model("p", "m")
    assert (info.context, info.output, info.tool_call, info.reasoning, info.vision) == (1000, 100, True, True, True)
    assert (info.input_cost, info.output_cost, info.cache_read_cost, info.cache_write_cost) == (1.0, 2.0, 0.1, None)
    assert info.priced


def test_catalog_strips_variant_suffixes() -> None:
    catalog = Catalog({"openrouter": {"models": {"a/b": {"limit": {"output": 5}}}}})
    assert catalog.model("openrouter", "a/b:free").output == 5


def test_empty_catalog_answers_none() -> None:
    catalog = Catalog()
    assert not catalog and catalog.model("x", "y") is None and catalog.provider("x") is None


def test_offline_mode_never_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("network used while offline")

    monkeypatch.setattr("scode.providers.catalog.requests.get", explode)
    set_catalog(None)
    assert not get_catalog()


def test_catalog_is_cached_on_disk(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    from scode.providers import catalog as module

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"p": {"models": {"m": {"limit": {"output": 7}}}}}

    calls = []
    monkeypatch.delenv("SCODE_OFFLINE")
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: calls.append(1) or Response())
    set_catalog(None)
    assert get_catalog().model("p", "m").output == 7
    assert json.loads(module.cache_path().read_text(encoding="utf-8"))["p"]
    set_catalog(None)
    get_catalog()  # fresh cache: no second fetch
    assert len(calls) == 1
