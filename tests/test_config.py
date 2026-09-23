from __future__ import annotations

import json
from pathlib import Path

import pytest

from scode import constants as C
from scode.config import (
    Settings,
    detect_key_provider,
    load_settings,
    mask_key,
    remove_project_permission,
    save_api_key,
    save_project_permission,
    save_user_setting,
    switch_provider,
    user_settings_path,
    validate,
    with_overrides,
)
from scode.errors import ConfigError


def write_project(workspace: Path, data: dict, *, local: bool = False) -> None:
    folder = workspace / C.PROJECT_DIR_NAME
    folder.mkdir(exist_ok=True)
    name = "settings.local.json" if local else "settings.json"
    (folder / name).write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------- defaults

def test_defaults_resolve_from_the_nvidia_profile(workspace: Path) -> None:
    settings = load_settings(workspace)
    assert settings.provider == "nvidia"
    assert settings.model == C.DEFAULT_MODEL
    assert settings.base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.small_model == C.DEFAULT_SMALL_MODEL
    assert settings.permission_mode == "default"
    assert settings.workspace == workspace.resolve()
    assert settings.api_key is None
    # Several providers reject any temperature, so none is sent by default.
    assert settings.temperature is None
    assert settings.profile is not None and settings.profile.name == "nvidia"


def test_every_builtin_provider_resolves(workspace: Path) -> None:
    from scode.providers.profiles import BUILTIN_PROFILES

    for name, profile in BUILTIN_PROFILES.items():
        overrides = {"provider": name}
        if not profile.base_url:
            overrides["base_url"] = "http://localhost:9999/v1"
        settings = load_settings(workspace, overrides)
        assert settings.provider == name
        assert settings.base_url.startswith("http")
        if profile.default_model:
            assert settings.model == profile.default_model


# ------------------------------------------------------------------ layers

def test_env_overrides_defaults(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCODE_MODEL", "z-ai/glm-5.3")
    monkeypatch.setenv("SCODE_TEMPERATURE", "0.2")
    monkeypatch.setenv("NVIDIA_API_KEY", "  nvapi-from-env  ")
    monkeypatch.setenv("SCODE_AUTO_COMPACT", "off")
    monkeypatch.setenv("SCODE_EFFORT", "high")

    settings = load_settings(workspace)
    assert settings.model == "z-ai/glm-5.3"
    assert settings.temperature == 0.2
    assert settings.api_key == "nvapi-from-env"
    assert settings.auto_compact is False
    assert settings.effort == "high"


def test_project_settings_beat_user_settings(workspace: Path) -> None:
    save_user_setting("model", "user/model")
    write_project(workspace, {"model": "project/model"})
    assert load_settings(workspace).model == "project/model"


def test_local_settings_beat_project_settings(workspace: Path) -> None:
    write_project(workspace, {"temperature": 0.9})
    write_project(workspace, {"temperature": 0.1}, local=True)
    assert load_settings(workspace).temperature == 0.1


def test_flags_beat_everything(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCODE_MODEL", "env/model")
    assert load_settings(workspace, {"model": "flag/model"}).model == "flag/model"


def test_none_overrides_are_ignored(workspace: Path) -> None:
    assert load_settings(workspace, {"model": None, "temperature": None}).model == C.DEFAULT_MODEL


def test_camel_case_keys_are_accepted(workspace: Path) -> None:
    write_project(workspace, {"appendSystemPrompt": "Be terse.", "maxOutputTokens": 1234})
    settings = load_settings(workspace)
    assert settings.append_system_prompt == "Be terse."
    assert settings.max_output_tokens == 1234


# ---------------------------------------------------------- provider switch

def test_a_higher_layer_provider_drops_a_lower_layer_model(workspace: Path) -> None:
    """A model saved for NVIDIA must not follow the user to OpenRouter."""
    save_user_setting("model", "nvidia/nemotron-3-ultra-550b-a55b")
    settings = load_settings(workspace, {"provider": "openrouter"})
    assert settings.provider == "openrouter"
    assert settings.model == "anthropic/claude-opus-5"


def test_model_and_provider_in_the_same_layer_both_apply(workspace: Path) -> None:
    write_project(workspace, {"provider": "openrouter", "model": "openai/gpt-6-sol"})
    settings = load_settings(workspace)
    assert (settings.provider, settings.model) == ("openrouter", "openai/gpt-6-sol")


def test_provider_colon_model_selects_both(workspace: Path) -> None:
    settings = load_settings(workspace, {"model": "anthropic:sonnet"})
    assert settings.provider == "anthropic"
    assert settings.model == "claude-sonnet-5"


def test_model_ids_with_colons_are_left_alone(workspace: Path) -> None:
    settings = load_settings(workspace, {"model": "openrouter:deepseek/deepseek-r1:free"})
    assert settings.provider == "openrouter"
    assert settings.model == "deepseek/deepseek-r1:free"
    plain = load_settings(workspace, {"model": "deepseek/deepseek-r1:free"})
    assert plain.provider == "nvidia"
    assert plain.model == "deepseek/deepseek-r1:free"


def test_aliases_resolve_per_provider(workspace: Path) -> None:
    assert load_settings(workspace, {"provider": "anthropic", "model": "opus"}).model == "claude-opus-5"
    assert load_settings(workspace, {"provider": "openrouter", "model": "haiku"}).model == "anthropic/claude-haiku-4.5"


def test_switch_provider_takes_the_new_defaults(workspace: Path) -> None:
    settings = load_settings(workspace, {"effort": "low"})
    moved = switch_provider(settings, "anthropic")
    assert moved.provider == "anthropic"
    assert moved.model == "claude-opus-5"
    assert moved.base_url == "https://api.anthropic.com"
    # The profile's default effort applies after a switch.
    assert moved.effort == "high"


def test_switch_provider_keeps_an_explicit_model(workspace: Path) -> None:
    moved = switch_provider(load_settings(workspace), "openrouter", "openai/gpt-6-sol")
    assert moved.model == "openai/gpt-6-sol"


def test_unknown_provider_is_a_clear_error(workspace: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown provider 'nope-provider'"):
        load_settings(workspace, {"provider": "nope-provider"})


def test_custom_provider_from_settings(workspace: Path) -> None:
    write_project(workspace, {"providers": {"mylab": {"base_url": "https://llm.example.com/v1",
                                                      "default_model": "lab-coder",
                                                      "api_key_env": "MYLAB_KEY"}}})
    settings = load_settings(workspace, {"provider": "mylab"})
    assert settings.base_url == "https://llm.example.com/v1"
    assert settings.model == "lab-coder"


def test_custom_provider_requires_a_base_url(workspace: Path) -> None:
    write_project(workspace, {"providers": {"broken": {"default_model": "x"}}})
    with pytest.raises(ConfigError, match="needs a base_url"):
        load_settings(workspace, {"provider": "broken"})


def test_builtin_provider_fields_can_be_overridden(workspace: Path) -> None:
    write_project(workspace, {"providers": {"omniroute": {"base_url": "http://gateway.lan:20128/v1"}}})
    settings = load_settings(workspace, {"provider": "omniroute"})
    assert settings.base_url == "http://gateway.lan:20128/v1"
    assert settings.model == "auto/coding"


def test_custom_provider_without_base_url_errors_early(workspace: Path) -> None:
    with pytest.raises(ConfigError, match="no base URL"):
        load_settings(workspace, {"provider": "custom", "model": "x"})


# -------------------------------------------------------------------- keys

def test_keys_are_found_per_provider(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    assert load_settings(workspace, {"provider": "openrouter"}).api_key == "sk-or-env"
    assert load_settings(workspace).api_key is None  # nvidia has none


def test_saved_keys_are_used_when_no_env(workspace: Path) -> None:
    save_api_key("anthropic", "sk-ant-saved")
    assert load_settings(workspace, {"provider": "anthropic"}).api_key == "sk-ant-saved"


def test_env_key_beats_saved_key(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_api_key("nvidia", "nvapi-saved")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-env")
    assert load_settings(workspace).api_key == "nvapi-env"


def test_explicit_key_beats_everything(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-env")
    assert load_settings(workspace, {"api_key": "nvapi-flag"}).api_key == "nvapi-flag"


def test_legacy_single_key_is_filed_under_its_provider(workspace: Path) -> None:
    save_user_setting("api_key", "nvapi-legacy")
    assert load_settings(workspace).api_key == "nvapi-legacy"
    assert load_settings(workspace, {"provider": "openrouter"}).api_key is None


def test_save_api_key_migrates_the_legacy_field(workspace: Path) -> None:
    save_user_setting("api_key", "nvapi-legacy")
    save_api_key("openrouter", "sk-or-new")
    data = json.loads(user_settings_path().read_text(encoding="utf-8"))
    assert "api_key" not in data
    assert data["api_keys"] == {"openrouter": "sk-or-new", "nvidia": "nvapi-legacy"}


def test_forgetting_a_key(workspace: Path) -> None:
    save_api_key("groq", "gsk_x")
    save_api_key("groq", None)
    assert "api_keys" not in json.loads(user_settings_path().read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "key, provider",
    [("nvapi-x", "nvidia"), ("sk-or-v1-x", "openrouter"), ("sk-ant-api03-x", "anthropic"),
     ("gsk_x", "groq"), ("sk-proj-x", None)],
)
def test_detect_key_provider(key: str, provider: str | None) -> None:
    assert detect_key_provider(key) == provider


def test_require_api_key_names_the_provider(workspace: Path) -> None:
    settings = load_settings(workspace, {"provider": "openrouter"})
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY") as info:
        settings.require_api_key()
    assert "openrouter.ai/keys" in str(info.value)


def test_keyless_providers_need_no_key(workspace: Path) -> None:
    assert load_settings(workspace, {"provider": "omniroute"}).require_api_key() == ""


# ------------------------------------------------------------- permissions

def test_permissions_block_is_read(workspace: Path) -> None:
    write_project(workspace, {"permissions": {"allow": ["Bash(git status:*)"], "deny": ["Bash(rm:*)"],
                                              "defaultMode": "acceptEdits"}})
    settings = load_settings(workspace)
    assert settings.allowed_tools == ("Bash(git status:*)",)
    assert settings.denied_tools == ("Bash(rm:*)",)
    assert settings.permission_mode == "acceptEdits"


def test_permission_rules_accumulate_across_layers(workspace: Path) -> None:
    write_project(workspace, {"permissions": {"allow": ["Read"]}})
    write_project(workspace, {"permissions": {"allow": ["Bash(ls:*)"]}}, local=True)
    settings = load_settings(workspace, {"allowed_tools": ("Grep",)})
    assert settings.allowed_tools == ("Read", "Bash(ls:*)", "Grep")


def test_save_project_permission_is_idempotent(workspace: Path) -> None:
    save_project_permission(workspace, "Bash(ls:*)")
    path = save_project_permission(workspace, "Bash(ls:*)")
    assert json.loads(path.read_text(encoding="utf-8"))["permissions"]["allow"] == ["Bash(ls:*)"]


def test_remove_project_permission(workspace: Path) -> None:
    save_project_permission(workspace, "Bash(ls:*)")
    assert remove_project_permission(workspace, "Bash(ls:*)") is True
    assert remove_project_permission(workspace, "Bash(ls:*)") is False


# ---------------------------------------------------------- mcp and hooks

def test_mcp_json_is_loaded(workspace: Path) -> None:
    (workspace / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"fs": {"command": "npx", "args": ["-y", "server-fs"]}}}), encoding="utf-8"
    )
    write_project(workspace, {"mcpServers": {"web": {"type": "http", "url": "https://x.dev/mcp"}}})
    settings = load_settings(workspace)
    assert set(settings.mcp_servers) == {"fs", "web"}


def test_hooks_are_loaded(workspace: Path) -> None:
    hooks = {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo"}]}]}
    write_project(workspace, {"hooks": hooks})
    assert load_settings(workspace).hooks == hooks


# -------------------------------------------------------------- validation

def test_invalid_json_is_reported(workspace: Path) -> None:
    folder = workspace / C.PROJECT_DIR_NAME
    folder.mkdir()
    (folder / "settings.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="Could not read settings"):
        load_settings(workspace)


def test_bad_bool_is_reported(workspace: Path) -> None:
    write_project(workspace, {"stream": "yes"})
    with pytest.raises(ConfigError, match="must be true or false"):
        load_settings(workspace)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"permission_mode": "nope"}, "Unknown permission mode"),
        ({"max_output_tokens": -1}, "max_output_tokens"),
        ({"temperature": 3.0}, "temperature"),
        ({"max_steps": -1}, "max_steps"),
        ({"base_url": "ftp://x"}, "base_url"),
        ({"effort": "extreme"}, "effort must be one of"),
    ],
)
def test_validation_rejects_bad_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        validate(Settings(**kwargs))


def test_with_overrides_validates_and_resolves_aliases(workspace: Path) -> None:
    base = load_settings(workspace, {"provider": "anthropic"})
    assert with_overrides(base, model="sonnet").model == "claude-sonnet-5"
    with pytest.raises(ConfigError):
        with_overrides(base, temperature=-1.0)


def test_save_user_setting_roundtrip() -> None:
    path = save_user_setting("theme", "light")
    assert path == user_settings_path()
    save_user_setting("model", "a/b")
    assert json.loads(path.read_text(encoding="utf-8")) == {"theme": "light", "model": "a/b"}
    save_user_setting("model", None)
    assert json.loads(path.read_text(encoding="utf-8")) == {"theme": "light"}


def test_mask_key() -> None:
    assert mask_key("nvapi-1234567890abcd") == "nvapi-1...abcd"
    assert mask_key("short") == "****"


def test_redacted_hides_every_key(workspace: Path) -> None:
    save_api_key("openrouter", "sk-or-abcdefghijklmnop")
    settings = load_settings(workspace, {"api_key": "nvapi-abcdefghijklmn"})
    text = json.dumps(settings.redacted())
    assert "abcdefghijk" not in text


def test_output_limit_is_capped_by_the_catalog(workspace: Path) -> None:
    from scode.providers.catalog import Catalog, set_catalog

    set_catalog(Catalog({"groq": {"models": {"qwen/qwen3.8-27b": {"limit": {"output": 16384, "context": 131072}}}}}))
    settings = load_settings(workspace, {"provider": "groq", "max_output_tokens": 50_000})
    assert settings.output_limit() == 16384
    assert settings.context_window() == 131072
