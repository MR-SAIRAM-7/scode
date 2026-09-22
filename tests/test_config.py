from __future__ import annotations

import json
from pathlib import Path

import pytest

from scode import constants as C
from scode.config import (
    Settings,
    load_settings,
    mask_key,
    save_project_permission,
    save_user_setting,
    user_settings_path,
    validate,
    with_overrides,
)
from scode.errors import ConfigError


def test_defaults(workspace: Path) -> None:
    settings = load_settings(workspace)
    assert settings.model == C.DEFAULT_MODEL
    assert settings.provider == "nvidia"
    assert settings.permission_mode == "default"
    assert settings.workspace == workspace.resolve()
    assert settings.api_key is None


def test_env_overrides_defaults(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCODE_MODEL", "z-ai/glm-5.3")
    monkeypatch.setenv("SCODE_TEMPERATURE", "0.2")
    monkeypatch.setenv("NVIDIA_API_KEY", "  nvapi-from-env  ")
    monkeypatch.setenv("SCODE_AUTO_COMPACT", "off")

    settings = load_settings(workspace)
    assert settings.model == "z-ai/glm-5.3"
    assert settings.temperature == 0.2
    assert settings.api_key == "nvapi-from-env"
    assert settings.auto_compact is False


def test_project_settings_beat_user_settings(workspace: Path) -> None:
    save_user_setting("model", "user/model")
    project = workspace / C.PROJECT_DIR_NAME
    project.mkdir()
    (project / "settings.json").write_text(json.dumps({"model": "project/model"}), encoding="utf-8")

    assert load_settings(workspace).model == "project/model"


def test_local_settings_beat_project_settings(workspace: Path) -> None:
    project = workspace / C.PROJECT_DIR_NAME
    project.mkdir()
    (project / "settings.json").write_text(json.dumps({"temperature": 0.9}), encoding="utf-8")
    (project / "settings.local.json").write_text(json.dumps({"temperature": 0.1}), encoding="utf-8")

    assert load_settings(workspace).temperature == 0.1


def test_flags_beat_everything(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCODE_MODEL", "env/model")
    settings = load_settings(workspace, {"model": "flag/model"})
    assert settings.model == "flag/model"


def test_none_overrides_are_ignored(workspace: Path) -> None:
    settings = load_settings(workspace, {"model": None, "temperature": None})
    assert settings.model == C.DEFAULT_MODEL


def test_permissions_block_is_read(workspace: Path) -> None:
    project = workspace / C.PROJECT_DIR_NAME
    project.mkdir()
    (project / "settings.json").write_text(
        json.dumps(
            {
                "permissions": {
                    "allow": ["Bash(git status:*)"],
                    "deny": ["Bash(rm:*)"],
                    "defaultMode": "acceptEdits",
                }
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(workspace)
    assert settings.allowed_tools == ("Bash(git status:*)",)
    assert settings.denied_tools == ("Bash(rm:*)",)
    assert settings.permission_mode == "acceptEdits"


def test_invalid_json_is_reported(workspace: Path) -> None:
    project = workspace / C.PROJECT_DIR_NAME
    project.mkdir()
    (project / "settings.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="Could not read settings"):
        load_settings(workspace)


def test_bad_bool_is_reported(workspace: Path) -> None:
    project = workspace / C.PROJECT_DIR_NAME
    project.mkdir()
    (project / "settings.json").write_text(json.dumps({"stream": "yes"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="must be true or false"):
        load_settings(workspace)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"permission_mode": "nope"}, "Unknown permission mode"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"temperature": 3.0}, "temperature"),
        ({"max_steps": -1}, "max_steps"),
        ({"base_url": "ftp://x"}, "base_url"),
    ],
)
def test_validation_rejects_bad_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        validate(Settings(**kwargs))


def test_save_user_setting_roundtrip() -> None:
    path = save_user_setting("theme", "light")
    assert path == user_settings_path()
    assert json.loads(path.read_text(encoding="utf-8"))["theme"] == "light"

    save_user_setting("model", "a/b")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"theme": "light", "model": "a/b"}


def test_save_project_permission_is_idempotent(workspace: Path) -> None:
    save_project_permission(workspace, "Bash(ls:*)")
    path = save_project_permission(workspace, "Bash(ls:*)")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["permissions"]["allow"] == ["Bash(ls:*)"]


def test_mask_key() -> None:
    assert mask_key("nvapi-1234567890abcd") == "nvapi-1...abcd"
    assert mask_key("short") == "****"


def test_require_api_key_message(workspace: Path) -> None:
    settings = load_settings(workspace)
    with pytest.raises(ConfigError, match="build.nvidia.com"):
        settings.require_api_key()


def test_redacted_hides_the_key(workspace: Path) -> None:
    settings = load_settings(workspace, {"api_key": "nvapi-abcdefghijklmn"})
    assert "abcdefghijk" not in json.dumps(settings.redacted())


def test_with_overrides_validates() -> None:
    base = Settings()
    assert with_overrides(base, model="x/y").model == "x/y"
    with pytest.raises(ConfigError):
        with_overrides(base, temperature=-1.0)
