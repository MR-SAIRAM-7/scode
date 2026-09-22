"""Layered settings: defaults < user file < project file < local file < env < flags."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from . import constants as C
from .errors import ConfigError

PERMISSION_MODES = ("default", "acceptEdits", "plan", "bypassPermissions")


@dataclass(frozen=True)
class Settings:
    provider: str = C.DEFAULT_PROVIDER
    model: str = C.DEFAULT_MODEL
    small_model: str = C.DEFAULT_SMALL_MODEL
    base_url: str = C.NVIDIA_BASE_URL
    api_key: str | None = None
    max_output_tokens: int = C.DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = C.DEFAULT_TEMPERATURE
    max_steps: int = C.DEFAULT_MAX_STEPS
    permission_mode: str = "default"
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    auto_compact: bool = True
    stream: bool = True
    theme: str = "dark"
    verbose: bool = False
    native_tools: bool = True
    workspace: Path = field(default_factory=Path.cwd)

    def context_window(self) -> int:
        return C.context_window_for(self.model)

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                "No API key found.\n"
                "  Set NVIDIA_API_KEY in your environment, or run /login inside scode.\n"
                "  Free keys: https://build.nvidia.com  (key looks like nvapi-...)"
            )
        return self.api_key

    def redacted(self) -> dict[str, Any]:
        data = asdict(self)
        data["workspace"] = str(self.workspace)
        if data.get("api_key"):
            data["api_key"] = mask_key(data["api_key"])
        data["allowed_tools"] = list(self.allowed_tools)
        data["denied_tools"] = list(self.denied_tools)
        return data


def mask_key(key: str) -> str:
    if len(key) <= 12:
        return "****"
    return f"{key[:7]}...{key[-4:]}"


_FILE_KEYS: dict[str, type] = {
    "provider": str,
    "model": str,
    "small_model": str,
    "base_url": str,
    "api_key": str,
    "max_output_tokens": int,
    "temperature": float,
    "max_steps": int,
    "permission_mode": str,
    "auto_compact": bool,
    "stream": bool,
    "theme": str,
    "verbose": bool,
    "native_tools": bool,
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"Could not read settings at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Settings at {path} must be a JSON object")
    return data


def _coerce(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, caster in _FILE_KEYS.items():
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        if caster is bool:
            if not isinstance(value, bool):
                raise ConfigError(f"{key} in {path} must be true or false")
            out[key] = value
            continue
        try:
            out[key] = caster(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid value for {key} in {path}: {value!r}") from exc

    permissions = raw.get("permissions")
    if isinstance(permissions, dict):
        for src, dest in (("allow", "allowed_tools"), ("deny", "denied_tools")):
            rules = permissions.get(src)
            if rules is None:
                continue
            if not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
                raise ConfigError(f"permissions.{src} in {path} must be a list of strings")
            out[dest] = tuple(rules)
        mode = permissions.get("defaultMode")
        if isinstance(mode, str):
            out["permission_mode"] = mode
    return out


def user_settings_path() -> Path:
    return C.home_dir() / C.USER_SETTINGS_FILE


def project_settings_paths(workspace: Path) -> tuple[Path, Path]:
    root = workspace / C.PROJECT_DIR_NAME
    return root / "settings.json", root / "settings.local.json"


def _env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    mapping: dict[str, tuple[str, type]] = {
        "SCODE_PROVIDER": ("provider", str),
        "SCODE_MODEL": ("model", str),
        "SCODE_SMALL_MODEL": ("small_model", str),
        "SCODE_BASE_URL": ("base_url", str),
        "SCODE_MAX_OUTPUT_TOKENS": ("max_output_tokens", int),
        "SCODE_TEMPERATURE": ("temperature", float),
        "SCODE_MAX_STEPS": ("max_steps", int),
        "SCODE_PERMISSION_MODE": ("permission_mode", str),
        "SCODE_THEME": ("theme", str),
    }
    for env_name, (key, caster) in mapping.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        try:
            out[key] = caster(raw)
        except ValueError as exc:
            raise ConfigError(f"{env_name} must be a valid {caster.__name__}: {raw!r}") from exc

    for env_name, key in (
        ("SCODE_AUTO_COMPACT", "auto_compact"),
        ("SCODE_NATIVE_TOOLS", "native_tools"),
    ):
        raw = os.getenv(env_name)
        if raw is not None and raw != "":
            out[key] = raw.strip().lower() not in {"0", "false", "no", "off"}

    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("SCODE_API_KEY")
    if api_key:
        out["api_key"] = api_key.strip()
    return out


def load_settings(
    workspace: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Merge every configuration layer into one immutable Settings."""
    ws = (workspace or Path.cwd()).resolve()
    merged: dict[str, Any] = {}

    user_path = user_settings_path()
    merged.update(_coerce(_read_json(user_path), user_path))
    for path in project_settings_paths(ws):
        merged.update(_coerce(_read_json(path), path))
    merged.update(_env_overrides())

    for key, value in (overrides or {}).items():
        if value is not None:
            merged[key] = value

    merged["workspace"] = ws
    return validate(Settings(**merged))


def validate(settings: Settings) -> Settings:
    if settings.permission_mode not in PERMISSION_MODES:
        raise ConfigError(
            f"Unknown permission mode {settings.permission_mode!r}. "
            f"Choose from: {', '.join(PERMISSION_MODES)}"
        )
    if settings.max_output_tokens <= 0:
        raise ConfigError("max_output_tokens must be positive")
    if not 0 <= settings.temperature <= 2:
        raise ConfigError("temperature must be between 0 and 2")
    if settings.max_steps <= 0:
        raise ConfigError("max_steps must be positive")
    if not settings.base_url.startswith(("http://", "https://")):
        raise ConfigError("base_url must start with http:// or https://")
    return settings


def save_user_setting(key: str, value: Any) -> Path:
    """Write one key into the user settings file, creating it if needed."""
    path = user_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_json(path)
    data[key] = value
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def save_project_permission(workspace: Path, rule: str, *, deny: bool = False) -> Path:
    """Append an allow/deny rule to the project's local settings file."""
    _, local_path = project_settings_paths(workspace)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_json(local_path)
    permissions = data.setdefault("permissions", {})
    bucket = permissions.setdefault("deny" if deny else "allow", [])
    if rule not in bucket:
        bucket.append(rule)
    local_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return local_path


def with_overrides(settings: Settings, **kwargs: Any) -> Settings:
    clean = {k: v for k, v in kwargs.items() if v is not None}
    return validate(replace(settings, **clean))
