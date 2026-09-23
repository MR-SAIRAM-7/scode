"""Layered settings: defaults < user file < project file < local file < env < flags.

Provider-dependent values (model, base URL, API key, output limit) are left
empty by the user and filled from the provider's profile at resolve time, so
switching providers never carries over a model that belongs to another one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import constants as C
from .errors import ConfigError
from .providers.profiles import (
    EFFORT_LEVELS,
    ProviderProfile,
    canonical_name,
    resolve_profile,
    split_model_spec,
)

PERMISSION_MODES = ("default", "acceptEdits", "plan", "bypassPermissions")

# Keys whose meaning depends on which provider is active. When a higher layer
# picks a different provider, a lower layer's value for these is discarded.
PROVIDER_BOUND = ("model", "small_model", "base_url")


@dataclass(frozen=True)
class Settings:
    provider: str = C.DEFAULT_PROVIDER
    model: str = ""
    small_model: str = ""
    base_url: str = ""
    api_key: str | None = None
    # Keys saved by /login, per provider.
    api_keys: dict[str, str] = field(default_factory=dict)
    # 0 means "the provider's default, capped at the model's output limit".
    max_output_tokens: int = 0
    # None means "don't send it": several providers reject any temperature.
    temperature: float | None = None
    effort: str = ""
    max_steps: int = C.DEFAULT_MAX_STEPS
    permission_mode: str = "default"
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    auto_compact: bool = True
    stream: bool = True
    theme: str = "dark"
    verbose: bool = False
    debug: bool = False
    native_tools: bool = True
    append_system_prompt: str = ""
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    hooks: dict[str, Any] = field(default_factory=dict)
    workspace: Path = field(default_factory=Path.cwd)
    profile: ProviderProfile | None = field(default=None, compare=False, repr=False)

    # --------------------------------------------------------------- lookups
    def get_profile(self) -> ProviderProfile:
        return self.profile or resolve_profile(self.provider, self.providers)

    def model_info(self, model: str | None = None):
        """Catalog metadata for a model, or None when unknown."""
        from .providers.catalog import get_catalog

        profile = self.get_profile()
        if not profile.catalog_id:
            return None
        return get_catalog().model(profile.catalog_id, model or self.model)

    def context_window(self) -> int:
        info = self.model_info()
        if info and info.context:
            return info.context
        return C.context_window_for(self.model)

    def output_limit(self) -> int:
        """max_tokens to send: the configured value, capped by the model."""
        wanted = self.max_output_tokens or self.get_profile().default_max_tokens
        info = self.model_info()
        if info and info.output:
            return min(wanted, info.output)
        return wanted

    def require_api_key(self) -> str:
        profile = self.get_profile()
        if self.api_key:
            return self.api_key
        if not profile.key_required:
            return ""
        lines = [f"No API key for {profile.label}."]
        if profile.api_key_env:
            lines.append(f"  Set {profile.api_key_env[0]} in your environment, or run /login {profile.name}.")
        else:
            lines.append(f"  Run /login {profile.name}, or pass --api-key.")
        if profile.signup_url:
            lines.append(f"  Get a key: {profile.signup_url}")
        raise ConfigError("\n".join(lines))

    def redacted(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for name in (
            "provider", "model", "small_model", "base_url", "max_output_tokens",
            "temperature", "effort", "max_steps", "permission_mode", "auto_compact",
            "stream", "theme", "verbose", "debug", "native_tools",
        ):
            data[name] = getattr(self, name)
        data["api_key"] = mask_key(self.api_key) if self.api_key else None
        data["api_keys"] = {k: mask_key(v) for k, v in self.api_keys.items()}
        data["allowed_tools"] = list(self.allowed_tools)
        data["denied_tools"] = list(self.denied_tools)
        data["workspace"] = str(self.workspace)
        data["mcp_servers"] = sorted(self.mcp_servers)
        data["hooks"] = sorted(self.hooks)
        data["custom_providers"] = sorted(self.providers)
        return data


def mask_key(key: str) -> str:
    if len(key) <= 12:
        return "****"
    return f"{key[:7]}...{key[-4:]}"


def detect_key_provider(key: str) -> str | None:
    """Guess which provider issued a key from its prefix."""
    prefixes = (
        ("nvapi-", "nvidia"),
        ("sk-or-", "openrouter"),
        ("sk-ant-", "anthropic"),
        ("gsk_", "groq"),
        ("xai-", "xai"),
        ("fw_", "fireworks"),
        ("csk-", "cerebras"),
    )
    for prefix, provider in prefixes:
        if key.startswith(prefix):
            return provider
    return None


# --------------------------------------------------------------- file layer

_SCALAR_KEYS: dict[str, type] = {
    "provider": str,
    "model": str,
    "small_model": str,
    "base_url": str,
    "max_output_tokens": int,
    "temperature": float,
    "effort": str,
    "max_steps": int,
    "permission_mode": str,
    "theme": str,
    "append_system_prompt": str,
}
_BOOL_KEYS = ("auto_compact", "stream", "verbose", "native_tools", "debug")
# Claude Code spells some keys in camelCase; accept both.
_ALIASES = {
    "mcpServers": "mcp_servers",
    "appendSystemPrompt": "append_system_prompt",
    "maxOutputTokens": "max_output_tokens",
    "smallModel": "small_model",
    "baseUrl": "base_url",
    "apiKeys": "api_keys",
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


def _dict_of_dicts(value: Any, key: str, path: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not all(isinstance(v, dict) for v in value.values()):
        raise ConfigError(f"{key} in {path} must be an object of objects")
    return {str(k): dict(v) for k, v in value.items()}


def _coerce(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    raw = {_ALIASES.get(k, k): v for k, v in raw.items()}
    out: dict[str, Any] = {}

    for key, caster in _SCALAR_KEYS.items():
        if raw.get(key) is None:
            continue
        try:
            out[key] = caster(raw[key])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid value for {key} in {path}: {raw[key]!r}") from exc

    for key in _BOOL_KEYS:
        if key in raw and raw[key] is not None:
            if not isinstance(raw[key], bool):
                raise ConfigError(f"{key} in {path} must be true or false")
            out[key] = raw[key]

    if raw.get("api_keys") is not None:
        keys = raw["api_keys"]
        if not isinstance(keys, dict) or not all(isinstance(v, str) for v in keys.values()):
            raise ConfigError(f"api_keys in {path} must map provider names to strings")
        out["api_keys"] = {canonical_name(k): v for k, v in keys.items()}

    # Older versions stored one bare key; file it under the provider it belongs to.
    if isinstance(raw.get("api_key"), str) and raw["api_key"].strip():
        key = raw["api_key"].strip()
        owner = detect_key_provider(key) or canonical_name(
            str(raw.get("provider") or C.DEFAULT_PROVIDER)
        )
        out.setdefault("api_keys", {}).setdefault(owner, key)

    for key in ("providers", "mcp_servers"):
        if raw.get(key) is not None:
            out[key] = _dict_of_dicts(raw[key], key, path)

    if raw.get("hooks") is not None:
        if not isinstance(raw["hooks"], dict):
            raise ConfigError(f"hooks in {path} must be an object")
        out["hooks"] = dict(raw["hooks"])

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


def _mcp_json(workspace: Path) -> dict[str, Any]:
    """Project `.mcp.json`, the format Claude Code uses for shared MCP servers."""
    path = workspace / ".mcp.json"
    raw = _read_json(path)
    servers = raw.get("mcpServers")
    if servers is None:
        return {}
    return {"mcp_servers": _dict_of_dicts(servers, "mcpServers", path)}


# ---------------------------------------------------------------- env layer

def _env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    mapping: dict[str, tuple[str, type]] = {
        "SCODE_PROVIDER": ("provider", str),
        "SCODE_MODEL": ("model", str),
        "SCODE_SMALL_MODEL": ("small_model", str),
        "SCODE_BASE_URL": ("base_url", str),
        "SCODE_MAX_OUTPUT_TOKENS": ("max_output_tokens", int),
        "SCODE_TEMPERATURE": ("temperature", float),
        "SCODE_EFFORT": ("effort", str),
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
        ("SCODE_DEBUG", "debug"),
    ):
        raw = os.getenv(env_name)
        if raw is not None and raw != "":
            out[key] = raw.strip().lower() not in {"0", "false", "no", "off"}

    generic_key = os.getenv("SCODE_API_KEY")
    if generic_key:
        out["api_key"] = generic_key.strip()
    return out


# ------------------------------------------------------------------ resolve

def lookup_api_key(
    profile: ProviderProfile,
    api_keys: dict[str, str],
    explicit: str | None = None,
) -> str | None:
    """Key precedence: explicit flag > provider env var > key saved by /login."""
    if explicit:
        return explicit
    for env_name in profile.api_key_env:
        value = os.getenv(env_name)
        if value and value.strip():
            return value.strip()
    saved = api_keys.get(profile.name)
    return saved.strip() if saved else None


def resolve(settings: Settings, *, explicit_key: str | None = None) -> Settings:
    """Fill provider-dependent fields from the profile and validate."""
    profile = resolve_profile(settings.provider, settings.providers)
    model = profile.resolve_alias(settings.model) if settings.model else profile.default_model
    small = profile.resolve_alias(settings.small_model) if settings.small_model else ""
    base_url = settings.base_url or profile.base_url
    if not base_url:
        raise ConfigError(
            f"Provider {profile.name!r} has no base URL. Pass --base-url or set "
            f"providers.{profile.name}.base_url in settings."
        )
    resolved = replace(
        settings,
        provider=profile.name,
        model=model,
        small_model=small or profile.small_model or model,
        base_url=base_url,
        api_key=lookup_api_key(profile, settings.api_keys, explicit_key or settings.api_key),
        effort=settings.effort or profile.default_effort,
        profile=profile,
    )
    return validate(resolved)


def load_settings(
    workspace: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Merge every configuration layer into one resolved, immutable Settings."""
    ws = (workspace or Path.cwd()).resolve()

    layers: list[dict[str, Any]] = []
    user_path = user_settings_path()
    layers.append(_coerce(_read_json(user_path), user_path))
    layers.append(_mcp_json(ws))
    for path in project_settings_paths(ws):
        layers.append(_coerce(_read_json(path), path))
    layers.append(_env_overrides())
    layers.append({k: v for k, v in (overrides or {}).items() if v is not None})

    merged: dict[str, Any] = {}
    origin: dict[str, int] = {}
    for index, layer in enumerate(layers):
        for key, value in layer.items():
            if key in {"providers", "mcp_servers", "hooks", "api_keys"}:
                merged[key] = {**merged.get(key, {}), **value}
            elif key in {"allowed_tools", "denied_tools"}:
                # Rules accumulate across layers, as in Claude Code.
                existing = list(merged.get(key, ()))
                merged[key] = tuple(existing + [r for r in value if r not in existing])
            else:
                merged[key] = value
            origin[key] = index

    # "provider:model" in the model field selects both at that layer.
    if merged.get("model"):
        spec_provider, spec_model = split_model_spec(merged["model"], merged.get("providers"))
        if spec_provider:
            merged["provider"] = spec_provider
            merged["model"] = spec_model
            origin["provider"] = max(origin.get("provider", -1), origin["model"])

    provider_layer = origin.get("provider", -1)
    for key in PROVIDER_BOUND:
        if key in merged and origin.get(key, -1) < provider_layer:
            del merged[key]

    explicit_key = merged.pop("api_key", None)
    merged["workspace"] = ws
    known = set(Settings.__dataclass_fields__) - {"profile"}
    unknown = set(merged) - known
    for key in unknown:
        merged.pop(key)
    return resolve(Settings(**merged), explicit_key=explicit_key)


def validate(settings: Settings) -> Settings:
    if settings.permission_mode not in PERMISSION_MODES:
        raise ConfigError(
            f"Unknown permission mode {settings.permission_mode!r}. "
            f"Choose from: {', '.join(PERMISSION_MODES)}"
        )
    if settings.max_output_tokens < 0:
        raise ConfigError("max_output_tokens must be positive")
    if settings.temperature is not None and not 0 <= settings.temperature <= 2:
        raise ConfigError("temperature must be between 0 and 2")
    if settings.max_steps <= 0:
        raise ConfigError("max_steps must be positive")
    if settings.effort and settings.effort not in EFFORT_LEVELS:
        raise ConfigError(f"effort must be one of: {', '.join(EFFORT_LEVELS)}")
    if settings.base_url and not settings.base_url.startswith(("http://", "https://")):
        raise ConfigError("base_url must start with http:// or https://")
    return settings


def with_overrides(settings: Settings, **kwargs: Any) -> Settings:
    """Change simple fields. Use switch_provider() to change provider."""
    clean = {k: v for k, v in kwargs.items() if v is not None}
    if "model" in clean:
        clean["model"] = settings.get_profile().resolve_alias(str(clean["model"]))
    return validate(replace(settings, **clean))


def switch_provider(settings: Settings, provider: str, model: str | None = None) -> Settings:
    """Move to another provider, taking its default model unless one is given."""
    fresh = replace(
        settings,
        provider=canonical_name(provider),
        model=model or "",
        small_model="",
        base_url="",
        api_key=None,
        effort="",
        profile=None,
    )
    return resolve(fresh)


# ------------------------------------------------------------- persistence

def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def save_user_setting(key: str, value: Any) -> Path:
    """Write one key into the user settings file, creating it if needed."""
    path = user_settings_path()
    data = _read_json(path)
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    _write_json(path, data)
    return path


def save_model_choice(provider: str, model: str) -> Path:
    """Remember a /provider or /model choice as the user's default."""
    path = user_settings_path()
    data = _read_json(path)
    name = canonical_name(provider)
    if canonical_name(str(data.get("provider") or C.DEFAULT_PROVIDER)) != name:
        # An endpoint or small model saved for the old provider would break the new one.
        for key in ("base_url", "baseUrl", "small_model", "smallModel"):
            data.pop(key, None)
    data["provider"] = name
    if model:
        data["model"] = model
    else:
        data.pop("model", None)
    _write_json(path, data)
    return path


def save_api_key(provider: str, key: str | None) -> Path:
    """Store (or with None, forget) the key /login saves for one provider."""
    path = user_settings_path()
    data = _read_json(path)
    keys = data.get("api_keys")
    if not isinstance(keys, dict):
        keys = {}
    name = canonical_name(provider)
    if key:
        keys[name] = key
    else:
        keys.pop(name, None)
    # Retire the old single-key field once keys are stored per provider.
    legacy = data.pop("api_key", None)
    if isinstance(legacy, str) and legacy and key is not None:
        owner = detect_key_provider(legacy) or C.DEFAULT_PROVIDER
        keys.setdefault(owner, legacy)
    if keys:
        data["api_keys"] = keys
    else:
        data.pop("api_keys", None)
    _write_json(path, data)
    return path


def save_project_permission(workspace: Path, rule: str, *, deny: bool = False) -> Path:
    """Append an allow/deny rule to the project's local settings file."""
    _, local_path = project_settings_paths(workspace)
    data = _read_json(local_path)
    permissions = data.setdefault("permissions", {})
    bucket = permissions.setdefault("deny" if deny else "allow", [])
    if rule not in bucket:
        bucket.append(rule)
    _write_json(local_path, data)
    return local_path


def remove_project_permission(workspace: Path, rule: str) -> bool:
    """Drop a rule from the local settings file. Returns True if it was there."""
    _, local_path = project_settings_paths(workspace)
    data = _read_json(local_path)
    permissions = data.get("permissions") or {}
    removed = False
    for bucket in ("allow", "deny"):
        rules = permissions.get(bucket) or []
        if rule in rules:
            rules.remove(rule)
            removed = True
    if removed:
        _write_json(local_path, data)
    return removed
