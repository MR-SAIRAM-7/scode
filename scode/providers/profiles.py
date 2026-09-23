"""Provider profiles: everything scode needs to talk to one model provider.

Adding a provider is data, not code. A built-in profile covers the common
providers; `providers` in settings.json overrides any field or defines a new
provider; and any OpenAI-compatible provider listed on models.dev resolves by
name with no configuration at all.

Model IDs below were checked against each provider's live catalog (or
models.dev) when this file was written. Catalogs move, so `/model --check`
probes what a key can actually reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any

from ..errors import ConfigError

WIRE_OPENAI = "openai"
WIRE_ANTHROPIC = "anthropic"

# How a provider expects a reasoning-effort setting to be sent.
EFFORT_NONE = "none"
EFFORT_REASONING_EFFORT = "reasoning_effort"  # top-level "reasoning_effort": "high"
EFFORT_OPENROUTER = "openrouter"              # "reasoning": {"effort": "high"}
EFFORT_ANTHROPIC = "anthropic"                # "output_config": {"effort": "high"}

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    label: str
    base_url: str
    wire: str = WIRE_OPENAI
    api_key_env: tuple[str, ...] = ()
    key_required: bool = True
    default_model: str = ""
    small_model: str = ""
    models: tuple[str, ...] = ()
    model_aliases: dict[str, str] = field(default_factory=dict)
    # models.dev provider id, for limits and pricing.
    catalog_id: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    max_tokens_param: str = "max_tokens"
    default_max_tokens: int = 32_000
    effort_style: str = EFFORT_NONE
    default_effort: str = ""
    supports_temperature: bool = True
    # OpenRouter wants reasoning_details echoed on assistant turns.
    echo_reasoning_details: bool = False
    # Local servers: use whichever model the server has loaded.
    auto_model: bool = False
    # Server-side refusal fallbacks (Anthropic API only).
    refusal_fallbacks: bool = False
    signup_url: str = ""
    notes: str = ""

    def resolve_alias(self, model: str) -> str:
        return self.model_aliases.get(model.strip().lower(), model.strip())

    def key_hint(self) -> str:
        if self.api_key_env:
            return f"Set {self.api_key_env[0]}, or run /login {self.name}"
        return f"Run /login {self.name}"


_CLAUDE_ALIASES = {
    "opus": "claude-opus-5",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
    "fable": "claude-fable-5-1",
}

BUILTIN_PROFILES: dict[str, ProviderProfile] = {
    p.name: p
    for p in (
        ProviderProfile(
            name="nvidia",
            label="NVIDIA NIM",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key_env=("NVIDIA_API_KEY",),
            default_model="nvidia/nemotron-3-super-120b-a12b",
            small_model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            models=(
                "nvidia/nemotron-3-super-120b-a12b",
                "nvidia/nemotron-3-ultra-550b-a55b",
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                "openai/gpt-oss-20b",
                "poolside/laguna-xs-2.1",
                "meta/llama-3.2-11b-vision-instruct",
            ),
            catalog_id="nvidia",
            effort_style=EFFORT_REASONING_EFFORT,
            signup_url="https://build.nvidia.com",
            notes="Free credits. The catalog lists far more models than a key can reach.",
        ),
        ProviderProfile(
            name="openrouter",
            label="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env=("OPENROUTER_API_KEY",),
            default_model="anthropic/claude-opus-5",
            small_model="anthropic/claude-sonnet-5",
            models=(
                "anthropic/claude-opus-5",
                "anthropic/claude-sonnet-5",
                "anthropic/claude-opus-5.5",
                "anthropic/claude-fable-5.1",
                "anthropic/claude-haiku-4.5",
                "openai/gpt-6-sol",
                "openai/gpt-6-astra",
                "google/gemini-3.8-flash",
                "deepseek/deepseek-v4-pro-0813",
                "moonshotai/kimi-k3",
                "z-ai/glm-5.3",
                "x-ai/grok-4.7",
                "qwen/qwen3-coder",
                "openrouter/auto",
            ),
            model_aliases={k: f"anthropic/{v}" for k, v in {
                "opus": "claude-opus-5",
                "sonnet": "claude-sonnet-5",
                "haiku": "claude-haiku-4.5",
                "fable": "claude-fable-5.1",
            }.items()},
            catalog_id="openrouter",
            headers={
                "HTTP-Referer": "https://github.com/MR-SAIRAM-7/scode",
                "X-Title": "scode",
            },
            effort_style=EFFORT_OPENROUTER,
            echo_reasoning_details=True,
            signup_url="https://openrouter.ai/keys",
            notes="One key for hundreds of models across providers.",
        ),
        ProviderProfile(
            name="omniroute",
            label="OmniRoute (local gateway)",
            base_url="http://localhost:20128/v1",
            api_key_env=("OMNIROUTE_API_KEY",),
            key_required=False,
            default_model="auto/coding",
            small_model="auto/fast",
            models=("auto/coding", "auto", "auto/fast", "auto/cheap", "auto/offline"),
            signup_url="https://github.com/diegosouzapw/OmniRoute",
            notes="Self-hosted router. Start it with `npx omniroute` (port 20128).",
        ),
        ProviderProfile(
            name="anthropic",
            label="Anthropic (native API)",
            base_url="https://api.anthropic.com",
            wire=WIRE_ANTHROPIC,
            api_key_env=("ANTHROPIC_API_KEY",),
            default_model="claude-opus-5",
            small_model="claude-sonnet-5",
            models=(
                "claude-opus-5",
                "claude-sonnet-5",
                "claude-haiku-4-5",
                "claude-fable-5-1",
                "claude-opus-5-5",
                "claude-opus-4-8",
            ),
            model_aliases=dict(_CLAUDE_ALIASES),
            catalog_id="anthropic",
            default_max_tokens=64_000,
            effort_style=EFFORT_ANTHROPIC,
            default_effort="high",
            supports_temperature=False,
            refusal_fallbacks=True,
            signup_url="https://console.anthropic.com/settings/keys",
            notes="Native Messages API with prompt caching and adaptive thinking.",
        ),
        ProviderProfile(
            name="openai",
            label="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key_env=("OPENAI_API_KEY",),
            default_model="gpt-6-sol",
            small_model="gpt-6-luna",
            models=("gpt-6-sol", "gpt-6-astra", "gpt-6-luna", "gpt-5.6-sol"),
            catalog_id="openai",
            # Reasoning models reject `max_tokens` and any non-default temperature.
            max_tokens_param="max_completion_tokens",
            supports_temperature=False,
            effort_style=EFFORT_REASONING_EFFORT,
            signup_url="https://platform.openai.com/api-keys",
        ),
        ProviderProfile(
            name="gemini",
            label="Google Gemini (OpenAI-compatible)",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            default_model="gemini-3.1-pro-preview",
            small_model="gemini-3.8-flash",
            models=("gemini-3.1-pro-preview", "gemini-3.8-flash", "gemini-3.5-flash-lite"),
            catalog_id="google",
            effort_style=EFFORT_REASONING_EFFORT,
            signup_url="https://aistudio.google.com/apikey",
        ),
        ProviderProfile(
            name="deepseek",
            label="DeepSeek",
            base_url="https://api.deepseek.com",
            api_key_env=("DEEPSEEK_API_KEY",),
            default_model="deepseek-v4-pro",
            small_model="deepseek-v4-flash",
            models=("deepseek-v4-pro", "deepseek-v4-flash"),
            catalog_id="deepseek",
            signup_url="https://platform.deepseek.com/api_keys",
        ),
        ProviderProfile(
            name="groq",
            label="Groq",
            base_url="https://api.groq.com/openai/v1",
            api_key_env=("GROQ_API_KEY",),
            default_model="qwen/qwen3.8-27b",
            small_model="openai/gpt-oss-20b",
            models=("qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"),
            catalog_id="groq",
            default_max_tokens=16_384,
            effort_style=EFFORT_REASONING_EFFORT,
            signup_url="https://console.groq.com/keys",
        ),
        ProviderProfile(
            name="together",
            label="Together AI",
            base_url="https://api.together.xyz/v1",
            api_key_env=("TOGETHER_API_KEY",),
            default_model="zai-org/GLM-5.3",
            small_model="deepseek-ai/DeepSeek-V4.1-Flash",
            models=("zai-org/GLM-5.3", "deepseek-ai/DeepSeek-V4.1-Flash", "moonshotai/Kimi-K3"),
            catalog_id="togetherai",
            signup_url="https://api.together.ai/settings/api-keys",
        ),
        ProviderProfile(
            name="mistral",
            label="Mistral",
            base_url="https://api.mistral.ai/v1",
            api_key_env=("MISTRAL_API_KEY",),
            default_model="mistral-medium-latest",
            small_model="mistral-small-latest",
            models=("mistral-medium-latest", "mistral-small-latest"),
            catalog_id="mistral",
            signup_url="https://console.mistral.ai/api-keys",
        ),
        ProviderProfile(
            name="xai",
            label="xAI",
            base_url="https://api.x.ai/v1",
            api_key_env=("XAI_API_KEY",),
            default_model="grok-4.7",
            small_model="grok-build-0.1",
            models=("grok-4.7", "grok-4.6", "grok-build-0.1"),
            catalog_id="xai",
            signup_url="https://console.x.ai",
        ),
        ProviderProfile(
            name="fireworks",
            label="Fireworks AI",
            base_url="https://api.fireworks.ai/inference/v1",
            api_key_env=("FIREWORKS_API_KEY",),
            default_model="accounts/fireworks/models/glm-5p3",
            small_model="accounts/fireworks/models/glm-5p3-flash",
            models=(
                "accounts/fireworks/models/glm-5p3",
                "accounts/fireworks/models/glm-5p3-flash",
                "accounts/fireworks/models/deepseek-v4p1-flash",
            ),
            catalog_id="fireworks-ai",
            signup_url="https://fireworks.ai/account/api-keys",
        ),
        ProviderProfile(
            name="cerebras",
            label="Cerebras",
            base_url="https://api.cerebras.ai/v1",
            api_key_env=("CEREBRAS_API_KEY",),
            default_model="qwen-3.8-27b",
            small_model="gpt-oss-120b",
            models=("qwen-3.8-27b", "gpt-oss-120b"),
            catalog_id="cerebras",
            default_max_tokens=32_768,
            signup_url="https://cloud.cerebras.ai",
        ),
        ProviderProfile(
            name="moonshot",
            label="Moonshot AI (Kimi)",
            base_url="https://api.moonshot.ai/v1",
            api_key_env=("MOONSHOT_API_KEY",),
            default_model="kimi-k3",
            small_model="kimi-k2.6",
            models=("kimi-k3", "kimi-k2.7-code", "kimi-k2.6"),
            catalog_id="moonshotai",
            signup_url="https://platform.moonshot.ai/console/api-keys",
        ),
        ProviderProfile(
            name="zai",
            label="Z.ai (GLM)",
            base_url="https://api.z.ai/api/paas/v4",
            api_key_env=("ZAI_API_KEY", "ZHIPU_API_KEY"),
            default_model="glm-5.3",
            small_model="glm-5.3-flash",
            models=("glm-5.3", "glm-5.3-flash"),
            catalog_id="zai",
            signup_url="https://z.ai/manage-apikey/apikey-list",
        ),
        ProviderProfile(
            name="ollama",
            label="Ollama (local)",
            base_url="http://localhost:11434/v1",
            api_key_env=("OLLAMA_API_KEY",),
            key_required=False,
            auto_model=True,
            notes="Uses whichever model you have pulled; pick one with /model.",
        ),
        ProviderProfile(
            name="lmstudio",
            label="LM Studio (local)",
            base_url="http://127.0.0.1:1234/v1",
            api_key_env=("LMSTUDIO_API_KEY",),
            key_required=False,
            auto_model=True,
            catalog_id="lmstudio",
            notes="Uses whichever model LM Studio has loaded.",
        ),
        ProviderProfile(
            name="custom",
            label="Custom OpenAI-compatible endpoint",
            base_url="",
            api_key_env=("SCODE_API_KEY",),
            key_required=False,
            notes="Set base_url and model in settings, or pass --base-url and --model.",
        ),
    )
}

# Spellings people reach for, mapped to the canonical profile name.
PROVIDER_ALIASES = {
    "nim": "nvidia",
    "open-router": "openrouter",
    "omni": "omniroute",
    "omni-route": "omniroute",
    "claude": "anthropic",
    "google": "gemini",
    "togetherai": "together",
    "together-ai": "together",
    "fireworks-ai": "fireworks",
    "moonshotai": "moonshot",
    "kimi": "moonshot",
    "z-ai": "zai",
    "zhipu": "zai",
    "lm-studio": "lmstudio",
    "openai-compatible": "custom",
    "x-ai": "xai",
    "grok": "xai",
}

_PROFILE_FIELDS = {f.name for f in fields(ProviderProfile)}
_TUPLE_FIELDS = {"api_key_env", "models"}
_DICT_FIELDS = {"headers", "model_aliases"}


def canonical_name(name: str) -> str:
    key = (name or "").strip().lower()
    return PROVIDER_ALIASES.get(key, key)


def _apply_overrides(profile: ProviderProfile, overrides: dict[str, Any], source: str) -> ProviderProfile:
    changes: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in _PROFILE_FIELDS or key == "name":
            continue
        if key in _TUPLE_FIELDS:
            if isinstance(value, str):
                value = (value,)
            if not isinstance(value, (list, tuple)):
                raise ConfigError(f"{source}.{key} must be a list of strings")
            value = tuple(str(v) for v in value)
        elif key in _DICT_FIELDS:
            if not isinstance(value, dict):
                raise ConfigError(f"{source}.{key} must be an object")
            value = {str(k): str(v) for k, v in value.items()}
        changes[key] = value
    try:
        return replace(profile, **changes)
    except TypeError as exc:
        raise ConfigError(f"Invalid provider settings in {source}: {exc}") from exc


def _from_catalog(name: str) -> ProviderProfile | None:
    from .catalog import get_catalog

    info = get_catalog().provider(name)
    if info is None or not info.openai_compatible:
        return None
    newest = info.newest_tool_models()
    return ProviderProfile(
        name=name,
        label=info.name or name,
        base_url=info.api,
        api_key_env=info.env,
        default_model=newest[0].id if newest else "",
        small_model=newest[1].id if len(newest) > 1 else "",
        models=tuple(m.id for m in newest[:8]),
        catalog_id=name,
        signup_url=info.doc,
        notes="Resolved from the models.dev catalog.",
    )


def known_provider_names(custom: dict[str, Any] | None = None) -> list[str]:
    names = set(BUILTIN_PROFILES)
    names.update(canonical_name(n) for n in (custom or {}))
    return sorted(names)


def resolve_profile(name: str, custom: dict[str, Any] | None = None) -> ProviderProfile:
    """Find the profile for a provider name.

    Order: built-in (with settings overrides applied), then a provider defined
    entirely in settings, then models.dev.
    """
    canonical = canonical_name(name)
    if not canonical:
        raise ConfigError("No provider configured.")

    overrides: dict[str, Any] | None = None
    for key, value in (custom or {}).items():
        if canonical_name(key) == canonical:
            if not isinstance(value, dict):
                raise ConfigError(f"providers.{key} in settings must be an object")
            overrides = value
            break

    if canonical in BUILTIN_PROFILES:
        profile = BUILTIN_PROFILES[canonical]
        if overrides:
            profile = _apply_overrides(profile, overrides, f"providers.{canonical}")
        return profile

    if overrides is not None:
        if not overrides.get("base_url"):
            raise ConfigError(
                f"providers.{canonical} needs a base_url (an OpenAI-compatible /v1 endpoint)."
            )
        base = ProviderProfile(name=canonical, label=str(overrides.get("label", canonical)), base_url="")
        return _apply_overrides(base, overrides, f"providers.{canonical}")

    from_catalog = _from_catalog(canonical)
    if from_catalog is not None:
        return from_catalog

    raise ConfigError(
        f"Unknown provider {name!r}. Built-in providers: "
        f"{', '.join(sorted(BUILTIN_PROFILES))}.\n"
        "Any other OpenAI-compatible endpoint works too: add it under \"providers\" "
        "in settings.json, or pass --provider custom --base-url URL --model ID."
    )


def split_model_spec(spec: str, custom: dict[str, Any] | None = None) -> tuple[str | None, str]:
    """Split "provider:model" into its parts.

    Only splits when the prefix names a known provider, because model ids
    themselves may contain colons (OpenRouter's ":free" variants).
    """
    text = (spec or "").strip()
    if ":" in text:
        prefix, rest = text.split(":", 1)
        canonical = canonical_name(prefix)
        if rest and (canonical in BUILTIN_PROFILES or canonical in {
            canonical_name(k) for k in (custom or {})
        }):
            return canonical, rest.strip()
    return None, text
