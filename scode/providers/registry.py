"""Builds the right client for the active provider."""

from __future__ import annotations

from typing import Any

from ..config import Settings
from .catalog import get_catalog
from .openai_compatible import OpenAICompatibleProvider
from .profiles import WIRE_ANTHROPIC, ProviderProfile


def _output_limits(profile: ProviderProfile):
    """A lookup of each model's output cap from the catalog, if known."""
    if not profile.catalog_id:
        return None

    def lookup(model: str) -> int | None:
        info = get_catalog().model(profile.catalog_id, model)
        return info.output if info and info.output else None

    return lookup


def build_provider(settings: Settings, *, model: str | None = None, require_key: bool = True) -> Any:
    """Construct the client for `settings.provider`."""
    profile = settings.get_profile()
    api_key = settings.require_api_key() if require_key else (settings.api_key or "")
    model_id = model or settings.model
    max_tokens = settings.max_output_tokens or profile.default_max_tokens

    if profile.wire == WIRE_ANTHROPIC:
        from .anthropic_native import AnthropicProvider

        return AnthropicProvider(
            api_key=api_key,
            model=model_id,
            base_url=settings.base_url,
            max_output_tokens=max_tokens,
            effort=settings.effort,
            refusal_fallbacks=profile.refusal_fallbacks,
            key_hint=profile.key_hint(),
            output_limits=_output_limits(profile),
        )

    return OpenAICompatibleProvider(
        name=profile.name,
        api_key=api_key,
        base_url=settings.base_url,
        model=model_id,
        max_output_tokens=max_tokens,
        temperature=settings.temperature,
        headers=dict(profile.headers),
        max_tokens_param=profile.max_tokens_param,
        effort_style=profile.effort_style,
        effort=settings.effort,
        supports_temperature=profile.supports_temperature,
        echo_reasoning_details=profile.echo_reasoning_details,
        key_hint=profile.key_hint(),
        output_limits=_output_limits(profile),
    )


# Kept for callers written against the single-provider API.
def get_provider(settings: Settings, *, model: str | None = None) -> Any:
    return build_provider(settings, model=model)


def list_catalog(settings: Settings) -> list[str]:
    """Live model list from the provider. Works without a key where the endpoint allows."""
    return build_provider(settings, require_key=False).list_models()


def check_model(settings: Settings, model: str, *, timeout: float = 25.0) -> tuple[str, str]:
    return build_provider(settings, require_key=False).check(model, timeout=timeout)


def autodetect_model(settings: Settings) -> str:
    """For local servers: the first model the server reports as loaded."""
    try:
        models = list_catalog(settings)
    except Exception:  # an unreachable local server surfaces on first request
        return ""
    return models[0] if models else ""
