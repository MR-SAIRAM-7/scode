"""Builds a provider from settings."""

from __future__ import annotations

from ..config import Settings
from ..constants import NVIDIA_BASE_URL
from ..errors import ConfigError
from .openai_compatible import OpenAICompatibleProvider

PROVIDERS = {
    "nvidia": NVIDIA_BASE_URL,
    # Any other endpoint that speaks the OpenAI chat API.
    "openai-compatible": None,
    "custom": None,
}


def get_provider(settings: Settings, *, model: str | None = None) -> OpenAICompatibleProvider:
    name = settings.provider.lower()
    if name not in PROVIDERS:
        raise ConfigError(
            f"Unknown provider {settings.provider!r}. Choose from: {', '.join(PROVIDERS)}"
        )

    base_url = settings.base_url
    if name == "nvidia" and not base_url:
        base_url = NVIDIA_BASE_URL
    if not base_url:
        raise ConfigError(
            f"Provider {name!r} needs a base_url. Set SCODE_BASE_URL or base_url in settings."
        )

    return OpenAICompatibleProvider(
        name=name,
        api_key=settings.require_api_key(),
        base_url=base_url,
        model=model or settings.model,
        max_output_tokens=settings.max_output_tokens,
        temperature=settings.temperature,
    )


def list_catalog(settings: Settings) -> list[str]:
    """Live model list from the provider. Works unauthenticated on NVIDIA."""
    provider = OpenAICompatibleProvider(
        name=settings.provider,
        api_key=settings.api_key or "",
        base_url=settings.base_url or NVIDIA_BASE_URL,
        model=settings.model,
    )
    return provider.list_models()
