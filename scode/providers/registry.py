from __future__ import annotations

from .openai_compatible import OpenAICompatibleProvider
from ..config import AppConfig


def get_provider(config: AppConfig) -> OpenAICompatibleProvider:
    provider = config.provider.lower()

    if provider == "nvidia":
        if not config.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is required for provider 'nvidia'")
        return OpenAICompatibleProvider(
            name="nvidia",
            api_key=config.nvidia_api_key,
            base_url=config.nvidia_base_url,
        )

    if provider in {"kimi", "kimi-k3"}:
        if not config.kimi_api_key:
            raise ValueError("KIMI_API_KEY is required for provider 'kimi-k3'")
        return OpenAICompatibleProvider(
            name="kimi-k3",
            api_key=config.kimi_api_key,
            base_url=config.kimi_base_url,
        )

    raise ValueError(f"Unsupported provider: {config.provider}")
