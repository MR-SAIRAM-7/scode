from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

DEFAULT_PROVIDER = "nvidia"
DEFAULT_MODEL = "moonshotai/kimi-k3"
DEFAULT_MAX_OUTPUT_TOKENS = 60_000


@dataclass(frozen=True)
class AppConfig:
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    nvidia_api_key: str | None = None
    kimi_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    kimi_base_url: str = "https://api.moonshot.ai/v1"
    workspace: Path = Path.cwd()

    @staticmethod
    def from_env() -> "AppConfig":
        max_tokens_raw = os.getenv("SCODE_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))
        max_output_tokens = int(max_tokens_raw)
        if max_output_tokens <= 0:
            raise ValueError("SCODE_MAX_OUTPUT_TOKENS must be positive")

        provider = os.getenv("SCODE_PROVIDER", DEFAULT_PROVIDER)
        model = os.getenv("SCODE_MODEL", DEFAULT_MODEL)

        return AppConfig(
            provider=provider,
            model=model,
            max_output_tokens=max_output_tokens,
            nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
            kimi_api_key=os.getenv("KIMI_API_KEY"),
            nvidia_base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            kimi_base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1"),
            workspace=Path(os.getenv("SCODE_WORKSPACE", str(Path.cwd()))).resolve(),
        )
