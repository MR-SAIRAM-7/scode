"""Static constants: paths, model catalog, defaults."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "scode"
VERSION = "2.0.0"

PRODUCT_TAGLINE = "Agentic coding CLI for any model provider"

# ---------------------------------------------------------------- filesystem

def home_dir() -> Path:
    """Root for user-level state (~/.scode), overridable for tests."""
    override = os.getenv("SCODE_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".scode"


def projects_dir() -> Path:
    return home_dir() / "projects"


USER_SETTINGS_FILE = "settings.json"
PROJECT_DIR_NAME = ".scode"
MEMORY_FILE = "SCODE.md"
# Also honoured so existing repos work unchanged.
ALT_MEMORY_FILES = ("CLAUDE.md", "AGENTS.md")

# ------------------------------------------------------------------ provider

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

DEFAULT_PROVIDER = "nvidia"
# Verified against the live NVIDIA endpoint: responds in ~1s and supports
# function calling. Run `/model --check` to see what your own key can reach —
# the public catalog lists far more models than any one account is granted.
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_SMALL_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

DEFAULT_MAX_OUTPUT_TOKENS = 32_000
DEFAULT_TEMPERATURE = 0.6
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_MAX_STEPS = 60

# Models verified working against the live NVIDIA endpoint, with the measured
# time to first token. Availability is per-account and changes over time, so
# `/model --check` probes for real rather than trusting this list.
MODEL_CATALOG: dict[str, dict[str, object]] = {
    "nvidia/nemotron-3-super-120b-a12b": {
        "label": "Nemotron 3 Super 120B",
        "context": 128_000,
        "tools": True,
        "note": "Flagship MoE, ~1s to first token. Default.",
    },
    "nvidia/nemotron-3-ultra-550b-a55b": {
        "label": "Nemotron 3 Ultra 550B",
        "context": 128_000,
        "tools": True,
        "note": "Largest Nemotron; slower (~19s) but strongest.",
    },
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": {
        "label": "Nemotron 3 Nano Omni 30B",
        "context": 128_000,
        "tools": True,
        "note": "Fast and cheap; used for summarising.",
    },
    "openai/gpt-oss-20b": {
        "label": "GPT-OSS 20B",
        "context": 128_000,
        "tools": True,
        "note": "Open-weight GPT with reasoning.",
    },
    "poolside/laguna-xs-2.1": {
        "label": "Laguna XS 2.1",
        "context": 128_000,
        "tools": True,
        "note": "Small coding model, no reasoning trace.",
    },
    "meta/llama-3.2-11b-vision-instruct": {
        "label": "Llama 3.2 11B Vision",
        "context": 128_000,
        "tools": True,
        "note": "Multimodal; accepts image_url content.",
    },
}

# Models whose gateway hangs or that are commonly ungranted. Named in errors so
# the failure is explained rather than just slow.
KNOWN_UNAVAILABLE = {
    "moonshotai/kimi-k3": "the gateway returns 504 after ~5 minutes",
    "moonshotai/kimi-k2.6": "not granted to most accounts",
    "deepseek-ai/deepseek-v4.1-flash": "the gateway times out",
    "z-ai/glm-5.3": "the gateway times out",
    "z-ai/glm-5.3-flash": "the gateway times out",
    "nvidia/nemotron-nano-3-30b-a3b": "not granted to most accounts",
    "meta/llama-3.3-70b-instruct": "reached end of life on 2026-08-26",
}


def context_window_for(model: str) -> int:
    entry = MODEL_CATALOG.get(model)
    if entry:
        return int(entry["context"])  # type: ignore[arg-type]
    return DEFAULT_CONTEXT_WINDOW


# ---------------------------------------------------------------- behaviour

# Fraction of the context window at which auto-compaction kicks in.
AUTO_COMPACT_THRESHOLD = 0.82

BASH_DEFAULT_TIMEOUT = 120
BASH_MAX_TIMEOUT = 600
MAX_TOOL_OUTPUT_CHARS = 30_000
MAX_READ_LINES = 2_000
MAX_LINE_CHARS = 2_000
