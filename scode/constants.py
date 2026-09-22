"""Static constants: paths, model catalog, defaults."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "scode"
VERSION = "1.0.0"

PRODUCT_TAGLINE = "Agentic coding CLI powered by NVIDIA NIM"

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
DEFAULT_MODEL = "moonshotai/kimi-k3"
DEFAULT_SMALL_MODEL = "nvidia/nemotron-nano-3-30b-a3b"

DEFAULT_MAX_OUTPUT_TOKENS = 32_000
DEFAULT_TEMPERATURE = 0.6
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_MAX_STEPS = 60

# Curated shortlist shown by `/model`. `/model --all` queries the live catalog,
# so this list never has to be exhaustive.
MODEL_CATALOG: dict[str, dict[str, object]] = {
    "moonshotai/kimi-k3": {
        "label": "Kimi K3",
        "context": 256_000,
        "tools": True,
        "note": "Strong agentic coder. Default.",
    },
    "moonshotai/kimi-k2.6": {
        "label": "Kimi K2.6",
        "context": 128_000,
        "tools": True,
        "note": "Previous Kimi generation.",
    },
    "deepseek-ai/deepseek-v4.1-flash": {
        "label": "DeepSeek V4.1 Flash",
        "context": 128_000,
        "tools": True,
        "note": "Fast, low latency.",
    },
    "z-ai/glm-5.3": {
        "label": "GLM 5.3",
        "context": 128_000,
        "tools": True,
        "note": "Solid general coder.",
    },
    "z-ai/glm-5.3-flash": {
        "label": "GLM 5.3 Flash",
        "context": 128_000,
        "tools": True,
        "note": "Speed-tuned GLM.",
    },
    "nvidia/nemotron-3-super-120b-a12b": {
        "label": "Nemotron 3 Super 120B",
        "context": 128_000,
        "tools": True,
        "note": "NVIDIA flagship MoE.",
    },
    "nvidia/nemotron-3-ultra-550b-a55b": {
        "label": "Nemotron 3 Ultra 550B",
        "context": 128_000,
        "tools": True,
        "note": "Largest Nemotron.",
    },
    "nvidia/nemotron-nano-3-30b-a3b": {
        "label": "Nemotron Nano 3 30B",
        "context": 128_000,
        "tools": True,
        "note": "Cheap/fast; good for summaries.",
    },
    "openai/gpt-oss-20b": {
        "label": "GPT-OSS 20B",
        "context": 128_000,
        "tools": True,
        "note": "Open-weight GPT.",
    },
    "meta/llama-3.3-70b-instruct": {
        "label": "Llama 3.3 70B",
        "context": 128_000,
        "tools": True,
        "note": "Meta general purpose.",
    },
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
