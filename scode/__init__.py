"""scode — an agentic coding CLI powered by NVIDIA NIM models."""

from __future__ import annotations

from .constants import VERSION as __version__

__all__ = ["__version__", "main"]


def main(argv: list[str] | None = None) -> int:
    """Entry point, imported lazily to keep `import scode` cheap."""
    from .cli import main as _main

    return _main(argv)
