"""Debug logging, off unless --debug or SCODE_DEBUG is set.

Logs request metadata (provider, model, sizes, timing, usage, stop reason),
errors, hooks and MCP activity. Never message content or credentials.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from .constants import home_dir

LOGGER_NAME = "scode"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(enabled: bool) -> Path | None:
    logger = get_logger()
    if not enabled:
        logger.addHandler(logging.NullHandler())
        return None
    path = home_dir() / "logs" / f"scode-{date.today().isoformat()}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return path
