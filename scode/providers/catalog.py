"""Model metadata from models.dev: output limits, pricing, and capabilities.

This is enrichment, never a dependency. scode works fully offline; the catalog
only sharpens defaults (output caps, vision support, prices) and lets scode
resolve providers it has no built-in profile for. The JSON is cached on disk
and refreshed at most once a day.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from ..constants import home_dir

MODELS_DEV_URL = "https://models.dev/api.json"
CACHE_MAX_AGE = 24 * 3600
FETCH_TIMEOUT = 8


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str = ""
    context: int = 0
    output: int = 0
    tool_call: bool = False
    reasoning: bool = False
    vision: bool = False
    # USD per million tokens; None when the catalog has no price.
    input_cost: float | None = None
    output_cost: float | None = None
    cache_read_cost: float | None = None
    cache_write_cost: float | None = None
    release_date: str = ""

    @property
    def priced(self) -> bool:
        return self.input_cost is not None and self.output_cost is not None


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    name: str = ""
    api: str = ""
    env: tuple[str, ...] = ()
    npm: str = ""
    doc: str = ""
    models: dict[str, ModelInfo] = field(default_factory=dict)

    @property
    def openai_compatible(self) -> bool:
        return bool(self.api) and "openai" in self.npm

    def newest_tool_models(self) -> list[ModelInfo]:
        tools = [m for m in self.models.values() if m.tool_call]
        return sorted(tools, key=lambda m: m.release_date, reverse=True)


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _parse_model(raw: dict[str, Any]) -> ModelInfo:
    cost = raw.get("cost") or {}
    limit = raw.get("limit") or {}
    modalities = raw.get("modalities") or {}
    return ModelInfo(
        id=str(raw.get("id", "")),
        name=str(raw.get("name", "")),
        context=int(limit.get("context") or 0),
        output=int(limit.get("output") or 0),
        tool_call=bool(raw.get("tool_call")),
        reasoning=bool(raw.get("reasoning")),
        vision="image" in (modalities.get("input") or []),
        input_cost=_float(cost.get("input")),
        output_cost=_float(cost.get("output")),
        cache_read_cost=_float(cost.get("cache_read")),
        cache_write_cost=_float(cost.get("cache_write")),
        release_date=str(raw.get("release_date") or ""),
    )


def _parse_provider(pid: str, raw: dict[str, Any]) -> ProviderInfo:
    models = {}
    for model_id, entry in (raw.get("models") or {}).items():
        if isinstance(entry, dict):
            models[model_id] = _parse_model({"id": model_id, **entry})
    env = raw.get("env") or ()
    return ProviderInfo(
        id=pid,
        name=str(raw.get("name", pid)),
        api=str(raw.get("api") or ""),
        env=tuple(str(e) for e in env),
        npm=str(raw.get("npm") or ""),
        doc=str(raw.get("doc") or ""),
        models=models,
    )


class Catalog:
    """Parsed models.dev data. An empty catalog is valid and answers None."""

    def __init__(self, raw: dict[str, Any] | None = None, *, fetched_at: float = 0.0) -> None:
        self._raw = raw or {}
        self._providers: dict[str, ProviderInfo] = {}
        self.fetched_at = fetched_at

    def __bool__(self) -> bool:
        return bool(self._raw)

    def provider(self, pid: str) -> ProviderInfo | None:
        if not pid:
            return None
        if pid not in self._providers:
            raw = self._raw.get(pid)
            if not isinstance(raw, dict):
                return None
            self._providers[pid] = _parse_provider(pid, raw)
        return self._providers[pid]

    def provider_ids(self) -> list[str]:
        return sorted(k for k, v in self._raw.items() if isinstance(v, dict))

    def model(self, pid: str, model_id: str) -> ModelInfo | None:
        provider = self.provider(pid)
        if provider is None or not model_id:
            return None
        found = provider.models.get(model_id)
        if found:
            return found
        # OpenRouter-style ids carry a ":variant" suffix the catalog may lack.
        base = model_id.split(":", 1)[0]
        return provider.models.get(base)


_lock = threading.Lock()
_cached: Catalog | None = None


def cache_path() -> Path:
    return home_dir() / "cache" / "models.dev.json"


def _network_allowed() -> bool:
    return os.getenv("SCODE_OFFLINE", "").lower() not in {"1", "true", "yes"}


def _read_cache() -> tuple[dict[str, Any] | None, float]:
    path = cache_path()
    try:
        mtime = path.stat().st_mtime
        return json.loads(path.read_text(encoding="utf-8")), mtime
    except (OSError, ValueError):
        return None, 0.0


def _fetch() -> dict[str, Any] | None:
    try:
        response = requests.get(
            MODELS_DEV_URL, timeout=FETCH_TIMEOUT, headers={"User-Agent": "scode-cli"}
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass  # a read-only home only costs us the cache
    return data


def get_catalog(*, refresh: bool = False) -> Catalog:
    """Return the catalog, loading from disk or network as needed.

    Never raises: with no cache and no network the result is an empty catalog.
    """
    global _cached
    with _lock:
        if _cached is not None and not refresh:
            return _cached

        data, mtime = _read_cache()
        stale = data is None or (time.time() - mtime) > CACHE_MAX_AGE
        if (stale or refresh) and _network_allowed():
            fresh = _fetch()
            if fresh is not None:
                data, mtime = fresh, time.time()

        _cached = Catalog(data, fetched_at=mtime)
        return _cached


def set_catalog(catalog: Catalog | None) -> None:
    """Replace the in-memory catalog (used by tests)."""
    global _cached
    with _lock:
        _cached = catalog
