"""Shared fixtures."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from scode.agent.loop import Agent
from scode.config import Settings, load_settings
from scode.permissions import PermissionEngine
from scode.providers.base import (
    AssistantMessage,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCall,
)
from scode.tools import ToolContext, build_registry
from scode.ui.console import UI
from scode.usage import Usage

# Every provider key a developer might have exported; tests must never see them.
_PROVIDER_KEYS = (
    "NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OMNIROUTE_API_KEY", "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY",
    "TOGETHER_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY", "FIREWORKS_API_KEY",
    "CEREBRAS_API_KEY", "MOONSHOT_API_KEY", "ZAI_API_KEY", "ZHIPU_API_KEY", "OLLAMA_API_KEY",
    "LMSTUDIO_API_KEY",
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to reach a real host. Local test servers are fine."""
    from urllib.parse import urlparse

    import requests.adapters

    original = requests.adapters.HTTPAdapter.send

    def guarded(self, request, *args, **kwargs):
        host = urlparse(request.url).hostname or ""
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise AssertionError(f"test tried to reach the network: {request.method} {request.url}")
        return original(self, request, *args, **kwargs)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", guarded)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every test out of the real ~/.scode, the network, and real keys."""
    from scode.providers.catalog import set_catalog

    home = tmp_path / "scode-home"
    home.mkdir()
    monkeypatch.setenv("SCODE_HOME", str(home))
    monkeypatch.setenv("SCODE_OFFLINE", "1")
    for name in (
        "SCODE_API_KEY", "SCODE_MODEL", "SCODE_PROVIDER", "SCODE_BASE_URL", "SCODE_TEMPERATURE",
        "SCODE_MAX_OUTPUT_TOKENS", "SCODE_MAX_STEPS", "SCODE_PERMISSION_MODE", "SCODE_THEME",
        "SCODE_AUTO_COMPACT", "SCODE_NATIVE_TOOLS", "SCODE_EFFORT", "SCODE_DEBUG",
        "SCODE_SMALL_MODEL", "SCODE_SHELL", *_PROVIDER_KEYS,
    ):
        monkeypatch.delenv(name, raising=False)
    # Each test starts with an empty in-memory catalog (and SCODE_OFFLINE stops fetches).
    set_catalog(None)
    yield home
    set_catalog(None)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "project"
    ws.mkdir()
    (ws / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (ws / "README.md").write_text("# Demo\n", encoding="utf-8")
    return ws


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return load_settings(
        workspace,
        {"api_key": "nvapi-test", "permission_mode": "bypassPermissions", "stream": False},
    )


@pytest.fixture
def ui() -> UI:
    return UI(quiet=True)


@pytest.fixture
def ctx(settings: Settings) -> ToolContext:
    return ToolContext(
        workspace=settings.workspace,
        settings=settings,
        permissions=PermissionEngine(settings),
    )


class FakeProvider:
    """Replays a scripted list of assistant turns."""

    def __init__(self, turns: list[AssistantMessage] | None = None, model: str = "fake/model") -> None:
        self.turns = list(turns or [])
        self.model = model
        self.name = "fake"
        self.supports_tools = True
        self.requests: list[list[dict[str, Any]]] = []
        self.tools_seen: list[Any] = []
        self.models_seen: list[str | None] = []

    def _next(self, messages: list[dict[str, Any]], tools: Any, model: str | None = None) -> AssistantMessage:
        self.requests.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        self.models_seen.append(model)
        if not self.turns:
            return AssistantMessage(content="(no more scripted turns)")
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    def complete(self, messages, *, tools=None, model=None, max_output_tokens=None, temperature=None):
        return self._next(messages, tools, model)

    def stream(self, messages, *, tools=None, model=None, max_output_tokens=None, temperature=None) -> Iterator[StreamEvent]:
        message = self._next(messages, tools, model)
        if message.content:
            yield TextDelta(message.content)
        yield StreamDone(message)

    def list_models(self) -> list[str]:
        return ["fake/model", "fake/other"]

    def check(self, model: str, *, timeout: float = 25.0) -> tuple[str, str]:
        return ("ok", "0.1s") if model in self.list_models() else ("unavailable", "not granted")


def tool_turn(name: str, arguments: dict[str, Any], *, content: str = "", call_id: str = "call_1") -> AssistantMessage:
    return AssistantMessage(
        content=content,
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                raw_arguments=json.dumps(arguments),
            )
        ],
    )


def multi_tool_turn(*calls: tuple[str, dict[str, Any]]) -> AssistantMessage:
    return AssistantMessage(
        tool_calls=[
            ToolCall(id=f"c{i}", name=name, arguments=args, raw_arguments=json.dumps(args))
            for i, (name, args) in enumerate(calls)
        ]
    )


def text_turn(content: str) -> AssistantMessage:
    return AssistantMessage(content=content)


@pytest.fixture
def make_agent(settings: Settings, ui: UI):
    def factory(
        turns: list[AssistantMessage],
        *,
        permission_mode: str | None = None,
        read_only: bool = False,
        hooks: Any = None,
        **overrides: Any,
    ) -> tuple[Agent, FakeProvider]:
        from scode.config import with_overrides

        local = with_overrides(settings, **overrides)
        engine = PermissionEngine(local)
        if permission_mode:
            engine.set_mode(permission_mode)
        provider = FakeProvider(turns)
        registry = build_registry(read_only=read_only, include_task=not read_only)
        context = ToolContext(
            workspace=local.workspace,
            settings=local,
            permissions=engine,
        )
        agent = Agent(
            provider=provider,  # type: ignore[arg-type]
            registry=registry,
            ctx=context,
            ui=ui,
            settings=local,
            usage=Usage(),
            hooks=hooks,
        )
        return agent, provider

    return factory


def sent_text(request: list[dict[str, Any]]) -> str:
    """Everything a request carried after the system prompt, as one string."""
    from scode.providers.base import text_of

    return "\n".join(text_of(m.get("content")) for m in request[1:])
