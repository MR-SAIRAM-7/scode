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


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every test out of the real ~/.scode."""
    home = tmp_path / "scode-home"
    home.mkdir()
    monkeypatch.setenv("SCODE_HOME", str(home))
    for name in (
        "NVIDIA_API_KEY",
        "SCODE_API_KEY",
        "SCODE_MODEL",
        "SCODE_PROVIDER",
        "SCODE_BASE_URL",
        "SCODE_TEMPERATURE",
        "SCODE_MAX_OUTPUT_TOKENS",
        "SCODE_MAX_STEPS",
        "SCODE_PERMISSION_MODE",
        "SCODE_THEME",
        "SCODE_AUTO_COMPACT",
        "SCODE_NATIVE_TOOLS",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


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

    def __init__(self, turns: list[AssistantMessage] | None = None) -> None:
        self.turns = list(turns or [])
        self.model = "fake/model"
        self.name = "fake"
        self.supports_tools = True
        self.requests: list[list[dict[str, Any]]] = []
        self.tools_seen: list[Any] = []

    def _next(self, messages: list[dict[str, Any]], tools: Any) -> AssistantMessage:
        self.requests.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        if not self.turns:
            return AssistantMessage(content="(no more scripted turns)")
        return self.turns.pop(0)

    def complete(self, messages, *, tools=None, model=None, max_output_tokens=None, temperature=None):
        return self._next(messages, tools)

    def stream(self, messages, *, tools=None, model=None, max_output_tokens=None, temperature=None) -> Iterator[StreamEvent]:
        message = self._next(messages, tools)
        if message.content:
            yield TextDelta(message.content)
        yield StreamDone(message)

    def list_models(self) -> list[str]:
        return ["fake/model", "fake/other"]


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


def text_turn(content: str) -> AssistantMessage:
    return AssistantMessage(content=content)


@pytest.fixture
def make_agent(settings: Settings, ui: UI):
    def factory(
        turns: list[AssistantMessage],
        *,
        permission_mode: str | None = None,
        read_only: bool = False,
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
        )
        return agent, provider

    return factory
