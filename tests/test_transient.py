"""Overloads, rate limits and dropped streams are retried instead of ending the turn."""

from __future__ import annotations

import pytest
import requests
from conftest import text_turn, tool_turn
from test_provider import FakeResponse, make_provider, sse

from scode.agent import loop as loop_module
from scode.errors import (
    ContextOverflow,
    ProviderError,
    TransientProviderError,
    is_transient,
)


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff waits instead of sleeping through them."""
    seen: list[float] = []
    monkeypatch.setattr(loop_module.Agent, "_wait", staticmethod(lambda s: seen.append(s) or True))
    return seen


# ------------------------------------------------------------ classification

@pytest.mark.parametrize("detail, code", [
    ("Service temporarily overloaded", None),
    ("Rate limit reached for requests", None),
    ("upstream timed out", None),
    ("anything", 503),
    ("anything", "overloaded_error"),
    ("Too Many Requests", None),
])
def test_transient_failures_are_recognised(detail, code) -> None:
    assert is_transient(detail, code)


@pytest.mark.parametrize("detail", ["Invalid API key", "model not found", "messages: field required"])
def test_permanent_failures_are_not_retried(detail) -> None:
    assert not is_transient(detail, "invalid_request_error")


# ------------------------------------------------------------ OpenAI wire

def _stream(provider):
    return list(provider.stream([{"role": "user", "content": "x"}]))


def test_an_overload_inside_the_stream_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [sse({"error": {"message": "Service temporarily overloaded"}}), "data: [DONE]"]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    with pytest.raises(TransientProviderError, match="temporarily overloaded"):
        _stream(provider)


def test_a_permanent_stream_error_is_not_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [sse({"error": {"message": "Invalid tool schema", "code": "invalid_request_error"}})]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    with pytest.raises(ProviderError) as info:
        _stream(provider)
    assert not isinstance(info.value, TransientProviderError)


def test_an_overflow_inside_the_stream_triggers_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [sse({"error": {"message": "This model's maximum context length is 8192 tokens"}})]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    with pytest.raises(ContextOverflow):
        _stream(provider)


def test_a_dropped_connection_mid_stream_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    class Dropping(FakeResponse):
        def iter_lines(self, decode_unicode: bool = False):
            yield sse({"choices": [{"delta": {"content": "par"}}]}).encode()
            raise requests.ConnectionError("Connection broken: IncompleteRead")

    provider = make_provider([Dropping()], monkeypatch)
    with pytest.raises(TransientProviderError, match="interrupted"):
        _stream(provider)


def test_rate_limits_that_outlast_http_retries_are_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scode.providers.openai_compatible.time.sleep", lambda s: None)
    limited = [FakeResponse(status_code=429, body={"error": {"message": "slow down"}}) for _ in range(4)]
    provider = make_provider(limited, monkeypatch)
    with pytest.raises(TransientProviderError, match="Rate limited"):
        _stream(provider)


# ------------------------------------------------------------ agent loop

def test_the_loop_waits_and_retries_a_transient_failure(make_agent, waits) -> None:
    overloaded = TransientProviderError("nvidia stream error: Service temporarily overloaded")
    agent, provider = make_agent([overloaded, overloaded, text_turn("done")])
    result = agent.run("do it")

    assert result.reason == "done" and result.text == "done"
    assert waits == [5.0, 10.0]
    # The failed attempts left nothing behind in the history.
    assert [m["role"] for m in agent.messages if m["role"] != "system"] == ["user", "assistant"]


def test_retries_do_not_spend_the_step_budget(make_agent, waits) -> None:
    overloaded = TransientProviderError("overloaded")
    agent, _ = make_agent([overloaded, overloaded, overloaded, text_turn("done")], max_steps=1)
    assert agent.run("go").reason == "done"


def test_the_retry_budget_resets_after_progress(make_agent, waits, workspace) -> None:
    (workspace / "a.txt").write_text("hi\n", encoding="utf-8")
    overloaded = TransientProviderError("overloaded")
    turns = [overloaded, overloaded, overloaded, tool_turn("Read", {"file_path": "a.txt"}),
             overloaded, overloaded, overloaded, text_turn("read it")]
    agent, _ = make_agent(turns)
    result = agent.run("read a.txt")
    assert result.reason == "done" and len(waits) == 6


def test_the_loop_gives_up_after_the_retry_budget(make_agent, waits) -> None:
    overloaded = TransientProviderError("overloaded")
    agent, _ = make_agent([overloaded] * (loop_module.MAX_TRANSIENT_RETRIES + 1) + [text_turn("never")])
    result = agent.run("go")
    assert result.reason == "error" and "overloaded" in result.error
    assert len(waits) == loop_module.MAX_TRANSIENT_RETRIES


def test_permanent_errors_are_not_retried(make_agent, waits) -> None:
    agent, _ = make_agent([ProviderError("model not found"), text_turn("never")])
    assert agent.run("go").reason == "error"
    assert waits == []


def test_ctrl_c_during_the_wait_interrupts_the_turn(make_agent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop_module.Agent, "_wait", staticmethod(lambda s: False))
    agent, _ = make_agent([TransientProviderError("overloaded"), text_turn("never")])
    assert agent.run("go").reason == "interrupted"


# ------------------------------------------------------------ Anthropic wire

def test_anthropic_overloads_and_rate_limits_are_transient() -> None:
    anthropic = pytest.importorskip("anthropic")
    # Build responses with the HTTP library this SDK version uses (httpx2 in 1.x).
    try:
        import httpx2 as httpx
    except ImportError:
        httpx = pytest.importorskip("httpx")

    from scode.providers.anthropic_native import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-ant-test", model="claude-opus-5", client=object())
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def status_error(cls, status: int, message: str):
        return cls(message, response=httpx.Response(status, request=request), body=None)

    assert isinstance(provider._translate(status_error(anthropic.RateLimitError, 429, "slow"), "m"),
                      TransientProviderError)
    assert isinstance(provider._translate(status_error(anthropic.InternalServerError, 529, "Overloaded"), "m"),
                      TransientProviderError)
    assert isinstance(provider._translate(status_error(anthropic.InternalServerError, 500, "boom"), "m"),
                      TransientProviderError)
    bad = provider._translate(status_error(anthropic.BadRequestError, 400, "tools.0: invalid"), "m")
    assert isinstance(bad, ProviderError) and not isinstance(bad, TransientProviderError)
