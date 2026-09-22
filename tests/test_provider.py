from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
import requests

from scode.errors import AuthError, ProviderError
from scode.providers.base import AssistantMessage, StreamDone, TextDelta, ToolCall, ToolCallStarted
from scode.providers.openai_compatible import OpenAICompatibleProvider, ToolsNotSupported


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: Any = None,
        lines: list[str] | None = None,
        headers: dict[str, str] | None = None,
        url: str = "https://example.test/v1/chat/completions",
    ) -> None:
        self.status_code = status_code
        self._body = body
        self._lines = lines or []
        self.headers = headers or {}
        self.url = url
        self.closed = False
        self.text = json.dumps(body) if body is not None else ""

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def iter_lines(self, decode_unicode: bool = False) -> Iterator[bytes]:
        for line in self._lines:
            yield line.encode("utf-8")

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload)


def make_provider(responses: list[FakeResponse], monkeypatch: pytest.MonkeyPatch) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(
        name="test",
        api_key="nvapi-test",
        base_url="https://example.test/v1",
        model="test/model",
    )
    queue = list(responses)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return queue.pop(0)

    monkeypatch.setattr(provider.session, "post", fake_post)
    provider.__dict__["_calls"] = calls
    return provider


# ------------------------------------------------------------------ payload

def test_payload_includes_tools_and_stream_options(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([FakeResponse(lines=["data: [DONE]"])], monkeypatch)
    tools = [{"type": "function", "function": {"name": "Read", "parameters": {}}}]
    list(provider.stream([{"role": "user", "content": "hi"}], tools=tools))

    payload = provider.__dict__["_calls"][0]["json"]
    assert payload["model"] == "test/model"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"


def test_tools_are_omitted_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([FakeResponse(lines=["data: [DONE]"])], monkeypatch)
    provider.supports_tools = False
    list(provider.stream([{"role": "user", "content": "hi"}], tools=[{"x": 1}]))
    assert "tools" not in provider.__dict__["_calls"][0]["json"]


# ----------------------------------------------------------------- streaming

def test_stream_collects_text(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        sse({"choices": [{"delta": {"content": "Hel"}}]}),
        sse({"choices": [{"delta": {"content": "lo"}}]}),
        sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    events = list(provider.stream([{"role": "user", "content": "hi"}]))

    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert deltas == ["Hel", "lo"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.message.content == "Hello"
    assert done.message.finish_reason == "stop"


def test_stream_ignores_comments_and_blank_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        "",
        ": keep-alive",
        "event: ping",
        sse({"choices": [{"delta": {"content": "ok"}}]}),
        "data: [DONE]",
    ]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    done = list(provider.stream([{"role": "user", "content": "x"}]))[-1]
    assert isinstance(done, StreamDone)
    assert done.message.content == "ok"


def test_stream_survives_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        "data: {not json}",
        sse({"choices": [{"delta": {"content": "fine"}}]}),
        "data: [DONE]",
    ]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    done = list(provider.stream([{"role": "user", "content": "x"}]))[-1]
    assert done.message.content == "fine"


def test_stream_assembles_tool_calls_across_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_a", "function": {"name": "Read", "arguments": '{"file'}}
        ]}}]}),
        sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '_path": "a.py"}'}}
        ]}}]}),
        sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]",
    ]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    events = list(provider.stream([{"role": "user", "content": "x"}]))

    assert any(isinstance(e, ToolCallStarted) and e.name == "Read" for e in events)
    message = events[-1].message
    assert len(message.tool_calls) == 1
    call = message.tool_calls[0]
    assert call.id == "call_a"
    assert call.name == "Read"
    assert call.arguments == {"file_path": "a.py"}


def test_stream_handles_parallel_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c0", "function": {"name": "Read", "arguments": '{"file_path":"a"}'}},
            {"index": 1, "id": "c1", "function": {"name": "LS", "arguments": "{}"}},
        ]}}]}),
        "data: [DONE]",
    ]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    message = list(provider.stream([{"role": "user", "content": "x"}]))[-1].message
    assert [c.name for c in message.tool_calls] == ["Read", "LS"]


def test_stream_captures_reasoning_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        sse({"choices": [{"delta": {"reasoning_content": "thinking..."}}]}),
        sse({"choices": [{"delta": {"content": "answer"}}]}),
        sse({"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 8,
                                      "prompt_tokens_details": {"cached_tokens": 40}}}),
        "data: [DONE]",
    ]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    message = list(provider.stream([{"role": "user", "content": "x"}]))[-1].message
    assert message.reasoning == "thinking..."
    assert message.input_tokens == 120
    assert message.output_tokens == 8
    assert message.cached_tokens == 40


def test_stream_handles_content_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        sse({"choices": [{"delta": {"content": [{"type": "text", "text": "part"}]}}]}),
        "data: [DONE]",
    ]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    assert list(provider.stream([{"role": "user", "content": "x"}]))[-1].message.content == "part"


def test_usage_is_estimated_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [sse({"choices": [{"delta": {"content": "hello there"}}]}), "data: [DONE]"]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    message = list(provider.stream([{"role": "user", "content": "x" * 400}]))[-1].message
    assert message.input_tokens > 0
    assert message.output_tokens > 0


# -------------------------------------------------------------- completion

def test_complete_parses_a_message(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "choices": [{
            "message": {
                "content": "done",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "Read", "arguments": '{"file_path":"a.py"}'}}
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    provider = make_provider([FakeResponse(body=body)], monkeypatch)
    message = provider.complete([{"role": "user", "content": "x"}])

    assert message.content == "done"
    assert message.tool_calls[0].arguments == {"file_path": "a.py"}
    assert message.input_tokens == 10


def test_complete_rejects_an_empty_choice_list(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([FakeResponse(body={"choices": []})], monkeypatch)
    with pytest.raises(ProviderError, match="no choices"):
        provider.complete([{"role": "user", "content": "x"}])


# ------------------------------------------------------------------- errors

def test_401_becomes_an_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(status_code=401, body={"error": {"message": "bad key"}})
    provider = make_provider([response], monkeypatch)
    with pytest.raises(AuthError, match="rejected the API key"):
        provider.complete([{"role": "user", "content": "x"}])


def test_404_says_the_model_is_not_on_this_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """NVIDIA returns 404 for catalog models an account was never granted."""
    provider = make_provider([FakeResponse(status_code=404, body={"detail": "nope"})], monkeypatch)
    with pytest.raises(ProviderError, match="not available on your account"):
        provider.complete([{"role": "user", "content": "x"}])


def test_410_reports_a_retired_model(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"detail": "The model has reached its end of life on 2026-08-26"}
    provider = make_provider([FakeResponse(status_code=410, body=body)], monkeypatch)
    with pytest.raises(ProviderError, match="retired by NVIDIA"):
        provider.complete([{"role": "user", "content": "x"}])


def test_504_is_retried_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wedged gateway answers only after the full read timeout."""
    monkeypatch.setattr("scode.providers.openai_compatible.time.sleep", lambda _s: None)
    responses = [FakeResponse(status_code=504) for _ in range(4)]
    provider = make_provider(responses, monkeypatch)
    with pytest.raises(ProviderError, match="did not respond"):
        provider.complete([{"role": "user", "content": "x"}])
    # Two attempts, not the full retry budget.
    assert len(provider.__dict__["_calls"]) == 2


def test_known_bad_models_explain_themselves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scode.providers.openai_compatible.time.sleep", lambda _s: None)
    provider = make_provider([FakeResponse(status_code=504) for _ in range(2)], monkeypatch)
    provider.model = "moonshotai/kimi-k3"
    with pytest.raises(ProviderError, match="504 after"):
        provider.complete([{"role": "user", "content": "x"}])


def test_check_model_classifies_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    from scode.providers.openai_compatible import check_model

    cases = {200: "ok", 404: "unavailable", 410: "retired", 401: "error", 500: "error"}
    for status, expected in cases.items():
        monkeypatch.setattr(
            "scode.providers.openai_compatible.requests.post",
            lambda *a, _s=status, **k: FakeResponse(status_code=_s, lines=["data: {}"]),
        )
        state, _ = check_model("https://x.test/v1", "k", "m/model")
        assert state == expected, f"{status} -> {state}"


def test_check_model_reports_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from scode.providers.openai_compatible import check_model

    def boom(*a, **k):
        raise requests.Timeout("slow")

    monkeypatch.setattr("scode.providers.openai_compatible.requests.post", boom)
    state, detail = check_model("https://x.test/v1", "k", "m/model")
    assert state == "timeout"
    assert "no response" in detail


def test_400_about_tools_signals_no_tool_support(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(
        status_code=400, body={"error": {"message": "tools are not supported for this model"}}
    )
    provider = make_provider([response], monkeypatch)
    with pytest.raises(ToolsNotSupported):
        provider.complete([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert provider.supports_tools is False


def test_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scode.providers.openai_compatible.time.sleep", lambda _s: None)
    responses = [
        FakeResponse(status_code=429, headers={"Retry-After": "0"}),
        FakeResponse(body={"choices": [{"message": {"content": "ok"}}]}),
    ]
    provider = make_provider(responses, monkeypatch)
    assert provider.complete([{"role": "user", "content": "x"}]).content == "ok"
    assert responses[0].closed is True


def test_connection_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAICompatibleProvider(
        name="t", api_key="k", base_url="https://example.test/v1", model="m"
    )

    def boom(*args: Any, **kwargs: Any):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(provider.session, "post", boom)
    with pytest.raises(ProviderError, match="Could not reach"):
        provider.complete([{"role": "user", "content": "x"}])


def test_list_models(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAICompatibleProvider(
        name="t", api_key="k", base_url="https://example.test/v1", model="m"
    )
    monkeypatch.setattr(
        provider.session,
        "get",
        lambda url, **kwargs: FakeResponse(body={"data": [{"id": "b/model"}, {"id": "a/model"}]}),
    )
    monkeypatch.setattr(FakeResponse, "raise_for_status", lambda self: None, raising=False)
    assert provider.list_models() == ["a/model", "b/model"]


# ------------------------------------------------------------- tool parsing

def test_tool_call_from_raw_handles_bad_json() -> None:
    call = ToolCall.from_raw("id", "Read", "{oops")
    assert call.parse_error
    assert call.arguments == {}


def test_tool_call_from_raw_rejects_non_objects() -> None:
    call = ToolCall.from_raw("id", "Read", "[1, 2]")
    assert "must be a JSON object" in (call.parse_error or "")


def test_tool_call_empty_arguments() -> None:
    call = ToolCall.from_raw("id", "LS", "")
    assert call.arguments == {}
    assert call.parse_error is None


def test_assistant_message_serialises_tool_calls() -> None:
    message = AssistantMessage(
        content="hi",
        tool_calls=[ToolCall(id="c1", name="Read", raw_arguments='{"file_path":"a"}')],
    )
    payload = message.to_message()
    assert payload["role"] == "assistant"
    assert payload["tool_calls"][0]["function"]["name"] == "Read"
    assert payload["tool_calls"][0]["type"] == "function"
