"""OpenAI-compatible adapter: parameter adaptation and request shape."""

from __future__ import annotations

import pytest
from test_provider import FakeResponse, make_provider, sse

from scode.errors import ContextOverflow, ProviderError
from scode.providers.base import AssistantMessage


def _bad(detail: str) -> FakeResponse:
    return FakeResponse(status_code=400, body={"error": {"message": detail}})


def _ok(content: str = "ok") -> FakeResponse:
    return FakeResponse(body={"choices": [{"message": {"content": content}}]})


def _calls(provider) -> list[dict]:
    return provider.__dict__["_calls"]


# ------------------------------------------------------- parameter adaptation

def test_switches_to_max_completion_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI reasoning models reject max_tokens outright."""
    provider = make_provider(
        [_bad("Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead."), _ok(), _ok()],
        monkeypatch,
    )
    assert provider.complete([{"role": "user", "content": "x"}]).content == "ok"
    calls = _calls(provider)
    assert "max_tokens" in calls[0]["json"]
    assert "max_completion_tokens" in calls[1]["json"] and "max_tokens" not in calls[1]["json"]
    # The fix is remembered: the next request is right the first time.
    provider.complete([{"role": "user", "content": "y"}])
    assert "max_completion_tokens" in calls[2]["json"]


def test_drops_a_rejected_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider(
        [_bad("temperature does not support 0.6 with this model. Only the default (1) value is supported."), _ok()],
        monkeypatch,
    )
    provider.temperature = 0.6
    provider.complete([{"role": "user", "content": "x"}])
    calls = _calls(provider)
    assert calls[0]["json"]["temperature"] == 0.6
    assert "temperature" not in calls[1]["json"]


def test_clamps_to_a_stated_output_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([_bad("max_tokens must be less than or equal to 16384, got 32000"), _ok()], monkeypatch)
    provider.complete([{"role": "user", "content": "x"}])
    assert _calls(provider)[1]["json"]["max_tokens"] == 16384


def test_drops_an_unsupported_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([_bad("Unrecognized request argument supplied: reasoning_effort"), _ok()], monkeypatch)
    provider.effort, provider.effort_style = "high", "reasoning_effort"
    provider.complete([{"role": "user", "content": "x"}])
    calls = _calls(provider)
    assert calls[0]["json"]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in calls[1]["json"]


def test_an_unfixable_400_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([_bad("messages[0].content is malformed")], monkeypatch)
    with pytest.raises(ProviderError, match="malformed"):
        provider.complete([{"role": "user", "content": "x"}])


def test_context_overflow_is_its_own_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider(
        [_bad("This model's maximum context length is 131072 tokens. However, you requested 140000 tokens")],
        monkeypatch,
    )
    with pytest.raises(ContextOverflow):
        provider.complete([{"role": "user", "content": "x"}])


def test_catalog_output_cap_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([_ok()], monkeypatch)
    provider.max_output_tokens = 32_000
    provider.output_limits = lambda model: 8192
    provider.complete([{"role": "user", "content": "x"}])
    assert _calls(provider)[0]["json"]["max_tokens"] == 8192


def test_a_tool_schema_error_is_not_mistaken_for_no_tool_support(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([_bad("Invalid 'tools[0].function.parameters': schema error")], monkeypatch)
    with pytest.raises(ProviderError, match="schema error"):
        provider.complete([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert provider.supports_tools is True


# ------------------------------------------------------------- request shape

def test_effort_styles(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([_ok(), _ok(), _ok()], monkeypatch)
    provider.effort = "high"
    for style, check in (
        ("reasoning_effort", lambda p: p["reasoning_effort"] == "high"),
        ("openrouter", lambda p: p["reasoning"] == {"effort": "high"}),
        ("none", lambda p: "reasoning" not in p and "reasoning_effort" not in p),
    ):
        provider.effort_style = style
        provider.complete([{"role": "user", "content": "x"}])
        assert check(_calls(provider)[-1]["json"]), style


def test_temperature_is_omitted_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([_ok()], monkeypatch)
    provider.complete([{"role": "user", "content": "x"}])
    assert "temperature" not in _calls(provider)[0]["json"]


def test_extra_headers_and_keyless_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([_ok()], monkeypatch)
    provider.api_key = ""
    provider.headers = {"X-Title": "scode"}
    provider.complete([{"role": "user", "content": "x"}])
    headers = _calls(provider)[0]["headers"]
    assert headers["X-Title"] == "scode"
    assert "Authorization" not in headers  # local servers need no key


def test_canonical_messages_are_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([_ok()], monkeypatch)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "look"},
                                     {"type": "image", "media_type": "image/png", "data": "QUJD"}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "LS", "arguments": "{}"}}],
         "provider_state": {"anthropic": {"content": [{"type": "thinking"}]}}},
        {"role": "tool", "tool_call_id": "c1", "name": "LS", "content": "a.py", "is_error": True},
    ]
    provider.complete(messages)
    wire = _calls(provider)[0]["json"]["messages"]
    assert wire[1]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    assert "provider_state" not in wire[2]
    assert wire[3] == {"role": "tool", "tool_call_id": "c1", "name": "LS", "content": "a.py"}


def test_reasoning_details_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenRouter wants reasoning_details echoed back on the next request."""
    lines = [
        sse({"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.text", "text": "Let me ", "index": 0}]}}]}),
        sse({"choices": [{"delta": {"reasoning_details": [{"text": "think.", "index": 0, "signature": "sig1"}]}}]}),
        sse({"choices": [{"delta": {"content": "done"}}]}),
        "data: [DONE]",
    ]
    provider = make_provider([FakeResponse(lines=lines), _ok()], monkeypatch)
    provider.echo_reasoning_details = True
    message = list(provider.stream([{"role": "user", "content": "x"}]))[-1].message

    details = message.provider_state["openai"]["reasoning_details"]
    assert details == [{"type": "reasoning.text", "text": "Let me think.", "index": 0, "signature": "sig1"}]
    assert message.reasoning == "Let me think."

    provider.complete([{"role": "user", "content": "x"}, message.to_message(), {"role": "user", "content": "y"}])
    assert _calls(provider)[1]["json"]["messages"][1]["reasoning_details"] == details


def test_reasoning_details_are_not_echoed_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider([_ok()], monkeypatch)
    assistant = {"role": "assistant", "content": "x",
                 "provider_state": {"openai": {"reasoning_details": [{"text": "t"}]}}}
    provider.complete([{"role": "user", "content": "a"}, assistant, {"role": "user", "content": "b"}])
    assert "reasoning_details" not in _calls(provider)[0]["json"]["messages"][1]


def test_stream_error_events_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [sse({"error": {"message": "upstream overloaded"}}), "data: [DONE]"]
    provider = make_provider([FakeResponse(lines=lines)], monkeypatch)
    with pytest.raises(ProviderError, match="upstream overloaded"):
        list(provider.stream([{"role": "user", "content": "x"}]))


def test_reported_cost_is_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.00042}}
    provider = make_provider([FakeResponse(body=body)], monkeypatch)
    assert provider.complete([{"role": "user", "content": "x"}]).reported_cost == 0.00042


@pytest.mark.parametrize(
    "finish, expected",
    [("stop", "end_turn"), ("tool_calls", "tool_use"), ("length", "max_tokens"),
     ("content_filter", "refusal"), (None, "end_turn")],
)
def test_stop_reasons_are_normalised(finish, expected) -> None:
    assert AssistantMessage(finish_reason=finish).stop_reason == expected


def test_registry_builds_the_right_client(workspace) -> None:
    from scode.config import load_settings
    from scode.providers.registry import build_provider

    settings = load_settings(workspace, {"provider": "openrouter", "api_key": "sk-or-x", "effort": "high"})
    client = build_provider(settings)
    assert client.base_url == "https://openrouter.ai/api/v1"
    assert client.headers["X-Title"] == "scode"
    assert client.echo_reasoning_details is True
    assert client.effort_style == "openrouter"

    openai = build_provider(load_settings(workspace, {"provider": "openai", "api_key": "sk-x"}))
    assert openai.max_tokens_param == "max_completion_tokens"
    assert openai.supports_temperature is False
