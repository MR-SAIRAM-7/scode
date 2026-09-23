"""Native Anthropic adapter: wire conversion, request parameters, and the real
SDK streaming against a local server that speaks the Messages SSE protocol."""

from __future__ import annotations

from pathlib import Path

import pytest
from anthropic_mock import AnthropicMock

from scode.errors import AuthError, ContextOverflow
from scode.providers.anthropic_native import (
    FALLBACK_BETA,
    AnthropicProvider,
    _live_blocks,
    supports_adaptive,
)
from scode.providers.base import ReasoningDelta, StreamDone, TextDelta, ToolCallStarted


class _NoNetwork:
    """Stands in for the SDK client where a test only inspects parameters."""


def offline(model: str = "claude-opus-5", **kwargs) -> AnthropicProvider:
    return AnthropicProvider(api_key="sk-ant-test", model=model, client=_NoNetwork(), **kwargs)


@pytest.fixture
def mock():
    server = AnthropicMock()
    yield server
    server.close()


def live(mock: AnthropicMock, model: str = "claude-opus-5", **kwargs) -> AnthropicProvider:
    return AnthropicProvider(api_key="sk-ant-test", model=model, base_url=mock.url, **kwargs)


# ------------------------------------------------------------- conversion

def test_system_prompt_becomes_a_cached_block() -> None:
    params = offline()._params([{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}], None, "claude-opus-5", None)
    assert params["system"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    # Automatic caching of the conversation tail, on top of the system breakpoint.
    assert params["cache_control"] == {"type": "ephemeral"}


def test_parallel_tool_results_share_one_user_message() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "LS", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "Read", "arguments": '{"file_path": "x"}'}},
        ]},
        {"role": "tool", "tool_call_id": "a", "name": "LS", "content": "x"},
        {"role": "tool", "tool_call_id": "b", "name": "Read", "content": "boom", "is_error": True},
        {"role": "user", "content": "<system-reminder>\nnote\n</system-reminder>"},
    ]
    _, converted = offline()._convert(messages)
    assert [m["role"] for m in converted] == ["user", "assistant", "user"]
    last = converted[2]["content"]
    assert [b["type"] for b in last] == ["tool_result", "tool_result", "text"]
    assert last[1] == {"type": "tool_result", "tool_use_id": "b", "content": "boom", "is_error": True}
    assert converted[1]["content"][1]["input"] == {"file_path": "x"}


def test_assistant_turns_replay_verbatim_from_provider_state() -> None:
    saved = [
        {"type": "thinking", "thinking": "hmm", "signature": "SIG"},
        {"type": "tool_use", "id": "toolu_1", "name": "LS", "input": {}},
    ]
    message = {"role": "assistant", "content": "", "tool_calls": [],
               "provider_state": {"anthropic": {"content": saved}}}
    _, converted = offline()._convert([{"role": "user", "content": "x"}, message])
    assert converted[1]["content"] == saved


def test_strip_thinking_drops_only_thinking() -> None:
    provider = offline()
    provider._flags.strip_thinking = True
    message = {"role": "assistant", "provider_state": {"anthropic": {"content": [
        {"type": "thinking", "thinking": "t", "signature": "s"},
        {"type": "text", "text": "answer"},
    ]}}}
    _, converted = provider._convert([{"role": "user", "content": "x"}, message])
    assert converted[1]["content"] == [{"type": "text", "text": "answer"}]


def test_foreign_tool_ids_are_sanitised_consistently() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call:9/x", "type": "function", "function": {"name": "LS", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call:9/x", "name": "LS", "content": "ok"},
    ]
    _, converted = offline()._convert(messages)
    assert converted[1]["content"][0]["id"] == converted[2]["content"][0]["tool_use_id"] == "call_9_x"


def test_images_become_base64_blocks() -> None:
    _, converted = offline()._convert([{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image", "media_type": "image/png", "data": "QUJD"},
    ]}])
    assert converted[0]["content"][1] == {
        "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}


# ------------------------------------------------------------- parameters

def test_current_models_get_adaptive_thinking_and_effort() -> None:
    params = offline(effort="high")._params([{"role": "user", "content": "x"}], None, "claude-opus-5", None)
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["output_config"] == {"effort": "high"}
    assert "temperature" not in params


def test_legacy_models_get_neither() -> None:
    params = offline(effort="high")._params([{"role": "user", "content": "x"}], None, "claude-haiku-4-5", None)
    assert "thinking" not in params and "output_config" not in params


@pytest.mark.parametrize("model, adaptive", [
    ("claude-opus-5", True), ("claude-sonnet-5", True), ("claude-fable-5-1", True), ("claude-opus-4-8", True),
    ("claude-haiku-4-5", False), ("claude-sonnet-4-5", False), ("claude-3-7-sonnet-latest", False),
])
def test_supports_adaptive(model: str, adaptive: bool) -> None:
    assert supports_adaptive(model) is adaptive


def test_eager_tool_streaming_only_on_the_real_api() -> None:
    tools = [{"type": "function", "function": {"name": "LS", "description": "list", "parameters": {"type": "object"}}}]
    official = offline()._params([{"role": "user", "content": "x"}], tools, "claude-opus-5", None)
    assert official["tools"][0]["eager_input_streaming"] is True
    proxied = AnthropicProvider(api_key="k", model="claude-opus-5", base_url="https://proxy.example.com", client=_NoNetwork())
    assert "eager_input_streaming" not in proxied._params([{"role": "user", "content": "x"}], tools, "claude-opus-5", None)["tools"][0]


def test_output_cap_applies() -> None:
    provider = offline(max_output_tokens=64_000, output_limits=lambda m: 8_000)
    assert provider._params([{"role": "user", "content": "x"}], None, "claude-opus-5", None)["max_tokens"] == 8_000


def test_refusal_fallbacks_use_the_beta_endpoint() -> None:
    calls = {}

    class Recorder:
        class beta:
            class messages:
                @staticmethod
                def stream(**kwargs):
                    calls["beta"] = kwargs
                    return None

        class messages:
            @staticmethod
            def stream(**kwargs):
                calls["plain"] = kwargs
                return None

    provider = AnthropicProvider(api_key="k", model="claude-opus-5", client=Recorder())
    provider._open_stream({"model": "claude-opus-5"})
    assert calls["beta"]["betas"] == [FALLBACK_BETA]
    assert calls["beta"]["extra_body"] == {"fallbacks": "default"}
    provider._open_stream({"model": "claude-sonnet-5"})  # not a documented fallback model
    assert "plain" in calls


def test_fallback_echo_rule() -> None:
    blocks = [
        {"type": "thinking", "thinking": "a", "signature": "s"},
        {"type": "text", "text": "partial answer"},
        {"type": "tool_use", "id": "t0", "name": "Bash", "input": {}},
        {"type": "fallback", "from": {"model": "claude-opus-5"}, "to": {"model": "claude-opus-4-8"}},
        {"type": "text", "text": "continued"},
        {"type": "tool_use", "id": "t1", "name": "LS", "input": {}},
    ]
    replay, live_part = _live_blocks(blocks)
    assert [b["type"] for b in replay] == ["text", "fallback", "text", "tool_use"]
    assert [b.get("id") for b in live_part if b["type"] == "tool_use"] == ["t1"]


def test_adapt_switches_off_what_a_400_names() -> None:
    provider = offline()
    assert provider._adapt("messages.3.content.0: Invalid `signature` in `thinking` block") is True
    assert provider._flags.strip_thinking is True
    assert provider._adapt("eager_input_streaming: Extra inputs are not permitted") is True
    assert provider._flags.eager is False
    assert provider._adapt("something unrelated") is False


# ------------------------------------------------------- the real SDK, live

def test_streams_text_thinking_and_a_tool_call(mock: AnthropicMock) -> None:
    mock.turns.append({
        "blocks": [
            {"type": "thinking", "thinking": "I should read it.", "signature": "SIG-1"},
            {"type": "text", "text": "Reading the file."},
            {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"file_path": "app.py", "offset": 1}},
        ],
        "usage": {"input": 120, "cache_read": 900, "cache_write": 40, "output": 55},
    })
    events = list(live(mock).stream([{"role": "system", "content": "S"}, {"role": "user", "content": "read app.py"}],
                                    tools=[{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}]))

    assert any(isinstance(e, ReasoningDelta) and "read it" in e.text for e in events)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Reading the file."
    assert any(isinstance(e, ToolCallStarted) and e.name == "Read" for e in events)

    message = events[-1].message
    assert isinstance(events[-1], StreamDone)
    assert message.tool_calls[0].arguments == {"file_path": "app.py", "offset": 1}
    assert message.stop_reason == "tool_use"
    assert (message.input_tokens, message.cached_tokens, message.cache_write_tokens, message.output_tokens) == (120, 900, 40, 55)
    saved = message.provider_state["anthropic"]["content"]
    assert saved[0] == {"type": "thinking", "thinking": "I should read it.", "signature": "SIG-1"}

    sent = mock.requests[0]
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["stream"] is True
    assert mock.headers[0]["x-api-key"] == "sk-ant-test"


def test_the_next_request_replays_thinking_and_groups_results(mock: AnthropicMock) -> None:
    mock.turns += [
        {"blocks": [{"type": "thinking", "thinking": "plan", "signature": "SIG-A"},
                    {"type": "tool_use", "id": "toolu_a", "name": "LS", "input": {}},
                    {"type": "tool_use", "id": "toolu_b", "name": "LS", "input": {"path": "src"}}]},
        {"blocks": [{"type": "text", "text": "Two directories."}]},
    ]
    provider = live(mock)
    history = [{"role": "system", "content": "S"}, {"role": "user", "content": "look around"}]
    first = list(provider.stream(history))[-1].message
    history.append(first.to_message())
    history.append({"role": "tool", "tool_call_id": "toolu_a", "name": "LS", "content": "a/"})
    history.append({"role": "tool", "tool_call_id": "toolu_b", "name": "LS", "content": "b/"})
    list(provider.stream(history))

    second = mock.requests[1]["messages"]
    assert second[1]["content"][0] == {"type": "thinking", "thinking": "plan", "signature": "SIG-A"}
    assert [b["type"] for b in second[2]["content"]] == ["tool_result", "tool_result"]
    # The earlier turns are byte-identical to what was sent before.
    assert second[0] == mock.requests[0]["messages"][0]


def test_signature_error_strips_thinking_and_retries(mock: AnthropicMock) -> None:
    mock.turns += [
        {"error": (400, "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation.")},
        {"blocks": [{"type": "text", "text": "recovered"}]},
    ]
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "provider_state": {"anthropic": {"content": [
            {"type": "thinking", "thinking": "t", "signature": "old"}, {"type": "text", "text": "prior"}]}}},
        {"role": "user", "content": "continue"},
    ]
    message = list(live(mock).stream(history))[-1].message
    assert message.content == "recovered"
    retried = mock.requests[1]["messages"][1]["content"]
    assert retried == [{"type": "text", "text": "prior"}]


def test_prompt_too_long_is_a_context_overflow(mock: AnthropicMock) -> None:
    mock.turns.append({"error": (400, "prompt is too long: 1200000 tokens > 1000000 maximum")})
    with pytest.raises(ContextOverflow):
        list(live(mock).stream([{"role": "user", "content": "x"}]))


def test_bad_key_is_an_auth_error(mock: AnthropicMock) -> None:
    mock.turns.append({"error": (401, "invalid x-api-key")})
    with pytest.raises(AuthError, match="ANTHROPIC_API_KEY"):
        list(live(mock).stream([{"role": "user", "content": "x"}]))


def test_list_and_check_models(mock: AnthropicMock) -> None:
    provider = live(mock)
    assert provider.list_models() == ["claude-opus-5"]
    assert provider.check("claude-nope")[0] == "unavailable"


def test_a_full_agent_turn_through_the_native_adapter(mock: AnthropicMock, workspace: Path) -> None:
    from scode.agent.loop import Agent
    from scode.config import load_settings
    from scode.permissions import PermissionEngine
    from scode.tools import ToolContext, build_registry
    from scode.ui.console import UI

    mock.turns += [
        {"blocks": [{"type": "thinking", "thinking": "look first", "signature": "S1"},
                    {"type": "tool_use", "id": "toolu_r", "name": "Read", "input": {"file_path": "app.py"}}]},
        {"blocks": [{"type": "text", "text": "It defines add(a, b)."}]},
    ]
    settings = load_settings(workspace, {"provider": "anthropic", "api_key": "sk-ant-test",
                                         "base_url": mock.url, "permission_mode": "bypassPermissions"})
    provider = AnthropicProvider(api_key="sk-ant-test", model=settings.model, base_url=mock.url)
    agent = Agent(provider=provider, registry=build_registry(), ctx=ToolContext(
        workspace=workspace, settings=settings, permissions=PermissionEngine(settings)),
        ui=UI(quiet=True), settings=settings)

    result = agent.run("what does app.py do?")
    assert result.text == "It defines add(a, b)."
    second = mock.requests[1]
    tool_result = second["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and "def add" in tool_result["content"]
    assert second["system"] == mock.requests[0]["system"]  # frozen system prompt
