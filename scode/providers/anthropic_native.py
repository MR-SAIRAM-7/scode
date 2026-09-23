"""Native Anthropic Messages API adapter, built on the official `anthropic` SDK.

Why native instead of an OpenAI-compatible shim: prompt caching (an agent loop
resends the whole conversation every step, and cache reads cost ~10% of input),
adaptive thinking with replayable signatures, and server-side refusal
fallbacks.

Rules this adapter keeps, because breaking them costs a 400 or the cache:
- The system prompt and tool list must be byte-stable across a session; the
  agent freezes them, and this adapter renders them deterministically.
- Assistant turns are replayed exactly as the API returned them (thinking
  blocks included, signatures intact) from `provider_state`.
- All results for one round of tool calls go back in a single user message,
  tool_result blocks first.
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from ..errors import (
    AuthError,
    ContextOverflow,
    Interrupted,
    ProviderError,
    TransientProviderError,
    is_context_overflow,
    is_transient,
)
from .base import (
    AssistantMessage,
    ReasoningDelta,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallStarted,
    text_of,
)

DEFAULT_BASE_URL = "https://api.anthropic.com"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models documented to support `fallbacks: "default"`.
FALLBACK_MODELS = frozenset({"claude-opus-5", "claude-fable-5-1"})
# Models that predate adaptive thinking and the `effort` control.
_LEGACY = re.compile(r"claude-(?:3|instant|haiku|sonnet-4-5|opus-4-5|opus-4-1|opus-4-0|sonnet-4-0)|claude-(?:sonnet|opus)-4-2025")
_TOOL_ID = re.compile(r"[^a-zA-Z0-9_-]")
MAX_JSON_REISSUES = 2


def supports_adaptive(model: str) -> bool:
    return not _LEGACY.search(model.lower())


def _safe_id(tool_id: str) -> str:
    return _TOOL_ID.sub("_", tool_id or "") or "toolu_missing"


@dataclass
class _Flags:
    """Request features, switched off one at a time if an endpoint rejects them."""

    fallbacks: bool = True
    eager: bool = True
    auto_cache: bool = True
    effort: bool = True
    thinking: bool = True
    strip_thinking: bool = False


def _live_blocks(blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the fallback echo rule. Returns (replayable blocks, live blocks).

    After a mid-output refusal fallback, blocks before the last `fallback`
    marker belong to the declined attempt: its text stays as context, but its
    thinking and tool calls must not be replayed or executed.
    """
    boundary = -1
    for index, block in enumerate(blocks):
        if block.get("type") == "fallback":
            boundary = index
    if boundary < 0:
        return blocks, blocks
    # Only text survives from the declined attempt (plus the audit markers).
    replay = [b for b in blocks[:boundary] if b.get("type") in {"text", "fallback"}]
    replay += blocks[boundary:]
    return replay, blocks[boundary + 1:]


class AnthropicProvider:
    name = "anthropic"
    supports_tools = True

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        max_output_tokens: int = 64_000,
        effort: str = "",
        refusal_fallbacks: bool = True,
        key_hint: str = "Set ANTHROPIC_API_KEY, or run /login anthropic.",
        output_limits: Callable[[str], int | None] | None = None,
        client: Any = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ProviderError(
                "The native Anthropic provider needs the Anthropic SDK:\n"
                "  pip install anthropic   (or, from the scode checkout: pip install -e \".[anthropic]\")\n"
                "Or reach Claude through OpenRouter: --provider openrouter --model opus"
            ) from exc

        self._sdk = anthropic
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.effort = effort
        self.key_hint = key_hint
        self.output_limits = output_limits
        official = self.base_url == DEFAULT_BASE_URL
        # Betas and eager tool streaming are Claude API features; proxies and
        # gateways in front of a custom base URL commonly reject them.
        self._flags = _Flags(fallbacks=refusal_fallbacks and official, eager=official)
        self.client = client or anthropic.Anthropic(
            api_key=api_key or None,
            base_url=self.base_url,
            max_retries=3,
            timeout=anthropic.Timeout(600.0, connect=20.0),
        )

    # ----------------------------------------------------------- conversion
    def _tools(self, schemas: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        tools = []
        for schema in schemas or []:
            function = schema.get("function") or {}
            tool = {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
            if self._flags.eager:
                tool["eager_input_streaming"] = True
            tools.append(tool)
        return tools

    @staticmethod
    def _user_blocks(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        blocks: list[dict[str, Any]] = []
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image":
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.get("media_type", "image/png"),
                            "data": part.get("data", ""),
                        },
                    }
                )
            elif part.get("text"):
                blocks.append({"type": "text", "text": str(part["text"])})
        return blocks

    def _assistant_blocks(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        state = (message.get("provider_state") or {}).get("anthropic") or {}
        saved = state.get("content")
        if isinstance(saved, list) and saved:
            blocks = copy.deepcopy(saved)
            if self._flags.strip_thinking:
                blocks = [b for b in blocks if b.get("type") not in {"thinking", "redacted_thinking"}]
            for block in blocks:
                if block.get("type") == "tool_use":
                    block["id"] = _safe_id(block.get("id", ""))
            if blocks:
                return blocks

        blocks = []
        text = text_of(message.get("content"))
        if text:
            blocks.append({"type": "text", "text": text})
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": _safe_id(call.get("id", "")),
                    "name": function.get("name", ""),
                    "input": arguments if isinstance(arguments, dict) else {},
                }
            )
        return blocks or [{"type": "text", "text": "(no content)"}]

    def _convert(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system = ""
        out: list[dict[str, Any]] = []

        def push(role: str, blocks: list[dict[str, Any]]) -> None:
            if not blocks:
                return
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})

        for message in messages:
            role = message.get("role")
            if role == "system":
                text = text_of(message.get("content"))
                if not out and not system:
                    system = text
                elif text:
                    push("user", [{"type": "text", "text": f"<system-reminder>\n{text}\n</system-reminder>"}])
            elif role == "user":
                push("user", self._user_blocks(message.get("content")))
            elif role == "assistant":
                push("assistant", self._assistant_blocks(message))
            elif role == "tool":
                result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": _safe_id(message.get("tool_call_id", "")),
                    "content": text_of(message.get("content")) or "(no output)",
                }
                if message.get("is_error"):
                    result["is_error"] = True
                push("user", [result])

        # Tool results must lead their user message; reminder text follows.
        for entry in out:
            if entry["role"] == "user":
                results = [b for b in entry["content"] if b.get("type") == "tool_result"]
                if results:
                    rest = [b for b in entry["content"] if b.get("type") != "tool_result"]
                    entry["content"] = results + rest
        return system, out

    # -------------------------------------------------------------- request
    def _params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        max_output_tokens: int | None,
    ) -> dict[str, Any]:
        system, converted = self._convert(messages)
        limit = max_output_tokens or self.max_output_tokens
        cap = self.output_limits(model) if self.output_limits else None
        if cap:
            limit = min(limit, cap)

        params: dict[str, Any] = {"model": model, "max_tokens": limit, "messages": converted}
        if system:
            # Explicit breakpoint on the frozen system prompt (tools render
            # before it, so this caches tools + system together) ...
            params["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if self._flags.auto_cache:
            # ... plus automatic caching of the growing conversation tail.
            params["cache_control"] = {"type": "ephemeral"}
        converted_tools = self._tools(tools)
        if converted_tools:
            params["tools"] = converted_tools
        if supports_adaptive(model):
            if self._flags.thinking:
                params["thinking"] = {"type": "adaptive", "display": "summarized"}
            if self.effort and self._flags.effort:
                params["output_config"] = {"effort": self.effort}
        return params

    def _open_stream(self, params: dict[str, Any]):
        if self._flags.fallbacks and params["model"] in FALLBACK_MODELS:
            return self.client.beta.messages.stream(
                **params, betas=[FALLBACK_BETA], extra_body={"fallbacks": "default"}
            )
        return self.client.messages.stream(**params)

    def _adapt(self, message: str) -> bool:
        """Switch off whatever a 400 complains about. True if a retry may help."""
        text = message.lower()
        if ("signature" in text or "bound to a different conversation" in text) and not self._flags.strip_thinking:
            # History no longer matches the thinking it carries. Recovery per
            # the API docs: drop every thinking block and retry once.
            self._flags.strip_thinking = True
            return True
        for flag, markers in (
            ("fallbacks", ("fallback",)),
            ("eager", ("eager_input_streaming",)),
            ("auto_cache", ("cache_control",)),
            ("effort", ("output_config", "effort")),
            ("thinking", ("thinking",)),
        ):
            if getattr(self._flags, flag) and any(m in text for m in markers):
                setattr(self._flags, flag, False)
                return True
        return False

    def _translate(self, exc: Exception, model: str) -> Exception:
        sdk = self._sdk
        detail = getattr(exc, "message", None) or str(exc)
        if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
            return AuthError(f"Anthropic rejected the API key. {detail}\n{self.key_hint}")
        if isinstance(exc, sdk.NotFoundError):
            return ProviderError(f"{model} was not found on the Anthropic API. {detail}\nRun /model to pick one.")
        if isinstance(exc, sdk.RateLimitError):
            return TransientProviderError(f"Rate limited by Anthropic. {detail}")
        if isinstance(exc, sdk.BadRequestError) and is_context_overflow(detail):
            return ContextOverflow(f"The conversation is too long for {model}. {detail}")
        if isinstance(exc, sdk.APIStatusError):
            status = getattr(exc, "status_code", "?")
            if status == 529 or isinstance(exc, getattr(sdk, "OverloadedError", ())):
                return TransientProviderError("Anthropic is overloaded right now.")
            # Error events inside a stream arrive with the stream's own status.
            if status in (500, 502, 503, 504) or (status not in (400, 401, 403, 404, 413, 422)
                                                  and is_transient(detail)):
                return TransientProviderError(f"Anthropic API error {status}: {detail}")
            return ProviderError(f"Anthropic API error {status}: {detail}")
        if isinstance(exc, sdk.APIConnectionError):
            return ProviderError(f"Could not reach {self.base_url}: {detail}")
        return ProviderError(f"Anthropic request failed: {detail}")

    # ----------------------------------------------------------- public API
    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,  # current Claude models reject sampling params
    ) -> Iterator[StreamEvent]:
        model_id = model or self.model
        started = time.monotonic()
        json_reissues = 0
        sdk = self._sdk

        while True:
            params = self._params(messages, tools, model_id, max_output_tokens)
            emitted = False
            try:
                with self._open_stream(params) as stream:
                    for event in stream:
                        kind = getattr(event, "type", "")
                        if kind == "content_block_start":
                            block = event.content_block
                            if getattr(block, "type", "") == "tool_use":
                                emitted = True
                                yield ToolCallStarted(block.name)
                        elif kind == "content_block_delta":
                            delta = event.delta
                            delta_type = getattr(delta, "type", "")
                            if delta_type == "text_delta" and delta.text:
                                emitted = True
                                yield TextDelta(delta.text)
                            elif delta_type == "thinking_delta" and getattr(delta, "thinking", ""):
                                emitted = True
                                yield ReasoningDelta(delta.thinking)
                    final = stream.get_final_message()
            except KeyboardInterrupt:
                raise Interrupted("Interrupted during streaming") from None
            except sdk.BadRequestError as exc:
                if not emitted and self._adapt(getattr(exc, "message", str(exc))):
                    continue
                raise self._translate(exc, model_id) from exc
            except sdk.APIError as exc:
                raise self._translate(exc, model_id) from exc
            except ValueError:
                # Streamed tool input the SDK could not parse at all. It raised
                # before the block completed, so there is no id to answer:
                # re-issue the turn, a bounded number of times.
                json_reissues += 1
                if json_reissues > MAX_JSON_REISSUES:
                    raise ProviderError("The model kept producing unparseable tool input.") from None
                continue
            break

        yield StreamDone(self._to_message(final, model_id, started))

    def _to_message(self, final: Any, model: str, started: float) -> AssistantMessage:
        blocks = [
            block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else dict(block)
            for block in (final.content or [])
        ]
        replay, live = _live_blocks(blocks)

        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        reasoning = "".join(b.get("thinking", "") for b in live if b.get("type") == "thinking")
        calls = [
            ToolCall.from_input(b.get("id", ""), b.get("name", ""), b.get("input"))
            for b in live
            if b.get("type") == "tool_use"
        ]

        usage = final.usage
        message = AssistantMessage(
            content=text,
            reasoning=reasoning,
            tool_calls=calls,
            finish_reason=getattr(final, "stop_reason", None),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cached_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            model=str(getattr(final, "model", "") or model),
            provider_state={"anthropic": {"content": replay, "model": model}},
        )
        if message.finish_reason == "refusal":
            details = getattr(final, "stop_details", None)
            message.refusal_category = getattr(details, "category", None) if details else None
        message.__dict__["_elapsed"] = time.monotonic() - started
        return message

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AssistantMessage:
        result: AssistantMessage | None = None
        for event in self.stream(
            messages, tools=tools, model=model, max_output_tokens=max_output_tokens
        ):
            if isinstance(event, StreamDone):
                result = event.message
        if result is None:
            raise ProviderError("The Anthropic stream ended without a message")
        return result

    def list_models(self) -> list[str]:
        try:
            return sorted(model.id for model in self.client.models.list())
        except self._sdk.APIError as exc:
            raise self._translate(exc, "") from exc

    def check(self, model: str, *, timeout: float = 25.0) -> tuple[str, str]:
        started = time.monotonic()
        sdk = self._sdk
        try:
            self.client.with_options(timeout=timeout).models.retrieve(model)
        except sdk.NotFoundError:
            return "unavailable", "not found"
        except (sdk.AuthenticationError, sdk.PermissionDeniedError):
            return "error", "key rejected"
        except sdk.APITimeoutError:
            return "timeout", f"no response in {timeout:.0f}s"
        except sdk.APIError as exc:
            return "error", str(getattr(exc, "message", exc))[:80]
        return "ok", f"{time.monotonic() - started:.1f}s"
