"""OpenAI-compatible chat client (NVIDIA NIM and anything that speaks the same API)."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import requests

from ..errors import AuthError, Interrupted, ProviderError
from .base import (
    AssistantMessage,
    ReasoningDelta,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallStarted,
)

CONNECT_TIMEOUT = 20
READ_TIMEOUT = 300
MAX_RETRIES = 4
RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


def _sleep_for(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(30.0, float(retry_after))
        except ValueError:
            pass
    return min(20.0, (2**attempt) + random.random())


@dataclass
class _ToolCallBuffer:
    """Accumulates one tool call across streaming deltas."""

    id: str = ""
    name: str = ""
    arguments: str = ""
    announced: bool = False


@dataclass
class OpenAICompatibleProvider:
    name: str
    api_key: str
    base_url: str
    model: str
    max_output_tokens: int = 32_000
    temperature: float = 0.6
    extra_body: dict[str, Any] = field(default_factory=dict)
    session: requests.Session = field(default_factory=requests.Session)
    # Flipped to False once the endpoint rejects function calling.
    supports_tools: bool = True

    # ------------------------------------------------------------ internals
    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _headers(self, *, stream: bool) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": "scode-cli",
        }

    def _payload(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_output_tokens: int | None,
        temperature: float | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_output_tokens or self.max_output_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": stream,
        }
        if tools and self.supports_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.extra_body)
        return payload

    def _post(self, payload: dict[str, Any], *, stream: bool) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=self._headers(stream=stream),
                    json=payload,
                    stream=stream,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                )
            except requests.Timeout as exc:
                last_error = exc
            except requests.RequestException as exc:
                raise ProviderError(f"Could not reach {self.base_url}: {exc}") from exc
            else:
                if response.status_code in RETRY_STATUSES and attempt < MAX_RETRIES - 1:
                    delay = _sleep_for(attempt, response.headers.get("Retry-After"))
                    response.close()
                    time.sleep(delay)
                    continue
                if response.status_code >= 400:
                    self._raise_for_status(response, payload)
                return response

            if attempt < MAX_RETRIES - 1:
                time.sleep(_sleep_for(attempt, None))

        raise ProviderError(f"Request to {self.base_url} timed out: {last_error}")

    def _raise_for_status(self, response: requests.Response, payload: dict[str, Any]) -> None:
        detail = self._error_detail(response)
        status = response.status_code
        response.close()

        if status in (401, 403):
            raise AuthError(
                f"Provider rejected the API key ({status}). {detail}\n"
                "Check NVIDIA_API_KEY, or run /login to set a new one."
            )
        if status == 404:
            raise ProviderError(
                f"Model {payload.get('model')!r} was not found at {self.base_url} ({detail}). "
                "Run /model to pick an available one."
            )
        if status == 400 and "tool" in detail.lower() and payload.get("tools"):
            raise _ToolsUnsupported(detail)
        if status == 429:
            raise ProviderError(f"Rate limited by the provider. {detail}")
        raise ProviderError(f"Provider error {status}: {detail}")

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return (response.text or "").strip()[:500]
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            if isinstance(error, str):
                return error
            for key in ("detail", "message", "title"):
                if key in body:
                    return str(body[key])
        return json.dumps(body)[:500]

    # ---------------------------------------------------------- public API
    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[StreamEvent]:
        try:
            yield from self._stream_once(
                messages,
                tools=tools,
                model=model,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        except _ToolsUnsupported:
            # The endpoint does not do function calling; the agent loop will
            # retry through the text protocol.
            self.supports_tools = False
            raise ToolsNotSupported(
                f"{model or self.model} does not support native tool calling"
            ) from None

    def _stream_once(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_output_tokens: int | None,
        temperature: float | None,
    ) -> Iterator[StreamEvent]:
        payload = self._payload(
            messages,
            tools=tools,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stream=True,
        )
        started = time.monotonic()
        response = self._post(payload, stream=True)

        content: list[str] = []
        reasoning: list[str] = []
        buffers: dict[int, _ToolCallBuffer] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] = {}

        try:
            for line in response.iter_lines(decode_unicode=False):
                if line is None:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]

                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                delta = choice.get("delta") or {}
                text = delta.get("content")
                if isinstance(text, list):  # some gateways send content parts
                    text = "".join(
                        part.get("text", "") for part in text if isinstance(part, dict)
                    )
                if text:
                    content.append(text)
                    yield TextDelta(text)

                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(thought, str) and thought:
                    reasoning.append(thought)
                    yield ReasoningDelta(thought)

                for raw_call in delta.get("tool_calls") or []:
                    index = raw_call.get("index", 0)
                    buffer = buffers.setdefault(index, _ToolCallBuffer())
                    if raw_call.get("id"):
                        buffer.id = raw_call["id"]
                    function = raw_call.get("function") or {}
                    if function.get("name"):
                        buffer.name += function["name"]
                    if function.get("arguments"):
                        buffer.arguments += function["arguments"]
                    if buffer.name and not buffer.announced:
                        buffer.announced = True
                        yield ToolCallStarted(buffer.name)
        except requests.RequestException as exc:
            raise ProviderError(f"Stream interrupted: {exc}") from exc
        except KeyboardInterrupt:
            raise Interrupted("Interrupted during streaming") from None
        finally:
            response.close()

        message = AssistantMessage(
            content="".join(content),
            reasoning="".join(reasoning),
            tool_calls=[
                ToolCall.from_raw(
                    buffer.id or f"call_{index}",
                    buffer.name,
                    buffer.arguments,
                )
                for index, buffer in sorted(buffers.items())
                if buffer.name
            ],
            finish_reason=finish_reason,
        )
        _apply_usage(message, usage, messages, started)
        yield StreamDone(message)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AssistantMessage:
        payload = self._payload(
            messages,
            tools=tools,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stream=False,
        )
        started = time.monotonic()
        try:
            response = self._post(payload, stream=False)
        except _ToolsUnsupported:
            self.supports_tools = False
            raise ToolsNotSupported(
                f"{model or self.model} does not support native tool calling"
            ) from None

        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError("Provider returned a non-JSON response") from exc
        finally:
            response.close()

        choices = body.get("choices") or []
        if not choices:
            raise ProviderError("Provider returned no choices")
        raw_message = choices[0].get("message") or {}

        content = raw_message.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))

        tool_calls = []
        for index, raw_call in enumerate(raw_message.get("tool_calls") or []):
            function = raw_call.get("function") or {}
            tool_calls.append(
                ToolCall.from_raw(
                    raw_call.get("id") or f"call_{index}",
                    function.get("name", ""),
                    function.get("arguments", ""),
                )
            )

        message = AssistantMessage(
            content=content,
            reasoning=raw_message.get("reasoning_content") or "",
            tool_calls=[c for c in tool_calls if c.name],
            finish_reason=choices[0].get("finish_reason"),
        )
        _apply_usage(message, body.get("usage") or {}, messages, started)
        return message

    # -------------------------------------------------------------- catalog
    def list_models(self) -> list[str]:
        url = f"{self.base_url.rstrip('/')}/models"
        try:
            response = self.session.get(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=(CONNECT_TIMEOUT, 30),
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ProviderError(f"Could not list models from {url}: {exc}") from exc
        return sorted(
            str(entry.get("id"))
            for entry in body.get("data", [])
            if isinstance(entry, dict) and entry.get("id")
        )


def _apply_usage(
    message: AssistantMessage,
    usage: dict[str, Any],
    messages: list[dict[str, Any]],
    started: float,
) -> None:
    from ..usage import estimate_tokens

    message.input_tokens = int(usage.get("prompt_tokens") or 0)
    message.output_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        message.cached_tokens = int(details.get("cached_tokens") or 0)

    if not message.input_tokens:
        message.input_tokens = sum(
            estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in messages
        )
    if not message.output_tokens:
        message.output_tokens = estimate_tokens(message.content) + estimate_tokens(
            "".join(c.raw_arguments for c in message.tool_calls)
        )
    message.__dict__["_elapsed"] = time.monotonic() - started


class ToolsNotSupported(ProviderError):
    """Raised when the endpoint cannot do native function calling."""


class _ToolsUnsupported(Exception):
    """Internal signal from the HTTP layer."""
