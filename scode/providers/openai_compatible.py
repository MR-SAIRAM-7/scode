"""OpenAI-compatible chat client.

Serves every provider that speaks `/chat/completions`: NVIDIA NIM, OpenRouter,
OmniRoute, OpenAI, Gemini's compatibility layer, DeepSeek, Groq, local
servers, and anything defined in settings.

Providers disagree on details (whether `max_tokens` is accepted, whether a
temperature is allowed, how reasoning effort is spelled). Rather than encode
every quirk, the client adapts: a 400 that names a rejected parameter is fixed
up and retried once, and the fix is remembered for the rest of the session.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import requests

from ..errors import AuthError, ContextOverflow, Interrupted, ProviderError, is_context_overflow
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
from .profiles import EFFORT_NONE, EFFORT_OPENROUTER, EFFORT_REASONING_EFFORT

CONNECT_TIMEOUT = 20
# Applies to each socket read, so for a stream this is the budget for the first
# token. Working models answer well inside it; a model whose gateway is wedged
# never answers at all, so waiting longer only delays the error.
READ_TIMEOUT = 180
MAX_RETRIES = 4
RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
# A wedged upstream returns these only after the full read timeout, so retrying
# three more times would hang for many minutes. One retry is enough.
SLOW_FAILURE_STATUSES = {502, 503, 504}
MAX_SLOW_FAILURE_ATTEMPTS = 2
MAX_ADAPTATIONS = 4

_UNSUPPORTED = re.compile(
    r"unsupported|not supported|does not support|isn't supported|is not allowed|"
    r"not permitted|unrecognized|unknown (?:field|parameter|argument|name)|"
    r"extra (?:inputs|fields)|only the default|invalid (?:parameter|argument|field)|"
    r"not available for this model"
)
_TOKEN_CAP_CONTEXT = re.compile(r"max_(?:completion_)?tokens|output tokens|completion tokens")


def _unavailable_note(model: str) -> str:
    from ..constants import KNOWN_UNAVAILABLE

    return KNOWN_UNAVAILABLE.get(model, "")


def _sleep_for(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(30.0, float(retry_after))
        except ValueError:
            pass
    return min(20.0, (2**attempt) + random.random())


def _token_cap_from_error(text: str, current: int) -> int | None:
    """Pull a smaller output limit out of an error like "max_tokens must be <= 16384"."""
    if not _TOKEN_CAP_CONTEXT.search(text):
        return None
    candidates = [int(n) for n in re.findall(r"\d{3,7}", text)]
    below = [n for n in candidates if 256 <= n < current]
    return max(below) if below else None


@dataclass
class _Adaptations:
    drop_temperature: bool = False
    drop_effort: bool = False
    drop_stream_options: bool = False
    max_tokens_param: str | None = None
    max_tokens_cap: int | None = None


@dataclass
class _ToolCallBuffer:
    """Accumulates one tool call across streaming deltas."""

    id: str = ""
    name: str = ""
    arguments: str = ""
    announced: bool = False


def _merge_reasoning_details(target: dict[int, dict[str, Any]], items: list[Any]) -> str:
    """Merge streamed reasoning_details fragments. Returns displayable text."""
    shown: list[str] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        index = item.get("index", position)
        if not isinstance(index, int):
            index = position
        slot = target.setdefault(index, {})
        for key, value in item.items():
            if value is None:
                continue
            if key in {"text", "summary", "data"} and isinstance(value, str):
                slot[key] = slot.get(key, "") + value
                if key in {"text", "summary"}:
                    shown.append(value)
            else:
                slot[key] = value
    return "".join(shown)


@dataclass
class OpenAICompatibleProvider:
    name: str
    api_key: str
    base_url: str
    model: str
    max_output_tokens: int = 32_000
    temperature: float | None = None
    headers: dict[str, str] = field(default_factory=dict)
    max_tokens_param: str = "max_tokens"
    effort_style: str = EFFORT_NONE
    effort: str = ""
    supports_temperature: bool = True
    echo_reasoning_details: bool = False
    key_hint: str = "Check your API key, or run /login."
    extra_body: dict[str, Any] = field(default_factory=dict)
    session: requests.Session = field(default_factory=requests.Session)
    # Flipped to False once the endpoint rejects function calling.
    supports_tools: bool = True
    # Per-model output caps from the catalog, looked up by the registry.
    output_limits: Callable[[str], int | None] | None = None
    _adaptations: dict[str, _Adaptations] = field(default_factory=dict)

    # ------------------------------------------------------------ internals
    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _headers(self, *, stream: bool) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": "scode-cli",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.headers)
        return headers

    def _adapt_state(self, model: str) -> _Adaptations:
        return self._adaptations.setdefault(model, _Adaptations())

    def _to_wire(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Canonical messages -> OpenAI chat messages. Unknown keys are dropped."""
        out: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                out.append({"role": "system", "content": text_of(content)})
            elif role == "user":
                if isinstance(content, list):
                    content = [self._part(p) for p in content if isinstance(p, dict)]
                out.append({"role": "user", "content": content if content is not None else ""})
            elif role == "assistant":
                wire: dict[str, Any] = {"role": "assistant", "content": text_of(content)}
                if message.get("tool_calls"):
                    wire["tool_calls"] = message["tool_calls"]
                if self.echo_reasoning_details:
                    state = (message.get("provider_state") or {}).get("openai") or {}
                    if state.get("reasoning_details"):
                        wire["reasoning_details"] = state["reasoning_details"]
                out.append(wire)
            elif role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get("tool_call_id", ""),
                        "name": message.get("name", ""),
                        "content": text_of(content),
                    }
                )
        return out

    @staticmethod
    def _part(part: dict[str, Any]) -> dict[str, Any]:
        if part.get("type") == "image":
            url = f"data:{part.get('media_type', 'image/png')};base64,{part.get('data', '')}"
            return {"type": "image_url", "image_url": {"url": url}}
        return {"type": "text", "text": str(part.get("text", ""))}

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
        model_id = model or self.model
        adapt = self._adapt_state(model_id)

        limit = max_output_tokens or self.max_output_tokens
        catalog_cap = self.output_limits(model_id) if self.output_limits else None
        if catalog_cap:
            limit = min(limit, catalog_cap)
        if adapt.max_tokens_cap:
            limit = min(limit, adapt.max_tokens_cap)

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": self._to_wire(messages),
            "stream": stream,
            (adapt.max_tokens_param or self.max_tokens_param): limit,
        }

        temp = self.temperature if temperature is None else temperature
        if temp is not None and self.supports_temperature and not adapt.drop_temperature:
            payload["temperature"] = temp

        if self.effort and not adapt.drop_effort:
            if self.effort_style == EFFORT_REASONING_EFFORT:
                payload["reasoning_effort"] = self.effort
            elif self.effort_style == EFFORT_OPENROUTER:
                payload["reasoning"] = {"effort": self.effort}

        if tools and self.supports_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream and not adapt.drop_stream_options:
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.extra_body)
        return payload

    def _adapt(self, detail: str, payload: dict[str, Any]) -> bool:
        """Learn from a 400 that names a parameter. Returns True if retrying helps."""
        text = detail.lower()
        adapt = self._adapt_state(str(payload.get("model", "")))
        unsupported = bool(_UNSUPPORTED.search(text))
        changed = False

        if "temperature" in text and "temperature" in payload and unsupported:
            adapt.drop_temperature = True
            changed = True

        if "max_completion_tokens" in text and "max_tokens" in payload:
            adapt.max_tokens_param = "max_completion_tokens"
            changed = True
        elif "max_completion_tokens" in payload and "max_completion_tokens" in text and unsupported:
            adapt.max_tokens_param = "max_tokens"
            changed = True

        current = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 0)
        cap = _token_cap_from_error(text, current) if current else None
        if cap:
            adapt.max_tokens_cap = cap
            changed = True

        effort_sent = "reasoning_effort" in payload or "reasoning" in payload
        if effort_sent and unsupported and ("reasoning" in text or "effort" in text):
            adapt.drop_effort = True
            changed = True

        if "stream_options" in text and "stream_options" in payload:
            adapt.drop_stream_options = True
            changed = True
        return changed

    def _request(self, build: Callable[[], dict[str, Any]], *, stream: bool) -> tuple[requests.Response, dict[str, Any]]:
        """POST with parameter adaptation. Returns the response and the payload sent."""
        for _ in range(MAX_ADAPTATIONS + 1):
            payload = build()
            try:
                return self._post(payload, stream=stream), payload
            except _BadRequest as exc:
                if self._adapt(exc.detail, payload):
                    continue
                raise ProviderError(f"Provider rejected the request (400): {exc.detail}") from None
        raise ProviderError("Provider kept rejecting the request parameters.")

    def _post(self, payload: dict[str, Any], *, stream: bool) -> requests.Response:
        last_error: Exception | None = None
        timeouts = 0
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
                timeouts += 1
                if timeouts >= MAX_SLOW_FAILURE_ATTEMPTS:
                    break
            except requests.RequestException as exc:
                raise ProviderError(f"Could not reach {self.base_url}: {exc}") from exc
            else:
                slow = response.status_code in SLOW_FAILURE_STATUSES
                budget = MAX_SLOW_FAILURE_ATTEMPTS if slow else MAX_RETRIES
                if response.status_code in RETRY_STATUSES and attempt < budget - 1:
                    delay = _sleep_for(attempt, response.headers.get("Retry-After"))
                    response.close()
                    time.sleep(delay)
                    continue
                if response.status_code >= 400:
                    self._raise_for_status(response, payload)
                return response

            if attempt < MAX_RETRIES - 1:
                time.sleep(_sleep_for(attempt, None))

        raise ProviderError(self._unreachable_message(payload, last_error))

    def _unreachable_message(self, payload: dict[str, Any], error: Exception | None) -> str:
        model = str(payload.get("model", ""))
        text = (
            f"{model} did not respond within {READ_TIMEOUT}s. "
            "The model is listed in the catalog but its backend is not answering."
        )
        note = _unavailable_note(model)
        if note:
            text = f"{model} is not usable: {note}."
        return text + "\nRun /model --check to see which models your key can reach."

    def _raise_for_status(self, response: requests.Response, payload: dict[str, Any]) -> None:
        detail = self._error_detail(response)
        status = response.status_code
        response.close()

        if status in (401, 403):
            raise AuthError(
                f"{self.name} rejected the API key ({status}). {detail}\n{self.key_hint}"
            )
        model = str(payload.get("model", ""))
        if status == 404:
            raise ProviderError(
                f"{model} is not available from {self.name}. {detail}\n"
                "Catalogs often list models a key cannot reach. "
                "Run /model --check to see yours."
            )
        if status == 410:
            raise ProviderError(
                f"{model} has been retired by {self.name}. {detail}\n"
                "Run /model --check to pick one that still works."
            )
        if status in SLOW_FAILURE_STATUSES:
            raise ProviderError(self._unreachable_message(payload, None))
        if status in (400, 422):
            lowered = detail.lower()
            if is_context_overflow(lowered):
                raise ContextOverflow(f"The conversation is too long for {model}. {detail}")
            if payload.get("tools") and "tool" in lowered and _UNSUPPORTED.search(lowered):
                raise _ToolsUnsupported(detail)
            raise _BadRequest(detail)
        if status == 429:
            raise ProviderError(f"Rate limited by {self.name}. {detail}")
        raise ProviderError(f"{self.name} error {status}: {detail}")

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
        started = time.monotonic()
        response, payload = self._request(
            lambda: self._payload(
                messages,
                tools=tools,
                model=model,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                stream=True,
            ),
            stream=True,
        )

        content: list[str] = []
        reasoning: list[str] = []
        details: dict[int, dict[str, Any]] = {}
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
                if not isinstance(event, dict):
                    continue

                if isinstance(event.get("error"), dict):
                    raise ProviderError(
                        f"{self.name} stream error: {event['error'].get('message', event['error'])}"
                    )
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
                raw_details = delta.get("reasoning_details")
                detail_text = ""
                if isinstance(raw_details, list) and raw_details:
                    detail_text = _merge_reasoning_details(details, raw_details)
                shown = thought if isinstance(thought, str) and thought else detail_text
                if shown:
                    reasoning.append(shown)
                    yield ReasoningDelta(shown)

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
                ToolCall.from_raw(buffer.id or f"call_{index}", buffer.name, buffer.arguments)
                for index, buffer in sorted(buffers.items())
                if buffer.name
            ],
            finish_reason=finish_reason,
            model=str(payload.get("model", "")),
        )
        if details and self.echo_reasoning_details:
            message.provider_state = {
                "openai": {"reasoning_details": [details[i] for i in sorted(details)]}
            }
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
        started = time.monotonic()
        try:
            response, payload = self._request(
                lambda: self._payload(
                    messages,
                    tools=tools,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    stream=False,
                ),
                stream=False,
            )
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
            reasoning=raw_message.get("reasoning_content") or raw_message.get("reasoning") or "",
            tool_calls=[c for c in tool_calls if c.name],
            finish_reason=choices[0].get("finish_reason"),
            model=str(payload.get("model", "")),
        )
        details = raw_message.get("reasoning_details")
        if self.echo_reasoning_details and isinstance(details, list) and details:
            message.provider_state = {"openai": {"reasoning_details": details}}
        _apply_usage(message, body.get("usage") or {}, messages, started)
        return message

    # -------------------------------------------------------------- catalog
    def list_models(self) -> list[str]:
        url = f"{self.base_url.rstrip('/')}/models"
        headers = {"User-Agent": "scode-cli", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = self.session.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, 30))
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ProviderError(f"Could not list models from {url}: {exc}") from exc
        entries = body.get("data", []) if isinstance(body, dict) else body
        return sorted(
            str(entry.get("id"))
            for entry in entries or []
            if isinstance(entry, dict) and entry.get("id")
        )

    def check(self, model: str, *, timeout: float = 25.0) -> tuple[str, str]:
        return check_model(self.base_url, self.api_key, model, timeout=timeout, headers=self.headers)


def check_model(
    base_url: str,
    api_key: str,
    model: str,
    *,
    timeout: float = 25.0,
    headers: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Probe one model. Returns (status, detail).

    status is "ok", "unavailable", "retired", "timeout" or "error".
    """
    started = time.monotonic()
    request_headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        **(headers or {}),
    }
    if api_key:
        request_headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=request_headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "stream": True,
            },
            stream=True,
            timeout=(CONNECT_TIMEOUT, timeout),
        )
    except requests.Timeout:
        return "timeout", f"no response in {timeout:.0f}s"
    except requests.RequestException as exc:
        return "error", str(exc)[:80]

    try:
        if response.status_code == 200:
            # The endpoint accepted the request, so the model is usable. Reading
            # a token is only for timing; some models close the stream first.
            try:
                for line in response.iter_lines():
                    if line:
                        break
            except requests.RequestException:
                return "ok", "accepted"
            return "ok", f"{time.monotonic() - started:.1f}s"
        if response.status_code == 404:
            return "unavailable", "not granted to this key"
        if response.status_code == 410:
            return "retired", "end of life"
        if response.status_code in (401, 403):
            return "error", "key rejected"
        return "error", f"HTTP {response.status_code}"
    except requests.RequestException:
        return "timeout", "stream broke"
    finally:
        response.close()


def _apply_usage(
    message: AssistantMessage,
    usage: dict[str, Any],
    messages: list[dict[str, Any]],
    started: float,
) -> None:
    """Normalise usage: input_tokens excludes cached tokens, as on Anthropic."""
    from ..usage import estimate_tokens

    prompt = int(usage.get("prompt_tokens") or 0)
    cached = 0
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens") or 0)
    message.cached_tokens = cached
    message.input_tokens = max(0, prompt - cached)
    message.output_tokens = int(usage.get("completion_tokens") or 0)

    cost = usage.get("cost")
    if isinstance(cost, (int, float)):
        message.reported_cost = float(cost)

    if not prompt:
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


class _BadRequest(Exception):
    """Internal: a 400 the client may be able to fix by adapting parameters."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail
