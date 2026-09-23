"""The agent loop: model request, tool execution, repeat.

Invariants the loop keeps, because providers reject or silently degrade
requests that break them:
- The system prompt is built once and not edited; changes reach the model as
  appended <system-reminder> messages.
- History is append-only between compactions.
- Every tool call gets exactly one result, even when the user interrupts or a
  hook stops the turn part-way through a batch.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..errors import (
    ContextOverflow,
    Interrupted,
    ProviderError,
    ScodeError,
    ToolError,
    TransientProviderError,
)
from ..hooks import HookOutcome, HookRunner
from ..log import get_logger
from ..permissions import Decision
from ..providers.base import (
    STOP_MAX_TOKENS,
    STOP_PAUSE,
    STOP_REFUSAL,
    AssistantMessage,
    ReasoningDelta,
    StreamDone,
    TextDelta,
    ToolCall,
    ToolCallStarted,
    text_of,
)
from ..providers.openai_compatible import ToolsNotSupported
from ..tools import PARALLEL_SAFE, Tool, ToolContext, ToolRegistry, ToolResult
from ..ui.console import UI
from ..usage import Usage, estimate_tokens, request_cost
from . import prompts
from .context import environment_block, load_project_memory

TOOL_USE_MARKER = "<tool_use>"
TOOL_USE_RE = re.compile(r"<tool_use>\s*(\{.*?\})\s*</tool_use>", re.DOTALL)
# How many empty model turns to ride out before giving up on the turn.
MAX_EMPTY_TURNS = 3
# Overloads and rate limits mid-task: wait and resend, rather than abandon the work.
MAX_TRANSIENT_RETRIES = 3
TRANSIENT_BACKOFF = (5.0, 10.0, 20.0)
# How many times to ask for smaller steps after a tool call was cut off.
MAX_TRUNCATIONS = 2
# How many times a Stop hook may send the model back to work in one turn.
MAX_STOP_HOOK_RUNS = 3
MAX_PARALLEL_TOOLS = 8
IMAGE_TOKEN_ESTIMATE = 1_500
log = get_logger()


@dataclass
class TurnResult:
    text: str
    # done | max_steps | interrupted | error | empty | refused | blocked
    reason: str = "done"
    steps: int = 0
    error: str | None = None


@dataclass
class _Outcome:
    content: str
    is_error: bool = False
    stop: bool = False


@dataclass
class _Prepared:
    call: ToolCall
    tool: Tool | None = None
    # Set when the call will not run (bad input, denied, blocked by a hook).
    outcome: _Outcome | None = None


def estimate_messages(messages: list[dict[str, Any]]) -> int:
    """Token estimate of what a request would carry, ignoring provider blobs."""
    total = 0
    for message in messages:
        content = message.get("content")
        total += estimate_tokens(text_of(content))
        if isinstance(content, list):
            total += IMAGE_TOKEN_ESTIMATE * sum(
                1 for part in content if isinstance(part, dict) and part.get("type") == "image"
            )
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            total += estimate_tokens(function.get("name", "")) + estimate_tokens(
                function.get("arguments", "")
            )
        total += 4  # per-message framing
    return total


@dataclass
class Agent:
    provider: Any
    registry: ToolRegistry
    ctx: ToolContext
    ui: UI
    settings: Settings
    usage: Usage = field(default_factory=Usage)
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Called with every message appended, for transcript persistence.
    on_message: Callable[[dict[str, Any]], None] | None = None
    # Called with the carrier message after a compaction.
    on_compact: Callable[[dict[str, Any]], None] | None = None
    # Prompts the user to approve a finished plan.
    plan_approver: Callable[[str], bool] | None = None
    subagent_prompt: str = ""
    quiet_tools: bool = False
    hooks: HookRunner | None = None
    # Extra system-prompt text, such as MCP server instructions.
    extra_system: str = ""
    _native_tools: bool = True
    _reminders: list[str] = field(default_factory=list)
    _last_prompt_tokens: int = 0
    _last_request_len: int = 0
    _stop_hook_runs: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._native_tools = self.settings.native_tools and bool(
            getattr(self.provider, "supports_tools", True)
        )
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(0, self._system_message())
        self._recount_context()

    # --------------------------------------------------------- system prompt
    def _system_message(self) -> dict[str, Any]:
        return {"role": "system", "content": self.system_prompt()}

    def system_prompt(self) -> str:
        return prompts.build_system_prompt(
            self.registry,
            environment=environment_block(
                self.ctx.workspace,
                model=self.provider.model,
                permission_mode=self.ctx.permissions.mode,
            ),
            project_memory=load_project_memory(self.ctx.workspace),
            native_tools=self._native_tools,
            plan_mode=self.ctx.permissions.mode == "plan",
            subagent=self.subagent_prompt,
            model=self.provider.model,
            append="\n\n".join(
                part for part in (self.extra_system, self.settings.append_system_prompt) if part
            ),
        )

    def refresh_system_prompt(self) -> None:
        """Rebuild the system prompt. Only for changes that reset the cache anyway
        (a new model, or dropping to the text tool protocol)."""
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = self._system_message()
        else:
            self.messages.insert(0, self._system_message())

    # --------------------------------------------------------------- reminders
    def remind(self, text: str) -> None:
        """Queue a note for the model, delivered with the next message we send."""
        if text.strip():
            self._reminders.append(text.strip())

    def _flush_reminders(self) -> None:
        if self._reminders:
            body = "\n\n".join(self._reminders)
            self._reminders.clear()
            self._append({"role": "user", "content": prompts.reminder(body)})

    def set_mode(self, mode: str) -> None:
        old = self.ctx.permissions.mode
        self.ctx.permissions.set_mode(mode)
        if mode == old:
            return
        if mode == "plan":
            self.remind(prompts.PLAN_MODE_ON)
        elif old == "plan":
            self.remind(f"{prompts.PLAN_MODE_OFF} Permission mode is now {mode}.")
        else:
            self.remind(f"The user changed the permission mode to {mode}.")

    # ------------------------------------------------------------- bookkeeping
    def _append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        if self.on_message:
            self.on_message(message)
        self._recount_context()

    def _recount_context(self) -> None:
        if self._last_prompt_tokens and self._last_request_len <= len(self.messages):
            newer = self.messages[self._last_request_len:]
            self.usage.context_tokens = self._last_prompt_tokens + estimate_messages(newer)
        else:
            self.usage.context_tokens = estimate_messages(self.messages)

    def _show_notices(self, outcome: HookOutcome) -> None:
        for notice in outcome.notices:
            self.ui.warn(notice)

    # ------------------------------------------------------------------- turn
    def run(self, user_content: str | list[dict[str, Any]]) -> TurnResult:
        self.ctx.turn += 1
        self._stop_hook_runs = 0

        if self.hooks and self.hooks.has("UserPromptSubmit"):
            outcome = self.hooks.run("UserPromptSubmit", {"prompt": text_of(user_content)})
            self._show_notices(outcome)
            if outcome.blocked or outcome.halt:
                reason = outcome.reason or "Blocked by a UserPromptSubmit hook."
                self.ui.warn(f"Prompt not sent: {reason}")
                return TurnResult("", reason="blocked", error=reason)
            for extra in outcome.context:
                self.remind(extra)

        self._maybe_compact(mid_turn=False)
        self._flush_reminders()
        self._append({"role": "user", "content": user_content})
        return self._loop()

    def _loop(self) -> TurnResult:
        final_text = ""
        empty = truncations = overflow_retries = transient = 0
        step = 0
        while step < self.settings.max_steps:
            step += 1
            if step > 1:
                self._maybe_compact(mid_turn=True)
            self._flush_reminders()

            try:
                message = self._request()
            except Interrupted:
                self._note_interrupt()
                return TurnResult(final_text, reason="interrupted", steps=step)
            except TransientProviderError as exc:
                transient += 1
                if transient > MAX_TRANSIENT_RETRIES:
                    log.warning("giving up after %d transient failures: %s", transient - 1, exc)
                    self.ui.error(str(exc))
                    return TurnResult(final_text, reason="error", steps=step, error=str(exc))
                delay = TRANSIENT_BACKOFF[min(transient, len(TRANSIENT_BACKOFF)) - 1]
                log.info("transient failure, retry %d in %.0fs: %s", transient, delay, exc)
                self.ui.warn(f"{exc} - retrying in {delay:.0f}s ({transient}/{MAX_TRANSIENT_RETRIES}).")
                if not self._wait(delay):
                    self._note_interrupt()
                    return TurnResult(final_text, reason="interrupted", steps=step)
                step -= 1  # a retry is not progress; don't spend the step budget on it
                continue
            except ContextOverflow as exc:
                if overflow_retries == 0 and len(self.messages) > 2:
                    overflow_retries += 1
                    self.ui.warn("The conversation outgrew the context window; compacting and retrying.")
                    self.compact(automatic=True, mid_turn=True, overflowed=True)
                    continue
                self.ui.error(str(exc))
                return TurnResult(final_text, reason="error", steps=step, error=str(exc))
            except ScodeError as exc:
                log.warning("request failed: %s: %s", type(exc).__name__, exc)
                self.ui.error(str(exc))
                return TurnResult(final_text, reason="error", steps=step, error=str(exc))

            transient = 0
            calls = list(message.tool_calls)
            if not calls and not self._native_tools:
                message, calls = self._extract_text_protocol_calls(message)

            if message.stop_reason == STOP_REFUSAL:
                category = f" ({message.refusal_category})" if message.refusal_category else ""
                self.ui.error(
                    f"The model declined this request{category}. Rephrase it, or try another "
                    "model with /model."
                )
                return TurnResult(final_text, reason="refused", steps=step, error="refused")

            # NVIDIA's gateway intermittently returns a 200 with an empty body.
            # That isn't an answer: retry rather than end the turn silently.
            if not calls and not message.content.strip():
                empty += 1
                if empty <= MAX_EMPTY_TURNS:
                    self.ui.muted(f"Empty response from the model; retrying ({empty}/{MAX_EMPTY_TURNS}).")
                    continue
                self.ui.error(
                    f"The model returned an empty response {empty} times in a row. "
                    "Try again, or switch models with /model --check."
                )
                return TurnResult(final_text, reason="empty", steps=step)
            empty = 0

            if calls and message.stop_reason == STOP_MAX_TOKENS:
                # A tool input cut off at the token limit can parse as a valid
                # partial object; running it would do the wrong thing.
                truncations += 1
                if truncations <= MAX_TRUNCATIONS:
                    self.ui.warn("The model ran out of output tokens mid tool call; asking for smaller steps.")
                    self.remind(
                        "Your previous response hit the output token limit before its tool call "
                        "was complete, so nothing ran. Make smaller changes per call, and write "
                        "large files in several parts."
                    )
                    continue
                error = "The model kept exceeding its output limit. Raise --max-output-tokens."
                self.ui.error(error)
                return TurnResult(final_text, reason="error", steps=step, error=error)

            self._append(message.to_message())
            if message.content.strip():
                final_text = message.content.strip()

            if not calls:
                if message.stop_reason == STOP_PAUSE:
                    continue  # a server-side tool loop paused; resending resumes it
                if message.stop_reason == STOP_MAX_TOKENS:
                    self.ui.warn("The model hit its output limit. Say 'continue', or raise --max-output-tokens.")
                if self._stop_blocked():
                    continue
                return TurnResult(final_text, reason="done", steps=step)

            try:
                stop = self._run_tool_calls(calls)
            except Interrupted:
                self._note_interrupt()
                return TurnResult(final_text, reason="interrupted", steps=step)
            if stop:
                return TurnResult(final_text, reason="done", steps=step)

        self.ui.warn(
            f"Stopped after {self.settings.max_steps} steps without finishing. Say 'continue' to keep going."
        )
        return TurnResult(final_text, reason="max_steps", steps=step)

    def _stop_blocked(self) -> bool:
        """Run Stop hooks. True when one sends the model back to work."""
        if not self.hooks or not self.hooks.has("Stop") or self._stop_hook_runs >= MAX_STOP_HOOK_RUNS:
            return False
        outcome = self.hooks.run("Stop", {"stop_hook_active": self._stop_hook_runs > 0})
        self._show_notices(outcome)
        if outcome.blocked and not outcome.halt:
            self._stop_hook_runs += 1
            self.remind(f"A Stop hook asked you to keep going: {outcome.reason}")
            return True
        return False

    @staticmethod
    def _wait(seconds: float) -> bool:
        """Pause before a retry. False when the user pressed Ctrl+C."""
        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            return False
        return True

    def _note_interrupt(self) -> None:
        self._append(
            {"role": "user", "content": "[The user interrupted. Stop, and wait for their next message.]"}
        )

    # ---------------------------------------------------------------- request
    def _request(self) -> AssistantMessage:
        tools = self.registry.schemas() if self._native_tools else None
        started = time.monotonic()
        request_len = len(self.messages)
        try:
            message = self._stream_or_complete(tools)
        except ToolsNotSupported:
            self.ui.warn(
                f"{self.provider.model} does not support native tool calling - "
                "switching to the text protocol."
            )
            self._native_tools = False
            self.refresh_system_prompt()
            message = self._stream_or_complete(None)
        elapsed = time.monotonic() - started
        self._record_usage(message, elapsed, request_len)
        log.debug(
            "request model=%s messages=%d tools=%d native=%s elapsed=%.1fs stop=%s "
            "in=%d cached=%d cache_write=%d out=%d calls=%d",
            message.model or self.provider.model, request_len, len(tools or []),
            self._native_tools, elapsed, message.finish_reason, message.input_tokens,
            message.cached_tokens, message.cache_write_tokens, message.output_tokens,
            len(message.tool_calls),
        )
        return message

    def _record_usage(self, message: AssistantMessage, seconds: float, request_len: int | None = None) -> None:
        cost = message.reported_cost
        if cost is None:
            try:
                info = self.settings.model_info(message.model or self.provider.model)
            except ScodeError:
                info = None
            cost = request_cost(
                info,
                input_tokens=message.input_tokens,
                output_tokens=message.output_tokens,
                cache_read_tokens=message.cached_tokens,
                cache_write_tokens=message.cache_write_tokens,
            )
        with self._lock:
            self.usage.record(
                input_tokens=message.input_tokens,
                output_tokens=message.output_tokens,
                cached_tokens=message.cached_tokens,
                cache_write_tokens=message.cache_write_tokens,
                seconds=seconds,
                cost=cost,
            )
        if request_len is not None:
            prompt = message.input_tokens + message.cached_tokens + message.cache_write_tokens
            if prompt:
                self._last_prompt_tokens = prompt + message.output_tokens
                self._last_request_len = request_len

    def _stream_or_complete(self, tools: list[dict[str, Any]] | None) -> AssistantMessage:
        if not self.settings.stream:
            with self.ui.status("Thinking"):
                return self.provider.complete(self.messages, tools=tools)
        return self._stream(tools)

    def _stream(self, tools: list[dict[str, Any]] | None) -> AssistantMessage:
        result: AssistantMessage | None = None
        # In text-protocol mode the tool call is markup, so keep it off screen.
        marker = None if self._native_tools else TOOL_USE_MARKER

        with self.ui.streaming(stop_marker=marker) as writer:
            status_cm = self.ui.status("Thinking")
            status = status_cm.__enter__()
            status_open = True
            thought = ""
            try:
                for event in self.provider.stream(self.messages, tools=tools):
                    if isinstance(event, TextDelta):
                        if status_open:
                            status_cm.__exit__(None, None, None)
                            status_open = False
                        writer.write(event.text)
                    elif isinstance(event, ReasoningDelta):
                        thought = (thought + event.text)[-400:]
                        if status is not None and status_open:
                            tail = " ".join(thought.split())[-70:]
                            status.update(f"Thinking: {tail}")
                    elif isinstance(event, ToolCallStarted):
                        if status is not None and status_open:
                            status.update(f"Preparing {event.name}")
                    elif isinstance(event, StreamDone):
                        result = event.message
            except KeyboardInterrupt:
                raise Interrupted("Interrupted") from None
            finally:
                if status_open:
                    status_cm.__exit__(None, None, None)

        if result is None:
            raise ProviderError("The model stream ended without a complete response")
        if self.settings.verbose and result.reasoning:
            self.ui.thinking(result.reasoning)
        return result

    # ------------------------------------------------------- text protocol
    def _extract_text_protocol_calls(
        self, message: AssistantMessage
    ) -> tuple[AssistantMessage, list[ToolCall]]:
        match = TOOL_USE_RE.search(message.content)
        if not match:
            return message, []

        raw = match.group(1)
        try:
            payload = json.loads(raw)
            name = str(payload["name"])
            args = payload.get("input") or payload.get("arguments") or {}
            if not isinstance(args, dict):
                raise ValueError("input must be an object")
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            call = ToolCall(
                id="text_call",
                name="__invalid__",
                raw_arguments=raw,
                parse_error=f"could not parse the <tool_use> block: {exc}",
            )
            return message, [call]

        # Strip the block so it is not echoed back as prose.
        message.content = TOOL_USE_RE.sub("", message.content).strip()
        call = ToolCall(
            id=f"text_{int(time.monotonic() * 1000)}",
            name=name,
            arguments=args,
            raw_arguments=json.dumps(args),
        )
        return message, [call]

    # ----------------------------------------------------------- tool calls
    def _parallel_ok(self, call: ToolCall) -> bool:
        return (
            call.name in PARALLEL_SAFE
            and not call.parse_error
            and self.registry.get(call.name) is not None
        )

    def _run_tool_calls(self, calls: list[ToolCall]) -> bool:
        """Execute the calls, batching adjacent pure reads. True ends the turn."""
        recorded: set[int] = set()

        def record(index: int, outcome: _Outcome) -> None:
            self._record_tool_result(calls[index], outcome)
            recorded.add(index)

        try:
            index = 0
            while index < len(calls):
                if self.ctx.cancelled():
                    raise Interrupted("Cancelled")
                end = index
                while end < len(calls) and self._parallel_ok(calls[end]):
                    end += 1
                if end - index > 1:
                    for i, outcome in self._run_parallel(calls, range(index, end)):
                        record(i, outcome)
                    index = end
                    continue

                outcome = self._run_one(calls[index])
                record(index, outcome)
                if outcome.stop:
                    for rest in range(index + 1, len(calls)):
                        record(rest, _Outcome("Not run: the turn was stopped before this call.", True))
                    return True
                index += 1
        except (Interrupted, KeyboardInterrupt):
            for i in range(len(calls)):
                if i not in recorded:
                    record(i, _Outcome("Interrupted by the user before this call ran.", True))
            raise Interrupted("Interrupted during tool calls") from None
        return False

    def _run_parallel(self, calls: list[ToolCall], indices: range) -> list[tuple[int, _Outcome]]:
        prepared = [(i, self._prepare(calls[i])) for i in indices]
        runnable = [(i, prep) for i, prep in prepared if prep.outcome is None]
        executed: dict[int, tuple[ToolResult | None, str | None]] = {}
        if runnable:
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_TOOLS, len(runnable))) as pool:
                futures = {i: pool.submit(self._execute, prep) for i, prep in runnable}
                for i, future in futures.items():
                    executed[i] = future.result()
        return [
            (i, prep.outcome if prep.outcome is not None else self._finish(prep, executed[i]))
            for i, prep in prepared
        ]

    def _run_one(self, call: ToolCall) -> _Outcome:
        prep = self._prepare(call)
        if prep.outcome is not None:
            return prep.outcome
        try:
            executed = self._execute(prep)
        except KeyboardInterrupt:
            raise Interrupted("Interrupted during a tool call") from None
        return self._finish(prep, executed)

    def _prepare(self, call: ToolCall) -> _Prepared:
        if call.parse_error:
            self.ui.tool_result(f"{call.name}: {call.parse_error}", error=True)
            return _Prepared(
                call,
                outcome=_Outcome(f"Error: {call.parse_error}. Send the call again with valid JSON.", True),
            )

        tool = self.registry.get(call.name)
        if tool is None:
            available = ", ".join(self.registry.names())
            self.ui.tool_result(f"Unknown tool: {call.name}", error=True)
            return _Prepared(
                call,
                outcome=_Outcome(f"Error: no tool named {call.name!r}. Available tools: {available}", True),
            )

        try:
            summary = tool.summarize_call(call.arguments, self.ctx)
        except Exception:  # a bad path must not crash the summary line
            summary = call.name
        if not self.quiet_tools:
            self.ui.tool_call(summary)

        if call.name == "ExitPlanMode":
            return _Prepared(call, tool, outcome=self._handle_exit_plan(call))

        hook_permission: str | None = None
        if self.hooks and self.hooks.has("PreToolUse"):
            hooked = self.hooks.run(
                "PreToolUse", {"tool_name": call.name, "tool_input": call.arguments}, subject=call.name
            )
            self._show_notices(hooked)
            if hooked.blocked or hooked.halt:
                reason = hooked.reason or "Blocked by a PreToolUse hook."
                self.ui.tool_result(reason, error=True)
                return _Prepared(
                    call, tool, outcome=_Outcome(f"The call was not run. {reason}", True, stop=hooked.halt)
                )
            hook_permission = hooked.permission

        try:
            request = tool.permission_request(call.arguments, self.ctx)
        except ToolError as exc:
            self.ui.tool_result(str(exc), error=True)
            return _Prepared(call, tool, outcome=_Outcome(f"Error: {exc}", True))

        permissions = self.ctx.permissions
        if hook_permission == "allow" and permissions.check(request) is not Decision.DENY:
            allowed, reason = True, "allowed by a hook"
        else:
            allowed, reason = permissions.authorize(request, force_ask=hook_permission == "ask")
        if not allowed:
            self.ui.tool_result(reason, error=True)
            return _Prepared(call, tool, outcome=_Outcome(f"The call was not run. {reason}", True))
        return _Prepared(call, tool)

    def _execute(self, prep: _Prepared) -> tuple[ToolResult | None, str | None]:
        """Run a tool. Returns (result, error); never raises except on Ctrl+C."""
        assert prep.tool is not None
        with self._lock:
            self.usage.tool_calls += 1
        try:
            return prep.tool.run(prep.call.arguments, self.ctx), None
        except ToolError as exc:
            return None, str(exc)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # a buggy tool must not kill the session
            return None, f"the {prep.call.name} tool failed with {type(exc).__name__}: {exc}"

    def _finish(self, prep: _Prepared, executed: tuple[ToolResult | None, str | None]) -> _Outcome:
        call = prep.call
        result, error = executed
        if error is not None or result is None:
            self.ui.tool_result(error or "failed", error=True)
            outcome = _Outcome(f"Error: {error}", True)
        else:
            if not self.quiet_tools:
                self.ui.tool_result(result.display or "done", error=result.is_error)
                if result.detail and call.name in {"Write", "Edit", "MultiEdit"}:
                    self.ui.diff(result.detail)
                elif result.detail and call.name == "TodoWrite":
                    self.ui.print(result.detail)
            outcome = _Outcome(result.truncated_output(), result.is_error)

        if self.hooks and self.hooks.has("PostToolUse"):
            hooked = self.hooks.run(
                "PostToolUse",
                {
                    "tool_name": call.name,
                    "tool_input": call.arguments,
                    "tool_response": {"output": outcome.content[:20_000], "is_error": outcome.is_error},
                },
                subject=call.name,
            )
            self._show_notices(hooked)
            if hooked.blocked and hooked.reason:
                outcome.content += f"\n\n[PostToolUse hook] {hooked.reason}"
            for extra in hooked.context:
                outcome.content += f"\n\n{extra}"
            outcome.stop = outcome.stop or hooked.halt
        return outcome

    def _record_tool_result(self, call: ToolCall, outcome: _Outcome) -> None:
        if self._native_tools:
            message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": outcome.content,
            }
            if outcome.is_error:
                message["is_error"] = True
            self._append(message)
        else:
            self._append(
                {
                    "role": "user",
                    "content": f"<tool_result name=\"{call.name}\">\n{outcome.content}\n</tool_result>",
                }
            )

    def _handle_exit_plan(self, call: ToolCall) -> _Outcome:
        plan = str(call.arguments.get("plan") or "").strip()
        if not plan:
            return _Outcome("Error: plan is empty", True)
        if self.plan_approver is None:
            return _Outcome(
                "Plan mode cannot be exited in this context. Present the plan as your final answer.",
                True,
                stop=True,
            )
        if not self.plan_approver(plan):
            return _Outcome(
                "The user did not approve the plan. Stay in plan mode, ask what they want changed, "
                "and don't modify anything.",
                stop=True,
            )
        self.ctx.plan_submitted = plan
        # Switch mode without a reminder: this tool result tells the model directly.
        self.ctx.permissions.set_mode("acceptEdits")
        return _Outcome("The user approved the plan. Plan mode is off - start implementing it now.")

    # ----------------------------------------------------------- compaction
    def _maybe_compact(self, *, mid_turn: bool) -> None:
        if not self.settings.auto_compact or len(self.messages) <= 2:
            return
        from ..constants import AUTO_COMPACT_THRESHOLD

        self._recount_context()
        if self.usage.context_tokens >= self.settings.context_window() * AUTO_COMPACT_THRESHOLD:
            self.compact(automatic=True, mid_turn=mid_turn)

    def compact(
        self,
        *,
        automatic: bool = False,
        instructions: str = "",
        mid_turn: bool = False,
        overflowed: bool = False,
    ) -> str:
        """Replace the transcript with a summary. Returns the summary."""
        from .compact import (
            build_carrier,
            last_user_request,
            summarize_in_place,
            summarize_transcript,
        )

        body = self.messages[1:]
        if len(body) < 2:
            return ""
        if self.hooks and self.hooks.has("PreCompact"):
            self._show_notices(
                self.hooks.run("PreCompact", {"trigger": "auto" if automatic else "manual"})
            )

        before = self.usage.context_tokens
        label = "Auto-compacting context" if automatic else "Compacting context"
        with self.ui.status(label):
            summary = None
            if not overflowed:
                tools = self.registry.schemas() if self._native_tools else None
                summary = summarize_in_place(
                    self.provider,
                    self.messages,
                    tools=tools,
                    instructions=instructions,
                    on_usage=lambda m: self._record_usage(m, 0.0),
                )
            if not summary:
                summary = summarize_transcript(
                    self.provider,
                    body,
                    settings=self.settings,
                    instructions=instructions,
                    on_usage=lambda m: self._record_usage(m, 0.0),
                )

        carrier = build_carrier(summary, last_user_request(body), mid_turn=mid_turn)
        self.messages = [self.messages[0], carrier]
        self._reminders.clear()
        self._last_prompt_tokens = 0
        self._last_request_len = 0
        self._recount_context()
        if self.on_compact:
            self.on_compact(carrier)
        self.ui.muted(f"Compacted context: {before:,} -> {self.usage.context_tokens:,} tokens.")
        return summary
