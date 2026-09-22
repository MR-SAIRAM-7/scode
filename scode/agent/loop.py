"""The agent loop: model request, tool execution, repeat."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..errors import Interrupted, ProviderError, ScodeError, ToolError
from ..permissions import PermissionEngine
from ..providers.base import (
    AssistantMessage,
    ReasoningDelta,
    StreamDone,
    TextDelta,
    ToolCall,
    ToolCallStarted,
)
from ..providers.openai_compatible import OpenAICompatibleProvider, ToolsNotSupported
from ..tools import ToolContext, ToolRegistry, build_registry
from ..ui.console import UI
from ..usage import Usage, estimate_tokens
from . import prompts
from .context import environment_block, load_project_memory

TOOL_USE_MARKER = "<tool_use>"
# How many empty model turns to ride out before giving up on the turn.
MAX_EMPTY_TURNS = 3
TOOL_USE_RE = re.compile(r"<tool_use>\s*(\{.*?\})\s*</tool_use>", re.DOTALL)


@dataclass
class TurnResult:
    text: str
    reason: str = "done"  # done | max_steps | interrupted | error
    steps: int = 0
    error: str | None = None


@dataclass
class Agent:
    provider: OpenAICompatibleProvider
    registry: ToolRegistry
    ctx: ToolContext
    ui: UI
    settings: Settings
    usage: Usage = field(default_factory=Usage)
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Called with every message appended, for transcript persistence.
    on_message: Callable[[dict[str, Any]], None] | None = None
    # Prompts the user to approve a finished plan.
    plan_approver: Callable[[str], bool] | None = None
    subagent_prompt: str = ""
    quiet_tools: bool = False
    _native_tools: bool = True
    _reasoning_shown: bool = False

    def __post_init__(self) -> None:
        self._native_tools = self.settings.native_tools
        if not self.messages:
            self.messages.append(self._system_message())

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
        )

    def refresh_system_prompt(self) -> None:
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = self._system_message()
        else:
            self.messages.insert(0, self._system_message())

    # ------------------------------------------------------------- bookkeeping
    def _append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        if self.on_message:
            self.on_message(message)
        self._recount_context()

    def _recount_context(self) -> None:
        self.usage.context_tokens = sum(
            estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in self.messages
        )

    # ------------------------------------------------------------------- turn
    def run(self, user_text: str) -> TurnResult:
        self._append({"role": "user", "content": user_text})
        return self._loop()

    def _loop(self) -> TurnResult:
        final_text = ""
        empty_turns = 0
        for step in range(1, self.settings.max_steps + 1):
            self._maybe_compact()

            try:
                message = self._request()
            except Interrupted:
                self._note_interrupt()
                return TurnResult(final_text, reason="interrupted", steps=step)
            except ScodeError as exc:
                self.ui.error(str(exc))
                return TurnResult(final_text, reason="error", steps=step, error=str(exc))

            calls = list(message.tool_calls)
            if not calls and not self._native_tools:
                message, calls = self._extract_text_protocol_calls(message)

            # NVIDIA's gateway intermittently returns a 200 with an empty body.
            # That is not an answer, so retry instead of ending the turn silently.
            if not calls and not message.content.strip():
                empty_turns += 1
                if empty_turns <= MAX_EMPTY_TURNS:
                    self.ui.muted(
                        f"Empty response from the model; retrying "
                        f"({empty_turns}/{MAX_EMPTY_TURNS})."
                    )
                    continue
                self.ui.error(
                    f"The model returned an empty response {empty_turns} times in a row. "
                    "Try again, or switch models with /model --check."
                )
                return TurnResult(final_text, reason="empty", steps=step)
            empty_turns = 0

            self._append(message.to_message())
            if message.content.strip():
                final_text = message.content.strip()

            if not calls:
                if message.finish_reason == "length":
                    self.ui.warn(
                        "The model hit its output limit. Ask it to continue, or raise "
                        "--max-output-tokens."
                    )
                return TurnResult(final_text, reason="done", steps=step)

            try:
                stop = self._run_tool_calls(calls)
            except Interrupted:
                self._note_interrupt()
                return TurnResult(final_text, reason="interrupted", steps=step)
            if stop:
                return TurnResult(final_text, reason="done", steps=step)

        self.ui.warn(
            f"Stopped after {self.settings.max_steps} steps without finishing. "
            "Say 'continue' to keep going."
        )
        return TurnResult(final_text, reason="max_steps", steps=self.settings.max_steps)

    def _note_interrupt(self) -> None:
        self._append(
            {
                "role": "user",
                "content": "[The user interrupted. Stop, and wait for their next message.]",
            }
        )

    # ---------------------------------------------------------------- request
    def _request(self) -> AssistantMessage:
        tools = self.registry.schemas() if self._native_tools else None
        started = time.monotonic()

        try:
            message = self._stream_or_complete(tools)
        except ToolsNotSupported:
            self.ui.warn(
                f"{self.provider.model} does not support native tool calling — "
                "switching to the text protocol."
            )
            self._native_tools = False
            self.refresh_system_prompt()
            message = self._stream_or_complete(None)

        self.usage.record(
            input_tokens=message.input_tokens,
            output_tokens=message.output_tokens,
            cached_tokens=message.cached_tokens,
            seconds=time.monotonic() - started,
        )
        return message

    def _stream_or_complete(self, tools: list[dict[str, Any]] | None) -> AssistantMessage:
        if not self.settings.stream:
            with self.ui.status("Thinking"):
                return self.provider.complete(self.messages, tools=tools)
        return self._stream(tools)

    def _stream(self, tools: list[dict[str, Any]] | None) -> AssistantMessage:
        result: AssistantMessage | None = None
        reasoning_seen = False

        # In text-protocol mode the tool call is markup, so keep it off screen.
        marker = None if self._native_tools else TOOL_USE_MARKER

        with self.ui.streaming(stop_marker=marker) as writer:
            status_cm = self.ui.status("Thinking")
            status = status_cm.__enter__()
            status_open = True
            try:
                for event in self.provider.stream(self.messages, tools=tools):
                    if isinstance(event, TextDelta):
                        if status_open:
                            status_cm.__exit__(None, None, None)
                            status_open = False
                        writer.write(event.text)
                    elif isinstance(event, ReasoningDelta):
                        if status is not None and self.settings.verbose:
                            status.update(f"Thinking: {event.text.strip()[-60:]}")
                        reasoning_seen = True
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
        if reasoning_seen and self.settings.verbose and result.reasoning:
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
    def _run_tool_calls(self, calls: list[ToolCall]) -> bool:
        """Execute each call. Returns True when the turn should stop."""
        for call in calls:
            if self.ctx.cancelled():
                raise Interrupted("Cancelled")

            result_text, is_error, stop = self._run_one(call)
            self._record_tool_result(call, result_text, is_error)
            if stop:
                return True
        return False

    def _record_tool_result(self, call: ToolCall, content: str, is_error: bool) -> None:
        if self._native_tools:
            self._append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": content,
                }
            )
        else:
            self._append(
                {
                    "role": "user",
                    "content": f"<tool_result name=\"{call.name}\">\n{content}\n</tool_result>",
                }
            )

    def _run_one(self, call: ToolCall) -> tuple[str, bool, bool]:
        """Returns (result_text, is_error, stop_turn)."""
        if call.parse_error:
            self.ui.tool_result(f"{call.name}: {call.parse_error}", error=True)
            return f"Error: {call.parse_error}. Send the call again with valid JSON.", True, False

        tool = self.registry.get(call.name)
        if tool is None:
            available = ", ".join(self.registry.names())
            self.ui.tool_result(f"Unknown tool: {call.name}", error=True)
            return f"Error: no tool named {call.name!r}. Available tools: {available}", True, False

        try:
            summary = tool.summarize_call(call.arguments, self.ctx)
        except Exception:  # a bad path should not crash the summary line
            summary = call.name

        if not self.quiet_tools:
            self.ui.tool_call(summary)

        if call.name == "ExitPlanMode":
            return self._handle_exit_plan(call)

        try:
            request = tool.permission_request(call.arguments, self.ctx)
        except ToolError as exc:
            self.ui.tool_result(str(exc), error=True)
            return f"Error: {exc}", True, False

        allowed, reason = self.ctx.permissions.authorize(request)
        if not allowed:
            self.ui.tool_result(reason, error=True)
            return f"The call was not run. {reason}", True, False

        self.usage.tool_calls += 1
        try:
            result = tool.run(call.arguments, self.ctx)
        except ToolError as exc:
            self.ui.tool_result(str(exc), error=True)
            return f"Error: {exc}", True, False
        except KeyboardInterrupt:
            raise Interrupted("Interrupted during a tool call") from None
        except Exception as exc:  # a buggy tool must not kill the session
            detail = f"{type(exc).__name__}: {exc}"
            self.ui.tool_result(detail, error=True)
            return f"Error: the {call.name} tool failed with {detail}", True, False

        if not self.quiet_tools:
            self.ui.tool_result(result.display or "done", error=result.is_error)
            if result.detail and call.name in {"Write", "Edit", "MultiEdit"}:
                self.ui.diff(result.detail)
            elif result.detail and call.name == "TodoWrite":
                self.ui.print(result.detail)

        return result.truncated_output(), result.is_error, False

    def _handle_exit_plan(self, call: ToolCall) -> tuple[str, bool, bool]:
        plan = str(call.arguments.get("plan") or "").strip()
        if not plan:
            return "Error: plan is empty", True, False

        if self.plan_approver is None:
            return (
                "Plan mode cannot be exited in this context. Present the plan as "
                "your final answer instead."
            ), True, True

        approved = self.plan_approver(plan)
        if not approved:
            return (
                "The user did not approve the plan. Stay in plan mode, ask what "
                "they want changed, and do not modify anything."
            ), False, True

        self.ctx.permissions.set_mode("acceptEdits")
        self.ctx.plan_submitted = plan
        self.refresh_system_prompt()
        return "The user approved the plan. Plan mode is off — start implementing it now.", False, False

    # ----------------------------------------------------------- compaction
    def _maybe_compact(self) -> None:
        if not self.settings.auto_compact:
            return
        from ..constants import AUTO_COMPACT_THRESHOLD

        window = self.settings.context_window()
        self._recount_context()
        if self.usage.context_tokens < window * AUTO_COMPACT_THRESHOLD:
            return
        self.compact(automatic=True)

    def compact(self, *, automatic: bool = False, instructions: str = "") -> str:
        """Replace the transcript with a summary, keeping the latest exchange."""
        from .compact import compact_messages

        before = self.usage.context_tokens
        label = "Auto-compacting context" if automatic else "Compacting context"
        with self.ui.status(label):
            summary, kept = compact_messages(
                self.provider,
                self.messages,
                settings=self.settings,
                instructions=instructions,
            )

        self.messages = [self._system_message(), *kept]
        self._recount_context()
        after = self.usage.context_tokens
        self.ui.muted(
            f"Compacted context: {before:,} → {after:,} estimated tokens."
        )
        return summary


def build_agent(
    *,
    provider: OpenAICompatibleProvider,
    settings: Settings,
    ui: UI,
    permissions: PermissionEngine,
    usage: Usage | None = None,
    read_only: bool = False,
    subagent_prompt: str = "",
    quiet_tools: bool = False,
) -> Agent:
    registry = build_registry(read_only=read_only, include_task=not read_only)
    ctx = ToolContext(
        workspace=settings.workspace,
        settings=settings,
        permissions=permissions,
    )
    return Agent(
        provider=provider,
        registry=registry,
        ctx=ctx,
        ui=ui,
        settings=settings,
        usage=usage or Usage(),
        subagent_prompt=subagent_prompt,
        quiet_tools=quiet_tools,
    )
