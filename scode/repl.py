"""The interactive loop."""

from __future__ import annotations

import html
from typing import Any

from . import constants as C
from .agent.context import append_memory, expand_file_mentions
from .agent.loop import Agent
from .commands import builtin as commands
from .config import Settings, with_overrides
from .errors import ConfigError, Interrupted, ProviderError, ScodeError
from .permissions import Answer, PermissionEngine, PermissionRequest
from .providers.openai_compatible import OpenAICompatibleProvider
from .providers.registry import get_provider, list_catalog
from .session.store import SessionStore, latest_session
from .tools import ToolContext, build_registry
from .ui.console import UI
from .ui.input import InputSession, choose
from .usage import Usage, format_tokens


class Repl:
    def __init__(
        self,
        settings: Settings,
        ui: UI,
        *,
        resume_id: str | None = None,
        continue_last: bool = False,
    ) -> None:
        self.settings = settings
        self.ui = ui
        self.usage = Usage()
        self.permissions = PermissionEngine(settings, asker=self._ask_permission)
        self.provider: OpenAICompatibleProvider = self._make_provider(settings)

        self.store = SessionStore(settings.workspace, model=settings.model).open()
        self.agent = self._build_agent()

        if continue_last and resume_id is None:
            previous = latest_session(settings.workspace)
            resume_id = previous.id if previous else None
            if resume_id is None:
                ui.muted("No previous session in this directory; starting a new one.")
        if resume_id:
            try:
                self.resume_session(resume_id)
            except (FileNotFoundError, ScodeError) as exc:
                ui.warn(str(exc))

        self.input = InputSession(
            settings.workspace,
            commands.listing,
            toolbar=self._toolbar,
        )

    # --------------------------------------------------------------- wiring
    @staticmethod
    def _make_provider(settings: Settings) -> OpenAICompatibleProvider:
        """Build a provider even without a key, so /login can fix it in place."""
        try:
            return get_provider(settings)
        except ConfigError:
            return OpenAICompatibleProvider(
                name=settings.provider,
                api_key="",
                base_url=settings.base_url or C.NVIDIA_BASE_URL,
                model=settings.model,
                max_output_tokens=settings.max_output_tokens,
                temperature=settings.temperature,
            )

    def _build_agent(self, messages: list[dict[str, Any]] | None = None) -> Agent:
        registry = build_registry()
        ctx = ToolContext(
            workspace=self.settings.workspace,
            settings=self.settings,
            permissions=self.permissions,
            emit=self.ui.muted,
        )
        ctx.spawn_subagent = self._spawn_subagent
        agent = Agent(
            provider=self.provider,
            registry=registry,
            ctx=ctx,
            ui=self.ui,
            settings=self.settings,
            usage=self.usage,
            messages=messages or [],
            on_message=self.store.append,
            plan_approver=self._approve_plan,
        )
        return agent

    def _toolbar(self) -> str:
        window = self.settings.context_window()
        percent = self.usage.context_percent(window)
        parts = [
            html.escape(self.settings.model),
            html.escape(self.permissions.mode),
            f"ctx {percent:.0f}%",
            f"{format_tokens(self.usage.total_tokens)} tok",
        ]
        return "  ".join(parts)

    # ---------------------------------------------------------- permissions
    def _ask_permission(self, request: PermissionRequest) -> tuple[Answer, str | None]:
        options: list[tuple[str, str]] = [("Yes", "run it once")]
        rules = [rule for rule in request.suggestions if rule]
        if rules:
            options.append((f"Yes, and don't ask again for {rules[0]}", "adds an allow rule"))
        options.append(("No", "tell the model what to do instead"))

        index = choose(
            self.ui,
            request.title,
            options,
            detail=request.detail,
            default=0,
        )
        if index == 0:
            return Answer.ONCE, None
        if rules and index == 1:
            return Answer.ALWAYS, rules[0]
        return Answer.NO, None

    def _approve_plan(self, plan: str) -> bool:
        self.ui.blank()
        self.ui.panel(plan, title="Plan", style="scode.accent")
        index = choose(
            self.ui,
            "Ready to code?",
            [
                ("Yes", "leave plan mode and start implementing"),
                ("No, keep planning", "stay read-only"),
            ],
            default=0,
        )
        return index == 0

    # ------------------------------------------------------------ subagents
    def _spawn_subagent(
        self,
        *,
        prompt: str,
        subagent_type: str,
        read_only: bool,
        label: str,
    ) -> str:
        from .agent.prompts import SUBAGENT_PROMPTS

        child_settings = with_overrides(self.settings, max_steps=min(self.settings.max_steps, 30))
        registry = build_registry(read_only=read_only, include_task=False)
        ctx = ToolContext(
            workspace=child_settings.workspace,
            settings=child_settings,
            permissions=self.permissions,
            emit=None,
        )
        child = Agent(
            provider=self.provider,
            registry=registry,
            ctx=ctx,
            ui=self.ui,
            settings=child_settings,
            usage=Usage(),
            subagent_prompt=SUBAGENT_PROMPTS.get(subagent_type, ""),
            quiet_tools=False,
        )

        self.ui.muted(f"  running subagent: {label}")
        try:
            result = child.run(prompt)
        except ScodeError as exc:
            return f"The subagent failed: {exc}"

        self.usage.merge(child.usage)
        if result.reason == "interrupted":
            return "The subagent was interrupted before it finished."
        if not result.text.strip():
            return "The subagent produced no report."
        return result.text

    # ---------------------------------------------------------- state changes
    def reset_conversation(self) -> None:
        self.store.close()
        self.store = SessionStore(self.settings.workspace, model=self.settings.model).open()
        self.usage = Usage()
        self.agent = self._build_agent()

    def resume_session(self, session_id: str) -> None:
        store, messages = SessionStore.resume(
            self.settings.workspace, session_id, model=self.settings.model
        )
        self.store.close()
        self.store = store
        restored = [m for m in messages if m.get("role") != "system"]
        self.agent = self._build_agent(messages=[])
        self.agent.messages.extend(restored)
        self.agent._recount_context()
        self.ui.muted(f"Restored {len(restored)} messages from session {store.id[:10]}.")

    def set_model(self, model: str) -> None:
        self.settings = with_overrides(self.settings, model=model)
        self.provider.model = model
        self.provider.supports_tools = True
        self.agent.settings = self.settings
        self.agent._native_tools = self.settings.native_tools
        self.agent.refresh_system_prompt()
        self.store.record_event("model_changed", model=model)

    def set_mode(self, mode: str) -> None:
        self.permissions.set_mode(mode)
        self.agent.refresh_system_prompt()
        self.store.record_event("mode_changed", mode=mode)

    def set_api_key(self, key: str) -> None:
        self.settings = with_overrides(self.settings, api_key=key)
        self.provider.api_key = key
        self.agent.settings = self.settings

    def set_theme(self, theme: str) -> None:
        self.settings = with_overrides(self.settings, theme=theme)
        new_ui = UI(theme, quiet=self.ui.quiet, markdown=self.ui.markdown)
        self.ui = new_ui
        self.agent.ui = new_ui

    def list_models(self) -> list[str]:
        return list_catalog(self.settings)

    # ----------------------------------------------------------------- loop
    def run(self, initial_prompt: str | None = None) -> int:
        self.ui.banner(
            model=self.settings.model,
            workspace=self.settings.workspace,
            mode=self.permissions.mode,
            api_key=bool(self.settings.api_key),
        )

        pending = initial_prompt
        while True:
            if pending is not None:
                text, pending = pending, None
            else:
                try:
                    text = self.input.ask()
                except KeyboardInterrupt:
                    continue  # Ctrl+C on an empty prompt just clears the line
                except EOFError:
                    self.ui.muted("Bye.")
                    break

            text = text.strip()
            if not text:
                continue

            try:
                if self._dispatch(text):
                    break
            except KeyboardInterrupt:
                self.ui.warn("Interrupted.")
            except Interrupted:
                self.ui.warn("Interrupted.")
            except ProviderError as exc:
                self.ui.error(str(exc))
            except ScodeError as exc:
                self.ui.error(str(exc))

        self.store.close()
        return 0

    def _dispatch(self, text: str) -> bool:
        """Handle one line of input. Returns True to exit."""
        if text.startswith("!"):
            self._run_shell(text[1:].strip())
            return False

        if text.startswith("#"):
            note = text[1:].strip()
            if note:
                path = append_memory(self.settings.workspace, note)
                self.ui.success(f"Noted in {path.name}")
            return False

        if text.startswith("/"):
            return self._run_command(text)

        self._send(text)
        return False

    def _run_command(self, text: str) -> bool:
        name, _, args = text[1:].partition(" ")
        command = commands.resolve(name)
        if command is None:
            self.ui.error(f"Unknown command /{name}. Try /help.")
            return False

        result = command.handler(self, args.strip())
        if result.prompt:
            self._send(result.prompt, echo=False)
        return result.exit

    def _run_shell(self, command: str) -> None:
        """`!cmd` runs a command directly and shows the model the result."""
        if not command:
            return
        tool = self.agent.registry.get("Bash")
        if tool is None:
            self.ui.error("The Bash tool is not available in this mode.")
            return

        self.ui.tool_call(f"Bash({command})")
        try:
            result = tool.run({"command": command}, self.agent.ctx)
        except ScodeError as exc:
            self.ui.error(str(exc))
            return

        self.ui.print(result.output)
        self.agent._append(
            {
                "role": "user",
                "content": (
                    f"[The user ran this command themselves]\n$ {command}\n\n"
                    f"{result.truncated_output()}"
                ),
            }
        )

    def _send(self, text: str, *, echo: bool = True) -> None:
        if not self.settings.api_key:
            self.ui.error(
                "No API key set. Run /login, or export NVIDIA_API_KEY and restart."
            )
            return

        expanded, attached = expand_file_mentions(text, self.settings.workspace)
        if attached:
            self.ui.muted(f"  attached: {', '.join(attached)}")

        self.ui.blank()
        result = self.agent.run(expanded)
        self.ui.blank()

        if result.reason == "error" and result.error:
            self.store.record_event("turn_error", error=result.error)

        window = self.settings.context_window()
        if self.usage.context_percent(window) > 90:
            self.ui.warn(
                f"Context is {self.usage.context_percent(window):.0f}% full. "
                "Run /compact to summarise, or /clear to start fresh."
            )


def run_repl(settings: Settings, ui: UI, **kwargs: Any) -> int:
    initial = kwargs.pop("initial_prompt", None)
    repl = Repl(settings, ui, **kwargs)
    return repl.run(initial)
