"""The interactive loop."""

from __future__ import annotations

import html
from typing import Any

from .agent.context import append_memory, expand_file_mentions
from .commands import builtin as commands
from .config import Settings
from .errors import Interrupted, ProviderError, ScodeError
from .extensions import load_commands
from .permissions import Answer, PermissionRequest
from .runtime import Runtime
from .ui.console import UI
from .ui.input import InputSession, choose, read_line
from .usage import format_tokens

# Shift+Tab cycles these, as in Claude Code; bypass is deliberately left out.
MODE_CYCLE = ("default", "acceptEdits", "plan")


class Repl(Runtime):
    def __init__(
        self,
        settings: Settings,
        ui: UI,
        *,
        resume_id: str | None = None,
        continue_last: bool = False,
    ) -> None:
        super().__init__(
            settings,
            ui,
            asker=self._ask_permission,
            plan_approver=self._approve_plan,
            ask_user=self._ask_user,
            interactive=True,
            resume_id=resume_id,
            continue_last=continue_last,
        )
        self.custom_commands = load_commands(self.settings.workspace)
        self.input = InputSession(
            self.settings.workspace,
            self._command_listing,
            toolbar=self._toolbar,
            on_cycle_mode=self._cycle_mode,
        )

    # ------------------------------------------------------------ interface
    def _command_listing(self) -> list[tuple[str, str]]:
        rows = list(commands.listing())
        rows += [(c.name, c.description or "custom command") for c in self.custom_commands.values()]
        return sorted(rows)

    def _toolbar(self) -> str:
        window = self.settings.context_window()
        parts = [
            html.escape(f"{self.settings.provider}:{self.settings.model}"),
            html.escape(self.permissions.mode) + " (shift+tab)",
            f"ctx {self.usage.context_percent(window):.0f}%",
            f"{format_tokens(self.usage.total_tokens)} tok",
        ]
        if self.usage.requests:
            parts.append(html.escape(self.usage.cost_label()))
        return "  ".join(parts)

    def _cycle_mode(self) -> None:
        current = self.permissions.mode
        position = MODE_CYCLE.index(current) if current in MODE_CYCLE else -1
        self.set_mode(MODE_CYCLE[(position + 1) % len(MODE_CYCLE)])

    # ---------------------------------------------------------- permissions
    def _ask_permission(self, request: PermissionRequest) -> tuple[Answer, str | None]:
        options: list[tuple[str, str]] = [("Yes", "run it once")]
        rules = [rule for rule in request.suggestions if rule]
        if rules:
            options.append((f"Yes, and don't ask again for {rules[0]}", "adds an allow rule"))
        options.append(("No", "tell the model what to do instead"))

        index = choose(self.ui, request.title, options, detail=request.detail, default=0)
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
            [("Yes", "leave plan mode and start implementing"), ("No, keep planning", "stay read-only")],
            default=0,
        )
        return index == 0

    def _ask_user(self, question: str, options: list[dict[str, str]], multi: bool) -> list[str] | None:
        labels = [(o["label"], o.get("description", "")) for o in options]
        labels.append(("Something else", "type your own answer"))
        if multi:
            self.ui.muted("Pick one; ask again for more.")
        index = choose(self.ui, question, labels, default=0)
        if index < len(options):
            return [options[index]["label"]]
        try:
            answer = read_line("Your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return [answer] if answer else None

    # ---------------------------------------------------------- state changes
    def set_theme(self, theme: str) -> None:
        from .config import with_overrides

        self.settings = with_overrides(self.settings, theme=theme)
        new_ui = UI(theme, quiet=self.ui.quiet, markdown=self.ui.markdown)
        self.ui = new_ui
        self.agent.ui = new_ui
        self.agent.ctx.emit = new_ui.muted

    # ----------------------------------------------------------------- loop
    def run(self, initial_prompt: str | None = None) -> int:
        self.ui.banner(
            model=f"{self.settings.provider}:{self.settings.model}",
            workspace=self.settings.workspace,
            mode=self.permissions.mode,
            api_key=bool(self.settings.api_key) or not self.settings.get_profile().key_required,
        )
        pending = initial_prompt
        try:
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
                except (KeyboardInterrupt, Interrupted):
                    self.ui.warn("Interrupted.")
                except ProviderError as exc:
                    self.ui.error(str(exc))
                except ScodeError as exc:
                    self.ui.error(str(exc))
        finally:
            self.close()
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
                # The system prompt is frozen for the session, so tell the model directly.
                self.agent.remind(f"The user added this standing instruction to {path.name}: {note}")
                self.ui.success(f"Noted in {path.name}")
            return False

        if text.startswith("/"):
            return self._run_command(text)

        self._send(text)
        return False

    def _run_command(self, text: str) -> bool:
        name, _, args = text[1:].partition(" ")
        command = commands.resolve(name)
        if command is not None:
            result = command.handler(self, args.strip())
            if result.prompt:
                self._send(result.prompt)
            return result.exit

        custom = self.custom_commands.get(name.lower())
        if custom is not None:
            self.ui.muted(f"  /{custom.name} ({custom.path.name})")
            self._send(custom.render(args.strip()))
            return False

        self.ui.error(f"Unknown command /{name}. Try /help.")
        return False

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

    def _send(self, text: str) -> None:
        profile = self.settings.get_profile()
        if not self.settings.api_key and profile.key_required:
            hint = f" Get one: {profile.signup_url}" if profile.signup_url else ""
            self.ui.error(f"No API key for {profile.label}. {profile.key_hint()}.{hint}")
            return

        info = None
        try:
            info = self.settings.model_info()
        except ScodeError:
            pass
        content, attached = expand_file_mentions(
            text, self.settings.workspace, allow_images=info.vision if info else True
        )
        if attached:
            self.ui.muted(f"  attached: {', '.join(attached)}")

        self.ui.blank()
        result = self.agent.run(content)
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
    return Repl(settings, ui, **kwargs).run(initial)
