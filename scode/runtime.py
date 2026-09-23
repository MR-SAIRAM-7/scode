"""One session's moving parts, shared by the interactive REPL and print mode."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .agent.loop import Agent
from .agent.prompts import SUBAGENT_PROMPTS
from .config import Settings, switch_provider, with_overrides
from .errors import ConfigError, ScodeError
from .extensions import AgentDefinition, load_agents
from .hooks import HookRunner
from .mcp import McpManager
from .permissions import Asker, PermissionEngine
from .providers.profiles import split_model_spec
from .providers.registry import autodetect_model, build_provider
from .session.store import SessionStore, latest_session
from .tools import ToolContext, build_registry
from .tools.background import kill_all
from .ui.console import UI
from .usage import Usage

SUBAGENT_MAX_STEPS = 40


def ensure_model(settings: Settings, ui: UI) -> Settings:
    """Local servers pick up whichever model is loaded when none is named."""
    profile = settings.get_profile()
    if settings.model or not profile.auto_model:
        return settings
    detected = autodetect_model(settings)
    if detected:
        ui.muted(f"Using {detected}, the model {profile.label} has loaded.")
        return with_overrides(settings, model=detected, small_model=settings.small_model or detected)
    ui.warn(f"No model found on {settings.base_url}. Load one, then pick it with /model <name>.")
    return settings


class Runtime:
    def __init__(
        self,
        settings: Settings,
        ui: UI,
        *,
        asker: Asker | None = None,
        plan_approver: Callable[[str], bool] | None = None,
        ask_user: Callable[..., list[str] | None] | None = None,
        interactive: bool = True,
        resume_id: str | None = None,
        continue_last: bool = False,
        start_mcp: bool = True,
    ) -> None:
        self.ui = ui
        self.settings = ensure_model(settings, ui)
        self.interactive = interactive
        self.ask_user = ask_user
        self.plan_approver = plan_approver
        self.usage = Usage()
        self.permissions = PermissionEngine(self.settings, asker=asker)
        self.provider = self._make_provider(self.settings)
        self.store = SessionStore(self.settings.workspace, model=self.settings.model).open()
        self.hooks = HookRunner(
            self.settings.hooks,
            self.settings.workspace,
            session_id=self.store.id,
            transcript_path=str(self.store.path),
        )
        for problem in self.hooks.errors:
            ui.warn(f"hooks: {problem}")
        self.custom_agents: dict[str, AgentDefinition] = load_agents(self.settings.workspace)

        self.mcp = McpManager(self.settings.mcp_servers, self.settings.workspace)
        if start_mcp and self.settings.mcp_servers:
            with ui.status(f"Starting {len(self.settings.mcp_servers)} MCP server(s)"):
                self.mcp.start()
            for failed in self.mcp.failures():
                ui.warn(f"MCP server {failed.name} failed: {failed.error}")

        self.agent = self._build_agent()

        if continue_last and resume_id is None:
            previous = latest_session(self.settings.workspace)
            resume_id = previous.id if previous else None
            if resume_id is None:
                ui.muted("No previous session in this directory; starting a new one.")
        if resume_id:
            try:
                self.resume_session(resume_id)
            except (FileNotFoundError, ScodeError) as exc:
                ui.warn(str(exc))

        if self.hooks.has("SessionStart"):
            outcome = self.hooks.run("SessionStart", {"source": "resume" if resume_id else "startup"})
            for notice in outcome.notices:
                ui.warn(notice)
            for extra in outcome.context:
                self.agent.remind(extra)

    # --------------------------------------------------------------- wiring
    @staticmethod
    def _make_provider(settings: Settings) -> Any:
        """Build the provider even without a key, so /login can fix it in place."""
        try:
            return build_provider(settings)
        except ConfigError:
            return build_provider(settings, require_key=False)

    def _build_agent(self, messages: list[dict[str, Any]] | None = None) -> Agent:
        registry = build_registry(
            custom_agents={name: a.description for name, a in self.custom_agents.items()},
            extra=self.mcp.tools(),
            interactive=self.interactive,
        )
        ctx = ToolContext(
            workspace=self.settings.workspace,
            settings=self.settings,
            permissions=self.permissions,
            emit=self.ui.muted,
            ask_user=self.ask_user,
        )
        ctx.spawn_subagent = self.spawn_subagent
        mcp_notes = self.mcp.instructions()
        return Agent(
            provider=self.provider,
            registry=registry,
            ctx=ctx,
            ui=self.ui,
            settings=self.settings,
            usage=self.usage,
            messages=messages or [],
            on_message=self.store.append,
            on_compact=self.store.record_compaction,
            plan_approver=self.plan_approver,
            hooks=self.hooks,
            extra_system=f"# MCP servers\n\n{mcp_notes}" if mcp_notes else "",
        )

    # ------------------------------------------------------------ subagents
    def _subagent_provider(self, definition: AgentDefinition | None, settings: Settings) -> tuple[Any, Settings]:
        spec = definition.model.strip() if definition else ""
        # "inherit" is Claude Code's spelling for "the parent's model".
        if not spec or spec.lower() == "inherit":
            return self.provider, settings
        if spec.lower() == "small":
            if not settings.small_model or settings.small_model == settings.model:
                return self.provider, settings
            spec = settings.small_model
        provider_name, model = split_model_spec(spec, settings.providers)
        try:
            if provider_name and provider_name != settings.provider:
                settings = switch_provider(settings, provider_name, model)
            else:
                settings = with_overrides(settings, model=model)
            return build_provider(settings), settings
        except ScodeError as exc:
            self.ui.warn(f"Agent {definition.name}: can't use model {definition.model!r} ({exc}); using {self.settings.model}.")
            return self.provider, self.settings

    def spawn_subagent(
        self,
        *,
        prompt: str,
        subagent_type: str,
        read_only: bool,
        label: str,
    ) -> str:
        definition = self.custom_agents.get(subagent_type)
        settings = with_overrides(self.settings, max_steps=min(self.settings.max_steps, SUBAGENT_MAX_STEPS))
        provider, settings = self._subagent_provider(definition, settings)

        registry = build_registry(
            read_only=read_only,
            include_task=False,
            extra=[] if read_only else self.mcp.tools(),
            only=definition.tools if definition and definition.tools else None,
            interactive=False,
        )
        parent = self.agent.ctx
        ctx = ToolContext(
            workspace=settings.workspace,
            settings=settings,
            permissions=self.permissions,
            checkpoints=parent.checkpoints,
            turn=parent.turn,
            shells=parent.shells,
        )
        if definition:
            system = (
                f"# You are the {definition.name} subagent\n\n{definition.prompt}\n\n"
                + SUBAGENT_PROMPTS["general-purpose"].split("\n\n", 1)[1]
            )
        else:
            system = SUBAGENT_PROMPTS.get(subagent_type, SUBAGENT_PROMPTS["general-purpose"])

        child = Agent(
            provider=provider,
            registry=registry,
            ctx=ctx,
            ui=self.ui,
            settings=settings,
            usage=Usage(),
            subagent_prompt=system,
            hooks=self.hooks,
        )
        self.ui.muted(f"  running subagent: {label}")
        try:
            result = child.run(prompt)
        except ScodeError as exc:
            return f"The subagent failed: {exc}"
        finally:
            self.usage.merge(child.usage)

        if self.hooks.has("SubagentStop"):
            for notice in self.hooks.run("SubagentStop", {"subagent_type": subagent_type}).notices:
                self.ui.warn(notice)
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
        store, messages = SessionStore.resume(self.settings.workspace, session_id, model=self.settings.model)
        self.store.close()
        self.store = store
        restored = [m for m in messages if m.get("role") != "system"]
        self.agent = self._build_agent(messages=[])
        self.agent.messages.extend(restored)
        self.agent._recount_context()
        self.ui.muted(f"Restored {len(restored)} messages from session {store.id[:10]}.")

    def set_model(self, model: str) -> None:
        provider_name, model_id = split_model_spec(model, self.settings.providers)
        if provider_name and provider_name != self.settings.provider:
            self.set_provider(provider_name, model_id)
            return
        self.settings = with_overrides(self.settings, model=model_id)
        self.provider = self._make_provider(self.settings)
        self._rebind()
        self.agent.refresh_system_prompt()
        self.store.record_event("model_changed", model=self.settings.model)

    def set_provider(self, provider: str, model: str | None = None) -> None:
        settings = switch_provider(self.settings, provider, model)
        settings = ensure_model(settings, self.ui)
        self.settings = settings
        self.provider = self._make_provider(settings)
        self._rebind()
        self.agent.refresh_system_prompt()
        self.store.record_event("provider_changed", provider=settings.provider, model=settings.model)

    def set_effort(self, effort: str) -> None:
        self.settings = with_overrides(self.settings, effort=effort)
        self.provider = self._make_provider(self.settings)
        self._rebind()

    def _rebind(self) -> None:
        """Point the live agent at new settings and provider, keeping history."""
        self.permissions.settings = self.settings
        self.agent.provider = self.provider
        self.agent.settings = self.settings
        self.agent.ctx.settings = self.settings
        self.agent._native_tools = self.settings.native_tools and bool(
            getattr(self.provider, "supports_tools", True)
        )

    def set_mode(self, mode: str) -> None:
        self.agent.set_mode(mode)
        self.store.record_event("mode_changed", mode=mode)

    def set_api_key(self, key: str, provider: str | None = None) -> None:
        if provider and provider != self.settings.provider:
            return  # stored for later; the active provider is unaffected
        self.settings = with_overrides(self.settings, api_key=key)
        self.provider = self._make_provider(self.settings)
        self._rebind()

    def list_models(self) -> list[str]:
        return build_provider(self.settings, require_key=False).list_models()

    def close(self) -> None:
        if self.hooks.has("SessionEnd"):
            try:
                self.hooks.run("SessionEnd", {"reason": "exit"})
            except Exception:
                pass
        kill_all(self.agent.ctx)
        self.mcp.close()
        self.store.close()
