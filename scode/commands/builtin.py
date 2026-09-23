"""Slash commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .. import constants as C
from ..config import (
    PERMISSION_MODES,
    detect_key_provider,
    mask_key,
    remove_project_permission,
    save_api_key,
    save_model_choice,
    save_user_setting,
    user_settings_path,
)
from ..errors import ProviderError, ScodeError
from ..providers.profiles import (
    BUILTIN_PROFILES,
    EFFORT_LEVELS,
    canonical_name,
    known_provider_names,
    resolve_profile,
)
from ..session.store import list_sessions
from ..usage import format_duration, format_tokens

if TYPE_CHECKING:  # pragma: no cover
    from ..repl import Repl


@dataclass
class CommandResult:
    exit: bool = False
    # When set, this text is sent to the model as if the user typed it.
    prompt: str | None = None


@dataclass
class Command:
    name: str
    description: str
    handler: Callable[[Repl, str], CommandResult]
    aliases: tuple[str, ...] = ()
    usage: str = ""


REGISTRY: dict[str, Command] = {}
_ALIASES: dict[str, str] = {}


def command(name: str, description: str, *, aliases: tuple[str, ...] = (), usage: str = ""):
    def decorator(func: Callable[[Repl, str], CommandResult]):
        REGISTRY[name] = Command(name, description, func, aliases, usage)
        for alias in aliases:
            _ALIASES[alias] = name
        return func

    return decorator


def resolve(name: str) -> Command | None:
    key = name.lstrip("/").lower()
    if key in REGISTRY:
        return REGISTRY[key]
    if key in _ALIASES:
        return REGISTRY[_ALIASES[key]]
    return None


def listing() -> list[tuple[str, str]]:
    rows = [(cmd.name, cmd.description) for cmd in REGISTRY.values()]
    rows += [(alias, f"alias for /{target}") for alias, target in _ALIASES.items()]
    return sorted(rows)


# ------------------------------------------------------------------ session

@command("help", "Show available commands", aliases=("h", "?"))
def _help(repl: Repl, args: str) -> CommandResult:
    rows = [
        [f"/{cmd.name}" + (f" {cmd.usage}" if cmd.usage else ""), cmd.description]
        for cmd in sorted(REGISTRY.values(), key=lambda c: c.name)
    ]
    repl.ui.blank()
    repl.ui.table(["command", "what it does"], rows, title="scode commands")
    custom = getattr(repl, "custom_commands", {})
    if custom:
        repl.ui.blank()
        repl.ui.table(
            ["custom command", "what it does"],
            [[f"/{c.name}" + (f" {c.argument_hint}" if c.argument_hint else ""), c.description]
             for c in sorted(custom.values(), key=lambda c: c.name)],
            title="your commands",
        )
    repl.ui.blank()
    repl.ui.muted(
        "Input:  @path attach a file or image   !command run it yourself   #note save to SCODE.md"
    )
    repl.ui.muted(
        "Keys:   shift+tab cycle mode   esc+enter or ctrl+j newline   ctrl+c interrupt   ctrl+d exit"
    )
    return CommandResult()


@command("exit", "Exit scode", aliases=("quit", "q"))
def _exit(repl: Repl, args: str) -> CommandResult:
    return CommandResult(exit=True)


@command("clear", "Start a fresh conversation, keeping settings", aliases=("reset", "new"))
def _clear(repl: Repl, args: str) -> CommandResult:
    repl.reset_conversation()
    repl.ui.success("Conversation cleared.")
    return CommandResult()


@command("compact", "Summarise the conversation to free up context", usage="[focus]")
def _compact(repl: Repl, args: str) -> CommandResult:
    if len(repl.agent.messages) <= 2:
        repl.ui.muted("Nothing to compact yet.")
        return CommandResult()
    repl.agent.compact(instructions=args)
    return CommandResult()


@command("status", "Show session, provider, context and cost")
def _status(repl: Repl, args: str) -> CommandResult:
    settings = repl.settings
    usage = repl.usage
    window = settings.context_window()
    profile = settings.get_profile()
    rows = [
        ["session", repl.store.id],
        ["provider", f"{profile.label} ({settings.base_url})"],
        ["model", settings.model],
        ["effort", settings.effort or "provider default"],
        ["api key", mask_key(settings.api_key) if settings.api_key else ("not needed" if not profile.key_required else "not set")],
        ["workspace", str(settings.workspace)],
        ["mode", repl.permissions.mode],
        ["tools", str(len(repl.agent.registry))],
        ["mcp servers", str(len(repl.mcp.clients))],
        ["native tools", "yes" if repl.agent._native_tools else "no (text protocol)"],
        ["context", f"{format_tokens(usage.context_tokens)} / {format_tokens(window)} ({usage.context_percent(window):.0f}%)"],
        ["requests", str(usage.requests)],
        ["tool calls", str(usage.tool_calls)],
        ["cost", usage.cost_label() if usage.requests else "-"],
        ["elapsed", format_duration(usage.wall_seconds)],
    ]
    repl.ui.blank()
    repl.ui.table(["", ""], rows, title="status")
    repl.ui.blank()
    return CommandResult()


@command("cost", "Show token usage and spend for this session", aliases=("usage", "tokens"))
def _cost(repl: Repl, args: str) -> CommandResult:
    usage = repl.usage
    rows = [
        ["input tokens", format_tokens(usage.input_tokens)],
        ["cache reads", f"{format_tokens(usage.cached_tokens)} ({usage.cache_hit_rate:.0f}% of prompt)"],
        ["cache writes", format_tokens(usage.cache_write_tokens)],
        ["output tokens", format_tokens(usage.output_tokens)],
        ["api requests", str(usage.requests)],
        ["tool calls", str(usage.tool_calls)],
        ["api time", format_duration(usage.api_seconds)],
        ["wall time", format_duration(usage.wall_seconds)],
        ["cost", usage.cost_label() if usage.requests else "-"],
    ]
    repl.ui.blank()
    repl.ui.table(["", ""], rows, title="usage")
    if usage.requests and not usage.fully_priced:
        repl.ui.muted("Some requests used a model with no published price, so the total is a lower bound.")
    if repl.settings.provider == "nvidia":
        repl.ui.muted("NVIDIA's free tier is metered in credits, not dollars.")
    repl.ui.blank()
    return CommandResult()


# ------------------------------------------------------------ provider/model

@command("provider", "Show providers or switch to one", usage="[name [model]]")
def _provider(repl: Repl, args: str) -> CommandResult:
    parts = args.split()
    if not parts:
        rows = []
        for name in known_provider_names(repl.settings.providers):
            try:
                profile = resolve_profile(name, repl.settings.providers)
            except ScodeError:
                continue
            marker = " (current)" if name == repl.settings.provider else ""
            keyed = any(os.getenv(e) for e in profile.api_key_env) or name in repl.settings.api_keys
            key = "set" if keyed else ("not needed" if not profile.key_required else "-")
            rows.append([name + marker, profile.label, profile.default_model or "(auto)", key])
        repl.ui.blank()
        repl.ui.table(["provider", "name", "default model", "key"], rows, title="providers")
        repl.ui.muted(
            "Switch with /provider <name> [model]. Any OpenAI-compatible endpoint can be added "
            "under \"providers\" in settings.json; any provider on models.dev works by name."
        )
        repl.ui.blank()
        return CommandResult()

    name = canonical_name(parts[0])
    model = parts[1] if len(parts) > 1 else None
    repl.set_provider(name, model)
    profile = repl.settings.get_profile()
    repl.ui.success(f"Provider: {profile.label}, model {repl.settings.model or '(none loaded)'}")
    _remember_choice(repl)
    if profile.key_required and not repl.settings.api_key:
        repl.ui.warn(f"No key for {profile.label}. {profile.key_hint()}.")
    return CommandResult()


def _remember_choice(repl: Repl) -> None:
    """Keep a /provider or /model choice for the next session, as Claude Code does."""
    try:
        path = save_model_choice(repl.settings.provider, repl.settings.model)
    except (ScodeError, OSError) as exc:
        repl.ui.warn(f"Switched for this session only; could not save the default: {exc}")
        return
    repl.ui.muted(f"  Saved as your default in {path}")


@command("model", "Show or change the model", usage="[name | provider:name | --check | --all]")
def _model(repl: Repl, args: str) -> CommandResult:
    arg = args.strip()

    if arg in {"--check", "-c"}:
        return _model_check(repl)

    if arg in {"--all", "--list", "-a"} or arg.startswith("--search"):
        query = arg.partition(" ")[2].strip().lower() if arg.startswith("--search") else ""
        try:
            with repl.ui.status("Fetching the model list"):
                models = repl.list_models()
        except ProviderError as exc:
            repl.ui.error(str(exc))
            return CommandResult()
        if query:
            models = [m for m in models if query in m.lower()]
        repl.ui.blank()
        repl.ui.table(["model"], [[m] for m in models], title=f"{len(models)} models at {repl.settings.base_url}")
        repl.ui.muted("Set one with /model <name>. Listed isn't the same as usable: /model --check probes.")
        repl.ui.blank()
        return CommandResult()

    if not arg:
        profile = repl.settings.get_profile()
        rows = []
        for name in profile.models or ((repl.settings.model,) if repl.settings.model else ()):
            marker = " (current)" if name == repl.settings.model else ""
            info = None
            try:
                info = repl.settings.model_info(name)
            except ScodeError:
                pass
            price = (
                f"${info.input_cost:g} / ${info.output_cost:g}" if info and info.priced else "-"
            )
            context = format_tokens(info.context) if info and info.context else "-"
            rows.append([name + marker, context, price])
        repl.ui.blank()
        repl.ui.table(["model", "context", "$/M in / out"], rows, title=f"{profile.label} models")
        if profile.model_aliases:
            repl.ui.muted("Aliases: " + ", ".join(f"{k} = {v}" for k, v in profile.model_aliases.items()))
        repl.ui.muted("Change with /model <name>, or /model provider:<name> to switch provider too.")
        repl.ui.blank()
        return CommandResult()

    note = C.KNOWN_UNAVAILABLE.get(arg)
    if note:
        repl.ui.warn(f"{arg} is known not to work: {note}.")
    repl.set_model(arg)
    repl.ui.success(f"Model: {repl.settings.provider}:{repl.settings.model}")
    _remember_choice(repl)
    return CommandResult()


def _model_check(repl: Repl) -> CommandResult:
    """Probe each candidate model so the user sees what their key can reach."""
    from ..providers.registry import check_model

    profile = repl.settings.get_profile()
    if profile.key_required and not repl.settings.api_key:
        repl.ui.error(f"No API key for {profile.label}. {profile.key_hint()}.")
        return CommandResult()

    candidates = list(profile.models)
    for extra in (repl.settings.model, repl.settings.small_model):
        if extra and extra not in candidates:
            candidates.append(extra)

    labels = {"ok": "works", "unavailable": "not on your key", "retired": "retired",
              "timeout": "no response", "error": "error"}
    rows: list[list[str]] = []
    working: list[str] = []
    repl.ui.blank()
    with repl.ui.status("Probing models") as status:
        for name in candidates:
            if status is not None:
                status.update(f"Probing {name}")
            state, detail = check_model(repl.settings, name)
            if state == "ok":
                working.append(name)
            current = " (current)" if name == repl.settings.model else ""
            rows.append([name + current, labels.get(state, state), detail])

    repl.ui.table(["model", "status", "detail"], rows, title=f"{profile.label} availability")
    if working:
        repl.ui.success(f"{len(working)} of {len(candidates)} models responded.")
        if repl.settings.model not in working:
            repl.ui.warn(f"Your current model isn't responding. Switch with: /model {working[0]}")
    else:
        repl.ui.error("No models responded. Check your key and network.")
    repl.ui.blank()
    return CommandResult()


@command("effort", "Show or set reasoning effort", usage="[low|medium|high|xhigh|max|default]")
def _effort(repl: Repl, args: str) -> CommandResult:
    arg = args.strip().lower()
    profile = repl.settings.get_profile()
    if not arg:
        current = repl.settings.effort or "provider default"
        repl.ui.info(f"Effort: {current}. Levels: {', '.join(EFFORT_LEVELS)}.")
        if profile.effort_style == "none":
            repl.ui.muted(f"{profile.label} has no effort control; the setting is ignored there.")
        return CommandResult()
    if arg == "default":
        repl.set_effort("")
        repl.ui.success("Effort: provider default")
        return CommandResult()
    if arg not in EFFORT_LEVELS:
        repl.ui.error(f"Effort must be one of: {', '.join(EFFORT_LEVELS)}, or default.")
        return CommandResult()
    repl.set_effort(arg)
    repl.ui.success(f"Effort: {arg}")
    return CommandResult()


@command("login", "Store an API key for a provider", usage="[provider] [key]")
def _login(repl: Repl, args: str) -> CommandResult:
    # /login | /login <provider> [key] | /login <key>  (provider guessed from the key's prefix)
    parts = args.split()
    provider = repl.settings.provider
    key = ""
    if parts and canonical_name(parts[0]) in known_provider_names(repl.settings.providers):
        provider = canonical_name(parts[0])
        key = parts[1] if len(parts) > 1 else ""
    elif parts:
        key = parts[0]
        provider = detect_key_provider(key) or provider

    profile = resolve_profile(provider, repl.settings.providers)
    if not key:
        if profile.signup_url:
            repl.ui.muted(f"Get a key for {profile.label}: {profile.signup_url}")
        try:
            from prompt_toolkit import prompt as ptk_prompt

            key = ptk_prompt(f"{profile.label} API key: ", is_password=True).strip()
        except (EOFError, KeyboardInterrupt):
            return CommandResult()
    if not key:
        repl.ui.warn("No key entered.")
        return CommandResult()

    path = save_api_key(provider, key)
    repl.settings.api_keys[provider] = key
    repl.set_api_key(key, provider)
    repl.ui.success(f"Saved the {profile.label} key to {path}")
    if provider != repl.settings.provider:
        repl.ui.muted(f"Switch to it with /provider {provider}")
    return CommandResult()


@command("logout", "Forget the stored key for a provider", usage="[provider]")
def _logout(repl: Repl, args: str) -> CommandResult:
    provider = canonical_name(args.strip()) if args.strip() else repl.settings.provider
    try:
        save_api_key(provider, None)
    except (OSError, ScodeError) as exc:
        repl.ui.error(f"Could not update {user_settings_path()}: {exc}")
        return CommandResult()
    repl.settings.api_keys.pop(provider, None)
    profile = BUILTIN_PROFILES.get(provider)
    env_hint = f" {profile.api_key_env[0]} in your environment still applies." if profile and profile.api_key_env else ""
    repl.ui.success(f"Removed the stored {provider} key.{env_hint}")
    return CommandResult()


# -------------------------------------------------------------- permissions

@command("mode", "Show or set the permission mode", usage="[default|acceptEdits|plan|bypassPermissions]")
def _mode(repl: Repl, args: str) -> CommandResult:
    arg = args.strip()
    if not arg:
        repl.ui.blank()
        repl.ui.table(
            ["mode", "behaviour"],
            [
                ["default", "asks before edits and shell commands"],
                ["acceptEdits", "file edits run automatically; shell still asks"],
                ["plan", "read-only: research and propose, change nothing"],
                ["bypassPermissions", "runs everything without asking"],
            ],
            title=f"permission mode (currently: {repl.permissions.mode})",
        )
        repl.ui.muted("Shift+Tab cycles default -> acceptEdits -> plan.")
        repl.ui.blank()
        return CommandResult()

    match = next((m for m in PERMISSION_MODES if m.lower() == arg.lower()), None)
    if match is None:
        repl.ui.error(f"Unknown mode {arg!r}. Choose from: {', '.join(PERMISSION_MODES)}")
        return CommandResult()

    if match == "bypassPermissions":
        from ..ui.input import confirm

        repl.ui.warn("bypassPermissions runs every command and edit without asking, including destructive ones.")
        if not confirm(repl.ui, "Enable it?"):
            return CommandResult()

    repl.set_mode(match)
    repl.ui.success(f"Permission mode: {match}")
    return CommandResult()


@command("permissions", "Show, add or remove allow/deny rules", usage="[allow|deny|remove RULE]")
def _permissions(repl: Repl, args: str) -> CommandResult:
    from ..config import save_project_permission

    parts = args.split(None, 1)
    if not parts:
        repl.ui.blank()
        repl.ui.info(repl.permissions.describe())
        repl.ui.muted(
            "Rules: Tool, Tool(prefix:*), Tool(glob), mcp__server. "
            "Add with /permissions allow 'Bash(npm test:*)'."
        )
        repl.ui.blank()
        return CommandResult()
    action = parts[0].lower()
    rule = parts[1].strip().strip("'\"") if len(parts) > 1 else ""
    if action not in {"allow", "deny", "remove"} or not rule:
        repl.ui.error("Usage: /permissions allow|deny|remove RULE")
        return CommandResult()
    if action == "remove":
        from ..permissions import Rule

        removed = remove_project_permission(repl.settings.workspace, rule)
        parsed = Rule.parse(rule)
        repl.permissions.session_allow = [r for r in repl.permissions.session_allow if r != parsed]
        repl.permissions.session_deny = [r for r in repl.permissions.session_deny if r != parsed]
        repl.ui.success(f"Removed {rule}" if removed else f"{rule} was not in .scode/settings.local.json")
        return CommandResult()
    repl.permissions.remember(rule, deny=action == "deny", persist=False)
    path = save_project_permission(repl.settings.workspace, rule, deny=action == "deny")
    repl.ui.success(f"{action.title()}ed {rule} (saved to {path.name})")
    return CommandResult()


@command("tools", "List the tools available to the model")
def _tools(repl: Repl, args: str) -> CommandResult:
    rows = []
    for tool in repl.agent.registry.all():
        kind = "write" if tool.mutating else "read"
        summary = " ".join(tool.description.split())
        rows.append([tool.name, kind, summary[:90] + ("..." if len(summary) > 90 else "")])
    repl.ui.blank()
    repl.ui.table(["tool", "kind", "description"], rows, title="tools")
    repl.ui.blank()
    return CommandResult()


@command("agents", "List the subagents the Task tool can launch")
def _agents(repl: Repl, args: str) -> CommandResult:
    from ..tools.task import SUBAGENT_TYPES

    rows = [[name, "built-in", " ".join(desc.split())] for name, desc in SUBAGENT_TYPES.items()]
    for name, agent in sorted(getattr(repl, "custom_agents", {}).items()):
        tools = ", ".join(agent.tools) if agent.tools else "all"
        rows.append([name, f"{agent.path.parent.parent.name}/{agent.path.parent.name}", f"{agent.description} (tools: {tools}{', model: ' + agent.model if agent.model else ''})"])
    repl.ui.blank()
    repl.ui.table(["subagent", "from", "what it does"], rows, title="subagents")
    repl.ui.muted("Define your own in .scode/agents/<name>.md (frontmatter: description, tools, model).")
    repl.ui.blank()
    return CommandResult()


@command("mcp", "Show MCP servers and their tools")
def _mcp(repl: Repl, args: str) -> CommandResult:
    statuses = list(repl.mcp.status.values())
    if not statuses:
        repl.ui.muted(
            "No MCP servers configured. Add them to .mcp.json:\n"
            '  {"mcpServers": {"name": {"command": "npx", "args": ["-y", "some-mcp-server"]}}}'
        )
        return CommandResult()
    rows = []
    for status in sorted(statuses, key=lambda s: s.name):
        detail = ", ".join(status.tools[:8]) + (" ..." if len(status.tools) > 8 else "")
        rows.append([status.name, status.state, str(len(status.tools)), status.error or detail])
    repl.ui.blank()
    repl.ui.table(["server", "state", "tools", "detail"], rows, title="MCP servers")
    repl.ui.muted("Allow a whole server without prompts: /permissions allow mcp__<server>")
    repl.ui.blank()
    return CommandResult()


@command("hooks", "Show configured hooks")
def _hooks(repl: Repl, args: str) -> CommandResult:
    rows = [[event, matcher, command] for event, matcher, command in repl.hooks.describe()]
    if not rows:
        repl.ui.muted(
            'No hooks configured. Add them under "hooks" in .scode/settings.json, '
            "in the same format Claude Code uses."
        )
        return CommandResult()
    repl.ui.blank()
    repl.ui.table(["event", "matcher", "command"], rows, title="hooks")
    repl.ui.blank()
    return CommandResult()


@command("bashes", "List background shells, or stop one", usage="[kill ID]")
def _bashes(repl: Repl, args: str) -> CommandResult:
    shells = repl.agent.ctx.shells
    parts = args.split()
    if len(parts) == 2 and parts[0] == "kill":
        shell = shells.get(parts[1])
        if shell is None:
            repl.ui.error(f"No shell {parts[1]}")
        else:
            shell.kill()
            repl.ui.success(f"Stopped {shell.id}")
        return CommandResult()
    if not shells:
        repl.ui.muted("No background shells.")
        return CommandResult()
    rows = [[s.id, s.status, s.command[:70]] for s in shells.values()]
    repl.ui.blank()
    repl.ui.table(["id", "status", "command"], rows, title="background shells")
    repl.ui.blank()
    return CommandResult()


@command("rewind", "Undo the agent's file changes from recent turns", usage="[turns]", aliases=("undo",))
def _rewind(repl: Repl, args: str) -> CommandResult:
    checkpoints = repl.agent.ctx.checkpoints
    if not checkpoints:
        repl.ui.muted("No file changes to rewind.")
        return CommandResult()
    turns = sorted({c.turn for c in checkpoints})
    try:
        count = max(1, int(args.strip() or "1"))
    except ValueError:
        repl.ui.error("Usage: /rewind [number of turns]")
        return CommandResult()
    targets = set(turns[-count:])

    restored: list[str] = []
    # Undo newest first, so a file edited twice ends at its oldest snapshot.
    for checkpoint in reversed([c for c in checkpoints if c.turn in targets]):
        try:
            if checkpoint.before is None:
                if checkpoint.path.exists():
                    checkpoint.path.unlink()
            else:
                checkpoint.path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.path.write_bytes(checkpoint.before)
        except OSError as exc:
            repl.ui.error(f"Could not restore {checkpoint.path}: {exc}")
            continue
        repl.agent.ctx.read_files.pop(str(checkpoint.path), None)
        restored.append(repl.agent.ctx.display_path(checkpoint.path))
    checkpoints[:] = [c for c in checkpoints if c.turn not in targets]

    unique = sorted(set(restored))
    repl.ui.success(f"Restored {len(unique)} file(s) from {len(targets)} turn(s): {', '.join(unique)}")
    repl.agent.remind(
        "The user rewound your file changes from the last "
        f"{len(targets)} turn(s). These files are back to their earlier contents: "
        + ", ".join(unique)
        + ". Read them again before editing."
    )
    return CommandResult()


# ------------------------------------------------------------------ project

@command("init", "Write a SCODE.md describing this project")
def _init(repl: Repl, args: str) -> CommandResult:
    target = repl.settings.workspace / C.MEMORY_FILE
    if target.exists():
        from ..ui.input import confirm

        if not confirm(repl.ui, f"{C.MEMORY_FILE} already exists. Rewrite it?"):
            return CommandResult()

    return CommandResult(
        prompt=(
            f"Analyse this codebase and write {C.MEMORY_FILE} at the workspace root. "
            "It is the instruction file future scode sessions will read, so write it "
            "for an engineer who has never seen this repo.\n\n"
            "Cover, briefly and accurately:\n"
            "- what the project is and what it does\n"
            "- how to build, run, test and lint it (take the exact commands from "
            "package.json scripts, Makefile, pyproject.toml, or CI config - do not guess)\n"
            "- the layout: which directory holds what\n"
            "- architecture and conventions a newcomer would otherwise get wrong\n\n"
            "Read the real files before writing anything. Keep it under 100 lines, "
            "state facts rather than advice, and leave out anything you could not verify. "
            "If a README or existing instruction file is already there, build on it."
        )
    )


@command("memory", "Show or edit project instruction files", usage="[edit]")
def _memory(repl: Repl, args: str) -> CommandResult:
    from ..agent.context import memory_files

    files = memory_files(repl.settings.workspace)
    if args.strip() == "edit":
        target = repl.settings.workspace / C.MEMORY_FILE
        if not target.exists():
            target.write_text(f"# {repl.settings.workspace.name}\n\n", encoding="utf-8")
        _open_in_editor(repl, target)
        repl.agent.remind(f"The user edited {target.name}; re-read it before relying on it.")
        return CommandResult()

    if not files:
        repl.ui.muted(
            f"No instruction file yet. Run /init to generate {C.MEMORY_FILE}, "
            "or use #note to add one line at a time."
        )
        return CommandResult()

    repl.ui.blank()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            repl.ui.error(f"{path}: {exc}")
            continue
        repl.ui.panel(text.strip()[:4000], title=str(path))
    repl.ui.blank()
    return CommandResult()


def _open_in_editor(repl: Repl, path: Path) -> None:
    editor = os.getenv("EDITOR") or os.getenv("VISUAL")
    if not editor:
        editor = "notepad" if sys.platform == "win32" else "nano"
    try:
        subprocess.run([editor, str(path)], check=False)
        repl.ui.success(f"Saved {path}")
    except OSError as exc:
        repl.ui.error(f"Could not open {editor}: {exc}")


# ---------------------------------------------------------------------- git

@command("diff", "Show the working tree diff", usage="[git args]")
def _diff(repl: Repl, args: str) -> CommandResult:
    extra = args.split() if args.strip() else []
    output = _run_git(repl, ["diff", "--stat", *extra])
    if output is None:
        return CommandResult()
    if not output.strip():
        repl.ui.muted("No unstaged changes.")
        return CommandResult()
    repl.ui.print(output)
    patch = _run_git(repl, ["diff", *extra]) or ""
    repl.ui.diff(patch, max_lines=200)
    return CommandResult()


@command("review", "Ask the model to review the current diff")
def _review(repl: Repl, args: str) -> CommandResult:
    target = args.strip() or "the uncommitted changes"
    return CommandResult(
        prompt=(
            f"Review {target} in this repository.\n\n"
            "Run `git diff HEAD` (and `git status`) to see what changed, then read "
            "enough surrounding code to judge each change in context.\n\n"
            "Report every real problem you find, most serious first, each with its "
            "severity and your confidence: correctness bugs, unhandled errors, security "
            "issues, breaking API changes, missing tests for new behaviour. For each, give "
            "the file and line, what goes wrong, and the concrete input or state that "
            "triggers it. Say so plainly if the diff looks fine - don't invent findings."
        )
    )


@command("commit", "Stage everything and write a commit message")
def _commit(repl: Repl, args: str) -> CommandResult:
    extra = f"\n\nThe user asked: {args.strip()}" if args.strip() else ""
    return CommandResult(
        prompt=(
            "Create a git commit for the current changes.\n\n"
            "1. Run `git status`, `git diff HEAD` and `git log --oneline -5` in parallel "
            "to see what changed and how this repo writes commit messages.\n"
            "2. Stage the relevant files. Don't stage unrelated changes, build output, "
            "or anything that looks like a secret.\n"
            "3. Commit with a message that says why the change was made, matching the "
            "style of recent commits in this repo.\n"
            "4. Run `git status` afterwards to confirm it succeeded.\n\n"
            "Do not push." + extra
        )
    )


def _run_git(repl: Repl, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repl.settings.workspace),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        repl.ui.error(f"git failed: {exc}")
        return None
    if completed.returncode != 0:
        repl.ui.error(completed.stderr.strip() or "git failed")
        return None
    return completed.stdout


# ----------------------------------------------------------------- sessions

@command("sessions", "List recent sessions in this project", aliases=("resume",))
def _sessions(repl: Repl, args: str) -> CommandResult:
    arg = args.strip()
    if arg:
        try:
            repl.resume_session(arg)
        except (FileNotFoundError, ScodeError) as exc:
            repl.ui.error(str(exc))
            return CommandResult()
        repl.ui.success(f"Resumed session {arg}")
        return CommandResult()

    sessions = list_sessions(repl.settings.workspace)
    if not sessions:
        repl.ui.muted("No saved sessions for this directory yet.")
        return CommandResult()

    rows = [[s.id[:10], s.age, str(s.message_count), s.title or "(no title)"] for s in sessions]
    repl.ui.blank()
    repl.ui.table(["id", "when", "msgs", "first message"], rows, title="sessions")
    repl.ui.muted("Resume one with /resume <id>, or start scode with --resume <id>")
    repl.ui.blank()
    return CommandResult()


@command("export", "Write the conversation to a markdown file", usage="[path]")
def _export(repl: Repl, args: str) -> CommandResult:
    from ..providers.base import text_of

    target = Path(args.strip()) if args.strip() else repl.settings.workspace / f"scode-{repl.store.id[:8]}.md"
    if not target.is_absolute():
        target = repl.settings.workspace / target

    lines = [f"# scode session {repl.store.id}", "", f"- model: {repl.settings.provider}:{repl.settings.model}", ""]
    for message in repl.agent.messages:
        role = message.get("role")
        if role == "system":
            continue
        content = text_of(message.get("content"))
        if role == "tool":
            lines.append(f"### tool result: {message.get('name', '')}\n")
            lines.append(f"```\n{content[:4000]}\n```\n")
            continue
        heading = {"user": "## User", "assistant": "## Assistant"}.get(str(role), f"## {role}")
        lines.append(heading + "\n")
        if content:
            lines.append(content + "\n")
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            lines.append(f"`{function.get('name')}({function.get('arguments', '')[:300]})`\n")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        repl.ui.error(f"Could not write {target}: {exc}")
        return CommandResult()
    repl.ui.success(f"Exported to {target}")
    return CommandResult()


# ------------------------------------------------------------------- config

@command("config", "Show the effective configuration")
def _config(repl: Repl, args: str) -> CommandResult:
    data = repl.settings.redacted()
    rows = [[key, value if isinstance(value, str) else json.dumps(value)] for key, value in sorted(data.items())]
    repl.ui.blank()
    repl.ui.table(["setting", "value"], rows, title="configuration")
    repl.ui.muted(f"User settings: {user_settings_path()}")
    repl.ui.muted(f"Project settings: {repl.settings.workspace / C.PROJECT_DIR_NAME / 'settings.json'}")
    repl.ui.blank()
    return CommandResult()


@command("theme", "Switch between the dark and light theme", usage="[dark|light]")
def _theme(repl: Repl, args: str) -> CommandResult:
    name = args.strip().lower() or ("light" if repl.settings.theme == "dark" else "dark")
    if name not in {"dark", "light"}:
        repl.ui.error("Theme must be 'dark' or 'light'")
        return CommandResult()
    repl.set_theme(name)
    save_user_setting("theme", name)
    repl.ui.success(f"Theme: {name}")
    return CommandResult()


@command("doctor", "Check the environment and connectivity")
def _doctor(repl: Repl, args: str) -> CommandResult:
    import shutil

    from ..tools.shell import detect_shell

    settings = repl.settings
    profile = settings.get_profile()
    shell = detect_shell()
    rows = [
        ["scode", C.VERSION],
        ["python", sys.version.split()[0]],
        ["shell", f"{shell.label} ({shell.executable})"],
        ["git", shutil.which("git") or "not found"],
        ["ripgrep", shutil.which("rg") or "not found (using the built-in search)"],
        ["home", str(C.home_dir())],
        ["provider", f"{profile.label} ({settings.base_url})"],
        ["model", settings.model or "(none)"],
        ["api key", mask_key(settings.api_key) if settings.api_key else ("not needed" if not profile.key_required else "NOT SET")],
        ["mcp servers", f"{len(repl.mcp.clients)} connected, {len(repl.mcp.failures())} failed"],
    ]
    repl.ui.blank()
    repl.ui.table(["check", "value"], rows, title="doctor")
    try:
        with repl.ui.status("Contacting the provider"):
            models = repl.list_models()
        repl.ui.success(f"Provider reachable - {len(models)} models listed")
        if settings.model and models and settings.model not in models:
            repl.ui.warn(f"{settings.model} is not in the list. Run /model --check.")
    except ProviderError as exc:
        repl.ui.error(str(exc))
    if profile.key_required and not settings.api_key:
        repl.ui.warn(f"No API key. {profile.key_hint()}.")
    repl.ui.blank()
    return CommandResult()
