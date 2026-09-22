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
from ..config import PERMISSION_MODES, mask_key, save_user_setting, user_settings_path
from ..errors import ProviderError, ScodeError
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
    repl.ui.blank()
    repl.ui.muted(
        "Input shortcuts:  @path  attach a file   !command  run a shell command   "
        "#note  save a project note"
    )
    repl.ui.muted(
        "Keys:  Esc+Enter or Ctrl+J  newline   Ctrl+C  interrupt   Ctrl+D  exit   "
        "Up/Down  history"
    )
    return CommandResult()


@command("exit", "Exit scode", aliases=("quit", "q"))
def _exit(repl: Repl, args: str) -> CommandResult:
    return CommandResult(exit=True)


@command("clear", "Start a fresh conversation, keeping settings", aliases=("reset",))
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


@command("status", "Show session, model and context status")
def _status(repl: Repl, args: str) -> CommandResult:
    settings = repl.settings
    usage = repl.usage
    window = settings.context_window()
    rows = [
        ["session", repl.store.id],
        ["model", settings.model],
        ["provider", f"{settings.provider} ({settings.base_url})"],
        ["api key", mask_key(settings.api_key) if settings.api_key else "not set"],
        ["workspace", str(settings.workspace)],
        ["mode", repl.permissions.mode],
        ["tools", str(len(repl.agent.registry))],
        ["native tools", "yes" if repl.agent._native_tools else "no (text protocol)"],
        ["messages", str(len(repl.agent.messages))],
        [
            "context",
            f"{format_tokens(usage.context_tokens)} / {format_tokens(window)} "
            f"({usage.context_percent(window):.0f}%)",
        ],
        ["turns", str(usage.requests)],
        ["tool calls", str(usage.tool_calls)],
        ["tokens", f"in {format_tokens(usage.input_tokens)} · out {format_tokens(usage.output_tokens)}"],
        ["elapsed", format_duration(usage.wall_seconds)],
    ]
    repl.ui.blank()
    repl.ui.table(["", ""], rows, title="status")
    repl.ui.blank()
    return CommandResult()


@command("cost", "Show token usage for this session", aliases=("usage", "tokens"))
def _cost(repl: Repl, args: str) -> CommandResult:
    usage = repl.usage
    repl.ui.blank()
    repl.ui.table(
        ["", ""],
        [
            ["input tokens", format_tokens(usage.input_tokens)],
            ["output tokens", format_tokens(usage.output_tokens)],
            ["cached", format_tokens(usage.cached_tokens)],
            ["total", format_tokens(usage.total_tokens)],
            ["api requests", str(usage.requests)],
            ["tool calls", str(usage.tool_calls)],
            ["api time", format_duration(usage.api_seconds)],
            ["wall time", format_duration(usage.wall_seconds)],
        ],
        title="usage",
    )
    repl.ui.muted("NVIDIA NIM free tier: usage is metered in credits, not dollars.")
    repl.ui.blank()
    return CommandResult()


# -------------------------------------------------------------------- model

@command("model", "Show or change the model", usage="[name|--check|--all|--search TEXT]")
def _model(repl: Repl, args: str) -> CommandResult:
    arg = args.strip()

    if arg in {"--check", "-c"}:
        return _model_check(repl)

    if arg in {"--all", "--list", "-a"} or arg.startswith("--search"):
        query = arg.partition(" ")[2].strip().lower() if arg.startswith("--search") else ""
        try:
            with repl.ui.status("Fetching the model catalog"):
                models = repl.list_models()
        except ProviderError as exc:
            repl.ui.error(str(exc))
            return CommandResult()
        if query:
            models = [m for m in models if query in m.lower()]
        repl.ui.blank()
        repl.ui.table(
            ["model"],
            [[m] for m in models],
            title=f"{len(models)} models at {repl.settings.base_url}",
        )
        repl.ui.muted("Set one with /model <name>")
        repl.ui.warn(
            "This is the public catalog; most entries are not granted to any one key. "
            "Run /model --check to see which ones actually answer."
        )
        repl.ui.blank()
        return CommandResult()

    if not arg:
        rows = []
        for name, info in C.MODEL_CATALOG.items():
            marker = " (current)" if name == repl.settings.model else ""
            rows.append([name + marker, str(info["label"]), str(info["note"])])
        repl.ui.blank()
        repl.ui.table(["model", "name", "notes"], rows, title="verified models")
        repl.ui.muted(
            "Change with /model <name>  ·  /model --check probes your key  ·  "
            "/model --all lists the full catalog"
        )
        repl.ui.blank()
        return CommandResult()

    note = C.KNOWN_UNAVAILABLE.get(arg)
    if note:
        repl.ui.warn(f"{arg} is known not to work: {note}.")
    repl.set_model(arg)
    repl.ui.success(f"Model set to {arg}")
    return CommandResult()


def _model_check(repl: Repl) -> CommandResult:
    """Probe each candidate model so the user sees what their key can reach."""
    from ..providers.openai_compatible import check_model

    if not repl.settings.api_key:
        repl.ui.error("No API key set. Run /login first.")
        return CommandResult()

    candidates = list(C.MODEL_CATALOG)
    for extra in (repl.settings.model, repl.settings.small_model):
        if extra and extra not in candidates:
            candidates.append(extra)

    marks = {
        "ok": ("scode.success", "works"),
        "unavailable": ("scode.warn", "not on your key"),
        "retired": ("scode.warn", "retired"),
        "timeout": ("scode.error", "no response"),
        "error": ("scode.error", "error"),
    }

    rows: list[list[str]] = []
    working: list[str] = []
    repl.ui.blank()
    with repl.ui.status("Probing models") as status:
        for name in candidates:
            if status is not None:
                status.update(f"Probing {name}")
            state, detail = check_model(
                repl.settings.base_url, repl.settings.api_key, name
            )
            if state == "ok":
                working.append(name)
            label = marks.get(state, ("scode.muted", state))[1]
            current = " (current)" if name == repl.settings.model else ""
            rows.append([name + current, label, detail])

    repl.ui.table(["model", "status", "detail"], rows, title="model availability")
    if working:
        repl.ui.success(f"{len(working)} of {len(candidates)} models responded.")
        if repl.settings.model not in working:
            repl.ui.warn(
                f"Your current model ({repl.settings.model}) is not responding. "
                f"Switch with: /model {working[0]}"
            )
    else:
        repl.ui.error("No models responded. Check your key and network.")
    repl.ui.blank()
    return CommandResult()


@command("login", "Store your NVIDIA API key")
def _login(repl: Repl, args: str) -> CommandResult:
    from prompt_toolkit import prompt as ptk_prompt

    repl.ui.muted("Get a free key at https://build.nvidia.com — it starts with nvapi-")
    key = args.strip()
    if not key:
        try:
            key = ptk_prompt("NVIDIA API key: ", is_password=True).strip()
        except (EOFError, KeyboardInterrupt):
            return CommandResult()
    if not key:
        repl.ui.warn("No key entered.")
        return CommandResult()

    path = save_user_setting("api_key", key)
    repl.set_api_key(key)
    repl.ui.success(f"Key saved to {path}")
    return CommandResult()


@command("logout", "Forget the stored API key")
def _logout(repl: Repl, args: str) -> CommandResult:
    path = user_settings_path()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.pop("api_key", None)
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except (OSError, json.JSONDecodeError) as exc:
            repl.ui.error(f"Could not update {path}: {exc}")
            return CommandResult()
    repl.ui.success("Stored key removed. NVIDIA_API_KEY in your environment still applies.")
    return CommandResult()


# -------------------------------------------------------------- permissions

@command(
    "mode",
    "Show or set the permission mode",
    aliases=("permissions",),
    usage="[default|acceptEdits|plan|bypassPermissions]",
)
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
        repl.ui.muted(repl.permissions.describe())
        repl.ui.blank()
        return CommandResult()

    match = next((m for m in PERMISSION_MODES if m.lower() == arg.lower()), None)
    if match is None:
        repl.ui.error(f"Unknown mode {arg!r}. Choose from: {', '.join(PERMISSION_MODES)}")
        return CommandResult()

    if match == "bypassPermissions":
        from ..ui.input import confirm

        repl.ui.warn(
            "bypassPermissions runs every command and edit without asking, "
            "including destructive ones."
        )
        if not confirm(repl.ui, "Enable it?"):
            return CommandResult()

    repl.set_mode(match)
    repl.ui.success(f"Permission mode: {match}")
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


@command("agents", "List the subagent types the Task tool can launch")
def _agents(repl: Repl, args: str) -> CommandResult:
    from ..tools.task import SUBAGENT_TYPES

    repl.ui.blank()
    repl.ui.table(
        ["subagent", "what it does"],
        [[name, " ".join(desc.split())] for name, desc in SUBAGENT_TYPES.items()],
        title="subagents",
    )
    repl.ui.blank()
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
            "package.json scripts, Makefile, pyproject.toml, or CI config — do not guess)\n"
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
            "Report only real problems, most serious first: correctness bugs, "
            "unhandled errors, security issues, breaking API changes, missing tests "
            "for new behaviour. For each one give the file and line, what goes wrong, "
            "and the concrete input or state that triggers it. Say so plainly if the "
            "diff looks fine — do not invent findings to fill a list."
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
            "2. Stage the relevant files. Do not stage unrelated changes, build output, "
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

    rows = [
        [s.id[:10], s.age, str(s.message_count), s.title or "(no title)"]
        for s in sessions
    ]
    repl.ui.blank()
    repl.ui.table(["id", "when", "msgs", "first message"], rows, title="sessions")
    repl.ui.muted("Resume one with /resume <id>, or start scode with --resume <id>")
    repl.ui.blank()
    return CommandResult()


@command("export", "Write the conversation to a markdown file", usage="[path]")
def _export(repl: Repl, args: str) -> CommandResult:
    target = Path(args.strip()) if args.strip() else repl.settings.workspace / f"scode-{repl.store.id[:8]}.md"
    if not target.is_absolute():
        target = repl.settings.workspace / target

    lines = [f"# scode session {repl.store.id}", "", f"- model: {repl.settings.model}", ""]
    for message in repl.agent.messages:
        role = message.get("role")
        if role == "system":
            continue
        content = message.get("content") or ""
        if role == "tool":
            lines.append(f"### tool result: {message.get('name', '')}\n")
            lines.append(f"```\n{str(content)[:4000]}\n```\n")
            continue
        heading = {"user": "## User", "assistant": "## Assistant"}.get(str(role), f"## {role}")
        lines.append(heading + "\n")
        if content:
            lines.append(str(content) + "\n")
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
    rows = [[key, json.dumps(value) if not isinstance(value, str) else value]
            for key, value in sorted(data.items())]
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
    rows: list[list[str]] = []

    rows.append(["python", sys.version.split()[0]])
    rows.append(["scode", C.VERSION])
    shell = detect_shell()
    rows.append(["shell", f"{shell.label} ({shell.executable})"])
    rows.append(["git", shutil.which("git") or "not found"])
    rows.append(["ripgrep", shutil.which("rg") or "not found (using the built-in search)"])
    rows.append(["unicode", "yes" if repl.ui.console.options.encoding != "ascii" else "ascii fallback"])
    rows.append(["home", str(C.home_dir())])
    rows.append(["api key", mask_key(settings.api_key) if settings.api_key else "NOT SET"])
    rows.append(["base url", settings.base_url])

    repl.ui.blank()
    repl.ui.table(["check", "value"], rows, title="doctor")

    try:
        with repl.ui.status("Contacting the provider"):
            models = repl.list_models()
        repl.ui.success(f"Provider reachable — {len(models)} models available")
        if settings.model not in models:
            repl.ui.warn(
                f"{settings.model} is not in the catalog. Run /model --all to see what is."
            )
    except ProviderError as exc:
        repl.ui.error(str(exc))

    if not settings.api_key:
        repl.ui.warn("No API key set. Run /login, or export NVIDIA_API_KEY.")
    repl.ui.blank()
    return CommandResult()
