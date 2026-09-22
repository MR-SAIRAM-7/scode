"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import constants as C
from .config import PERMISSION_MODES, Settings, load_settings
from .errors import ConfigError, Interrupted, ScodeError
from .ui.console import UI, plain_print
from .ui.glyphs import enable_utf8_stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scode",
        description=f"{C.PRODUCT_TAGLINE}. Run with no arguments for an interactive session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("prompt", nargs="*", help="Prompt to run. Omit for interactive mode.")

    run = parser.add_argument_group("running")
    run.add_argument(
        "-p", "--print", dest="print_mode", action="store_true",
        help="Run one turn non-interactively and print the result",
    )
    run.add_argument(
        "--output-format", choices=["text", "json"], default="text",
        help="Output format for --print (default: text)",
    )
    run.add_argument("-c", "--continue", dest="continue_last", action="store_true",
                     help="Continue the most recent session in this directory")
    run.add_argument("-r", "--resume", metavar="ID", help="Resume a session by id")
    run.add_argument("-C", "--workspace", metavar="DIR", help="Directory to work in (default: cwd)")

    model = parser.add_argument_group("model")
    model.add_argument("--model", "-m", help=f"Model id (default: {C.DEFAULT_MODEL})")
    model.add_argument("--small-model", help="Cheaper model used for summarisation")
    model.add_argument("--provider", help="Provider name (default: nvidia)")
    model.add_argument("--base-url", help="OpenAI-compatible endpoint")
    model.add_argument("--api-key", help="API key (prefer the NVIDIA_API_KEY env var)")
    model.add_argument("--max-output-tokens", type=int, help="Cap on tokens per reply")
    model.add_argument("--temperature", type=float, help="Sampling temperature (0-2)")
    model.add_argument("--max-steps", type=int, help="Max tool-use steps per turn")
    model.add_argument("--no-native-tools", action="store_true",
                       help="Use the text tool protocol instead of function calling")

    perms = parser.add_argument_group("permissions")
    perms.add_argument("--permission-mode", choices=PERMISSION_MODES,
                       help="How tool calls are approved (default: default)")
    perms.add_argument("--allowed-tools", metavar="RULE", action="append", default=None,
                       help="Auto-approve a tool or rule, e.g. 'Bash(git status:*)'. Repeatable.")
    perms.add_argument("--disallowed-tools", metavar="RULE", action="append", default=None,
                       help="Always refuse a tool or rule. Repeatable.")
    perms.add_argument("--dangerously-skip-permissions", action="store_true",
                       help="Run every tool without asking. Use only in a sandbox.")

    display = parser.add_argument_group("display")
    display.add_argument("--theme", choices=["dark", "light"], help="Colour theme")
    display.add_argument("--no-stream", action="store_true", help="Wait for the full reply")
    display.add_argument("--no-markdown", action="store_true", help="Print raw text, no rendering")
    display.add_argument("-q", "--quiet", action="store_true", help="Suppress decoration")
    display.add_argument("-v", "--verbose", action="store_true", help="Show reasoning and detail")

    info = parser.add_argument_group("information")
    info.add_argument("--version", action="store_true", help="Print the version and exit")
    info.add_argument("--list-models", action="store_true", help="List provider models and exit")
    info.add_argument("--sessions", action="store_true", help="List saved sessions and exit")
    info.add_argument("--doctor", action="store_true", help="Check the environment and exit")
    return parser


EPILOG = """\
examples:
  scode                                  start an interactive session
  scode "add retry logic to the client"  start with a first prompt
  scode -p "what does src/app.py do?"    one-shot, prints and exits
  scode -c                               continue the last session here
  scode --permission-mode plan           research without changing anything
  scode --allowed-tools 'Bash(npm test:*)' -p "run the tests"

setup:
  export NVIDIA_API_KEY=nvapi-...        free keys at https://build.nvidia.com
"""


def settings_from_args(args: argparse.Namespace) -> Settings:
    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else Path.cwd()
    if not workspace.is_dir():
        raise ConfigError(f"{workspace} is not a directory")

    overrides: dict[str, Any] = {
        "model": args.model,
        "small_model": args.small_model,
        "provider": args.provider,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "permission_mode": args.permission_mode,
        "theme": args.theme,
        "verbose": True if args.verbose else None,
        "stream": False if args.no_stream else None,
        "native_tools": False if args.no_native_tools else None,
    }
    if args.dangerously_skip_permissions:
        overrides["permission_mode"] = "bypassPermissions"
    if args.allowed_tools:
        overrides["allowed_tools"] = tuple(args.allowed_tools)
    if args.disallowed_tools:
        overrides["denied_tools"] = tuple(args.disallowed_tools)

    return load_settings(workspace, overrides)


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"scode {C.VERSION}")
        return 0

    try:
        settings = settings_from_args(args)
    except ConfigError as exc:
        print(f"scode: {exc}", file=sys.stderr)
        return 2

    ui = UI(settings.theme, quiet=args.quiet, markdown=not args.no_markdown)

    if args.list_models:
        return _list_models(settings, ui)
    if args.sessions:
        return _list_sessions(settings, ui)
    if args.doctor:
        return _doctor(settings, ui)

    prompt = " ".join(args.prompt).strip()
    # Only -p reads a piped prompt; interactively, stdin is the command stream.
    if not prompt and args.print_mode:
        prompt = _stdin_prompt()

    try:
        if args.print_mode:
            if not prompt:
                print("scode: --print needs a prompt (argument or stdin)", file=sys.stderr)
                return 2
            return run_print_mode(settings, ui, prompt, output_format=args.output_format)
        return run_interactive(
            settings, ui, prompt, resume=args.resume, continue_last=args.continue_last
        )
    except ConfigError as exc:
        ui.error(str(exc))
        return 2
    except ScodeError as exc:
        ui.error(str(exc))
        return 1
    except KeyboardInterrupt:
        ui.blank()
        ui.muted("Interrupted.")
        return 130


def _stdin_prompt() -> str:
    """Accept a piped prompt: `echo "..." | scode -p`."""
    if sys.stdin is None or sys.stdin.isatty():
        return ""
    try:
        return sys.stdin.read().strip()
    except (OSError, UnicodeDecodeError):
        return ""


def run_interactive(
    settings: Settings,
    ui: UI,
    prompt: str,
    *,
    resume: str | None,
    continue_last: bool,
) -> int:
    from .repl import Repl

    if not settings.api_key:
        # The REPL still starts, so /login can set a key without restarting.
        ui.warn(
            "No API key set. Run /login inside scode, or export NVIDIA_API_KEY "
            "(free keys: https://build.nvidia.com)."
        )

    repl = Repl(settings, ui, resume_id=resume, continue_last=continue_last)
    return repl.run(prompt or None)


def run_print_mode(
    settings: Settings,
    ui: UI,
    prompt: str,
    *,
    output_format: str = "text",
) -> int:
    """One turn, no prompts, machine-friendly output."""
    from .agent.context import expand_file_mentions
    from .agent.loop import Agent
    from .permissions import PermissionEngine
    from .providers.registry import get_provider
    from .session.store import SessionStore
    from .tools import ToolContext, build_registry
    from .usage import Usage

    settings.require_api_key()

    # JSON output (and -q) suppress the live rendering; the result is printed once at the end.
    suppress = output_format == "json" or ui.quiet
    quiet_ui = UI(settings.theme, quiet=suppress, markdown=ui.markdown)
    permissions = PermissionEngine(settings, asker=None)
    provider = get_provider(settings)
    usage = Usage()

    store = SessionStore(settings.workspace, model=settings.model).open()
    registry = build_registry()
    ctx = ToolContext(
        workspace=settings.workspace,
        settings=settings,
        permissions=permissions,
    )
    agent = Agent(
        provider=provider,
        registry=registry,
        ctx=ctx,
        ui=quiet_ui,
        settings=settings,
        usage=usage,
        on_message=store.append,
    )

    expanded, _ = expand_file_mentions(prompt, settings.workspace)
    try:
        result = agent.run(expanded)
    except Interrupted:
        store.close()
        return 130
    finally:
        store.close()

    if output_format == "json":
        payload = {
            "result": result.text,
            "reason": result.reason,
            "steps": result.steps,
            "error": result.error,
            "session_id": store.id,
            "model": settings.model,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "requests": usage.requests,
                "tool_calls": usage.tool_calls,
            },
        }
        plain_print(json.dumps(payload, indent=2))
    elif suppress:
        if result.text:
            plain_print(result.text)
    else:
        quiet_ui.blank()

    return 0 if result.reason == "done" else 1


def _list_models(settings: Settings, ui: UI) -> int:
    from .providers.registry import list_catalog

    try:
        with ui.status("Fetching the model catalog"):
            models = list_catalog(settings)
    except ScodeError as exc:
        ui.error(str(exc))
        return 1
    for model in models:
        marker = "*" if model == settings.model else " "
        print(f"{marker} {model}")
    return 0


def _list_sessions(settings: Settings, ui: UI) -> int:
    from .session.store import list_sessions

    sessions = list_sessions(settings.workspace)
    if not sessions:
        ui.muted("No saved sessions for this directory.")
        return 0
    ui.table(
        ["id", "when", "msgs", "first message"],
        [[s.id[:10], s.age, str(s.message_count), s.title or "(no title)"] for s in sessions],
        title=f"sessions in {settings.workspace}",
    )
    return 0


def _doctor(settings: Settings, ui: UI) -> int:
    import shutil

    from .providers.registry import list_catalog
    from .tools.shell import detect_shell

    shell = detect_shell()
    ui.table(
        ["check", "value"],
        [
            ["scode", C.VERSION],
            ["python", sys.version.split()[0]],
            ["shell", f"{shell.label} ({shell.executable})"],
            ["git", shutil.which("git") or "not found"],
            ["ripgrep", shutil.which("rg") or "not found (built-in search will be used)"],
            ["workspace", str(settings.workspace)],
            ["home", str(C.home_dir())],
            ["model", settings.model],
            ["base url", settings.base_url],
            ["api key", "set" if settings.api_key else "NOT SET"],
        ],
        title="doctor",
    )
    try:
        with ui.status("Contacting the provider"):
            models = list_catalog(settings)
        ui.success(f"Provider reachable — {len(models)} models")
        if settings.model not in models:
            ui.warn(f"{settings.model} is not in the catalog; run scode --list-models")
    except ScodeError as exc:
        ui.error(str(exc))
        return 1
    if not settings.api_key:
        ui.warn("No API key. Export NVIDIA_API_KEY or run /login in an interactive session.")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
