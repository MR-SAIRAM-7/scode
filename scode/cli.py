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
from .providers.profiles import EFFORT_LEVELS
from .ui.console import UI, plain_print
from .ui.glyphs import enable_utf8_stdout

OUTPUT_FORMATS = ("text", "json", "stream-json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scode",
        description=f"{C.PRODUCT_TAGLINE}. Run with no arguments for an interactive session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("prompt", nargs="*", help="Prompt to run. Omit for interactive mode.")

    run = parser.add_argument_group("running")
    run.add_argument("-p", "--print", dest="print_mode", action="store_true",
                     help="Run one turn non-interactively and print the result")
    run.add_argument("--output-format", choices=OUTPUT_FORMATS, default="text",
                     help="Output for --print: text, json, or stream-json (one JSON event per line)")
    run.add_argument("-c", "--continue", dest="continue_last", action="store_true",
                     help="Continue the most recent session in this directory")
    run.add_argument("-r", "--resume", metavar="ID", help="Resume a session by id")
    run.add_argument("-C", "--workspace", metavar="DIR", help="Directory to work in (default: cwd)")

    model = parser.add_argument_group("model and provider")
    model.add_argument("--provider", help="Provider: nvidia, openrouter, omniroute, anthropic, openai, ... "
                                         "(see --list-providers)")
    model.add_argument("--model", "-m", help="Model id, alias (opus, sonnet), or provider:model")
    model.add_argument("--small-model", help="Cheaper model for summarising")
    model.add_argument("--base-url", help="OpenAI-compatible endpoint (for --provider custom)")
    model.add_argument("--api-key", help="API key (prefer the provider's environment variable)")
    model.add_argument("--effort", choices=EFFORT_LEVELS, help="Reasoning effort, where the model supports it")
    model.add_argument("--max-output-tokens", type=int, help="Cap on tokens per reply")
    model.add_argument("--temperature", type=float, help="Sampling temperature (0-2)")
    model.add_argument("--max-steps", "--max-turns", dest="max_steps", type=int,
                       help="Max tool-use steps per turn")
    model.add_argument("--no-native-tools", action="store_true",
                       help="Use the text tool protocol instead of function calling")
    model.add_argument("--append-system-prompt", metavar="TEXT", help="Extra system-prompt text")
    model.add_argument("--mcp-config", metavar="FILE", action="append",
                       help="Load MCP servers from a JSON file ({\"mcpServers\": {...}}). Repeatable.")

    perms = parser.add_argument_group("permissions")
    perms.add_argument("--permission-mode", choices=PERMISSION_MODES,
                       help="How tool calls are approved (default: default)")
    perms.add_argument("--allowed-tools", "--allowedTools", dest="allowed_tools", metavar="RULE",
                       action="append", default=None,
                       help="Auto-approve a tool or rule, e.g. 'Bash(git status:*)'. Repeatable.")
    perms.add_argument("--disallowed-tools", "--disallowedTools", dest="disallowed_tools", metavar="RULE",
                       action="append", default=None, help="Always refuse a tool or rule. Repeatable.")
    perms.add_argument("--dangerously-skip-permissions", action="store_true",
                       help="Run every tool without asking. Use only in a sandbox.")

    display = parser.add_argument_group("display")
    display.add_argument("--theme", choices=["dark", "light"], help="Colour theme")
    display.add_argument("--no-stream", action="store_true", help="Wait for the full reply")
    display.add_argument("--no-markdown", action="store_true", help="Print raw text, no rendering")
    display.add_argument("-q", "--quiet", action="store_true", help="Suppress decoration")
    display.add_argument("-v", "--verbose", action="store_true", help="Show reasoning and detail")
    display.add_argument("--debug", action="store_true", help="Write a debug log to ~/.scode/logs")

    info = parser.add_argument_group("information")
    info.add_argument("--version", action="store_true", help="Print the version and exit")
    info.add_argument("--list-providers", action="store_true", help="List providers and exit")
    info.add_argument("--list-models", action="store_true", help="List the provider's models and exit")
    info.add_argument("--check-models", action="store_true",
                      help="Probe which models your key can actually reach, then exit")
    info.add_argument("--sessions", action="store_true", help="List saved sessions and exit")
    info.add_argument("--doctor", action="store_true", help="Check the environment and exit")
    return parser


EPILOG = """\
examples:
  scode                                        interactive session in this directory
  scode "add retry logic to the client"        start with a first prompt
  scode -p "what does src/app.py do?"          one-shot, prints and exits
  scode -c                                     continue the last session here
  scode --provider openrouter --model sonnet   switch provider and model
  scode --model anthropic:opus                 provider:model in one flag
  scode --permission-mode plan                 research without changing anything

keys (set the one for your provider):
  NVIDIA_API_KEY  OPENROUTER_API_KEY  ANTHROPIC_API_KEY  OPENAI_API_KEY  ...
  or run /login inside scode
"""


def _load_mcp_files(paths: list[str] | None) -> dict[str, dict[str, Any]]:
    servers: dict[str, dict[str, Any]] = {}
    for raw in paths or []:
        path = Path(raw).expanduser()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Could not read MCP config {path}: {exc}") from exc
        entries = data.get("mcpServers") if isinstance(data, dict) else None
        if not isinstance(entries, dict):
            raise ConfigError(f"{path} must contain an \"mcpServers\" object")
        servers.update({str(k): dict(v) for k, v in entries.items() if isinstance(v, dict)})
    return servers


def settings_from_args(args: argparse.Namespace) -> Settings:
    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else Path.cwd()
    if not workspace.is_dir():
        raise ConfigError(f"{workspace} is not a directory")

    overrides: dict[str, Any] = {
        "provider": args.provider,
        "model": args.model,
        "small_model": args.small_model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "effort": args.effort,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "permission_mode": args.permission_mode,
        "theme": args.theme,
        "append_system_prompt": args.append_system_prompt,
        "verbose": True if args.verbose else None,
        "debug": True if args.debug else None,
        "stream": False if args.no_stream else None,
        "native_tools": False if args.no_native_tools else None,
    }
    if args.dangerously_skip_permissions:
        overrides["permission_mode"] = "bypassPermissions"
    if args.allowed_tools:
        overrides["allowed_tools"] = tuple(args.allowed_tools)
    if args.disallowed_tools:
        overrides["denied_tools"] = tuple(args.disallowed_tools)
    mcp = _load_mcp_files(args.mcp_config)
    if mcp:
        overrides["mcp_servers"] = mcp

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
        if args.list_providers:
            print(f"scode: {exc}\n", file=sys.stderr)
            return _list_providers({})
        print(f"scode: {exc}", file=sys.stderr)
        return 2

    if args.list_providers:
        return _list_providers(settings.providers, settings.provider)

    from .log import setup_logging

    log_path = setup_logging(settings.debug)
    ui = UI(settings.theme, quiet=args.quiet, markdown=not args.no_markdown)
    if log_path and not args.quiet:
        ui.muted(f"Debug log: {log_path}")

    if args.list_models:
        return _list_models(settings, ui)
    if args.check_models:
        return _check_models(settings, ui)
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
            return run_print_mode(settings, ui, prompt, output_format=args.output_format,
                                  resume=args.resume, continue_last=args.continue_last)
        return run_interactive(settings, ui, prompt, resume=args.resume, continue_last=args.continue_last)
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


def run_interactive(settings: Settings, ui: UI, prompt: str, *, resume: str | None, continue_last: bool) -> int:
    from .repl import Repl

    profile = settings.get_profile()
    if profile.key_required and not settings.api_key:
        # The REPL still starts, so /login can set a key without restarting.
        ui.warn(f"No API key for {profile.label}. {profile.key_hint()}." +
                (f" Get one: {profile.signup_url}" if profile.signup_url else ""))

    repl = Repl(settings, ui, resume_id=resume, continue_last=continue_last)
    return repl.run(prompt or None)


def _event_for(message: dict[str, Any]) -> dict[str, Any]:
    from .providers.base import text_of

    role = message.get("role")
    if role == "tool":
        return {
            "type": "tool_result",
            "tool_call_id": message.get("tool_call_id"),
            "name": message.get("name"),
            "content": text_of(message.get("content")),
            "is_error": bool(message.get("is_error")),
        }
    event: dict[str, Any] = {"type": role, "content": text_of(message.get("content"))}
    if message.get("tool_calls"):
        event["tool_calls"] = [
            {
                "id": call.get("id"),
                "name": (call.get("function") or {}).get("name"),
                "input": _json_or_raw((call.get("function") or {}).get("arguments", "")),
            }
            for call in message["tool_calls"]
        ]
    return event


def _json_or_raw(text: str) -> Any:
    try:
        return json.loads(text or "{}")
    except json.JSONDecodeError:
        return text


def run_print_mode(
    settings: Settings,
    ui: UI,
    prompt: str,
    *,
    output_format: str = "text",
    resume: str | None = None,
    continue_last: bool = False,
) -> int:
    """One turn, no prompts, machine-friendly output."""
    from .agent.context import expand_file_mentions
    from .runtime import Runtime

    settings.require_api_key()

    # Structured output (and -q) suppress live rendering; the result is printed once at the end.
    suppress = output_format != "text" or ui.quiet
    quiet_ui = UI(settings.theme, quiet=suppress, markdown=ui.markdown)
    runtime = Runtime(
        settings,
        quiet_ui,
        asker=None,
        interactive=False,
        resume_id=resume,
        continue_last=continue_last,
    )
    agent = runtime.agent

    if output_format == "stream-json":
        persist = agent.on_message

        def emit(message: dict[str, Any]) -> None:
            if persist:
                persist(message)
            plain_print(json.dumps(_event_for(message), ensure_ascii=False))

        agent.on_message = emit
        plain_print(json.dumps({
            "type": "system",
            "subtype": "init",
            "session_id": runtime.store.id,
            "provider": runtime.settings.provider,
            "model": runtime.settings.model,
            "cwd": str(runtime.settings.workspace),
            "permission_mode": runtime.permissions.mode,
            "tools": agent.registry.names(),
        }))

    content, _ = expand_file_mentions(prompt, runtime.settings.workspace)
    try:
        result = agent.run(content)
    except Interrupted:
        runtime.close()
        return 130
    finally:
        runtime.close()

    usage = runtime.usage
    summary = {
        "result": result.text,
        "reason": result.reason,
        "is_error": result.reason != "done",
        "steps": result.steps,
        "error": result.error,
        "session_id": runtime.store.id,
        "provider": runtime.settings.provider,
        "model": runtime.settings.model,
        "total_cost_usd": round(usage.cost_usd, 6) if usage.fully_priced else None,
        "usage": {
            "input_tokens": usage.input_tokens,
            "cache_read_tokens": usage.cached_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "output_tokens": usage.output_tokens,
            "requests": usage.requests,
            "tool_calls": usage.tool_calls,
        },
    }
    if output_format == "json":
        plain_print(json.dumps(summary, indent=2))
    elif output_format == "stream-json":
        plain_print(json.dumps({"type": "result", **summary}))
    elif suppress:
        if result.text:
            plain_print(result.text)
    else:
        quiet_ui.blank()

    return 0 if result.reason == "done" else 1


def _list_providers(custom: dict[str, Any], current: str = "") -> int:
    from .providers.profiles import known_provider_names, resolve_profile

    for name in known_provider_names(custom):
        try:
            profile = resolve_profile(name, custom)
        except ScodeError as exc:
            print(f"{name:11s} invalid: {exc}")
            continue
        env = profile.api_key_env[0] if profile.api_key_env else "-"
        marker = "*" if name == current else " "
        print(f"{marker}{profile.name:11s} {profile.label:36s} {env:20s} {profile.default_model or '(auto)'}")
    print("\nAny OpenAI-compatible provider listed on models.dev also works by name,")
    print('and any other endpoint can be added under "providers" in settings.json.')
    return 0


def _list_models(settings: Settings, ui: UI) -> int:
    from .providers.registry import list_catalog

    try:
        with ui.status("Fetching the model list"):
            models = list_catalog(settings)
    except ScodeError as exc:
        ui.error(str(exc))
        return 1
    for model in models:
        marker = "*" if model == settings.model else " "
        print(f"{marker} {model}")
    return 0


def _check_models(settings: Settings, ui: UI) -> int:
    """Probe each candidate model. Catalogs list far more than a key can use."""
    from .providers.registry import check_model

    profile = settings.get_profile()
    if profile.key_required and not settings.api_key:
        ui.error(f"No API key for {profile.label}. {profile.key_hint()}.")
        return 2

    candidates = list(profile.models)
    for extra in (settings.model, settings.small_model):
        if extra and extra not in candidates:
            candidates.append(extra)

    labels = {"ok": "works", "unavailable": "not on your key", "retired": "retired",
              "timeout": "no response", "error": "error"}
    rows: list[list[str]] = []
    working: list[str] = []
    with ui.status("Probing models") as status:
        for name in candidates:
            if status is not None:
                status.update(f"Probing {name}")
            state, detail = check_model(settings, name)
            if state == "ok":
                working.append(name)
            current = " (current)" if name == settings.model else ""
            rows.append([name + current, labels.get(state, state), detail])

    ui.table(["model", "status", "detail"], rows, title=f"{profile.label} availability")
    if not working:
        ui.error("No models responded. Check your key and network.")
        return 1
    ui.success(f"{len(working)} of {len(candidates)} models responded.")
    if settings.model not in working:
        ui.warn(f"{settings.model} is not responding. Try: scode --model {working[0]}")
        return 1
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

    profile = settings.get_profile()
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
            ["provider", f"{profile.label} ({settings.base_url})"],
            ["model", settings.model or "(auto)"],
            ["api key", "set" if settings.api_key else ("not needed" if not profile.key_required else "NOT SET")],
            ["mcp servers", str(len(settings.mcp_servers))],
        ],
        title="doctor",
    )
    try:
        with ui.status("Contacting the provider"):
            models = list_catalog(settings)
        ui.success(f"Provider reachable - {len(models)} models listed")
        if settings.model and models and settings.model not in models:
            ui.warn(f"{settings.model} is not in the list; run scode --check-models")
    except ScodeError as exc:
        ui.error(str(exc))
        return 1
    if profile.key_required and not settings.api_key:
        ui.warn(f"No API key. {profile.key_hint()}.")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
