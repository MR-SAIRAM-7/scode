from __future__ import annotations

import argparse
from dataclasses import replace

from .agent import Agent
from .config import AppConfig
from .providers.registry import get_provider
from .tools import build_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scode",
        description="Agentic coding CLI with NVIDIA and Kimi K3 provider support.",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt for the coding agent")
    parser.add_argument("--provider", choices=["nvidia", "kimi-k3"], help="Model provider")
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--max-output-tokens", type=int, help="Maximum output tokens")
    parser.add_argument("--workspace", help="Workspace root for tools")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.prompt:
        parser.print_help()
        return 0

    config = AppConfig.from_env()
    if args.provider:
        config = replace(config, provider=args.provider)
    if args.model:
        config = replace(config, model=args.model)
    if args.max_output_tokens is not None:
        if args.max_output_tokens <= 0:
            raise SystemExit("--max-output-tokens must be positive")
        config = replace(config, max_output_tokens=args.max_output_tokens)
    if args.workspace:
        from pathlib import Path

        config = replace(config, workspace=Path(args.workspace).resolve())

    try:
        provider = get_provider(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    tools = build_tools(config.workspace)
    agent = Agent(
        provider,
        model=config.model,
        max_output_tokens=config.max_output_tokens,
        tools=tools,
    )

    print(agent.run(args.prompt))
    return 0
