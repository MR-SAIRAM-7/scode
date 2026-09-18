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
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--seed", type=int, help="Sampling seed")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "max"], help="Reasoning effort level")
    parser.add_argument("--stream", action="store_true", help="Stream model output for direct chat mode")
    parser.add_argument("--direct", action="store_true", help="Use direct chat completion instead of agent mode")
    parser.add_argument("--image-url", help="Optional image URL for multimodal direct chat")
    parser.add_argument("--workspace", help="Workspace root for tools")
    return parser


def _build_direct_messages(prompt: str, image_url: str | None) -> list[dict[str, object]]:
    if not image_url:
        return [{"role": "user", "content": prompt}]

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]


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
    if args.temperature is not None:
        if args.temperature < 0:
            raise SystemExit("--temperature must be non-negative")
        config = replace(config, temperature=args.temperature)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.reasoning_effort:
        config = replace(config, reasoning_effort=args.reasoning_effort)
    if args.workspace:
        from pathlib import Path

        config = replace(config, workspace=Path(args.workspace).resolve())

    try:
        provider = get_provider(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.direct:
        messages = _build_direct_messages(args.prompt, args.image_url)
        if args.stream:
            for chunk in provider.stream_text(
                messages=messages,
                model=config.model,
                max_output_tokens=config.max_output_tokens,
                temperature=config.temperature,
                seed=config.seed,
                reasoning_effort=config.reasoning_effort,
            ):
                print(chunk, end="", flush=True)
            print()
        else:
            message = provider.complete(
                messages=messages,
                model=config.model,
                max_output_tokens=config.max_output_tokens,
                temperature=config.temperature,
                seed=config.seed,
                reasoning_effort=config.reasoning_effort,
            )
            print(message.get("content", ""))
        return 0

    tools = build_tools(config.workspace)
    agent = Agent(
        provider,
        model=config.model,
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
        seed=config.seed,
        reasoning_effort=config.reasoning_effort,
        tools=tools,
    )
    print(agent.run(args.prompt))
    return 0
