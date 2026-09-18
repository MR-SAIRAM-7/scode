from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AgentTaskResult:
    index: int
    prompt: str
    output: str


def run_parallel_tasks(
    prompts: list[str],
    runner: Callable[[str], str],
    *,
    max_workers: int,
) -> list[AgentTaskResult]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    if not prompts:
        return []

    bounded_workers = min(max_workers, len(prompts))
    results: list[AgentTaskResult | None] = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=bounded_workers) as executor:
        futures = {executor.submit(runner, prompt): (index, prompt) for index, prompt in enumerate(prompts)}
        for future in as_completed(futures):
            index, prompt = futures[future]
            try:
                output = future.result()
            except Exception as exc:  # pragma: no cover
                output = f"Agent failed: {exc}"
            results[index] = AgentTaskResult(index=index, prompt=prompt, output=output)

    return [result for result in results if result is not None]
