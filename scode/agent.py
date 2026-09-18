from __future__ import annotations

import json
from typing import Any

from .providers.base import ChatProvider
from .tools import Tool


SYSTEM_PROMPT = (
    "You are a production coding agent. "
    "Respond with JSON only. "
    "Use {'type':'tool','name':'<tool>','arguments':{...}} to call tools, "
    "or {'type':'final','content':'...'} when done."
)


class Agent:
    def __init__(
        self,
        provider: ChatProvider,
        *,
        model: str,
        max_output_tokens: int,
        tools: dict[str, Tool],
        max_steps: int = 12,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.tools = tools
        self.max_steps = max_steps

    def run(self, user_prompt: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for _ in range(self.max_steps):
            message = self.provider.complete(
                messages=messages,
                model=self.model,
                max_output_tokens=self.max_output_tokens,
            )
            content = message.get("content", "")
            messages.append({"role": "assistant", "content": content})

            action = self._parse_action(content)
            if action["type"] == "final":
                return str(action["content"])

            if action["type"] == "tool":
                tool_name = action["name"]
                tool = self.tools.get(tool_name)
                if tool is None:
                    tool_result = f"Unknown tool: {tool_name}"
                else:
                    try:
                        tool_result = tool.handler(**action.get("arguments", {}))
                    except Exception as exc:  # pragma: no cover
                        tool_result = f"Tool error: {exc}"

                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps({"tool_result": tool_result}),
                    }
                )
                continue

            raise RuntimeError("Invalid agent action")

        raise RuntimeError("Agent exceeded max steps without final answer")

    @staticmethod
    def _parse_action(content: str) -> dict[str, Any]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Model response must be JSON") from exc

        action_type = payload.get("type")
        if action_type == "final" and "content" in payload:
            return payload

        if action_type == "tool" and "name" in payload:
            payload.setdefault("arguments", {})
            return payload

        raise RuntimeError("Unsupported action format")
