import json
import unittest

from scode.agent import Agent


class _FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, messages, model, max_output_tokens, temperature, seed, reasoning_effort):
        self.calls += 1
        if self.calls == 1:
            return {"content": json.dumps({"type": "tool", "name": "echo", "arguments": {"text": "ok"}})}
        return {"content": json.dumps({"type": "final", "content": "done"})}


class AgentTests(unittest.TestCase):
    def test_agent_executes_tool_then_finishes(self) -> None:
        provider = _FakeProvider()

        def echo(text: str) -> str:
            return text

        agent = Agent(
            provider,
            model="test-model",
            max_output_tokens=60000,
            temperature=1.0,
            seed=0,
            reasoning_effort="max",
            tools={"echo": type("T", (), {"handler": staticmethod(echo)})()},
        )

        result = agent.run("hello")

        self.assertEqual(result, "done")
        self.assertEqual(provider.calls, 2)
