import json
import unittest
from unittest import mock

from scode.providers.openai_compatible import OpenAICompatibleProvider


class _FakeResponse:
    def __init__(self, payload=None, lines=None):
        self._payload = payload or {}
        self._lines = lines or []

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_lines(self):
        for line in self._lines:
            yield line

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class ProviderTests(unittest.TestCase):
    def test_complete_sends_expected_payload(self) -> None:
        provider = OpenAICompatibleProvider(name="nvidia", api_key="k", base_url="https://example.com/v1")
        captured = {}

        def fake_post(url, headers, json, timeout, **kwargs):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["kwargs"] = kwargs
            return _FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]})

        with mock.patch("scode.providers.openai_compatible.requests.post", side_effect=fake_post):
            message = provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="moonshotai/kimi-k3",
                max_output_tokens=60000,
                temperature=1.0,
                seed=0,
                reasoning_effort="max",
            )

        self.assertEqual(message["content"], "ok")
        self.assertEqual(captured["url"], "https://example.com/v1/chat/completions")
        self.assertEqual(captured["headers"]["Accept"], "application/json")
        self.assertEqual(captured["json"]["stream"], False)
        self.assertEqual(captured["json"]["max_tokens"], 60000)
        self.assertEqual(captured["json"]["reasoning_effort"], "max")

    def test_stream_text_reads_sse_events(self) -> None:
        provider = OpenAICompatibleProvider(name="nvidia", api_key="k", base_url="https://example.com/v1")
        lines = [
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}",
            b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}",
            b"data: [DONE]",
        ]

        with mock.patch("scode.providers.openai_compatible.requests.post", return_value=_FakeResponse(lines=lines)):
            output = "".join(
                provider.stream_text(
                    messages=[{"role": "user", "content": "hello"}],
                    model="moonshotai/kimi-k3",
                    max_output_tokens=60000,
                    temperature=1.0,
                    seed=0,
                    reasoning_effort="max",
                )
            )

        self.assertEqual(output, "Hello world")
