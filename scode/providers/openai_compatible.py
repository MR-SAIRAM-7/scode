from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import json
from typing import Any

import requests


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    name: str
    api_key: str
    base_url: str

    def _make_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        max_output_tokens: int,
        temperature: float,
        seed: int,
        reasoning_effort: str,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "seed": seed,
            "stream": stream,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
        }

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        max_output_tokens: int,
        temperature: float,
        seed: int,
        reasoning_effort: str,
    ) -> dict[str, Any]:
        payload = self._make_payload(
            messages=messages,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            stream=False,
        )
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Accept": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response format from provider '{self.name}'") from exc

    def stream_text(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        max_output_tokens: int,
        temperature: float,
        seed: int,
        reasoning_effort: str,
    ) -> Iterator[str]:
        payload = self._make_payload(
            messages=messages,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            stream=True,
        )
        with requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Accept": "text/event-stream",
            },
            json=payload,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data:"):
                    continue
                data = decoded[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    delta = event["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if delta:
                    yield str(delta)
