from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib import request


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    name: str
    api_key: str
    base_url: str

    def complete(self, *, messages: list[dict[str, Any]], model: str, max_output_tokens: int) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
        }

        req = request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
        )

        with request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))

        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response format from provider '{self.name}'") from exc
