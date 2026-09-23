"""A local stand-in for POST /v1/messages that streams Anthropic's SSE format.

Each scripted turn is a list of content blocks; the server replays them as
message_start / content_block_* / message_delta / message_stop events and
records every request body so tests can inspect the wire format.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class AnthropicMock:
    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                model = {"id": "claude-opus-5", "type": "model", "display_name": "Opus",
                         "created_at": "2026-07-24T00:00:00Z"}
                path = self.path.split("?", 1)[0]
                if path == "/v1/models":
                    self._json(200, {"data": [model], "has_more": False,
                                     "first_id": model["id"], "last_id": model["id"]})
                elif path == f"/v1/models/{model['id']}":
                    self._json(200, model)
                else:
                    self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": "nope"}})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                mock.requests.append(body)
                mock.headers.append({k.lower(): v for k, v in self.headers.items()})
                turn = mock.turns.pop(0) if mock.turns else {"blocks": [{"type": "text", "text": "(done)"}]}
                if turn.get("error"):
                    status, message = turn["error"]
                    self._json(status, {"type": "error", "error": {"type": "invalid_request_error", "message": message}})
                    return
                self._stream(turn, body.get("model", "claude-opus-5"))

            def _json(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _event(self, name: str, payload: dict[str, Any]) -> None:
                self.wfile.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            def _stream(self, turn: dict[str, Any], model: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                usage = turn.get("usage", {})
                self._event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": "msg_mock", "type": "message", "role": "assistant", "model": model,
                        "content": [], "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": usage.get("input", 100), "output_tokens": 1,
                                  "cache_read_input_tokens": usage.get("cache_read", 0),
                                  "cache_creation_input_tokens": usage.get("cache_write", 0)},
                    },
                })
                for index, block in enumerate(turn["blocks"]):
                    kind = block["type"]
                    if kind == "text":
                        self._event("content_block_start", {"type": "content_block_start", "index": index,
                                                            "content_block": {"type": "text", "text": ""}})
                        self._event("content_block_delta", {"type": "content_block_delta", "index": index,
                                                            "delta": {"type": "text_delta", "text": block["text"]}})
                    elif kind == "thinking":
                        self._event("content_block_start", {"type": "content_block_start", "index": index,
                                                            "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
                        self._event("content_block_delta", {"type": "content_block_delta", "index": index,
                                                            "delta": {"type": "thinking_delta", "thinking": block["thinking"]}})
                        self._event("content_block_delta", {"type": "content_block_delta", "index": index,
                                                            "delta": {"type": "signature_delta", "signature": block["signature"]}})
                    elif kind == "tool_use":
                        self._event("content_block_start", {"type": "content_block_start", "index": index,
                                                            "content_block": {"type": "tool_use", "id": block["id"],
                                                                              "name": block["name"], "input": {}}})
                        raw = json.dumps(block["input"])
                        # Split the JSON the way a real stream does.
                        for cut in range(0, len(raw), 7):
                            self._event("content_block_delta", {"type": "content_block_delta", "index": index,
                                                                "delta": {"type": "input_json_delta", "partial_json": raw[cut:cut + 7]}})
                    self._event("content_block_stop", {"type": "content_block_stop", "index": index})
                stop = turn.get("stop_reason") or (
                    "tool_use" if any(b["type"] == "tool_use" for b in turn["blocks"]) else "end_turn"
                )
                self._event("message_delta", {"type": "message_delta",
                                              "delta": {"stop_reason": stop, "stop_sequence": None},
                                              "usage": {"output_tokens": usage.get("output", 42)}})
                self._event("message_stop", {"type": "message_stop"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
