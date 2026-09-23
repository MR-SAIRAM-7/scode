"""Model Context Protocol client: stdio and Streamable HTTP transports.

Only what an agent needs: the initialize handshake, tools/list, tools/call,
and polite answers to the few requests servers send back (ping, roots/list).
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any

import requests

from ..constants import VERSION

PROTOCOL_VERSION = "2025-06-18"
DEFAULT_TIMEOUT = 60.0


class McpError(Exception):
    """A server failed to start, answer, or speak the protocol."""


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand(value: Any) -> Any:
    """Expand ${VAR} and ${VAR:-default} the way Claude Code's .mcp.json does."""
    if isinstance(value, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


class _Transport:
    def request(self, method: str, params: dict[str, Any] | None, timeout: float) -> Any:
        raise NotImplementedError

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class StdioTransport(_Transport):
    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None,
        cwd: Path,
        roots: list[dict[str, str]],
    ) -> None:
        resolved = shutil.which(command) or command
        argv = [resolved, *args]
        if sys.platform == "win32" and resolved.lower().endswith((".cmd", ".bat")):
            # CreateProcess can't launch batch shims such as npx.cmd directly.
            argv = ["cmd", "/d", "/c", resolved, *args]
        full_env = {**os.environ, **(env or {})}
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            self.process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd),
                env=full_env,
                **kwargs,
            )
        except OSError as exc:
            raise McpError(f"could not start {command!r}: {exc}") from exc

        self._roots = roots
        self._ids = itertools.count(1)
        self._pending: dict[int, dict[str, Any]] = {}
        self._events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self.stderr_tail: deque[str] = deque(maxlen=40)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _send(self, message: dict[str, Any]) -> None:
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(data)
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise McpError(f"server stdin closed: {exc}") from exc

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for raw in iter(self.process.stdout.readline, b""):
            try:
                message = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue  # servers sometimes log to stdout; ignore non-JSON
            if isinstance(message, dict):
                self._dispatch(message)
        # Process exited: wake every waiter so nothing hangs.
        with self._lock:
            for event in self._events.values():
                event.set()

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for raw in iter(self.process.stderr.readline, b""):
            self.stderr_tail.append(raw.decode("utf-8", errors="replace").rstrip())

    def _dispatch(self, message: dict[str, Any]) -> None:
        if "method" in message and "id" in message:
            self._answer_server_request(message)
            return
        if "id" in message and ("result" in message or "error" in message):
            with self._lock:
                self._pending[message["id"]] = message
                event = self._events.get(message["id"])
            if event:
                event.set()

    def _answer_server_request(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if method == "ping":
            reply: dict[str, Any] = {"result": {}}
        elif method == "roots/list":
            reply = {"result": {"roots": self._roots}}
        else:
            reply = {"error": {"code": -32601, "message": f"scode does not support {method}"}}
        try:
            self._send({"jsonrpc": "2.0", "id": message["id"], **reply})
        except McpError:
            pass

    def request(self, method: str, params: dict[str, Any] | None, timeout: float) -> Any:
        request_id = next(self._ids)
        event = threading.Event()
        with self._lock:
            self._events[request_id] = event
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)
        finished = event.wait(timeout)
        with self._lock:
            self._events.pop(request_id, None)
            response = self._pending.pop(request_id, None)
        if response is None:
            if self.process.poll() is not None:
                tail = " | ".join(list(self.stderr_tail)[-3:])
                raise McpError(f"server exited (code {self.process.returncode}). {tail}".strip())
            if not finished:
                raise McpError(f"{method} timed out after {timeout:.0f}s")
            raise McpError(f"{method} got no response")
        if "error" in response:
            error = response["error"] or {}
            raise McpError(f"{method} failed: {error.get('message', error)}")
        return response.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()


class HttpTransport(_Transport):
    """MCP Streamable HTTP: JSON-RPC over POST, answered as JSON or SSE."""

    def __init__(self, url: str, headers: dict[str, str] | None) -> None:
        self.url = url
        self.headers = headers or {}
        self.session = requests.Session()
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self._ids = itertools.count(1)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        return headers

    def _post(self, message: dict[str, Any], timeout: float) -> requests.Response:
        try:
            response = self.session.post(
                self.url, json=message, headers=self._headers(), stream=True, timeout=(15, timeout)
            )
        except requests.RequestException as exc:
            raise McpError(f"could not reach {self.url}: {exc}") from exc
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self.session_id = session_id
        if response.status_code in (401, 403):
            response.close()
            raise McpError(
                f"{self.url} rejected the request ({response.status_code}). "
                "Add an Authorization header in the server's \"headers\" config."
            )
        if response.status_code >= 400:
            text = response.text[:200]
            response.close()
            raise McpError(f"{self.url} returned HTTP {response.status_code}: {text}")
        return response

    def request(self, method: str, params: dict[str, Any] | None, timeout: float) -> Any:
        request_id = next(self._ids)
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        response = self._post(message, timeout)
        try:
            ctype = response.headers.get("Content-Type", "")
            if "text/event-stream" in ctype:
                reply = self._reply_from_sse(response, request_id)
            else:
                try:
                    reply = response.json()
                except ValueError as exc:
                    raise McpError(f"{method}: server sent non-JSON") from exc
        finally:
            response.close()
        if isinstance(reply, list):  # a JSON-RPC batch
            reply = next((r for r in reply if isinstance(r, dict) and r.get("id") == request_id), None)
        if not isinstance(reply, dict):
            raise McpError(f"{method} got no response")
        if "error" in reply:
            error = reply["error"] or {}
            raise McpError(f"{method} failed: {error.get('message', error)}")
        return reply.get("result")

    @staticmethod
    def _reply_from_sse(response: requests.Response, request_id: int) -> Any:
        data_lines: list[str] = []
        for raw in response.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            if raw == "":
                if data_lines:
                    try:
                        event = json.loads("\n".join(data_lines))
                    except json.JSONDecodeError:
                        event = None
                    data_lines = []
                    if isinstance(event, dict) and event.get("id") == request_id:
                        return event
                continue
            if raw.startswith("data:"):
                data_lines.append(raw[5:].lstrip())
        if data_lines:
            try:
                event = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                return None
            if isinstance(event, dict) and event.get("id") == request_id:
                return event
        return None

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._post(message, 15).close()

    def close(self) -> None:
        if self.session_id:
            try:
                self.session.delete(self.url, headers=self._headers(), timeout=5)
            except requests.RequestException:
                pass
        self.session.close()


class McpClient:
    def __init__(self, name: str, transport: _Transport) -> None:
        self.name = name
        self.transport = transport
        self.server_info: dict[str, Any] = {}
        self.instructions = ""

    def initialize(self, timeout: float = 30.0) -> None:
        result = self.transport.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"roots": {"listChanged": False}},
                "clientInfo": {"name": "scode", "version": VERSION},
            },
            timeout,
        ) or {}
        self.server_info = result.get("serverInfo") or {}
        self.instructions = str(result.get("instructions") or "")
        if isinstance(self.transport, HttpTransport):
            self.transport.protocol_version = str(result.get("protocolVersion") or PROTOCOL_VERSION)
        self.transport.notify("notifications/initialized")

    def list_tools(self, timeout: float = 30.0) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(50):  # pagination guard
            params = {"cursor": cursor} if cursor else {}
            result = self.transport.request("tools/list", params, timeout) or {}
            tools.extend(t for t in result.get("tools") or [] if isinstance(t, dict))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
        result = self.transport.request("tools/call", {"name": name, "arguments": arguments}, timeout)
        return result if isinstance(result, dict) else {"content": []}

    def close(self) -> None:
        self.transport.close()


def connect(name: str, config: dict[str, Any], workspace: Path) -> McpClient:
    """Start (stdio) or reach (http) one configured server and initialise it."""
    config = expand(config)
    kind = str(config.get("type") or ("http" if config.get("url") else "stdio")).lower()
    if kind == "stdio":
        command = str(config.get("command") or "").strip()
        if not command:
            raise McpError("stdio server needs a command")
        args = [str(a) for a in config.get("args") or []]
        env = {str(k): str(v) for k, v in (config.get("env") or {}).items()}
        roots = [{"uri": workspace.resolve().as_uri(), "name": workspace.name}]
        transport: _Transport = StdioTransport(command, args, env, workspace, roots)
    elif kind in {"http", "streamable-http", "streamable_http"}:
        url = str(config.get("url") or "").strip()
        if not url:
            raise McpError("http server needs a url")
        headers = {str(k): str(v) for k, v in (config.get("headers") or {}).items()}
        transport = HttpTransport(url, headers)
    elif kind == "sse":
        raise McpError("the legacy SSE transport isn't supported; use the server's HTTP endpoint")
    else:
        raise McpError(f"unknown transport type {kind!r}")

    client = McpClient(name, transport)
    try:
        client.initialize(timeout=float(config.get("timeout") or 30))
    except McpError:
        client.close()
        raise
    return client
