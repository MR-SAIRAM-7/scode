"""MCP servers as scode tools."""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..permissions import PermissionRequest
from ..tools.base import Tool, ToolContext, ToolResult
from .client import McpClient, McpError, connect

_NAME = re.compile(r"[^a-zA-Z0-9_-]")
# Providers cap tool names at 64 characters.
MAX_TOOL_NAME = 64


def tool_name(server: str, tool: str) -> str:
    name = f"mcp__{_NAME.sub('_', server)}__{_NAME.sub('_', tool)}"
    return name[:MAX_TOOL_NAME]


def _render(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(str(item.get("text", "")))
        elif kind == "image":
            size = len(str(item.get("data", ""))) * 3 // 4
            parts.append(f"[image: {item.get('mimeType', 'image')}, ~{size} bytes]")
        elif kind == "resource":
            resource = item.get("resource") or {}
            parts.append(str(resource.get("text") or f"[resource: {resource.get('uri', '')}]"))
        elif kind == "resource_link":
            parts.append(f"[resource link: {item.get('uri', '')}]")
    if not parts and isinstance(result.get("structuredContent"), (dict, list)):
        import json

        parts.append(json.dumps(result["structuredContent"], indent=2))
    return "\n".join(parts) or "(no output)"


class McpTool(Tool):
    verb = "Calling"

    def __init__(self, server: str, client: McpClient, spec: dict[str, Any]) -> None:
        self.server = server
        self.client = client
        self.remote_name = str(spec.get("name", ""))
        self.name = tool_name(server, self.remote_name)
        description = str(spec.get("description") or "").strip()
        self.description = f"{description} (MCP server: {server})".strip()
        schema = spec.get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        self.parameters = schema
        annotations = spec.get("annotations") or {}
        # Unknown side effects ask for permission unless the server says read-only.
        self.mutating = not bool(annotations.get("readOnlyHint"))

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"{self.server} - {self.remote_name}"

    def permission_request(self, args: dict[str, Any], ctx: ToolContext) -> PermissionRequest:
        import json

        preview = json.dumps(args, indent=2, ensure_ascii=False)
        return PermissionRequest(
            tool=self.name,
            specifier="",
            title=f"{self.server}: {self.remote_name}",
            detail=preview[:2000],
            mutating=self.mutating,
            suggestions=(self.name, f"mcp__{_NAME.sub('_', self.server)}"),
        )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = self.client.call_tool(self.remote_name, args)
        except McpError as exc:
            raise ToolError(f"MCP server {self.server}: {exc}") from exc
        text = _render(result)
        first = next((line for line in text.splitlines() if line.strip()), "(no output)")
        return ToolResult(
            output=text,
            display=first[:100],
            is_error=bool(result.get("isError")),
        )


@dataclass
class ServerStatus:
    name: str
    state: str = "pending"  # connected | failed | pending
    error: str = ""
    tools: list[str] = field(default_factory=list)
    instructions: str = ""


class McpManager:
    """Connects every configured server in parallel and owns their lifetimes."""

    def __init__(self, servers: dict[str, dict[str, Any]], workspace: Path) -> None:
        self.servers = servers
        self.workspace = workspace
        self.clients: dict[str, McpClient] = {}
        self.status: dict[str, ServerStatus] = {name: ServerStatus(name) for name in servers}
        self._tools: list[McpTool] = []
        self._lock = threading.Lock()

    def _start_one(self, name: str, config: dict[str, Any]) -> None:
        status = self.status[name]
        if config.get("disabled"):
            status.state, status.error = "disabled", ""
            return
        try:
            client = connect(name, config, self.workspace)
            specs = client.list_tools()
        except McpError as exc:
            status.state, status.error = "failed", str(exc)
            return
        except Exception as exc:  # a broken server must not break startup
            status.state, status.error = "failed", f"{type(exc).__name__}: {exc}"
            return
        tools = [McpTool(name, client, spec) for spec in specs if spec.get("name")]
        with self._lock:
            self.clients[name] = client
            self._tools.extend(tools)
        status.state = "connected"
        status.tools = [t.remote_name for t in tools]
        status.instructions = client.instructions

    def start(self, timeout: float = 45.0) -> None:
        if not self.servers:
            return
        pool = ThreadPoolExecutor(max_workers=min(8, len(self.servers)))
        futures = [pool.submit(self._start_one, n, c) for n, c in self.servers.items()]
        wait(futures, timeout=timeout)
        pool.shutdown(wait=False, cancel_futures=True)
        for status in self.status.values():
            if status.state == "pending":
                status.state, status.error = "failed", f"did not start within {timeout:.0f}s"

    def tools(self) -> list[McpTool]:
        with self._lock:
            return sorted(self._tools, key=lambda t: t.name)

    def failures(self) -> list[ServerStatus]:
        return [s for s in self.status.values() if s.state == "failed"]

    def instructions(self) -> str:
        """Server-provided usage notes, for the system prompt."""
        blocks = [
            f"## {s.name}\n{s.instructions.strip()}"
            for s in self.status.values()
            if s.state == "connected" and s.instructions.strip()
        ]
        return "\n\n".join(blocks)

    def close(self) -> None:
        for client in list(self.clients.values()):
            try:
                client.close()
            except Exception:
                pass
        self.clients.clear()
