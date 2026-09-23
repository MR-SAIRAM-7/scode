"""MCP client: stdio against a real subprocess server, HTTP against a local one."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from conftest import text_turn, tool_turn

from scode.mcp import McpManager, McpTool, expand, tool_name
from scode.mcp.client import McpError, connect
from scode.permissions import PermissionEngine, Rule

SERVER = Path(__file__).with_name("mcp_server.py")


def stdio_config() -> dict:
    return {"command": sys.executable, "args": [str(SERVER)]}


@pytest.fixture
def manager(workspace: Path):
    mgr = McpManager({"mock": stdio_config()}, workspace)
    mgr.start(timeout=30)
    yield mgr
    mgr.close()


# ------------------------------------------------------------------- stdio

def test_stdio_server_connects_and_pages_tools(manager: McpManager) -> None:
    status = manager.status["mock"]
    assert status.state == "connected", status.error
    assert status.tools == ["echo", "add", "fail"]  # both pages of tools/list
    assert status.instructions == "Use echo to repeat things."


def test_tools_are_namespaced(manager: McpManager) -> None:
    names = [t.name for t in manager.tools()]
    assert names == ["mcp__mock__add", "mcp__mock__echo", "mcp__mock__fail"]


def test_calling_a_tool(manager: McpManager, ctx) -> None:
    add = next(t for t in manager.tools() if t.remote_name == "add")
    assert add.run({"a": 2, "b": 3}, ctx).output == "5"


def test_error_results_are_flagged(manager: McpManager, ctx) -> None:
    failing = next(t for t in manager.tools() if t.remote_name == "fail")
    result = failing.run({}, ctx)
    assert result.is_error and "something broke" in result.output


def test_read_only_hint_skips_permission(manager: McpManager, settings) -> None:
    tools = {t.remote_name: t for t in manager.tools()}
    assert tools["echo"].mutating is False
    assert tools["add"].mutating is True


def test_server_instructions_reach_the_system_prompt(manager: McpManager) -> None:
    assert "Use echo to repeat things." in manager.instructions()


def test_a_broken_command_fails_cleanly(workspace: Path) -> None:
    mgr = McpManager({"ghost": {"command": "definitely-not-a-real-binary-xyz"}}, workspace)
    mgr.start(timeout=10)
    assert mgr.status["ghost"].state == "failed"
    assert mgr.tools() == []


def test_a_server_that_exits_is_reported(workspace: Path) -> None:
    mgr = McpManager({"quitter": {"command": sys.executable, "args": ["-c", "print('bye')"]}}, workspace)
    mgr.start(timeout=15)
    assert mgr.status["quitter"].state == "failed"
    assert "exited" in mgr.status["quitter"].error or "no response" in mgr.status["quitter"].error


def test_disabled_servers_are_skipped(workspace: Path) -> None:
    mgr = McpManager({"off": {**stdio_config(), "disabled": True}}, workspace)
    mgr.start()
    assert mgr.status["off"].state == "disabled"


def test_agent_calls_an_mcp_tool(make_agent, manager: McpManager, settings) -> None:
    agent, _ = make_agent([tool_turn("mcp__mock__echo", {"text": "hi there"}), text_turn("done")])
    for tool in manager.tools():
        agent.registry.register(tool)
    agent.run("echo something")
    result = next(m for m in agent.messages if m.get("role") == "tool")
    assert result["content"] == "echo: hi there"


def test_mutating_mcp_tools_need_approval(make_agent, manager: McpManager) -> None:
    agent, _ = make_agent([tool_turn("mcp__mock__add", {"a": 1, "b": 1}), text_turn("ok")], permission_mode="default")
    for tool in manager.tools():
        agent.registry.register(tool)
    agent.run("add")
    result = next(m for m in agent.messages if m.get("role") == "tool")
    assert "was not run" in result["content"]


def test_a_server_wide_rule_allows_all_its_tools(settings) -> None:
    rule = Rule.parse("mcp__github")
    assert rule.matches("mcp__github__create_issue", "")
    assert not rule.matches("mcp__gitlab__create_issue", "")
    engine = PermissionEngine(settings)
    engine.remember("mcp__mock", persist=False)


def test_tool_names_are_sanitised_and_capped() -> None:
    assert tool_name("my server", "do.thing") == "mcp__my_server__do_thing"
    assert len(tool_name("s" * 40, "t" * 40)) == 64


def test_env_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "abc")
    config = {"headers": {"Authorization": "Bearer ${GH_TOKEN}"}, "args": ["${MISSING:-fallback}", "plain"]}
    assert expand(config) == {"headers": {"Authorization": "Bearer abc"}, "args": ["fallback", "plain"]}


# -------------------------------------------------------------------- http

class _HttpMcp:
    """Streamable HTTP server: JSON or SSE replies, a session id, 202 for notifications."""

    def __init__(self, *, sse: bool) -> None:
        self.sse = sse
        self.seen: list[dict] = []
        self.session_headers: list[str | None] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_DELETE(self):
                self.send_response(200)
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append(body)
                outer.session_headers.append(self.headers.get("Mcp-Session-Id"))
                if "id" not in body:
                    self.send_response(202)
                    self.end_headers()
                    return
                method = body["method"]
                if method == "initialize":
                    result = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "h"}}
                elif method == "tools/list":
                    result = {"tools": [{"name": "search", "description": "Search docs",
                                         "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}}]}
                else:
                    result = {"content": [{"type": "text", "text": f"results for {body['params']['arguments']['q']}"}]}
                reply = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result})
                self.send_response(200)
                self.send_header("Mcp-Session-Id", "sess-42")
                if outer.sse:
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.write(f"event: message\ndata: {reply}\n\n".encode())
                else:
                    data = reply.encode()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.mark.parametrize("sse", [False, True])
def test_http_transport(sse: bool, workspace: Path) -> None:
    server = _HttpMcp(sse=sse)
    try:
        client = connect("docs", {"type": "http", "url": server.url, "headers": {"X-Key": "k"}}, workspace)
        tools = client.list_tools()
        assert [t["name"] for t in tools] == ["search"]
        result = client.call_tool("search", {"q": "caching"})
        assert result["content"][0]["text"] == "results for caching"
        client.close()
    finally:
        server.close()
    # The session id from initialize is sent on every later request.
    assert server.session_headers[0] is None
    assert all(h == "sess-42" for h in server.session_headers[1:])
    assert server.seen[1] == {"jsonrpc": "2.0", "method": "notifications/initialized"}


def test_http_errors_are_reported(workspace: Path) -> None:
    with pytest.raises(McpError, match="could not reach"):
        connect("down", {"type": "http", "url": "http://127.0.0.1:9/mcp"}, workspace)


def test_legacy_sse_transport_is_refused(workspace: Path) -> None:
    with pytest.raises(McpError, match="legacy SSE"):
        connect("old", {"type": "sse", "url": "http://x"}, workspace)


def test_mcp_tool_wrapper_renders_content_types(workspace: Path, ctx) -> None:
    class FakeClient:
        def call_tool(self, name, arguments, timeout=60):
            return {"content": [
                {"type": "text", "text": "line one"},
                {"type": "image", "mimeType": "image/png", "data": "QUJDRA=="},
                {"type": "resource", "resource": {"uri": "file:///x", "text": "resource body"}},
            ]}

    tool = McpTool("srv", FakeClient(), {"name": "multi", "inputSchema": {"type": "object"}})
    output = tool.run({}, ctx).output
    assert "line one" in output and "[image: image/png" in output and "resource body" in output
