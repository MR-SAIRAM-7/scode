"""A minimal stdio MCP server for tests.

Speaks newline-delimited JSON-RPC: initialize, tools/list (paginated),
tools/call. It also writes a non-JSON line to stdout at startup (some real
servers log there) and asks the client for roots/list once, to exercise the
client's handling of server-initiated requests.
"""

import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the text back.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
    },
    {
        "name": "fail",
        "description": "Always reports an error.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main():
    sys.stdout.write("mock server starting (not JSON)\n")
    sys.stdout.flush()
    asked_roots = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        method = message.get("method")
        if "id" in message and method is None:
            continue  # the client answering our roots/list request
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": message["id"], "result": {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock", "version": "1"},
                "instructions": "Use echo to repeat things.",
            }})
        elif method == "notifications/initialized":
            if not asked_roots:
                asked_roots = True
                send({"jsonrpc": "2.0", "id": "srv-1", "method": "roots/list"})
        elif method == "tools/list":
            cursor = (message.get("params") or {}).get("cursor")
            if cursor is None:
                send({"jsonrpc": "2.0", "id": message["id"], "result": {"tools": TOOLS[:2], "nextCursor": "page2"}})
            else:
                send({"jsonrpc": "2.0", "id": message["id"], "result": {"tools": TOOLS[2:]}})
        elif method == "tools/call":
            name = message["params"]["name"]
            args = message["params"].get("arguments") or {}
            if name == "echo":
                result = {"content": [{"type": "text", "text": f"echo: {args.get('text', '')}"}]}
            elif name == "add":
                result = {"content": [{"type": "text", "text": str(args.get("a", 0) + args.get("b", 0))}]}
            else:
                result = {"content": [{"type": "text", "text": "something broke"}], "isError": True}
            send({"jsonrpc": "2.0", "id": message["id"], "result": result})
        elif "id" in message:
            send({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "unknown"}})


if __name__ == "__main__":
    main()
