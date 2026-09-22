"""WebFetch: pull a URL into the conversation as readable text."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import requests

from ..errors import ToolError
from ..permissions import PermissionRequest
from .base import Tool, ToolContext, ToolResult

MAX_CHARS = 40_000
SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "nav", "footer"}
BLOCK_TAGS = {
    "p", "div", "section", "article", "br", "li", "tr", "h1", "h2", "h3",
    "h4", "h5", "h6", "pre", "blockquote", "table",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self._skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in BLOCK_TAGS:
            self.chunks.append("\n")
        if tag in {"h1", "h2", "h3"}:
            self.chunks.append("\n## ")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip()
        if self._skip_depth:
            return
        if data.strip():
            self.chunks.append(data)

    def text(self) -> str:
        joined = "".join(self.chunks)
        joined = re.sub(r"[ \t\f\v]+", " ", joined)
        joined = re.sub(r"\n\s*\n\s*\n+", "\n\n", joined)
        return joined.strip()


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup should still yield what we parsed
        pass
    return parser.title, parser.text()


class WebFetchTool(Tool):
    name = "WebFetch"
    verb = "Fetching"
    description = (
        "Fetch a URL and return its readable text. Use it for documentation, "
        "issues, release notes and API references. HTML is converted to text; "
        "JSON and plain text come back as-is."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL"},
            "prompt": {
                "type": "string",
                "description": "What you are looking for on the page",
            },
        },
        "required": ["url"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"WebFetch({args.get('url', '')})"

    def permission_request(self, args: dict[str, Any], ctx: ToolContext) -> PermissionRequest:
        url = str(args.get("url", ""))
        host = urlparse(url).netloc or url
        return PermissionRequest(
            tool=self.name,
            specifier=url,
            title=f"Fetch {url}",
            detail=str(args.get("prompt") or ""),
            mutating=True,
            suggestions=(f"WebFetch({urlparse(url).scheme}://{host}/*)", "WebFetch"),
        )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        url = str(args["url"]).strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolError("url must start with http:// or https://")
        if not parsed.netloc:
            raise ToolError(f"{url!r} is not a valid URL")

        try:
            response = requests.get(
                url,
                timeout=(15, 45),
                headers={
                    "User-Agent": "scode-cli/1.0 (+https://github.com/MR-SAIRAM-7/scode)",
                    "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.8",
                },
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            raise ToolError(f"Could not fetch {url}: {exc}") from exc

        if response.status_code >= 400:
            raise ToolError(f"{url} returned HTTP {response.status_code}")

        content_type = response.headers.get("Content-Type", "")
        body = response.text
        title = ""
        if "html" in content_type.lower() or body.lstrip()[:200].lower().startswith(
            ("<!doctype html", "<html")
        ):
            title, body = html_to_text(body)

        if len(body) > MAX_CHARS:
            body = body[:MAX_CHARS] + f"\n\n[truncated at {MAX_CHARS} characters]"

        final_url = response.url
        header = f"# {title}\n" if title else ""
        note = ""
        if final_url != url:
            note = f"(redirected to {final_url})\n"

        return ToolResult(
            output=f"{header}{note}Source: {final_url}\n\n{body or '(empty response)'}",
            display=f"Fetched {parsed.netloc} ({len(body)} chars)",
            metadata={"url": final_url, "status": response.status_code},
        )
