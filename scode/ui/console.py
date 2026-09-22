"""All terminal output goes through the UI class."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..constants import PRODUCT_TAGLINE, VERSION
from .glyphs import enable_utf8_stdout, g, supports_unicode
from .theme import get_theme

# Streaming above this size stops re-rendering markdown and appends plain text,
# which keeps very long answers fast and scrollback-safe.
LIVE_MARKDOWN_LIMIT = 12_000
LIVE_REFRESH_SECONDS = 0.08


class UI:
    def __init__(self, theme: str = "dark", *, quiet: bool = False, markdown: bool = True) -> None:
        enable_utf8_stdout()
        self.console = Console(
            theme=get_theme(theme),
            soft_wrap=False,
            highlight=False,
            emoji=False,
        )
        self.quiet = quiet
        self.markdown = markdown
        self._status: Status | None = None

    # ----------------------------------------------------------- primitives
    def print(self, *args, **kwargs) -> None:
        if not self.quiet:
            self.console.print(*args, **kwargs)

    def blank(self) -> None:
        if not self.quiet:
            self.console.print()

    def rule(self, label: str = "") -> None:
        if not self.quiet:
            self.console.print(Rule(label, style="scode.muted"))

    def muted(self, text: str) -> None:
        self.print(Text(text, style="scode.muted"))

    def error(self, text: str) -> None:
        self.console.print(Text(f"{g('cross')} {text}", style="scode.error"))

    def warn(self, text: str) -> None:
        self.print(Text(f"{g('warn')} {text}", style="scode.warn"))

    def success(self, text: str) -> None:
        self.print(Text(f"{g('check')} {text}", style="scode.success"))

    def info(self, text: str) -> None:
        self.print(Text(text))

    # --------------------------------------------------------------- banner
    def banner(self, *, model: str, workspace: Path, mode: str, api_key: bool) -> None:
        if self.quiet:
            return
        title = Text()
        title.append("scode", style="scode.brand")
        title.append(f"  v{VERSION}", style="scode.muted")

        body = Table.grid(padding=(0, 2))
        body.add_column(style="scode.muted", justify="right")
        body.add_column()
        body.add_row("model", model)
        body.add_row("cwd", str(workspace))
        body.add_row("mode", mode)
        if not api_key:
            body.add_row("key", Text("not set — run /login", style="scode.warn"))

        self.console.print(
            Panel(
                Group(title, Text(PRODUCT_TAGLINE, style="scode.muted"), Text(), body),
                border_style="scode.accent",
                padding=(1, 2),
            )
        )
        self.console.print(
            Text("  /help for commands  ·  /exit to quit  ·  Ctrl+C to interrupt",
                 style="scode.muted")
        )
        self.blank()

    # ------------------------------------------------------------ messaging
    def user_echo(self, text: str) -> None:
        self.print(Text(f"{g('bullet')} ", style="scode.user").append(text, style="default"))

    def assistant_markdown(self, text: str) -> None:
        """Render a finished assistant message."""
        if not text.strip():
            return
        if self.markdown:
            self.console.print(Markdown(text, code_theme="monokai"))
        else:
            self.console.print(Text(text))

    def tool_call(self, summary: str) -> None:
        self.print(Text(f"{g('arrow')} ", style="scode.tool").append(summary, style="scode.tool"))

    def tool_result(self, line: str, *, error: bool = False) -> None:
        style = "scode.error" if error else "scode.tool.result"
        prefix = Text(f"  {g('branch')}  ", style="scode.tool.result")
        self.print(prefix.append(line, style=style))

    def thinking(self, text: str) -> None:
        if text.strip():
            self.print(Text(text.strip(), style="scode.thinking"))

    # ---------------------------------------------------------------- diffs
    def diff(self, patch: str, *, max_lines: int = 60) -> None:
        if not patch.strip() or self.quiet:
            return
        lines = patch.splitlines()
        shown = lines[:max_lines]
        body = Text()
        for line in shown:
            if line.startswith(("+++", "---")):
                style = "scode.diff.meta"
            elif line.startswith("@@"):
                style = "scode.diff.hunk"
            elif line.startswith("+"):
                style = "scode.diff.add"
            elif line.startswith("-"):
                style = "scode.diff.del"
            else:
                style = "scode.muted"
            body.append(line + "\n", style=style)
        if len(lines) > max_lines:
            body.append(f"... {len(lines) - max_lines} more diff lines\n", style="scode.muted")
        self.console.print(Panel(body, border_style="scode.muted", padding=(0, 1)))

    def code(self, text: str, language: str = "text", *, title: str = "") -> None:
        if self.quiet:
            return
        syntax = Syntax(text, language, theme="monokai", word_wrap=True, line_numbers=False)
        self.console.print(Panel(syntax, title=title or None, border_style="scode.muted"))

    def table(self, columns: list[str], rows: list[list[str]], *, title: str = "") -> None:
        if self.quiet:
            return
        table = Table(
            title=title or None,
            box=None,
            pad_edge=False,
            title_style="scode.brand",
            header_style="scode.muted",
        )
        for column in columns:
            table.add_column(column, overflow="fold")
        for row in rows:
            table.add_row(*row)
        self.console.print(table)

    def panel(self, renderable, *, title: str = "", style: str = "scode.accent") -> None:
        if self.quiet:
            return
        self.console.print(Panel(renderable, title=title or None, border_style=style, padding=(0, 1)))

    # --------------------------------------------------------------- status
    @contextmanager
    def status(self, label: str) -> Iterator[Status | None]:
        if self.quiet or not self.console.is_terminal:
            yield None
            return
        spinner = "dots" if supports_unicode() else "line"
        with self.console.status(Text(label, style="scode.muted"), spinner=spinner) as status:
            self._status = status
            try:
                yield status
            finally:
                self._status = None

    @contextmanager
    def streaming(self, stop_marker: str | None = None) -> Iterator[StreamWriter]:
        """Context manager that renders assistant text as it arrives."""
        writer = StreamWriter(self, stop_marker=stop_marker)
        try:
            yield writer
        finally:
            writer.close()


class StreamWriter:
    """Incrementally renders streamed assistant output.

    `stop_marker` hides everything from that marker onwards. The text-protocol
    mode uses it so a `<tool_use>` block is parsed but never shown.
    """

    def __init__(self, ui: UI, stop_marker: str | None = None) -> None:
        self.ui = ui
        self.stop_marker = stop_marker
        self.buffer: list[str] = []
        self._live: Live | None = None
        self._last_refresh = 0.0
        self._plain_mode = False
        self._closed = False
        self._started = False
        self._printed = 0

    @property
    def text(self) -> str:
        return "".join(self.buffer)

    def visible(self, *, final: bool = False) -> str:
        """The part of the stream the user should see."""
        text = self.text
        if not self.stop_marker:
            return text
        index = text.find(self.stop_marker)
        if index != -1:
            return text[:index].rstrip()
        if not final:
            # Hold back enough characters that a marker split across chunks is
            # never printed before it can be recognised.
            keep = len(text) - (len(self.stop_marker) - 1)
            return text[: max(0, keep)]
        return text

    def write(self, chunk: str) -> None:
        if not chunk or self._closed:
            return
        self.buffer.append(chunk)

        if self.ui.quiet:
            return
        if not self._started:
            self._start()

        visible = self.visible()
        if self._plain_mode:
            self._print_plain(visible)
            return

        now = time.monotonic()
        if len(self.text) > LIVE_MARKDOWN_LIMIT:
            self._switch_to_plain()
            return
        if now - self._last_refresh < LIVE_REFRESH_SECONDS:
            return
        self._last_refresh = now
        self._render(visible)

    def _print_plain(self, visible: str) -> None:
        pending = visible[self._printed :]
        if pending:
            self.ui.console.print(Text(pending), end="", soft_wrap=True)
            self._printed = len(visible)

    def _start(self) -> None:
        self._started = True
        if not self.ui.markdown or not self.ui.console.is_terminal:
            self._plain_mode = True
            return
        self._live = Live(
            Markdown(self.visible(), code_theme="monokai"),
            console=self.ui.console,
            refresh_per_second=12,
            vertical_overflow="visible",
            transient=False,
        )
        self._live.start()

    def _render(self, visible: str) -> None:
        if self._live is not None:
            self._live.update(Markdown(visible, code_theme="monokai"))

    def _switch_to_plain(self) -> None:
        """Long answers stop re-rendering markdown and just append."""
        visible = self.visible()
        if self._live is not None:
            self._live.update(Markdown(visible, code_theme="monokai"))
            self._live.stop()
            self._live = None
        self._plain_mode = True
        self._printed = len(visible)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.ui.quiet:
            return

        final = self.visible(final=True)
        if self._live is not None:
            self._live.update(Markdown(final, code_theme="monokai"))
            self._live.stop()
            self._live = None
        elif self._plain_mode:
            self._print_plain(final)
            if final:
                self.ui.console.print()


def plain_print(text: str) -> None:
    """Bypass rich entirely — used by -p/--print output."""
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
