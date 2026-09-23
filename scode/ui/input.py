"""Interactive input: the prompt line, completions and confirmation prompts."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

from ..constants import home_dir
from .glyphs import g

PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "#76b900 bold",
        "bottom-toolbar": "#666666 bg:default",
        "completion-menu.completion": "bg:#1f1f1f #cccccc",
        "completion-menu.completion.current": "bg:#76b900 #000000",
        "completion-menu.meta.completion": "bg:#1f1f1f #888888",
        "completion-menu.meta.completion.current": "bg:#5c8f00 #000000",
    }
)


class ScodeCompleter(Completer):
    """Completes /commands and @file references."""

    def __init__(self, workspace: Path, commands: Callable[[], Iterable[tuple[str, str]]]) -> None:
        self.workspace = workspace
        self.commands = commands

    def get_completions(self, document: Document, complete_event):  # noqa: ANN001
        text = document.text_before_cursor

        if text.startswith("/") and " " not in text:
            prefix = text[1:].lower()
            for name, description in self.commands():
                if name.startswith(prefix):
                    yield Completion(
                        name,
                        start_position=-len(prefix),
                        display=f"/{name}",
                        display_meta=description,
                    )
            return

        at_index = text.rfind("@")
        if at_index != -1 and (at_index == 0 or text[at_index - 1].isspace()):
            partial = text[at_index + 1 :]
            if any(ch in partial for ch in ('"', "'")):
                return
            yield from self._file_completions(partial)

    def _file_completions(self, partial: str) -> Iterable[Completion]:
        raw = Path(partial)
        directory = self.workspace / raw.parent if partial else self.workspace
        stem = raw.name if partial else ""
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        shown = 0
        for entry in entries:
            if entry.name.startswith(".") and not stem.startswith("."):
                continue
            if entry.name in {"node_modules", "__pycache__", ".git", ".venv"}:
                continue
            if stem and not entry.name.lower().startswith(stem.lower()):
                continue
            try:
                relative = entry.relative_to(self.workspace).as_posix()
            except ValueError:
                relative = entry.name
            suffix = "/" if entry.is_dir() else ""
            yield Completion(
                relative + suffix,
                start_position=-len(partial),
                display=entry.name + suffix,
                display_meta="dir" if entry.is_dir() else "file",
            )
            shown += 1
            if shown >= 40:
                return


def history_path() -> Path:
    path = home_dir() / "history"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def interactive_terminal() -> bool:
    """True when a full-screen prompt can actually be drawn."""
    try:
        return bool(sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


class InputSession:
    """Wraps prompt_toolkit, falling back to plain reads outside a terminal."""

    def __init__(
        self,
        workspace: Path,
        commands: Callable[[], Iterable[tuple[str, str]]],
        toolbar: Callable[[], str] | None = None,
        on_cycle_mode: Callable[[], None] | None = None,
    ) -> None:
        self.session: PromptSession[str] | None = None
        if not interactive_terminal():
            return
        bindings = KeyBindings()

        if on_cycle_mode is not None:

            @bindings.add(Keys.BackTab)
            def _(event) -> None:  # noqa: ANN001 - Shift+Tab cycles the permission mode
                on_cycle_mode()
                event.app.invalidate()

        @bindings.add(Keys.Escape, Keys.Enter)
        def _(event) -> None:  # noqa: ANN001 - Esc then Enter inserts a newline
            event.current_buffer.insert_text("\n")

        @bindings.add(Keys.ControlJ)
        def _(event) -> None:  # noqa: ANN001 - Ctrl+J inserts a newline
            event.current_buffer.insert_text("\n")

        try:
            self.session = PromptSession(
                history=FileHistory(str(history_path())),
                completer=ScodeCompleter(workspace, commands),
                complete_while_typing=True,
                key_bindings=bindings,
                style=PROMPT_STYLE,
                multiline=False,
                bottom_toolbar=(lambda: HTML(toolbar())) if toolbar else None,
                enable_history_search=True,
                mouse_support=False,
            )
        except Exception:
            # Some terminals (and CI) cannot host a full prompt; read plainly.
            self.session = None

    def ask(self) -> str:
        """Read one message. Raises EOFError on Ctrl+D, KeyboardInterrupt on Ctrl+C."""
        if self.session is None:
            return read_plain(f"{g('prompt')} ")

        text = self.session.prompt(HTML(f'<prompt>{g("prompt")} </prompt>'))
        # A trailing backslash means "keep typing".
        while text.rstrip().endswith("\\"):
            text = text.rstrip()[:-1] + "\n"
            more = self.session.prompt(HTML('<prompt>. </prompt>'))
            text += more
        return text


def read_line(prompt_text: str) -> str:
    """One line of free text, with or without a full terminal."""
    return _read_key(prompt_text)


def read_plain(prompt_text: str) -> str:
    """A prompt that works without a terminal (pipes, CI, dumb consoles)."""
    sys.stdout.write(prompt_text)
    sys.stdout.flush()
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\n")


def choose(
    ui,  # ui.console.UI
    title: str,
    options: list[tuple[str, str]],
    *,
    detail: str = "",
    default: int = 0,
) -> int:
    """Numbered single-key chooser. Returns the index of the chosen option."""
    from rich.text import Text

    ui.blank()
    ui.panel(_detail_renderable(detail) if detail else Text(title), title=title if detail else "")

    for index, (label, hint) in enumerate(options, start=1):
        line = Text(f"  {index}. ", style="scode.accent")
        line.append(label, style="bold" if index - 1 == default else "default")
        if hint:
            line.append(f"  {hint}", style="scode.muted")
        ui.print(line)
    ui.blank()

    valid = {str(i) for i in range(1, len(options) + 1)}
    while True:
        try:
            answer = _read_key(f"Choose 1-{len(options)} [{default + 1}]: ")
        except (EOFError, KeyboardInterrupt):
            return len(options) - 1  # treat abort as the last (safest) option
        answer = answer.strip().lower()
        if answer == "":
            return default
        if answer in valid:
            return int(answer) - 1
        if answer in {"y", "yes"}:
            return 0
        if answer in {"n", "no", "q"}:
            return len(options) - 1


def _read_key(prompt_text: str) -> str:
    if not interactive_terminal():
        return read_plain(prompt_text)
    from prompt_toolkit import prompt as ptk_prompt

    try:
        return ptk_prompt(prompt_text)
    except Exception:
        return read_plain(prompt_text)


def _detail_renderable(detail: str):
    """Render diffs with colour, everything else as plain text."""
    from rich.text import Text

    if not any(line.startswith(("+++", "@@", "--- ")) for line in detail.splitlines()):
        return Text(detail)

    body = Text()
    for line in detail.splitlines()[:40]:
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
    return body


def confirm(ui, question: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = _read_key(f"{question} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not answer:
        return default
    return answer in {"y", "yes"}
