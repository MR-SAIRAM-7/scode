from __future__ import annotations

import pytest

from scode.tools.base import TodoItem, ToolResult
from scode.tools.workflow import render_todos
from scode.ui.console import UI, StreamWriter
from scode.ui.glyphs import g, supports_unicode
from scode.usage import Usage, estimate_tokens, format_duration, format_tokens

# ------------------------------------------------------------------- usage

def test_usage_accumulates() -> None:
    usage = Usage()
    usage.record(input_tokens=100, output_tokens=20, cached_tokens=5, seconds=1.5)
    usage.record(input_tokens=50, output_tokens=10)

    assert usage.input_tokens == 150
    assert usage.output_tokens == 30
    assert usage.cached_tokens == 5
    # Every token processed counts, cached prompt tokens included.
    assert usage.total_tokens == 185
    assert usage.requests == 2
    assert usage.api_seconds == pytest.approx(1.5)


def test_usage_ignores_negative_counts() -> None:
    usage = Usage()
    usage.record(input_tokens=-5, output_tokens=10)
    assert usage.input_tokens == 0
    assert usage.output_tokens == 10


def test_usage_merge() -> None:
    parent, child = Usage(), Usage()
    child.record(input_tokens=10, output_tokens=5)
    child.tool_calls = 3
    parent.merge(child)

    assert parent.input_tokens == 10
    assert parent.tool_calls == 3
    assert parent.requests == 1


def test_context_percent_is_capped() -> None:
    usage = Usage()
    usage.context_tokens = 500
    assert usage.context_percent(1000) == 50.0
    assert usage.context_percent(100) == 100.0
    assert usage.context_percent(0) == 0.0


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("x" * 400) == 100


@pytest.mark.parametrize(
    "count, expected",
    [(0, "0"), (999, "999"), (1500, "1.5k"), (12000, "12k"), (2_500_000, "2.50M")],
)
def test_format_tokens(count: int, expected: str) -> None:
    assert format_tokens(count) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [(5.25, "5.2s"), (90, "1m 30s"), (3700, "1h 1m")],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


# ------------------------------------------------------------------ glyphs

def test_glyphs_always_resolve() -> None:
    for name in ("bullet", "arrow", "branch", "check", "cross", "todo_done", "prompt"):
        assert g(name)


def test_ascii_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCODE_ASCII", "1")
    supports_unicode.cache_clear()
    try:
        assert supports_unicode() is False
        assert g("todo_done") == "[x]"
        assert g("arrow") == ">"
    finally:
        supports_unicode.cache_clear()


def test_todos_render_with_safe_glyphs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCODE_ASCII", "1")
    supports_unicode.cache_clear()
    try:
        text = render_todos([
            TodoItem("done thing", "completed"),
            TodoItem("doing thing", "in_progress"),
            TodoItem("todo thing", "pending"),
        ])
        text.encode("cp1252")  # must survive a legacy Windows console
        assert "[x] done thing" in text
        assert "[>] doing thing" in text
        assert "[ ] todo thing" in text
    finally:
        supports_unicode.cache_clear()


# ---------------------------------------------------------------------- ui

def test_quiet_ui_prints_nothing(capsys: pytest.CaptureFixture) -> None:
    ui = UI(quiet=True)
    ui.info("hidden")
    ui.tool_call("Read(x)")
    ui.success("also hidden")
    assert capsys.readouterr().out == ""


def test_errors_print_even_when_quiet(capsys: pytest.CaptureFixture) -> None:
    UI(quiet=True).error("something broke")
    assert "something broke" in capsys.readouterr().out


def test_diff_is_rendered(capsys: pytest.CaptureFixture) -> None:
    UI().diff("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n")
    out = capsys.readouterr().out
    assert "-old" in out
    assert "+new" in out


def test_diff_is_truncated(capsys: pytest.CaptureFixture) -> None:
    patch = "\n".join(f"+line {i}" for i in range(200))
    UI().diff(patch, max_lines=10)
    assert "more diff lines" in capsys.readouterr().out


def test_stream_writer_collects_text() -> None:
    writer = StreamWriter(UI(quiet=True))
    writer.write("Hello, ")
    writer.write("world")
    writer.close()
    assert writer.text == "Hello, world"


def test_stream_writer_close_is_idempotent() -> None:
    writer = StreamWriter(UI(quiet=True))
    writer.write("x")
    writer.close()
    writer.close()
    assert writer.text == "x"


def test_stream_writer_ignores_writes_after_close() -> None:
    writer = StreamWriter(UI(quiet=True))
    writer.close()
    writer.write("late")
    assert writer.text == ""


def test_streaming_context_manager() -> None:
    ui = UI(quiet=True)
    with ui.streaming() as writer:
        writer.write("streamed")
    assert writer.text == "streamed"


def test_status_is_a_noop_when_quiet() -> None:
    with UI(quiet=True).status("working") as status:
        assert status is None


# ------------------------------------------------------------- tool result

def test_tool_result_truncation() -> None:
    result = ToolResult(output="x" * 100)
    assert result.truncated_output(limit=1000) == "x" * 100

    long = ToolResult(output="y" * 5000)
    trimmed = long.truncated_output(limit=1000)
    assert "characters truncated" in trimmed
    assert len(trimmed) < 5000


def test_table_renders(capsys: pytest.CaptureFixture) -> None:
    UI().table(["a", "b"], [["1", "2"]], title="t")
    out = capsys.readouterr().out
    assert "1" in out and "2" in out


# ------------------------------------------------- non-terminal fallbacks

def test_input_session_falls_back_without_a_terminal(tmp_path, monkeypatch) -> None:
    from scode.commands import listing
    from scode.ui.input import InputSession

    monkeypatch.setattr("scode.ui.input.interactive_terminal", lambda: False)
    session = InputSession(tmp_path, listing)
    assert session.session is None


def test_read_plain_returns_a_line(monkeypatch) -> None:
    import io

    from scode.ui.input import read_plain

    monkeypatch.setattr("sys.stdin", io.StringIO("hello there\n"))
    assert read_plain("> ") == "hello there"


def test_read_plain_raises_at_eof(monkeypatch) -> None:
    import io

    from scode.ui.input import read_plain

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(EOFError):
        read_plain("> ")


def test_choose_falls_back_to_the_safe_option_at_eof(monkeypatch) -> None:
    import io

    from scode.ui.input import choose

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("scode.ui.input.interactive_terminal", lambda: False)
    index = choose(UI(quiet=True), "Run it?", [("Yes", ""), ("No", "")])
    assert index == 1  # the last option is the safe one


def test_choose_reads_a_number(monkeypatch) -> None:
    import io

    from scode.ui.input import choose

    monkeypatch.setattr("sys.stdin", io.StringIO("2\n"))
    monkeypatch.setattr("scode.ui.input.interactive_terminal", lambda: False)
    assert choose(UI(quiet=True), "Pick", [("a", ""), ("b", ""), ("c", "")]) == 1


def test_choose_accepts_yes_and_no(monkeypatch) -> None:
    import io

    from scode.ui.input import choose

    monkeypatch.setattr("scode.ui.input.interactive_terminal", lambda: False)
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    assert choose(UI(quiet=True), "Pick", [("a", ""), ("b", "")]) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    assert choose(UI(quiet=True), "Pick", [("a", ""), ("b", "")]) == 1


def test_stream_writer_does_not_duplicate_the_first_chunk(capsys) -> None:
    """Regression: plain mode printed the opening chunk twice."""
    ui = UI(quiet=False, markdown=True)
    with ui.streaming() as writer:
        writer.write("Hello ")
        writer.write("world")
    out = capsys.readouterr().out
    assert out.count("Hello") == 1
    assert writer.text == "Hello world"


# ---------------------------------------------------- stream stop markers

def test_stop_marker_hides_the_tool_block() -> None:
    writer = StreamWriter(UI(quiet=True), stop_marker="<tool_use>")
    writer.write('Let me look.\n<tool_use>\n{"name": "LS"}\n</tool_use>')
    assert writer.visible(final=True) == "Let me look."
    assert "<tool_use>" in writer.text  # the parser still sees it


def test_stop_marker_holds_back_a_split_marker() -> None:
    writer = StreamWriter(UI(quiet=True), stop_marker="<tool_use>")
    writer.write("Checking.")
    writer.write("<tool")
    # The partial marker must not be shown while it could still complete.
    assert "<tool" not in writer.visible()
    writer.write("_use>")
    assert writer.visible(final=True) == "Checking."


def test_without_a_marker_everything_is_visible() -> None:
    writer = StreamWriter(UI(quiet=True))
    writer.write("plain <tool_use> text")
    assert writer.visible(final=True) == "plain <tool_use> text"


def test_marker_text_is_printed_once(capsys) -> None:
    ui = UI(quiet=False, markdown=True)
    with ui.streaming(stop_marker="<tool_use>") as writer:
        writer.write("Reading the file.")
        writer.write('\n<tool_use>{"name":"LS"}</tool_use>')
    out = capsys.readouterr().out
    assert out.count("Reading the file.") == 1
    assert "tool_use" not in out
