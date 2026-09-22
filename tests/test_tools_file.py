from __future__ import annotations

from pathlib import Path

import pytest

from scode.errors import ToolError
from scode.tools import ToolContext
from scode.tools.file_tools import (
    EditTool,
    MultiEditTool,
    ReadTool,
    WriteTool,
    unified_diff,
)


def read(ctx: ToolContext, **args):
    return ReadTool().run(args, ctx)


# --------------------------------------------------------------------- Read

def test_read_numbers_lines(ctx: ToolContext) -> None:
    result = read(ctx, file_path="app.py")
    assert "     1\tdef add(a, b):" in result.output
    assert "     2\t    return a + b" in result.output


def test_read_marks_the_file_as_seen(ctx: ToolContext) -> None:
    read(ctx, file_path="app.py")
    assert str(ctx.workspace / "app.py") in ctx.read_files


def test_read_offset_and_limit(ctx: ToolContext) -> None:
    (ctx.workspace / "many.txt").write_text("\n".join(f"line{i}" for i in range(1, 21)), encoding="utf-8")
    result = read(ctx, file_path="many.txt", offset=5, limit=3)
    assert "     5\tline5" in result.output
    assert "     8\tline8" not in result.output
    assert "offset=8" in result.output


def test_read_rejects_a_missing_file(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        read(ctx, file_path="nope.py")


def test_read_rejects_a_directory(ctx: ToolContext) -> None:
    (ctx.workspace / "sub").mkdir()
    with pytest.raises(ToolError, match="is a directory"):
        read(ctx, file_path="sub")


def test_read_rejects_binary(ctx: ToolContext) -> None:
    (ctx.workspace / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    with pytest.raises(ToolError, match="binary"):
        read(ctx, file_path="blob.bin")


def test_read_empty_file(ctx: ToolContext) -> None:
    (ctx.workspace / "empty.txt").write_text("", encoding="utf-8")
    assert "empty" in read(ctx, file_path="empty.txt").output


def test_read_offset_past_end(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="past the end"):
        read(ctx, file_path="app.py", offset=500)


def test_read_requires_a_path(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="required"):
        ReadTool().run({}, ctx)


# -------------------------------------------------------------------- Write

def test_write_creates_nested_directories(ctx: ToolContext) -> None:
    result = WriteTool().run({"file_path": "a/b/c.txt", "content": "hi\n"}, ctx)
    assert (ctx.workspace / "a/b/c.txt").read_text(encoding="utf-8") == "hi\n"
    assert "Created" in result.output


def test_write_over_an_unread_file_is_refused(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="before editing"):
        WriteTool().run({"file_path": "app.py", "content": "x"}, ctx)


def test_write_over_a_read_file_works(ctx: ToolContext) -> None:
    read(ctx, file_path="app.py")
    result = WriteTool().run({"file_path": "app.py", "content": "new\n"}, ctx)
    assert (ctx.workspace / "app.py").read_text(encoding="utf-8") == "new\n"
    assert "Updated" in result.output
    assert result.detail.startswith("---")


def test_write_permission_request_shows_a_diff(ctx: ToolContext) -> None:
    request = WriteTool().permission_request(
        {"file_path": "app.py", "content": "def add(a, b):\n    return 0\n"}, ctx
    )
    assert "Overwrite" in request.title
    assert "+    return 0" in request.detail


# --------------------------------------------------------------------- Edit

def test_edit_replaces_exact_text(ctx: ToolContext) -> None:
    read(ctx, file_path="app.py")
    result = EditTool().run(
        {"file_path": "app.py", "old_string": "a + b", "new_string": "a - b"}, ctx
    )
    assert "a - b" in (ctx.workspace / "app.py").read_text(encoding="utf-8")
    assert "+1/-1" in result.output


def test_edit_requires_a_prior_read(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="before editing"):
        EditTool().run({"file_path": "app.py", "old_string": "a + b", "new_string": "x"}, ctx)


def test_edit_rejects_a_missing_string(ctx: ToolContext) -> None:
    read(ctx, file_path="app.py")
    with pytest.raises(ToolError, match="was not found"):
        EditTool().run({"file_path": "app.py", "old_string": "nope", "new_string": "x"}, ctx)


def test_edit_hints_when_only_whitespace_differs(ctx: ToolContext) -> None:
    read(ctx, file_path="app.py")
    with pytest.raises(ToolError, match="whitespace differs"):
        EditTool().run(
            {"file_path": "app.py", "old_string": "  return a + b  ", "new_string": "x"}, ctx
        )


def test_edit_rejects_an_ambiguous_string(ctx: ToolContext) -> None:
    (ctx.workspace / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    read(ctx, file_path="dup.py")
    with pytest.raises(ToolError, match="appears 2 times"):
        EditTool().run({"file_path": "dup.py", "old_string": "x = 1", "new_string": "x = 2"}, ctx)


def test_edit_replace_all(ctx: ToolContext) -> None:
    (ctx.workspace / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    read(ctx, file_path="dup.py")
    EditTool().run(
        {"file_path": "dup.py", "old_string": "x = 1", "new_string": "x = 2", "replace_all": True},
        ctx,
    )
    assert (ctx.workspace / "dup.py").read_text(encoding="utf-8") == "x = 2\nx = 2\n"


def test_edit_rejects_identical_strings(ctx: ToolContext) -> None:
    read(ctx, file_path="app.py")
    with pytest.raises(ToolError, match="identical"):
        EditTool().run({"file_path": "app.py", "old_string": "a", "new_string": "a"}, ctx)


def test_edit_detects_an_external_change(ctx: ToolContext, monkeypatch) -> None:
    path = ctx.workspace / "app.py"
    read(ctx, file_path="app.py")
    # Simulate another process writing the file after it was read.
    ctx.read_files[str(path)] = path.stat().st_mtime - 100
    with pytest.raises(ToolError, match="changed on disk"):
        EditTool().run({"file_path": "app.py", "old_string": "a + b", "new_string": "x"}, ctx)


# ---------------------------------------------------------------- MultiEdit

def test_multiedit_applies_in_order(ctx: ToolContext) -> None:
    read(ctx, file_path="app.py")
    result = MultiEditTool().run(
        {
            "file_path": "app.py",
            "edits": [
                {"old_string": "def add", "new_string": "def total"},
                {"old_string": "a + b", "new_string": "sum((a, b))"},
            ],
        },
        ctx,
    )
    text = (ctx.workspace / "app.py").read_text(encoding="utf-8")
    assert text == "def total(a, b):\n    return sum((a, b))\n"
    assert "Applied 2 edits" in result.output


def test_multiedit_is_atomic(ctx: ToolContext) -> None:
    original = (ctx.workspace / "app.py").read_text(encoding="utf-8")
    read(ctx, file_path="app.py")
    with pytest.raises(ToolError, match="edit 2 of 2 failed"):
        MultiEditTool().run(
            {
                "file_path": "app.py",
                "edits": [
                    {"old_string": "def add", "new_string": "def total"},
                    {"old_string": "does not exist", "new_string": "x"},
                ],
            },
            ctx,
        )
    assert (ctx.workspace / "app.py").read_text(encoding="utf-8") == original


def test_multiedit_needs_edits(ctx: ToolContext) -> None:
    read(ctx, file_path="app.py")
    with pytest.raises(ToolError):
        MultiEditTool().run({"file_path": "app.py", "edits": []}, ctx)


# --------------------------------------------------------------------- misc

def test_unified_diff_format() -> None:
    patch = unified_diff("a\n", "b\n", "f.txt")
    assert patch.startswith("--- a/f.txt")
    assert "+b" in patch


def test_paths_outside_the_workspace_are_flagged(ctx: ToolContext, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    request = WriteTool().permission_request(
        {"file_path": str(outside), "content": "changed"}, ctx
    )
    assert "outside the workspace" in request.detail


def test_display_path_is_relative(ctx: ToolContext) -> None:
    assert ctx.display_path(ctx.workspace / "a" / "b.py") == "a/b.py"


def test_crlf_is_preserved(ctx: ToolContext) -> None:
    path = ctx.workspace / "crlf.txt"
    path.write_bytes(b"one\r\ntwo\r\n")
    read(ctx, file_path="crlf.txt")
    EditTool().run({"file_path": "crlf.txt", "old_string": "one", "new_string": "ONE"}, ctx)
    assert path.read_bytes() == b"ONE\r\ntwo\r\n"
