from __future__ import annotations

import pytest

from scode.errors import ToolError
from scode.tools import ToolContext
from scode.tools.search_tools import GlobTool, GrepTool, LSTool


@pytest.fixture
def tree(ctx: ToolContext) -> ToolContext:
    root = ctx.workspace
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("import os\n\n\ndef run():\n    return 'TODO: wire up'\n", encoding="utf-8")
    (root / "src" / "util.py").write_text("def helper():\n    pass  # TODO later\n", encoding="utf-8")
    (root / "src" / "app.ts").write_text("export const x = 1;\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.py").write_text("TODO ignore me\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("# Guide\nTODO write this\n", encoding="utf-8")
    return ctx


# --------------------------------------------------------------------- Glob

def test_glob_finds_by_pattern(tree: ToolContext) -> None:
    result = GlobTool().run({"pattern": "**/*.py"}, tree)
    assert "src/main.py" in result.output
    assert "src/util.py" in result.output


def test_glob_skips_ignored_directories(tree: ToolContext) -> None:
    result = GlobTool().run({"pattern": "**/*.py"}, tree)
    assert "node_modules" not in result.output


def test_glob_reports_no_matches(tree: ToolContext) -> None:
    result = GlobTool().run({"pattern": "**/*.rs"}, tree)
    assert "No files match" in result.output


def test_glob_respects_a_subdirectory(tree: ToolContext) -> None:
    result = GlobTool().run({"pattern": "*.md", "path": "docs"}, tree)
    assert "guide.md" in result.output
    assert "main.py" not in result.output


def test_glob_rejects_a_missing_directory(tree: ToolContext) -> None:
    with pytest.raises(ToolError, match="not a directory"):
        GlobTool().run({"pattern": "*", "path": "nowhere"}, tree)


def test_glob_limit(tree: ToolContext) -> None:
    result = GlobTool().run({"pattern": "**/*.py", "limit": 1}, tree)
    assert "more matches not shown" in result.output


# --------------------------------------------------------------------- Grep

def _grep(tree: ToolContext, **args):
    # Force the built-in engine so the test is identical with or without ripgrep.
    return GrepTool()._run_python(
        str(args.pop("pattern")),
        tree.resolve(str(args.pop("path", tree.workspace))),
        args,
        tree,
        mode=str(args.get("output_mode") or "files_with_matches"),
        limit=int(args.get("head_limit") or 100),
        ignore_case=bool(args.get("-i")),
        show_numbers=args.get("-n") is not False,
        context_lines=int(args.get("-C") or 0),
    )


def test_grep_lists_matching_files(tree: ToolContext) -> None:
    result = _grep(tree, pattern="TODO")
    assert "src/main.py" in result.output
    assert "docs/guide.md" in result.output
    assert "node_modules" not in result.output


def test_grep_content_mode_has_line_numbers(tree: ToolContext) -> None:
    result = _grep(tree, pattern="def helper", output_mode="content")
    assert "src/util.py:1:def helper():" in result.output


def test_grep_count_mode(tree: ToolContext) -> None:
    result = _grep(tree, pattern="TODO", output_mode="count")
    assert any(line.endswith(":1") for line in result.output.splitlines())


def test_grep_case_insensitive(tree: ToolContext) -> None:
    assert "No matches" in _grep(tree, pattern="todo").output
    assert "src/main.py" in _grep(tree, pattern="todo", **{"-i": True}).output


def test_grep_glob_filter(tree: ToolContext) -> None:
    result = _grep(tree, pattern="TODO", glob="*.md")
    assert "docs/guide.md" in result.output
    assert "main.py" not in result.output


def test_grep_type_filter(tree: ToolContext) -> None:
    result = _grep(tree, pattern="TODO", type="md")
    assert "guide.md" in result.output
    assert "main.py" not in result.output


def test_grep_context_lines(tree: ToolContext) -> None:
    result = _grep(tree, pattern="def run", output_mode="content", **{"-C": 1})
    assert "src/main.py:3:" in result.output
    assert "src/main.py:5:" in result.output


def test_grep_rejects_a_bad_regex(tree: ToolContext) -> None:
    with pytest.raises(ToolError, match="Invalid regular expression"):
        _grep(tree, pattern="([")


def test_grep_rejects_an_unknown_mode(tree: ToolContext) -> None:
    with pytest.raises(ToolError, match="Unknown output_mode"):
        GrepTool().run({"pattern": "x", "output_mode": "weird"}, tree)


def test_grep_rejects_a_missing_path(tree: ToolContext) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        GrepTool().run({"pattern": "x", "path": "nope"}, tree)


# ----------------------------------------------------------------------- LS

def test_ls_lists_directories_first(tree: ToolContext) -> None:
    lines = LSTool().run({}, tree).output.splitlines()
    entries = [line.strip() for line in lines[1:]]
    assert entries[0].endswith("/")
    assert any(e.startswith("app.py") for e in entries)


def test_ls_hides_ignored_directories(tree: ToolContext) -> None:
    assert "node_modules" not in LSTool().run({}, tree).output


def test_ls_ignore_patterns(tree: ToolContext) -> None:
    output = LSTool().run({"ignore": ["*.md"]}, tree).output
    assert "README.md" not in output


def test_ls_rejects_a_file(tree: ToolContext) -> None:
    with pytest.raises(ToolError, match="not a directory"):
        LSTool().run({"path": "app.py"}, tree)


def test_ls_empty_directory(tree: ToolContext) -> None:
    (tree.workspace / "blank").mkdir()
    assert "is empty" in LSTool().run({"path": "blank"}, tree).output
