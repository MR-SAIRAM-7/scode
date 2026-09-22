"""Glob, Grep and LS."""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..errors import ToolError
from .base import Tool, ToolContext, ToolResult
from .file_tools import read_text

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".next",
    ".tox",
    ".idea",
    ".gradle",
    "target",
}

FILE_TYPES = {
    "py": ("*.py", "*.pyi"),
    "js": ("*.js", "*.jsx", "*.mjs", "*.cjs"),
    "ts": ("*.ts", "*.tsx", "*.mts"),
    "rust": ("*.rs",),
    "go": ("*.go",),
    "java": ("*.java",),
    "c": ("*.c", "*.h"),
    "cpp": ("*.cpp", "*.cc", "*.hpp", "*.hh", "*.cxx"),
    "cs": ("*.cs",),
    "rb": ("*.rb",),
    "php": ("*.php",),
    "sh": ("*.sh", "*.bash", "*.zsh"),
    "json": ("*.json",),
    "yaml": ("*.yaml", "*.yml"),
    "toml": ("*.toml",),
    "md": ("*.md", "*.markdown"),
    "html": ("*.html", "*.htm"),
    "css": ("*.css", "*.scss", "*.sass"),
    "sql": ("*.sql",),
}


def _is_ignored(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part in IGNORED_DIRS for part in parts)


def walk_files(root: Path, *, limit: int = 200_000) -> Iterator[Path]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        base = Path(dirpath)
        for name in filenames:
            yield base / name
            count += 1
            if count >= limit:
                return


class GlobTool(Tool):
    name = "Glob"
    description = (
        "Find files by glob pattern (e.g. **/*.py, src/**/test_*.ts). "
        "Returns paths sorted by modification time, newest first."
    )
    mutating = False
    verb = "Searching"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "path": {"type": "string", "description": "Directory to search (default: workspace)"},
            "limit": {"type": "integer", "description": "Max results (default 200)"},
        },
        "required": ["pattern"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        root = ctx.resolve(str(args.get("path") or ctx.workspace))
        if not root.is_dir():
            raise ToolError(f"{ctx.display_path(root)} is not a directory")
        pattern = str(args["pattern"])
        limit = max(1, min(int(args.get("limit") or 200), 2000))

        try:
            matches = [
                p
                for p in root.glob(pattern)
                if p.is_file() and not _is_ignored(p, root)
            ]
        except (ValueError, OSError) as exc:
            raise ToolError(f"Invalid glob pattern {pattern!r}: {exc}") from exc

        matches.sort(key=lambda p: _safe_mtime(p), reverse=True)
        shown = matches[:limit]
        if not shown:
            return ToolResult(
                output=f"No files match {pattern!r} under {ctx.display_path(root)}",
                display=f"Glob {pattern} — no matches",
            )

        listing = "\n".join(ctx.display_path(p) for p in shown)
        if len(matches) > limit:
            listing += f"\n\n[{len(matches) - limit} more matches not shown]"
        return ToolResult(
            output=listing,
            display=f"Glob {pattern} — {len(matches)} file(s)",
            metadata={"count": len(matches)},
        )


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


class GrepTool(Tool):
    name = "Grep"
    description = (
        "Search file contents with a regular expression. Uses ripgrep when it is "
        "installed and falls back to a built-in engine. output_mode can be "
        "'content', 'files_with_matches' (default) or 'count'."
    )
    mutating = False
    verb = "Searching"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression"},
            "path": {"type": "string", "description": "File or directory to search"},
            "glob": {"type": "string", "description": "Only search files matching this glob"},
            "type": {
                "type": "string",
                "description": "File type filter: " + ", ".join(sorted(FILE_TYPES)),
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
            },
            "-i": {"type": "boolean", "description": "Case insensitive"},
            "-n": {"type": "boolean", "description": "Show line numbers with content mode"},
            "-C": {"type": "integer", "description": "Context lines around each match"},
            "head_limit": {"type": "integer", "description": "Max results (default 100)"},
        },
        "required": ["pattern"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"Grep({args.get('pattern', '')})"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        pattern = str(args["pattern"])
        root = ctx.resolve(str(args.get("path") or ctx.workspace))
        mode = str(args.get("output_mode") or "files_with_matches")
        if mode not in {"content", "files_with_matches", "count"}:
            raise ToolError(f"Unknown output_mode {mode!r}")
        limit = max(1, min(int(args.get("head_limit") or 100), 1000))
        ignore_case = bool(args.get("-i"))
        show_numbers = args.get("-n") is not False
        context_lines = max(0, int(args.get("-C") or 0))

        if not root.exists():
            raise ToolError(f"{ctx.display_path(root)} does not exist")

        rg = shutil.which("rg")
        if rg:
            try:
                return self._run_ripgrep(
                    rg, pattern, root, args, ctx,
                    mode=mode, limit=limit, ignore_case=ignore_case,
                    show_numbers=show_numbers, context_lines=context_lines,
                )
            except (OSError, subprocess.SubprocessError):
                pass  # fall through to the pure-Python path

        return self._run_python(
            pattern, root, args, ctx,
            mode=mode, limit=limit, ignore_case=ignore_case,
            show_numbers=show_numbers, context_lines=context_lines,
        )

    # --------------------------------------------------------------- engines
    def _run_ripgrep(
        self,
        rg: str,
        pattern: str,
        root: Path,
        args: dict[str, Any],
        ctx: ToolContext,
        *,
        mode: str,
        limit: int,
        ignore_case: bool,
        show_numbers: bool,
        context_lines: int,
    ) -> ToolResult:
        cmd = [rg, "--color=never", "--no-heading"]
        if ignore_case:
            cmd.append("-i")
        if mode == "files_with_matches":
            cmd.append("-l")
        elif mode == "count":
            cmd.append("-c")
        else:
            if show_numbers:
                cmd.append("-n")
            if context_lines:
                cmd += ["-C", str(context_lines)]
        if args.get("glob"):
            cmd += ["-g", str(args["glob"])]
        if args.get("type"):
            for suffix_glob in FILE_TYPES.get(str(args["type"]), ()):
                cmd += ["-g", suffix_glob]
        cmd += ["-e", pattern, str(root)]

        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        )
        if completed.returncode not in (0, 1):
            raise subprocess.SubprocessError(completed.stderr.strip() or "ripgrep failed")

        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        return self._format(lines, pattern, ctx, mode=mode, limit=limit, root=root)

    def _run_python(
        self,
        pattern: str,
        root: Path,
        args: dict[str, Any],
        ctx: ToolContext,
        *,
        mode: str,
        limit: int,
        ignore_case: bool,
        show_numbers: bool,
        context_lines: int,
    ) -> ToolResult:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"Invalid regular expression {pattern!r}: {exc}") from exc

        globs: tuple[str, ...] = ()
        if args.get("glob"):
            globs = (str(args["glob"]),)
        elif args.get("type"):
            globs = FILE_TYPES.get(str(args["type"]), ())

        candidates = [root] if root.is_file() else list(walk_files(root))
        results: list[str] = []
        scanned = 0

        for path in candidates:
            if globs and not any(
                fnmatch.fnmatch(path.name, g) or fnmatch.fnmatch(str(path), g) for g in globs
            ):
                continue
            try:
                if path.stat().st_size > 8_000_000:
                    continue
                text = read_text(path)
            except (ToolError, OSError):
                continue
            scanned += 1

            lines = text.splitlines()
            hits = [i for i, line in enumerate(lines) if regex.search(line)]
            if not hits:
                continue

            display = ctx.display_path(path)
            if mode == "files_with_matches":
                results.append(display)
            elif mode == "count":
                results.append(f"{display}:{len(hits)}")
            else:
                emitted: set[int] = set()
                for index in hits:
                    low = max(0, index - context_lines)
                    high = min(len(lines), index + context_lines + 1)
                    for i in range(low, high):
                        if i in emitted:
                            continue
                        emitted.add(i)
                        prefix = f"{display}:{i + 1}:" if show_numbers else f"{display}:"
                        results.append(prefix + lines[i])
                    if len(results) >= limit:
                        break
            if len(results) >= limit and mode == "content":
                break

        return self._format(results, pattern, ctx, mode=mode, limit=limit, root=root)

    def _format(
        self,
        lines: list[str],
        pattern: str,
        ctx: ToolContext,
        *,
        mode: str,
        limit: int,
        root: Path,
    ) -> ToolResult:
        if not lines:
            return ToolResult(
                output=f"No matches for {pattern!r} in {ctx.display_path(root)}",
                display=f"Grep {pattern} — no matches",
            )
        shown = lines[:limit]
        body = "\n".join(shown)
        if len(lines) > limit:
            body += f"\n\n[{len(lines) - limit} more results not shown]"
        noun = "file" if mode == "files_with_matches" else "match"
        return ToolResult(
            output=body,
            display=f"Grep {pattern} — {len(lines)} {noun}(es)",
            metadata={"count": len(lines)},
        )


class LSTool(Tool):
    name = "LS"
    description = "List the entries of a directory, directories first."
    mutating = False
    verb = "Listing"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list"},
            "ignore": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Glob patterns to skip",
            },
        },
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"LS({args.get('path') or '.'})"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.resolve(str(args.get("path") or ctx.workspace))
        if not root.exists():
            raise ToolError(f"{ctx.display_path(root)} does not exist")
        if not root.is_dir():
            raise ToolError(f"{ctx.display_path(root)} is not a directory")

        ignore = [str(p) for p in (args.get("ignore") or [])]
        entries = []
        try:
            for entry in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                if entry.name in IGNORED_DIRS:
                    continue
                if any(fnmatch.fnmatch(entry.name, pattern) for pattern in ignore):
                    continue
                if entry.is_dir():
                    entries.append(f"{entry.name}/")
                else:
                    entries.append(f"{entry.name}  ({_human_size(entry)})")
        except OSError as exc:
            raise ToolError(f"Could not list {ctx.display_path(root)}: {exc}") from exc

        if not entries:
            return ToolResult(
                output=f"{ctx.display_path(root)} is empty",
                display=f"LS {ctx.display_path(root)} — empty",
            )
        listing = f"{ctx.display_path(root)}/\n" + "\n".join(f"  {e}" for e in entries)
        return ToolResult(
            output=listing,
            display=f"LS {ctx.display_path(root)} — {len(entries)} entries",
        )


def _human_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"
