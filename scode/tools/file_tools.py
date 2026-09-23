"""Read, Write, Edit and MultiEdit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..constants import MAX_LINE_CHARS, MAX_READ_LINES
from ..errors import ToolError
from ..permissions import PermissionRequest
from .base import Tool, ToolContext, ToolResult

TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg"}


def read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ToolError(f"Could not read {path}: {exc}") from exc
    if b"\x00" in raw[:8192]:
        raise ToolError(
            f"{path.name} looks like a binary file, so it cannot be read as text."
        )
    for encoding in TEXT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # newline="" keeps whatever line endings the model produced.
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
    except OSError as exc:
        raise ToolError(f"Could not write {path}: {exc}") from exc


def unified_diff(before: str, after: str, path: str) -> str:
    import difflib

    lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    return "".join(lines)


def _count_changes(diff: str) -> tuple[int, int]:
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    return added, removed


def _require_prior_read(path: Path, ctx: ToolContext) -> None:
    """Editing a file the agent has not read is how models clobber work."""
    key = str(path)
    if key not in ctx.read_files:
        raise ToolError(
            f"Read {ctx.display_path(path)} before editing it, so the edit is based on "
            "the file's current contents."
        )
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return
    if mtime > ctx.read_files[key] + 1e-6:
        raise ToolError(
            f"{ctx.display_path(path)} changed on disk since it was read. "
            "Read it again before editing."
        )


def _mark_read(path: Path, ctx: ToolContext) -> None:
    try:
        ctx.read_files[str(path)] = path.stat().st_mtime
    except OSError:
        ctx.read_files[str(path)] = 0.0


def format_numbered(text: str, *, start: int = 1) -> str:
    out = []
    for offset, line in enumerate(text.splitlines()):
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + f"... [{len(line) - MAX_LINE_CHARS} more chars]"
        out.append(f"{start + offset:6d}\t{line}")
    return "\n".join(out)


class ReadTool(Tool):
    name = "Read"
    description = (
        "Read a file from the filesystem. Returns contents with line numbers in "
        "`cat -n` format. Use offset/limit for large files. Always read a file "
        "before editing it."
    )
    mutating = False
    verb = "Reading"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute or workspace-relative path"},
            "offset": {"type": "integer", "description": "1-based line to start from"},
            "limit": {"type": "integer", "description": "How many lines to read"},
        },
        "required": ["file_path"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"Read({ctx.display_path(ctx.resolve(str(args.get('file_path', ''))))})"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        path = ctx.resolve(str(args["file_path"]))
        if not path.exists():
            raise ToolError(f"{ctx.display_path(path)} does not exist")
        if path.is_dir():
            raise ToolError(
                f"{ctx.display_path(path)} is a directory. Use LS or Glob instead."
            )
        if path.suffix.lower() in IMAGE_SUFFIXES and path.suffix.lower() != ".svg":
            _mark_read(path, ctx)
            size = path.stat().st_size
            return ToolResult(
                output=f"[image file: {path.name}, {size} bytes — not rendered as text]",
                display=f"Read {ctx.display_path(path)} (image)",
            )

        text = read_text(path)
        _mark_read(path, ctx)

        if text == "":
            return ToolResult(
                output="[file is empty]",
                display=f"Read {ctx.display_path(path)} (empty)",
            )

        lines = text.splitlines()
        offset = max(1, int(args.get("offset") or 1))
        limit = int(args.get("limit") or MAX_READ_LINES)
        limit = max(1, min(limit, MAX_READ_LINES))
        window = lines[offset - 1 : offset - 1 + limit]
        if not window:
            raise ToolError(
                f"Offset {offset} is past the end of {ctx.display_path(path)} "
                f"({len(lines)} lines)"
            )

        body = format_numbered("\n".join(window), start=offset)
        shown_to = offset - 1 + len(window)
        if shown_to < len(lines):
            body += (
                f"\n\n[showing lines {offset}-{shown_to} of {len(lines)}; "
                f"use offset={shown_to + 1} to continue]"
            )
        return ToolResult(
            output=body,
            display=f"Read {ctx.display_path(path)} ({len(window)} lines)",
            metadata={"lines": len(lines)},
        )


class WriteTool(Tool):
    name = "Write"
    description = (
        "Write a file, replacing it entirely if it exists. Read an existing file "
        "before overwriting it. Prefer Edit for changing part of a file."
    )
    verb = "Writing"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string", "description": "Full contents to write"},
        },
        "required": ["file_path", "content"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"Write({ctx.display_path(ctx.resolve(str(args.get('file_path', ''))))})"

    def permission_request(self, args: dict[str, Any], ctx: ToolContext) -> PermissionRequest:
        path = ctx.resolve(str(args.get("file_path", "")))
        display = ctx.display_path(path)
        exists = path.exists()
        before = read_text(path) if exists and path.is_file() else ""
        diff = unified_diff(before, str(args.get("content", "")), display)
        outside = "" if ctx.inside_workspace(path) else "  ⚠ outside the workspace\n"
        return PermissionRequest(
            tool=self.name,
            specifier=display,
            title=("Overwrite " if exists else "Create ") + display,
            detail=outside + diff,
            mutating=True,
            suggestions=(f"Write({display})", "Write"),
        )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        path = ctx.resolve(str(args["file_path"]))
        content = str(args.get("content", ""))
        if path.is_dir():
            raise ToolError(f"{ctx.display_path(path)} is a directory")

        existed = path.is_file()
        before = read_text(path) if existed else ""
        if existed:
            _require_prior_read(path, ctx)

        ctx.checkpoint(path)
        write_text(path, content)
        _mark_read(path, ctx)

        diff = unified_diff(before, content, ctx.display_path(path))
        added, removed = _count_changes(diff)
        verb = "Updated" if existed else "Created"
        line_count = len(content.splitlines())
        return ToolResult(
            output=(
                f"{verb} {ctx.display_path(path)} ({line_count} lines, "
                f"+{added}/-{removed})"
            ),
            display=f"{verb} {ctx.display_path(path)}",
            detail=diff,
            metadata={"added": added, "removed": removed, "path": str(path)},
        )


def _apply_replacement(
    text: str,
    old: str,
    new: str,
    *,
    replace_all: bool,
    display: str,
) -> str:
    if old == new:
        raise ToolError("old_string and new_string are identical; nothing to change")
    occurrences = text.count(old)
    if occurrences == 0:
        hint = ""
        stripped = old.strip()
        if stripped and stripped in text:
            hint = " (the text is present but the surrounding whitespace differs)"
        elif old.replace("\r\n", "\n") in text.replace("\r\n", "\n"):
            hint = " (line endings differ)"
        raise ToolError(
            f"old_string was not found in {display}{hint}. "
            "Read the file again and copy the exact text, including indentation."
        )
    if occurrences > 1 and not replace_all:
        raise ToolError(
            f"old_string appears {occurrences} times in {display}. "
            "Add surrounding context to make it unique, or pass replace_all=true."
        )
    return text.replace(old, new) if replace_all else text.replace(old, new, 1)


class EditTool(Tool):
    name = "Edit"
    description = (
        "Replace an exact string in a file. old_string must match the file "
        "byte-for-byte, including indentation, and must be unique unless "
        "replace_all is true. Read the file first."
    )
    verb = "Editing"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence"},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"Edit({ctx.display_path(ctx.resolve(str(args.get('file_path', ''))))})"

    def _preview(self, args: dict[str, Any], ctx: ToolContext) -> tuple[Path, str, str]:
        path = ctx.resolve(str(args.get("file_path", "")))
        display = ctx.display_path(path)
        if not path.is_file():
            raise ToolError(f"{display} does not exist")
        before = read_text(path)
        after = _apply_replacement(
            before,
            str(args.get("old_string", "")),
            str(args.get("new_string", "")),
            replace_all=bool(args.get("replace_all")),
            display=display,
        )
        return path, before, after

    def permission_request(self, args: dict[str, Any], ctx: ToolContext) -> PermissionRequest:
        path = ctx.resolve(str(args.get("file_path", "")))
        display = ctx.display_path(path)
        try:
            _, before, after = self._preview(args, ctx)
            detail = unified_diff(before, after, display)
        except ToolError as exc:
            detail = f"(could not preview: {exc})"
        return PermissionRequest(
            tool=self.name,
            specifier=display,
            title=f"Edit {display}",
            detail=detail,
            mutating=True,
            suggestions=(f"Edit({display})", "Edit"),
        )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        path, before, after = self._preview(args, ctx)
        _require_prior_read(path, ctx)

        ctx.checkpoint(path)
        write_text(path, after)
        _mark_read(path, ctx)

        display = ctx.display_path(path)
        diff = unified_diff(before, after, display)
        added, removed = _count_changes(diff)
        return ToolResult(
            output=f"Edited {display} (+{added}/-{removed})",
            display=f"Edited {display}",
            detail=diff,
            metadata={"added": added, "removed": removed, "path": str(path)},
        )


class MultiEditTool(Tool):
    name = "MultiEdit"
    description = (
        "Apply several exact-string edits to one file in order. All edits must "
        "succeed or none are written. Read the file first."
    )
    verb = "Editing"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "edits": {
                "type": "array",
                "description": "Edits applied in sequence",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": ["file_path", "edits"],
    }

    def summarize_call(self, args: dict[str, Any], ctx: ToolContext) -> str:
        count = len(args.get("edits") or [])
        path = ctx.display_path(ctx.resolve(str(args.get("file_path", ""))))
        return f"MultiEdit({path}, {count} edits)"

    def _preview(self, args: dict[str, Any], ctx: ToolContext) -> tuple[Path, str, str]:
        path = ctx.resolve(str(args.get("file_path", "")))
        display = ctx.display_path(path)
        edits = args.get("edits") or []
        if not isinstance(edits, list) or not edits:
            raise ToolError("edits must be a non-empty array")
        if not path.is_file():
            raise ToolError(f"{display} does not exist")

        before = read_text(path)
        text = before
        for index, edit in enumerate(edits, start=1):
            if not isinstance(edit, dict):
                raise ToolError(f"edits[{index}] must be an object")
            try:
                text = _apply_replacement(
                    text,
                    str(edit.get("old_string", "")),
                    str(edit.get("new_string", "")),
                    replace_all=bool(edit.get("replace_all")),
                    display=display,
                )
            except ToolError as exc:
                raise ToolError(f"edit {index} of {len(edits)} failed: {exc}") from exc
        return path, before, text

    def permission_request(self, args: dict[str, Any], ctx: ToolContext) -> PermissionRequest:
        path = ctx.resolve(str(args.get("file_path", "")))
        display = ctx.display_path(path)
        try:
            _, before, after = self._preview(args, ctx)
            detail = unified_diff(before, after, display)
        except ToolError as exc:
            detail = f"(could not preview: {exc})"
        return PermissionRequest(
            tool=self.name,
            specifier=display,
            title=f"Edit {display} ({len(args.get('edits') or [])} changes)",
            detail=detail,
            mutating=True,
            suggestions=(f"MultiEdit({display})", "MultiEdit"),
        )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.validate(args)
        path, before, after = self._preview(args, ctx)
        _require_prior_read(path, ctx)

        ctx.checkpoint(path)
        write_text(path, after)
        _mark_read(path, ctx)

        display = ctx.display_path(path)
        diff = unified_diff(before, after, display)
        added, removed = _count_changes(diff)
        count = len(args.get("edits") or [])
        return ToolResult(
            output=f"Applied {count} edits to {display} (+{added}/-{removed})",
            display=f"Edited {display} ({count} changes)",
            detail=diff,
            metadata={"added": added, "removed": removed, "path": str(path)},
        )


def relative_or_absolute(path: Path, workspace: Path) -> str:
    try:
        return os.path.relpath(path, workspace).replace(os.sep, "/")
    except ValueError:
        return str(path)
