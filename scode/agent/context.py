"""Project context: environment block, memory files and @file mentions."""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

from ..constants import ALT_MEMORY_FILES, MEMORY_FILE, PROJECT_DIR_NAME, home_dir

MAX_MEMORY_CHARS = 24_000
MAX_MENTION_CHARS = 60_000


def _git(workspace: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_summary(workspace: Path) -> dict[str, str]:
    if _git(workspace, "rev-parse", "--is-inside-work-tree") != "true":
        return {}
    info: dict[str, str] = {"repo": "yes"}
    branch = _git(workspace, "rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        info["branch"] = branch
    status = _git(workspace, "status", "--porcelain")
    if status is not None:
        changed = [line for line in status.splitlines() if line.strip()]
        info["status"] = f"{len(changed)} changed file(s)" if changed else "clean"
    remote = _git(workspace, "remote", "get-url", "origin")
    if remote:
        info["remote"] = remote
    return info


def directory_snapshot(workspace: Path, *, limit: int = 30) -> str:
    from ..tools.search_tools import IGNORED_DIRS

    try:
        entries = sorted(
            (p for p in workspace.iterdir() if p.name not in IGNORED_DIRS and not p.name.startswith(".")),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except OSError:
        return ""
    names = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:limit]]
    if len(entries) > limit:
        names.append(f"... and {len(entries) - limit} more")
    return ", ".join(names)


def environment_block(workspace: Path, *, model: str, permission_mode: str) -> str:
    lines = [
        f"Working directory: {workspace}",
        f"Platform: {platform.system()} {platform.release()} ({sys.platform})",
        f"Python: {platform.python_version()}",
        f"Today: {date.today().isoformat()}",
        f"Model: {model}",
        f"Permission mode: {permission_mode}",
    ]
    git = git_summary(workspace)
    if git:
        lines.append(
            "Git: "
            + ", ".join(f"{key}={value}" for key, value in git.items() if key != "repo")
        )
    else:
        lines.append("Git: not a repository")

    snapshot = directory_snapshot(workspace)
    if snapshot:
        lines.append(f"Top level: {snapshot}")
    return "\n".join(lines)


def memory_files(workspace: Path) -> list[Path]:
    """Instruction files, nearest last so the closest one wins."""
    found: list[Path] = []

    user_memory = home_dir() / MEMORY_FILE
    if user_memory.is_file():
        found.append(user_memory)

    for name in (MEMORY_FILE, *ALT_MEMORY_FILES):
        candidate = workspace / name
        if candidate.is_file():
            found.append(candidate)
            break

    nested = workspace / PROJECT_DIR_NAME / MEMORY_FILE
    if nested.is_file():
        found.append(nested)
    return found


def load_project_memory(workspace: Path) -> str:
    chunks = []
    for path in memory_files(workspace):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        chunks.append(f"--- from {path} ---\n{text}")
    joined = "\n\n".join(chunks)
    if len(joined) > MAX_MEMORY_CHARS:
        joined = joined[:MAX_MEMORY_CHARS] + "\n\n[instruction file truncated]"
    return joined


def append_memory(workspace: Path, note: str, *, user_level: bool = False) -> Path:
    """Append a `#` note to the project (or user) instruction file."""
    path = (home_dir() / MEMORY_FILE) if user_level else (workspace / MEMORY_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if not existing:
        existing = f"# {workspace.name}\n\nProject instructions for scode.\n"
    if not existing.endswith("\n"):
        existing += "\n"
    path.write_text(f"{existing}\n- {note.strip()}\n", encoding="utf-8")
    return path


MENTION_RE = re.compile(r"(?:^|(?<=\s))@([\w./\\~-]+)")
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def expand_file_mentions(
    text: str,
    workspace: Path,
    *,
    allow_images: bool = True,
) -> tuple[str | list[dict[str, str]], list[str]]:
    """Attach @-mentioned files to a message, as Claude Code does.

    Text files are inlined. Images become image parts, so the result is a
    part list instead of a string when any image is attached.
    """
    import base64

    from ..errors import ToolError
    from ..providers.base import image_part
    from ..tools.file_tools import read_text

    attached: list[str] = []
    images: list[dict[str, str]] = []
    budget = MAX_MENTION_CHARS

    for match in MENTION_RE.finditer(text):
        raw = match.group(1)
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate
        if not candidate.is_file():
            continue
        try:
            display = candidate.resolve().relative_to(workspace).as_posix()
        except ValueError:
            display = str(candidate)

        media_type = IMAGE_TYPES.get(candidate.suffix.lower())
        if media_type:
            if not allow_images:
                continue
            try:
                data = candidate.read_bytes()
            except OSError:
                continue
            if len(data) > MAX_IMAGE_BYTES:
                continue
            images.append(image_part(media_type, base64.b64encode(data).decode("ascii")))
            attached.append(display)
            continue

        try:
            content = read_text(candidate)
        except (ToolError, OSError):
            continue
        if len(content) > budget:
            content = content[:budget] + "\n[truncated]"
        budget -= len(content)
        attached.append(display)
        text += f"\n\n<attached path=\"{display}\">\n{content}\n</attached>"
        if budget <= 0:
            break

    if images:
        return [{"type": "text", "text": text}, *images], attached
    return text, attached
