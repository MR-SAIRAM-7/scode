"""Transcripts on disk, so --continue and --resume work."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..constants import projects_dir


def slugify(path: Path) -> str:
    text = str(path).replace("\\", "/")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug[:120] or "root"


def session_dir(workspace: Path) -> Path:
    return projects_dir() / slugify(workspace)


@dataclass
class Session:
    id: str
    workspace: Path
    created: float
    updated: float
    model: str = ""
    title: str = ""
    message_count: int = 0
    path: Path | None = None

    @property
    def age(self) -> str:
        seconds = max(0.0, time.time() - self.updated)
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{int(seconds // 60)}m ago"
        if seconds < 86400:
            return f"{int(seconds // 3600)}h ago"
        return f"{int(seconds // 86400)}d ago"


class SessionStore:
    """Append-only JSONL transcript for one session."""

    def __init__(self, workspace: Path, session_id: str | None = None, *, model: str = "") -> None:
        self.workspace = workspace
        self.id = session_id or uuid.uuid4().hex[:16]
        self.directory = session_dir(workspace)
        self.path = self.directory / f"{self.id}.jsonl"
        self.model = model
        self.title = ""
        self._handle = None
        self._enabled = True

    # ------------------------------------------------------------------ io
    def open(self) -> SessionStore:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            is_new = not self.path.exists()
            self._handle = open(self.path, "a", encoding="utf-8")
            if is_new:
                self._write(
                    {
                        "type": "meta",
                        "id": self.id,
                        "cwd": str(self.workspace),
                        "created": time.time(),
                        "model": self.model,
                    }
                )
        except OSError:
            # A read-only home directory should not stop the CLI from working.
            self._enabled = False
        return self

    def _write(self, record: dict[str, Any]) -> None:
        if not self._enabled or self._handle is None:
            return
        try:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()
        except (OSError, TypeError, ValueError):
            self._enabled = False

    def append(self, message: dict[str, Any]) -> None:
        if not self.title and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                self.title = " ".join(content.split())[:80]
                self._write({"type": "title", "title": self.title})
        self._write({"type": "message", "message": message, "at": time.time()})

    def record_event(self, kind: str, **payload: Any) -> None:
        self._write({"type": "event", "kind": kind, "at": time.time(), **payload})

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None

    def __enter__(self) -> SessionStore:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------------------- loading
    @staticmethod
    def read_messages(path: Path) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for record in _read_records(path):
            if record.get("type") == "message":
                message = record.get("message")
                if isinstance(message, dict) and message.get("role"):
                    messages.append(message)
        return messages

    @classmethod
    def resume(cls, workspace: Path, session_id: str, *, model: str = "") -> tuple[SessionStore, list[dict[str, Any]]]:
        path = session_dir(workspace) / f"{session_id}.jsonl"
        if not path.is_file():
            match = _find_by_prefix(workspace, session_id)
            if match is None:
                raise FileNotFoundError(f"No session {session_id!r} for {workspace}")
            path = match
            session_id = path.stem
        store = cls(workspace, session_id, model=model)
        messages = cls.read_messages(path)
        store.open()
        store.record_event("resumed")
        return store, messages


def _read_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
    except OSError:
        return


def _find_by_prefix(workspace: Path, prefix: str) -> Path | None:
    directory = session_dir(workspace)
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"{prefix}*.jsonl"))
    return matches[0] if matches else None


def list_sessions(workspace: Path, *, limit: int = 20) -> list[Session]:
    directory = session_dir(workspace)
    if not directory.is_dir():
        return []

    sessions: list[Session] = []
    for path in directory.glob("*.jsonl"):
        meta: dict[str, Any] = {}
        title = ""
        count = 0
        last = 0.0
        for record in _read_records(path):
            kind = record.get("type")
            if kind == "meta":
                meta = record
            elif kind == "title":
                title = str(record.get("title") or "")
            elif kind == "message":
                count += 1
                last = float(record.get("at") or last)
        if not meta and count == 0:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = last
        sessions.append(
            Session(
                id=str(meta.get("id") or path.stem),
                workspace=workspace,
                created=float(meta.get("created") or mtime),
                updated=last or mtime,
                model=str(meta.get("model") or ""),
                title=title,
                message_count=count,
                path=path,
            )
        )

    sessions.sort(key=lambda s: s.updated, reverse=True)
    return sessions[:limit]


def latest_session(workspace: Path) -> Session | None:
    sessions = list_sessions(workspace, limit=1)
    return sessions[0] if sessions else None
