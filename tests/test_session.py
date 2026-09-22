from __future__ import annotations

from pathlib import Path

import pytest

from scode.session.store import (
    SessionStore,
    latest_session,
    list_sessions,
    session_dir,
    slugify,
)


def test_slugify_is_filesystem_safe() -> None:
    slug = slugify(Path("D:/PROJECTS/my app"))
    assert "/" not in slug and "\\" not in slug and ":" not in slug
    assert slug == "d-projects-my-app"


def test_messages_round_trip(workspace: Path) -> None:
    with SessionStore(workspace, model="m/1") as store:
        store.append({"role": "user", "content": "hello"})
        store.append({"role": "assistant", "content": "hi"})
        path = store.path

    assert SessionStore.read_messages(path) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_tool_messages_survive(workspace: Path) -> None:
    with SessionStore(workspace) as store:
        store.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
        ]})
        store.append({"role": "tool", "tool_call_id": "c1", "name": "Read", "content": "text"})
        path = store.path

    messages = SessionStore.read_messages(path)
    assert messages[0]["tool_calls"][0]["function"]["name"] == "Read"
    assert messages[1]["role"] == "tool"


def test_title_comes_from_the_first_user_message(workspace: Path) -> None:
    with SessionStore(workspace) as store:
        store.append({"role": "user", "content": "fix   the\nbug please"})
        store.append({"role": "user", "content": "second"})
        assert store.title == "fix the bug please"

    session = list_sessions(workspace)[0]
    assert session.title == "fix the bug please"


def test_resume_restores_messages(workspace: Path) -> None:
    with SessionStore(workspace) as store:
        store.append({"role": "user", "content": "first"})
        session_id = store.id

    resumed, messages = SessionStore.resume(workspace, session_id)
    resumed.close()
    assert [m["content"] for m in messages] == ["first"]


def test_resume_accepts_an_id_prefix(workspace: Path) -> None:
    with SessionStore(workspace) as store:
        store.append({"role": "user", "content": "x"})
        session_id = store.id

    resumed, messages = SessionStore.resume(workspace, session_id[:6])
    resumed.close()
    assert resumed.id == session_id
    assert len(messages) == 1


def test_resume_rejects_an_unknown_id(workspace: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SessionStore.resume(workspace, "doesnotexist")


def test_list_sessions_is_newest_first(workspace: Path) -> None:
    import time

    ids = []
    for index in range(3):
        with SessionStore(workspace) as store:
            store.append({"role": "user", "content": f"session {index}"})
            ids.append(store.id)
        time.sleep(0.01)

    listed = [s.id for s in list_sessions(workspace)]
    assert listed[0] == ids[-1]
    assert set(listed) == set(ids)


def test_list_sessions_counts_messages(workspace: Path) -> None:
    with SessionStore(workspace) as store:
        for index in range(4):
            store.append({"role": "user", "content": str(index)})

    assert list_sessions(workspace)[0].message_count == 4


def test_latest_session(workspace: Path) -> None:
    assert latest_session(workspace) is None
    with SessionStore(workspace) as store:
        store.append({"role": "user", "content": "x"})
        expected = store.id
    assert latest_session(workspace).id == expected


def test_sessions_are_scoped_per_directory(workspace: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    with SessionStore(workspace) as store:
        store.append({"role": "user", "content": "here"})

    assert len(list_sessions(workspace)) == 1
    assert list_sessions(other) == []


def test_corrupt_lines_are_skipped(workspace: Path) -> None:
    with SessionStore(workspace) as store:
        store.append({"role": "user", "content": "good"})
        path = store.path

    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{ not json\n")
        handle.write("\n")

    assert len(SessionStore.read_messages(path)) == 1


def test_an_unwritable_home_does_not_crash(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "mkdir", boom)
    store = SessionStore(workspace).open()
    store.append({"role": "user", "content": "still works"})
    store.close()
    assert store._enabled is False


def test_events_are_recorded(workspace: Path) -> None:
    with SessionStore(workspace) as store:
        store.record_event("model_changed", model="a/b")
        path = store.path
    assert "model_changed" in path.read_text(encoding="utf-8")


def test_session_dir_is_under_the_scode_home(workspace: Path, isolated_home: Path) -> None:
    assert isolated_home in session_dir(workspace).parents
