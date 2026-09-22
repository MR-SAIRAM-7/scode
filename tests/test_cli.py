from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeProvider, text_turn, tool_turn

from scode.cli import build_parser, main, run_print_mode, settings_from_args
from scode.config import load_settings
from scode.errors import ConfigError
from scode.ui.console import UI


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


# ------------------------------------------------------------------ parser

def test_prompt_words_are_joined(workspace: Path) -> None:
    args = parse(["fix", "the", "bug"])
    assert " ".join(args.prompt) == "fix the bug"


def test_flags_map_onto_settings(workspace: Path) -> None:
    args = parse([
        "-C", str(workspace),
        "--model", "a/b",
        "--temperature", "0.3",
        "--max-steps", "9",
        "--permission-mode", "plan",
        "--allowed-tools", "Bash(ls:*)",
        "--allowed-tools", "Read",
        "--disallowed-tools", "WebFetch",
        "--no-stream",
        "--no-native-tools",
        "-v",
    ])
    settings = settings_from_args(args)

    assert settings.model == "a/b"
    assert settings.temperature == 0.3
    assert settings.max_steps == 9
    assert settings.permission_mode == "plan"
    assert settings.allowed_tools == ("Bash(ls:*)", "Read")
    assert settings.denied_tools == ("WebFetch",)
    assert settings.stream is False
    assert settings.native_tools is False
    assert settings.verbose is True


def test_dangerously_skip_permissions(workspace: Path) -> None:
    args = parse(["-C", str(workspace), "--dangerously-skip-permissions"])
    assert settings_from_args(args).permission_mode == "bypassPermissions"


def test_missing_workspace_is_rejected(tmp_path: Path) -> None:
    args = parse(["-C", str(tmp_path / "nope")])
    with pytest.raises(ConfigError, match="not a directory"):
        settings_from_args(args)


def test_unset_flags_do_not_override_config(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCODE_MODEL", "env/model")
    args = parse(["-C", str(workspace)])
    assert settings_from_args(args).model == "env/model"


def test_version_exits_zero(capsys: pytest.CaptureFixture) -> None:
    assert main(["--version"]) == 0
    assert "scode" in capsys.readouterr().out


def test_bad_workspace_returns_two(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["-C", str(tmp_path / "missing")]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_print_mode_without_a_prompt_returns_two(workspace: Path, monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", None)
    assert main(["-C", str(workspace), "-p"]) == 2


# -------------------------------------------------------------- print mode

@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch):
    holder: dict = {}

    def install(turns):
        provider = FakeProvider(turns)
        holder["provider"] = provider
        monkeypatch.setattr("scode.providers.registry.get_provider", lambda *a, **k: provider)
        return provider

    return install


def test_print_mode_returns_the_answer(workspace: Path, fake_provider, capsys) -> None:
    fake_provider([text_turn("app.py defines add().")])
    settings = load_settings(workspace, {"api_key": "nvapi-x", "stream": False})

    code = run_print_mode(settings, UI(quiet=True), "what does app.py do?")
    assert code == 0
    assert "app.py defines add()." in capsys.readouterr().out


def test_print_mode_json_output(workspace: Path, fake_provider, capsys) -> None:
    fake_provider([text_turn("done")])
    settings = load_settings(workspace, {"api_key": "nvapi-x", "stream": False})

    run_print_mode(settings, UI(quiet=True), "hi", output_format="json")
    payload = json.loads(capsys.readouterr().out)

    assert payload["result"] == "done"
    assert payload["reason"] == "done"
    assert payload["model"] == settings.model
    assert payload["usage"]["requests"] == 1
    assert payload["session_id"]


def test_print_mode_denies_mutations_without_approval(workspace: Path, fake_provider) -> None:
    fake_provider([
        tool_turn("Write", {"file_path": "new.txt", "content": "x"}),
        text_turn("I could not write the file."),
    ])
    settings = load_settings(workspace, {"api_key": "nvapi-x", "stream": False})

    run_print_mode(settings, UI(quiet=True), "create new.txt")
    assert not (workspace / "new.txt").exists()


def test_print_mode_allows_mutations_when_told_to(workspace: Path, fake_provider) -> None:
    fake_provider([
        tool_turn("Write", {"file_path": "new.txt", "content": "hello\n"}),
        text_turn("Created it."),
    ])
    settings = load_settings(
        workspace, {"api_key": "nvapi-x", "stream": False, "permission_mode": "acceptEdits"}
    )

    run_print_mode(settings, UI(quiet=True), "create new.txt")
    assert (workspace / "new.txt").read_text(encoding="utf-8") == "hello\n"


def test_print_mode_needs_an_api_key(workspace: Path) -> None:
    settings = load_settings(workspace, {"stream": False})
    with pytest.raises(ConfigError, match="No API key"):
        run_print_mode(settings, UI(quiet=True), "hi")


def test_print_mode_writes_a_transcript(workspace: Path, fake_provider) -> None:
    from scode.session.store import list_sessions

    fake_provider([text_turn("hi")])
    settings = load_settings(workspace, {"api_key": "nvapi-x", "stream": False})

    run_print_mode(settings, UI(quiet=True), "hello")
    sessions = list_sessions(workspace)
    assert sessions and sessions[0].message_count == 2


def test_print_mode_expands_mentions(workspace: Path, fake_provider) -> None:
    provider = fake_provider([text_turn("ok")])
    settings = load_settings(workspace, {"api_key": "nvapi-x", "stream": False})

    run_print_mode(settings, UI(quiet=True), "explain @app.py")
    sent = provider.requests[0][-1]["content"]
    assert "def add(a, b):" in sent


def test_print_mode_nonzero_on_failure(workspace: Path, fake_provider, monkeypatch) -> None:
    from scode.errors import ProviderError

    provider = fake_provider([text_turn("never")])

    def boom(messages, **kwargs):
        raise ProviderError("down")

    provider.complete = boom  # type: ignore[assignment]
    settings = load_settings(workspace, {"api_key": "nvapi-x", "stream": False})
    assert run_print_mode(settings, UI(quiet=True), "hi") == 1


# ------------------------------------------------------------ informational

def test_list_sessions_flag(workspace: Path, capsys) -> None:
    from scode.session.store import SessionStore

    with SessionStore(workspace) as store:
        store.append({"role": "user", "content": "earlier work"})

    assert main(["-C", str(workspace), "--sessions"]) == 0
    assert "earlier work" in capsys.readouterr().out


def test_list_models_flag(workspace: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("scode.providers.registry.list_catalog", lambda s: ["x/one", "x/two"])
    assert main(["-C", str(workspace), "--list-models"]) == 0
    assert "x/one" in capsys.readouterr().out


def test_doctor_flags_a_missing_key(workspace: Path, monkeypatch) -> None:
    monkeypatch.setattr("scode.providers.registry.list_catalog", lambda s: ["a/b"])
    assert main(["-C", str(workspace), "--doctor"]) == 1


def test_doctor_passes_with_a_key(workspace: Path, monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    monkeypatch.setattr(
        "scode.providers.registry.list_catalog",
        lambda s: [load_settings(workspace).model],
    )
    assert main(["-C", str(workspace), "--doctor"]) == 0


def test_interactive_mode_does_not_eat_piped_stdin(workspace: Path, monkeypatch) -> None:
    """Regression: piped input was swallowed as the initial prompt."""
    import io

    from scode.cli import main

    monkeypatch.setattr("sys.stdin", io.StringIO("/exit\n"))
    captured: dict = {}

    def fake_interactive(settings, ui, prompt, **kwargs):
        captured["prompt"] = prompt
        return 0

    monkeypatch.setattr("scode.cli.run_interactive", fake_interactive)
    main(["-C", str(workspace)])
    assert captured["prompt"] == ""


def test_print_mode_still_reads_piped_stdin(workspace: Path, fake_provider, monkeypatch) -> None:
    import io

    provider = fake_provider([text_turn("ok")])
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    monkeypatch.setattr("sys.stdin", io.StringIO("summarise the repo"))
    main(["-C", str(workspace), "-p", "--no-stream"])
    assert "summarise the repo" in provider.requests[0][-1]["content"]
