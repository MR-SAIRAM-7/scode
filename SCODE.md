# scode

An agentic coding CLI (a Claude Code-style terminal assistant) that runs on NVIDIA NIM
models. Pure Python, no framework.

## Commands

```bash
pytest                          # full suite (351 tests)
pytest tests/test_agent.py -q   # one file
ruff check scode tests          # lint
ruff check scode tests --fix    # autofix
pip install -e ".[dev]"         # dev install
python -m scode --doctor        # check the environment and provider
```

There is no build step. `scode` is installed as a console script via
`[project.scripts]` in `pyproject.toml`.

## Layout

| Path | Holds |
| --- | --- |
| `scode/cli.py` | argparse entry point, `--print` mode, `--doctor`/`--list-models` |
| `scode/repl.py` | the interactive loop; owns the provider, agent, session and permission asker |
| `scode/config.py` | `Settings` and the layered loader (user → project → local → env → flags) |
| `scode/permissions.py` | rule parsing (`Bash(git status:*)`), the four modes, the approval flow |
| `scode/agent/loop.py` | the agent loop: request → tool calls → repeat |
| `scode/agent/prompts.py` | system prompts, including the no-function-calling fallback protocol |
| `scode/agent/context.py` | environment block, `SCODE.md` loading, `@file` expansion |
| `scode/agent/compact.py` | context compaction |
| `scode/providers/` | OpenAI-compatible client: SSE streaming, tool-call assembly, retries |
| `scode/tools/` | one module per tool family; `build_registry()` assembles them |
| `scode/session/store.py` | JSONL transcripts under `~/.scode/projects/<slug>/` |
| `scode/commands/builtin.py` | slash commands, registered with the `@command` decorator |
| `scode/ui/` | rich console, streaming renderer, diffs, prompt_toolkit input, glyphs |

## Conventions

- `from __future__ import annotations` at the top of every module.
- Errors that users should see subclass `ScodeError` (`scode/errors.py`). A `ToolError`
  raised inside a tool is caught by the agent loop and fed back to the model as text — it
  never crashes the turn. Anything else is a bug.
- Tools subclass `Tool` and declare `name`, `description`, `parameters` (JSON Schema) and
  `mutating`. Register them in `scode/tools/__init__.py`.
- `ToolResult.output` goes to the model; `.display` is the one-line summary for the user;
  `.detail` is optional rich content such as a diff.
- All terminal output goes through `scode/ui/console.py::UI`. Never `print()` directly —
  `UI` handles quiet mode, themes and encoding.
- Non-ASCII glyphs come from `scode/ui/glyphs.py::g()`, which falls back to ASCII on
  consoles that cannot encode them (Windows cp1252).

## Things that are easy to get wrong

- **Compaction must not orphan tool results.** A `tool` message whose preceding assistant
  `tool_calls` message was dropped is rejected by the API. `_last_safe_split()` only ever
  cuts at a user message.
- **Read-before-edit is enforced** in `_require_prior_read()`, including an mtime check.
  Tests depend on this; do not relax it.
- **File writes use `newline=""`** so CRLF files stay CRLF.
- **`-p` mode has no asker**, so `PermissionEngine.authorize()` denies mutating tools by
  default. That is intentional.
- **Streaming**: `StreamWriter._start()` must not print the first chunk — the caller does.
  Printing in both places duplicated it once already.
- **Interactive mode must not read stdin as a prompt.** Only `-p` consumes piped stdin;
  in the REPL, stdin is the command stream.
- **The model sometimes returns a 200 with an empty body** (no content, no tool calls).
  That is not an answer: `_loop` retries up to `MAX_EMPTY_TURNS` and never appends the
  empty message. Treating it as "done" made scode exit 0 having silently done nothing.
- **The NVIDIA catalog is not an availability list.** `/v1/models` advertises models the
  account cannot call (404), that are retired (410), or whose gateway hangs (504 after
  ~5 minutes). `constants.MODEL_CATALOG` holds only verified ones and `KNOWN_UNAVAILABLE`
  explains the common failures; `check_model()` is the authoritative probe.
- **502/503/504 are retried only once.** A wedged gateway answers only after the full read
  timeout, so the normal retry budget would hang for many minutes.

## Testing

`tests/conftest.py` provides `FakeProvider`, which replays scripted `AssistantMessage`
turns, plus `tool_turn()` / `text_turn()` helpers and a `make_agent` factory. Use these to
test the loop rather than mocking HTTP. `isolated_home` redirects `~/.scode` to a tmpdir
and clears every `SCODE_*` env var, so tests never touch real user state.

For provider-level tests, `tests/test_provider.py` fakes `requests.Session.post` with an
SSE line generator.
