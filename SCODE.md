# scode

A Claude Code-style agentic coding CLI that runs on any model provider (NVIDIA NIM,
OpenRouter, OmniRoute, Anthropic, OpenAI and more). Pure Python, no agent framework.

## Commands

```bash
pytest                          # full suite; no network or keys needed
pytest tests/test_agent.py -q   # one file
ruff check scode tests          # lint
ruff check scode tests --fix    # autofix
pip install -e ".[dev]"         # dev install (includes the anthropic SDK)
python -m scode --doctor        # check the environment and provider
```

There is no build step. `scode` is installed as a console script via
`[project.scripts]` in `pyproject.toml`. CI runs ruff and pytest on Linux and Windows,
Python 3.10 to 3.13, so avoid syntax and stdlib APIs newer than 3.10.

## Layout

| Path | Holds |
| --- | --- |
| `scode/cli.py` | argparse entry point, `--print` mode (text/json/stream-json), `--doctor` and the listings |
| `scode/runtime.py` | `Runtime`: owns provider, agent, permissions, session, hooks, MCP; model/provider switching; subagents |
| `scode/repl.py` | `Repl(Runtime)`: the interactive loop, permission prompts, Shift+Tab mode cycling |
| `scode/config.py` | `Settings` and the layered loader (user → project → local → env → flags) |
| `scode/providers/profiles.py` | `ProviderProfile` and `BUILTIN_PROFILES`: one data record per provider |
| `scode/providers/catalog.py` | models.dev catalog: limits, prices, unknown providers by name |
| `scode/providers/registry.py` | `build_provider()`: picks the wire for a profile |
| `scode/providers/openai_compatible.py` | OpenAI-style wire: SSE streaming, tool-call assembly, retries, 400 adaptation |
| `scode/providers/anthropic_native.py` | native Anthropic wire over the official SDK: caching, adaptive thinking, effort |
| `scode/agent/loop.py` | the agent loop: request → tool calls → repeat; compaction, hooks, reminders |
| `scode/agent/prompts.py` | system prompts, subagent prompts, the no-function-calling text protocol |
| `scode/agent/context.py` | environment block, `SCODE.md` loading, `@file` expansion |
| `scode/agent/compact.py` | summarising the conversation |
| `scode/tools/` | one module per tool family; `build_registry()` assembles them |
| `scode/mcp/` | MCP client (stdio, Streamable HTTP) and the manager that exposes server tools |
| `scode/hooks.py` | Claude Code-compatible hooks |
| `scode/extensions.py` | custom slash commands and subagents from `.scode/` / `.claude/` markdown |
| `scode/permissions.py` | rule parsing (`Bash(git status:*)`), the four modes, the approval flow |
| `scode/session/store.py` | JSONL transcripts under `~/.scode/projects/<slug>/` |
| `scode/commands/builtin.py` | slash commands, registered with the `@command` decorator |
| `scode/usage.py` | token and cost accounting |
| `scode/ui/` | rich console, streaming renderer, diffs, prompt_toolkit input, glyphs |

## Adding a provider

Prefer data over code. Add a `ProviderProfile` to `BUILTIN_PROFILES` in
`scode/providers/profiles.py`, with its endpoint, key variables, default and small model,
and the quirks the wire needs (`max_tokens_param`, `effort_style`,
`supports_temperature`, `headers`). Only a provider with a non-OpenAI wire needs a new
module in `scode/providers/`, and it must return the canonical `AssistantMessage`.
Users can do the same without touching code, under `providers` in settings.json.

## Conventions

- `from __future__ import annotations` at the top of every module.
- Messages are canonical OpenAI-style dicts. Each wire converts at its edge. Opaque data
  a provider needs replayed (Anthropic thinking signatures, OpenRouter
  `reasoning_details`) lives in `provider_state` on the assistant message.
- Errors that users should see subclass `ScodeError` (`scode/errors.py`). A `ToolError`
  raised inside a tool is caught by the agent loop and fed back to the model as text, so
  it never crashes the turn. Anything else is a bug.
- Tools subclass `Tool` and declare `name`, `description`, `parameters` (JSON Schema) and
  `mutating`. Register them in `scode/tools/__init__.py`. Read-only tools that are safe to
  run concurrently go in `PARALLEL_SAFE`.
- `ToolResult.output` goes to the model; `.display` is the one-line summary for the user;
  `.detail` is optional rich content such as a diff.
- All terminal output goes through `scode/ui/console.py::UI`. Never `print()` directly,
  except in `cli.py` for machine-readable output. `UI` handles quiet mode, themes and
  encoding.
- Non-ASCII glyphs come from `scode/ui/glyphs.py::g()`, which falls back to ASCII on
  consoles that can't encode them (Windows cp1252).

## Things that are easy to get wrong

- **The system prompt is frozen for the session, and history is append-only.** Mode
  changes, `#` notes and similar go in as `<system-reminder>` user messages
  (`Agent.remind()`), never as edits to the system prompt or to earlier messages.
  Editing either breaks prompt caching on every provider.
- **Every tool call gets a result**, even when the turn is interrupted or a hook blocks it.
  An orphaned `tool_calls` entry makes the next request fail.
- **Compaction replaces the history** with one carrier message: the summary plus the
  latest user request. Old turns are never replayed after a compaction.
- **Read-before-edit is enforced** in `_require_prior_read()`, including an mtime check.
  Tests depend on this; don't relax it.
- **File writes use `newline=""`** so CRLF files stay CRLF.
- **`-p` mode has no asker**, so `PermissionEngine.authorize()` denies mutating tools by
  default. That's intentional.
- **Interactive mode must not read stdin as a prompt.** Only `-p` consumes piped stdin;
  in the REPL, stdin is the command stream.
- **A 200 with an empty body isn't an answer.** `_loop` retries and never appends the
  empty message.
- **Provider parameters are adapted, not hard-coded.** A 400 naming `max_tokens`,
  `temperature`, the effort field or `stream_options` is fixed in
  `OpenAICompatibleProvider._adapt()` and the request is retried. Add new quirks there
  or as a profile field.
- **A catalog isn't an availability list.** NVIDIA's `/v1/models` advertises models a key
  can't call (404), retired ones (410), and ones whose gateway hangs. `check_model()` is
  the authoritative probe.
- **502/503/504 are retried only once.** A wedged gateway answers only after the full
  read timeout, so a normal retry budget would hang for many minutes.

## Testing

`tests/conftest.py` provides `FakeProvider`, which replays scripted `AssistantMessage`
turns (or raises scripted exceptions), plus `tool_turn()` / `text_turn()` helpers and a
`make_agent` factory. Use these to test the loop rather than mocking HTTP.

`isolated_home` redirects `~/.scode` to a tmpdir, clears every `SCODE_*` and provider key
variable, and turns off the models.dev fetch. The autouse `no_network` guard fails any
test that tries to reach a non-local host.

Provider tests use real local servers: `tests/test_provider.py` fakes SSE for the OpenAI
wire, `tests/anthropic_mock.py` serves the Anthropic Messages API to the real SDK, and
`tests/mcp_server.py` is a stdio MCP server.
