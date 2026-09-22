# scode

An agentic coding CLI for your terminal, powered by **NVIDIA NIM** models. It reads your
code, edits files, runs commands, and asks before it changes anything — the Claude Code
workflow, running on NVIDIA's free model catalog.

```
⏺ Read(src/client.py)
  ⎿  Read src/client.py (128 lines)

⏺ Edit(src/client.py)
  ⎿  Edited src/client.py
┌──────────────────────────────────────────────────────┐
│ @@ -42,6 +42,9 @@                                    │
│      def fetch(self, url):                           │
│ -        return requests.get(url)                    │
│ +        for attempt in range(3):                    │
│ +            try:                                    │
│ +                return requests.get(url, timeout=10)│
└──────────────────────────────────────────────────────┘

⏺ Bash(pytest -q)
  ⎿  12 passed in 1.4s

Added a retry loop with a timeout to `src/client.py:42`. Tests pass.
```

## Install

```bash
git clone https://github.com/MR-SAIRAM-7/scode.git
cd scode
pip install -e .
```

Requires Python 3.10+. Installs `requests`, `rich` and `prompt_toolkit`.

## Set up a key

NVIDIA gives free credits at [build.nvidia.com](https://build.nvidia.com). Your key looks
like `nvapi-...`.

```bash
export NVIDIA_API_KEY=nvapi-your-key-here
```

On Windows PowerShell:

```powershell
$env:NVIDIA_API_KEY = "nvapi-your-key-here"
```

Or start `scode` and run `/login`, which stores the key in `~/.scode/settings.json`.

Check everything is wired up:

```bash
scode --doctor
```

## Use it

```bash
scode                                  # interactive session in the current directory
scode "add retry logic to the client"  # start with a first prompt
scode -p "what does src/app.py do?"    # one-shot, prints and exits
scode -c                               # continue the last session here
scode --permission-mode plan           # research and propose, change nothing
```

Inside a session:

| Input | What it does |
| --- | --- |
| `explain @src/app.py` | attaches the file's contents to your message |
| `!pytest -q` | runs a shell command yourself; scode sees the output |
| `#always run make lint` | saves a note to `SCODE.md` for future sessions |
| `/help` | lists every command |

Keys: `Esc`+`Enter` or `Ctrl+J` for a newline, `Ctrl+C` to interrupt a turn, `Ctrl+D` to
exit, `Up`/`Down` for history, `Tab` to complete commands and file paths.

## Commands

| Command | What it does |
| --- | --- |
| `/help` | list commands |
| `/model [name]` | show or change the model; `/model --all` lists the live catalog |
| `/mode [name]` | change how tool calls are approved |
| `/status` | session, model, context and token status |
| `/cost` | token usage for the session |
| `/compact [focus]` | summarise the conversation to free up context |
| `/clear` | start a fresh conversation |
| `/init` | generate a `SCODE.md` describing the project |
| `/memory` | show or edit the instruction files |
| `/diff` | show the working tree diff |
| `/review` | have the model review the current diff |
| `/commit` | stage and commit with a written message |
| `/sessions`, `/resume <id>` | list and resume earlier sessions |
| `/export [path]` | write the conversation to markdown |
| `/tools`, `/agents` | list tools and subagent types |
| `/login`, `/logout` | store or forget the API key |
| `/config`, `/theme`, `/doctor` | configuration and diagnostics |
| `/exit` | quit |

## Permissions

scode asks before it changes anything. Four modes:

| Mode | Behaviour |
| --- | --- |
| `default` | asks before file edits and shell commands |
| `acceptEdits` | file edits run automatically; shell commands still ask |
| `plan` | read-only — research and propose, change nothing |
| `bypassPermissions` | runs everything without asking (sandboxes only) |

When it asks, you can allow the call once, or allow it permanently. Permanent choices are
saved as rules in `.scode/settings.local.json`:

```json
{ "permissions": { "allow": ["Bash(npm test:*)", "Edit(src/**)"], "deny": ["Bash(rm:*)"] } }
```

Rules are `Tool` (every use), `Tool(prefix:*)` (commands starting with a prefix), or
`Tool(glob)` (path or argument match). You can also pass them per-run:

```bash
scode --allowed-tools 'Bash(npm test:*)' -p "run the tests and fix what fails"
```

Commands that would destroy a filesystem (`rm -rf /`, `mkfs`, fork bombs) are refused in
every mode, including `bypassPermissions`.

## Tools

| Tool | What it does |
| --- | --- |
| `Read` | read a file with line numbers, with offset/limit for large ones |
| `Write` | create or overwrite a file |
| `Edit` | replace an exact, unique string |
| `MultiEdit` | several edits to one file, applied atomically |
| `Glob` | find files by pattern, newest first |
| `Grep` | regex search; uses ripgrep when installed |
| `LS` | list a directory |
| `Bash` | run a shell command in the workspace |
| `TodoWrite` | track multi-step work |
| `Task` | delegate to a subagent with its own context |
| `WebFetch` | fetch a URL as readable text |
| `ExitPlanMode` | present a plan and ask to start work |

`Edit`, `MultiEdit` and `Write` refuse to touch a file the model has not read this session,
and refuse again if it changed on disk since — so an edit can never silently clobber work.

## Project instructions

Put standing instructions for a repo in `SCODE.md` at its root. scode loads it into every
session. `CLAUDE.md` and `AGENTS.md` are read too, so existing repos work unchanged.
`~/.scode/SCODE.md` applies to every project.

Run `/init` to have scode read the codebase and write one for you.

## Models

The default is `moonshotai/kimi-k3`. `/model` shows a shortlist; `/model --all` queries
NVIDIA's live catalog. Any model that supports function calling works well; for one that
does not, scode detects the rejection and falls back to a text-based tool protocol
automatically.

```bash
scode --model nvidia/nemotron-3-super-120b-a12b
scode --list-models
```

## Configuration

Settings merge in this order, later winning: user file → project file → project-local file
→ environment → command-line flags.

- `~/.scode/settings.json` — your defaults
- `.scode/settings.json` — shared project settings, commit this
- `.scode/settings.local.json` — your machine only, git-ignore this

```json
{
  "model": "moonshotai/kimi-k3",
  "temperature": 0.6,
  "max_output_tokens": 32000,
  "auto_compact": true,
  "permissions": { "defaultMode": "acceptEdits", "allow": ["Bash(git status:*)"] }
}
```

Environment variables: `NVIDIA_API_KEY`, `SCODE_MODEL`, `SCODE_BASE_URL`,
`SCODE_TEMPERATURE`, `SCODE_MAX_OUTPUT_TOKENS`, `SCODE_MAX_STEPS`, `SCODE_PERMISSION_MODE`,
`SCODE_THEME`, `SCODE_AUTO_COMPACT`, `SCODE_SHELL`, `SCODE_HOME`, `SCODE_ASCII`.

## Sessions and context

Every session is written to `~/.scode/projects/<project>/<id>.jsonl`. Resume with `scode -c`
(the last one) or `scode --resume <id>`. When the context window fills past ~82%, scode
summarises the older turns automatically and keeps going; `/compact` does it on demand.

## Scripting

`-p` runs one turn and exits, so scode composes with other tools:

```bash
scode -p "summarise the architecture" > docs/architecture.md
echo "what changed in the last commit?" | scode -p
scode -p --output-format json "list the TODOs" | jq -r .result
```

In `-p` mode nothing can prompt you, so mutating tools are refused unless you allow them
explicitly with `--allowed-tools` or `--permission-mode`. Exit codes: `0` success,
`1` the turn failed, `2` bad configuration, `130` interrupted.

## Other endpoints

Anything that speaks the OpenAI chat API works:

```bash
scode --base-url http://localhost:11434/v1 --model qwen2.5-coder --api-key ollama
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check scode tests
```

## Licence

MIT
