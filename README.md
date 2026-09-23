# scode

An agentic coding CLI for your terminal, in the style of Claude Code, that runs on **any
model provider**: NVIDIA's free NIM models, OpenRouter, a local OmniRoute gateway,
Anthropic, OpenAI, Gemini, DeepSeek, Groq and more, or a model on your own machine. It
reads your code, edits files and runs commands. It asks before it changes anything.

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

You need Python 3.10 or newer.

```bash
git clone https://github.com/MR-SAIRAM-7/scode.git
cd scode
pip install -e .
```

To talk to Claude through Anthropic's native API (prompt caching, adaptive thinking),
also install the Anthropic SDK:

```bash
pip install -e ".[anthropic]"
```

## Pick a provider

Every provider works the same way: set its key, then pick it. NVIDIA is the default
because it's free.

| Provider | `--provider` | Key variable | Default model | Notes |
| --- | --- | --- | --- | --- |
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY` | `nvidia/nemotron-3-super-120b-a12b` | Free credits at [build.nvidia.com](https://build.nvidia.com) |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `anthropic/claude-opus-5` | One key reaches hundreds of models |
| OmniRoute | `omniroute` | none (local) | `auto/coding` | Self-hosted router on `localhost:20128` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-5` | Native API; needs the `anthropic` extra |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-6-sol` | |
| Google Gemini | `gemini` | `GEMINI_API_KEY` | `gemini-3.1-pro-preview` | |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` | |
| Groq | `groq` | `GROQ_API_KEY` | `qwen/qwen3.8-27b` | |
| Together AI | `together` | `TOGETHER_API_KEY` | `zai-org/GLM-5.3` | |
| Mistral | `mistral` | `MISTRAL_API_KEY` | `mistral-medium-latest` | |
| xAI | `xai` | `XAI_API_KEY` | `grok-4.7` | |
| Fireworks | `fireworks` | `FIREWORKS_API_KEY` | `glm-5p3` | |
| Cerebras | `cerebras` | `CEREBRAS_API_KEY` | `qwen-3.8-27b` | |
| Moonshot (Kimi) | `moonshot` | `MOONSHOT_API_KEY` | `kimi-k3` | |
| Z.ai (GLM) | `zai` | `ZAI_API_KEY` | `glm-5.3` | |
| Ollama | `ollama` | none (local) | whatever is loaded | `localhost:11434` |
| LM Studio | `lmstudio` | none (local) | whatever is loaded | `127.0.0.1:1234` |
| Anything else | `custom` | `SCODE_API_KEY` | you choose | Any OpenAI-compatible endpoint |

`scode --list-providers` prints this table from your install. It includes any providers
you have added yourself.

### Set the key

Keep keys out of your shell history and out of any file you commit.

**Windows (PowerShell).** This sets the key for the current terminal only:

```powershell
$env:NVIDIA_API_KEY = "nvapi-your-key-here"
```

To set it permanently, run the command below, then **open a new terminal**. `setx`
doesn't affect the terminal you run it in.

```powershell
setx NVIDIA_API_KEY "nvapi-your-key-here"
```

**macOS / Linux.** Add the line to `~/.bashrc` or `~/.zshrc` to make it permanent:

```bash
export NVIDIA_API_KEY=nvapi-your-key-here
```

Or skip all of that: start scode and run `/login`. It asks for the key without echoing it
and stores it in `~/.scode/settings.json`, separately for each provider. `/login
sk-or-v1-...` recognises an OpenRouter key by its prefix, and the same goes for
Anthropic, NVIDIA, Groq, xAI, Fireworks and Cerebras keys.

## Run it

**scode works on whatever directory you start it in**, so `cd` into the project you want
to work on first. Starting it in your home folder points it at your entire user profile.

```bash
cd path/to/your/project
scode
```

If `scode` isn't found, use the module form, which always works:

```bash
python -m scode
```

<details>
<summary>Windows: getting the short <code>scode</code> command</summary>

`pip install` puts `scode.exe` in your Python scripts folder, which is often not on
`PATH`. pip prints a warning about this during install. Either use `python -m scode`, or
add the folder to `PATH` and open a new terminal. Change the Python version in the path
to match yours:

```powershell
setx PATH "$env:PATH;$env:APPDATA\Python\Python314\Scripts"
```

</details>

Check the install, and find out which models your key can actually reach:

```bash
scode --doctor
```

```bash
scode --check-models
```

### Ways to start it

```bash
scode                                        # interactive session in the current directory
scode "add retry logic to the client"        # start with a first prompt
scode -p "what does src/app.py do?"          # one-shot: print the answer and exit
scode -c                                     # continue the last session here
scode --permission-mode plan                 # research and propose, change nothing
scode --provider openrouter --model sonnet   # choose provider and model
scode --model anthropic:opus                 # the same, in one flag
```

If you haven't used a tool like this before, start in plan mode. It reads your code and
proposes changes but can't touch a single file. That way you can see how it behaves on a
repo you care about before you let it edit.

### What to expect

- It **asks before every edit and shell command**. Press `1` to allow the action once,
  `2` to allow that kind of action from now on, or `3` to refuse and tell it what to do
  instead.
- **Shift+Tab** cycles the permission mode: `default` → `acceptEdits` → `plan`. The
  current mode is shown in the bottom toolbar.
- If a provider's gateway returns an empty reply, you'll see `Empty response from the
  model; retrying`. scode recovers on its own; it isn't an error.

Inside a session:

| Input | What it does |
| --- | --- |
| `explain @src/app.py` | attaches the file to your message (images too, for vision models) |
| `!pytest -q` | runs a shell command yourself; scode sees the output |
| `#always run make lint` | saves a standing instruction to `SCODE.md` |
| `/help` | lists every command, including your custom ones |

Keys: `Esc`+`Enter` or `Ctrl+J` for a newline, `Ctrl+C` to interrupt a turn, `Ctrl+D` to
exit, `Up`/`Down` for history, `Tab` to complete commands and file paths, `Shift+Tab`
to change the permission mode.

## Switching providers and models

All of these work, from quickest to most permanent:

```bash
scode --provider openrouter                    # the provider's default model
scode --provider openrouter --model sonnet     # aliases: opus, sonnet, haiku, fable
scode --model anthropic:claude-sonnet-5        # provider:model in one flag
SCODE_PROVIDER=groq scode                      # environment
```

Inside a session:

```
/provider                     list providers, which have keys, and their default models
/provider openrouter          switch; the provider's default model is used
/provider anthropic sonnet    switch provider and model
/model                        list this provider's models with context size and price
/model gpt-6-luna             change the model
/model openrouter:opus        change provider and model together
/model --check                probe which models your key can actually reach
/effort high                  reasoning effort: low, medium, high, xhigh or max
```

`/provider` and `/model` also save your choice to `~/.scode/settings.json`, so the next
session starts with it. This works the way `/model` does in Claude Code.

When you switch provider, the old provider's endpoint and small model are dropped. You
can't end up sending an OpenRouter model name to NVIDIA by accident.

**A model catalog isn't the same as what your key can use.** NVIDIA in particular lists
far more models than a free key can call. `/model --check` asks each one:

```
model                                          status       detail
nvidia/nemotron-3-super-120b-a12b (current)    works        0.5s
nvidia/nemotron-3-ultra-550b-a55b              works        1.7s
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning  works        0.3s
openai/gpt-oss-20b                             works        5.2s
```

Model limits and prices come from [models.dev](https://models.dev), cached for a day in
`~/.scode/cache/`. scode uses them for context-window tracking, the output cap, and cost.
Set `SCODE_OFFLINE=1` to stop scode fetching the catalog.

## Adding a provider

There are three ways, and none of them needs a code change.

**1. It's already on models.dev.** Any OpenAI-compatible provider listed on
[models.dev](https://models.dev) resolves by name. scode reads its endpoint, key
variable and newest models from the catalog:

```bash
scode --provider <models.dev id>
```

**2. Add it to your settings.** Any OpenAI-compatible endpoint (a company gateway, a
proxy, a new provider) goes under `providers` in `~/.scode/settings.json` or a project's
`.scode/settings.json`:

```json
{
  "provider": "acme",
  "providers": {
    "acme": {
      "label": "Acme AI gateway",
      "base_url": "https://llm.acme.internal/v1",
      "api_key_env": ["ACME_API_KEY"],
      "default_model": "acme-coder-large",
      "small_model": "acme-coder-small",
      "models": ["acme-coder-large", "acme-coder-small"],
      "model_aliases": { "big": "acme-coder-large" },
      "headers": { "X-Team": "platform" }
    }
  }
}
```

The same block can also override fields of a built-in provider. For example, this routes
OpenRouter through your own proxy:

```json
{ "providers": { "openrouter": { "base_url": "https://proxy.example.com/openrouter/v1" } } }
```

**3. One-off.** Pass the endpoint and model on the command line:

```bash
scode --provider custom --base-url http://localhost:8000/v1 --model my-model
```

| Field | Meaning |
| --- | --- |
| `base_url` | the `/v1` endpoint |
| `wire` | `openai` (default) or `anthropic` for an Anthropic-compatible endpoint |
| `api_key_env` | environment variables to read the key from, first match wins |
| `key_required` | `false` for local servers |
| `default_model`, `small_model` | the working model, and a cheaper one used to summarise the conversation |
| `models`, `model_aliases` | what `/model` lists, and short names for them |
| `headers` | extra HTTP headers on every request |
| `max_tokens_param` | `max_tokens` (default) or `max_completion_tokens` |
| `default_max_tokens` | output cap when you haven't set one |
| `effort_style` | how reasoning effort is sent: `none`, `reasoning_effort`, `openrouter` or `anthropic` |
| `supports_temperature` | `false` for models that reject a temperature |
| `auto_model` | `true` to use whichever model a local server has loaded |

You don't have to get the parameters right up front. If a provider rejects
`max_tokens`, `temperature`, the effort parameter or usage streaming, scode adapts the
request and retries on its own.

To make a provider **built in**, add a `ProviderProfile` to `BUILTIN_PROFILES` in
[scode/providers/profiles.py](scode/providers/profiles.py). It's data, not code.

## Commands

| Command | What it does |
| --- | --- |
| `/help` | list commands |
| `/provider [name [model]]` | list providers, or switch |
| `/model [name \| provider:name]` | list models, or switch |
| `/model --check`, `--all`, `--search q` | probe availability, or list the full catalog |
| `/effort [level]` | reasoning effort, where the model supports it |
| `/mode [name]` | change how tool calls are approved |
| `/permissions [allow\|deny\|remove RULE]` | show or edit permission rules |
| `/status` | session, provider, context and cost |
| `/cost` | token usage, cache hits and spend for the session |
| `/compact [focus]` | summarise the conversation to free up context |
| `/clear` | start a fresh conversation |
| `/rewind [turns]` | undo the agent's file changes from recent turns |
| `/init` | generate a `SCODE.md` describing the project |
| `/memory [edit]` | show or edit the instruction files |
| `/diff`, `/review`, `/commit` | working tree diff, a review of it, or a commit |
| `/sessions`, `/resume <id>` | list and resume earlier sessions |
| `/export [path]` | write the conversation to markdown |
| `/tools`, `/agents` | tools and subagent types |
| `/mcp`, `/hooks`, `/bashes` | MCP servers, configured hooks, background shells |
| `/login`, `/logout` | store or forget a provider's API key |
| `/config`, `/theme`, `/doctor` | configuration and diagnostics |
| `/exit` | quit |

## Permissions

scode asks before it changes anything. Four modes:

| Mode | Behaviour |
| --- | --- |
| `default` | asks before file edits and shell commands |
| `acceptEdits` | file edits run automatically; shell commands still ask |
| `plan` | read-only: it researches and proposes, then asks to start |
| `bypassPermissions` | runs everything without asking (for sandboxes only) |

When it asks, you can allow the call once or allow it permanently. Permanent choices are
saved as rules in `.scode/settings.local.json`:

```json
{ "permissions": { "allow": ["Bash(npm test:*)", "Edit(src/**)"], "deny": ["Bash(rm:*)"] } }
```

A rule takes one of these forms:

- `Tool` matches every use of the tool.
- `Tool(prefix:*)` matches commands that start with a prefix.
- `Tool(glob)` matches a path or argument.
- `mcp__server` matches every tool of one MCP server; `mcp__server__tool` matches a single tool.

You can also pass rules for a single run:

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
| `MultiEdit` | apply several edits to one file atomically |
| `Glob` | find files by pattern, newest first |
| `Grep` | regex search; uses ripgrep when it's installed |
| `LS` | list a directory |
| `Bash` | run a shell command; `run_in_background` for servers and watchers |
| `BashOutput`, `KillShell` | read from or stop a background shell |
| `TodoWrite` | track multi-step work |
| `Task` | delegate to a subagent with its own context |
| `AskUserQuestion` | ask you a multiple-choice question when a decision is yours |
| `WebFetch` | fetch a URL as readable text |
| `ExitPlanMode` | present a plan and ask to start work |

Read-only tools called together (`Read`, `Glob`, `Grep`, `LS`, `BashOutput`) run in
parallel.

`Edit`, `MultiEdit` and `Write` refuse to touch a file the model hasn't read in this
session. They refuse again if the file changed on disk since then, so an edit can never
silently overwrite your work. Every `Write`, `Edit` and `MultiEdit` is checkpointed
first, so `/rewind` can undo it. Changes made through `Bash` aren't tracked.

## Project instructions, custom commands and agents

Put standing instructions for a repo in `SCODE.md` at its root, and scode loads it into
every session. `CLAUDE.md` and `AGENTS.md` are read too, so existing repos work
unchanged. Instructions in `~/.scode/SCODE.md` apply to every project. Run `/init` to
have scode read the codebase and write one for you.

**Custom slash commands** are markdown files in `.scode/commands/` (or `.claude/commands/`,
or the same folders under your home directory). `.scode/commands/fix-issue.md` becomes
`/fix-issue`, and a file in a subfolder like `frontend/lint.md` becomes `/frontend:lint`.

```markdown
---
description: Fix a GitHub issue
argument-hint: [issue number]
---
Read issue #$ARGUMENTS with `gh issue view $1`, find the cause, fix it and add a test.
```

**Custom subagents** are markdown files in `.scode/agents/` (or `.claude/agents/`). The
`Task` tool can launch them by name, with their own prompt, tools and model:

```markdown
---
name: reviewer
description: Reviews a diff for bugs and missing tests
tools: Read, Grep, Glob, Bash
model: small
---
You are a meticulous code reviewer. Report problems with file:line references.
```

## MCP servers

scode is an MCP client. It supports stdio servers and Streamable HTTP servers, which are
the same formats Claude Code uses. Put servers in `.mcp.json` at the project root (commit
this one to share it), under `mcpServers` in a settings file, or pass `--mcp-config
file.json`:

```json
{
  "mcpServers": {
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] },
    "tracker": { "type": "http", "url": "https://mcp.example.com/mcp",
                 "headers": { "Authorization": "Bearer ${TRACKER_TOKEN}" } }
  }
}
```

`${VAR}` is replaced from the environment. A server's tools appear to the model as
`mcp__<server>__<tool>`. Tools a server marks read-only run without a prompt; the rest go
through the normal permission check. `/mcp` shows each server's status and tools.

## Hooks

Hooks run your own commands at points in the agent's work. They use Claude Code's
format, so hook scripts written for Claude Code work unchanged. Configure them under
`hooks` in a settings file:

```json
{
  "hooks": {
    "PostToolUse": [
      { "matcher": "Edit|MultiEdit|Write",
        "hooks": [{ "type": "command", "command": "ruff format --quiet ." }] }
    ],
    "PreToolUse": [
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": "python .scode/check_bash.py" }] }
    ]
  }
}
```

Events: `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`, `SubagentStop`,
`SessionStart`, `SessionEnd`, `PreCompact`.

Each hook gets the event as JSON on stdin, with `SCODE_PROJECT_DIR` and
`CLAUDE_PROJECT_DIR` set in its environment. What the hook returns decides what happens
next:

- Exit code `0`: continue.
- Exit code `2`: block. For a tool event or `Stop`, the hook's stderr goes to the model;
  for `UserPromptSubmit`, it's shown to you and the prompt isn't sent.
- JSON on stdout: `{"decision": "block", "reason": "..."}` blocks with a reason;
  `{"hookSpecificOutput": {"permissionDecision": "allow" | "deny" | "ask"}}` settles the
  permission question.

## Sessions, context and cost

Every session is saved to `~/.scode/projects/<project>/<id>.jsonl`. Resume the last one
with `scode -c`, or a specific one with `scode --resume <id>`.

When the context window is about 82% full, scode summarises the conversation and keeps
going. `/compact` does the same on demand. If a provider rejects a request for being too
long, scode compacts the conversation and retries once.

The system prompt stays the same for the whole session and history is only ever appended
to, so providers with prompt caching (Anthropic, OpenRouter, DeepSeek, OpenAI and others)
can serve most of each request from cache. `/cost` shows cache hits and the spend worked
out from models.dev prices. A provider's own reported cost (OpenRouter's, for example) is
used when it gives one.

## Scripting

`-p` runs one turn and exits, so scode works with other tools:

```bash
scode -p "summarise the architecture" > docs/architecture.md
echo "what changed in the last commit?" | scode -p
scode -p --output-format json "list the TODOs" | jq -r .result
scode -p --output-format stream-json "fix the failing test"    # one JSON event per line
```

In `-p` mode nothing can prompt you, so mutating tools are refused unless you allow them
explicitly with `--allowed-tools` or `--permission-mode`. Exit codes: `0` success, `1`
the turn failed, `2` bad configuration, `130` interrupted.

## Configuration

Settings merge in this order, later winning: user file → project file → project-local
file → environment → command-line flags.

- `~/.scode/settings.json`: your defaults
- `.scode/settings.json`: shared project settings; commit this one
- `.scode/settings.local.json`: settings for your machine only; git-ignore this one

```json
{
  "provider": "openrouter",
  "model": "anthropic/claude-sonnet-5",
  "effort": "high",
  "max_output_tokens": 32000,
  "auto_compact": true,
  "append_system_prompt": "Prefer small, reviewable diffs.",
  "permissions": { "defaultMode": "acceptEdits", "allow": ["Bash(git status:*)"] }
}
```

A project file that sets `provider` also resets `model`, `small_model` and `base_url`
inherited from your user file. That way a project pinned to one provider never picks up
another provider's model.

Environment variables:

| Variable | Sets |
| --- | --- |
| `SCODE_PROVIDER`, `SCODE_MODEL`, `SCODE_SMALL_MODEL` | provider and models |
| `SCODE_BASE_URL`, `SCODE_API_KEY` | endpoint and key for the current provider |
| `SCODE_EFFORT`, `SCODE_TEMPERATURE`, `SCODE_MAX_OUTPUT_TOKENS`, `SCODE_MAX_STEPS` | model behaviour |
| `SCODE_PERMISSION_MODE`, `SCODE_AUTO_COMPACT`, `SCODE_NATIVE_TOOLS` | agent behaviour |
| `SCODE_THEME`, `SCODE_ASCII`, `SCODE_SHELL` | display and shell |
| `SCODE_HOME` | where settings, sessions and logs live (default `~/.scode`) |
| `SCODE_OFFLINE` | `1` to skip fetching the models.dev catalog |
| `SCODE_DEBUG` | `1` to write a debug log, like `--debug` |

## Troubleshooting

- `scode --doctor` checks Python, the key, the endpoint, and whether the model answers.
- `scode --debug` writes every request, retry and tool call to `~/.scode/logs/`.
- **"Model not found"**: run `/model --check` to see which models your key can reach.
- **A local provider "could not be reached"**: start the server first (`ollama serve`,
  `npx omniroute`, or LM Studio's server tab).
- **A model can't call tools**: scode notices when a model rejects function calling and
  switches to a text-based tool protocol. `--no-native-tools` forces this protocol from
  the start.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check scode tests
```

The test suite runs without network access or API keys. Provider tests use local mock
servers for both the OpenAI-compatible and the Anthropic wire.

## Licence

MIT
