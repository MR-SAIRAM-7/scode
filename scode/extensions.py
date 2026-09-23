"""Custom slash commands and custom subagents, defined as markdown files.

Commands:  .scode/commands/<name>.md       ->  /<name> [args]
Agents:    .scode/agents/<name>.md         ->  Task(subagent_type="<name>")

Both are looked up in the user's home (~/.scode/...) and the project, and in
the matching .claude/ directories so setups written for Claude Code work
unchanged. Later locations win: project over user, .scode over .claude.

A file may start with a frontmatter block:

    ---
    description: Review the staged diff
    argument-hint: [focus]
    model: sonnet
    tools: Read, Grep, Glob
    ---
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .constants import home_dir

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_POSITIONAL = re.compile(r"\$([1-9])")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, text[match.end():]


def _split_list(value: str) -> list[str]:
    value = value.strip().strip("[]")
    return [item.strip().strip("\"'") for item in value.split(",") if item.strip()]


def _search_dirs(workspace: Path, kind: str) -> list[Path]:
    home = home_dir()
    return [
        Path.home() / ".claude" / kind,
        home / kind,
        workspace / ".claude" / kind,
        workspace / ".scode" / kind,
    ]


@dataclass(frozen=True)
class CustomCommand:
    name: str
    body: str
    path: Path
    description: str = ""
    argument_hint: str = ""
    model: str = ""

    def render(self, arguments: str) -> str:
        words = arguments.split()
        text = self.body.replace("$ARGUMENTS", arguments)
        text = _POSITIONAL.sub(lambda m: words[int(m.group(1)) - 1] if int(m.group(1)) <= len(words) else "", text)
        if arguments and "$ARGUMENTS" not in self.body and not _POSITIONAL.search(self.body):
            text = f"{text.rstrip()}\n\n{arguments}"
        return text.strip()


def load_commands(workspace: Path) -> dict[str, CustomCommand]:
    found: dict[str, CustomCommand] = {}
    for directory in _search_dirs(workspace, "commands"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            meta, body = parse_frontmatter(text)
            relative = path.relative_to(directory).with_suffix("")
            name = ":".join(relative.parts).lower()
            description = meta.get("description") or _first_line(body)
            found[name] = CustomCommand(
                name=name,
                body=body.strip(),
                path=path,
                description=description,
                argument_hint=meta.get("argument-hint", ""),
                model=meta.get("model", ""),
            )
    return found


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    prompt: str
    path: Path
    tools: tuple[str, ...] = ()
    model: str = ""
    read_only: bool = False
    extra: dict[str, str] = field(default_factory=dict)


def load_agents(workspace: Path) -> dict[str, AgentDefinition]:
    found: dict[str, AgentDefinition] = {}
    for directory in _search_dirs(workspace, "agents"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            meta, body = parse_frontmatter(text)
            name = (meta.get("name") or path.stem).strip().lower()
            if not name or not body.strip():
                continue
            tools = tuple(_split_list(meta.get("tools", "")))
            model = meta.get("model", "")
            found[name] = AgentDefinition(
                name=name,
                description=meta.get("description") or _first_line(body),
                prompt=body.strip(),
                path=path,
                tools=tools,
                model="" if model.lower() == "inherit" else model,
            )
    return found


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:100]
    return ""
