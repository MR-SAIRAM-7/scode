# scode

A modular, agentic coding CLI scaffold with provider integrations for:

- NVIDIA hosted models (`--provider nvidia`)
- Kimi K3 (`--provider kimi-k3`)

## Key behavior

- Default model: `moonshotai/kimi-k3`
- Default max output tokens: `60000`
- Agent loop supports tool execution for CLI-style autonomous steps.

## Quick start

```bash
python -m scode --help
```

Set credentials via environment variables:

- `NVIDIA_API_KEY` for `nvidia`
- `KIMI_API_KEY` for `kimi-k3`

Optional overrides:

- `SCODE_PROVIDER`
- `SCODE_MODEL`
- `SCODE_MAX_OUTPUT_TOKENS`
- `SCODE_WORKSPACE`
