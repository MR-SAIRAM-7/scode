# scode

A modular, agentic coding CLI scaffold with provider integrations for:

- NVIDIA hosted models (`--provider nvidia`)
- Kimi K3 (`--provider kimi-k3`)
- Parallel multi-agent task execution (`--task` + `--parallel-agents`)

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
- `SCODE_TEMPERATURE`
- `SCODE_SEED`
- `SCODE_REASONING_EFFORT`

## NVIDIA/Kimi direct mode (request-style)

Direct mode maps to chat completion requests like NVIDIA's example payloads, including streaming and image URL input:

```bash
python -m scode --provider nvidia --direct --stream --image-url "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1b.jpg" "What is in this image?"
```

For non-streaming direct mode:

```bash
python -m scode --provider nvidia --direct "Write a production-ready Python CLI skeleton"
```

Never hardcode API keys in source code; set `NVIDIA_API_KEY` / `KIMI_API_KEY` in your environment.

## Parallel multi-agent mode

Run multiple agent tasks concurrently in one CLI invocation:

```bash
python -m scode "Plan architecture" --task "Write tests strategy" --task "Suggest rollout checklist" --parallel-agents 3
```

Each task runs as its own agent workflow and results are printed per agent section.
