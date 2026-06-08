# Baseline ISD Agent

Single prompt ADDIE instructional design generator with multi-provider support.

## Overview

Baseline is a reference agent that generates complete ADDIE outputs with a single prompt.
Used as a comparison baseline for evaluating other agents.

Uses the shared provider-neutral LLM configuration for hosted and local models.

## Features

- Single API call generates complete ADDIE output
- Provider-neutral LLM support (hosted and local backends)
- Bloom's Taxonomy based learning objective design
- Gagné's 9 Events of Instruction application
- Structured JSON output

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Basic execution
baseline run --input scenario.json --output result.json

# Show info
baseline info
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--input, -i` | Input scenario JSON file |
| `--output, -o` | Output ADDIE result file |
| `--trajectory, -t` | Trajectory save file (optional) |
| `--model` | LLM model (default: varies by provider) |
| `--verbose, -v` | Verbose output |

## LLM Configuration

Baseline uses the shared provider-neutral LLM layer from `shared/llm`.
Pass `--agent-model-*` flags to `run_benchmark.py`, or configure optional `.env` defaults/secrets at the repository root. Judge models are configured separately with `--judge-model-*`.

```bash
# OpenRouter example
AGENT_MODEL_PROVIDER=openrouter
AGENT_MODEL_API_SPEC=openai_compatible
AGENT_MODEL_NAME=openai/gpt-4o-mini
AGENT_MODEL_BASE_URL=https://openrouter.ai/api/v1
AGENT_MODEL_API_KEY_ENV=OPENROUTER_API_KEY
```

For local backends, use an OpenAI-compatible endpoint such as Ollama,
LM Studio, or vLLM:

```bash
AGENT_MODEL_PROVIDER=local-lmstudio
AGENT_MODEL_API_SPEC=openai_compatible
AGENT_MODEL_NAME=local-model
AGENT_MODEL_BASE_URL=http://localhost:1234/v1
AGENT_MODEL_API_KEY=not-needed
```

## Implementation Details

This implementation follows **Zero-shot Chain-of-Thought (CoT)** and **Single Prompting** approach as a baseline model.

### Key Characteristics
- **Single-Turn Generation**: Generates complete results with a single system prompt and user input, without complex agent interactions
- **Zero-shot CoT**: Induces step-by-step reasoning through prompt instructions only, without few-shot examples
- **Provider-neutral LLM Support**: Serves as a benchmark baseline across hosted and local OpenAI-compatible or native API specs
