# ISD-Agent-Bench

A comprehensive benchmark for evaluating LLM-based Instructional Systems Design (ISD) agents.

## Quick Start

### 1. Setup

```bash
# Run setup script
chmod +x scripts/*.sh
./scripts/1_setup.sh

# Optional: create .env only if you want persistent defaults/secrets
cp .env.example .env
```

### 2. Configure Models

ISD-Agent-Bench has two model roles:

- `AGENT_MODEL_*`: the model used by benchmark agents to generate outputs.
- `JUDGE_MODEL_*`: the model(s) used by the evaluator to score outputs.

Use `provider` for a backend preset such as `openrouter`, `upstage`, or
`local-lmstudio`. Use `api_spec` for the API contract, usually
`openai_compatible` or `anthropic`.

Command-line flags are enough by themselves; `.env` is only a convenience for
persistent defaults or API key env vars.

**Option A: OpenRouter**
```bash
AGENT_MODEL_PROVIDER=openrouter
AGENT_MODEL_API_SPEC=openai_compatible
AGENT_MODEL_NAME=anthropic/claude-opus-4.5
AGENT_MODEL_BASE_URL=https://openrouter.ai/api/v1
AGENT_MODEL_API_KEY_ENV=OPENROUTER_API_KEY
OPENROUTER_API_KEY=your-key-here

JUDGE_MODEL_PROVIDER=openrouter
JUDGE_MODEL_NAMES=openai/gpt-4o-mini,google/gemini-2.5-flash-lite
JUDGE_MODEL_BASE_URL=https://openrouter.ai/api/v1
JUDGE_MODEL_API_KEY_ENV=OPENROUTER_API_KEY
```

**Option B: Upstage Solar with round-robin keys**
```bash
AGENT_MODEL_PROVIDER=upstage
AGENT_MODEL_API_SPEC=openai_compatible
AGENT_MODEL_NAME=solar-pro3
AGENT_MODEL_BASE_URL=https://api.upstage.ai/v1/solar
AGENT_MODEL_API_KEY_ENVS=UPSTAGE_API_KEY,UPSTAGE_API_KEY2,UPSTAGE_API_KEY3
AGENT_MODEL_CREDENTIAL_STRATEGY=round_robin
UPSTAGE_API_KEY=your-key-here
UPSTAGE_API_KEY2=
UPSTAGE_API_KEY3=
```

**Option C: Local OpenAI-compatible backend**
```bash
# Ollama
AGENT_MODEL_PROVIDER=local-ollama
AGENT_MODEL_API_SPEC=openai_compatible
AGENT_MODEL_NAME=qwen2.5:14b
AGENT_MODEL_BASE_URL=http://localhost:11434/v1
AGENT_MODEL_API_KEY=not-needed

# LM Studio
# AGENT_MODEL_PROVIDER=local-lmstudio
# AGENT_MODEL_BASE_URL=http://localhost:1234/v1
```

**Option D: Two judge models from different providers**
```bash
JUDGE_MODEL_NAMES=openai/gpt-4o-mini,solar-pro3
JUDGE_MODEL_PROVIDERS=openrouter,upstage
JUDGE_MODEL_BASE_URLS=https://openrouter.ai/api/v1,https://api.upstage.ai/v1/solar
JUDGE_MODEL_API_KEY_ENVS=OPENROUTER_API_KEY|OPENROUTER_API_KEY2,UPSTAGE_API_KEY
JUDGE_MODEL_CREDENTIAL_STRATEGIES=round_robin,first
```

Use commas to separate judge models and pipes to group multiple credentials for
one judge model.

Equivalent CLI override:

```bash
python run_benchmark.py \
  --scenario scenarios/test/scenario_bal_large2_0075.json \
  --agents baseline,eduplanner \
  --agent-model-api-spec openai_compatible \
  --agent-model-base-url http://localhost:1234/v1 \
  --agent-model-name local-model \
  --agent-model-api-key not-needed \
  --judge-model-provider openrouter \
  --judge-model-base-url https://openrouter.ai/api/v1 \
  --judge-model-names openai/gpt-4o-mini,google/gemini-2.5-flash-lite \
  --judge-model-api-key "$OPENROUTER_API_KEY"
```

Cloud example without `.env`:

```bash
python run_benchmark.py \
  --scenario scenarios/test/scenario_bal_large2_0075.json \
  --agents baseline,eduplanner \
  --agent-model-provider openrouter \
  --agent-model-base-url https://openrouter.ai/api/v1 \
  --agent-model-name openai/gpt-4o-mini \
  --agent-model-api-key "$OPENROUTER_API_KEY" \
  --judge-model-provider openrouter \
  --judge-model-base-url https://openrouter.ai/api/v1 \
  --judge-model-names google/gemini-2.5-flash-lite \
  --judge-model-api-key "$OPENROUTER_API_KEY"
```

### 3. Run Benchmark

```bash
# Split data into train/test
./scripts/2_split_data.sh

# Run benchmark test
./scripts/3_run_benchmark_test.sh

# Run full benchmark
./scripts/4_run_benchmark.sh
```

Benchmark scripts pass base URLs explicitly. Override `OPENROUTER_BASE_URL`,
`UPSTAGE_BASE_URL`, `AGENT_MODEL_BASE_URL`, or `JUDGE_MODEL_BASE_URL` before
running a script when targeting a proxy or local OpenAI-compatible endpoint.

## Dataset

| Folder | Count | Description |
|--------|-------|-------------|
| scenarios/idld_aligned | 8,842 | SCOPUS paper-based scenarios |
| scenarios/context_variant/part1 | 9,000 | Context Matrix variations |
| scenarios/context_variant/part2 | 7,953 | Context Matrix variations |
| **Total** | **25,795** | All scenarios |

## Agents

| Agent | Type | Description |
|-------|------|-------------|
| Baseline | General | Single LLM call |
| ReAct-ISD | General | ReAct pattern with ISD tools |
| EduPlanner | ISD-Specialized | Multi-agent collaboration |
| ADDIE-Agent | ISD-Specialized | ADDIE framework |
| Dick-Carey-Agent | ISD-Specialized | Dick & Carey model |
| RPISD-Agent | ISD-Specialized | Rapid Prototyping ISD |

## Directory Structure

```
isd-agent-bench/
├── .env.example          # Environment template
├── README.md             # This file
├── run_benchmark.py      # Main benchmark runner
├── scripts/              # Shell scripts
│   ├── 1_setup.sh        # Install dependencies
│   ├── 2_split_data.sh   # Split train/test
│   ├── 3_run_benchmark_test.sh  # Quick test
│   └── 4_run_benchmark.sh       # Full benchmark
├── scenarios/            # ISD scenarios (25,795)
│   ├── idld_aligned/     # SCOPUS-based (8,842)
│   ├── context_variant/  # Augmented (16,953)
│   ├── train/            # Training set
│   └── test/             # Test set
├── agents/               # 6 ISD agents
├── evaluator/            # ADDIE rubric evaluator
└── shared/               # Common schemas, utilities, and LLM factory
```

## LLM Architecture

`shared/llm` owns provider-specific behavior:

- `LLMConfig`: provider-neutral model/base URL/key configuration
- `create_chat_model()`: creates LangChain chat models
- `configure_default_llm()`: supplies tool-style agents with the active config

Agent implementations should not read provider-specific environment variables
or instantiate `ChatOpenAI`/`ChatAnthropic` directly. Add new cloud or local
providers by adding configuration, not by changing agent code.

Judge configuration is intentionally separate and consumed via
`JUDGE_MODEL_*` settings. For multi-provider evaluation, set
`JUDGE_MODEL_NAMES`, `JUDGE_MODEL_PROVIDERS`, `JUDGE_MODEL_BASE_URLS`, and
`JUDGE_MODEL_API_KEY_ENVS` as aligned comma-separated lists. Within one judge
entry, use `|` to provide multiple API keys/env vars for round-robin
credentials.

## License

MIT License
