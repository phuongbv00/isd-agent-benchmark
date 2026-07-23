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

The `scripts/` prefix numbers follow the execution order of the full pipeline
(the end-to-end guide, including the self-hosted RunPod/vLLM serving setup and
the RQ1 "model ladder", lives in the parent monorepo:
`../docs/benchmark_guides.md`):

```bash
# Data (one-time; results are committed)
./scripts/2_split_data.sh                      # 95/5 stratified train/test split
python scripts/2_split_stratified_subset.py \
    --source scenarios/train --target scenarios/train_30              # tuning set
python scripts/2_split_stratified_subset.py \
    --source scenarios/test --target scenarios/test_90 --per-cell 3   # bench set

# Serving env (self-hosted pods): sync env-quads into .env
python scripts/3_sync_runpod_pods_env.py

# Smoke run (1 scenario)
./scripts/3_run_benchmark_test.sh

# Tune / general multi-model runs (one tmux session per slot)
DATASET=train_30 AGENT_MODEL_SLOTS=qwen2b RUN_TAGS=tune1 ./scripts/4_run_benchmark.sh

# RQ1 model ladder (4 sizes x 3 runs x 7 agents on test_90) + audits
python scripts/alignmentgraph-isd-bench/5_audit_preflight.py
./scripts/alignmentgraph-isd-bench/5_run_ladder.sh
python scripts/alignmentgraph-isd-bench/6_audit_inflight.py             # while running
python scripts/alignmentgraph-isd-bench/6_audit_postrun.py              # after each run

# Post-processing: RQ2 alignment scoring, pooling, paper tables
for RUN in results/test_90_benchmark_*_r*; do python scripts/alignmentgraph-isd-bench/7_score_alignment.py "$RUN"; done
python scripts/alignmentgraph-isd-bench/7_pool_ladder_runs.py --auto-glob 'results/test_90_benchmark_*_r*'
python scripts/alignmentgraph-isd-bench/8_gen_paper_tables.py --pooled results/pooled_ladder.json
```

## Configuration Matrix

Configuration is resolved in this order: command-line flags, environment
variables loaded from the shell or `.env`, then defaults in code. The benchmark
uses three configuration layers:

| Layer | Scope | Typical use |
|-------|-------|-------------|
| `run_benchmark.py` CLI flags | One benchmark process | Override dataset, agents, model backend, judge backend, and parallelism for a run. |
| `.env` / shell environment | Persistent defaults and secrets | Store API keys and reusable model defaults without passing every flag. |
| `scripts/4_run_benchmark.sh` variables | Multi-model tmux wrapper | Launch several benchmark sessions, each with a different agent model slot. |

### Benchmark Selection Flags

| Flag | Meaning |
|------|---------|
| `--scenario PATH`, `-s PATH` | Run one scenario JSON file. Results are saved under `results/single_<timestamp>/`. |
| `--dataset NAME`, `-d NAME` | Run a named dataset directory under `scenarios/`, e.g. `train`, `test`, or `test_90`. Cannot be combined with `--variant`. |
| `--variant idld_aligned,context_variant`, `-t ...` | Run existing variant folders. Cannot be combined with `--dataset`. |
| `--agents baseline,eduplanner,...`, `-a ...` | Comma-separated agent IDs to run. |
| `--verbose`, `-v` | Print additional execution and evaluation details. |
| `--install` | Install agent and evaluator packages in editable mode. |
| `--check` | Check whether agent modules can be imported. |

Default agents are `eduplanner`, `baseline`, `react-isd`, `addie-agent`,
`dick-carey-agent`, and `rpisd-agent`. `alignmentgraph-isd` is available but
must be added explicitly with `--agents`.

### Parallelism and Rate Limits

| Flag | Meaning |
|------|---------|
| `--no-parallel` | Disable parallel agent execution inside each scenario. |
| `--max-workers N`, `-w N` | Maximum number of agents running concurrently inside one scenario. |
| `--no-scenario-parallel` | Disable scenario-level parallel execution. |
| `--scenario-max-workers N` | Maximum number of scenarios running concurrently. |
| `--rate-limit conservative\|moderate\|aggressive\|turbo`, `-r ...` | Apply a preset for worker counts and benchmark delay. |

| Rate limit | Agent workers | Scenario workers | Delay |
|------------|---------------|------------------|-------|
| `conservative` | 2 | 2 | `2.0s` |
| `moderate` | 3 | 4 | `0.5s` |
| `aggressive` | 6 | 8 | `0.1s` |
| `turbo` | 6 | 16 | `0.0s` |

`--rate-limit` sets `BENCHMARK_DELAY` and overrides default worker counts unless
`--max-workers` or `--scenario-max-workers` are passed explicitly.

### Agent Model Matrix

The agent model generates instructional-design outputs.

| CLI flag | Environment variable | Meaning |
|----------|----------------------|---------|
| `--agent-model-provider` | `AGENT_MODEL_PROVIDER` | Provider preset such as `openrouter`, `openai`, `upstage`, `anthropic`, `local-ollama`, `local-lmstudio`, or `local-vllm`. |
| `--agent-model-api-spec` | `AGENT_MODEL_API_SPEC` | API contract: `openai_compatible`, `openai`, or `anthropic`. |
| `--agent-model-base-url` | `AGENT_MODEL_BASE_URL` | OpenAI-compatible endpoint URL. |
| `--agent-model-name` | `AGENT_MODEL_NAME` | Model name sent to the backend. |
| `--agent-model-api-key` | `AGENT_MODEL_API_KEY` | Direct API key value. Prefer env vars for reusable runs. |
| `--agent-model-api-key-env` | `AGENT_MODEL_API_KEY_ENV` | Name of an environment variable containing one API key. |
| `--agent-model-api-key-envs` | `AGENT_MODEL_API_KEY_ENVS` | Comma-separated API key env vars for multiple credentials. |

Additional agent-model environment variables:

| Environment variable | Meaning |
|----------------------|---------|
| `AGENT_MODEL_CREDENTIAL_STRATEGY` | Credential selection strategy, usually `first` or `round_robin`. |
| `AGENT_MODEL_TEMPERATURE` | Generation temperature. Default is `0.7`. |
| `AGENT_MODEL_MAX_TOKENS` | Default max tokens. Default is `4096`; some agents override this internally. |
| `REASONING_BUDGET` | Optional reasoning budget for compatible backends. |

### Judge Model Matrix

Judge models evaluate agent outputs. They are configured separately from the
agent model to reduce self-preference bias.

| CLI flag | Environment variable | Meaning |
|----------|----------------------|---------|
| `--judge-model-provider` | `JUDGE_MODEL_PROVIDER` | Default judge provider preset or label. |
| `--judge-model-providers` | `JUDGE_MODEL_PROVIDERS` | Comma-separated provider labels aligned with judge model names. |
| `--judge-model-names` | `JUDGE_MODEL_NAMES` | Comma-separated judge model names. |
| `--judge-model-base-url` | `JUDGE_MODEL_BASE_URL` | Default OpenAI-compatible judge endpoint. |
| `--judge-model-base-urls` | `JUDGE_MODEL_BASE_URLS` | Comma-separated judge endpoints aligned with judge model names. |
| `--judge-model-api-key` | `JUDGE_MODEL_API_KEY` | Default direct judge API key value. |
| `--judge-model-api-keys` | `JUDGE_MODEL_API_KEYS` | Comma-separated judge key groups; use `|` for multiple keys in one model group. |
| `--judge-model-api-key-env` | `JUDGE_MODEL_API_KEY_ENV` | Environment variable containing the default judge API key. |
| `--judge-model-api-key-envs` | `JUDGE_MODEL_API_KEY_ENVS` | Comma-separated judge env-var groups; use `|` for multiple env vars in one model group. |
| `--judge-model-credential-strategies` | `JUDGE_MODEL_CREDENTIAL_STRATEGIES` | Comma-separated credential strategies aligned with judge model names. |

Evaluation uses multi-judge mode by default. Pass `--single-judge` to disable
multi-judge evaluation.

### Full Benchmark Wrapper Matrix

`scripts/4_run_benchmark.sh` launches one tmux session per agent-model slot.

| Wrapper variable | Default | Meaning |
|------------------|---------|---------|
| `AGENTS` | `baseline,eduplanner,react-isd,addie-agent,dick-carey-agent,rpisd-agent` | Agents passed to `run_benchmark.py`. |
| `RATE_LIMIT` | `turbo` | Rate-limit mode passed to `--rate-limit`. |
| `DATASET` | `test` | Dataset passed to `--dataset`. |
| `AGENT_MODEL_SLOTS` | `gpt,gemini,solar` | Comma-separated slot labels. Each slot becomes one tmux session (parallel). |
| `RUN_TAGS` | *(empty)* | Comma/space-separated run tags (e.g. `tune1,tune2`), run sequentially inside each slot session; empty = one untagged run. Same var in `5_run_ladder.sh` (default `r1 r2 r3`). |

Each slot reads these variables, where `<SLOT>` is uppercased and `-` becomes
`_`, for example `GPT_AGENT_MODEL_NAME`:

| Slot variable pattern | Meaning |
|-----------------------|---------|
| `<SLOT>_AGENT_MODEL_PROVIDER` | Provider for this tmux session. |
| `<SLOT>_AGENT_MODEL_BASE_URL` | Base URL for this tmux session. |
| `<SLOT>_AGENT_MODEL_NAME` | Agent model for this tmux session. |
| `<SLOT>_AGENT_MODEL_API_KEY_ENVS` | Comma-separated credential env vars for this tmux session. |
| `<SLOT>_RATE_LIMIT` | Optional per-slot override of the global `RATE_LIMIT` preset. |

### Other Environment Variables

| Environment variable | Meaning |
|----------------------|---------|
| `OPENROUTER_API_KEY`, `UPSTAGE_API_KEY`, `UPSTAGE_API_KEY2`, `UPSTAGE_API_KEY3`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Common secret variables referenced by `*_API_KEY_ENV` or `*_API_KEY_ENVS`. |
| `BENCHMARK_DELAY` | Delay between submissions. Normally set by `--rate-limit`. |
| `ISD_EVAL_PROGRESS` | Internal evaluator progress marker flag set by the benchmark runner. |
| `AGENT_MODEL_MAX_TOKENS_CAP` | Central clamp on every agent's completion budget (`shared/llm/factory.py`). Unset = agents' own budgets apply unchanged. Ladder runs use `8096` for the 16k-context self-hosted models. |
| `AGENT_MODEL_STREAMING` | `1` enables client streaming (`stream_usage=True`); required behind proxies with read timeouts (RunPod/Cloudflare 524). |
| `AGENT_MODEL_DISABLE_THINKING` | `1` requests non-thinking mode (`chat_template_kwargs: {enable_thinking: false}`); required for small Qwen3.5 models whose `<think>` otherwise exhausts the completion budget. |

Note: `SCENARIO_MAX_WORKERS` and `AGENT_MAX_WORKERS` appear in `.env.example`
as optional notes, but `run_benchmark.py` currently reads worker counts from
CLI flags and `--rate-limit`, not directly from those env vars.

## Dataset

| Folder | Count | Description |
|--------|-------|-------------|
| scenarios/idld_aligned | 8,842 | SCOPUS paper-based scenarios |
| scenarios/context_variant/part1 | 9,000 | Context Matrix variations |
| scenarios/context_variant/part2 | 7,953 | Context Matrix variations |
| **Total** | **25,795** | All scenarios |

Derived splits (committed, seed 42): `scenarios/train` (~24.6k) / `scenarios/test`
(~1.2k) from the 95/5 stratified split; `scenarios/train_30` (tuning) and
`scenarios/test_90` (benchmark reporting) are fixed domain×difficulty
stratified subsets with metadata JSONs alongside.

## Agents

| Agent | Type | Description |
|-------|------|-------------|
| Baseline | General | Single LLM call |
| ReAct-ISD | General | ReAct pattern with ISD tools |
| EduPlanner | ISD-Specialized | Multi-agent collaboration |
| ADDIE-Agent | ISD-Specialized | ADDIE framework |
| Dick-Carey-Agent | ISD-Specialized | Dick & Carey model |
| RPISD-Agent | ISD-Specialized | Rapid Prototyping ISD |
| AlignmentGraph-ISD | ISD-Specialized | Alignment-graph multi-agent harness (thesis contribution; thin adapter over the top-level `alignmentgraph-isd` package — not in the default agent list, add via `--agents`) |

## Directory Structure

```
isd-agent-bench/
├── .env.example          # Environment template
├── README.md             # This file
├── run_benchmark.py      # Main benchmark runner
├── scripts/              # Numbered by pipeline order
│   ├── 1_setup.sh                   # Install dependencies
│   ├── 2_split_data.sh              # 95/5 train/test split
│   ├── 2_split_stratified_subset.py # Cut train_30 / test_90 subsets
│   ├── 3_sync_runpod_pods_env.py    # Sync RunPod pod env-quads into .env
│   ├── 3_run_benchmark_test.sh      # Smoke run (1 scenario)
│   ├── 4_run_benchmark.sh           # General tmux launcher (tuning)
│   ├── 5_audit_preflight.py         # Pod/auth/flags/disk checks
│   ├── 5_run_ladder.sh              # RQ1 model-ladder launcher
│   ├── 6_audit_inflight.py          # Red-flag counters while running
│   ├── 6_audit_postrun.py           # Completeness/integrity/metrics audit
│   ├── 7_score_alignment.py         # RQ2 alignment scoring (LLM-free)
│   ├── 7_pool_ladder_runs.py        # Pool runs -> stats + token usage
│   └── 8_gen_paper_tables.py        # Paper .tex tables + macros
├── scenarios/            # ISD scenarios (25,795)
│   ├── idld_aligned/     # SCOPUS-based (8,842)
│   ├── context_variant/  # Augmented (16,953)
│   ├── train/ test/      # 95/5 split (committed)
│   └── train_30/ test_90/  # Fixed stratified subsets (+ *_metadata.json)
├── agents/               # 7 ISD agents (6 defaults + alignmentgraph-isd adapter)
├── evaluator/            # ADDIE rubric + trajectory + alignment evaluator
├── results/              # Run dirs: <dataset>_benchmark_<model>_<tag>_<ts>/
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
