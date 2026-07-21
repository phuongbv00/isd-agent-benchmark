# Model-ladder runs (RQ1) — manual launch guide

`6_run_ladder.sh` launches the RQ1 "model ladder": the full benchmark on
4 Qwen sizes (0.8B / 2B / 4B / 9B), **3 independent runs per size**, all
**7 agents** (6 defaults + `alignmentgraph-isd`), dataset **test_90**.

## Scale and cost — read before launching

- 4 sizes x 3 runs x 7 agents x 90 scenarios ≈ **7,560 agent generations**,
  plus 2 judge-model evaluations per generation (multi-judge default).
- The previous single `test_90` run of Qwen3.5-9B (7 agents, 1 run) took on
  the order of **several hours** at `turbo` rate limits. Expect the whole
  ladder to take **roughly a day** of wall-clock time (sizes run in parallel,
  the 3 runs of a size run sequentially), and budget API cost accordingly
  (rough order: 12x the cost of that single 9B run; small sizes are cheaper
  per token but produce a similar call count, and judge cost is identical
  across sizes).
- **Verify the OpenRouter model IDs first.** Only the 9B ID
  (`qwen/qwen3.5-9b:nitro`) is confirmed from the 2026-07-14 artifact run.
  The 0.8B/2B/4B defaults in the script are *inferred from the 9B naming
  pattern* and marked `TODO(user)` — check https://openrouter.ai/models and
  override via env if needed:

  ```bash
  QWEN08B_AGENT_MODEL_NAME="qwen/<exact-id>" \
  QWEN2B_AGENT_MODEL_NAME="qwen/<exact-id>" \
  QWEN4B_AGENT_MODEL_NAME="qwen/<exact-id>" \
  ./scripts/6_run_ladder.sh
  ```

## Self-hosting sizes on RunPod (alternative to OpenRouter)

Instead of guessing OpenRouter IDs for the small Qwen sizes (see the
TODO(user) warning above), you can self-host each size as a RunPod
Serverless vLLM endpoint and point the ladder's env-quads at it — no changes
needed to `6_run_ladder.sh` itself, since `--agent-model-base-url` already
takes any OpenAI-compatible URL (`shared/llm`).

```bash
export RUNPOD_API_KEY=...
python scripts/runpod_deploy_ladder.py gpu-types                       # verify GPU ids first
python scripts/runpod_deploy_ladder.py create-volume --name qwen-ladder-cache --size 100
python scripts/runpod_deploy_ladder.py deploy --volume-id <id-from-above>
source results/runpod_ladder_<ts>/ladder_env.sh
./scripts/6_run_ladder.sh
# after the ladder finishes:
python scripts/runpod_deploy_ladder.py teardown --manifest results/runpod_ladder_<ts>/manifest.json
```

**Verify the HF model IDs and GPU tiers first** — same caution as the
OpenRouter TODOs: the defaults in `runpod_deploy_ladder.py` (`DEFAULT_SLOTS`)
are inferred names, not confirmed checkpoints. Endpoints scale to zero
(`workersMin=0`) so idle cost is ~0, but review before deploying.

## Per-slot rate limit

`RATE_LIMIT` (default `turbo`) is the global rate-limit preset. Because
different providers/models impose different limits, each slot may override it
with `<SLOT>_RATE_LIMIT`; unset slots fall back to the global value. The chosen
rate is echoed per slot at launch and recorded per slot in the manifest.

```bash
# Global moderate, but throttle the small models on a tighter-limited endpoint:
RATE_LIMIT=moderate \
QWEN08B_RATE_LIMIT=conservative \
QWEN2B_RATE_LIMIT=conservative \
./scripts/6_run_ladder.sh
```

The same `<SLOT>_RATE_LIMIT` override works in `4_run_benchmark.sh`.

## How to run

```bash
cd isd-agent-benchmark
source .env            # OPENROUTER_API_KEY + judge config must be set
./scripts/6_run_ladder.sh
# type "yes" at the confirmation prompt
```

Mechanics (same as `4_run_benchmark.sh`):

- One **tmux session per model size** (`ladder-qwen08b`, `ladder-qwen2b`,
  `ladder-qwen4b`, `ladder-qwen9b`) — sizes run in parallel.
- Inside each session the 3 runs execute **sequentially** with
  `--run-tag r1|r2|r3`, so each run lands in its own results dir:
  `results/test_90_benchmark_<model>_r<N>_<timestamp>/`.
- A reproducibility manifest is written **before** launch to
  `results/ladder_<timestamp>/ladder_manifest.json` (exact model IDs,
  provider/base URL, dataset, agents, run tags, judge models, and the git
  commits of both the benchmark checkout and the parent monorepo that
  contains the `alignmentgraph-isd` package). Keep this file with the runs.
- Per-slot logs: `results/ladder_<timestamp>/<slot>.log`.

Monitor:

```bash
tmux ls
tmux attach -t ladder-qwen2b     # detach: Ctrl+B, D
tail -f results/ladder_*/qwen2b.log
```

## Resume / partial re-runs

Each (size, run) is an independent `run_benchmark.py` invocation with its own
results dir, so resuming = re-launching only what is missing. Examples:

```bash
# Only 4B, runs 2 and 3 (run 1 already finished):
LADDER_SLOTS="qwen4b" RUNS="2 3" ./scripts/6_run_ladder.sh

# Only 0.8B, all 3 runs:
LADDER_SLOTS="qwen08b" ./scripts/6_run_ladder.sh
```

A re-launch writes a new `ladder_<timestamp>/` manifest; keep all manifests.
Do **not** delete a partially-scored run dir — the pooling script reports
missing/failed scenarios per run explicitly, and a re-run of the same tag is
distinguishable by its timestamp (use the newer dir, delete the aborted one
only after pooling confirms the newer one is complete).

## After the runs

```bash
# Pool 3 runs per size into per-scenario means + stats (Wilcoxon/Holm/bootstrap):
python scripts/7_pool_ladder_runs.py --auto-glob 'results/test_90_benchmark_*_r*'

# Generate the paper tables + macros from the pooled JSON:
python scripts/8_gen_paper_tables.py --pooled results/pooled_ladder.json
```

See `7_pool_ladder_runs.py --help` for explicit per-model dir mapping if the
auto-glob grouping picks up unwanted dirs.
