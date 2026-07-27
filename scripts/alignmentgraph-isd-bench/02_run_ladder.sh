#!/bin/bash
# ---------------------------------------------------------------------------
# 02_run_ladder.sh — RQ1 ladder + ablation arms, in ONE set of runs
#
# Runs the full benchmark on 4 Qwen model sizes (0.8B / 2B / 4B / 9B),
# 3 independent runs per size, all 10 agents, dataset test_90.
# One tmux session per model size (sizes run in PARALLEL, but the 3 runs of a
# size run SEQUENTIALLY inside its session — safe w.r.t. per-key rate limits,
# same mechanism as 4_run_benchmark.sh).
#
# The 10 agents = the 7 ladder agents + the 3 ablation arms. Running the arms
# here rather than in their own session is what makes A0 literally the same
# runs for both analyses: no table has to reconcile two different numbers for
# the same full pipeline, and the arms get the ladder's 3-run averaging.
#
# THIS SCRIPT IS EXPENSIVE. It asks for confirmation before launching.
# Total ≈ 4 sizes x 3 runs x 10 agents x 90 scenarios = 10,800 generations
# (+ judge calls). See ../docs/benchmark_guides.md (parent repo) for cost/time
# estimates and how to resume a partial ladder.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../.."
source .env 2>/dev/null || true

# ---------------------------------------------------------------------------
# Ladder config
# ---------------------------------------------------------------------------
DATASET="${DATASET:-test_90}"
RATE_LIMIT="${RATE_LIMIT:-turbo}"
# All 10 agents: 6 benchmark defaults + the proposed system (NOT in defaults)
# + the 3 ablation arms (registry-only agent ids, same adapter with the two
# HarnessRunConfig flags pinned per id). alignmentgraph-isd doubles as the
# ladder's proposed system and the ablation's A0 arm — that is the point.
AGENTS="${AGENTS:-eduplanner,baseline,react-isd,addie-agent,dick-carey-agent,rpisd-agent,alignmentgraph-isd,alignmentgraph-isd-no-verifier,alignmentgraph-isd-no-graph-ctx,alignmentgraph-isd-skeleton}"
# Run tags: 3 independent runs per model size. Comma- or space-separated
# full tags (same var and mechanism as RUN_TAGS in 4_run_benchmark.sh).
# For resume, override e.g.
#   RUN_TAGS="r2,r3" LADDER_SLOTS="qwen08b" ./scripts/alignmentgraph-isd-bench/02_run_ladder.sh
RUN_TAGS="${RUN_TAGS:-r1 r2 r3}"
_RUN_TAGS_LIST="${RUN_TAGS//,/ }"

# ---------------------------------------------------------------------------
# Model slots. Same env-quad convention as 4_run_benchmark.sh:
#   <SLOT>_AGENT_MODEL_PROVIDER / _BASE_URL / _NAME / _API_KEY_ENVS
#   <SLOT>_RATE_LIMIT  (optional; overrides the global RATE_LIMIT for this slot
#                       only — e.g. a smaller Qwen on a tighter-limited provider)
#
# Backing: RunPod Pods created manually in the console from the public vLLM
# template (see ../docs/benchmark_guides.md, section 4), one pod per size.
# BASE_URL depends on the pod id RunPod assigns at deploy time, so there is
# no static default for it -- after the pods are RUNNING, run:
#   python scripts/3_sync_runpod_pods_env.py
# which rewrites the managed env block at the end of .env; the `source .env`
# at the top of this script then picks the quads up. This script hard-fails
# with instructions if BASE_URL is still unset for any slot -- see the
# validation block below.
# ---------------------------------------------------------------------------
LADDER_SLOTS="${LADDER_SLOTS:-qwen08b,qwen2b,qwen4b,qwen9b}"
_LADDER_SLOTS_LIST="${LADDER_SLOTS//,/ }"

# Fallback defaults only -- the managed block in .env (written by
# scripts/3_sync_runpod_pods_env.py) overrides these with what each pod
# actually serves. VLLM_API_KEY is the Bearer token of the self-hosted vLLM
# servers (NOT the RunPod account key).
QWEN08B_AGENT_MODEL_PROVIDER="${QWEN08B_AGENT_MODEL_PROVIDER:-runpod-vllm}"
QWEN08B_AGENT_MODEL_BASE_URL="${QWEN08B_AGENT_MODEL_BASE_URL:-}"
QWEN08B_AGENT_MODEL_NAME="${QWEN08B_AGENT_MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
QWEN08B_AGENT_MODEL_API_KEY_ENVS="${QWEN08B_AGENT_MODEL_API_KEY_ENVS:-VLLM_API_KEY}"

QWEN2B_AGENT_MODEL_PROVIDER="${QWEN2B_AGENT_MODEL_PROVIDER:-runpod-vllm}"
QWEN2B_AGENT_MODEL_BASE_URL="${QWEN2B_AGENT_MODEL_BASE_URL:-}"
QWEN2B_AGENT_MODEL_NAME="${QWEN2B_AGENT_MODEL_NAME:-Qwen/Qwen3.5-2B}"
QWEN2B_AGENT_MODEL_API_KEY_ENVS="${QWEN2B_AGENT_MODEL_API_KEY_ENVS:-VLLM_API_KEY}"

QWEN4B_AGENT_MODEL_PROVIDER="${QWEN4B_AGENT_MODEL_PROVIDER:-runpod-vllm}"
QWEN4B_AGENT_MODEL_BASE_URL="${QWEN4B_AGENT_MODEL_BASE_URL:-}"
QWEN4B_AGENT_MODEL_NAME="${QWEN4B_AGENT_MODEL_NAME:-Qwen/Qwen3.5-4B}"
QWEN4B_AGENT_MODEL_API_KEY_ENVS="${QWEN4B_AGENT_MODEL_API_KEY_ENVS:-VLLM_API_KEY}"

QWEN9B_AGENT_MODEL_PROVIDER="${QWEN9B_AGENT_MODEL_PROVIDER:-runpod-vllm}"
QWEN9B_AGENT_MODEL_BASE_URL="${QWEN9B_AGENT_MODEL_BASE_URL:-}"
QWEN9B_AGENT_MODEL_NAME="${QWEN9B_AGENT_MODEL_NAME:-Qwen/Qwen3.5-9B}"
QWEN9B_AGENT_MODEL_API_KEY_ENVS="${QWEN9B_AGENT_MODEL_API_KEY_ENVS:-VLLM_API_KEY}"

slot_upper() {
    echo "$1" | tr '[:lower:]-' '[:upper:]_'
}

# A slot may be served by SEVERAL pods of the same model: set
# <SLOT>_AGENT_MODEL_BASE_URLS to a comma-separated list and the agents
# round-robin across them (shared/llm/config.py::resolve_base_url), the same
# way EMBED_BASE_URLS works for the encoder. Fold the plural into the singular
# here so everything downstream stays unchanged, and strip whitespace: the
# value is interpolated unquoted into the tmux command string below, so a
# space would split it into two arguments.
for slot in $_LADDER_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    plural_var="${upper}_AGENT_MODEL_BASE_URLS"
    single_var="${upper}_AGENT_MODEL_BASE_URL"
    _urls="${!plural_var:-${!single_var:-}}"
    printf -v "$single_var" '%s' "${_urls//[[:space:]]/}"
done

# Fail fast: an empty *_BASE_URL would otherwise vanish from the unquoted
# run_benchmark.py invocation below (bash drops empty unquoted expansions),
# silently shifting --agent-model-base-url's value onto the next flag.
_missing_base_url=""
for slot in $_LADDER_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    base_url_var="${upper}_AGENT_MODEL_BASE_URL"
    if [[ -z "${!base_url_var:-}" ]]; then
        _missing_base_url="${_missing_base_url} ${slot}"
    fi
done
if [[ -n "$_missing_base_url" ]]; then
    echo "ERROR: no RunPod BASE_URL set for slot(s):${_missing_base_url}" >&2
    echo "Deploy the pods in the console (template link in ../docs/benchmark_guides.md, section 4)," >&2
    echo "then sync their env-quads into .env and relaunch:" >&2
    echo "  python scripts/3_sync_runpod_pods_env.py" >&2
    echo "  ./scripts/alignmentgraph-isd-bench/02_run_ladder.sh" >&2
    exit 1
fi

# Export every API key env referenced by any slot or the judge config so child
# tmux sessions inherit them (same mechanism as 4_run_benchmark.sh).
# NOTE: no associative arrays here -- macOS ships bash 3.2 as /bin/bash (no
# `declare -A`), which combined with `set -u` turns a silently-broken assoc
# array into a hard "unbound variable" crash (see git history for details).
_exported_keys_list=""
register_key_envs() {
    local IFS=','
    for env_name in $1; do
        env_name="${env_name// /}"
        case " $_exported_keys_list " in
            *" $env_name "*) ;;
            *)
                if [[ -n "$env_name" ]]; then
                    export "$env_name"
                    _exported_keys_list="$_exported_keys_list $env_name"
                fi
                ;;
        esac
    done
}
for slot in $_LADDER_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    key_envs_var="${upper}_AGENT_MODEL_API_KEY_ENVS"
    register_key_envs "${!key_envs_var:-}"
done
register_key_envs "${JUDGE_MODEL_API_KEY_ENV:-}"

NUM_SLOTS=$(echo "$_LADDER_SLOTS_LIST" | wc -w | tr -d ' ')
NUM_RUNS=$(echo "$_RUN_TAGS_LIST" | wc -w | tr -d ' ')
NUM_AGENTS=$(echo "${AGENTS//,/ }" | wc -w | tr -d ' ')

# ---------------------------------------------------------------------------
# Cost warning + confirmation
# ---------------------------------------------------------------------------
echo "=============================================================="
echo "  MODEL LADDER — RQ1 (quality vs model scale)"
echo "=============================================================="
echo "  Dataset:    $DATASET"
echo "  Agents ($NUM_AGENTS): $AGENTS"
echo "  Runs/model: $NUM_RUNS (tags: $_RUN_TAGS_LIST)"
echo "  Rate limit: $RATE_LIMIT (global default; per-slot <SLOT>_RATE_LIMIT overrides)"
echo "  Model slots ($NUM_SLOTS):"
for slot in $_LADDER_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    name_var="${upper}_AGENT_MODEL_NAME"
    provider_var="${upper}_AGENT_MODEL_PROVIDER"
    rate_var="${upper}_RATE_LIMIT"
    echo "    - ${slot}: ${!name_var} (provider=${!provider_var}, rate=${!rate_var:-$RATE_LIMIT})"
done
echo ""
echo "  !! WARNING: this launches ~$((NUM_SLOTS * NUM_RUNS)) full benchmark runs"
echo "  !! (~$((NUM_SLOTS * NUM_RUNS * NUM_AGENTS * 90)) agent generations on test_90, plus judge calls)."
echo "  !! This costs REAL MONEY and takes MANY HOURS."
echo "  !! Also billing RunPod GPU time per endpoint on top of this."
echo "=============================================================="
read -r -p "Type 'yes' to launch the ladder: " CONFIRM
if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted. Nothing launched."
    exit 1
fi

# Export slot env quads (+ optional per-slot rate) so both the manifest python
# below and the tmux child shells see them (they may only be defined by this
# script's defaults, not by .env, so they are plain shell vars until exported).
for slot in $_LADDER_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    for suffix in PROVIDER BASE_URL NAME API_KEY_ENVS; do
        export "${upper}_AGENT_MODEL_${suffix}"
    done
    export "${upper}_RATE_LIMIT"
done

# ---------------------------------------------------------------------------
# Manifest (M8 reproducibility): exact model IDs, dataset, agents, run tags,
# git commits of benchmark + alignmentgraph-isd package.
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/ladder_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

BENCH_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
BENCH_DIRTY=$(git status --porcelain 2>/dev/null | head -1 | grep -q . && echo "true" || echo "false")
PKG_COMMIT=$(git -C .. rev-parse HEAD 2>/dev/null || echo "unknown")
PKG_DIRTY=$(git -C .. status --porcelain 2>/dev/null | head -1 | grep -q . && echo "true" || echo "false")

MANIFEST="$LOG_DIR/ladder_manifest.json"
LADDER_TS="$TIMESTAMP" DATASET="$DATASET" AGENTS="$AGENTS" RUN_TAGS="$_RUN_TAGS_LIST" \
RATE_LIMIT="$RATE_LIMIT" SLOTS="$_LADDER_SLOTS_LIST" \
BENCH_COMMIT="$BENCH_COMMIT" BENCH_DIRTY="$BENCH_DIRTY" \
PKG_COMMIT="$PKG_COMMIT" PKG_DIRTY="$PKG_DIRTY" \
JUDGE_MODEL_NAMES="${JUDGE_MODEL_NAMES:-}" JUDGE_MODEL_PROVIDER="${JUDGE_MODEL_PROVIDER:-}" \
python3 - "$MANIFEST" <<'PYEOF'
import json, os, sys

slots = []
for slot in os.environ["SLOTS"].split():
    upper = slot.upper().replace("-", "_")
    slots.append({
        "slot": slot,
        "model": os.environ.get(f"{upper}_AGENT_MODEL_NAME", ""),
        "provider": os.environ.get(f"{upper}_AGENT_MODEL_PROVIDER", ""),
        "base_url": os.environ.get(f"{upper}_AGENT_MODEL_BASE_URL", ""),
        "api_key_envs": os.environ.get(f"{upper}_AGENT_MODEL_API_KEY_ENVS", ""),
        "rate_limit": os.environ.get(f"{upper}_RATE_LIMIT") or os.environ["RATE_LIMIT"],
    })

manifest = {
    "ladder_timestamp": os.environ["LADDER_TS"],
    "dataset": os.environ["DATASET"],
    "agents": os.environ["AGENTS"].split(","),
    "run_tags": os.environ["RUN_TAGS"].split(),
    "rate_limit": os.environ["RATE_LIMIT"],
    "model_slots": slots,
    "judge_models": [m for m in os.environ.get("JUDGE_MODEL_NAMES", "").split(",") if m],
    "judge_provider": os.environ.get("JUDGE_MODEL_PROVIDER", ""),
    "git": {
        "isd_agent_benchmark_commit": os.environ["BENCH_COMMIT"],
        "isd_agent_benchmark_dirty": os.environ["BENCH_DIRTY"] == "true",
        "master_thesis_commit": os.environ["PKG_COMMIT"],
        "master_thesis_dirty": os.environ["PKG_DIRTY"] == "true",
        "note": "master_thesis_commit covers the alignmentgraph-isd package "
                "(tracked directly in the parent monorepo).",
    },
    "run_dir_pattern": "results/{dataset}_benchmark_{model_safe}_{run_tag}_{timestamp}",
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"Manifest written: {sys.argv[1]}")
PYEOF

# ---------------------------------------------------------------------------
# Launch: one tmux session per model size; runs r1..rN sequential inside it.
# ---------------------------------------------------------------------------
for slot in $_LADDER_SLOTS_LIST; do
    tmux kill-session -t "ladder-${slot}" 2>/dev/null
done

i=0
for slot in $_LADDER_SLOTS_LIST; do
    i=$((i + 1))
    upper=$(slot_upper "$slot")
    provider_var="${upper}_AGENT_MODEL_PROVIDER"
    base_url_var="${upper}_AGENT_MODEL_BASE_URL"
    name_var="${upper}_AGENT_MODEL_NAME"
    key_envs_var="${upper}_AGENT_MODEL_API_KEY_ENVS"
    rate_var="${upper}_RATE_LIMIT"
    slot_rate="${!rate_var:-$RATE_LIMIT}"

    session="ladder-${slot}"
    log_file="$LOG_DIR/${slot}.log"

    echo "[${i}/${NUM_SLOTS}] ${slot}: ${!name_var} (rate=$slot_rate) x runs [$_RUN_TAGS_LIST]..."

    tmux new-session -d -s "$session" \
"cd $(pwd) || exit 1; source .env 2>/dev/null || true; \
for tag in $_RUN_TAGS_LIST; do \
    echo \"=== ${slot} run \$tag starting: \$(date) ===\"; \
    python run_benchmark.py \
        --dataset $DATASET \
        --agents $AGENTS \
        --rate-limit $slot_rate \
        --agent-model-provider ${!provider_var} \
        --agent-model-base-url ${!base_url_var} \
        --agent-model-name ${!name_var} \
        --agent-model-api-key-envs ${!key_envs_var} \
        --run-tag \$tag \
        2>&1 | tee -a $log_file; \
    echo \"=== ${slot} run \$tag finished: \$(date) ===\"; \
done; echo 'Ladder slot done!'; read"
done

echo ""
echo "=============================================================="
echo "  ${NUM_SLOTS} ladder tmux sessions running (${NUM_RUNS} sequential runs each)"
echo "=============================================================="
echo ""
echo "Manifest: $MANIFEST"
echo ""
echo "Attach:"
for slot in $_LADDER_SLOTS_LIST; do
    echo "  tmux attach -t ladder-${slot}"
done
echo ""
echo "Logs:"
for slot in $_LADDER_SLOTS_LIST; do
    echo "  tail -f $LOG_DIR/${slot}.log"
done
echo ""
echo "Stop everything: tmux kill-server"
echo ""
echo "After all runs finish, pool with:"
echo "  python scripts/alignmentgraph-isd-bench/07_pool_ladder_runs.py --auto-glob 'results/${DATASET}_benchmark_*_r*'"
