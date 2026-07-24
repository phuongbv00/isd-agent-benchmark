#!/bin/bash
# ---------------------------------------------------------------------------
# ablation/5_run_ablation.sh — ablation-study launcher (thesis defense)
#
# Runs the 4 ablation arms of alignmentgraph-isd as 4 separate agents inside
# ONE benchmark run per Qwen size (0.8B / 2B / 4B / 9B), dataset test_90,
# a single run per size (tag `abl1`). All arms share one judge session per
# scenario, so arm-vs-arm deltas are free of judge drift; the arm set also
# satisfies the >=2-successful-agents evaluation gate by itself.
#
#   A0 alignmentgraph-isd                (full pipeline — re-run in-session;
#                                         the ladder r1–r3 runs stay as an A0
#                                         consistency check, see 7_pool_ablation_runs.py)
#   A1 alignmentgraph-isd-no-verifier    (QC verify+repair off)
#   A2 alignmentgraph-isd-no-graph-ctx   (graph-context prompt injection off)
#   A3 alignmentgraph-isd-skeleton       (both off)
#
# One tmux session per model size (sizes run in PARALLEL), same slot/env-quad
# conventions as 5_run_ladder.sh. Tag `abl1` deliberately does not match the
# ladder's r\d+ tags, so the ladder pooling auto-glob never ingests these dirs.
#
# DRY_RUN=1 prints the per-slot commands without launching anything.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../../.."
source .env 2>/dev/null || true

# ---------------------------------------------------------------------------
# Ablation config
# ---------------------------------------------------------------------------
DATASET="${DATASET:-test_90}"
RATE_LIMIT="${RATE_LIMIT:-turbo}"
# Deliberately NOT the generic $AGENTS var: .env exports AGENTS for the
# ladder/benchmark scripts and would silently replace the arm list here.
# Override with ABLATION_AGENTS if you ever need a different arm set.
AGENTS="${ABLATION_AGENTS:-alignmentgraph-isd,alignmentgraph-isd-no-verifier,alignmentgraph-isd-no-graph-ctx,alignmentgraph-isd-skeleton}"
# Single run per size; bump to abl2 later if a delta lands borderline.
RUN_TAG="${RUN_TAG:-abl1}"

ABLATION_SLOTS="${ABLATION_SLOTS:-qwen08b,qwen2b,qwen4b,qwen9b}"
_ABLATION_SLOTS_LIST="${ABLATION_SLOTS//,/ }"

# Fallback defaults only -- the managed block in .env (written by
# scripts/3_sync_runpod_pods_env.py) overrides these with what each pod
# actually serves (same mechanism as 5_run_ladder.sh).
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

# Fail fast: an empty *_BASE_URL would otherwise vanish from the unquoted
# run_benchmark.py invocation below (bash drops empty unquoted expansions),
# silently shifting --agent-model-base-url's value onto the next flag.
_missing_base_url=""
for slot in $_ABLATION_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    base_url_var="${upper}_AGENT_MODEL_BASE_URL"
    if [[ -z "${!base_url_var:-}" ]]; then
        _missing_base_url="${_missing_base_url} ${slot}"
    fi
done
if [[ -n "$_missing_base_url" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "ERROR: no RunPod BASE_URL set for slot(s):${_missing_base_url}" >&2
    echo "Deploy the pods in the console (template link in ../docs/benchmark_guides.md, section 4)," >&2
    echo "then sync their env-quads into .env and relaunch:" >&2
    echo "  python scripts/3_sync_runpod_pods_env.py" >&2
    echo "  ./scripts/alignmentgraph-isd-bench/ablation/5_run_ablation.sh" >&2
    exit 1
fi

# Export every API key env referenced by any slot or the judge config so child
# tmux sessions inherit them. No associative arrays -- macOS bash 3.2 (see
# 5_run_ladder.sh for the full rationale).
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
for slot in $_ABLATION_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    key_envs_var="${upper}_AGENT_MODEL_API_KEY_ENVS"
    register_key_envs "${!key_envs_var:-}"
done
register_key_envs "${JUDGE_MODEL_API_KEY_ENV:-}"

NUM_SLOTS=$(echo "$_ABLATION_SLOTS_LIST" | wc -w | tr -d ' ')
NUM_AGENTS=$(echo "${AGENTS//,/ }" | wc -w | tr -d ' ')

# ---------------------------------------------------------------------------
# Cost warning + confirmation
# ---------------------------------------------------------------------------
echo "=============================================================="
echo "  ABLATION STUDY — component contribution (defense)"
echo "=============================================================="
echo "  Dataset:    $DATASET"
echo "  Arms ($NUM_AGENTS):   $AGENTS"
echo "  Run tag:    $RUN_TAG (single run per size)"
echo "  Rate limit: $RATE_LIMIT (global default; per-slot <SLOT>_RATE_LIMIT overrides)"
echo "  Model slots ($NUM_SLOTS):"
for slot in $_ABLATION_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    name_var="${upper}_AGENT_MODEL_NAME"
    provider_var="${upper}_AGENT_MODEL_PROVIDER"
    rate_var="${upper}_RATE_LIMIT"
    echo "    - ${slot}: ${!name_var} (provider=${!provider_var}, rate=${!rate_var:-$RATE_LIMIT})"
done
echo ""
echo "  !! WARNING: this launches ${NUM_SLOTS} full benchmark runs"
echo "  !! (~$((NUM_SLOTS * NUM_AGENTS * 90)) graph-heavy agent generations on test_90, plus judge calls)."
echo "  !! This costs REAL MONEY and takes MANY HOURS."
echo "  !! Also billing RunPod GPU time per endpoint on top of this."
echo "=============================================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] Commands that WOULD run (one tmux session per slot):"
    for slot in $_ABLATION_SLOTS_LIST; do
        upper=$(slot_upper "$slot")
        provider_var="${upper}_AGENT_MODEL_PROVIDER"
        base_url_var="${upper}_AGENT_MODEL_BASE_URL"
        name_var="${upper}_AGENT_MODEL_NAME"
        key_envs_var="${upper}_AGENT_MODEL_API_KEY_ENVS"
        rate_var="${upper}_RATE_LIMIT"
        echo ""
        echo "  [ablation-${slot}]"
        echo "  python run_benchmark.py --dataset $DATASET --agents $AGENTS \\"
        echo "      --rate-limit ${!rate_var:-$RATE_LIMIT} \\"
        echo "      --agent-model-provider ${!provider_var} \\"
        echo "      --agent-model-base-url ${!base_url_var:-<UNSET>} \\"
        echo "      --agent-model-name ${!name_var} \\"
        echo "      --agent-model-api-key-envs ${!key_envs_var} \\"
        echo "      --run-tag $RUN_TAG"
    done
    exit 0
fi

read -r -p "Type 'yes' to launch the ablation runs: " CONFIRM
if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted. Nothing launched."
    exit 1
fi

# Export slot env quads (+ optional per-slot rate) so both the manifest python
# below and the tmux child shells see them.
for slot in $_ABLATION_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    for suffix in PROVIDER BASE_URL NAME API_KEY_ENVS; do
        export "${upper}_AGENT_MODEL_${suffix}"
    done
    export "${upper}_RATE_LIMIT"
done

# ---------------------------------------------------------------------------
# Manifest (M8 reproducibility): arms, dataset, model slots, git commits of
# benchmark + alignmentgraph-isd package.
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/ablation_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

BENCH_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
BENCH_DIRTY=$(git status --porcelain 2>/dev/null | head -1 | grep -q . && echo "true" || echo "false")
PKG_COMMIT=$(git -C .. rev-parse HEAD 2>/dev/null || echo "unknown")
PKG_DIRTY=$(git -C .. status --porcelain 2>/dev/null | head -1 | grep -q . && echo "true" || echo "false")

MANIFEST="$LOG_DIR/ablation_manifest.json"
ABLATION_TS="$TIMESTAMP" DATASET="$DATASET" AGENTS="$AGENTS" RUN_TAG="$RUN_TAG" \
RATE_LIMIT="$RATE_LIMIT" SLOTS="$_ABLATION_SLOTS_LIST" \
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
    "ablation_timestamp": os.environ["ABLATION_TS"],
    "dataset": os.environ["DATASET"],
    "arms": os.environ["AGENTS"].split(","),
    "run_tag": os.environ["RUN_TAG"],
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
# Launch: one tmux session per model size, a single tagged run each.
# ---------------------------------------------------------------------------
for slot in $_ABLATION_SLOTS_LIST; do
    tmux kill-session -t "ablation-${slot}" 2>/dev/null
done

i=0
for slot in $_ABLATION_SLOTS_LIST; do
    i=$((i + 1))
    upper=$(slot_upper "$slot")
    provider_var="${upper}_AGENT_MODEL_PROVIDER"
    base_url_var="${upper}_AGENT_MODEL_BASE_URL"
    name_var="${upper}_AGENT_MODEL_NAME"
    key_envs_var="${upper}_AGENT_MODEL_API_KEY_ENVS"
    rate_var="${upper}_RATE_LIMIT"
    slot_rate="${!rate_var:-$RATE_LIMIT}"

    session="ablation-${slot}"
    log_file="$LOG_DIR/${slot}.log"

    echo "[${i}/${NUM_SLOTS}] ${slot}: ${!name_var} (rate=$slot_rate) run [$RUN_TAG]..."

    tmux new-session -d -s "$session" \
"cd $(pwd) || exit 1; source .env 2>/dev/null || true; \
echo \"=== ${slot} ablation run $RUN_TAG starting: \$(date) ===\"; \
python run_benchmark.py \
    --dataset $DATASET \
    --agents $AGENTS \
    --rate-limit $slot_rate \
    --agent-model-provider ${!provider_var} \
    --agent-model-base-url ${!base_url_var} \
    --agent-model-name ${!name_var} \
    --agent-model-api-key-envs ${!key_envs_var} \
    --run-tag $RUN_TAG \
    2>&1 | tee -a $log_file; \
echo \"=== ${slot} ablation run $RUN_TAG finished: \$(date) ===\"; \
echo 'Ablation slot done!'; read"
done

echo ""
echo "=============================================================="
echo "  ${NUM_SLOTS} ablation tmux sessions running (1 run each)"
echo "=============================================================="
echo ""
echo "Manifest: $MANIFEST"
echo ""
echo "Attach:"
for slot in $_ABLATION_SLOTS_LIST; do
    echo "  tmux attach -t ablation-${slot}"
done
echo ""
echo "Logs:"
for slot in $_ABLATION_SLOTS_LIST; do
    echo "  tail -f $LOG_DIR/${slot}.log"
done
echo ""
echo "Stop everything: tmux kill-server"
echo ""
echo "After the runs finish:"
echo "  python scripts/alignmentgraph-isd-bench/6_audit_postrun.py results/${DATASET}_benchmark_*_${RUN_TAG}_*"
echo "  python scripts/alignmentgraph-isd-bench/7_score_alignment.py <each run dir>   # RQ2 per arm"
echo "  python scripts/alignmentgraph-isd-bench/ablation/7_pool_ablation_runs.py --auto-glob 'results/${DATASET}_benchmark_*_${RUN_TAG}_*'"
