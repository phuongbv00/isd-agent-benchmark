#!/bin/bash
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

# ---------------------------------------------------------------------------
# Shared benchmark config
# ---------------------------------------------------------------------------
AGENTS="${AGENTS:-baseline,eduplanner,react-isd,addie-agent,dick-carey-agent,rpisd-agent}"
RATE_LIMIT="${RATE_LIMIT:-turbo}"
DATASET="${DATASET:-test}"

# ---------------------------------------------------------------------------
# Agent model slots — comma-separated session labels. Each slot reads:
#   <SLOT>_AGENT_MODEL_PROVIDER
#   <SLOT>_AGENT_MODEL_BASE_URL
#   <SLOT>_AGENT_MODEL_NAME
#   <SLOT>_AGENT_MODEL_API_KEY_ENVS   (comma-separated env names; 1+ allowed)
#   <SLOT>_RATE_LIMIT                 (optional; overrides the global RATE_LIMIT
#                                      for this slot only — different providers
#                                      have different rate limits)
# Override AGENT_MODEL_SLOTS to add/remove/reorder parallel sessions, e.g.
#   AGENT_MODEL_SLOTS="claude,gpt" ./scripts/4_run_benchmark.sh
# ---------------------------------------------------------------------------
AGENT_MODEL_SLOTS="${AGENT_MODEL_SLOTS:-gpt,gemini,solar}"
# Normalize to whitespace-separated for word-split loops below.
_AGENT_MODEL_SLOTS_LIST="${AGENT_MODEL_SLOTS//,/ }"

slot_upper() {
    echo "$1" | tr '[:lower:]-' '[:upper:]_'
}

# Export every API key env referenced by any slot or by the judge config so
# child tmux sessions inherit them.
# NOTE: no associative arrays here -- macOS ships bash 3.2 as /bin/bash (no
# `declare -A`), which silently degrades to a broken integer-indexed array
# (see 6_run_ladder.sh, which has `set -u` and crashes outright on this).
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
for slot in $_AGENT_MODEL_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    key_envs_var="${upper}_AGENT_MODEL_API_KEY_ENVS"
    register_key_envs "${!key_envs_var}"
done
register_key_envs "$JUDGE_MODEL_API_KEY_ENV"

# Log dir
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/benchmark_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

NUM_SLOTS=$(echo "$_AGENT_MODEL_SLOTS_LIST" | wc -w | tr -d ' ')

echo "=============================================="
echo "  Running ${NUM_SLOTS} models in parallel via tmux"
echo "  Dataset: $DATASET"
echo "  Rate Limit: $RATE_LIMIT (global default; per-slot <SLOT>_RATE_LIMIT overrides)"
echo "  Model config: --agent-model-* and --judge-model-* flags"
echo "  Judge models: $JUDGE_MODEL_NAMES (provider=$JUDGE_MODEL_PROVIDER, base=$JUDGE_MODEL_BASE_URL)"
echo "  Log directory: $LOG_DIR"
echo ""
echo "  Agent model slots:"
for slot in $_AGENT_MODEL_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    provider_var="${upper}_AGENT_MODEL_PROVIDER"
    name_var="${upper}_AGENT_MODEL_NAME"
    key_envs_var="${upper}_AGENT_MODEL_API_KEY_ENVS"
    rate_var="${upper}_RATE_LIMIT"
    echo "    - ${slot}: ${!name_var} (provider=${!provider_var}, keys=${!key_envs_var}, rate=${!rate_var:-$RATE_LIMIT})"
done
echo "=============================================="

# Kill any prior sessions for these slots so re-runs are clean
for slot in $_AGENT_MODEL_SLOTS_LIST; do
    tmux kill-session -t "bench-${slot}" 2>/dev/null
done

i=0
for slot in $_AGENT_MODEL_SLOTS_LIST; do
    i=$((i + 1))
    upper=$(slot_upper "$slot")
    provider_var="${upper}_AGENT_MODEL_PROVIDER"
    base_url_var="${upper}_AGENT_MODEL_BASE_URL"
    name_var="${upper}_AGENT_MODEL_NAME"
    key_envs_var="${upper}_AGENT_MODEL_API_KEY_ENVS"
    rate_var="${upper}_RATE_LIMIT"
    slot_rate="${!rate_var:-$RATE_LIMIT}"

    session="bench-${slot}"
    log_file="$LOG_DIR/${slot}.log"

    echo "[${i}/${NUM_SLOTS}] ${slot}: ${!name_var} (rate=$slot_rate)..."

    tmux new-session -d -s "$session" \
"cd $(pwd) || exit 1; source .env 2>/dev/null || true; python run_benchmark.py \
    --dataset $DATASET \
    --agents $AGENTS \
    --rate-limit $slot_rate \
    --agent-model-provider ${!provider_var} \
    --agent-model-base-url ${!base_url_var} \
    --agent-model-name ${!name_var} \
    --agent-model-api-key-envs ${!key_envs_var} \
    2>&1 | tee $log_file; echo 'Done!'; read"
done

echo ""
echo "=============================================="
echo "  ${NUM_SLOTS} tmux sessions running!"
echo "=============================================="
echo ""
echo "📺 Session list:"
tmux ls
echo ""
echo "📊 Live view (attach):"
for slot in $_AGENT_MODEL_SLOTS_LIST; do
    echo "  tmux attach -t bench-${slot}"
done
echo ""
echo "📁 Log files (saved live):"
for slot in $_AGENT_MODEL_SLOTS_LIST; do
    echo "  tail -f $LOG_DIR/${slot}.log"
done
echo ""
echo "⌨️ Detach from a session: Ctrl+B, D"
echo ""
echo "🛑 Stop everything:"
echo "  tmux kill-server"
