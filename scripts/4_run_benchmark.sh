#!/bin/bash
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

# ---------------------------------------------------------------------------
# Shared benchmark config
# ---------------------------------------------------------------------------
AGENTS="${AGENTS:-baseline,eduplanner,react-isd,addie-agent,dick-carey-agent,rpisd-agent}"
RATE_LIMIT="${RATE_LIMIT:-turbo}"
DATASET="${DATASET:-test}"

OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
UPSTAGE_BASE_URL="${UPSTAGE_BASE_URL:-https://api.upstage.ai/v1/solar}"

# ---------------------------------------------------------------------------
# Judge model config
# ---------------------------------------------------------------------------
JUDGE_MODEL_PROVIDER="${JUDGE_MODEL_PROVIDER:-openrouter}"
JUDGE_MODEL_BASE_URL="${JUDGE_MODEL_BASE_URL:-$OPENROUTER_BASE_URL}"
JUDGE_MODEL_NAMES="${JUDGE_MODEL_NAMES:-openai/gpt-4o-mini,google/gemini-2.5-flash-lite}"
JUDGE_MODEL_API_KEY_ENV="${JUDGE_MODEL_API_KEY_ENV:-OPENROUTER_API_KEY}"

# ---------------------------------------------------------------------------
# Agent model slots — comma-separated session labels. Each slot reads:
#   <SLOT>_AGENT_MODEL_PROVIDER
#   <SLOT>_AGENT_MODEL_BASE_URL
#   <SLOT>_AGENT_MODEL_NAME
#   <SLOT>_AGENT_MODEL_API_KEY_ENVS   (comma-separated env names; 1+ allowed)
# Override AGENT_MODEL_SLOTS to add/remove/reorder parallel sessions, e.g.
#   AGENT_MODEL_SLOTS="claude,gpt" ./scripts/4_run_benchmark.sh
# ---------------------------------------------------------------------------
AGENT_MODEL_SLOTS="${AGENT_MODEL_SLOTS:-gpt,gemini,solar}"
# Normalize to whitespace-separated for word-split loops below.
_AGENT_MODEL_SLOTS_LIST="${AGENT_MODEL_SLOTS//,/ }"

# Built-in slot defaults (overridable per slot)
GPT_AGENT_MODEL_PROVIDER="${GPT_AGENT_MODEL_PROVIDER:-openrouter}"
GPT_AGENT_MODEL_BASE_URL="${GPT_AGENT_MODEL_BASE_URL:-$OPENROUTER_BASE_URL}"
GPT_AGENT_MODEL_NAME="${GPT_AGENT_MODEL_NAME:-openai/gpt-5-mini}"
GPT_AGENT_MODEL_API_KEY_ENVS="${GPT_AGENT_MODEL_API_KEY_ENVS:-OPENROUTER_API_KEY}"

GEMINI_AGENT_MODEL_PROVIDER="${GEMINI_AGENT_MODEL_PROVIDER:-openrouter}"
GEMINI_AGENT_MODEL_BASE_URL="${GEMINI_AGENT_MODEL_BASE_URL:-$OPENROUTER_BASE_URL}"
GEMINI_AGENT_MODEL_NAME="${GEMINI_AGENT_MODEL_NAME:-google/gemini-3-flash-preview}"
GEMINI_AGENT_MODEL_API_KEY_ENVS="${GEMINI_AGENT_MODEL_API_KEY_ENVS:-OPENROUTER_API_KEY}"

SOLAR_AGENT_MODEL_PROVIDER="${SOLAR_AGENT_MODEL_PROVIDER:-upstage}"
SOLAR_AGENT_MODEL_BASE_URL="${SOLAR_AGENT_MODEL_BASE_URL:-$UPSTAGE_BASE_URL}"
SOLAR_AGENT_MODEL_NAME="${SOLAR_AGENT_MODEL_NAME:-solar-pro3}"
SOLAR_AGENT_MODEL_API_KEY_ENVS="${SOLAR_AGENT_MODEL_API_KEY_ENVS:-UPSTAGE_API_KEY,UPSTAGE_API_KEY2,UPSTAGE_API_KEY3}"

slot_upper() {
    echo "$1" | tr '[:lower:]-' '[:upper:]_'
}

# Export every API key env referenced by any slot or by the judge config so
# child tmux sessions inherit them.
declare -A _exported_keys=()
register_key_envs() {
    local IFS=','
    for env_name in $1; do
        env_name="${env_name// /}"
        if [[ -n "$env_name" && -z "${_exported_keys[$env_name]:-}" ]]; then
            export "$env_name"
            _exported_keys[$env_name]=1
        fi
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
echo "  ${NUM_SLOTS}개 모델 tmux 병렬 실행"
echo "  Dataset: $DATASET"
echo "  Rate Limit: $RATE_LIMIT"
echo "  Model config: --agent-model-* and --judge-model-* flags"
echo "  Judge models: $JUDGE_MODEL_NAMES (provider=$JUDGE_MODEL_PROVIDER, base=$JUDGE_MODEL_BASE_URL)"
echo "  로그 디렉토리: $LOG_DIR"
echo ""
echo "  Agent model slots:"
for slot in $_AGENT_MODEL_SLOTS_LIST; do
    upper=$(slot_upper "$slot")
    provider_var="${upper}_AGENT_MODEL_PROVIDER"
    name_var="${upper}_AGENT_MODEL_NAME"
    key_envs_var="${upper}_AGENT_MODEL_API_KEY_ENVS"
    echo "    - ${slot}: ${!name_var} (provider=${!provider_var}, keys=${!key_envs_var})"
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

    session="bench-${slot}"
    log_file="$LOG_DIR/${slot}.log"

    echo "[${i}/${NUM_SLOTS}] ${slot}: ${!name_var}..."

    tmux new-session -d -s "$session" \
"cd $(pwd) || exit 1; source .env 2>/dev/null || true; python run_benchmark.py \
    --dataset $DATASET \
    --agents $AGENTS \
    --rate-limit $RATE_LIMIT \
    --agent-model-provider ${!provider_var} \
    --agent-model-base-url ${!base_url_var} \
    --agent-model-name ${!name_var} \
    --agent-model-api-key-envs ${!key_envs_var} \
    --judge-model-provider $JUDGE_MODEL_PROVIDER \
    --judge-model-base-url $JUDGE_MODEL_BASE_URL \
    --judge-model-names $JUDGE_MODEL_NAMES \
    --judge-model-api-key-env $JUDGE_MODEL_API_KEY_ENV \
    2>&1 | tee $log_file; echo '완료!'; read"
done

echo ""
echo "=============================================="
echo "  ${NUM_SLOTS}개 tmux 세션 실행 중!"
echo "=============================================="
echo ""
echo "📺 세션 목록:"
tmux ls
echo ""
echo "📊 실시간 확인 (attach):"
for slot in $_AGENT_MODEL_SLOTS_LIST; do
    echo "  tmux attach -t bench-${slot}"
done
echo ""
echo "📁 로그 파일 (실시간 저장):"
for slot in $_AGENT_MODEL_SLOTS_LIST; do
    echo "  tail -f $LOG_DIR/${slot}.log"
done
echo ""
echo "⌨️  세션에서 나가기: Ctrl+B, D"
echo ""
echo "🛑 전체 중지:"
echo "  tmux kill-server"
