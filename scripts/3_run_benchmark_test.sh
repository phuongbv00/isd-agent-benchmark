#!/bin/bash
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

AGENTS="${AGENTS:-baseline,eduplanner,react-isd,addie-agent,dick-carey-agent,rpisd-agent}"
RATE_LIMIT="${RATE_LIMIT:-turbo}"
SCENARIO="${SCENARIO:-scenarios/test/scenario_bal_large2_0075.json}"

python run_benchmark.py \
  --scenario "$SCENARIO" \
  --agents "$AGENTS" \
  --rate-limit "$RATE_LIMIT" \
  --verbose
