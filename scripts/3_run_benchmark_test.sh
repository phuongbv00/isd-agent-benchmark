#!/bin/bash
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

# Test: 1 scenario, all 7 agents
AGENTS="baseline,react-isd,alignmentgraph-isd"
RATE_LIMIT="turbo"

python run_benchmark.py \
  --scenario scenarios/test/scenario_bal_large2_0075.json \
  --agents "$AGENTS" \
  --rate-limit "$RATE_LIMIT" \
  --verbose
