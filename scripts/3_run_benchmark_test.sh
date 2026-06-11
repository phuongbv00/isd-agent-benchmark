#!/bin/bash
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

# Test: 1 scenario, all 7 agents
AGENTS="baseline,eduplanner,react-isd,addie-agent,dick-carey-agent,rpisd-agent,alignmentgraph-isd"
RATE_LIMIT="conservative"

python run_benchmark.py \
  --scenario scenarios/test/scenario_bal_large2_0075.json \
  --agents "$AGENTS" \
  --rate-limit "$RATE_LIMIT" \
  --verbose
