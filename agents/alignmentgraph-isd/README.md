# AlignmentGraph-ISD Agent (benchmark adapter)

Thin benchmark adapter for the **`alignmentgraph-isd`** package — the thesis
contribution, tracked at the top level of the parent monorepo (its own README
documents the package API, alignment-graph dump, and RAG add-on). This adapter
contains **no instructional-design logic**; everything lives in the package.

## What the adapter does

`src/alignmentgraph_isd_agent/agent.py`, ~90 lines total:

1. **Vocabulary bridge** — `_scenario_to_design_brief()` maps the benchmark's
   *scenario* schema to the package's canonical `DesignBrief` contract
   (`scenario_id` → `id`, etc.). This is the only translation point between
   the two vocabularies.
2. **LLM injection** — `_llm_factory_from_benchmark_config()` wraps the
   benchmark's `LLMConfig` into a `ChatModelFactory` via
   `shared.llm.create_chat_model`, so the package never reads provider env
   vars itself. A default `max_tokens=16384` is supplied only when the caller
   set none — the benchmark's uniform max-tokens normalization (and the
   `AGENT_MODEL_MAX_TOKENS_CAP` clamp) stays authoritative.
3. **Rate-limit posture** — the package's empty-section regen loop inherits
   the benchmark's `BENCHMARK_DELAY` as its backoff (overridable via
   `HARNESS_REGEN_BACKOFF`; attempt budget via `HARNESS_REGEN_BUDGET`), so
   per-agent retries respect the same concurrency posture as the outer
   scheduler.
4. **Run** — `AlignmentGraphISDAgent.run(scenario)` calls
   `MetaAgent(config).run(brief)` and returns the standard benchmark result
   dict: `{"addie_output", "trajectory", "metadata"}` plus the agent-specific
   `"graph"` / `"graph_dot"` alignment-graph dump, which the benchmark runner
   saves as `<agent_id>_graph.json` / `<agent_id>_graph.dot` per scenario.

## Usage

Not in the benchmark's default agent list — add explicitly:

```bash
python run_benchmark.py --dataset test_90 --agents baseline,alignmentgraph-isd ...
```

## Install

Installed by `scripts/1_setup.sh` / `run_benchmark.py --install` in editable
mode; depends only on the top-level `alignmentgraph-isd` package (plus
`shared/` at runtime for the LLM factory).
