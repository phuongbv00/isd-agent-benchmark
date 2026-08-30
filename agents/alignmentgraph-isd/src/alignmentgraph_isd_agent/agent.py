from __future__ import annotations
import os
from typing import Any

from alignmentgraph_isd import DesignBrief, MetaAgent
from alignmentgraph_isd.config import ChatModelFactory, HarnessRunConfig


#: In-flight LLM requests this agent allows itself, per benchmark rate preset.
#:
#: The benchmark's posture (``-r``) bounds how many AGENTS run at once, which is
#: all an orchestrator can bound. This agent's own topology fans out — the multi
#: arm runs content_designer and rubric_designer in the SAME superstep and each
#: issues one Developer call per work set, peaking at ~12 concurrent requests
#: from a single run — so it has to bound itself, and this table is where it
#: translates the benchmark's posture into its own limit. Owning the mapping here
#: keeps ``run_benchmark.py`` free of any agent's internal knobs: it publishes
#: ``BENCHMARK_RATE_LIMIT`` and nothing more.
#:
#: A shared endpoint answered the unbounded overflow with HTTP 429 on the two
#: multi arms only (9-25 per run over three runs, against 0-2 on the sequential
#: single arms), so the throttling sat on the ``agent_mode`` ablation axis.
LLM_CONCURRENCY_BY_RATE_LIMIT = {
    "conservative": 2,
    "moderate": 3,
    "aggressive": 4,
    "turbo": 6,
}
#: Used when the benchmark names no preset (direct package use, an older
#: launcher). Parallel enough to keep the fan-out meaningful, well inside a
#: typical per-user concurrency cap.
DEFAULT_LLM_MAX_CONCURRENCY = 4


def _llm_concurrency_from_env() -> int:
    """Requests this run may keep in flight: the explicit override if set,
    otherwise whatever the benchmark's rate preset implies."""
    override = os.getenv("HARNESS_LLM_MAX_CONCURRENCY")
    if override is not None:
        try:
            return max(0, int(override))
        except ValueError:
            pass
    preset = (os.getenv("BENCHMARK_RATE_LIMIT") or "").strip().lower()
    return LLM_CONCURRENCY_BY_RATE_LIMIT.get(preset, DEFAULT_LLM_MAX_CONCURRENCY)


def _regen_settings_from_env() -> dict[str, Any]:
    """Rate-limit posture for the package, sourced from the benchmark environment.

    The regen backoff defaults to the benchmark's own rate knob
    (``BENCHMARK_DELAY``, set by the ``-r`` preset) so per-agent re-runs respect
    the same concurrency posture as the outer scheduler; a 429 window then gets a
    chance to drain rather than being hammered by immediate retries.
    ``HARNESS_REGEN_BACKOFF`` overrides the delay, ``HARNESS_REGEN_BUDGET`` the
    number of extra attempts.

    The call-gate cap comes from ``BENCHMARK_RATE_LIMIT`` via
    :data:`LLM_CONCURRENCY_BY_RATE_LIMIT`; ``HARNESS_LLM_MAX_CONCURRENCY``
    overrides it (0 opts out). The outer scheduler only ever counted a whole
    agent as one unit, so without this the in-agent fan-out was outside the rate
    posture entirely. ``HARNESS_LLM_MIN_INTERVAL`` additionally spaces call
    starts, for an endpoint that meters rate rather than concurrency."""
    settings: dict[str, Any] = {
        "llm_max_concurrency": _llm_concurrency_from_env(),
    }
    interval = os.getenv("HARNESS_LLM_MIN_INTERVAL")
    if interval is not None:
        try:
            settings["llm_min_interval_seconds"] = max(0.0, float(interval))
        except ValueError:
            pass
    backoff = os.getenv("HARNESS_REGEN_BACKOFF") or os.getenv("BENCHMARK_DELAY")
    if backoff is not None:
        try:
            settings["regen_backoff_seconds"] = float(backoff)
        except ValueError:
            pass
    budget = os.getenv("HARNESS_REGEN_BUDGET")
    if budget is not None:
        try:
            settings["regen_budget"] = int(budget)
        except ValueError:
            pass
    return settings


def _llm_factory_from_benchmark_config(llm_config: Any) -> ChatModelFactory | None:
    if llm_config is None:
        return None

    # Only supply a default when the caller set nothing; never override an
    # explicit budget. The benchmark normalizes max_tokens uniformly across all
    # agents (see run_benchmark._normalize_agent_max_tokens), so overriding here
    # would reintroduce the very asymmetry that normalization removes.
    if getattr(llm_config, "max_tokens", None) is None:
        llm_config = llm_config.copy_with(max_tokens=16384)

    def factory() -> Any:
        from shared.llm import create_chat_model

        return create_chat_model(llm_config)

    return factory


def _scenario_to_design_brief(scenario: dict | DesignBrief) -> DesignBrief:
    """Bridge the benchmark's "scenario" vocabulary to the package's design brief contract.

    The benchmark's ``scenario_id`` becomes the design brief ``id``. All other
    fields share the same names, so this is the one place the two vocabularies meet.
    """
    if isinstance(scenario, DesignBrief):
        return scenario
    data = dict(scenario)
    if "id" not in data and "scenario_id" in data:
        data["id"] = data["scenario_id"]
    return DesignBrief(**data)


class AlignmentGraphISDAgent:
    def __init__(
        self,
        llm_config=None,
        llm_factory: ChatModelFactory | None = None,
        *,
        agent_mode: str = "multi",
        context_mode: str = "graph",
        control_mode: str = "scripted",
    ) -> None:
        # The factory is the single source of truth for the model; the harness
        # reads model provenance for its metadata by introspecting it (no parallel
        # LLMConfig to keep in sync).
        #
        # The ablation axes are kwargs (not env vars) on purpose: the benchmark
        # registry pins them per agent_id (alignmentgraph-isd-single-prose etc.),
        # so an arm can never run with the wrong config because of a missing
        # export. Defaults = the full multi-agent, graph-context pipeline.
        self.config = HarnessRunConfig(
            llm_factory=llm_factory or _llm_factory_from_benchmark_config(llm_config),
            agent_mode=agent_mode,
            context_mode=context_mode,
            **_regen_settings_from_env(),
        )
        # EXPERIMENTAL: "agentic" routes run() to the package's tool-loop
        # control mode (alignmentgraph_isd.core.agentic) instead of the
        # scripted pipeline. Not a HarnessRunConfig field — it selects which
        # runner is called, not how the pipeline is configured.
        self.control_mode = control_mode

    def run(self, scenario: dict | DesignBrief) -> dict:
        # The result carries the full alignment-graph dump as its own keys
        # ("graph" / "graph_dot"); the benchmark runner writes them to
        # <agent_id>_graph.json / <agent_id>_graph.dot next to the output.
        if self.control_mode == "agentic":
            from alignmentgraph_isd.core.agentic import run_agentic_brief

            return run_agentic_brief(self.config, _scenario_to_design_brief(scenario))
        return MetaAgent(self.config).run(_scenario_to_design_brief(scenario))
