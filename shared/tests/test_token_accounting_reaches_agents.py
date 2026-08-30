"""Every agent's LLM calls must land in the counter run_benchmark reads.

``_run_agent_task`` reads ``resolved_llm_config._token_counter`` AFTER the
runner returns and writes that snapshot into the run's ``token_usage``. So the
contract is: whatever config an agent ends up building its models from must
still carry the counter ``_get_agent_runner`` opened.

The unit tests in ``test_llm_config.py`` pin the lineage rule itself. This file
pins it against the AGENTS — it constructs each one the way the benchmark does
and checks the config it settled on, because the bug being guarded here was not
in the rule but in six agents independently doing something the rule did not
survive: every baseline copies its config in ``__init__`` to pin a model or
temperature, and while ``copy_with()`` handed out fresh counters that silently
zeroed their ``token_usage`` from 2026-08-29. ``alignmentgraph-isd`` was the
only agent still reporting real numbers, so the failure flattered the agent
under study — the worst direction available.

No network: constructors only. Anything that cannot be built offline is skipped
rather than silently passing.
"""
from __future__ import annotations

import pytest

from shared.llm import LLMConfig


def _benchmark_config() -> LLMConfig:
    """The config ``_get_agent_runner`` hands an agent: max_tokens normalized to
    the uniform budget, then one fresh accounting scope for this run."""
    base = LLMConfig(
        model="test-model",
        provider="custom",
        base_url="http://localhost:9/v1",
        api_key="dummy",
        allow_dummy_api_key=True,
    )
    return base.copy_with(max_tokens=16384).new_token_scope()


def _build(agent_id: str, config: LLMConfig):
    """Construct one agent the way ``_get_agent_runner`` does, and return the
    LLMConfig it will actually build its chat models from."""
    if agent_id == "react-isd":
        from react_isd.agent import ReActISDAgent
        return ReActISDAgent(llm_config=config).llm_config
    if agent_id == "addie-agent":
        from addie_agent.agent import ADDIEAgent
        return ADDIEAgent(llm_config=config).llm_config
    if agent_id == "dick-carey-agent":
        from dick_carey_agent.agent import DickCareyAgent
        return DickCareyAgent(llm_config=config).llm_config
    if agent_id == "rpisd-agent":
        from rpisd_agent.agent import RPISDAgent
        return RPISDAgent(llm_config=config).llm_config
    if agent_id == "baseline":
        from baseline.generator import BaselineGenerator
        return BaselineGenerator(llm_config=config, max_tokens=16384).llm_config
    if agent_id == "eduplanner":
        from eduplanner.agents.base import AgentConfig
        return AgentConfig(
            model=config.model, provider=config.provider, llm_config=config,
        ).to_llm_config()
    if agent_id == "alignmentgraph-isd":
        # This one hands the package a factory rather than a config, so the
        # config it will build models from is whatever that closure captured.
        from alignmentgraph_isd_agent.agent import AlignmentGraphISDAgent
        factory = AlignmentGraphISDAgent(llm_config=config).config.llm_factory
        captured = [cell.cell_contents for cell in (factory.__closure__ or ())]
        return next(c for c in captured if isinstance(c, LLMConfig))
    raise AssertionError(f"unhandled agent id {agent_id}")


AGENTS = [
    "baseline", "eduplanner", "react-isd", "addie-agent",
    "dick-carey-agent", "rpisd-agent", "alignmentgraph-isd",
]


@pytest.mark.parametrize("agent_id", AGENTS)
def test_agent_reports_into_the_runs_token_counter(agent_id):
    config = _benchmark_config()
    try:
        used = _build(agent_id, config)
    except ImportError as exc:  # agent package not installed in this env
        pytest.skip(f"{agent_id} not importable: {exc}")
    except (AttributeError, IndexError, TypeError, StopIteration) as exc:
        pytest.fail(
            f"{agent_id}: could not reach the config it uses ({exc}). The probe, "
            "not the agent, is stale — fix _build rather than deleting the case."
        )

    assert used._token_counter is config._token_counter, (
        f"{agent_id} builds its models from a config detached from the run's "
        "token counter, so its token_usage will be 0 on every run. An agent may "
        "copy the config it is handed; copy_with() has to carry the counter."
    )
