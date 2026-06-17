from __future__ import annotations
import os
from typing import Any

from alignmentgraph_isd.config import AblationConfig, ChatModelFactory, HarnessRunConfig
from alignmentgraph_isd.orchestrator import MetaAgent
from alignmentgraph_isd.schema import DesignBrief


def _llm_factory_from_benchmark_config(llm_config: Any) -> ChatModelFactory | None:
    if llm_config is None:
        return None

    if getattr(llm_config, "max_tokens", None) and llm_config.max_tokens < 8192:
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
        ablation_key: str | None = None,
        llm_factory: ChatModelFactory | None = None,
    ) -> None:
        key = ablation_key or os.environ.get("HARNESS_ABLATION", "full")
        self.config = HarnessRunConfig(
            ablation=AblationConfig.from_key(key),
            llm_factory=llm_factory or _llm_factory_from_benchmark_config(llm_config),
        )

    def run(self, scenario: dict | DesignBrief) -> dict:
        return MetaAgent(self.config).run(_scenario_to_design_brief(scenario))
