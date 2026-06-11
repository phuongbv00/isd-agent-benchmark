from __future__ import annotations
import os
from alignmentgraph_isd.config import HarnessRunConfig, AblationConfig
from alignmentgraph_isd.orchestrator import MetaAgent


class AlignmentGraphISDAgent:
    def __init__(self, llm_config=None, ablation_key: str | None = None) -> None:
        key = ablation_key or os.environ.get("HARNESS_ABLATION", "full")
        self.config = HarnessRunConfig(
            ablation=AblationConfig.from_key(key),
            bench_llm_config=llm_config,
        )

    def run(self, scenario: dict) -> dict:
        return MetaAgent(self.config).run(scenario)
