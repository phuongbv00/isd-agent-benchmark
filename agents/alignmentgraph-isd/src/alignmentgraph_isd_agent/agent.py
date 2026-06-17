from __future__ import annotations
import os
from typing import Any

from alignmentgraph_isd.config import AblationConfig, ChatModelFactory, HarnessRunConfig
from alignmentgraph_isd.orchestrator import MetaAgent


def _llm_factory_from_benchmark_config(llm_config: Any) -> ChatModelFactory | None:
    if llm_config is None:
        return None

    if getattr(llm_config, "max_tokens", None) and llm_config.max_tokens < 8192:
        llm_config = llm_config.copy_with(max_tokens=16384)

    def factory() -> Any:
        from shared.llm import create_chat_model

        return create_chat_model(llm_config)

    return factory


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

    def run(self, scenario: dict) -> dict:
        return MetaAgent(self.config).run(scenario)
