"""
ReAct-ISD Agent (v0.5.0 - 5 Phase Tools)

5개 ADDIE 단계별 도구를 사용하여 교수설계를 수행합니다.
각 도구가 표준 스키마 섹션을 직접 반환하므로 변환 로직이 없습니다.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

from shared.llm import (
    LLMConfig,
    configure_default_llm,
    create_chat_model,
    llm_config_from_legacy,
)

from react_isd.tools.phases import (
    run_analysis,
    run_design,
    run_development,
    run_implementation,
    run_evaluation,
)

UPSTAGE_DEFAULT_MODEL = "solar-mini"


class ReActISDAgent:
    """5개 ADDIE 단계 도구 기반 교수설계 에이전트"""

    def __init__(
        self,
        model: str = UPSTAGE_DEFAULT_MODEL,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        provider: str = "upstage",
        llm_config: Optional[LLMConfig] = None,
    ):
        selected_model = model if llm_config is None or model != UPSTAGE_DEFAULT_MODEL else llm_config.model
        self.llm_config = (
            llm_config.copy_with(model=selected_model, temperature=temperature)
            if llm_config is not None
            else llm_config_from_legacy(
                provider=provider,
                model=selected_model,
                api_key=api_key,
                temperature=temperature,
            )
        )

        configure_default_llm(self.llm_config)

        self.model_name = self.llm_config.model
        self.temperature = temperature
        self.provider = self.llm_config.provider
        self.llm = create_chat_model(self.llm_config)

    def run(self, scenario: dict) -> dict:
        """
        시나리오를 입력받아 ADDIE 산출물을 생성합니다.

        5개 ADDIE 단계 도구를 호출하고 결과를 병합합니다.
        - Analysis → Design → Development/Implementation/Evaluation (병렬)
        """
        start_time = datetime.now()
        tool_calls = []

        # 시나리오 정보 추출
        context = scenario.get("context", {})
        title = scenario.get("title", "Training program")
        target_audience = context.get("target_audience", "general learners")
        learning_environment = context.get("learning_environment", "not specified")
        duration = context.get("duration", "not specified")
        prior_knowledge = context.get("prior_knowledge")
        learning_goals = scenario.get("learning_goals", [])
        class_size = self._parse_class_size(context.get("class_size"))

        # ========================================
        # Phase 1: Analysis (순차 - 다른 단계의 기반)
        # ========================================
        analysis_result = run_analysis.invoke({
            "title": title,
            "target_audience": target_audience,
            "learning_environment": learning_environment,
            "duration": duration,
            "prior_knowledge": prior_knowledge,
            "learning_goals": learning_goals,
        })
        tool_calls.append({
            "step": 1,
            "tool": "run_analysis",
            "timestamp": datetime.now().isoformat(),
        })

        # ========================================
        # Phase 2: Design (순차 - Development 의존)
        # ========================================
        design_result = run_design.invoke({
            "title": title,
            "target_audience": target_audience,
            "duration": duration,
            "learning_goals": learning_goals,
        })
        tool_calls.append({
            "step": 2,
            "tool": "run_design",
            "timestamp": datetime.now().isoformat(),
        })

        # ========================================
        # Phase 3-5: Development, Implementation, Evaluation (병렬)
        # ========================================
        parallel_tools = {
            "run_development": (run_development, {
                "title": title,
                "target_audience": target_audience,
                "learning_environment": learning_environment,
                "learning_goals": learning_goals,
            }),
            "run_implementation": (run_implementation, {
                "title": title,
                "target_audience": target_audience,
                "learning_environment": learning_environment,
                "class_size": class_size,
            }),
            "run_evaluation": (run_evaluation, {
                "title": title,
                "target_audience": target_audience,
                "learning_goals": learning_goals,
            }),
        }

        parallel_results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(tool.invoke, args): name
                for name, (tool, args) in parallel_tools.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    parallel_results[name] = future.result()
                except Exception as e:
                    print(f"[WARN] {name} failed: {e}")
                    parallel_results[name] = {}

        # 병렬 도구 호출 기록
        parallel_timestamp = datetime.now().isoformat()
        for i, name in enumerate(["run_development", "run_implementation", "run_evaluation"]):
            tool_calls.append({
                "step": 3 + i,
                "tool": name,
                "timestamp": parallel_timestamp,
            })

        # ========================================
        # 결과 병합 (단순 딕셔너리 병합)
        # ========================================
        addie_output = {
            "analysis": analysis_result,
            "design": design_result,
            "development": parallel_results.get("run_development", {}),
            "implementation": parallel_results.get("run_implementation", {}),
            "evaluation": parallel_results.get("run_evaluation", {}),
        }

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        return {
            "scenario_id": scenario.get("scenario_id", "unknown"),
            "agent_id": "react-isd",
            "timestamp": end_time.isoformat(),
            "addie_output": addie_output,
            "trajectory": {
                "tool_calls": tool_calls,
                "reasoning_steps": [f"Phase {tc['step']}: {tc['tool']}" for tc in tool_calls],
                "agent_interactions": [{
                    "iteration": 1,
                    "agent": "react-isd",
                    "action": "execute",
                    "timestamp": start_time.isoformat(),
                }],
            },
            "metadata": {
                "model": self.model_name,
                "total_tokens": 0,
                "execution_time_seconds": execution_time,
                "cost_usd": 0.0,
                "agent_version": "0.5.0",
                "iterations": 1,
                "tool_calls_count": len(tool_calls),
            },
        }

    def _parse_class_size(self, class_size_raw) -> Optional[int]:
        """class_size 파싱"""
        if class_size_raw is None:
            return None
        if isinstance(class_size_raw, int):
            return class_size_raw
        if isinstance(class_size_raw, str):
            nums = re.findall(r'\d+', class_size_raw)
            return int(nums[0]) if nums else None
        return None
