"""
RPISD Agent: StateGraph 기반 래피드 프로토타이핑 교수설계 에이전트

RPISD 모형의 순환적/반복적 프로세스를 LangGraph StateGraph로 구현합니다.
이중 루프 구조를 통해 프로토타입 품질을 점진적으로 개선합니다.

[START] → [착수] → [분석] → [설계] ←─────── [사용성평가] (내부 루프: 프로토타입)
                              │                    ↑
                              │                    │
                              ▼                    │
                           [개발] ─────────────────┘ (외부 루프: 개발)
                              │
                              ▼
                           [실행] → [평가] → [END]

33개 ADDIE 소항목 산출물 완전성 확보:
- Analysis: 7개 (learner_analysis, context_analysis, task_analysis 등)
- Design: 7개 (learning_objectives, instructional_strategy, assessment_plan 등)
- Development: 8개 (lesson_plan, modules, materials, quiz_items 등)
- Implementation: 6개 (delivery_method, guides, pilot_plan 등)
- Evaluation: 5개 (quiz_items, rubric, program_evaluation 등)
"""

from datetime import datetime
from typing import Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END
from shared.llm import LLMConfig, configure_default_llm, llm_config_from_legacy

from rpisd_agent.state import (
    RPISDState,
    ToolCall,
    PrototypeVersion,
    create_initial_state,
    record_prototype_version,
    map_to_addie_output,
)
from rpisd_agent.tools import (
    # 착수
    kickoff_meeting,
    # 분석
    analyze_gap,
    analyze_performance,
    analyze_learner_characteristics,
    analyze_initial_task,
    # 설계
    design_instruction,
    develop_prototype,
    analyze_task_detailed,
    # 사용성 평가
    evaluate_with_client,
    evaluate_with_expert,
    evaluate_with_learner,
    aggregate_feedback,
    # 개발
    develop_final_program,
    # 실행
    implement_program,
    # 평가
    create_quiz_items,
    create_rubric,
    create_program_evaluation,
)


class RPISDAgent:
    """
    RPISD 모형 기반 래피드 프로토타이핑 교수설계 에이전트

    LangGraph StateGraph를 사용하여 RPISD 6단계를 실행합니다.
    이중 루프를 통해 프로토타입 및 개발 품질을 반복 개선합니다.
    """

    def __init__(
        self,
        model: str = "solar-mini",
        temperature: float = 0.7,
        max_iterations: int = 2,  # 3→2: 래피드 프로토타이핑 최적화 (#78)
        quality_threshold: float = 0.75,  # 0.8→0.75: 현실적 임계값 (#78)
        debug: bool = False,
        llm_config: Optional[LLMConfig] = None,
    ):
        selected_model = model if llm_config is None or model != "solar-mini" else llm_config.model
        self.llm_config = (
            llm_config.copy_with(model=selected_model, temperature=temperature)
            if llm_config is not None
            else llm_config_from_legacy(
                provider="upstage",
                model=selected_model,
                temperature=temperature,
            )
        )
        configure_default_llm(self.llm_config)

        self.model = self.llm_config.model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.debug = debug

        # StateGraph 빌드
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """이중 루프 StateGraph 빌드"""
        workflow = StateGraph(RPISDState)

        # 노드 추가 (7단계: 6단계 + 평가)
        workflow.add_node("kickoff", self._kickoff_node)
        workflow.add_node("analysis", self._analysis_node)
        workflow.add_node("design", self._design_node)
        workflow.add_node("usability_evaluation", self._usability_evaluation_node)
        workflow.add_node("development", self._development_node)
        workflow.add_node("implementation", self._implementation_node)
        workflow.add_node("evaluation", self._evaluation_node)

        # 엣지 추가
        workflow.set_entry_point("kickoff")
        workflow.add_edge("kickoff", "analysis")
        workflow.add_edge("analysis", "design")

        # 설계 → 사용성평가 (프로토타입 루프 시작)
        workflow.add_edge("design", "usability_evaluation")

        # 사용성평가 후 조건부 분기 (이중 루프 핵심)
        workflow.add_conditional_edges(
            "usability_evaluation",
            self._determine_next_step,
            {
                "design": "design",              # 프로토타입 루프: 설계로 회귀
                "development": "development",    # 프로토타입 통과: 개발로 진행
                "usability_evaluation": "usability_evaluation",  # 개발 루프: 재평가
                "implementation": "implementation",  # 개발 통과: 실행으로
            }
        )

        # 개발 → 사용성평가 (개발 루프)
        workflow.add_edge("development", "usability_evaluation")

        # 실행 → 평가 → 종료
        workflow.add_edge("implementation", "evaluation")
        workflow.add_edge("evaluation", END)

        return workflow.compile()

    def _log(self, message: str):
        """디버그 로깅"""
        if self.debug:
            print(f"[RPISD] {message}")

    def _record_tool_call(
        self,
        state: RPISDState,
        tool_name: str,
        args: dict,
        result: str,
        start_time: datetime,
    ) -> ToolCall:
        """도구 호출 기록"""
        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return ToolCall(
            step=len(state.get("tool_calls", [])) + 1,
            tool=tool_name,
            args=args,
            result=result,
            timestamp=start_time.isoformat(),
            duration_ms=duration_ms,
            success=True,
        )

    def _determine_next_step(self, state: RPISDState) -> str:
        """
        사용성 평가 후 다음 단계 결정 (이중 루프 핵심 로직)

        Returns:
            "design": 프로토타입 루프 - 설계로 회귀
            "development": 프로토타입 통과 - 개발로 진행
            "usability_evaluation": 개발 루프 - 재평가 (development에서 edge로 처리됨)
            "implementation": 개발 통과 - 실행으로 진행

        최적화 (#78):
            - 동적 임계값: 반복 횟수에 따라 품질 기준 완화 (0.75 → 0.70 → 0.65)
            - 조기 종료: 첫 반복에서 0.7 이상이면 즉시 다음 단계 진행
        """
        loop_source = state.get("loop_source", "prototype")
        current_quality = state.get("current_quality", 0.0)
        base_threshold = state.get("quality_threshold", self.quality_threshold)
        prototype_iteration = state.get("prototype_iteration", 0)
        development_iteration = state.get("development_iteration", 0)
        max_iterations = state.get("max_iterations", self.max_iterations)

        # 동적 임계값 계산 (#78): 반복마다 0.05씩 완화
        if loop_source == "prototype":
            iteration = prototype_iteration
        else:
            iteration = development_iteration

        # 조기 종료 임계값 (#78): 첫 반복에서 0.65 이상이면 즉시 다음 단계로
        # (폴백 함수 평균 0.755를 고려한 공격적 조기 종료)
        early_exit_threshold = 0.65
        # 동적 임계값: 반복 횟수가 증가할수록 기준 완화
        quality_threshold = max(base_threshold - (iteration - 1) * 0.05, 0.60)

        self._log(f"분기 결정: source={loop_source}, quality={current_quality:.2f}, threshold={quality_threshold:.2f} (base={base_threshold})")
        self._log(f"  prototype_iter={prototype_iteration}, dev_iter={development_iteration}, max={max_iterations}")

        if loop_source == "prototype":
            # 프로토타입 루프
            # 조기 종료: 첫 반복에서 early_exit_threshold(0.7) 이상이면 즉시 개발로 (#78)
            if prototype_iteration == 1 and current_quality >= early_exit_threshold:
                self._log(f"조기 종료: 첫 반복 품질 {current_quality:.2f} >= {early_exit_threshold} → 개발로 진행")
                return "development"
            elif current_quality >= quality_threshold:
                self._log("프로토타입 품질 충족 → 개발로 진행")
                return "development"
            elif prototype_iteration >= max_iterations:
                self._log("프로토타입 최대 반복 도달 → 개발로 진행")
                return "development"
            else:
                self._log("프로토타입 품질 미달 → 설계로 회귀")
                return "design"
        else:
            # 개발 루프
            # 조기 종료: 첫 반복에서 early_exit_threshold(0.7) 이상이면 즉시 실행으로 (#78)
            if development_iteration == 1 and current_quality >= early_exit_threshold:
                self._log(f"조기 종료: 첫 반복 품질 {current_quality:.2f} >= {early_exit_threshold} → 실행으로 진행")
                return "implementation"
            elif current_quality >= quality_threshold:
                self._log("개발 품질 충족 → 실행으로 진행")
                return "implementation"
            elif development_iteration >= max_iterations:
                self._log("개발 최대 반복 도달 → 실행으로 진행")
                return "implementation"
            else:
                self._log("개발 품질 미달 → 재개발 필요 (implementation으로 진행)")
                # 개발 루프에서 품질 미달 시 실행으로 진행 (무한 루프 방지)
                return "implementation"

    # ========== 1단계: 프로젝트 착수 ==========
    def _kickoff_node(self, state: RPISDState) -> dict:
        """프로젝트 착수 회의"""
        self._log("1단계: 프로젝트 착수 회의 시작")
        scenario = state["scenario"]
        context = scenario.get("context", {})
        tool_calls = list(state.get("tool_calls", []))
        reasoning_steps = list(state.get("reasoning_steps", []))
        errors = list(state.get("errors", []))

        reasoning_steps.append("Step 1: 프로젝트 착수 회의 - 범위 정의, 역할 공식화")

        start_time = datetime.now()
        try:
            kickoff_result = kickoff_meeting.invoke({
                "project_title": scenario.get("title", "교육 프로그램"),
                "learning_goals": scenario.get("learning_goals", []),
                "target_audience": context.get("target_audience", "일반 학습자"),
                "duration": context.get("duration"),
                "stakeholders": scenario.get("stakeholders"),
                "constraints": scenario.get("constraints", {}).get("resources"),
            })
            tool_calls.append(self._record_tool_call(
                state, "kickoff_meeting",
                {"project_title": scenario.get("title", "")},
                f"착수 회의 완료: {len(kickoff_result.get('success_criteria', []))}개 성공 기준",
                start_time,
            ))
        except Exception as e:
            errors.append(f"kickoff_meeting 실패: {str(e)}")
            kickoff_result = {}

        self._log(f"1단계 완료: {kickoff_result.get('project_title', '')}")

        return {
            "kickoff_result": kickoff_result,
            "current_phase": "analysis",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 2단계: 분석 ==========
    def _analysis_node(self, state: RPISDState) -> dict:
        """빠른 분석 (Gap, Performance, Learner, Task) - 4개 분석 병렬 실행 (#78)"""
        self._log("2단계: 분석 시작 (병렬 실행)")
        scenario = state["scenario"]
        context = scenario.get("context", {})
        tool_calls = list(state.get("tool_calls", []))
        reasoning_steps = list(state.get("reasoning_steps", []))
        errors = list(state.get("errors", []))

        reasoning_steps.append("Step 2: 분석 - 차이, 수행, 학습자, 과제 분석 (병렬)")

        # 분석 4단계 병렬 실행 (#78 성능 최적화)
        parallel_start_time = datetime.now()
        gap_result = {}
        performance_result = {}
        learner_result = {}
        task_result = {}

        def invoke_gap():
            return analyze_gap.invoke({
                "learning_goals": scenario.get("learning_goals", []),
                "current_state": context.get("prior_knowledge"),
                "desired_state": scenario.get("expected_outcomes", [None])[0] if scenario.get("expected_outcomes") else None,
            })

        def invoke_performance():
            return analyze_performance.invoke({
                "learning_goals": scenario.get("learning_goals", []),
                "performance_issues": context.get("performance_issues"),
                "organizational_context": context.get("additional_context"),
            })

        def invoke_learner():
            return analyze_learner_characteristics.invoke({
                "target_audience": context.get("target_audience", "일반 학습자"),
                "prior_knowledge": context.get("prior_knowledge"),
                "learning_environment": context.get("learning_environment"),
                "additional_context": context.get("additional_context"),
            })

        def invoke_task():
            return analyze_initial_task.invoke({
                "learning_goals": scenario.get("learning_goals", []),
                "domain": scenario.get("domain"),
                "complexity_level": scenario.get("difficulty"),
            })

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(invoke_gap): "gap",
                executor.submit(invoke_performance): "performance",
                executor.submit(invoke_learner): "learner",
                executor.submit(invoke_task): "task",
            }
            for future in as_completed(futures):
                analysis_type = futures[future]
                try:
                    result = future.result()
                    if analysis_type == "gap":
                        gap_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "analyze_gap",
                            {"learning_goals": scenario.get("learning_goals", [])[:2]},
                            f"Gap 분석 완료: {len(gap_result.get('gaps', []))}개 갭",
                            parallel_start_time,
                        ))
                    elif analysis_type == "performance":
                        performance_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "analyze_performance",
                            {},
                            f"수행 분석 완료: training_solution={performance_result.get('is_training_solution', True)}",
                            parallel_start_time,
                        ))
                    elif analysis_type == "learner":
                        learner_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "analyze_learner_characteristics",
                            {"target_audience": context.get("target_audience", "")},
                            f"학습자 분석 완료: {len(learner_result.get('learning_preferences', []))}개 선호도",
                            parallel_start_time,
                        ))
                    elif analysis_type == "task":
                        task_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "analyze_initial_task",
                            {"learning_goals": scenario.get("learning_goals", [])[:2]},
                            f"과제 분석 완료: {len(task_result.get('main_topics', []))}개 주제",
                            parallel_start_time,
                        ))
                except Exception as e:
                    errors.append(f"analyze_{analysis_type} 실패: {str(e)}")

        self._log(f"2단계 완료: gap={len(gap_result.get('gaps', []))}, learner={len(learner_result.get('challenges', []))}")

        return {
            "analysis_result": {
                "gap_analysis": gap_result,
                "performance_analysis": performance_result,
                "learner_characteristics": learner_result,
                "initial_task": task_result,
            },
            "current_phase": "design",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 3단계: 설계 (프로토타입 루프) ==========
    def _design_node(self, state: RPISDState) -> dict:
        """설계 및 프로토타입 개발"""
        prototype_iteration = state.get("prototype_iteration", 0) + 1
        self._log(f"3단계: 설계 시작 (프로토타입 v{prototype_iteration})")

        scenario = state["scenario"]
        context = scenario.get("context", {})
        analysis = state.get("analysis_result", {})
        tool_calls = list(state.get("tool_calls", []))
        reasoning_steps = list(state.get("reasoning_steps", []))
        errors = list(state.get("errors", []))
        prototype_versions = list(state.get("prototype_versions", []))

        # 이전 피드백 확인
        previous_feedback = None
        focus_areas = None
        if prototype_iteration > 1:
            usability = state.get("usability_feedback", {})
            previous_feedback = [
                usability.get("client_feedback", {}),
                usability.get("expert_feedback", {}),
                usability.get("learner_feedback", {}),
            ]
            focus_areas = usability.get("improvement_areas", [])
            focus_areas = [area.get("area") if isinstance(area, dict) else area for area in focus_areas]

        reasoning_steps.append(f"Step 3: 설계 (v{prototype_iteration}) - 교수설계, 프로토타입 개발")

        # 교수설계 (첫 번째 반복 시)
        design_result = state.get("design_result", {})
        if prototype_iteration == 1:
            start_time = datetime.now()
            try:
                design_result = design_instruction.invoke({
                    "learning_goals": scenario.get("learning_goals", []),
                    "learner_characteristics": analysis.get("learner_characteristics", {}),
                    "duration": context.get("duration"),
                    "learning_environment": context.get("learning_environment"),
                })
                tool_calls.append(self._record_tool_call(
                    state, "design_instruction",
                    {"learning_goals": scenario.get("learning_goals", [])[:2]},
                    f"교수설계 완료: {len(design_result.get('objectives', []))}개 목표",
                    start_time,
                ))
            except Exception as e:
                errors.append(f"design_instruction 실패: {str(e)}")

        # 프로토타입 개발
        start_time = datetime.now()
        try:
            prototype_result = develop_prototype.invoke({
                "design_result": design_result,
                "prototype_version": prototype_iteration,
                "previous_feedback": previous_feedback,
                "focus_areas": focus_areas,
            })
            tool_calls.append(self._record_tool_call(
                state, "develop_prototype",
                {"version": prototype_iteration},
                f"프로토타입 v{prototype_iteration} 개발 완료: {len(prototype_result.get('modules', []))}개 모듈",
                start_time,
            ))

            # 프로토타입 버전 기록
            new_version = record_prototype_version(
                state,
                content=prototype_result,
                feedback=[],
                quality_score=0.0,
            )
            prototype_versions.append(new_version)

        except Exception as e:
            errors.append(f"develop_prototype 실패: {str(e)}")
            prototype_result = {}

        # 상세 과제 분석 (첫 번째 반복 후)
        if prototype_iteration == 1:
            start_time = datetime.now()
            try:
                task_detailed = analyze_task_detailed.invoke({
                    "prototype": prototype_result,
                    "initial_task_analysis": analysis.get("initial_task", {}),
                    "feedback": None,
                })
                tool_calls.append(self._record_tool_call(
                    state, "analyze_task_detailed",
                    {},
                    f"상세 과제 분석 완료: {len(task_detailed.get('refined_topics', []))}개 주제",
                    start_time,
                ))
            except Exception as e:
                errors.append(f"analyze_task_detailed 실패: {str(e)}")

        self._log(f"3단계 완료: v{prototype_iteration}, modules={len(prototype_result.get('modules', []))}")

        return {
            "design_result": design_result,
            "prototype_versions": prototype_versions,
            "prototype_iteration": prototype_iteration,
            "loop_source": "prototype",
            "current_phase": "usability_evaluation",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 4단계: 사용성 평가 ==========
    def _usability_evaluation_node(self, state: RPISDState) -> dict:
        """사용성 평가 (의뢰인, 전문가, 학습자)"""
        loop_source = state.get("loop_source", "prototype")
        iteration = state.get("prototype_iteration", 1) if loop_source == "prototype" else state.get("development_iteration", 1)
        self._log(f"4단계: 사용성 평가 ({loop_source} v{iteration})")

        scenario = state["scenario"]
        context = scenario.get("context", {})
        kickoff = state.get("kickoff_result", {})
        design = state.get("design_result", {})
        analysis = state.get("analysis_result", {})
        tool_calls = list(state.get("tool_calls", []))
        reasoning_steps = list(state.get("reasoning_steps", []))
        errors = list(state.get("errors", []))

        # 평가 대상 결정
        if loop_source == "prototype":
            versions = state.get("prototype_versions", [])
            eval_target = versions[-1].get("content", {}) if versions else {}
        else:
            eval_target = state.get("development_result", {})

        reasoning_steps.append(f"Step 4: 사용성 평가 ({loop_source}) - 의뢰인, 전문가, 학습자 (병렬)")

        # 사용성평가 3단계 병렬 실행 (#73 성능 최적화)
        parallel_start_time = datetime.now()
        client_result = {"overall_score": 0.7}
        expert_result = {"overall_score": 0.75}
        learner_result = {"overall_score": 0.7}

        def invoke_client():
            return evaluate_with_client.invoke({
                "prototype": eval_target,
                "project_scope": kickoff.get("scope", {}),
                "success_criteria": kickoff.get("success_criteria", []),
            })

        def invoke_expert():
            return evaluate_with_expert.invoke({
                "prototype": eval_target,
                "design_result": design,
                "domain": scenario.get("domain"),
            })

        def invoke_learner():
            return evaluate_with_learner.invoke({
                "prototype": eval_target,
                "learner_characteristics": analysis.get("learner_characteristics", {}),
                "sample_size": 5,
            })

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(invoke_client): "client",
                executor.submit(invoke_expert): "expert",
                executor.submit(invoke_learner): "learner",
            }
            for future in as_completed(futures):
                eval_type = futures[future]
                try:
                    result = future.result()
                    if eval_type == "client":
                        client_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "evaluate_with_client",
                            {},
                            f"의뢰인 평가 완료: {client_result.get('overall_score', 0):.2f}",
                            parallel_start_time,
                        ))
                    elif eval_type == "expert":
                        expert_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "evaluate_with_expert",
                            {},
                            f"전문가 평가 완료: {expert_result.get('overall_score', 0):.2f}",
                            parallel_start_time,
                        ))
                    elif eval_type == "learner":
                        learner_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "evaluate_with_learner",
                            {},
                            f"학습자 평가 완료: {learner_result.get('overall_score', 0):.2f}",
                            parallel_start_time,
                        ))
                except Exception as e:
                    errors.append(f"evaluate_with_{eval_type} 실패: {str(e)}")

        # 피드백 통합
        start_time = datetime.now()
        try:
            aggregated = aggregate_feedback.invoke({
                "client_feedback": client_result,
                "expert_feedback": expert_result,
                "learner_feedback": learner_result,
                "quality_threshold": state.get("quality_threshold", self.quality_threshold),
            })
            tool_calls.append(self._record_tool_call(
                state, "aggregate_feedback",
                {"quality_threshold": state.get("quality_threshold", self.quality_threshold)},
                f"피드백 통합 완료: {aggregated.get('aggregated_score', 0):.2f}, pass={aggregated.get('pass_threshold', False)}",
                start_time,
            ))
        except Exception as e:
            errors.append(f"aggregate_feedback 실패: {str(e)}")
            aggregated = {"aggregated_score": 0.72, "pass_threshold": False}

        current_quality = aggregated.get("aggregated_score", 0.0)

        # 프로토타입 버전에 피드백 및 점수 업데이트
        if loop_source == "prototype":
            versions = list(state.get("prototype_versions", []))
            if versions:
                versions[-1]["feedback"] = [client_result, expert_result, learner_result]
                versions[-1]["quality_score"] = current_quality

        self._log(f"4단계 완료: quality={current_quality:.2f}, threshold={state.get('quality_threshold', self.quality_threshold)}")

        return {
            "usability_feedback": {
                "client_feedback": client_result,
                "expert_feedback": expert_result,
                "learner_feedback": learner_result,
                "aggregated_score": current_quality,
                "improvement_areas": aggregated.get("improvement_areas", []),
                "recommendations": aggregated.get("recommendations", []),
            },
            "current_quality": current_quality,
            "prototype_versions": state.get("prototype_versions", []),
            "current_phase": "check_quality",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 5단계: 개발 ==========
    def _development_node(self, state: RPISDState) -> dict:
        """최종 프로그램 개발"""
        development_iteration = state.get("development_iteration", 0) + 1
        self._log(f"5단계: 개발 시작 (v{development_iteration})")

        scenario = state["scenario"]
        design = state.get("design_result", {})
        usability = state.get("usability_feedback", {})
        versions = state.get("prototype_versions", [])
        tool_calls = list(state.get("tool_calls", []))
        reasoning_steps = list(state.get("reasoning_steps", []))
        errors = list(state.get("errors", []))

        final_prototype = versions[-1].get("content", {}) if versions else {}

        reasoning_steps.append(f"Step 5: 개발 (v{development_iteration}) - 최종 프로그램 개발")

        start_time = datetime.now()
        try:
            development_result = develop_final_program.invoke({
                "final_prototype": final_prototype,
                "aggregated_feedback": usability,
                "design_result": design,
                "project_title": scenario.get("title"),
            })
            tool_calls.append(self._record_tool_call(
                state, "develop_final_program",
                {"project_title": scenario.get("title", "")},
                f"최종 개발 완료: {len(development_result.get('modules', []))}개 모듈, {len(development_result.get('materials', []))}개 자료",
                start_time,
            ))
        except Exception as e:
            errors.append(f"develop_final_program 실패: {str(e)}")
            development_result = {}

        self._log(f"5단계 완료: modules={len(development_result.get('modules', []))}")

        return {
            "development_result": development_result,
            "development_iteration": development_iteration,
            "loop_source": "development",
            "current_phase": "usability_evaluation",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 6단계: 실행 ==========
    def _implementation_node(self, state: RPISDState) -> dict:
        """프로그램 실행 및 유지관리"""
        self._log("6단계: 실행 시작")

        scenario = state["scenario"]
        context = scenario.get("context", {})
        development = state.get("development_result", {})
        tool_calls = list(state.get("tool_calls", []))
        reasoning_steps = list(state.get("reasoning_steps", []))
        errors = list(state.get("errors", []))

        reasoning_steps.append("Step 6: 실행 - 프로그램 실행 및 유지관리 계획")

        start_time = datetime.now()
        try:
            implementation_result = implement_program.invoke({
                "development_result": development,
                "learning_environment": context.get("learning_environment"),
                "target_audience": context.get("target_audience"),
                "project_title": scenario.get("title"),
            })
            tool_calls.append(self._record_tool_call(
                state, "implement_program",
                {"project_title": scenario.get("title", "")},
                f"실행 계획 완료: {implementation_result.get('delivery_method', '')}",
                start_time,
            ))
        except Exception as e:
            errors.append(f"implement_program 실패: {str(e)}")
            implementation_result = {}

        self._log(f"6단계 완료: {implementation_result.get('delivery_method', '')}")

        return {
            "implementation_result": implementation_result,
            "current_phase": "evaluation",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 7단계: 평가 ==========
    def _evaluation_node(self, state: RPISDState) -> dict:
        """평가 문항, 루브릭, 프로그램 평가 생성"""
        self._log("7단계: 평가 시작")

        scenario = state["scenario"]
        context = scenario.get("context", {})
        design = state.get("design_result", {})
        analysis = state.get("analysis_result", {})
        prototype_versions = state.get("prototype_versions", [])
        tool_calls = list(state.get("tool_calls", []))
        reasoning_steps = list(state.get("reasoning_steps", []))
        errors = list(state.get("errors", []))

        reasoning_steps.append("Step 7: 평가 - 퀴즈 문항, 루브릭, 프로그램 평가 계획 수립")

        objectives = design.get("objectives", [])
        main_topics = analysis.get("initial_task", {}).get("main_topics", [])

        # 프로토타입 이력 정리
        prototype_history = [
            {
                "version": p.get("version"),
                "quality_score": p.get("quality_score"),
                "timestamp": p.get("timestamp", ""),
            }
            for p in prototype_versions
        ]

        # 1. 퀴즈 문항 생성
        start_time = datetime.now()
        try:
            quiz_items = create_quiz_items.invoke({
                "objectives": objectives,
                "main_topics": main_topics,
                "difficulty": scenario.get("difficulty"),
                "num_items": 10,
            })
            tool_calls.append(self._record_tool_call(
                state, "create_quiz_items",
                {"num_items": 10},
                f"퀴즈 문항 생성 완료: {len(quiz_items)}개 문항",
                start_time,
            ))
        except Exception as e:
            errors.append(f"create_quiz_items 실패: {str(e)}")
            quiz_items = []

        # 2. 평가 루브릭 생성
        start_time = datetime.now()
        try:
            rubric = create_rubric.invoke({
                "objectives": objectives,
                "assessment_type": "종합 평가",
            })
            tool_calls.append(self._record_tool_call(
                state, "create_rubric",
                {"assessment_type": "종합 평가"},
                f"루브릭 생성 완료: {len(rubric.get('criteria', []))}개 기준",
                start_time,
            ))
        except Exception as e:
            errors.append(f"create_rubric 실패: {str(e)}")
            rubric = {}

        # 3. 프로그램 평가 계획 생성 (Kirkpatrick 4단계)
        start_time = datetime.now()
        try:
            program_evaluation = create_program_evaluation.invoke({
                "program_title": scenario.get("title", "교육 프로그램"),
                "objectives": objectives,
                "target_audience": context.get("target_audience"),
                "prototype_history": prototype_history,
            })
            tool_calls.append(self._record_tool_call(
                state, "create_program_evaluation",
                {"program_title": scenario.get("title", "")},
                f"프로그램 평가 계획 완료: {program_evaluation.get('evaluation_model', '')}",
                start_time,
            ))
        except Exception as e:
            errors.append(f"create_program_evaluation 실패: {str(e)}")
            program_evaluation = {}

        # 채택 결정 및 개선 계획 추출
        adoption_decision = program_evaluation.get("adoption_decision", {})
        improvement_plan = program_evaluation.get("improvement_plan", {})

        self._log(f"7단계 완료: quiz={len(quiz_items)}개, rubric={len(rubric.get('criteria', []))}개 기준")

        return {
            "evaluation_result": {
                "quiz_items": quiz_items,
                "rubric": rubric,
                "program_evaluation": program_evaluation,
                "usability_summary": state.get("usability_feedback", {}),
                "adoption_decision": adoption_decision,
                "improvement_plan": improvement_plan,
            },
            "current_phase": "complete",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    def run(self, scenario: dict) -> dict:
        """
        시나리오를 입력받아 RPISD 산출물을 생성합니다.

        Args:
            scenario: 시나리오 딕셔너리

        Returns:
            dict: 완전한 결과 (ADDIE 호환 출력 + RPISD 출력 + trajectory + metadata)
        """
        start_time = datetime.now()

        # 초기 상태 생성
        initial_state = create_initial_state(scenario)
        initial_state["max_iterations"] = self.max_iterations
        initial_state["quality_threshold"] = self.quality_threshold

        # StateGraph 실행
        final_state = self.graph.invoke(initial_state)

        # 메타데이터 생성
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # ADDIE 호환 출력 조립
        addie_output = map_to_addie_output(final_state)

        result = {
            "scenario_id": scenario.get("scenario_id", "unknown"),
            "agent_id": "rpisd-agent",
            "timestamp": end_time.isoformat(),
            "addie_output": addie_output,
            "rpisd_output": {
                "kickoff": final_state.get("kickoff_result", {}),
                "analysis": final_state.get("analysis_result", {}),
                "design": final_state.get("design_result", {}),
                "prototype_versions": final_state.get("prototype_versions", []),
                "usability_feedback": final_state.get("usability_feedback", {}),
                "development": final_state.get("development_result", {}),
                "implementation": final_state.get("implementation_result", {}),
                "evaluation": final_state.get("evaluation_result", {}),
            },
            "trajectory": {
                "tool_calls": final_state.get("tool_calls", []),
                "reasoning_steps": final_state.get("reasoning_steps", []),
            },
            "metadata": {
                "model": self.model,
                "total_tokens": 0,
                "execution_time_seconds": execution_time,
                "agent_version": "0.1.0",
                "tool_calls_count": len(final_state.get("tool_calls", [])),
                "prototype_iterations": final_state.get("prototype_iteration", 0),
                "development_iterations": final_state.get("development_iteration", 0),
                "final_quality_score": final_state.get("current_quality", 0.0),
                "quality_threshold": self.quality_threshold,
                "max_iterations": self.max_iterations,
                "errors": final_state.get("errors", []),
            },
        }

        self._log(f"RPISD 완료: {execution_time:.2f}초, {len(final_state.get('tool_calls', []))}개 도구 호출")
        self._log(f"  프로토타입 반복: {final_state.get('prototype_iteration', 0)}회")
        self._log(f"  개발 반복: {final_state.get('development_iteration', 0)}회")
        self._log(f"  최종 품질: {final_state.get('current_quality', 0.0):.2f}")

        return result
