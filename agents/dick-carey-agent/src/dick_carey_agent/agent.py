"""
Dick & Carey Agent: StateGraph 기반 체제적 교수설계 에이전트

Dick & Carey 모형의 10단계 프로세스를 LangGraph StateGraph로 구현합니다.
형성평가-수정 피드백 루프를 통한 반복적 개선을 수행합니다.

[START] → [Goal] → [InstructionalAnalysis] ─┬→ [LearnerContext] → [PerformanceObjectives]
                                            └→ [LearnerContext] ─┘
                                                                    ↓
[END] ← [SummativeEvaluation] ← [should_revise] ← [FormativeEvaluation] ← [InstructionalMaterials] ← [InstructionalStrategy] ← [AssessmentInstruments]
                                      ↓ (if revision needed)
                              [Revision] → [FormativeEvaluation]
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Literal, Optional

from langgraph.graph import StateGraph, END
from shared.llm import LLMConfig, configure_default_llm, llm_config_from_legacy

from dick_carey_agent.state import (
    DickCareyState,
    ScenarioInput,
    ToolCall,
    create_initial_state,
    map_to_addie_output,
)
from dick_carey_agent.tools import (
    # 1-3단계
    set_instructional_goal,
    analyze_instruction,
    analyze_entry_behaviors,
    analyze_context,
    # 4-5단계
    write_performance_objectives,
    develop_assessment_instruments,
    # 6-7단계
    develop_instructional_strategy,
    develop_instructional_materials,
    # 8-10단계
    conduct_formative_evaluation,
    revise_instruction,
    conduct_summative_evaluation,
)


class DickCareyAgent:
    """
    Dick & Carey 모형 기반 체제적 교수설계 에이전트

    LangGraph StateGraph를 사용하여 Dick & Carey 10단계를 실행합니다.
    형성평가-수정 피드백 루프를 통해 품질 기준 달성까지 반복합니다.
    """

    def __init__(
        self,
        model: str = "solar-mini",
        temperature: float = 0.7,
        max_iterations: int = 3,
        quality_threshold: float = 6.5,  # 최적화: 7.0 → 6.5 (#80)
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
        """StateGraph 빌드 (피드백 루프 포함)"""
        workflow = StateGraph(DickCareyState)

        # 노드 추가 (10단계 - 5-6단계 병렬화 #80)
        workflow.add_node("goal", self._goal_node)
        workflow.add_node("instructional_analysis", self._instructional_analysis_node)
        workflow.add_node("learner_context", self._learner_context_node)
        workflow.add_node("performance_objectives", self._performance_objectives_node)
        workflow.add_node("assessment_and_strategy", self._assessment_and_strategy_node)  # 5-6단계 병렬
        workflow.add_node("instructional_materials", self._instructional_materials_node)
        workflow.add_node("formative_evaluation", self._formative_evaluation_node)
        workflow.add_node("revision", self._revision_node)
        workflow.add_node("summative_evaluation", self._summative_evaluation_node)

        # 엣지 추가 (순차 + 피드백 루프)
        workflow.set_entry_point("goal")
        workflow.add_edge("goal", "instructional_analysis")
        workflow.add_edge("instructional_analysis", "learner_context")
        workflow.add_edge("learner_context", "performance_objectives")
        workflow.add_edge("performance_objectives", "assessment_and_strategy")  # 5-6단계 병렬 노드로
        workflow.add_edge("assessment_and_strategy", "instructional_materials")
        workflow.add_edge("instructional_materials", "formative_evaluation")

        # 조건부 분기: 형성평가 후 수정 여부 결정
        workflow.add_conditional_edges(
            "formative_evaluation",
            self._should_revise,
            {
                "revision": "revision",
                "summative_evaluation": "summative_evaluation",
            }
        )

        # 수정 후 다시 형성평가로
        workflow.add_edge("revision", "formative_evaluation")

        # 총괄평가 후 종료
        workflow.add_edge("summative_evaluation", END)

        return workflow.compile()

    def _log(self, message: str):
        """디버그 로깅"""
        if self.debug:
            print(f"[Dick&Carey] {message}")

    def _record_tool_call(
        self,
        state: DickCareyState,
        tool_name: str,
        args: dict,
        result: str,
        start_time: datetime,
    ) -> ToolCall:
        """도구 호출 기록"""
        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        tool_call = ToolCall(
            step=len(state.get("tool_calls", [])) + 1,
            tool=tool_name,
            args=args,
            result=result,
            timestamp=start_time.isoformat(),
            duration_ms=duration_ms,
            success=True,
        )

        return tool_call

    def _should_revise(self, state: DickCareyState) -> Literal["revision", "summative_evaluation"]:
        """형성평가 결과 기반 분기 결정 (최적화 #80)"""
        formative = state.get("formative_evaluation", {})
        quality_score = formative.get("quality_score", 0)
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", self.max_iterations)
        quality_threshold = state.get("quality_threshold", self.quality_threshold)
        score_history = state.get("quality_score_history", [])

        self._log(f"Quality score: {quality_score}, iteration: {iteration_count}/{max_iterations}, threshold: {quality_threshold}")

        # 탈출 조건 (#73 + #80 성능 최적화)
        # 1. 품질 기준 충족
        if quality_score >= quality_threshold:
            self._log("Quality threshold met → proceeding to summative evaluation")
            return "summative_evaluation"
        # 2. 최대 반복 도달
        elif iteration_count >= max_iterations:
            self._log("Max iterations reached → proceeding to summative evaluation")
            return "summative_evaluation"
        # 3. 준수한 점수 달성 시 조기 종료 (#80 성능 최적화)
        elif quality_score >= 6.0:
            self._log(f"Acceptable score reached ({quality_score:.2f} >= 6.0) → proceeding to summative evaluation")
            return "summative_evaluation"
        # 4. 점수 개선 없음 (#73)
        elif len(score_history) >= 2 and quality_score <= score_history[-2]:
            self._log(f"No score improvement ({score_history[-2]:.2f} → {quality_score:.2f}) → proceeding to summative evaluation")
            return "summative_evaluation"
        else:
            self._log("Quality below threshold → proceeding to revision")
            return "revision"

    # ========== 1단계: 교수목적 설정 ==========
    def _goal_node(self, state: DickCareyState) -> dict:
        """1단계: 교수목적 설정"""
        self._log("Step 1: Identify instructional goal - start")
        scenario = state["scenario"]
        context = scenario.get("context", {})
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])

        reasoning_steps.append("Step 1: Identify instructional goal - define what learners can perform after instruction")

        start_time = datetime.now()
        try:
            goal_result = set_instructional_goal.invoke({
                "learning_goals": scenario.get("learning_goals", []),
                "target_audience": context.get("target_audience", "general learners"),
                "current_state": context.get("prior_knowledge"),
                "desired_state": None,
            })
            tool_calls.append(self._record_tool_call(
                state, "set_instructional_goal",
                {"learning_goals": scenario.get("learning_goals", [])},
                f"Instructional goal identified: {goal_result.get('goal_statement', '')[:50]}...",
                start_time,
            ))
        except Exception as e:
            errors.append(f"set_instructional_goal failed: {str(e)}")
            goal_result = {}

        self._log(f"Step 1 complete: {goal_result.get('goal_statement', '')[:50]}...")

        return {
            "goal": goal_result,
            "current_phase": "instructional_analysis",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 2단계: 교수분석 ==========
    def _instructional_analysis_node(self, state: DickCareyState) -> dict:
        """2단계: 교수분석"""
        self._log("Step 2: Instructional analysis - start")
        scenario = state["scenario"]
        goal = state.get("goal", {})
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])

        reasoning_steps.append("Step 2: Instructional analysis - analyze sub-skills and procedures")

        start_time = datetime.now()
        try:
            analysis_result = analyze_instruction.invoke({
                "instructional_goal": goal.get("goal_statement", ""),
                "domain": scenario.get("domain"),
                "learning_goals": scenario.get("learning_goals", []),
            })
            tool_calls.append(self._record_tool_call(
                state, "analyze_instruction",
                {"instructional_goal": goal.get("goal_statement", "")[:50]},
                f"Instructional analysis complete: {len(analysis_result.get('sub_skills', []))} sub-skills",
                start_time,
            ))
        except Exception as e:
            errors.append(f"analyze_instruction failed: {str(e)}")
            analysis_result = {}

        self._log(f"Step 2 complete: {len(analysis_result.get('sub_skills', []))} sub-skills")

        return {
            "instructional_analysis": analysis_result,
            "current_phase": "learner_context",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 3단계: 학습자/환경 분석 (병렬화 #80) ==========
    def _learner_context_node(self, state: DickCareyState) -> dict:
        """3단계: 학습자/환경 분석 (병렬 실행으로 최적화)"""
        self._log("Step 3: Learner/context analysis - start (parallel execution)")
        scenario = state["scenario"]
        context = scenario.get("context", {})
        analysis = state.get("instructional_analysis", {})
        tool_calls = list(state.get("tool_calls", []))
        reasoning_steps = list(state.get("reasoning_steps", []))
        errors = list(state.get("errors", []))

        reasoning_steps.append("Step 3: Learner/context analysis - entry behaviours, context analysis (parallel execution)")

        # class_size 파싱 (문자열인 경우 숫자 추출)
        import re
        raw_class_size = context.get("class_size")
        parsed_class_size = None
        if raw_class_size is not None:
            if isinstance(raw_class_size, int):
                parsed_class_size = raw_class_size
            elif isinstance(raw_class_size, str):
                numbers = re.findall(r'\d+', raw_class_size)
                if numbers:
                    if len(numbers) >= 2:
                        parsed_class_size = (int(numbers[0]) + int(numbers[1])) // 2
                    else:
                        parsed_class_size = int(numbers[0])

        # 병렬 실행을 위한 함수 정의
        def run_learner_analysis():
            start = datetime.now()
            result = analyze_entry_behaviors.invoke({
                "target_audience": context.get("target_audience", "general learners"),
                "prior_knowledge": context.get("prior_knowledge"),
                "entry_skills": analysis.get("entry_skills", []),
            })
            return ("learner", result, start)

        def run_context_analysis():
            start = datetime.now()
            result = analyze_context.invoke({
                "learning_environment": context.get("learning_environment", "Unspecified"),
                "duration": context.get("duration", "Unspecified"),
                "performance_context": context.get("additional_context"),
                "class_size": parsed_class_size,
                "resources": scenario.get("constraints", {}).get("resources"),
            })
            return ("context", result, start)

        learner_result = {}
        context_result = {}

        # ThreadPoolExecutor로 병렬 실행
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_learner_analysis),
                executor.submit(run_context_analysis),
            ]

            for future in as_completed(futures):
                try:
                    task_type, result, start_time = future.result()
                    if task_type == "learner":
                        learner_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "analyze_entry_behaviors",
                            {"target_audience": context.get("target_audience", "")},
                            f"Learner analysis complete: {len(result.get('entry_behaviors', []))} entry behaviours",
                            start_time,
                        ))
                    else:  # context
                        context_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "analyze_context",
                            {"learning_environment": context.get("learning_environment", "")},
                            f"Context analysis complete: {len(result.get('constraints', []))} constraints",
                            start_time,
                        ))
                except Exception as e:
                    error_msg = str(e)
                    if "learner" in error_msg.lower() or "entry" in error_msg.lower():
                        errors.append(f"analyze_entry_behaviors failed: {error_msg}")
                    else:
                        errors.append(f"analyze_context failed: {error_msg}")
                        # 환경 분석 폴백
                        from dick_carey_agent.tools.goal_analysis import _fallback_analyze_context
                        context_result = _fallback_analyze_context(
                            learning_environment=context.get("learning_environment", "Unspecified"),
                            duration=context.get("duration", "Unspecified"),
                            performance_context=context.get("additional_context"),
                            class_size=None,
                            resources=scenario.get("constraints", {}).get("resources"),
                        )

        self._log(f"Step 3 complete (parallel): learner={len(learner_result.get('entry_behaviors', []))}, context={len(context_result.get('constraints', []))}")

        return {
            "learner_context": {
                "learner": learner_result,
                "context": context_result,
            },
            "current_phase": "performance_objectives",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 4단계: 수행목표 진술 ==========
    def _performance_objectives_node(self, state: DickCareyState) -> dict:
        """4단계: 수행목표 진술"""
        self._log("Step 4: Write performance objectives - start")
        scenario = state["scenario"]
        context = scenario.get("context", {})
        goal = state.get("goal", {})
        analysis = state.get("instructional_analysis", {})
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])

        reasoning_steps.append("Step 4: Write performance objectives - author objectives in ABCD format")

        start_time = datetime.now()
        try:
            objectives_result = write_performance_objectives.invoke({
                "instructional_goal": goal.get("goal_statement", ""),
                "sub_skills": analysis.get("sub_skills", []),
                "target_audience": context.get("target_audience", "general learners"),
            })
            enabling_count = len(objectives_result.get("enabling_objectives", []))
            tool_calls.append(self._record_tool_call(
                state, "write_performance_objectives",
                {"instructional_goal": goal.get("goal_statement", "")[:50]},
                f"Performance objectives written: 1 terminal + {enabling_count} enabling objectives",
                start_time,
            ))
        except Exception as e:
            errors.append(f"write_performance_objectives failed: {str(e)}")
            objectives_result = {}

        self._log(f"Step 4 complete: {len(objectives_result.get('enabling_objectives', []))} objectives")

        return {
            "performance_objectives": objectives_result,
            "current_phase": "assessment_and_strategy",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 5-6단계: 평가도구 + 교수전략 개발 (병렬화 #80) ==========
    def _assessment_and_strategy_node(self, state: DickCareyState) -> dict:
        """5-6단계: 평가도구 개발 + 교수전략 개발 (병렬 실행)"""
        self._log("Steps 5-6: Develop assessment instruments/instructional strategy - start (parallel execution)")
        scenario = state["scenario"]
        context = scenario.get("context", {})
        objectives = state.get("performance_objectives", {})
        learner_context = state.get("learner_context", {})
        tool_calls = list(state.get("tool_calls", []))
        reasoning_steps = list(state.get("reasoning_steps", []))
        errors = list(state.get("errors", []))

        reasoning_steps.append("Step 5-6: Develop assessment instruments + develop instructional strategy (parallel execution)")

        # 병렬 실행을 위한 함수 정의
        def run_assessment():
            start = datetime.now()
            result = develop_assessment_instruments.invoke({
                "performance_objectives": objectives,
                "learning_environment": context.get("learning_environment", "Unspecified"),
                "duration": context.get("duration", "Unspecified"),
            })
            return ("assessment", result, start)

        def run_strategy():
            start = datetime.now()
            result = develop_instructional_strategy.invoke({
                "performance_objectives": objectives,
                "learner_analysis": learner_context.get("learner", {}),
                "learning_environment": context.get("learning_environment", "Unspecified"),
                "duration": context.get("duration", "Unspecified"),
            })
            return ("strategy", result, start)

        assessment_result = {}
        strategy_result = {}

        # ThreadPoolExecutor로 병렬 실행
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_assessment),
                executor.submit(run_strategy),
            ]

            for future in as_completed(futures):
                try:
                    task_type, result, start_time = future.result()
                    if task_type == "assessment":
                        assessment_result = result
                        post_count = len(result.get("post_test", []))
                        tool_calls.append(self._record_tool_call(
                            state, "develop_assessment_instruments",
                            {"objectives_count": len(objectives.get("enabling_objectives", []))},
                            f"Assessment instruments developed: {post_count} post-test items",
                            start_time,
                        ))
                    else:  # strategy
                        strategy_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "develop_instructional_strategy",
                            {"learning_environment": context.get("learning_environment", "")},
                            f"Instructional strategy developed: {result.get('delivery_method', '')}",
                            start_time,
                        ))
                except Exception as e:
                    error_msg = str(e)
                    if "assessment" in error_msg.lower():
                        errors.append(f"develop_assessment_instruments failed: {error_msg}")
                    else:
                        errors.append(f"develop_instructional_strategy failed: {error_msg}")

        self._log(f"Steps 5-6 complete (parallel): assessment={len(assessment_result.get('post_test', []))}, strategy={strategy_result.get('delivery_method', '')}")

        return {
            "assessment_instruments": assessment_result,
            "instructional_strategy": strategy_result,
            "current_phase": "instructional_materials",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 7단계: 교수자료 개발 ==========
    def _instructional_materials_node(self, state: DickCareyState) -> dict:
        """7단계: 교수자료 개발"""
        self._log("Step 7: Develop instructional materials - start")
        scenario = state["scenario"]
        context = scenario.get("context", {})
        objectives = state.get("performance_objectives", {})
        strategy = state.get("instructional_strategy", {})
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])

        reasoning_steps.append("Step 7: Develop instructional materials - instructor guide, learner materials, media")

        start_time = datetime.now()
        try:
            materials_result = develop_instructional_materials.invoke({
                "instructional_strategy": strategy,
                "performance_objectives": objectives,
                "learning_environment": context.get("learning_environment", "Unspecified"),
                "duration": context.get("duration", "Unspecified"),
                "topic_title": scenario.get("title", "Training program"),
            })
            learner_count = len(materials_result.get("learner_materials", []))
            slide_count = len(materials_result.get("slide_contents", []))
            tool_calls.append(self._record_tool_call(
                state, "develop_instructional_materials",
                {"topic_title": scenario.get("title", "")},
                f"Instructional materials developed: {learner_count} material types, {slide_count} slides",
                start_time,
            ))
        except Exception as e:
            errors.append(f"develop_instructional_materials failed: {str(e)}")
            materials_result = {}

        self._log(f"Step 7 complete: {len(materials_result.get('learner_materials', []))} material types")

        return {
            "instructional_materials": materials_result,
            "current_phase": "formative_evaluation",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 8단계: 형성평가 실시 ==========
    def _formative_evaluation_node(self, state: DickCareyState) -> dict:
        """8단계: 형성평가 실시"""
        iteration = state.get("iteration_count", 0) + 1
        self._log(f"Step 8: Conduct formative evaluation (round {iteration})")

        materials = state.get("instructional_materials", {})
        objectives = state.get("performance_objectives", {})
        assessment = state.get("assessment_instruments", {})
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])

        reasoning_steps.append(f"Step 8: Conduct formative evaluation (round {iteration}) - one-to-one, small group, field trial")

        start_time = datetime.now()
        try:
            formative_result = conduct_formative_evaluation.invoke({
                "instructional_materials": materials,
                "performance_objectives": objectives,
                "assessment_instruments": assessment,
                "iteration": iteration,
            })
            quality_score = formative_result.get("quality_score", 0)
            tool_calls.append(self._record_tool_call(
                state, "conduct_formative_evaluation",
                {"iteration": iteration},
                f"Formative evaluation complete: quality score {quality_score}",
                start_time,
            ))
        except Exception as e:
            errors.append(f"conduct_formative_evaluation failed: {str(e)}")
            formative_result = {"quality_score": 5.0}

        self._log(f"Step 8 complete: quality score {formative_result.get('quality_score', 0)}")

        # 점수 이력 업데이트 (#73 성능 최적화)
        score_history = list(state.get("quality_score_history", []))
        score_history.append(formative_result.get("quality_score", 0))

        return {
            "formative_evaluation": formative_result,
            "iteration_count": iteration,
            "quality_score_history": score_history,
            "current_phase": "revision_check",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 9단계: 교수프로그램 수정 ==========
    def _revision_node(self, state: DickCareyState) -> dict:
        """9단계: 교수프로그램 수정"""
        iteration = state.get("iteration_count", 1)
        self._log(f"Step 9: Revise instruction (round {iteration})")

        formative = state.get("formative_evaluation", {})
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])
        revision_log = state.get("revision_log", [])

        reasoning_steps.append(f"Step 9: Revise instruction (round {iteration}) - improvement based on formative evaluation")

        start_time = datetime.now()
        try:
            revision_result = revise_instruction.invoke({
                "formative_evaluation": formative,
                "current_state": {
                    "materials": state.get("instructional_materials", {}),
                    "strategy": state.get("instructional_strategy", {}),
                },
                "iteration": iteration,
            })
            revision_count = len(revision_result.get("revision_items", []))
            tool_calls.append(self._record_tool_call(
                state, "revise_instruction",
                {"iteration": iteration},
                f"Revision complete: {revision_count} items",
                start_time,
            ))
            # 수정 이력 추가
            revision_log.append(revision_result)
        except Exception as e:
            errors.append(f"revise_instruction failed: {str(e)}")
            revision_result = {"iteration": iteration, "revision_items": [], "summary": "Revision failed"}

        self._log(f"Step 9 complete: {len(revision_result.get('revision_items', []))} items revised")

        return {
            "revision_log": revision_log,
            "revision_triggered": True,
            "current_phase": "formative_evaluation",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    # ========== 10단계: 총괄평가 실시 ==========
    def _summative_evaluation_node(self, state: DickCareyState) -> dict:
        """10단계: 총괄평가 실시"""
        self._log("Step 10: Conduct summative evaluation")

        objectives = state.get("performance_objectives", {})
        iteration_count = state.get("iteration_count", 1)
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])

        reasoning_steps.append(f"Step 10: Conduct summative evaluation - final effectiveness evaluation ({iteration_count} formative evaluation rounds in total)")

        start_time = datetime.now()
        try:
            summative_result = conduct_summative_evaluation.invoke({
                "final_state": {
                    "goal": state.get("goal", {}),
                    "materials": state.get("instructional_materials", {}),
                    "strategy": state.get("instructional_strategy", {}),
                },
                "performance_objectives": objectives,
                "total_iterations": iteration_count,
            })
            tool_calls.append(self._record_tool_call(
                state, "conduct_summative_evaluation",
                {"total_iterations": iteration_count},
                f"Summative evaluation complete: effectiveness {summative_result.get('effectiveness_score', 0)}, decision: {summative_result.get('decision', '')}",
                start_time,
            ))
        except Exception as e:
            errors.append(f"conduct_summative_evaluation failed: {str(e)}")
            summative_result = {"effectiveness_score": 7.0, "decision": "conditional_adopt"}

        self._log(f"Step 10 complete: {summative_result.get('decision', '')}")

        return {
            "summative_evaluation": summative_result,
            "current_phase": "complete",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    def run(self, scenario: dict) -> dict:
        """
        시나리오를 입력받아 Dick & Carey 산출물을 생성합니다.

        Args:
            scenario: 시나리오 딕셔너리

        Returns:
            dict: 완전한 결과 (ADDIE 호환 출력 + trajectory + metadata)
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
            "agent_id": "dick-carey-agent",
            "timestamp": end_time.isoformat(),
            "addie_output": addie_output,
            "dick_carey_output": {
                "goal": final_state.get("goal", {}),
                "instructional_analysis": final_state.get("instructional_analysis", {}),
                "learner_context": final_state.get("learner_context", {}),
                "performance_objectives": final_state.get("performance_objectives", {}),
                "assessment_instruments": final_state.get("assessment_instruments", {}),
                "instructional_strategy": final_state.get("instructional_strategy", {}),
                "instructional_materials": final_state.get("instructional_materials", {}),
                "formative_evaluation": final_state.get("formative_evaluation", {}),
                "revision_log": final_state.get("revision_log", []),
                "summative_evaluation": final_state.get("summative_evaluation", {}),
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
                "iteration_count": final_state.get("iteration_count", 0),
                "quality_threshold": self.quality_threshold,
                "max_iterations": self.max_iterations,
                "final_quality_score": final_state.get("formative_evaluation", {}).get("quality_score", 0),
                "final_decision": final_state.get("summative_evaluation", {}).get("decision", ""),
                "errors": final_state.get("errors", []),
            },
        }

        self._log(f"Dick & Carey complete: {execution_time:.2f}s, {len(final_state.get('tool_calls', []))} tool calls, {final_state.get('iteration_count', 0)} iterations")

        return result
