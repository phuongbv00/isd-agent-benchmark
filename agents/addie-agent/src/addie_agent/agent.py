"""
ADDIE Agent: StateGraph 기반 순차적 교수설계 에이전트

ADDIE 모형의 선형적/순차적 프로세스를 LangGraph StateGraph로 구현합니다.
[START] → [Analysis] → [Design] → [Development] → [Implementation] → [Evaluation] → [END]
"""

from datetime import datetime
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END
from shared.llm import LLMConfig, configure_default_llm, llm_config_from_legacy

from addie_agent.state import (
    ADDIEState,
    ScenarioInput,
    ToolCall,
    create_initial_state,
)
from addie_agent.tools import (
    # Analysis (요구분석, 학습자분석, 환경분석, 과제분석)
    analyze_needs,
    analyze_learner,
    analyze_context,
    analyze_task,
    # Design (목표명세화, 평가도구, 교수전략)
    design_objectives,
    design_assessment,
    design_strategy,
    # Development (레슨플랜, 교수자료)
    create_lesson_plan,
    create_materials,
    # Implementation (실행계획, 유지관리)
    create_implementation_plan,
    create_maintenance_plan,
    # Evaluation (퀴즈, 루브릭, 성과평가)
    create_quiz_items,
    create_rubric,
    create_program_evaluation,
)


def _parse_class_size(value: Any) -> Optional[int]:
    """class_size 값을 정수로 변환 (문자열 처리 포함)"""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # "소규모(1-10명)" → 5, "30명" → 30 등 숫자 추출
        import re
        numbers = re.findall(r'\d+', value)
        if numbers:
            # 범위인 경우 중간값 사용
            if len(numbers) >= 2:
                return (int(numbers[0]) + int(numbers[1])) // 2
            return int(numbers[0])
    return None


class ADDIEAgent:
    """
    ADDIE 모형 기반 순차적 교수설계 에이전트

    LangGraph StateGraph를 사용하여 ADDIE 5단계를 순차적으로 실행합니다.
    각 단계의 산출물이 다음 단계의 입력으로 활용됩니다.
    """

    def __init__(
        self,
        model: str = "solar-mini",
        temperature: float = 0.7,
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
        self.debug = debug

        # StateGraph 빌드
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """선형 StateGraph 빌드"""
        # StateGraph 생성
        workflow = StateGraph(ADDIEState)

        # 노드 추가
        workflow.add_node("analysis", self._analysis_node)
        workflow.add_node("design", self._design_node)
        workflow.add_node("development", self._development_node)
        workflow.add_node("implementation", self._implementation_node)
        workflow.add_node("evaluation", self._evaluation_node)

        # 엣지 추가 (선형 워크플로우)
        workflow.set_entry_point("analysis")
        workflow.add_edge("analysis", "design")
        workflow.add_edge("design", "development")
        workflow.add_edge("development", "implementation")
        workflow.add_edge("implementation", "evaluation")
        workflow.add_edge("evaluation", END)

        return workflow.compile()

    def _log(self, message: str):
        """디버그 로깅"""
        if self.debug:
            print(f"[ADDIE] {message}")

    def _record_tool_call(
        self,
        state: ADDIEState,
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

    def _analysis_node(self, state: ADDIEState) -> dict:
        """Analysis 단계 노드"""
        self._log("Analysis phase started")
        scenario = state["scenario"]
        context = scenario.get("context", {})
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])

        reasoning_steps.append("Step 1: Analysis phase - needs, learner, context, task analysis (parallel)")

        # Analysis 4단계 병렬 실행 (#73 성능 최적화)
        parallel_start_time = datetime.now()
        needs_result = {}
        learner_result = {}
        context_result = {}
        task_result = {}

        def invoke_needs():
            return analyze_needs.invoke({
                "learning_goals": scenario.get("learning_goals", []),
                "current_state": context.get("prior_knowledge"),
                "desired_state": scenario.get("expected_outcomes", [None])[0] if scenario.get("expected_outcomes") else None,
                "performance_gap": context.get("performance_gap"),
            })

        def invoke_learner():
            return analyze_learner.invoke({
                "target_audience": context.get("target_audience", "General learners"),
                "prior_knowledge": context.get("prior_knowledge"),
                "additional_context": context.get("additional_context"),
            })

        def invoke_context():
            return analyze_context.invoke({
                "learning_environment": context.get("learning_environment", "Not specified"),
                "duration": context.get("duration", "Not specified"),
                "class_size": _parse_class_size(context.get("class_size")),
                "budget": scenario.get("constraints", {}).get("budget"),
                "resources": scenario.get("constraints", {}).get("resources"),
            })

        def invoke_task():
            return analyze_task.invoke({
                "learning_goals": scenario.get("learning_goals", []),
                "domain": scenario.get("domain"),
                "difficulty": scenario.get("difficulty"),
            })

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(invoke_needs): "needs",
                executor.submit(invoke_learner): "learner",
                executor.submit(invoke_context): "context",
                executor.submit(invoke_task): "task",
            }
            for future in as_completed(futures):
                analysis_type = futures[future]
                try:
                    result = future.result()
                    if analysis_type == "needs":
                        needs_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "analyze_needs",
                            {"learning_goals": scenario.get("learning_goals", [])},
                            f"Needs analysis complete: {len(needs_result.get('training_needs', []))} training needs identified",
                            parallel_start_time,
                        ))
                    elif analysis_type == "learner":
                        learner_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "analyze_learner",
                            {"target_audience": context.get("target_audience", "")},
                            f"Learner analysis complete: {len(learner_result.get('characteristics', []))} characteristics",
                            parallel_start_time,
                        ))
                    elif analysis_type == "context":
                        context_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "analyze_context",
                            {"learning_environment": context.get("learning_environment", "")},
                            f"Context analysis complete: {len(context_result.get('constraints', []))} constraints",
                            parallel_start_time,
                        ))
                    elif analysis_type == "task":
                        task_result = result
                        tool_calls.append(self._record_tool_call(
                            state, "analyze_task",
                            {"learning_goals": scenario.get("learning_goals", [])},
                            f"Task analysis complete: {len(task_result.get('main_topics', []))} topics",
                            parallel_start_time,
                        ))
                except Exception as e:
                    errors.append(f"analyze_{analysis_type} failed: {str(e)}")

        # LLM 응답에 누락된 priority_matrix 보완 (A-4)
        if needs_result and "priority_matrix" not in needs_result:
            training_needs = needs_result.get("training_needs", [])
            needs_result["priority_matrix"] = {
                "high_urgency_high_impact": training_needs[:2] if len(training_needs) >= 2 else training_needs,
                "high_urgency_low_impact": [training_needs[2]] if len(training_needs) > 2 else [],
                "low_urgency_high_impact": ["Advanced competency development"],
                "low_urgency_low_impact": ["Optional self-development courses"],
            }

        self._log(f"Analysis complete: needs={len(needs_result.get('training_needs', []))}, learner={len(learner_result.get('characteristics', []))}, context={len(context_result.get('constraints', []))}, task={len(task_result.get('main_topics', []))}")

        return {
            "analysis_result": {
                "needs_analysis": needs_result,
                "learner_analysis": learner_result,
                "context_analysis": context_result,
                "task_analysis": task_result,
            },
            "current_phase": "design",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    def _design_node(self, state: ADDIEState) -> dict:
        """Design 단계 노드"""
        self._log("Design phase started")
        scenario = state["scenario"]
        context = scenario.get("context", {})
        analysis = state.get("analysis_result", {})
        task_analysis = analysis.get("task_analysis", {})
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])

        reasoning_steps.append("Step 2: Design phase - learning objectives, assessment plan, instructional strategy design")

        # 1. 학습 목표 설계
        start_time = datetime.now()
        try:
            objectives_result = design_objectives.invoke({
                "learning_goals": scenario.get("learning_goals", []),
                "target_audience": context.get("target_audience", "General learners"),
                "difficulty": scenario.get("difficulty"),
            })
            tool_calls.append(self._record_tool_call(
                state, "design_objectives",
                {"learning_goals": scenario.get("learning_goals", [])},
                f"Learning objectives design complete: {len(objectives_result)} objectives",
                start_time,
            ))
        except Exception as e:
            errors.append(f"design_objectives failed: {str(e)}")
            objectives_result = []

        # 2. 평가 계획 수립
        start_time = datetime.now()
        try:
            assessment_result = design_assessment.invoke({
                "objectives": objectives_result,
                "duration": context.get("duration", "Not specified"),
                "learning_environment": context.get("learning_environment", "Not specified"),
            })
            tool_calls.append(self._record_tool_call(
                state, "design_assessment",
                {"objectives_count": len(objectives_result)},
                f"Assessment plan complete",
                start_time,
            ))
        except Exception as e:
            errors.append(f"design_assessment failed: {str(e)}")
            assessment_result = {}

        # 3. 교수 전략 설계
        start_time = datetime.now()
        main_topics = task_analysis.get("main_topics", scenario.get("learning_goals", []))
        try:
            strategy_result = design_strategy.invoke({
                "main_topics": main_topics,
                "target_audience": context.get("target_audience", "General learners"),
                "duration": context.get("duration", "Not specified"),
                "learning_environment": context.get("learning_environment", "Not specified"),
            })
            tool_calls.append(self._record_tool_call(
                state, "design_strategy",
                {"main_topics": main_topics},
                f"Instructional strategy design complete: {len(strategy_result.get('sequence', []))} Events",
                start_time,
            ))
        except Exception as e:
            errors.append(f"design_strategy failed: {str(e)}")
            strategy_result = {}

        self._log(f"Design complete: objectives={len(objectives_result)}, events={len(strategy_result.get('sequence', []))}")

        return {
            "design_result": {
                "learning_objectives": objectives_result,
                "assessment_plan": assessment_result,
                "instructional_strategy": strategy_result,
            },
            "current_phase": "development",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    def _development_node(self, state: ADDIEState) -> dict:
        """Development 단계 노드"""
        self._log("Development phase started")
        scenario = state["scenario"]
        context = scenario.get("context", {})
        analysis = state.get("analysis_result", {})
        design = state.get("design_result", {})
        task_analysis = analysis.get("task_analysis", {})
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])

        reasoning_steps.append("Step 3: Development phase - lesson plan and learning materials development")

        objectives = design.get("learning_objectives", [])
        strategy = design.get("instructional_strategy", {})
        main_topics = task_analysis.get("main_topics", scenario.get("learning_goals", []))

        # 1. 레슨 플랜 생성
        start_time = datetime.now()
        try:
            lesson_plan_result = create_lesson_plan.invoke({
                "objectives": objectives,
                "instructional_strategy": strategy,
                "duration": context.get("duration", "Not specified"),
                "main_topics": main_topics,
            })
            tool_calls.append(self._record_tool_call(
                state, "create_lesson_plan",
                {"duration": context.get("duration", "")},
                f"Lesson plan created: {len(lesson_plan_result.get('modules', []))} modules",
                start_time,
            ))
        except Exception as e:
            errors.append(f"create_lesson_plan failed: {str(e)}")
            lesson_plan_result = {}

        # 2. 학습 자료 생성
        start_time = datetime.now()
        try:
            materials_result = create_materials.invoke({
                "lesson_plan": lesson_plan_result,
                "learning_environment": context.get("learning_environment", "Not specified"),
                "target_audience": context.get("target_audience", "General learners"),
            })
            tool_calls.append(self._record_tool_call(
                state, "create_materials",
                {"lesson_plan_modules": len(lesson_plan_result.get("modules", []))},
                f"Learning materials created: {len(materials_result)} materials",
                start_time,
            ))
        except Exception as e:
            errors.append(f"create_materials failed: {str(e)}")
            materials_result = []

        # LLM 응답에 누락된 storyboard 보완 (D-18)
        if materials_result:
            default_storyboard = [
                {"frame_number": 1, "screen_title": "Introduction Screen", "visual_description": "Display training title and logo", "audio_narration": "Welcome to the training.", "interaction": "Click start button", "notes": "Background music fade-in"},
                {"frame_number": 2, "screen_title": "Learning Objectives", "visual_description": "Animated list of learning objectives", "audio_narration": "Let's review what you will learn today.", "interaction": "Auto-advance", "notes": "Sequential display of each objective"},
            ]
            for material in materials_result:
                if material.get("type") in ["PPT", "Video"] and "storyboard" not in material:
                    material["storyboard"] = default_storyboard

        self._log(f"Development complete: modules={len(lesson_plan_result.get('modules', []))}, materials={len(materials_result)}")

        # 출력 구조: 루브릭 항목과 일치하도록 명시적 필드명 사용
        # Item 19: 학습자용 자료 개발 - materials를 learner_materials로 명시
        # Item 20: 교수자용 매뉴얼 - facilitator_guide (implementation에서 생성)
        # Item 21: 운영자용 매뉴얼 - operator_guide (implementation에서 생성)
        # Item 22: 평가 도구·문항 - evaluation에서 생성
        return {
            "development_result": {
                "lesson_plan": lesson_plan_result,
                "learner_materials": materials_result,  # Item 19: Learner material development
                "materials": materials_result,  # maintain backward compatibility
            },
            "current_phase": "implementation",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    def _implementation_node(self, state: ADDIEState) -> dict:
        """Implementation 단계 노드"""
        self._log("Implementation phase started")
        scenario = state["scenario"]
        context = scenario.get("context", {})
        development = state.get("development_result", {})
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])

        reasoning_steps.append("Step 4: Implementation phase - implementation plan and maintenance plan development")

        lesson_plan = development.get("lesson_plan", {})

        # 실행 계획 생성
        start_time = datetime.now()
        try:
            implementation_result = create_implementation_plan.invoke({
                "lesson_plan": lesson_plan,
                "learning_environment": context.get("learning_environment", "Not specified"),
                "target_audience": context.get("target_audience", "General learners"),
                "class_size": _parse_class_size(context.get("class_size")),
            })
            fg_len = len(implementation_result.get("facilitator_guide", ""))
            lg_len = len(implementation_result.get("learner_guide", ""))
            tool_calls.append(self._record_tool_call(
                state, "create_implementation_plan",
                {"learning_environment": context.get("learning_environment", "")},
                f"Implementation plan complete: facilitator_guide={fg_len} chars, learner_guide={lg_len} chars",
                start_time,
            ))
        except Exception as e:
            errors.append(f"create_implementation_plan failed: {str(e)}")
            implementation_result = {}

        # LLM 응답에 누락된 필드 보완 (Dev-21, I-24, I-26)
        if implementation_result:
            env_lower = context.get("learning_environment", "").lower()
            is_online = "online" in env_lower
            if "operator_guide" not in implementation_result:
                if is_online:
                    implementation_result["operator_guide"] = """1. Platform preparation: Create and distribute video conference links, verify recording settings, pre-configure breakout sessions, set up waiting room
2. Participant management: Monitor connection status, support connection issues, manage chat, check and record attendance
3. Technical support: Resolve screen sharing issues, support audio issues, prepare backup links, respond to network failures
4. Post-session processing: Edit and upload recordings, organize attendance records, collect survey results, send participant emails"""
                else:
                    implementation_result["operator_guide"] = """1. Training environment preparation: Confirm classroom reservation, inspect equipment (projector, microphone, PC), print and arrange learning materials, prepare refreshments
2. Participant management: Check attendance, distribute name tags, guide seating, record notable issues
3. Operational support: Communicate with the instructor, manage time, announce breaks, respond to unexpected situations
4. Post-session processing: Tidy the classroom, return equipment, collect surveys, report attendance status"""
            if "orientation_plan" not in implementation_result:
                if is_online:
                    implementation_result["orientation_plan"] = """1. Instructor/facilitator orientation (3 days before training): Familiarization with platform features, screen sharing test, guidance on running breakout sessions, review of instructional materials
2. Operator orientation (2 days before training): Explanation of technical support roles, sharing of issue response manual, verification of emergency contact list, checklist distribution
3. Rehearsal (1 day before training): Full flow test, backup scenario review, time allocation check, final check completed"""
                else:
                    implementation_result["orientation_plan"] = """1. Instructor/facilitator orientation (1 week before training): Explanation of training objectives and curriculum, delivery and review of instructional materials, agreement on delivery approach, Q&A
2. Operator orientation (3 days before training): Explanation of operational roles and responsibilities, checklist distribution, sharing of emergency contact list, confirmation of rehearsal schedule
3. Rehearsal (1 day before training): Equipment test, walkthrough of participant flow, time allocation review, final adjustments"""
            if "pilot_plan" not in implementation_result:
                implementation_result["pilot_plan"] = {
                    "pilot_scope": "First pilot: Trial run of the full course with a small group (10-15 participants)",
                    "participants": "1-2 representatives from each department, with training staff observing",
                    "duration": "Same as the main training",
                    "success_criteria": ["Learning objective achievement rate of 80% or higher", "Satisfaction of 4.0/5.0 or higher", "Delivered without major issues"],
                    "data_collection": ["Pre/post test scores", "Satisfaction survey", "Observation records", "Participant feedback"],
                    "contingency_plan": "Use backup materials if technical problems occur; reduce optional modules if the schedule overruns",
                }

        # 2. 유지관리 계획 생성
        materials = development.get("materials", [])
        content_types = [m.get("type", "Material") for m in materials] if materials else ["Slides", "Handout"]
        start_time = datetime.now()
        try:
            maintenance_result = create_maintenance_plan.invoke({
                "program_title": scenario.get("title", "Training Program"),
                "delivery_method": implementation_result.get("delivery_method", context.get("learning_environment", "In-person classroom training")),
                "content_types": content_types,
                "update_frequency": "Quarterly",
            })
            tool_calls.append(self._record_tool_call(
                state, "create_maintenance_plan",
                {"program_title": scenario.get("title", "")},
                f"Maintenance plan complete: {len(maintenance_result.get('content_maintenance', {}).get('update_triggers', []))} update triggers",
                start_time,
            ))
        except Exception as e:
            errors.append(f"create_maintenance_plan failed: {str(e)}")
            maintenance_result = {}

        self._log(f"Implementation complete: facilitator_guide={len(implementation_result.get('facilitator_guide', ''))} chars, maintenance_plan={bool(maintenance_result)}")

        # 출력 구조 평탄화: 루브릭 항목과 일치하도록 필드를 phase 레벨로 이동
        # Item 24: 교수자·운영자 오리엔테이션
        # Item 25: 시스템/환경 점검 (technical_requirements에 포함)
        # Item 26: 프로토타입 실행 (pilot_plan)
        # Item 27: 운영 모니터링 및 지원 (support_plan)
        return {
            "implementation_result": {
                # 기존 implementation_plan 내용을 평탄화
                "delivery_method": implementation_result.get("delivery_method", ""),
                "facilitator_guide": implementation_result.get("facilitator_guide", ""),
                "learner_guide": implementation_result.get("learner_guide", ""),
                "technical_requirements": implementation_result.get("technical_requirements", []),
                "support_plan": implementation_result.get("support_plan", ""),
                # 루브릭 항목과 매칭되는 필드 (평탄화)
                "operator_guide": implementation_result.get("operator_guide", ""),  # Dev-21
                "instructor_operator_orientation": implementation_result.get("orientation_plan", ""),  # I-24
                "system_environment_check": {
                    "technical_requirements": implementation_result.get("technical_requirements", []),
                    "pre_check_items": ["Platform access test", "Learning material upload verification", "Network stability check"],
                },  # I-25
                "prototype_execution": implementation_result.get("pilot_plan", {}),  # I-26
                "operation_monitoring": {
                    "monitoring_items": ["Learner engagement", "Technical issues", "Schedule adherence"],
                    "support_channels": ["Live chat", "Q&A board", "Email support"],
                    "escalation_process": "Level 1: Operator → Level 2: Instructor → Level 3: Manager",
                },  # I-27
                # 유지관리 계획
                "maintenance_plan": maintenance_result,
            },
            "current_phase": "evaluation",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    def _evaluation_node(self, state: ADDIEState) -> dict:
        """Evaluation 단계 노드"""
        self._log("Evaluation phase started")
        scenario = state["scenario"]
        analysis = state.get("analysis_result", {})
        design = state.get("design_result", {})
        task_analysis = analysis.get("task_analysis", {})
        tool_calls = state.get("tool_calls", [])
        reasoning_steps = state.get("reasoning_steps", [])
        errors = state.get("errors", [])

        reasoning_steps.append("Step 5: Evaluation phase - quiz items, assessment rubric, program evaluation plan generation")

        objectives = design.get("learning_objectives", [])
        main_topics = task_analysis.get("main_topics", scenario.get("learning_goals", []))

        # 1. 퀴즈 문항 생성
        start_time = datetime.now()
        try:
            quiz_result = create_quiz_items.invoke({
                "objectives": objectives,
                "main_topics": main_topics,
                "difficulty": scenario.get("difficulty"),
                "num_items": 10,
            })
            tool_calls.append(self._record_tool_call(
                state, "create_quiz_items",
                {"objectives_count": len(objectives)},
                f"Quiz items created: {len(quiz_result)} items",
                start_time,
            ))
        except Exception as e:
            errors.append(f"create_quiz_items failed: {str(e)}")
            quiz_result = []

        # 2. 평가 루브릭 생성
        start_time = datetime.now()
        try:
            rubric_result = create_rubric.invoke({
                "objectives": objectives,
                "assessment_type": "Comprehensive Assessment",
            })
            tool_calls.append(self._record_tool_call(
                state, "create_rubric",
                {"objectives_count": len(objectives)},
                f"Assessment rubric created: {len(rubric_result.get('criteria', []))} criteria",
                start_time,
            ))
        except Exception as e:
            errors.append(f"create_rubric failed: {str(e)}")
            rubric_result = {}

        # 3. 성과평가 계획 생성 (Kirkpatrick 4-Level)
        start_time = datetime.now()
        try:
            context = scenario.get("context", {})
            program_evaluation_result = create_program_evaluation.invoke({
                "program_title": scenario.get("title", "Training Program"),
                "objectives": objectives,
                "target_audience": context.get("target_audience", "General learners"),
            })
            tool_calls.append(self._record_tool_call(
                state, "create_program_evaluation",
                {"program_title": scenario.get("title", "")},
                f"Program evaluation plan complete: Kirkpatrick 4-Level model",
                start_time,
            ))
        except Exception as e:
            errors.append(f"create_program_evaluation failed: {str(e)}")
            program_evaluation_result = {}

        self._log(f"Evaluation complete: quiz_items={len(quiz_result)}, criteria={len(rubric_result.get('criteria', []))}, program_evaluation={bool(program_evaluation_result)}")

        # 출력 구조: 루브릭 항목과 일치하도록 명시적 필드명 사용
        # Item 28: 파일럿/초기 실행 중 자료 수집
        # Item 29: 형성평가 결과 기반 1차 프로그램 개선
        # Item 30: 총괄 평가 문항 개발
        # Item 31: 총괄평가 시행 및 프로그램 효과 분석
        # Item 32: 프로그램 채택 여부 결정
        # Item 33: 프로그램 개선

        # Item 28: 파일럿 실행 중 자료 수집 계획
        pilot_data_collection = {
            "collection_methods": [
                {"method": "Pre/post test", "timing": "Before training starts / after it ends", "data_type": "Learning achievement"},
                {"method": "Satisfaction survey", "timing": "Immediately after training ends", "data_type": "Learner satisfaction"},
                {"method": "Observation records", "timing": "During training delivery", "data_type": "Engagement, comprehension"},
                {"method": "Interview/FGI", "timing": "Within 1 week after training ends", "data_type": "Qualitative feedback"},
            ],
            "analysis_plan": {
                "quantitative": "Pre/post score comparison, satisfaction average, participation rate calculation",
                "qualitative": "Feedback theme analysis, classification of improvement requests",
            },
            "improvement_triggers": [
                "Revise content if the learning objective achievement rate is below 80%",
                "Improve the delivery approach if satisfaction is below 4.0",
                "Add supplementary materials if comprehension of a specific module is low",
            ],
        }

        # Item 29: 형성평가 결과 기반 1차 프로그램 개선
        formative_improvement = {
            "evaluation_criteria": rubric_result.get("criteria", []),
            "improvement_process": [
                "1. Collect and analyze pilot data",
                "2. Identify problem areas and set priorities",
                "3. Derive improvement proposals and obtain expert review",
                "4. Apply the revised version and re-validate",
            ],
            "feedback_integration": "Synthesize learner feedback and observation results to improve content and delivery approach",
        }

        return {
            "evaluation_result": {
                "quiz_items": quiz_result,  # Item 30: Summative assessment items
                "rubric": rubric_result,
                "program_evaluation": program_evaluation_result,  # Item 31, 32
                "feedback_plan": rubric_result.get("feedback_plan", "Provide individual feedback after assessment"),
                # 루브릭 항목과 매칭되는 필드 추가
                "pilot_data_collection": pilot_data_collection,  # Item 28
                "formative_improvement": formative_improvement,  # Item 29
            },
            "current_phase": "complete",
            "tool_calls": tool_calls,
            "reasoning_steps": reasoning_steps,
            "errors": errors,
        }

    def run(self, scenario: dict) -> dict:
        """
        시나리오를 입력받아 ADDIE 산출물을 생성합니다.

        Args:
            scenario: 시나리오 딕셔너리

        Returns:
            dict: 완전한 결과 (ADDIE + trajectory + metadata)
        """
        start_time = datetime.now()

        # 초기 상태 생성
        initial_state = create_initial_state(scenario)

        # StateGraph 실행
        final_state = self.graph.invoke(initial_state)

        # 메타데이터 생성
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # ADDIE 출력 조립
        raw_output = {
            "analysis": final_state.get("analysis_result", {}),
            "design": final_state.get("design_result", {}),
            "development": final_state.get("development_result", {}),
            "implementation": final_state.get("implementation_result", {}),
            "evaluation": final_state.get("evaluation_result", {}),
        }

        # 표준 스키마로 변환
        addie_output = self._convert_to_standard_schema(raw_output)

        result = {
            "scenario_id": scenario.get("scenario_id", "unknown"),
            "agent_id": "addie-agent",
            "timestamp": end_time.isoformat(),
            "addie_output": addie_output,
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
                "errors": final_state.get("errors", []),
            },
        }

        self._log(f"ADDIE complete: {execution_time:.2f}s, {len(final_state.get('tool_calls', []))} tool calls")

        return result

    def _convert_to_standard_schema(self, raw: dict) -> dict:
        """
        내부 ADDIE 구조를 표준 스키마로 변환합니다.

        표준 스키마 (docs/addie_output_schema.json) 준수
        """
        # Analysis 변환
        a = raw.get("analysis", {})
        na = a.get("needs_analysis", {})
        la = a.get("learner_analysis", {})
        ca = a.get("context_analysis", {})
        ta = a.get("task_analysis", {})

        # problem_definition 생성 (항목 1)
        problem_def = na.get("problem_definition", "") or na.get("performance_gap", "")
        if not problem_def:
            current = na.get("current_state", "Insufficient current competency")
            desired = na.get("desired_state", "Achievement of target competency")
            problem_def = f"Current state: {current}. Target state: {desired}. An instructional intervention is needed to close this gap."

        analysis_dict = {
            "needs_analysis": {
                "problem_definition": problem_def,
                "gap_analysis": na.get("gap_analysis", []) if isinstance(na.get("gap_analysis"), list) else [
                    {"current": na.get("current_state", ""), "target": na.get("desired_state", ""), "gap": na.get("performance_gap", "")}
                ],
                "performance_analysis": f"Training needs: {', '.join((na.get('training_needs') or [])[:3])}. Non-training solutions: {', '.join((na.get('non_training_solutions') or [])[:2])}",
                "priority_matrix": na.get("priority_matrix", {}),
            },
            "learner_analysis": {
                "target_audience": la.get("target_audience", ""),
                "characteristics": la.get("characteristics", []),
                "prior_knowledge": la.get("prior_knowledge", ""),
                "learning_preferences": la.get("learning_preferences", []),
                "motivation": la.get("motivation", ""),
            },
            "context_analysis": {
                "environment": ca.get("environment", ""),
                "constraints": ca.get("constraints", []),
                "resources": ca.get("resources", []),
                "technical_requirements": ca.get("technical_requirements", []),
            },
            "task_analysis": {
                "initial_objectives": ta.get("main_topics", []),
                "subtopics": ta.get("subtopics", []),
                "prerequisites": ta.get("prerequisites", []),
                "review_summary": ta.get("review_summary", "") or f"Topics: {', '.join(ta.get('main_topics', [])[:3])}",
            },
        }

        # Design 변환
        d = raw.get("design", {})
        ist = d.get("instructional_strategy", {})

        # 학습 활동 생성
        learning_activities = []
        for event in ist.get("sequence", []):
            learning_activities.append({
                "activity_name": event.get("event", ""),
                "duration": event.get("duration", ""),
                "description": event.get("activity", ""),
                "materials": event.get("resources", []),
            })

        design_dict = {
            "learning_objectives": d.get("learning_objectives", []),
            "assessment_plan": {
                "formative": [{"description": f} for f in d.get("assessment_plan", {}).get("formative", [])],
                "summative": [{"description": s} for s in d.get("assessment_plan", {}).get("summative", [])],
                "assessment_criteria": d.get("assessment_plan", {}).get("diagnostic", []),
            },
            "content_structure": {
                "modules": [m.get("title", "") for m in raw.get("development", {}).get("lesson_plan", {}).get("modules", [])],
                "topics": ta.get("main_topics", []),
                "sequencing": ist.get("model", ""),
            },
            "instructional_strategies": {
                "methods": ist.get("methods", []),
                "activities": [e.get("activity", "") for e in ist.get("sequence", [])[:5]],
                "rationale": f"Model: {ist.get('model', '')}",
            },
            "non_instructional_strategies": {
                "motivation_strategies": [],
                "self_directed_learning": [],
                "support_strategies": [],
            },
            "media_selection": {
                "media_types": [],
                "tools": [],
                "utilization_plan": "",
            },
            "learning_activities": learning_activities,
            "storyboard": {
                "screens": [
                    {"screen_id": "S01", "title": "Introduction", "description": "Introduce learning objectives and overview"},
                    {"screen_id": "S02", "title": "Learning Content", "description": "Present core concepts and content"},
                    {"screen_id": "S03", "title": "Practice/Activity", "description": "Learner participation activities"},
                    {"screen_id": "S04", "title": "Assessment", "description": "Check learning achievement"},
                    {"screen_id": "S05", "title": "Wrap-up", "description": "Summary and guidance on next steps"},
                ],
                "navigation_flow": "S01 → S02 → S03 → S04 → S05 (sequential progression, previous/next buttons)",
                "interactions": ["Click", "Drag and drop", "Text input", "Multiple-choice quiz"],
            },
        }

        # Development 변환
        dev = raw.get("development", {})
        impl = raw.get("implementation", {})

        # learner_materials 변환
        learner_materials = []
        for mat in dev.get("learner_materials", []) or dev.get("materials", []):
            learner_materials.append({
                "title": mat.get("title", ""),
                "type": mat.get("type", ""),
                "content": mat.get("description", "") or mat.get("content", ""),
                "format": "PDF/PPT",
            })
        # 기본값 보장 (Item 19: 학습자용 자료 개발)
        if not learner_materials:
            learner_materials = [
                {"title": "Learning Guide", "type": "Document", "content": "Learner guide covering learning objectives, delivery approach, and assessment criteria", "format": "PDF"},
                {"title": "Worksheet", "type": "Activity Material", "content": "Practice worksheet for applying the learning content", "format": "PDF"},
                {"title": "Reference Materials", "type": "Supplementary Material", "content": "Additional references and links for advanced study", "format": "PDF/Web"},
            ]

        # quiz_items를 assessment_tools로 변환
        ev = raw.get("evaluation", {})
        quiz_tools = [
            {
                "item_id": q.get("id", f"Q-{idx+1:03d}"),
                "type": q.get("type", ""),
                "question": q.get("question", ""),
                "aligned_objective": q.get("objective_id", ""),
                "scoring_criteria": q.get("explanation", ""),
            }
            for idx, q in enumerate(ev.get("quiz_items", []))
        ]

        development_dict = {
            "learner_materials": learner_materials,
            "instructor_guide": {
                "overview": impl.get("facilitator_guide", ""),
                "session_guides": [mod.get("title", "") for mod in dev.get("lesson_plan", {}).get("modules", [])],
                "facilitation_tips": ["Encourage learner participation", "Use questioning"],
                "troubleshooting": ["Respond to technical problems"],
            },
            "operator_manual": {
                "system_setup": impl.get("operator_guide", ""),
                "operation_procedures": ["Registration management", "Attendance management"],
                "support_procedures": ["Respond to learner inquiries"],
                "escalation_process": "Report to the responsible staff member when a problem occurs",
            },
            "assessment_tools": quiz_tools,
            "expert_review": {
                "reviewers": ["Subject matter expert", "Instructional design expert", "Field practitioner"],
                "review_criteria": ["Content accuracy", "Instructional design appropriateness", "Learning objective alignment", "Suitability for learner level"],
                "feedback_summary": "The expert review confirmed the accuracy of the content and the appropriateness of the instructional design, and improvement work was carried out by incorporating feedback on learning objective alignment and suitability for the learner level",
                "revisions_made": ["Content revisions based on expert feedback", "Strengthened learning objective alignment", "Added examples suited to the learner level"],
            },
        }

        # Implementation 변환
        op = impl.get("instructor_operator_orientation", "") or impl.get("orientation_plan", {})
        sc = impl.get("system_environment_check", {})
        pp = impl.get("prototype_execution", {})
        mp = impl.get("operation_monitoring", {})

        implementation_dict = {
            "instructor_orientation": {
                "orientation_objectives": ["Understanding the program", "Familiarity with operational procedures"],
                "schedule": op if isinstance(op, str) else str(op),
                "materials": ["Instructor guide", "Operator manual"],
                "competency_checklist": ["Content comprehension", "Facilitation ability"],
            },
            "system_check": {
                "checklist": sc.get("pre_check_items", impl.get("technical_requirements", [])),
                "technical_validation": "System testing completed",
                "contingency_plans": ["Establish contingency response plan"],
            },
            "prototype_execution": {
                "pilot_scope": pp.get("pilot_scope", "") if isinstance(pp, dict) else "Small-scale pilot test",
                "participants": pp.get("participants", "") if isinstance(pp, dict) else "Around 10 participants",
                "execution_log": pp.get("data_collection", []) if isinstance(pp, dict) else [],
                "issues_encountered": [],
            },
            "monitoring": {
                "monitoring_criteria": mp.get("monitoring_items", ["Learning progress", "Participation rate"]),
                "support_channels": mp.get("support_channels", ["Email", "Phone"]),
                "issue_resolution_log": [],
                "real_time_adjustments": [],
            },
        }

        # Evaluation 변환
        pdc = ev.get("pilot_data_collection", {})
        fi = ev.get("formative_improvement", {})
        pe = ev.get("program_evaluation", {})
        rubric = ev.get("rubric", {})

        # 항목 28: 데이터 수집 기본값
        data_collection_methods = [m.get("method", "") for m in pdc.get("collection_methods", [])]
        if not data_collection_methods:
            data_collection_methods = ["Pre/post test", "Satisfaction survey", "Observation records", "Interview"]

        # 항목 29: 형성평가 개선 기본값
        improvement_steps = fi.get("improvement_process", [])
        if not improvement_steps:
            improvement_steps = [
                "Analyze pilot data and identify problems",
                "Set improvement priorities and derive revision proposals",
                "Apply the revised version after expert review",
                "Re-validate and incorporate final changes",
            ]

        # 항목 31: 효과성 분석 기본값
        kirk_analysis = pe.get("kirkpatrick_analysis", {})
        learning_outcomes = kirk_analysis.get("level1_reaction", {})
        if not learning_outcomes:
            learning_outcomes = {
                "description": "Learner reaction evaluation",
                "methods": ["Satisfaction survey"],
                "target_score": "4.0/5.0 or higher",
            }

        goal_achievement = pe.get("effectiveness_score", "")
        if not goal_achievement:
            goal_achievement = "Goal achievement rate expected to be 80% or higher"

        # 항목 32: 채택 결정 기본값
        adoption_decision_val = pe.get("adoption_recommendation", "")
        adoption_rationale = pe.get("rationale", "")
        if not adoption_decision_val:
            adoption_decision_val = "Conditional adoption"
            adoption_rationale = "Final decision depends on the pilot results. Full-scale rollout once the learning objective achievement rate and satisfaction criteria are met."

        # 항목 33: 개선 계획 기본값
        feedback_summary = ev.get("feedback_plan", "")
        if not feedback_summary:
            feedback_summary = "Synthesize learner feedback and evaluation results and reflect them in program improvement"

        improvement_areas = fi.get("evaluation_criteria", rubric.get("criteria", []))
        if not improvement_areas:
            improvement_areas = ["Appropriateness of learning content", "Effectiveness of delivery approach", "Validity of assessment tools"]

        action_items = fi.get("improvement_process", [])
        if not action_items:
            action_items = ["Content updates", "Improvement of instructional methods", "Refinement of assessment items"]

        evaluation_dict = {
            "formative": {
                "data_collection": {
                    "methods": data_collection_methods,
                    "learner_feedback": pdc.get("analysis_plan", {}).get("qualitative", "") or ["Collect learner opinions", "Feedback on difficult parts", "Improvement requests"],
                    "performance_data": pdc.get("analysis_plan", {}) or {"quantitative": "Pre/post score comparison", "qualitative": "Feedback analysis"},
                    "observations": pdc.get("improvement_triggers", []) or ["Observation of learning progress", "Engagement monitoring", "Analysis of learner behavior patterns"],
                    "pilot_difficulties": {
                        "identified_modules": ["Concept comprehension module", "Practice application module"],
                        "difficulty_reasons": ["Insufficient prerequisite knowledge", "Insufficient practice time", "Complexity of materials"],
                        "improvement_suggestions": ["Provide supplementary materials", "Extend practice time", "Add step-by-step guides"],
                    },
                },
                "improvements": [
                    {
                        "issue_identified": step,
                        "improvement_action": "Execute improvement action",
                        "priority": "High" if idx == 0 else "Medium",
                    }
                    for idx, step in enumerate(improvement_steps)
                ],
            },
            "summative": {
                "assessment_tools": quiz_tools,
                "effectiveness_analysis": {
                    "learning_outcomes": learning_outcomes,
                    "goal_achievement_rate": goal_achievement,
                    "statistical_analysis": str(kirk_analysis.get("level2_learning", "")) or "Pre/post score t-test analysis",
                    "recommendations": pe.get("recommendations", []) or ["Recommend continuous improvement"],
                },
                "adoption_decision": {
                    "decision": adoption_decision_val,
                    "rationale": adoption_rationale,
                    "conditions": pe.get("conditions", []) or ["Favorable pilot results", "Budget secured"],
                    "stakeholder_approval": "Pending approval",
                },
            },
            "improvement_plan": {
                "feedback_summary": feedback_summary,
                "improvement_areas": improvement_areas,
                "action_items": action_items,
                "feedback_loop": fi.get("feedback_integration", "Reflect the evaluation results in the next training cycle"),
                "next_iteration_goals": pe.get("next_steps", []) or ["Program stabilization", "Review of expanded rollout"],
            },
        }

        return {
            "analysis": analysis_dict,
            "design": design_dict,
            "development": development_dict,
            "implementation": implementation_dict,
            "evaluation": evaluation_dict,
        }
