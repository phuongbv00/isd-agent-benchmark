"""
Optimizer Agent: 교수설계 최적화 에이전트

평가 피드백을 기반으로 교수설계를 개선합니다.
- 순차적 ADDIE 파이프라인으로 단계별 최적화
- 약점 보완 (ADDIE Rubric 점수가 낮은 단계 집중)
- 학습자 맞춤화
"""

import json
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from eduplanner.agents.base import BaseAgent, AgentConfig
from eduplanner.agents.analyst import AnalysisResult
from eduplanner.models.schemas import (
    ADDIEOutput,
    EvaluationFeedback,
    Analysis,
    Design,
    Development,
    Implementation,
    Evaluation,
    LearnerAnalysis,
    ContextAnalysis,
    TaskAnalysis,
    LearningObjective,
    AssessmentPlan,
    InstructionalStrategy,
    InstructionalEvent,
    LessonPlan,
    Module,
    Activity,
    Material,
    QuizItem,
    Rubric,
)
from eduplanner.models.skill_tree import LearnerProfile
from eduplanner.agents.prompts import (
    ANALYSIS_PROMPT,
    DESIGN_PROMPT,
    DEVELOPMENT_PROMPT,
    IMPLEMENTATION_PROMPT,
    EVALUATION_PROMPT,
    get_optimization_prompt,
)


class OptimizerAgent(BaseAgent):
    """교수설계 최적화 에이전트"""

    def __init__(self, config: Optional[AgentConfig] = None, debug: bool = False):
        if config is None:
            # Optimizer는 안정적이고 일관된 개선을 위해 낮은 temperature 사용
            config = AgentConfig(
                temperature=0.3,
                max_tokens=8192,
            )
        super().__init__(config)
        self.debug = debug

    @property
    def name(self) -> str:
        return "Optimizer Agent"

    @property
    def role(self) -> str:
        return "Improves the instructional design based on evaluation feedback."

    def run(
        self,
        addie_output: ADDIEOutput,
        feedback: EvaluationFeedback,
        analysis_result: Optional[AnalysisResult] = None,
        learner_profile: Optional[LearnerProfile] = None,
        scenario_context: Optional[str] = None,
    ) -> ADDIEOutput:
        """
        피드백을 반영하여 교수설계를 순차적 파이프라인으로 최적화합니다.

        Args:
            addie_output: 현재 ADDIE 산출물
            feedback: 평가 피드백
            analysis_result: Analyst의 상세 분석 결과
            learner_profile: 학습자 프로필
            scenario_context: 시나리오 맥락

        Returns:
            ADDIEOutput: 최적화된 ADDIE 산출물
        """
        if self.debug:
            print("\n" + "="*60)
            print("[Optimizer] Starting sequential pipeline optimization")
            print(f"  Current score: {feedback.score:.1f}")
            print(f"  Weighted score: {feedback.weighted_score}")
            print(f"  ADDIE Scores: {feedback.addie_scores}")
            print("="*60)

        # 피드백 요약 생성
        feedback_summary = self._build_feedback_summary(feedback, analysis_result)

        # 현재 ADDIE 데이터를 딕셔너리로 변환
        current_data = {
            "analysis": self._addie_analysis_to_dict(addie_output.analysis),
            "design": self._addie_design_to_dict(addie_output.design),
            "development": self._addie_development_to_dict(addie_output.development),
            "implementation": self._addie_implementation_to_dict(addie_output.implementation),
            "evaluation": self._addie_evaluation_to_dict(addie_output.evaluation),
        }

        # ADDIE Rubric 점수가 낮은 단계 식별
        weak_stages = self._identify_weak_stages(feedback.addie_scores)

        if self.debug:
            print(f"\n[Optimizer] Stages needing improvement: {weak_stages if weak_stages else 'none (full optimization)'}")

        # 순차적 파이프라인 최적화
        optimized_data = self._optimize_sequential_pipeline(
            current_data=current_data,
            feedback_summary=feedback_summary,
            weak_stages=weak_stages,
            scenario_context=scenario_context,
            learner_profile=learner_profile,
        )

        # 최적화된 데이터를 ADDIEOutput으로 변환
        optimized_output = self._assemble_addie_output(optimized_data, addie_output)

        # 선택적 병합: 더 나은 부분만 적용
        merged_output = self._selective_merge(addie_output, optimized_output)

        if self.debug:
            print("\n[Optimizer] Optimization complete")
            print("="*60 + "\n")

        return merged_output

    def _identify_weak_stages(self, addie_scores: dict) -> list[str]:
        """ADDIE Rubric 점수가 낮은 ADDIE 단계 식별"""
        # ADDIE 항목과 단계 매핑
        item_to_stage = {
            "A1": "analysis", "A2": "analysis", "A3": "analysis",
            "D1": "design", "D2": "design", "D3": "design",
            "Dev1": "development", "Dev2": "development",
            "I1": "implementation", "I2": "implementation",
            "E1": "evaluation", "E2": "evaluation", "E3": "evaluation",
        }

        # 단계별 평균 점수 계산
        stage_scores = {}
        stage_counts = {}
        for item, score in addie_scores.items():
            stage = item_to_stage.get(item)
            if stage:
                stage_scores[stage] = stage_scores.get(stage, 0) + score
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

        # 평균 점수가 7점 미만인 단계 식별 (10점 만점 기준)
        weak_stages = []
        for stage, total in stage_scores.items():
            avg = total / stage_counts.get(stage, 1)
            if avg < 7.0:  # below 7 points means improvement needed
                weak_stages.append(stage)

        return weak_stages

    def _optimize_sequential_pipeline(
        self,
        current_data: dict,
        feedback_summary: str,
        weak_stages: list[str],
        scenario_context: Optional[str],
        learner_profile: Optional[LearnerProfile],
    ) -> dict:
        """순차적 파이프라인으로 ADDIE 단계 최적화"""
        import copy
        optimized = copy.deepcopy(current_data)

        # 최적화할 단계 결정 (약한 단계가 없으면 implementation만)
        stages_to_optimize = weak_stages if weak_stages else ["implementation"]

        # 항상 implementation은 포함 (가이드 상세화 필요)
        if "implementation" not in stages_to_optimize:
            stages_to_optimize.append("implementation")

        learner_context = ""
        if learner_profile:
            learner_context = learner_profile.skill_tree.to_prompt_context()

        for stage in ["analysis", "design", "development", "implementation", "evaluation"]:
            if stage not in stages_to_optimize:
                continue

            if self.debug:
                print(f"\n  [{stage.upper()}] Optimizing...")

            # 단계별 최적화 프롬프트 생성
            system_prompt = get_optimization_prompt(stage, feedback_summary)

            # 이전 단계 결과를 컨텍스트로 제공
            previous_context = self._build_previous_context(optimized, stage)

            user_prompt = f"""## Scenario Information
{scenario_context or ''}

## Learner Profile
{learner_context}

{previous_context}

## Current {stage.upper()} Stage Data
```json
{json.dumps(optimized.get(stage, {}), ensure_ascii=False, indent=2)}
```

Improve the data above according to the feedback.
Keep the existing content as much as possible and fix only the problems."""

            # LLM 호출
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            # 응답 파싱
            stage_data = self._parse_json_response(response.content)
            if stage_data:
                optimized[stage] = stage_data

                if self.debug:
                    if stage == "implementation":
                        fg_len = len(stage_data.get("facilitator_guide", ""))
                        lg_len = len(stage_data.get("learner_guide", ""))
                        print(f"    → facilitator_guide: {fg_len} chars, learner_guide: {lg_len} chars")

        return optimized

    def _build_previous_context(self, data: dict, current_stage: str) -> str:
        """현재 단계 이전의 결과를 컨텍스트로 구성"""
        stage_order = ["analysis", "design", "development", "implementation", "evaluation"]
        current_idx = stage_order.index(current_stage)

        if current_idx == 0:
            return ""

        parts = ["## Previous Stage Results"]
        for i in range(current_idx):
            prev_stage = stage_order[i]
            prev_data = data.get(prev_stage, {})

            # 요약 정보만 포함 (토큰 절약)
            if prev_stage == "analysis":
                la = prev_data.get("learner_analysis", {})
                parts.append(f"### Analysis\n- Target: {la.get('target_audience', '')}")
            elif prev_stage == "design":
                objs = prev_data.get("learning_objectives", [])
                parts.append(f"### Design\n- Learning objectives: {len(objs)}")
            elif prev_stage == "development":
                modules = prev_data.get("lesson_plan", {}).get("modules", [])
                parts.append(f"### Development\n- Modules: {len(modules)}")

        return "\n".join(parts)

    def _build_feedback_summary(
        self,
        feedback: EvaluationFeedback,
        analysis_result: Optional[AnalysisResult] = None,
    ) -> str:
        """피드백을 요약 문자열로 변환"""
        parts = [f"**Current score:** {feedback.score:.1f}/100"]
        if feedback.weighted_score:
            parts.append(f"**Weighted score:** {feedback.weighted_score:.1f}/100")

        # ADDIE 단계별 점수 요약
        parts.append("\n**ADDIE Rubric scores:**")
        stage_items = {
            "Analysis": ["A1", "A2", "A3"],
            "Design": ["D1", "D2", "D3"],
            "Development": ["Dev1", "Dev2"],
            "Implementation": ["I1", "I2"],
            "Evaluation": ["E1", "E2", "E3"],
        }
        for stage, items in stage_items.items():
            scores = [feedback.addie_scores.get(item, 0) for item in items]
            avg = sum(scores) / len(scores) if scores else 0
            status = "⚠️ Needs improvement" if avg < 7.0 else "✓"
            item_str = ", ".join(f"{item}:{feedback.addie_scores.get(item, 0):.1f}" for item in items)
            parts.append(f"- {stage}: average {avg:.1f}/10 {status} ({item_str})")

        if feedback.weaknesses:
            parts.append("\n**Weaknesses:**")
            for w in feedback.weaknesses[:5]:
                parts.append(f"- {w}")

        if feedback.suggestions:
            parts.append("\n**Improvement suggestions:**")
            for s in feedback.suggestions[:3]:
                parts.append(f"- {s}")

        if analysis_result and analysis_result.errors:
            parts.append("\n**Errors:**")
            for err in analysis_result.errors[:3]:
                parts.append(f"- {err.get('description', '')}")

        return "\n".join(parts)

    def _parse_json_response(self, response_text: str) -> dict:
        """LLM 응답에서 JSON 파싱"""
        import re

        # JSON 블록 추출
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else response_text

        try:
            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            return {}

    def _addie_analysis_to_dict(self, analysis: Analysis) -> dict:
        """Analysis를 딕셔너리로 변환"""
        return {
            "learner_analysis": {
                "target_audience": analysis.learner_analysis.target_audience,
                "characteristics": analysis.learner_analysis.characteristics,
                "prior_knowledge": analysis.learner_analysis.prior_knowledge,
                "learning_preferences": analysis.learner_analysis.learning_preferences,
                "motivation": analysis.learner_analysis.motivation,
                "challenges": analysis.learner_analysis.challenges,
            },
            "context_analysis": {
                "environment": analysis.context_analysis.environment,
                "duration": analysis.context_analysis.duration,
                "constraints": analysis.context_analysis.constraints,
                "resources": analysis.context_analysis.resources,
                "technical_requirements": analysis.context_analysis.technical_requirements,
            },
            "task_analysis": {
                "main_topics": analysis.task_analysis.main_topics,
                "subtopics": analysis.task_analysis.subtopics,
                "prerequisites": analysis.task_analysis.prerequisites,
            },
        }

    def _addie_design_to_dict(self, design: Design) -> dict:
        """Design을 딕셔너리로 변환"""
        return {
            "learning_objectives": [
                {
                    "id": obj.id,
                    "level": obj.level,
                    "statement": obj.statement,
                    "bloom_verb": obj.bloom_verb,
                    "measurable": obj.measurable,
                }
                for obj in design.learning_objectives
            ],
            "assessment_plan": {
                "diagnostic": design.assessment_plan.diagnostic,
                "formative": design.assessment_plan.formative,
                "summative": design.assessment_plan.summative,
            },
            "instructional_strategy": {
                "model": design.instructional_strategy.model,
                "sequence": [
                    {
                        "event": e.event,
                        "activity": e.activity,
                        "duration": e.duration,
                        "resources": e.resources,
                    }
                    for e in design.instructional_strategy.sequence
                ],
                "methods": design.instructional_strategy.methods,
            },
        }

    def _addie_development_to_dict(self, development: Development) -> dict:
        """Development를 딕셔너리로 변환"""
        return {
            "lesson_plan": {
                "total_duration": development.lesson_plan.total_duration,
                "modules": [
                    {
                        "title": m.title,
                        "duration": m.duration,
                        "objectives": m.objectives,
                        "activities": [
                            {
                                "time": a.time,
                                "activity": a.activity,
                                "description": a.description,
                                "resources": a.resources,
                            }
                            for a in m.activities
                        ],
                    }
                    for m in development.lesson_plan.modules
                ],
            },
            "materials": [
                {
                    "type": m.type,
                    "title": m.title,
                    "description": m.description,
                    "slides": m.slides,
                    "pages": m.pages,
                    "duration": m.duration,
                }
                for m in development.materials
            ],
        }

    def _addie_implementation_to_dict(self, impl: Implementation) -> dict:
        """Implementation을 딕셔너리로 변환"""
        return {
            "delivery_method": impl.delivery_method,
            "facilitator_guide": impl.facilitator_guide,
            "learner_guide": impl.learner_guide,
            "technical_requirements": impl.technical_requirements,
            "support_plan": impl.support_plan,
        }

    def _addie_evaluation_to_dict(self, evaluation: Evaluation) -> dict:
        """Evaluation을 딕셔너리로 변환"""
        result = {
            "quiz_items": [
                {
                    "id": q.id,
                    "question": q.question,
                    "type": q.type,
                    "options": q.options,
                    "answer": q.answer,
                    "explanation": q.explanation,
                    "objective_id": q.objective_id,
                    "difficulty": q.difficulty,
                }
                for q in evaluation.quiz_items
            ],
            "feedback_plan": evaluation.feedback_plan,
        }

        if evaluation.rubric:
            result["rubric"] = {
                "criteria": evaluation.rubric.criteria,
                "levels": evaluation.rubric.levels,
            }

        return result

    def _assemble_addie_output(self, data: dict, original: ADDIEOutput) -> ADDIEOutput:
        """딕셔너리 데이터를 ADDIEOutput으로 조립"""
        import copy

        # Analysis
        analysis_data = data.get("analysis", {})
        la_data = analysis_data.get("learner_analysis", {})
        ca_data = analysis_data.get("context_analysis", {})
        ta_data = analysis_data.get("task_analysis", {})

        analysis = Analysis(
            learner_analysis=LearnerAnalysis(
                target_audience=la_data.get("target_audience", original.analysis.learner_analysis.target_audience),
                characteristics=la_data.get("characteristics", original.analysis.learner_analysis.characteristics),
                prior_knowledge=la_data.get("prior_knowledge", original.analysis.learner_analysis.prior_knowledge),
                learning_preferences=la_data.get("learning_preferences", original.analysis.learner_analysis.learning_preferences),
                motivation=la_data.get("motivation", original.analysis.learner_analysis.motivation),
                challenges=la_data.get("challenges", original.analysis.learner_analysis.challenges),
            ),
            context_analysis=ContextAnalysis(
                environment=ca_data.get("environment", original.analysis.context_analysis.environment),
                duration=ca_data.get("duration", original.analysis.context_analysis.duration),
                constraints=ca_data.get("constraints", original.analysis.context_analysis.constraints),
                resources=ca_data.get("resources", original.analysis.context_analysis.resources),
                technical_requirements=ca_data.get("technical_requirements", original.analysis.context_analysis.technical_requirements),
            ),
            task_analysis=TaskAnalysis(
                main_topics=ta_data.get("main_topics", original.analysis.task_analysis.main_topics),
                subtopics=ta_data.get("subtopics", original.analysis.task_analysis.subtopics),
                prerequisites=ta_data.get("prerequisites", original.analysis.task_analysis.prerequisites),
            ),
        )

        # Design
        design_data = data.get("design", {})
        objectives = []
        for i, obj in enumerate(design_data.get("learning_objectives", [])):
            objectives.append(LearningObjective(
                id=obj.get("id", f"OBJ-{i+1:02d}"),
                level=obj.get("level", "Understand"),
                statement=obj.get("statement", ""),
                bloom_verb=obj.get("bloom_verb", "explain"),
                measurable=obj.get("measurable", True),
            ))
        if not objectives:
            objectives = copy.deepcopy(original.design.learning_objectives)

        strategy_data = design_data.get("instructional_strategy", {})
        events = []
        for event in strategy_data.get("sequence", []):
            # resources가 문자열인 경우 리스트로 변환
            resources_val = event.get("resources", [])
            if isinstance(resources_val, str):
                resources_val = [resources_val]
            elif not isinstance(resources_val, list):
                resources_val = []
            else:
                # 리스트 내 None 값 필터링
                resources_val = [r for r in resources_val if r is not None and isinstance(r, str)]

            events.append(InstructionalEvent(
                event=event.get("event", ""),
                activity=event.get("activity", ""),
                duration=event.get("duration"),
                resources=resources_val,
            ))
        if not events:
            events = copy.deepcopy(original.design.instructional_strategy.sequence)

        assessment_data = design_data.get("assessment_plan", {})
        design = Design(
            learning_objectives=objectives,
            assessment_plan=AssessmentPlan(
                diagnostic=assessment_data.get("diagnostic", original.design.assessment_plan.diagnostic),
                formative=assessment_data.get("formative", original.design.assessment_plan.formative),
                summative=assessment_data.get("summative", original.design.assessment_plan.summative),
            ),
            instructional_strategy=InstructionalStrategy(
                model=strategy_data.get("model", original.design.instructional_strategy.model),
                sequence=events,
                methods=strategy_data.get("methods", original.design.instructional_strategy.methods),
            ),
        )

        # Development
        dev_data = data.get("development", {})
        lesson_data = dev_data.get("lesson_plan", {})
        modules = []
        for mod in lesson_data.get("modules", []):
            activities = []
            for act in mod.get("activities", []):
                # resources 타입 변환 및 None 필터링
                act_resources = act.get("resources", [])
                if isinstance(act_resources, str):
                    act_resources = [act_resources]
                elif isinstance(act_resources, list):
                    act_resources = [r for r in act_resources if r is not None and isinstance(r, str)]
                else:
                    act_resources = []

                activities.append(Activity(
                    time=act.get("time", ""),
                    activity=act.get("activity", ""),
                    description=act.get("description"),
                    resources=act_resources,
                ))
            modules.append(Module(
                title=mod.get("title", ""),
                duration=mod.get("duration", ""),
                objectives=mod.get("objectives", []),
                activities=activities,
            ))
        if not modules:
            modules = copy.deepcopy(original.development.lesson_plan.modules)

        materials = []
        for mat in dev_data.get("materials", []):
            # slides/pages가 숫자 문자열인 경우 정수로 변환
            slides_val = mat.get("slides")
            if isinstance(slides_val, str):
                try:
                    slides_val = int(slides_val)
                except ValueError:
                    slides_val = None

            pages_val = mat.get("pages")
            if isinstance(pages_val, str):
                try:
                    pages_val = int(pages_val)
                except ValueError:
                    pages_val = None

            materials.append(Material(
                type=mat.get("type", ""),
                title=mat.get("title", ""),
                description=mat.get("description"),
                slides=slides_val,
                duration=mat.get("duration"),
                pages=pages_val,
            ))
        if not materials:
            materials = copy.deepcopy(original.development.materials)

        development = Development(
            lesson_plan=LessonPlan(
                total_duration=lesson_data.get("total_duration", original.development.lesson_plan.total_duration),
                modules=modules,
            ),
            materials=materials,
        )

        # Implementation
        impl_data = data.get("implementation", {})
        implementation = Implementation(
            delivery_method=impl_data.get("delivery_method", original.implementation.delivery_method),
            facilitator_guide=impl_data.get("facilitator_guide", original.implementation.facilitator_guide),
            learner_guide=impl_data.get("learner_guide", original.implementation.learner_guide),
            technical_requirements=impl_data.get("technical_requirements", original.implementation.technical_requirements),
            support_plan=impl_data.get("support_plan", original.implementation.support_plan),
        )

        # Evaluation
        eval_data = data.get("evaluation", {})
        quiz_items = []
        for i, item in enumerate(eval_data.get("quiz_items", [])):
            # answer가 리스트인 경우 쉼표로 연결 (다답형 문항 처리)
            answer_val = item.get("answer", "")
            if isinstance(answer_val, list):
                answer_val = ", ".join(str(a) for a in answer_val)
            elif not isinstance(answer_val, str):
                answer_val = str(answer_val) if answer_val is not None else ""

            # options가 dict 리스트인 경우 문자열 리스트로 변환
            options_val = item.get("options", [])
            if isinstance(options_val, list):
                options_val = [
                    str(opt.get("text", opt.get("label", str(opt)))) if isinstance(opt, dict) else str(opt)
                    for opt in options_val if opt is not None
                ]
            else:
                options_val = []

            quiz_items.append(QuizItem(
                id=item.get("id", f"Q-{i+1:02d}"),
                question=item.get("question", ""),
                type=item.get("type", "multiple_choice"),
                options=options_val,
                answer=answer_val,
                explanation=item.get("explanation"),
                objective_id=item.get("objective_id"),
                difficulty=item.get("difficulty"),
            ))
        if not quiz_items:
            quiz_items = copy.deepcopy(original.evaluation.quiz_items)

        rubric_data = eval_data.get("rubric")
        rubric = None
        if rubric_data:
            rubric = Rubric(
                criteria=rubric_data.get("criteria", []),
                levels=rubric_data.get("levels", {}),
            )
        elif original.evaluation.rubric:
            rubric = copy.deepcopy(original.evaluation.rubric)

        evaluation = Evaluation(
            quiz_items=quiz_items,
            rubric=rubric,
            feedback_plan=eval_data.get("feedback_plan", original.evaluation.feedback_plan),
        )

        return ADDIEOutput(
            analysis=analysis,
            design=design,
            development=development,
            implementation=implementation,
            evaluation=evaluation,
        )

    def _selective_merge(
        self,
        original: ADDIEOutput,
        optimized: ADDIEOutput,
    ) -> ADDIEOutput:
        """
        원본과 최적화된 결과를 비교하여 더 나은 부분만 병합합니다.
        - 리스트 항목 수가 줄어들면 원본 유지
        - 필수 요소가 누락되면 원본 유지
        """
        import copy

        result = copy.deepcopy(original)

        # 디버깅: 병합 결정 추적
        merge_decisions = []

        # Analysis 병합
        orig_la = original.analysis.learner_analysis
        opt_la = optimized.analysis.learner_analysis

        # characteristics: 개수가 같거나 늘어나면 적용
        cond = len(opt_la.characteristics) >= len(orig_la.characteristics)
        merge_decisions.append(f"characteristics: orig={len(orig_la.characteristics)}, opt={len(opt_la.characteristics)} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.analysis.learner_analysis.characteristics = opt_la.characteristics

        # learning_preferences: 개수 유지
        cond = len(opt_la.learning_preferences) >= len(orig_la.learning_preferences)
        merge_decisions.append(f"learning_preferences: orig={len(orig_la.learning_preferences)}, opt={len(opt_la.learning_preferences)} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.analysis.learner_analysis.learning_preferences = opt_la.learning_preferences

        # challenges: 개수 유지
        cond = len(opt_la.challenges) >= len(orig_la.challenges)
        merge_decisions.append(f"challenges: orig={len(orig_la.challenges)}, opt={len(opt_la.challenges)} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.analysis.learner_analysis.challenges = opt_la.challenges

        # motivation: 내용이 더 길면 적용
        cond = opt_la.motivation and len(str(opt_la.motivation)) >= len(str(orig_la.motivation or ""))
        merge_decisions.append(f"motivation: orig_len={len(str(orig_la.motivation or ''))}, opt_len={len(str(opt_la.motivation or ''))} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.analysis.learner_analysis.motivation = opt_la.motivation

        # Design 병합
        orig_d = original.design
        opt_d = optimized.design

        # learning_objectives: 개수가 늘어나면 적용 (개선으로 간주)
        # min 조건 제거: Generator/재시도에서 최소 요구사항 보장
        cond = len(opt_d.learning_objectives) >= len(orig_d.learning_objectives)
        merge_decisions.append(f"learning_objectives: orig={len(orig_d.learning_objectives)}, opt={len(opt_d.learning_objectives)} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.design.learning_objectives = opt_d.learning_objectives

        # instructional_strategy.sequence: 9개 Event 유지
        cond1 = len(opt_d.instructional_strategy.sequence) >= 9
        cond2 = len(opt_d.instructional_strategy.sequence) >= len(orig_d.instructional_strategy.sequence)
        merge_decisions.append(f"instructional_strategy: orig={len(orig_d.instructional_strategy.sequence)}, opt={len(opt_d.instructional_strategy.sequence)}, min=9 -> {'APPLY' if (cond1 or cond2) else 'REJECT'}")
        if cond1:
            result.design.instructional_strategy = opt_d.instructional_strategy
        elif cond2:
            result.design.instructional_strategy = opt_d.instructional_strategy

        # assessment_plan: 각 항목 개수 유지
        cond = len(opt_d.assessment_plan.formative) >= len(orig_d.assessment_plan.formative)
        merge_decisions.append(f"assessment_formative: orig={len(orig_d.assessment_plan.formative)}, opt={len(opt_d.assessment_plan.formative)} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.design.assessment_plan.formative = opt_d.assessment_plan.formative

        cond = len(opt_d.assessment_plan.summative) >= len(orig_d.assessment_plan.summative)
        merge_decisions.append(f"assessment_summative: orig={len(orig_d.assessment_plan.summative)}, opt={len(opt_d.assessment_plan.summative)} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.design.assessment_plan.summative = opt_d.assessment_plan.summative

        cond = len(opt_d.assessment_plan.diagnostic) >= len(orig_d.assessment_plan.diagnostic)
        merge_decisions.append(f"assessment_diagnostic: orig={len(orig_d.assessment_plan.diagnostic)}, opt={len(opt_d.assessment_plan.diagnostic)} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.design.assessment_plan.diagnostic = opt_d.assessment_plan.diagnostic

        # Development 병합
        orig_dev = original.development
        opt_dev = optimized.development

        # modules: 개수가 늘어나면 적용
        cond = len(opt_dev.lesson_plan.modules) >= len(orig_dev.lesson_plan.modules)
        merge_decisions.append(f"modules: orig={len(orig_dev.lesson_plan.modules)}, opt={len(opt_dev.lesson_plan.modules)} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.development.lesson_plan = opt_dev.lesson_plan

        # materials: 개수 유지
        cond = len(opt_dev.materials) >= len(orig_dev.materials)
        merge_decisions.append(f"materials: orig={len(orig_dev.materials)}, opt={len(opt_dev.materials)} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.development.materials = opt_dev.materials

        # Implementation 병합 (가이드가 더 길면 적용)
        orig_impl = original.implementation
        opt_impl = optimized.implementation

        cond = opt_impl.facilitator_guide and len(str(opt_impl.facilitator_guide)) >= len(str(orig_impl.facilitator_guide or ""))
        merge_decisions.append(f"facilitator_guide: orig_len={len(str(orig_impl.facilitator_guide or ''))}, opt_len={len(str(opt_impl.facilitator_guide or ''))} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.implementation.facilitator_guide = opt_impl.facilitator_guide

        cond = opt_impl.learner_guide and len(str(opt_impl.learner_guide)) >= len(str(orig_impl.learner_guide or ""))
        merge_decisions.append(f"learner_guide: orig_len={len(str(orig_impl.learner_guide or ''))}, opt_len={len(str(opt_impl.learner_guide or ''))} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.implementation.learner_guide = opt_impl.learner_guide

        cond = len(opt_impl.technical_requirements) >= len(orig_impl.technical_requirements)
        merge_decisions.append(f"technical_requirements: orig={len(orig_impl.technical_requirements)}, opt={len(opt_impl.technical_requirements)} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.implementation.technical_requirements = opt_impl.technical_requirements

        # Evaluation 병합
        orig_eval = original.evaluation
        opt_eval = optimized.evaluation

        # quiz_items: 개수 유지 (최소 10개 권장)
        cond = len(opt_eval.quiz_items) >= len(orig_eval.quiz_items)
        merge_decisions.append(f"quiz_items: orig={len(orig_eval.quiz_items)}, opt={len(opt_eval.quiz_items)} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.evaluation.quiz_items = opt_eval.quiz_items

        # rubric: 기준 개수 유지
        if opt_eval.rubric and orig_eval.rubric:
            cond = len(opt_eval.rubric.criteria) >= len(orig_eval.rubric.criteria)
            merge_decisions.append(f"rubric_criteria: orig={len(orig_eval.rubric.criteria)}, opt={len(opt_eval.rubric.criteria)} -> {'APPLY' if cond else 'REJECT'}")
            if cond:
                result.evaluation.rubric = opt_eval.rubric
        elif opt_eval.rubric:
            merge_decisions.append(f"rubric: orig=None, opt=exists -> APPLY")
            result.evaluation.rubric = opt_eval.rubric

        # feedback_plan: 내용이 더 길면 적용
        cond = opt_eval.feedback_plan and len(str(opt_eval.feedback_plan)) >= len(str(orig_eval.feedback_plan or ""))
        merge_decisions.append(f"feedback_plan: orig_len={len(str(orig_eval.feedback_plan or ''))}, opt_len={len(str(opt_eval.feedback_plan or ''))} -> {'APPLY' if cond else 'REJECT'}")
        if cond:
            result.evaluation.feedback_plan = opt_eval.feedback_plan

        # 디버깅 출력
        if self.debug:
            print("\n" + "="*60)
            print("[SELECTIVE_MERGE] Merge decision details:")
            print("="*60)
            for decision in merge_decisions:
                status = "✓" if "APPLY" in decision else "✗"
                print(f"  {status} {decision}")

            # REJECT된 항목 수 집계
            rejected = sum(1 for d in merge_decisions if "REJECT" in d)
            applied = sum(1 for d in merge_decisions if "APPLY" in d)
            print(f"\n  Total: APPLY={applied}, REJECT={rejected}")
            print("="*60 + "\n")

        return result
