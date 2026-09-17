"""
EduPlanner Main Agent: 3-Agent 협업 오케스트레이터

Evaluator, Optimizer, Analyst의 협업을 조율하여
교수설계 산출물을 생성하고 개선합니다.

협업 흐름:
1. Generator가 초기 ADDIE 산출물 생성
2. Evaluator가 CIDPP 평가 수행
3. 목표 점수 미달 시:
   - Analyst가 문제점 분석
   - Optimizer가 개선 수행
4. 반복하여 품질 목표 달성
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from eduplanner.agents.base import BaseAgent, AgentConfig
from eduplanner.agents.evaluator import EvaluatorAgent
from eduplanner.agents.optimizer import OptimizerAgent
from eduplanner.agents.analyst import AnalystAgent
from eduplanner.models.schemas import (
    ScenarioInput,
    ADDIEOutput,
    AgentResult,
    Trajectory,
    Metadata,
    ToolCall,
    Analysis,
    Design,
    Development,
    Implementation,
    Evaluation,
    LearnerAnalysis,
    ContextAnalysis,
    TaskAnalysis,
    NeedsAnalysis,
    LearningObjective,
    AssessmentPlan,
    InstructionalStrategy,
    InstructionalEvent,
    PrototypeDesign,
    LessonPlan,
    Module,
    Activity,
    Material,
    SlideContent,
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
)


# 기존 통합 프롬프트 (Optimizer에서 사용 - 하위 호환성 유지용)
GENERATOR_SYSTEM_PROMPT = """You are an instructional design expert with 20 years of experience.

## Role
Generate systematic and **detailed** instructional design outputs according to the ADDIE model.

## ⚠️ Important: MINIMUM REQUIREMENTS

The requirements below must be met. Failure to meet them is evaluated as falling short of the quality standard.

### Analysis Stage
- learner_analysis.characteristics: **Minimum 5** specific characteristics (at least 1 sentence each)
- learner_analysis.learning_preferences: **Minimum 4**
- learner_analysis.challenges: **Minimum 3** anticipated difficulties
- learner_analysis.motivation: Explain motivation level and reasons in **2-3 sentences**
- context_analysis.constraints: **Minimum 3**
- context_analysis.resources: **Minimum 3**
- context_analysis.technical_requirements: **Minimum 2**
- task_analysis.main_topics: **Minimum 3**
- task_analysis.subtopics: **Minimum 6** (at least 2 per main_topic)
- task_analysis.prerequisites: **Minimum 2**

### Design Stage
- learning_objectives: **Minimum 5** (Bloom's level distribution required)
  - Remember/Understand: 1-2
  - Apply/Analyze: 2-3
  - Evaluate/Create: 1-2
- assessment_plan.diagnostic: **Minimum 2** methods
- assessment_plan.formative: **Minimum 2** methods
- assessment_plan.summative: **Minimum 2** methods
- instructional_strategy.sequence: Include **all 9 Events** (required!)
- instructional_strategy.methods: **Minimum 3**

### Development Stage
- lesson_plan.modules: **Minimum 3** modules
- Each module.activities: **Minimum 3** activities
- materials: **Minimum 5** materials (slides, pages values required - no null)
  - **content field required**: Write the actual content of each material
  - Handouts: Actual text content to be distributed (minimum 500 characters)
  - Quiz materials: Include items and answer options
  - **slide_contents required for presentation/slide materials** (must be included!):
    - Detailed content for each slide (slide_number, title, bullet_points, speaker_notes)
    - 3-5 key bullet_points per slide
    - Include detailed explanations for the presenter in speaker_notes
    - Example format:
    ```json
    {
      "type": "Presentation",
      "title": "Lecture Slides",
      "slides": 10,
      "slide_contents": [
        {"slide_number": 1, "title": "Training Introduction", "bullet_points": ["Welcome", "Learning objectives", "Schedule overview"], "speaker_notes": "Welcome the participants and introduce the training objectives."},
        {"slide_number": 2, "title": "Core Concepts", "bullet_points": ["Explanation of concept 1", "Explanation of concept 2", "Real examples"], "speaker_notes": "Explain the core concepts together with examples."}
      ]
    }
    ```

### Evaluation Stage
- quiz_items: **Minimum 10** (distributed by difficulty)
  - easy: 3-4
  - medium: 4-5
  - hard: 2-3
  - **options required**: Provide 4 answer options for multiple choice
  - **answer required**: Specify the correct answer
  - **explanation required**: Provide an answer explanation

### Implementation Stage
- facilitator_guide: **Minimum 200 characters** detailed guide (specific facilitation instructions)
- learner_guide: **Minimum 200 characters** detailed guide (guidance on how to learn)
- technical_requirements: **Minimum 2**

- rubric.criteria: **Minimum 5** assessment criteria
- rubric.levels: Specify **concrete criteria** for each level (excellent/good/needs_improvement) (1-2 sentences each)
- feedback_plan: **2-3 sentence** detailed plan

## ADDIE Framework

### 1. Analysis
- Learner analysis: Target audience, characteristics, prior knowledge, preferences, motivation, difficulties
- Context analysis: Learning environment, time, constraints, resources, technical requirements
- Task analysis: Main topics, subtopics, prerequisite learning

### 2. Design
- Learning objectives: Set objectives by Bloom's Taxonomy level
- Assessment plan: Diagnostic/formative/summative assessment methods
- Instructional strategy: Instructional events based on Gagné's 9 Events

### 3. Development
- Lesson plan: Module-by-module structure, time allocation
- Learning materials: Required textbooks, slides, media

### 4. Implementation
- Delivery method, facilitator guide, learner guide
- Technical requirements, support plan

### 5. Evaluation
- Quiz items, assessment rubric, feedback plan

## Bloom's Taxonomy Verbs
- Remember: define, list, recognize, recall, name
- Understand: explain, summarize, interpret, classify, exemplify
- Apply: apply, demonstrate, use, execute, implement
- Analyze: analyze, compare, distinguish, organize, attribute
- Evaluate: evaluate, judge, critique, justify, verify
- Create: design, develop, generate, construct, plan

## Gagné's 9 Events (must include all!)
1. Gain attention
2. Inform learners of objectives
3. Stimulate recall of prior learning
4. Present content
5. Provide learning guidance
6. Elicit performance
7. Provide feedback
8. Assess performance
9. Enhance retention and transfer

## 📋 Quality Self-Verification (must check after generation)

After generating the output, verify the following:
□ Do all learning_objectives start with a measurable verb?
□ Does instructional_strategy.sequence include all 9 Events?
□ Is each quiz_item linked (objective_id) to a specific learning_objective?
□ Does the total time allocation match duration?
□ Is every description at least 2 sentences?
□ Are the slides and pages values of materials all entered as numbers?
□ **Is facilitator_guide 200+ characters and does it include step-by-step instructions?**
□ **Is learner_guide 100+ characters and does it include specific learning guidance?**

## ⚠️ Required Elements of the Implementation Guides

**facilitator_guide** (minimum 200 characters):
- Step-by-step facilitation instructions (use numbers 1, 2, 3...)
- Specific guidance for each module/activity
- Time management tips
- Methods for encouraging learner participation

Example:
"1. Gain attention (5 min): Play a video and spark interest with a simple question
2. Present objectives (3 min): Present the learning objectives visually on the whiteboard
3. Deliver content (20 min): Explain core concepts while going through the slides..."

**learner_guide** (minimum 100 characters):
- Guidance on before/during/after learning activities
- How to participate
- How to ask questions/request help

## Output Example (for reference on level of detail - write content to fit the scenario)

Below is an example of the **level of detail**. Write the actual content to fit the given scenario:

```json
{
  "analysis": {
    "learner_analysis": {
      "target_audience": "[Target audience of the scenario]",
      "characteristics": [
        "[Demographic characteristics such as age range and background of the target audience]",
        "[Education level and field of expertise]",
        "[Level of relevant experience]",
        "[Learning attitude and disposition]",
        "[Ability to use technology/tools]"
      ],
      "prior_knowledge": "[Describe in detail in 2-3 sentences what the target audience already knows and what they lack]",
      "learning_preferences": [
        "[Preferred learning mode 1]",
        "[Preferred learning mode 2]",
        "[Preferred content type]",
        "[Learning environment preference]"
      ],
      "motivation": "[Describe learning motivation specifically in 2-3 sentences. Include both intrinsic and extrinsic motivation]",
      "challenges": [
        "[Anticipated learning difficulty 1]",
        "[Anticipated learning difficulty 2]",
        "[Anticipated learning difficulty 3]"
      ]
    },
    "context_analysis": {
      "environment": "[Learning environment]",
      "duration": "[Total learning time]",
      "constraints": ["[Constraint 1]", "[Constraint 2]", "[Constraint 3]"],
      "resources": ["[Available resource 1]", "[Available resource 2]", "[Available resource 3]", "[Available resource 4]"],
      "technical_requirements": ["[Technical requirement 1]", "[Technical requirement 2]", "[Technical requirement 3]"]
    },
    "task_analysis": {
      "main_topics": ["[Main topic 1]", "[Main topic 2]", "[Main topic 3]"],
      "subtopics": ["[Subtopic 1-1]", "[Subtopic 1-2]", "[Subtopic 2-1]", "[Subtopic 2-2]", "[Subtopic 3-1]", "[Subtopic 3-2]"],
      "prerequisites": ["[Prerequisite learning 1]", "[Prerequisite learning 2]"]
    }
  },
  "design": {
    "learning_objectives": [
      {"id": "OBJ-01", "level": "Remember", "statement": "[Objective starting with a measurable action verb]", "bloom_verb": "[verb]", "measurable": true},
      {"id": "OBJ-02", "level": "Understand", "statement": "[Objective starting with a measurable action verb]", "bloom_verb": "[verb]", "measurable": true},
      {"id": "OBJ-03", "level": "Apply", "statement": "[Objective starting with a measurable action verb]", "bloom_verb": "[verb]", "measurable": true},
      {"id": "OBJ-04", "level": "Apply", "statement": "[Objective starting with a measurable action verb]", "bloom_verb": "[verb]", "measurable": true},
      {"id": "OBJ-05", "level": "Analyze", "statement": "[Objective starting with a measurable action verb]", "bloom_verb": "[verb]", "measurable": true}
    ],
    "assessment_plan": {
      "diagnostic": ["[Diagnostic assessment method 1]", "[Diagnostic assessment method 2]"],
      "formative": ["[Formative assessment method 1]", "[Formative assessment method 2]", "[Formative assessment method 3]"],
      "summative": ["[Summative assessment method 1]", "[Summative assessment method 2]"]
    },
    "instructional_strategy": {
      "model": "Gagné's 9 Events",
      "sequence": [
        {"event": "Gain attention", "activity": "[Specific activity description]", "duration": "[time]", "resources": ["[resource]"]},
        {"event": "Inform learners of objectives", "activity": "[Specific activity description]", "duration": "[time]", "resources": ["[resource]"]},
        {"event": "Stimulate recall of prior learning", "activity": "[Specific activity description]", "duration": "[time]", "resources": ["[resource]"]},
        {"event": "Present content", "activity": "[Specific activity description]", "duration": "[time]", "resources": ["[resource]"]},
        {"event": "Provide learning guidance", "activity": "[Specific activity description]", "duration": "[time]", "resources": ["[resource]"]},
        {"event": "Elicit performance", "activity": "[Specific activity description]", "duration": "[time]", "resources": ["[resource]"]},
        {"event": "Provide feedback", "activity": "[Specific activity description]", "duration": "[time]", "resources": ["[resource]"]},
        {"event": "Assess performance", "activity": "[Specific activity description]", "duration": "[time]", "resources": ["[resource]"]},
        {"event": "Enhance retention and transfer", "activity": "[Specific activity description]", "duration": "[time]", "resources": ["[resource]"]}
      ],
      "methods": ["[Instructional method 1]", "[Instructional method 2]", "[Instructional method 3]"]
    }
  },
  "implementation": {
    "delivery_method": "[Delivery method]",
    "facilitator_guide": "1. Pre-preparation (10 min before): Inspect the classroom, test the projector, lay out learning materials\n2. Opening (5 min): Welcome, guide to today's learning objectives, icebreaking activity\n3. Module 1 delivery (20 min): Group discussion after slide explanation, Q&A\n4. Practice guidance (15 min): Support individual practice, 1:1 help for struggling learners\n5. Wrap-up (5 min): Summarize key content, guide next steps",
    "learner_guide": "1. Before learning: Complete the pre-survey, set personal learning goals\n2. During learning: Ask questions actively, participate in group activities, collaborate with peers during practice\n3. After learning: Review the handouts, establish a plan for applying it on the job",
    "technical_requirements": ["[Technical requirement 1]", "[Technical requirement 2]", "[Technical requirement 3]"],
    "support_plan": "[Learner support plan]"
  }
}
```

## Output Format
Referring to the level of detail in the example above, output a complete JSON structure **that fits the scenario**:

```json
{
  "analysis": {...},
  "design": {...},
  "development": {...},
  "implementation": {...},
  "evaluation": {...}
}
```
"""


class EduPlannerAgent(BaseAgent):
    """EduPlanner 메인 에이전트 (3-Agent 오케스트레이터)"""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        max_iterations: int = 2,
        target_score: float = 90.0,
        debug: bool = False,
    ):
        if config is None:
            config = AgentConfig(
                temperature=0.7,
                max_tokens=16384,
            )
        super().__init__(config)

        self.max_iterations = max_iterations
        self.target_score = target_score
        self.debug = debug

        # 하위 에이전트들
        self._evaluator: Optional[EvaluatorAgent] = None
        self._optimizer: Optional[OptimizerAgent] = None
        self._analyst: Optional[AnalystAgent] = None

    def _agent_config(self, *, temperature: float, max_tokens: int) -> AgentConfig:
        """하위 에이전트별 생성 파라미터만 조정하고 LLM provider/key는 공유합니다."""
        return self.config.model_copy(update={
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

    @property
    def name(self) -> str:
        return "EduPlanner Agent"

    @property
    def role(self) -> str:
        return "Generates high-quality instructional design outputs through 3-Agent collaboration."

    @property
    def evaluator(self) -> EvaluatorAgent:
        """Evaluator Agent (지연 초기화)"""
        if self._evaluator is None:
            self._evaluator = EvaluatorAgent(
                config=self._agent_config(temperature=0.7, max_tokens=4096)
            )
        return self._evaluator

    @property
    def optimizer(self) -> OptimizerAgent:
        """Optimizer Agent (지연 초기화)"""
        if self._optimizer is None:
            self._optimizer = OptimizerAgent(
                config=self._agent_config(temperature=0.3, max_tokens=8192),
                debug=self.debug,
            )
        return self._optimizer

    @property
    def analyst(self) -> AnalystAgent:
        """Analyst Agent (지연 초기화)"""
        if self._analyst is None:
            self._analyst = AnalystAgent(
                config=self._agent_config(temperature=0.7, max_tokens=4096)
            )
        return self._analyst

    def run(self, scenario_input: ScenarioInput) -> AgentResult:
        """
        교수설계 시나리오를 입력받아 ADDIE 산출물을 생성합니다.

        Args:
            scenario_input: 시나리오 입력

        Returns:
            AgentResult: 최종 결과 (ADDIE 산출물 + 궤적 + 메타데이터)
        """
        start_time = datetime.now()
        trajectory = Trajectory()
        total_tokens = 0
        step_counter = 0  # tool_calls step counter

        # target_audience 추론: 기존 필드가 없으면 IDLD 필드에서 조합
        target_audience = scenario_input.context.target_audience
        if not target_audience:
            parts_audience = []
            if scenario_input.context.learner_age:
                parts_audience.append(scenario_input.context.learner_age)
            if scenario_input.context.learner_role:
                parts_audience.append(scenario_input.context.learner_role)
            if scenario_input.context.learner_education:
                parts_audience.append(f"({scenario_input.context.learner_education})")
            target_audience = " ".join(parts_audience) if parts_audience else "Learners"

        # 학습자 프로필 생성
        learner_profile = LearnerProfile.from_scenario(
            target_audience=target_audience,
            prior_knowledge=scenario_input.context.prior_knowledge,
            learning_environment=scenario_input.context.learning_environment or "TBD",
        )

        # 시나리오 컨텍스트 문자열
        scenario_context = self._build_scenario_context(scenario_input)

        # 1단계: 초기 ADDIE 산출물 생성
        trajectory.reasoning_steps.append("Step 1: Generate initial ADDIE output")
        gen_start = datetime.now()
        addie_output = self._generate_initial_output(scenario_input, learner_profile)
        gen_end = datetime.now()
        step_counter += 1

        trajectory.tool_calls.append(ToolCall(
            step=step_counter,
            tool="generate_initial_addie",
            args={"scenario_id": scenario_input.scenario_id},
            result="Initial ADDIE output generation complete",
            timestamp=gen_start,
            duration_ms=int((gen_end - gen_start).total_seconds() * 1000),
        ))

        # 2단계: 반복적 개선 루프
        best_output = addie_output
        best_score = 0.0
        score_history = []  # track score history

        for iteration in range(1, self.max_iterations + 1):
            trajectory.reasoning_steps.append(f"Step {iteration + 1}: Evaluation and improvement iteration {iteration}")

            # 2.1 & 2.2: Evaluator와 Analyst 병렬 실행 (#79 성능 최적화)
            parallel_start = datetime.now()

            # ThreadPoolExecutor로 동기 함수들을 병렬 실행
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Evaluator 실행
                eval_future = executor.submit(
                    self.evaluator.run,
                    addie_output=addie_output,
                    learner_profile=learner_profile,
                    scenario_context=scenario_context,
                )
                # Analyst 실행
                analyst_future = executor.submit(
                    self.analyst.run,
                    addie_output=addie_output,
                    scenario_input=scenario_input,
                    learner_profile=learner_profile,
                )

                # 결과 수집
                feedback = eval_future.result()
                analysis_result = analyst_future.result()

            parallel_end = datetime.now()

            # Evaluator 결과 기록
            step_counter += 1
            trajectory.tool_calls.append(ToolCall(
                step=step_counter,
                tool="evaluate_addie",
                args={"iteration": iteration},
                result=f"ADDIE evaluation complete: {feedback.score:.1f} points",
                timestamp=parallel_start,
                duration_ms=int((parallel_end - parallel_start).total_seconds() * 1000 / 2),
                output_data={"score": feedback.score, "addie": feedback.addie_scores},
                feedback={"suggestions": feedback.suggestions[:2]} if feedback.suggestions else None,
            ))

            # Analyst 결과 기록
            step_counter += 1
            trajectory.tool_calls.append(ToolCall(
                step=step_counter,
                tool="analyze_addie",
                args={"iteration": iteration},
                result=f"Analysis complete: quality={analysis_result.quality_level}, errors={len(analysis_result.errors)}",
                timestamp=parallel_start,
                duration_ms=int((parallel_end - parallel_start).total_seconds() * 1000 / 2),
                output_data={
                    "quality": analysis_result.quality_level,
                    "errors": len(analysis_result.errors),
                    "missing": len(analysis_result.missing_elements),
                },
                feedback={"summary": analysis_result.summary} if hasattr(analysis_result, 'summary') and analysis_result.summary else None,
            ))

            # 점수 이력 추가
            score_history.append(feedback.score)

            # 최고 점수 갱신
            if feedback.score > best_score:
                best_score = feedback.score
                best_output = addie_output

            # 조기 종료 조건 (#73 성능 최적화)
            # 1. 목표 점수(85점) 도달 시 종료
            # 2. 점수 개선 없으면 종료 (반복 낭비 방지)
            if feedback.score >= 85.0:
                trajectory.reasoning_steps.append(
                    f"Early termination: target score reached ({feedback.score:.1f} points)"
                )
                break
            if len(score_history) >= 2 and feedback.score <= score_history[-2]:
                trajectory.reasoning_steps.append(
                    f"Early termination: no score improvement ({score_history[-2]:.1f} → {feedback.score:.1f})"
                )
                break

            # 2.3: 최적화 (Evaluator feedback + Analyst 분석 결과 모두 전달)
            opt_start = datetime.now()
            addie_output = self.optimizer.run(
                addie_output=addie_output,
                feedback=feedback,
                analysis_result=analysis_result,
                learner_profile=learner_profile,
                scenario_context=scenario_context,
            )
            opt_end = datetime.now()
            step_counter += 1

            trajectory.tool_calls.append(ToolCall(
                step=step_counter,
                tool="optimize_addie",
                args={"iteration": iteration},
                result="ADDIE output optimization complete",
                timestamp=opt_start,
                duration_ms=int((opt_end - opt_start).total_seconds() * 1000),
                output_data={"optimized": True, "feedback_score": feedback.score},
                feedback={"improvement_areas": feedback.weaknesses[:3]} if feedback.weaknesses else None,
            ))

        # 최고 점수 버전으로 복원 (점수가 떨어진 경우 대비)
        addie_output = best_output
        trajectory.reasoning_steps.append(
            f"Using highest-scoring version: {best_score:.1f} points"
        )

        # 최종 평가
        final_eval_start = datetime.now()
        final_feedback = self.evaluator.run(
            addie_output=addie_output,
            learner_profile=learner_profile,
            scenario_context=scenario_context,
        )
        final_eval_end = datetime.now()
        step_counter += 1

        trajectory.tool_calls.append(ToolCall(
            step=step_counter,
            tool="final_evaluate_addie",
            args={"type": "final"},
            result=f"Final CIDPP evaluation: {final_feedback.score:.1f} points",
            timestamp=final_eval_start,
            duration_ms=int((final_eval_end - final_eval_start).total_seconds() * 1000),
        ))

        trajectory.reasoning_steps.append(
            f"Final score: {final_feedback.score:.1f}/100"
        )

        # 메타데이터 생성
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        metadata = Metadata(
            model=self.config.model,
            total_tokens=total_tokens,
            execution_time_seconds=execution_time,
            agent_version="0.1.0",
            iterations=len([
                tc for tc in trajectory.tool_calls
                if tc.tool == "optimize_addie"
            ]),
        )

        # Final fallback: slide_contents가 없는 경우 자동 생성
        if addie_output and addie_output.development:
            materials = addie_output.development.materials or []
            # modules는 development.lesson_plan에 있음
            lesson_plan = addie_output.development.lesson_plan
            modules = lesson_plan.modules if lesson_plan else []
            has_slide_contents = any(
                mat.slide_contents for mat in materials if mat
            )
            if not has_slide_contents and modules:
                # learning_objectives 가져오기
                learning_objectives = addie_output.design.learning_objectives if addie_output.design else []
                fallback_slides = self._generate_fallback_slides(modules, scenario_input, learning_objectives)
                if fallback_slides:
                    from eduplanner.models.schemas import Material
                    new_material = Material(
                        type="Presentation",
                        title="Training Slides",
                        description="Slides auto-generated from module information",
                        slides=len(fallback_slides),
                        slide_contents=fallback_slides,
                    )
                    addie_output.development.materials.append(new_material)

        return AgentResult(
            scenario_id=scenario_input.scenario_id,
            agent_id="eduplanner",
            timestamp=end_time,
            addie_output=addie_output,
            trajectory=trajectory,
            metadata=metadata,
        )

    def _build_scenario_context(self, scenario_input: ScenarioInput) -> str:
        """시나리오 컨텍스트 문자열 생성"""
        # target_audience 추론: 기존 필드가 없으면 IDLD 필드에서 조합
        target_audience = scenario_input.context.target_audience
        if not target_audience:
            # IDLD 시나리오 필드에서 조합
            parts_audience = []
            if scenario_input.context.learner_age:
                parts_audience.append(scenario_input.context.learner_age)
            if scenario_input.context.learner_role:
                parts_audience.append(scenario_input.context.learner_role)
            if scenario_input.context.learner_education:
                parts_audience.append(f"({scenario_input.context.learner_education})")
            target_audience = " ".join(parts_audience) if parts_audience else "Learners"

        parts = [
            f"**Title:** {scenario_input.title}",
            f"**Target Audience:** {target_audience}",
            f"**Duration:** {scenario_input.context.duration or 'TBD'}",
            f"**Environment:** {scenario_input.context.learning_environment or 'TBD'}",
            f"**Goals:** {', '.join(scenario_input.learning_goals)}",
        ]

        if scenario_input.context.prior_knowledge:
            parts.append(f"**Prior Knowledge:** {scenario_input.context.prior_knowledge}")

        if scenario_input.context.class_size:
            parts.append(f"**Class Size:** {scenario_input.context.class_size}")

        if scenario_input.context.institution_type:
            parts.append(f"**Institution Type:** {scenario_input.context.institution_type}")

        if scenario_input.constraints:
            if scenario_input.constraints.budget:
                parts.append(f"**Budget:** {scenario_input.constraints.budget}")
            if scenario_input.constraints.resources:
                parts.append(f"**Resources:** {', '.join(scenario_input.constraints.resources)}")
            if scenario_input.constraints.tech_requirements:
                parts.append(f"**Technical Requirements:** {scenario_input.constraints.tech_requirements}")

        return "\n".join(parts)

    def _generate_initial_output(
        self,
        scenario_input: ScenarioInput,
        learner_profile: LearnerProfile,
    ) -> ADDIEOutput:
        """
        ADDIE 산출물을 순차적 파이프라인으로 생성

        각 단계의 출력이 다음 단계의 입력으로 사용됩니다:
        Analysis → Design → Development → Implementation → Evaluation
        """
        import json

        if self.debug:
            print("\n" + "="*60)
            print("[Sequential ADDIE Pipeline] Starting sequential generation")
            print("="*60)

        # 시나리오 컨텍스트 (모든 단계에서 공통 사용)
        scenario_context = self._build_scenario_context(scenario_input)

        # ============================================================
        # Step 1: Analysis 단계
        # ============================================================
        if self.debug:
            print("\n[Step 1/5] Generating Analysis stage...")

        analysis_prompt = f"""## Scenario Information
{scenario_context}

## Learner Profile
{learner_profile.skill_tree.to_prompt_context()}

Perform the Analysis stage for the scenario above."""

        analysis_response = self.llm.invoke([
            SystemMessage(content=ANALYSIS_PROMPT),
            HumanMessage(content=analysis_prompt),
        ])
        analysis_data = self._parse_json_response(analysis_response.content)

        if self.debug:
            print(f"  → Analysis complete: characteristics={len(analysis_data.get('learner_analysis', {}).get('characteristics', []))}")

        # ============================================================
        # Step 2: Design 단계 (Analysis 결과 입력)
        # ============================================================
        if self.debug:
            print("\n[Step 2/5] Generating Design stage...")

        design_prompt = f"""## Scenario Information
{scenario_context}

## Previous Stage Result: Analysis
```json
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}
```

Based on the Analysis result above, perform the Design stage."""

        design_response = self.llm.invoke([
            SystemMessage(content=DESIGN_PROMPT),
            HumanMessage(content=design_prompt),
        ])
        design_data = self._parse_json_response(design_response.content)

        if self.debug:
            print(f"  → Design complete: objectives={len(design_data.get('learning_objectives', []))}, events={len(design_data.get('instructional_strategy', {}).get('sequence', []))}")

        # ============================================================
        # Step 3: Development 단계 (Analysis + Design 결과 입력)
        # ============================================================
        if self.debug:
            print("\n[Step 3/5] Generating Development stage...")

        development_prompt = f"""## Scenario Information
{scenario_context}

## Previous Stage Result: Analysis
```json
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}
```

## Previous Stage Result: Design
```json
{json.dumps(design_data, ensure_ascii=False, indent=2)}
```

Based on the results above, perform the Development stage.
Link the learning_objectives IDs from Design to the objectives of the modules."""

        development_response = self.llm.invoke([
            SystemMessage(content=DEVELOPMENT_PROMPT),
            HumanMessage(content=development_prompt),
        ])
        development_data = self._parse_json_response(development_response.content)

        if self.debug:
            print(f"  → Development complete: modules={len(development_data.get('lesson_plan', {}).get('modules', []))}, materials={len(development_data.get('materials', []))}")

        # ============================================================
        # Step 4: Implementation 단계 (이전 단계 결과 입력)
        # ============================================================
        if self.debug:
            print("\n[Step 4/5] Generating Implementation stage...")

        implementation_prompt = f"""## Scenario Information
{scenario_context}

## Summary of Previous Stage Results

### Analysis
- Target audience: {analysis_data.get('learner_analysis', {}).get('target_audience', '')}
- Environment: {analysis_data.get('context_analysis', {}).get('environment', '')}
- Duration: {analysis_data.get('context_analysis', {}).get('duration', '')}

### Design
- Learning objectives: {len(design_data.get('learning_objectives', []))}
- Instructional strategy: {design_data.get('instructional_strategy', {}).get('model', '')}

### Development - Lesson Plan
```json
{json.dumps(development_data.get('lesson_plan', {}), ensure_ascii=False, indent=2)}
```

Based on the results above, perform the Implementation stage.

⚠️ **Important**: facilitator_guide and learner_guide must be written in detail!
- facilitator_guide: 200+ characters, include step numbering and time allocation
- learner_guide: 150+ characters, separated into before/during/after learning"""

        implementation_response = self.llm.invoke([
            SystemMessage(content=IMPLEMENTATION_PROMPT),
            HumanMessage(content=implementation_prompt),
        ])
        implementation_data = self._parse_json_response(implementation_response.content)

        if self.debug:
            fg_len = len(implementation_data.get('facilitator_guide', ''))
            lg_len = len(implementation_data.get('learner_guide', ''))
            print(f"  → Implementation complete: facilitator_guide={fg_len} chars, learner_guide={lg_len} chars")

        # ============================================================
        # Step 5: Evaluation 단계 (이전 단계 결과 입력)
        # ============================================================
        if self.debug:
            print("\n[Step 5/5] Generating Evaluation stage...")

        evaluation_prompt = f"""## Scenario Information
{scenario_context}

## Previous Stage Result: Design - Learning Objectives
```json
{json.dumps(design_data.get('learning_objectives', []), ensure_ascii=False, indent=2)}
```

Perform the Evaluation stage in line with the learning objectives above.
The objective_id of each quiz_item must be linked to the id of the learning_objectives above."""

        evaluation_response = self.llm.invoke([
            SystemMessage(content=EVALUATION_PROMPT),
            HumanMessage(content=evaluation_prompt),
        ])
        evaluation_data = self._parse_json_response(evaluation_response.content)

        if self.debug:
            print(f"  → Evaluation complete: quiz_items={len(evaluation_data.get('quiz_items', []))}")
            print("\n" + "="*60)
            print("[Sequential ADDIE Pipeline] Generation complete!")
            print("="*60 + "\n")

        # ============================================================
        # 최종 ADDIEOutput 조립
        # ============================================================
        return self._assemble_addie_output(
            scenario_input=scenario_input,
            analysis_data=analysis_data,
            design_data=design_data,
            development_data=development_data,
            implementation_data=implementation_data,
            evaluation_data=evaluation_data,
        )

    def _parse_json_response(self, response_text: str) -> dict:
        """LLM 응답에서 JSON 파싱"""
        import json
        import re

        # JSON 블록 추출
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else response_text

        try:
            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 빈 딕셔너리 반환
            return {}

    def _assemble_addie_output(
        self,
        scenario_input: ScenarioInput,
        analysis_data: dict,
        design_data: dict,
        development_data: dict,
        implementation_data: dict,
        evaluation_data: dict,
    ) -> ADDIEOutput:
        """각 단계 데이터를 ADDIEOutput으로 조립"""

        # Analysis
        la_data = analysis_data.get("learner_analysis", {})
        ca_data = analysis_data.get("context_analysis", {})
        ta_data = analysis_data.get("task_analysis", {})
        na_data = analysis_data.get("needs_analysis", {})

        # target_audience 추론
        target_audience = la_data.get("target_audience") or scenario_input.context.target_audience
        if not target_audience:
            parts_audience = []
            if scenario_input.context.learner_age:
                parts_audience.append(scenario_input.context.learner_age)
            if scenario_input.context.learner_role:
                parts_audience.append(scenario_input.context.learner_role)
            target_audience = " ".join(parts_audience) if parts_audience else "Learners"

        # NeedsAnalysis (Item 1-4)
        needs_analysis = None
        if na_data:
            needs_analysis = NeedsAnalysis(
                problem_definition=na_data.get("problem_definition"),
                gap_analysis=na_data.get("gap_analysis"),
                performance_analysis=na_data.get("performance_analysis"),
                needs_prioritization=na_data.get("needs_prioritization"),
            )

        analysis = Analysis(
            learner_analysis=LearnerAnalysis(
                target_audience=target_audience,
                characteristics=la_data.get("characteristics", []),
                prior_knowledge=la_data.get("prior_knowledge", scenario_input.context.prior_knowledge),
                learning_preferences=la_data.get("learning_preferences", []),
                motivation=la_data.get("motivation"),
                challenges=la_data.get("challenges", []),
            ),
            context_analysis=ContextAnalysis(
                environment=ca_data.get("environment", scenario_input.context.learning_environment or "TBD"),
                duration=ca_data.get("duration", scenario_input.context.duration or "TBD"),
                constraints=ca_data.get("constraints", []),
                resources=ca_data.get("resources", []),
                technical_requirements=ca_data.get("technical_requirements", []),
                # Item 6: 물리/조직/기술 환경 상세
                physical_environment=ca_data.get("physical_environment"),
                organizational_environment=ca_data.get("organizational_environment"),
                technology_environment=ca_data.get("technology_environment"),
            ),
            task_analysis=TaskAnalysis(
                main_topics=ta_data.get("main_topics", scenario_input.learning_goals),
                subtopics=ta_data.get("subtopics", []),
                prerequisites=ta_data.get("prerequisites", []),
                # Item 7-10: 과제 및 목표분석
                initial_learning_objectives=ta_data.get("initial_learning_objectives"),
                sub_skills=ta_data.get("sub_skills", []),
                entry_behaviors=ta_data.get("entry_behaviors"),
                task_analysis_review=ta_data.get("task_analysis_review"),
            ),
            # Item 1-4: 요구분석
            needs_analysis=needs_analysis,
        )

        # Design
        objectives = []
        for i, obj in enumerate(design_data.get("learning_objectives", [])):
            objectives.append(LearningObjective(
                id=obj.get("id", f"OBJ-{i+1:02d}"),
                level=obj.get("level", "Understand"),
                statement=obj.get("statement", ""),
                bloom_verb=obj.get("bloom_verb", "explain"),
                measurable=obj.get("measurable", True),
            ))

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

        assessment_data = design_data.get("assessment_plan", {})
        prototype_data = design_data.get("prototype_design", {})

        # PrototypeDesign (Item 18)
        prototype_design = None
        if prototype_data:
            prototype_design = PrototypeDesign(
                storyboard=prototype_data.get("storyboard"),
                screen_flow=prototype_data.get("screen_flow", []),
                navigation_structure=prototype_data.get("navigation_structure"),
            )

        design = Design(
            learning_objectives=objectives,
            assessment_plan=AssessmentPlan(
                diagnostic=assessment_data.get("diagnostic", []),
                formative=assessment_data.get("formative", []),
                summative=assessment_data.get("summative", []),
            ),
            instructional_strategy=InstructionalStrategy(
                model=strategy_data.get("model", "Gagné's 9 Events"),
                sequence=events,
                methods=strategy_data.get("methods", []),
                # Item 14-17: 교수전략 확장
                instructional_strategies=strategy_data.get("instructional_strategies"),
                non_instructional_strategies=strategy_data.get("non_instructional_strategies"),
                media_selection=strategy_data.get("media_selection", []),
            ),
            # Item 18: 프로토타입 구조 설계
            prototype_design=prototype_design,
        )

        # Development
        lesson_data = development_data.get("lesson_plan", {})
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

        materials = []
        for mat in development_data.get("materials", []):
            slide_contents_data = mat.get("slide_contents", [])
            slide_contents = None
            if slide_contents_data:
                slide_contents = [
                    SlideContent(
                        slide_number=sc.get("slide_number", i + 1),
                        title=sc.get("title", ""),
                        bullet_points=sc.get("bullet_points", []),
                        speaker_notes=sc.get("speaker_notes"),
                    )
                    for i, sc in enumerate(slide_contents_data)
                ]

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
                slide_contents=slide_contents,
            ))

        development = Development(
            lesson_plan=LessonPlan(
                total_duration=lesson_data.get("total_duration", scenario_input.context.duration or "TBD"),
                modules=modules,
            ),
            materials=materials,
            # Item 20-23: 개발 단계 확장
            facilitator_manual=development_data.get("facilitator_manual"),
            operator_manual=development_data.get("operator_manual"),
            assessment_tools=development_data.get("assessment_tools", []),
            expert_review_plan=development_data.get("expert_review_plan"),
        )

        # Implementation
        implementation = Implementation(
            delivery_method=implementation_data.get("delivery_method", "In-person training"),
            facilitator_guide=implementation_data.get("facilitator_guide"),
            learner_guide=implementation_data.get("learner_guide"),
            technical_requirements=implementation_data.get("technical_requirements", []),
            support_plan=implementation_data.get("support_plan"),
            # Item 24-27: 실행 단계 확장
            orientation_plan=implementation_data.get("orientation_plan"),
            system_check_plan=implementation_data.get("system_check_plan"),
            pilot_execution_plan=implementation_data.get("pilot_execution_plan"),
            monitoring_plan=implementation_data.get("monitoring_plan"),
        )

        # Evaluation
        quiz_items = []
        for i, item in enumerate(evaluation_data.get("quiz_items", [])):
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

        rubric_data = evaluation_data.get("rubric")
        rubric = None
        if rubric_data:
            rubric = Rubric(
                criteria=rubric_data.get("criteria", []),
                levels=rubric_data.get("levels", {}),
            )

        evaluation = Evaluation(
            quiz_items=quiz_items,
            rubric=rubric,
            feedback_plan=evaluation_data.get("feedback_plan"),
            # Item 28-33: 형성평가/총괄평가/프로그램 개선 필드
            pilot_data_collection=evaluation_data.get("pilot_data_collection"),
            formative_improvement=evaluation_data.get("formative_improvement"),
            summative_evaluation_plan=evaluation_data.get("summative_evaluation_plan"),
            adoption_decision_criteria=evaluation_data.get("adoption_decision_criteria"),
            program_improvement=evaluation_data.get("program_improvement"),
        )

        return ADDIEOutput(
            analysis=analysis,
            design=design,
            development=development,
            implementation=implementation,
            evaluation=evaluation,
        )

    def _validate_minimum_requirements(self, addie_output: ADDIEOutput) -> tuple[int, list[str]]:
        """최소 요구사항 검증 (0-100 점수 반환)"""
        issues = []
        score = 100

        # Analysis 검증
        la = addie_output.analysis.learner_analysis
        if len(la.characteristics) < 5:
            issues.append(f"characteristics: {len(la.characteristics)}/5")
            score -= 10
        if len(la.learning_preferences) < 4:
            issues.append(f"learning_preferences: {len(la.learning_preferences)}/4")
            score -= 5
        if len(la.challenges) < 3:
            issues.append(f"challenges: {len(la.challenges)}/3")
            score -= 5

        # Design 검증
        d = addie_output.design
        if len(d.learning_objectives) < 5:
            issues.append(f"learning_objectives: {len(d.learning_objectives)}/5")
            score -= 20  # learning objectives are important
        if len(d.instructional_strategy.sequence) < 9:
            issues.append(f"instructional_strategy.sequence: {len(d.instructional_strategy.sequence)}/9")
            score -= 15

        # Development 검증
        dev = addie_output.development
        if len(dev.lesson_plan.modules) < 3:
            issues.append(f"modules: {len(dev.lesson_plan.modules)}/3")
            score -= 10
        if len(dev.materials) < 5:
            issues.append(f"materials: {len(dev.materials)}/5")
            score -= 5

        # Evaluation 검증
        ev = addie_output.evaluation
        if len(ev.quiz_items) < 10:
            issues.append(f"quiz_items: {len(ev.quiz_items)}/10")
            score -= 10

        # Implementation 검증 (가이드 길이)
        impl = addie_output.implementation
        fg_len = len(str(impl.facilitator_guide or ""))
        lg_len = len(str(impl.learner_guide or ""))
        if fg_len < 200:
            issues.append(f"facilitator_guide: {fg_len}/200 chars")
            score -= 15  # important item
        if lg_len < 100:
            issues.append(f"learner_guide: {lg_len}/100 chars")
            score -= 10

        return max(0, score), issues

    def _build_generation_prompt(
        self,
        scenario_input: ScenarioInput,
        learner_profile: LearnerProfile,
    ) -> str:
        """생성 프롬프트 구성"""
        parts = [
            "Please generate an ADDIE instructional design output that fits the following scenario.\n",
            "## Scenario Information",
            f"**Title:** {scenario_input.title}",
            f"**Scenario ID:** {scenario_input.scenario_id}",
            f"\n**Learning Context:**",
            f"- Target audience: {scenario_input.context.target_audience}",
            f"- Duration: {scenario_input.context.duration}",
            f"- Environment: {scenario_input.context.learning_environment}",
        ]

        if scenario_input.context.prior_knowledge:
            parts.append(f"- Prior knowledge: {scenario_input.context.prior_knowledge}")

        if scenario_input.context.class_size:
            parts.append(f"- Class size: {scenario_input.context.class_size} learners")

        parts.append(f"\n**Learning Goals:**")
        for goal in scenario_input.learning_goals:
            parts.append(f"- {goal}")

        if scenario_input.constraints:
            parts.append("\n**Constraints:**")
            if scenario_input.constraints.budget:
                parts.append(f"- Budget: {scenario_input.constraints.budget}")
            if scenario_input.constraints.resources:
                parts.append(f"- Available resources: {', '.join(scenario_input.constraints.resources)}")
            if scenario_input.constraints.accessibility:
                parts.append(f"- Accessibility: {', '.join(scenario_input.constraints.accessibility)}")

        # 학습자 프로필
        parts.append("\n" + learner_profile.skill_tree.to_prompt_context())

        parts.append("\nBased on the information above, please generate a complete ADDIE output.")

        return "\n".join(parts)

    def _parse_addie_response(
        self,
        response_text: str,
        scenario_input: ScenarioInput
    ) -> ADDIEOutput:
        """LLM 응답을 ADDIEOutput으로 파싱"""
        import json
        import re

        # JSON 블록 추출
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            return self._dict_to_addie_output(data, scenario_input)

        # JSON 블록 없이 직접 파싱 시도
        try:
            data = json.loads(response_text)
            return self._dict_to_addie_output(data, scenario_input)
        except json.JSONDecodeError:
            return self._create_default_output(scenario_input)

    def _dict_to_addie_output(
        self,
        data: dict,
        scenario_input: ScenarioInput
    ) -> ADDIEOutput:
        """딕셔너리를 ADDIEOutput으로 변환"""
        # Analysis
        analysis_data = data.get("analysis", {})
        analysis = Analysis(
            learner_analysis=LearnerAnalysis(
                target_audience=scenario_input.context.target_audience,
                characteristics=analysis_data.get("learner_analysis", {}).get("characteristics", []),
                prior_knowledge=scenario_input.context.prior_knowledge,
                learning_preferences=analysis_data.get("learner_analysis", {}).get("learning_preferences", []),
                motivation=analysis_data.get("learner_analysis", {}).get("motivation"),
                challenges=analysis_data.get("learner_analysis", {}).get("challenges", []),
            ),
            context_analysis=ContextAnalysis(
                environment=scenario_input.context.learning_environment,
                duration=scenario_input.context.duration,
                constraints=analysis_data.get("context_analysis", {}).get("constraints", []),
                resources=analysis_data.get("context_analysis", {}).get("resources", []),
                technical_requirements=analysis_data.get("context_analysis", {}).get("technical_requirements", []),
            ),
            task_analysis=TaskAnalysis(
                main_topics=analysis_data.get("task_analysis", {}).get("main_topics", scenario_input.learning_goals),
                subtopics=analysis_data.get("task_analysis", {}).get("subtopics", []),
                prerequisites=analysis_data.get("task_analysis", {}).get("prerequisites", []),
            ),
        )

        # Design
        design_data = data.get("design", {})
        objectives = []
        for i, obj in enumerate(design_data.get("learning_objectives", [])):
            objectives.append(LearningObjective(
                id=f"OBJ-{i+1:02d}",
                level=obj.get("level", "Understand"),
                statement=obj.get("statement", ""),
                bloom_verb=obj.get("bloom_verb", "explain"),
                measurable=obj.get("measurable", True),
            ))

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

        design = Design(
            learning_objectives=objectives,
            assessment_plan=AssessmentPlan(
                formative=design_data.get("assessment_plan", {}).get("formative", []),
                summative=design_data.get("assessment_plan", {}).get("summative", []),
                diagnostic=design_data.get("assessment_plan", {}).get("diagnostic", []),
            ),
            instructional_strategy=InstructionalStrategy(
                model=strategy_data.get("model", "Gagné's 9 Events"),
                sequence=events,
                methods=strategy_data.get("methods", []),
            ),
        )

        # Development
        dev_data = data.get("development", {})
        modules = []
        for mod in dev_data.get("lesson_plan", {}).get("modules", []):
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

        materials = []
        for mat in dev_data.get("materials", []):
            # slide_contents 파싱
            slide_contents_data = mat.get("slide_contents", [])
            slide_contents = None
            if slide_contents_data:
                slide_contents = [
                    SlideContent(
                        slide_number=sc.get("slide_number", i + 1),
                        title=sc.get("title", ""),
                        bullet_points=sc.get("bullet_points", []),
                        speaker_notes=sc.get("speaker_notes"),
                        visual_suggestion=sc.get("visual_suggestion"),
                    )
                    for i, sc in enumerate(slide_contents_data)
                ]

            # Fallback: 프레젠테이션인데 slide_contents가 없으면 자동 생성
            mat_type = mat.get("type", "").lower()
            if not slide_contents and ("presentation" in mat_type or "slide" in mat_type or "presentation" in mat_type):
                slide_contents = self._generate_fallback_slides(modules, scenario_input, objectives)

            # slides/pages가 숫자 문자열인 경우 정수로 변환
            slides_val = mat.get("slides")
            if slides_val is None and slide_contents:
                slides_val = len(slide_contents)
            elif isinstance(slides_val, str):
                try:
                    slides_val = int(slides_val)
                except ValueError:
                    slides_val = len(slide_contents) if slide_contents else None

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
                slide_contents=slide_contents,
            ))

        # Fallback: materials 처리 후 slide_contents가 있는 material이 없으면 자동 생성
        has_slide_contents = any(mat.slide_contents for mat in materials)
        if not has_slide_contents and modules:
            fallback_slides = self._generate_fallback_slides(modules, scenario_input, objectives)
            if fallback_slides:
                materials.append(Material(
                    type="Presentation",
                    title="Training Slides",
                    description="Slides auto-generated from module information",
                    slides=len(fallback_slides),
                    slide_contents=fallback_slides,
                ))

        development = Development(
            lesson_plan=LessonPlan(
                total_duration=scenario_input.context.duration,
                modules=modules,
            ),
            materials=materials,
        )

        # Implementation
        impl_data = data.get("implementation", {})
        implementation = Implementation(
            delivery_method=impl_data.get("delivery_method", "In-person training"),
            facilitator_guide=impl_data.get("facilitator_guide"),
            learner_guide=impl_data.get("learner_guide"),
            technical_requirements=impl_data.get("technical_requirements", []),
            support_plan=impl_data.get("support_plan"),
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
                id=f"Q-{i+1:02d}",
                question=item.get("question", ""),
                type=item.get("type", "multiple_choice"),
                options=options_val,
                answer=answer_val,
                explanation=item.get("explanation"),
                objective_id=item.get("objective_id"),
                difficulty=item.get("difficulty"),
            ))

        rubric_data = eval_data.get("rubric")
        rubric = None
        if rubric_data:
            rubric = Rubric(
                criteria=rubric_data.get("criteria", []),
                levels=rubric_data.get("levels", {}),
            )

        evaluation = Evaluation(
            quiz_items=quiz_items,
            rubric=rubric,
            feedback_plan=eval_data.get("feedback_plan"),
            # Item 28-33: 형성평가/총괄평가/프로그램 개선 필드
            pilot_data_collection=eval_data.get("pilot_data_collection"),
            formative_improvement=eval_data.get("formative_improvement"),
            summative_evaluation_plan=eval_data.get("summative_evaluation_plan"),
            adoption_decision_criteria=eval_data.get("adoption_decision_criteria"),
            program_improvement=eval_data.get("program_improvement"),
        )

        return ADDIEOutput(
            analysis=analysis,
            design=design,
            development=development,
            implementation=implementation,
            evaluation=evaluation,
        )

    def _generate_fallback_slides(
        self,
        modules: list[Module],
        scenario_input: ScenarioInput,
        learning_objectives: list[LearningObjective] = None,
    ) -> list[SlideContent]:
        """모듈 정보를 기반으로 폴백 슬라이드 콘텐츠 생성"""
        slide_contents = []
        slide_num = 1

        # objective ID → statement 매핑 생성
        obj_id_to_statement = {}
        if learning_objectives:
            for obj in learning_objectives:
                obj_id_to_statement[obj.id] = obj.statement

        def resolve_objective(obj_ref: str) -> str:
            """objective ID를 실제 statement로 변환, 없으면 원본 반환"""
            if obj_ref in obj_id_to_statement:
                return obj_id_to_statement[obj_ref]
            # OBJ-xx 패턴이면 매핑에서 찾기
            if obj_ref.startswith("OBJ-"):
                return obj_id_to_statement.get(obj_ref, obj_ref)
            return obj_ref

        # 도입 슬라이드
        slide_contents.append(SlideContent(
            slide_number=slide_num,
            title="Training Introduction",
            bullet_points=["Welcome", "Training objectives", "Schedule overview"],
            speaker_notes="Welcome the participants and clearly convey the training objectives.",
        ))
        slide_num += 1

        # 학습 목표 슬라이드
        if scenario_input.learning_goals:
            slide_contents.append(SlideContent(
                slide_number=slide_num,
                title="Learning Objectives",
                bullet_points=scenario_input.learning_goals[:5],
                speaker_notes="Explain the objectives to be achieved through today's learning.",
            ))
            slide_num += 1

        # 모듈별 슬라이드 생성
        for module in modules:
            module_title = module.title if module.title else "Learning Module"

            # 모듈 시작 슬라이드 - objective ID를 실제 statement로 변환
            raw_objectives = module.objectives[:3] if module.objectives else []
            resolved_objectives = [resolve_objective(obj) for obj in raw_objectives] if raw_objectives else ["Learning objectives", "Key content"]

            bullet_points = resolved_objectives + [f"Estimated duration: {module.duration}"] if module.duration else resolved_objectives
            slide_contents.append(SlideContent(
                slide_number=slide_num,
                title=module_title,
                bullet_points=bullet_points,
                speaker_notes=f"Explain the learning objectives and overview of {module_title}.",
            ))
            slide_num += 1

            # 활동별 슬라이드 (최대 3개)
            activities = module.activities if module.activities else []
            for activity in activities[:3]:
                activity_name = activity.activity if activity.activity else "Learning Activity"
                description = activity.description if activity.description else ""
                bullet_points = [description] if description else ["Activity description"]

                # 자원 정보 추가
                if activity.resources:
                    bullet_points.extend([f"Resource: {r}" for r in activity.resources[:2]])

                slide_contents.append(SlideContent(
                    slide_number=slide_num,
                    title=activity_name,
                    bullet_points=bullet_points,
                    speaker_notes=f"Explain how to carry out {activity_name}.",
                ))
                slide_num += 1

        # 마무리 슬라이드
        slide_contents.append(SlideContent(
            slide_number=slide_num,
            title="Wrap-up and Q&A",
            bullet_points=["Summary of today's learning content", "Recap of key points", "Q&A"],
            speaker_notes="Summarize the key content and take questions.",
        ))

        return slide_contents

    def _create_default_output(self, scenario_input: ScenarioInput) -> ADDIEOutput:
        """기본 ADDIE 산출물 생성 (파싱 실패 시)"""
        return ADDIEOutput(
            analysis=Analysis(
                learner_analysis=LearnerAnalysis(
                    target_audience=scenario_input.context.target_audience,
                    characteristics=[],
                    prior_knowledge=scenario_input.context.prior_knowledge,
                ),
                context_analysis=ContextAnalysis(
                    environment=scenario_input.context.learning_environment,
                    duration=scenario_input.context.duration,
                ),
                task_analysis=TaskAnalysis(
                    main_topics=scenario_input.learning_goals,
                ),
            ),
            design=Design(
                learning_objectives=[
                    LearningObjective(
                        id="OBJ-01",
                        level="Understand",
                        statement=scenario_input.learning_goals[0] if scenario_input.learning_goals else "",
                        bloom_verb="explain",
                    )
                ],
                assessment_plan=AssessmentPlan(),
                instructional_strategy=InstructionalStrategy(),
            ),
            development=Development(
                lesson_plan=LessonPlan(
                    total_duration=scenario_input.context.duration,
                ),
            ),
            implementation=Implementation(
                delivery_method="In-person training",
            ),
            evaluation=Evaluation(),
        )
