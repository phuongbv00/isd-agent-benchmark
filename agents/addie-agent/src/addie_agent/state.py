"""
ADDIE Agent State Schema

ADDIE 에이전트의 상태를 정의하는 TypedDict 스키마입니다.
LangGraph StateGraph에서 사용됩니다.
"""

from typing import TypedDict, Optional, List, Any, Literal
from datetime import datetime


# ADDIE 단계 타입
ADDIEPhase = Literal["analysis", "design", "development", "implementation", "evaluation", "complete"]


class ToolCall(TypedDict, total=False):
    """도구 호출 기록 (trajectory_schema.json 준수)"""
    step: int                    # Call order
    tool: str                    # Tool name
    args: dict                   # Tool arguments
    result: str                  # Execution result summary
    result_detail: dict          # Execution result detail
    timestamp: str               # Call timestamp (ISO 8601)
    duration_ms: int             # Execution time (milliseconds)
    tokens_used: int             # Number of tokens used
    success: bool                # Success flag
    error: str                   # Error message


class LearnerAnalysis(TypedDict, total=False):
    """학습자 분석 결과"""
    target_audience: str
    characteristics: List[str]   # at least 5
    prior_knowledge: str
    learning_preferences: List[str]
    motivation: str              # 2-3 sentences
    challenges: List[str]        # at least 3


class ContextAnalysis(TypedDict, total=False):
    """환경 분석 결과"""
    environment: str
    duration: str
    constraints: List[str]
    resources: List[str]
    technical_requirements: List[str]


class TaskAnalysis(TypedDict, total=False):
    """과제 분석 결과"""
    main_topics: List[str]       # at least 3
    subtopics: List[str]         # at least 6
    prerequisites: List[str]
    knowledge_structure: dict
    review_summary: str          # A-10: Review/organize task analysis results


class NeedsAnalysis(TypedDict, total=False):
    """요구분석 결과 (analyze_needs 반환값)"""
    gap_analysis: List[str]
    root_causes: List[str]
    training_needs: List[str]
    non_training_solutions: List[str]
    priority: str
    priority_matrix: dict        # A-4: Needs prioritization decision extension
    recommendation: str


class AnalysisResult(TypedDict, total=False):
    """Analysis 단계 산출물"""
    needs_analysis: NeedsAnalysis
    learner_analysis: LearnerAnalysis
    context_analysis: ContextAnalysis
    task_analysis: TaskAnalysis


class LearningObjective(TypedDict, total=False):
    """학습 목표"""
    id: str                      # LO-001 format
    level: str                   # Bloom's Taxonomy level
    statement: str
    bloom_verb: str
    measurable: bool


class AssessmentPlan(TypedDict, total=False):
    """평가 계획"""
    diagnostic: List[str]        # at least 2
    formative: List[str]         # at least 2
    summative: List[str]         # at least 2


class InstructionalEvent(TypedDict, total=False):
    """Gagné's 9 Events 교수사태"""
    event: str
    activity: str
    duration: str
    resources: List[str]


class InstructionalStrategy(TypedDict, total=False):
    """교수 전략"""
    model: str                   # Gagné's 9 Events
    sequence: List[InstructionalEvent]  # 9 required
    methods: List[str]


class DesignResult(TypedDict, total=False):
    """Design 단계 산출물"""
    learning_objectives: List[LearningObjective]  # at least 5
    assessment_plan: AssessmentPlan
    instructional_strategy: InstructionalStrategy


class Activity(TypedDict, total=False):
    """학습 활동"""
    time: str
    activity: str
    description: str
    resources: List[str]


class Module(TypedDict, total=False):
    """레슨 모듈"""
    title: str
    duration: str
    objectives: List[str]        # LO-xxx reference
    activities: List[Activity]   # at least 3


class LessonPlan(TypedDict, total=False):
    """레슨 플랜"""
    total_duration: str
    modules: List[Module]        # at least 3


class SlideContent(TypedDict, total=False):
    """슬라이드 콘텐츠"""
    slide_number: int
    title: str
    bullet_points: List[str]
    speaker_notes: str
    visual_suggestion: str


class StoryboardFrame(TypedDict, total=False):
    """스토리보드 프레임"""
    frame_number: int
    screen_title: str
    visual_description: str
    audio_narration: str
    interaction: str
    notes: str


class Material(TypedDict, total=False):
    """학습 자료"""
    type: str                    # PPT, video, quiz, worksheet, etc.
    title: str
    description: str
    slides: int
    duration: str
    pages: int
    questions: int
    slide_contents: List[SlideContent]
    storyboard: List[StoryboardFrame]  # D-18: Storyboard / screen flow design


class DevelopmentResult(TypedDict, total=False):
    """Development 단계 산출물"""
    lesson_plan: LessonPlan
    materials: List[Material]    # at least 5


class MaintenancePlan(TypedDict, total=False):
    """유지관리 계획 (create_maintenance_plan 반환값)"""
    program_title: str
    maintenance_period: str
    content_maintenance: dict
    technical_maintenance: dict
    quality_assurance: dict
    version_control: dict
    support_resources: dict


class PilotPlan(TypedDict, total=False):
    """파일럿 실행 계획 (I-26)"""
    pilot_scope: str
    participants: str
    duration: str
    success_criteria: List[str]
    data_collection: List[str]
    contingency_plan: str


class ImplementationResult(TypedDict, total=False):
    """Implementation 단계 산출물"""
    delivery_method: str
    facilitator_guide: str       # 200+ characters
    learner_guide: str           # 200+ characters
    technical_requirements: List[str]  # 2 or more
    support_plan: str
    schedule: dict
    operator_guide: str          # Dev-21: Operator manual
    orientation_plan: str        # I-24: Instructor/operator orientation
    pilot_plan: PilotPlan        # I-26: Prototype execution
    maintenance_plan: MaintenancePlan  # Maintenance plan


class QuizItem(TypedDict, total=False):
    """퀴즈 문항"""
    id: str
    question: str
    type: str                    # multiple choice, short answer, true/false, etc.
    options: List[str]           # 4 if multiple choice
    answer: str
    explanation: str
    objective_id: str            # LO-xxx reference
    difficulty: str              # easy, medium, hard


class Rubric(TypedDict, total=False):
    """평가 루브릭"""
    criteria: List[str]          # at least 5
    levels: dict                 # excellent, good, needs_improvement


class AdoptionDecision(TypedDict, total=False):
    """프로그램 채택 여부 결정 (E-32)"""
    recommendation: str          # adopt, modify, reject
    rationale: str
    conditions: List[str]
    next_steps: List[str]


class ProgramEvaluation(TypedDict, total=False):
    """성과평가 계획 (create_program_evaluation 반환값)"""
    program_title: str
    evaluation_model: str        # Kirkpatrick 4-Level
    levels: dict                 # level_1_reaction, level_2_learning, etc.
    roi_calculation: dict
    evaluation_schedule: List[dict]
    success_criteria: dict
    adoption_decision: AdoptionDecision  # E-32: Program adoption decision


class EvaluationResult(TypedDict, total=False):
    """Evaluation 단계 산출물"""
    quiz_items: List[QuizItem]   # at least 10
    rubric: Rubric
    feedback_plan: str
    program_evaluation: ProgramEvaluation  # Program evaluation plan


class ScenarioContext(TypedDict, total=False):
    """시나리오 컨텍스트"""
    target_audience: str
    duration: str
    learning_environment: str
    prior_knowledge: str
    class_size: int
    additional_context: str


class ScenarioConstraints(TypedDict, total=False):
    """시나리오 제약조건"""
    budget: str
    resources: List[str]
    accessibility: List[str]


class ScenarioInput(TypedDict, total=False):
    """시나리오 입력"""
    scenario_id: str
    title: str
    context: ScenarioContext
    learning_goals: List[str]
    constraints: ScenarioConstraints
    domain: str
    difficulty: str


class Metadata(TypedDict, total=False):
    """메타데이터"""
    model: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    execution_time_seconds: float
    cost_usd: float
    agent_version: str
    tool_calls_count: int


class Trajectory(TypedDict, total=False):
    """에이전트 궤적"""
    tool_calls: List[ToolCall]
    reasoning_steps: List[str]


class ADDIEState(TypedDict, total=False):
    """
    ADDIE 에이전트 상태 스키마

    LangGraph StateGraph에서 사용되는 상태 정의입니다.
    각 단계의 산출물이 다음 단계의 입력으로 활용됩니다.
    """
    # 입력
    scenario: ScenarioInput

    # 단계별 산출물
    analysis_result: AnalysisResult
    design_result: DesignResult
    development_result: DevelopmentResult
    implementation_result: ImplementationResult
    evaluation_result: EvaluationResult

    # 상태 관리
    current_phase: ADDIEPhase
    errors: List[str]

    # 궤적 및 메타데이터
    tool_calls: List[ToolCall]
    reasoning_steps: List[str]
    metadata: Metadata


def create_initial_state(scenario: ScenarioInput) -> ADDIEState:
    """초기 상태 생성"""
    return ADDIEState(
        scenario=scenario,
        analysis_result={},
        design_result={},
        development_result={},
        implementation_result={},
        evaluation_result={},
        current_phase="analysis",
        errors=[],
        tool_calls=[],
        reasoning_steps=[],
        metadata={
            "agent_version": "0.1.0",
            "tool_calls_count": 0,
        },
    )
