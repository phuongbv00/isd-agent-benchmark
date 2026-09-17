"""
Dick & Carey Agent State Schema

Dick & Carey 모형의 10단계 체제적 교수설계 상태를 정의하는 TypedDict 스키마입니다.
LangGraph StateGraph에서 사용됩니다.

10단계:
1. 교수목적 설정 (Instructional Goal)
2. 교수분석 (Instructional Analysis)
3. 학습자/환경 분석 (Entry Behaviors & Context)
4. 수행목표 진술 (Performance Objectives)
5. 평가도구 개발 (Assessment Instruments)
6. 교수전략 개발 (Instructional Strategy)
7. 교수자료 개발 (Instructional Materials)
8. 형성평가 실시 (Formative Evaluation)
9. 교수프로그램 수정 (Revision)
10. 총괄평가 실시 (Summative Evaluation)

ADDIE 33개 소항목 완전성 확보:
- A-1 ~ A-10: Analysis 단계
- D-11 ~ D-18: Design 단계
- Dev-19 ~ Dev-23: Development 단계
- I-24 ~ I-27: Implementation 단계
- E-28 ~ E-33: Evaluation 단계
"""

from typing import TypedDict, Optional, List, Any, Literal
from datetime import datetime


# Dick & Carey 단계 타입
DickCareyPhase = Literal[
    "goal",                    # 1단계
    "instructional_analysis",  # 2단계
    "learner_context",         # 3단계
    "performance_objectives",  # 4단계
    "assessment_instruments",  # 5단계
    "instructional_strategy",  # 6단계
    "instructional_materials", # 7단계
    "formative_evaluation",    # 8단계
    "revision",                # 9단계
    "summative_evaluation",    # 10단계
    "complete",
]


class ToolCall(TypedDict, total=False):
    """도구 호출 기록 (trajectory_schema.json 준수)"""
    step: int                    # 호출 순서
    tool: str                    # 도구 이름
    args: dict                   # 도구 인자
    result: str                  # 실행 결과 요약
    result_detail: dict          # 실행 결과 상세
    timestamp: str               # 호출 시각 (ISO 8601)
    duration_ms: int             # 실행 시간 (밀리초)
    tokens_used: int             # 사용 토큰 수
    success: bool                # 성공 여부
    error: str                   # 오류 메시지


# ========== 1단계: 교수목적 설정 ==========
class NeedsAnalysis(TypedDict, total=False):
    """요구분석 결과 (A-1 ~ A-4)"""
    gap_analysis: List[str]           # A-1: 문제 확인 및 정의
    root_causes: List[str]            # A-3: 수행분석 (근본 원인)
    training_needs: List[str]         # 교육적 해결이 필요한 요구
    non_training_solutions: List[str]  # D-15: 비교수적 전략 수립을 위한 요구
    priority_matrix: dict             # A-4: 요구 우선순위 결정
    recommendation: str               # 최종 권고


class InstructionalGoal(TypedDict, total=False):
    """교수목적 설정 결과"""
    goal_statement: str          # 교수 목적 진술
    target_domain: str           # 목표 도메인 (인지적, 정의적, 심동적)
    current_state: str           # 현재 상태
    desired_state: str           # 목표 상태
    performance_gap: str         # 수행 격차
    needs_analysis: NeedsAnalysis  # A-1 ~ A-4: 요구분석


# ========== 2단계: 교수분석 ==========
class SubSkill(TypedDict, total=False):
    """하위 기능"""
    id: str                      # SS-001 형식
    description: str
    type: str                    # 지적 기능, 언어 정보, 인지 전략, 운동 기능, 태도
    prerequisites: List[str]     # 선수 기능 ID 목록


class InstructionalAnalysis(TypedDict, total=False):
    """교수분석 결과"""
    task_type: str               # 절차적, 위계적, 조합형, 군집형
    sub_skills: List[SubSkill]   # 하위 기능 목록 (최소 5개)
    skill_hierarchy: dict        # 기능 위계 구조
    entry_skills: List[str]      # 출발점 기능
    review_summary: str          # A-10: 과제분석 결과 검토/정리


# ========== 3단계: 학습자/환경 분석 ==========
class LearnerAnalysis(TypedDict, total=False):
    """학습자 분석 결과"""
    target_audience: str
    entry_behaviors: List[str]   # 출발점 행동 (최소 3개)
    prior_knowledge: str
    learning_preferences: List[str]
    motivation: str
    characteristics: List[str]   # 최소 5개


class ContextAnalysis(TypedDict, total=False):
    """환경 분석 결과"""
    performance_context: str     # 수행 환경 (학습 결과 적용 환경)
    learning_context: str        # 학습 환경
    constraints: List[str]
    resources: List[str]
    technical_requirements: List[str]


class LearnerContextResult(TypedDict, total=False):
    """3단계 통합 결과"""
    learner: LearnerAnalysis
    context: ContextAnalysis


# ========== 4단계: 수행목표 진술 ==========
class PerformanceObjective(TypedDict, total=False):
    """수행목표 (ABCD 형식)"""
    id: str                      # PO-001 형식
    audience: str                # A: 대상 학습자
    behavior: str                # B: 관찰 가능한 행동
    condition: str               # C: 수행 조건
    degree: str                  # D: 성취 기준
    statement: str               # 통합 진술문
    sub_skill_id: str            # 관련 하위 기능 ID
    bloom_level: str             # Bloom's Taxonomy 수준


class PerformanceObjectivesResult(TypedDict, total=False):
    """4단계 결과"""
    terminal_objective: PerformanceObjective  # 최종 수행목표
    enabling_objectives: List[PerformanceObjective]  # 가능 수행목표 (최소 5개)


# ========== 5단계: 평가도구 개발 ==========
class AssessmentItem(TypedDict, total=False):
    """평가 문항"""
    id: str
    objective_id: str            # 관련 수행목표 ID
    type: str                    # 객관식, 주관식, 수행평가, 관찰 등
    question: str
    options: List[str]           # 객관식 선택지
    answer: str
    rubric: str                  # 채점 기준


class AssessmentInstrumentsResult(TypedDict, total=False):
    """5단계 결과"""
    entry_test: List[AssessmentItem]       # 진단평가 (최소 3개)
    practice_tests: List[AssessmentItem]   # 연습평가 (최소 3개)
    post_test: List[AssessmentItem]        # 사후평가 (최소 5개)
    alignment_matrix: dict                 # 목표-평가 정렬 매트릭스


# ========== 6단계: 교수전략 개발 ==========
class PreInstructionalActivity(TypedDict, total=False):
    """교수 전 활동"""
    motivation: str              # 동기 유발
    objectives_info: str         # 목표 제시
    prerequisite_review: str     # 선수학습 확인


class ContentPresentation(TypedDict, total=False):
    """내용 제시 및 학습 안내"""
    sequence: List[str]          # 내용 제시 순서
    examples: List[str]          # 예시
    non_examples: List[str]      # 비예시
    practice_guidance: str       # 연습 안내


class LearnerParticipation(TypedDict, total=False):
    """학습자 참여"""
    practice_activities: List[str]  # 연습 활동 (최소 3개)
    feedback_strategy: str       # 피드백 전략


class Assessment(TypedDict, total=False):
    """평가 및 후속 활동"""
    assessment_strategy: str
    retention_transfer: str      # 파지 및 전이 활동


class NonInstructionalStrategy(TypedDict, total=False):
    """비교수적 전략 (D-15)"""
    strategies: List[str]        # 비교수적 해결 전략 목록
    rationale: str               # 비교수적 접근 이유
    implementation: str          # 실행 방안


class MediaSelection(TypedDict, total=False):
    """매체 선정과 활용 계획 (D-16)"""
    selected_media: List[str]    # 선정된 매체 목록
    selection_criteria: str      # 선정 기준
    utilization_plan: str        # 활용 계획


class ContentSelection(TypedDict, total=False):
    """교수 내용 선정 (D-13)"""
    core_content: List[str]      # 핵심 내용
    supplementary_content: List[str]  # 보충 내용
    selection_rationale: str     # 선정 근거


class InstructionalStrategyResult(TypedDict, total=False):
    """6단계 결과"""
    pre_instructional: PreInstructionalActivity
    content_presentation: ContentPresentation
    learner_participation: LearnerParticipation
    assessment: Assessment
    delivery_method: str         # 전달 방법
    grouping_strategy: str       # 집단 구성
    content_selection: ContentSelection  # D-13: 교수 내용 선정
    non_instructional_strategy: NonInstructionalStrategy  # D-15: 비교수적 전략
    media_selection: MediaSelection  # D-16: 매체 선정과 활용 계획


# ========== 7단계: 교수자료 개발 ==========
class StoryboardFrame(TypedDict, total=False):
    """스토리보드 프레임 (D-18)"""
    frame_number: int
    screen_title: str
    visual_description: str
    audio_narration: str
    interaction: str
    notes: str


class Material(TypedDict, total=False):
    """교수 자료"""
    type: str                    # 교수자 가이드, 학습자 자료, PPT, 동영상, 워크시트 등
    title: str
    description: str
    content_outline: List[str]   # 내용 개요
    pages: int
    duration: str
    storyboard: List[StoryboardFrame]  # D-18: 스토리보드/화면 흐름 설계


class SlideContent(TypedDict, total=False):
    """슬라이드 콘텐츠"""
    slide_number: int
    title: str
    bullet_points: List[str]
    speaker_notes: str
    visual_suggestion: str


class ExpertReview(TypedDict, total=False):
    """전문가 검토 (Dev-23)"""
    reviewer: str                # 검토자
    review_date: str             # 검토 일자
    review_areas: List[str]      # 검토 영역
    findings: List[str]          # 검토 결과
    recommendations: List[str]   # 권고사항
    approval_status: str         # 승인 상태


class InstructionalMaterialsResult(TypedDict, total=False):
    """7단계 결과"""
    instructor_guide: Material
    learner_materials: List[Material]  # 학습자 자료 (최소 3개)
    media_list: List[Material]         # 미디어 자료
    slide_contents: List[SlideContent] # PPT 슬라이드 (최소 10개)
    instructor_manual: str       # Dev-20: 교수자용 매뉴얼
    operator_manual: str         # Dev-21: 운영자용 매뉴얼
    expert_review: ExpertReview  # Dev-23: 전문가 검토


# ========== Implementation 관련 TypedDict (I-24 ~ I-27) ==========
class OrientationPlan(TypedDict, total=False):
    """교수자/운영자 오리엔테이션 (I-24)"""
    facilitator_orientation: str  # 교수자 오리엔테이션 계획
    operator_orientation: str     # 운영자 오리엔테이션 계획
    rehearsal_plan: str           # 리허설 계획
    schedule: List[str]           # 일정


class SystemCheck(TypedDict, total=False):
    """시스템/환경 점검 (I-25)"""
    checklist: List[str]         # 점검 체크리스트
    technical_tests: List[str]   # 기술적 테스트 항목
    contingency_plan: str        # 비상 대응 계획


class PilotPlan(TypedDict, total=False):
    """프로토타입 실행 계획 (I-26)"""
    pilot_scope: str             # 파일럿 범위
    participants: str            # 참가자
    duration: str                # 기간
    success_criteria: List[str]  # 성공 기준
    data_collection: List[str]   # 데이터 수집 방법
    contingency_plan: str        # 비상 계획


class OperationMonitoring(TypedDict, total=False):
    """운영 모니터링 및 지원 (I-27)"""
    monitoring_metrics: List[str]  # 모니터링 지표
    support_channels: List[str]    # 지원 채널
    escalation_process: str        # 에스컬레이션 프로세스
    feedback_collection: str       # 피드백 수집 방법


# ========== 8단계: 형성평가 실시 ==========
class FormativeEvaluationResult(TypedDict, total=False):
    """8단계 결과"""
    quality_score: float         # 품질 점수 (0-10)
    one_to_one_findings: List[str]      # 일대일 평가 결과
    small_group_findings: List[str]     # 소집단 평가 결과
    field_trial_findings: List[str]     # 현장 평가 결과
    strengths: List[str]         # 강점
    weaknesses: List[str]        # 약점
    revision_recommendations: List[str]  # 수정 권고사항 (최소 3개)
    # Implementation 단계 (I-24 ~ I-27)
    orientation_plan: OrientationPlan  # I-24: 오리엔테이션
    system_check: SystemCheck          # I-25: 시스템/환경 점검
    pilot_plan: PilotPlan              # I-26: 프로토타입 실행
    operation_monitoring: OperationMonitoring  # I-27: 운영 모니터링


# ========== 9단계: 교수프로그램 수정 ==========
class RevisionItem(TypedDict, total=False):
    """수정 항목"""
    issue: str                   # 문제점
    target_phase: str            # 수정 대상 단계
    action: str                  # 수정 조치
    status: str                  # 완료, 진행중, 보류


class RevisionResult(TypedDict, total=False):
    """9단계 결과"""
    iteration: int               # 반복 횟수
    revision_items: List[RevisionItem]  # 수정 항목 목록
    summary: str                 # 수정 요약


# ========== 10단계: 총괄평가 실시 ==========
class EffectivenessAnalysis(TypedDict, total=False):
    """총괄평가 시행 및 효과 분석 (E-31)"""
    kirkpatrick_levels: dict     # Kirkpatrick 4단계 평가
    roi_calculation: dict        # ROI 계산
    impact_assessment: str       # 영향 평가


class AdoptionDecision(TypedDict, total=False):
    """프로그램 채택 여부 결정 (E-32)"""
    recommendation: str          # adopt, modify, reject
    rationale: str               # 결정 근거
    conditions: List[str]        # 채택 조건
    next_steps: List[str]        # 다음 단계


class ProgramImprovement(TypedDict, total=False):
    """프로그램 개선 (E-33)"""
    improvement_areas: List[str]  # 개선 영역
    improvement_actions: List[str]  # 개선 조치
    timeline: str                 # 일정
    responsible: str              # 담당자


class SummativeEvaluationResult(TypedDict, total=False):
    """10단계 결과"""
    effectiveness_score: float   # 효과성 점수 (0-10)
    efficiency_analysis: str     # 효율성 분석
    learner_satisfaction: str    # 학습자 만족도
    goal_achievement: str        # 목표 달성도
    recommendations: List[str]   # 최종 권고사항
    decision: str                # 채택/수정/폐기
    # Evaluation 단계 (E-31 ~ E-33)
    effectiveness_analysis: EffectivenessAnalysis  # E-31: 효과 분석
    adoption_decision: AdoptionDecision  # E-32: 채택 여부 결정
    program_improvement: ProgramImprovement  # E-33: 프로그램 개선


# ========== 시나리오 입력 ==========
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


# ========== 메타데이터 ==========
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
    iteration_count: int         # 피드백 루프 반복 횟수


class Trajectory(TypedDict, total=False):
    """에이전트 궤적"""
    tool_calls: List[ToolCall]
    reasoning_steps: List[str]


# ========== 메인 State 스키마 ==========
class DickCareyState(TypedDict, total=False):
    """
    Dick & Carey 에이전트 상태 스키마

    LangGraph StateGraph에서 사용되는 상태 정의입니다.
    10단계 체제적 교수설계 프로세스를 지원합니다.
    형성평가-수정 피드백 루프를 통한 반복적 개선이 가능합니다.
    """
    # 입력
    scenario: ScenarioInput

    # 10단계별 산출물
    goal: InstructionalGoal                           # 1단계
    instructional_analysis: InstructionalAnalysis     # 2단계
    learner_context: LearnerContextResult             # 3단계
    performance_objectives: PerformanceObjectivesResult  # 4단계
    assessment_instruments: AssessmentInstrumentsResult  # 5단계
    instructional_strategy: InstructionalStrategyResult  # 6단계
    instructional_materials: InstructionalMaterialsResult  # 7단계
    formative_evaluation: FormativeEvaluationResult   # 8단계
    revision_log: List[RevisionResult]                # 9단계 (반복 기록)
    summative_evaluation: SummativeEvaluationResult   # 10단계

    # 순환 제어
    iteration_count: int          # 현재 반복 횟수
    max_iterations: int           # 최대 반복 횟수 (기본값: 3)
    quality_threshold: float      # 품질 기준 점수 (기본값: 7.0)
    revision_triggered: bool      # 수정 트리거 여부
    quality_score_history: List[float]  # 점수 이력

    # 상태 관리
    current_phase: DickCareyPhase
    errors: List[str]

    # 궤적 및 메타데이터
    tool_calls: List[ToolCall]
    reasoning_steps: List[str]
    metadata: Metadata


def create_initial_state(scenario: ScenarioInput) -> DickCareyState:
    """초기 상태 생성"""
    return DickCareyState(
        scenario=scenario,
        goal={},
        instructional_analysis={},
        learner_context={},
        performance_objectives={},
        assessment_instruments={},
        instructional_strategy={},
        instructional_materials={},
        formative_evaluation={},
        revision_log=[],
        summative_evaluation={},
        # 순환 제어
        iteration_count=0,
        max_iterations=3,
        quality_threshold=7.0,
        revision_triggered=False,
        quality_score_history=[],
        # 상태 관리
        current_phase="goal",
        errors=[],
        # 궤적
        tool_calls=[],
        reasoning_steps=[],
        metadata={
            "agent_version": "0.1.0",
            "tool_calls_count": 0,
            "iteration_count": 0,
        },
    )


def map_to_addie_output(state: DickCareyState) -> dict:
    """
    Dick & Carey 산출물을 ADDIE 스키마로 변환 (33개 소항목 완전 매핑)

    Dick & Carey → ADDIE 매핑:
    - 1-3단계 → Analysis (A-1 ~ A-10)
    - 4-5단계 → Design (D-11 ~ D-18)
    - 6-7단계 → Development (Dev-19 ~ Dev-23)
    - 8-9단계 → Implementation (I-24 ~ I-27)
    - 10단계 → Evaluation (E-28 ~ E-33)

    표준 스키마 (docs/addie_output_schema.json) 준수
    """
    learner_context = state.get("learner_context", {})
    goal = state.get("goal", {})
    instructional_analysis = state.get("instructional_analysis", {})
    instructional_strategy = state.get("instructional_strategy", {})
    instructional_materials = state.get("instructional_materials", {})
    formative_evaluation = state.get("formative_evaluation", {})
    summative_evaluation = state.get("summative_evaluation", {})
    assessment_instruments = state.get("assessment_instruments", {})
    performance_objectives = state.get("performance_objectives", {})

    # 학습 목표를 표준 형식으로 변환
    learning_objectives = []
    terminal_obj = performance_objectives.get("terminal_objective", {})
    if terminal_obj:
        learning_objectives.append({
            "id": terminal_obj.get("id", "TO-001"),
            "level": "terminal",
            "statement": terminal_obj.get("statement", ""),
            "bloom_verb": terminal_obj.get("bloom_level", ""),
            "measurable": True,
        })
    for idx, obj in enumerate(performance_objectives.get("enabling_objectives", [])):
        learning_objectives.append({
            "id": obj.get("id", f"EO-{idx+1:03d}"),
            "level": "enabling",
            "statement": obj.get("statement", ""),
            "bloom_verb": obj.get("bloom_level", ""),
            "measurable": True,
        })

    # 학습 활동 생성 (교수 전략에서 추출)
    learning_activities = []
    pre_inst = instructional_strategy.get("pre_instructional", {})
    if pre_inst:
        learning_activities.append({
            "activity_name": "Pre-instructional activities",
            "duration": "15 minutes",
            "description": f"Motivation: {pre_inst.get('motivation', '')}, Objective presentation: {pre_inst.get('objectives_info', '')}",
            "materials": ["Motivational materials"],
        })
    content_pres = instructional_strategy.get("content_presentation", {})
    if content_pres:
        learning_activities.append({
            "activity_name": "Content presentation",
            "duration": "30 minutes",
            "description": f"Examples: {', '.join(content_pres.get('examples', [])[:3])}",
            "materials": content_pres.get("sequence", [])[:3],
        })
    learner_part = instructional_strategy.get("learner_participation", {})
    if learner_part:
        learning_activities.append({
            "activity_name": "Learner participation activities",
            "duration": "25 minutes",
            "description": f"Practice: {', '.join(learner_part.get('practice_activities', [])[:3])}",
            "materials": ["Practice exercises", "Worksheets"],
        })

    return {
        "analysis": {
            # A1: 요구분석 (소항목 1-4)
            "needs_analysis": {
                # [1] 문제 확인 및 정의
                "problem_definition": goal.get("performance_gap", "") or f"Current state: {goal.get('current_state', '')}, Desired state: {goal.get('desired_state', '')}",
                # [2] 차이분석
                "gap_analysis": goal.get("needs_analysis", {}).get("gap_analysis", []) or [
                    {"current": goal.get("current_state", ""), "target": goal.get("desired_state", ""), "gap": goal.get("performance_gap", "")}
                ],
                # [3] 수행분석
                "performance_analysis": f"Root causes: {', '.join(goal.get('needs_analysis', {}).get('root_causes', []))}. Training needs: {', '.join(goal.get('needs_analysis', {}).get('training_needs', []))}",
                # [4] 요구 우선순위 결정
                "priority_matrix": goal.get("needs_analysis", {}).get("priority_matrix", {
                    "high_priority": goal.get("needs_analysis", {}).get("training_needs", [])[:2],
                    "medium_priority": [],
                    "low_priority": goal.get("needs_analysis", {}).get("non_training_solutions", []),
                }),
            },
            # A2: 학습자 및 환경분석 (소항목 5-6)
            "learner_analysis": {
                "target_audience": learner_context.get("learner", {}).get("target_audience", ""),
                "characteristics": learner_context.get("learner", {}).get("characteristics", []),
                "prior_knowledge": learner_context.get("learner", {}).get("prior_knowledge", ""),
                "learning_preferences": learner_context.get("learner", {}).get("learning_preferences", []),
                "motivation": learner_context.get("learner", {}).get("motivation", ""),
            },
            "context_analysis": {
                "environment": learner_context.get("context", {}).get("learning_context", ""),
                "constraints": learner_context.get("context", {}).get("constraints", []),
                "resources": learner_context.get("context", {}).get("resources", []),
                "technical_requirements": learner_context.get("context", {}).get("technical_requirements", []),
            },
            # A3: 과제 및 목표분석 (소항목 7-10)
            "task_analysis": {
                # [7] 초기 학습목표 분석
                "initial_objectives": [obj.get("statement", "") for obj in performance_objectives.get("enabling_objectives", [])[:5]] or [goal.get("goal_statement", "")],
                # [8] 하위 기능 분석
                "subtopics": [skill.get("description", "") for skill in instructional_analysis.get("sub_skills", [])],
                # [9] 출발점 행동 분석
                "prerequisites": instructional_analysis.get("entry_skills", []) or learner_context.get("learner", {}).get("entry_behaviors", []),
                # [10] 과제분석 결과 검토·정리
                "review_summary": instructional_analysis.get("review_summary", "") or f"Task type: {instructional_analysis.get('task_type', '')}. Analysis completed for {len(instructional_analysis.get('sub_skills', []))} sub-skills.",
            },
        },
        "design": {
            # [11] 학습목표 정교화
            "learning_objectives": learning_objectives,
            # [12] 평가 계획 수립
            "assessment_plan": {
                "formative": [{"type": item.get("type", ""), "description": item.get("question", "")} for item in assessment_instruments.get("practice_tests", [])[:3]],
                "summative": [{"type": item.get("type", ""), "description": item.get("question", "")} for item in assessment_instruments.get("post_test", [])[:3]],
                "assessment_criteria": [item.get("rubric", "") for item in assessment_instruments.get("post_test", [])[:5]],
            },
            # [13] 교수 내용 선정
            "content_structure": {
                "modules": instructional_strategy.get("content_selection", {}).get("core_content", []),
                "topics": instructional_strategy.get("content_selection", {}).get("supplementary_content", []),
                "sequencing": instructional_strategy.get("content_selection", {}).get("selection_rationale", "") or ", ".join(instructional_strategy.get("content_presentation", {}).get("sequence", [])),
            },
            # [14] 교수적 전략 수립
            "instructional_strategies": {
                "methods": [instructional_strategy.get("delivery_method", ""), instructional_strategy.get("grouping_strategy", "")],
                "activities": instructional_strategy.get("learner_participation", {}).get("practice_activities", []),
                "rationale": f"Delivery method: {instructional_strategy.get('delivery_method', '')}. {instructional_strategy.get('learner_participation', {}).get('feedback_strategy', '')}",
            },
            # [15] 비교수적 전략 수립
            "non_instructional_strategies": {
                "motivation_strategies": [instructional_strategy.get("pre_instructional", {}).get("motivation", "")],
                "self_directed_learning": instructional_strategy.get("non_instructional_strategy", {}).get("strategies", []),
                "support_strategies": [instructional_strategy.get("non_instructional_strategy", {}).get("implementation", "")],
            },
            # [16] 매체 선정과 활용 계획
            "media_selection": {
                "media_types": instructional_strategy.get("media_selection", {}).get("selected_media", []),
                "tools": [],
                "utilization_plan": instructional_strategy.get("media_selection", {}).get("utilization_plan", ""),
            },
            # [17] 학습활동 및 시간 구조화
            "learning_activities": learning_activities,
            # [18] 스토리보드/화면 흐름 설계
            "storyboard": {
                "screens": _extract_storyboards(instructional_materials),
                "navigation_flow": "Sequential progression",
                "interactions": ["Click", "Drag and drop", "Text input"],
            },
        },
        "development": {
            # [19] 학습자용 자료 개발
            "learner_materials": [
                {
                    "title": mat.get("title", ""),
                    "type": mat.get("type", ""),
                    "content": mat.get("description", ""),
                    "format": "PDF/PPT",
                }
                for mat in instructional_materials.get("learner_materials", [])
            ],
            # [20] 교수자용 매뉴얼 개발
            "instructor_guide": {
                "overview": instructional_materials.get("instructor_guide", {}).get("description", "") or instructional_materials.get("instructor_manual", ""),
                "session_guides": instructional_materials.get("instructor_guide", {}).get("content_outline", []),
                "facilitation_tips": ["Encourage learner participation", "Use questioning", "Provide feedback"],
                "troubleshooting": ["Handle technical issues", "Adjust learning pace"],
            },
            # [21] 운영자용 매뉴얼 개발
            "operator_manual": {
                "system_setup": instructional_materials.get("operator_manual", "") or "System setup guide",
                "operation_procedures": ["Registration management", "Attendance management", "Grade management"],
                "support_procedures": ["Respond to learner inquiries", "Technical support"],
                "escalation_process": "Report to the person in charge when an issue occurs",
            },
            # [22] 평가 도구·문항 개발
            "assessment_tools": [
                {
                    "item_id": item.get("id", f"Q-{idx+1:03d}"),
                    "type": item.get("type", ""),
                    "question": item.get("question", ""),
                    "aligned_objective": item.get("objective_id", ""),
                    "scoring_criteria": item.get("rubric", ""),
                }
                for idx, item in enumerate(assessment_instruments.get("post_test", []))
            ],
            # [23] 전문가 검토
            "expert_review": {
                "reviewers": [instructional_materials.get("expert_review", {}).get("reviewer", "Subject matter expert")],
                "review_criteria": instructional_materials.get("expert_review", {}).get("review_areas", ["Content accuracy", "Instructional design appropriateness"]),
                "feedback_summary": "; ".join(instructional_materials.get("expert_review", {}).get("findings", [])),
                "revisions_made": instructional_materials.get("expert_review", {}).get("recommendations", []),
            },
        },
        "implementation": {
            # [24] 교수자·운영자 오리엔테이션
            "instructor_orientation": {
                "orientation_objectives": formative_evaluation.get("orientation_plan", {}).get("schedule", []) or ["Understand the program", "Master operational procedures"],
                "schedule": formative_evaluation.get("orientation_plan", {}).get("facilitator_orientation", "") or "One week in advance",
                "materials": ["Instructor guide", "Operations manual"],
                "competency_checklist": ["Content comprehension", "Facilitation ability", "Technology utilization ability"],
            },
            # [25] 시스템/환경 점검
            "system_check": {
                "checklist": formative_evaluation.get("system_check", {}).get("checklist", ["Network connection", "Equipment check", "Material preparation"]),
                "technical_validation": "; ".join(formative_evaluation.get("system_check", {}).get("technical_tests", [])) or "System testing completed",
                "contingency_plans": [formative_evaluation.get("system_check", {}).get("contingency_plan", "Contingency plan established")],
            },
            # [26] 프로토타입 실행
            "prototype_execution": {
                "pilot_scope": formative_evaluation.get("pilot_plan", {}).get("pilot_scope", "Small-scale pilot test"),
                "participants": formative_evaluation.get("pilot_plan", {}).get("participants", "About 10 participants"),
                "execution_log": formative_evaluation.get("pilot_plan", {}).get("data_collection", []),
                "issues_encountered": formative_evaluation.get("weaknesses", []),
            },
            # [27] 운영 모니터링 및 지원
            "monitoring": {
                "monitoring_criteria": formative_evaluation.get("operation_monitoring", {}).get("monitoring_metrics", ["Learning progress", "Participation rate", "Satisfaction"]),
                "support_channels": formative_evaluation.get("operation_monitoring", {}).get("support_channels", ["Email", "Phone", "Online forum"]),
                "issue_resolution_log": [],
                "real_time_adjustments": formative_evaluation.get("revision_recommendations", []),
            },
        },
        "evaluation": {
            # E1: 형성평가 (소항목 28-29)
            "formative": {
                # [28] 파일럿/초기 실행 중 자료 수집
                "data_collection": {
                    "methods": ["One-to-one evaluation", "Small group evaluation", "Field trial"],
                    "learner_feedback": formative_evaluation.get("one_to_one_findings", []) + formative_evaluation.get("small_group_findings", []),
                    "performance_data": {
                        "quality_score": formative_evaluation.get("quality_score", 0),
                        "strengths": formative_evaluation.get("strengths", []),
                        "weaknesses": formative_evaluation.get("weaknesses", []),
                    },
                    "observations": formative_evaluation.get("field_trial_findings", []),
                },
                # [29] 형성평가 결과 기반 1차 프로그램 개선
                "improvements": [
                    {
                        "issue_identified": rec,
                        "improvement_action": f"Improvement action for {rec}",
                        "priority": "High" if idx < 2 else "Medium",
                    }
                    for idx, rec in enumerate(formative_evaluation.get("revision_recommendations", []))
                ],
            },
            # E2: 총괄평가 및 채택 결정 (소항목 30-32)
            "summative": {
                # [30] 총괄 평가 문항 개발
                "assessment_tools": [
                    {
                        "item_id": item.get("id", f"SQ-{idx+1:03d}"),
                        "type": item.get("type", ""),
                        "question": item.get("question", ""),
                        "scoring_rubric": item.get("rubric", ""),
                    }
                    for idx, item in enumerate(assessment_instruments.get("post_test", []))
                ],
                # [31] 총괄평가 시행 및 프로그램 효과 분석
                "effectiveness_analysis": {
                    "learning_outcomes": summative_evaluation.get("effectiveness_analysis", {}).get("kirkpatrick_levels", {}),
                    "goal_achievement_rate": summative_evaluation.get("goal_achievement", "") or f"{summative_evaluation.get('effectiveness_score', 0)*10}%",
                    "statistical_analysis": summative_evaluation.get("efficiency_analysis", ""),
                    "recommendations": summative_evaluation.get("recommendations", []),
                },
                # [32] 프로그램 채택 여부 결정
                "adoption_decision": {
                    "decision": summative_evaluation.get("adoption_decision", {}).get("recommendation", "") or summative_evaluation.get("decision", ""),
                    "rationale": summative_evaluation.get("adoption_decision", {}).get("rationale", ""),
                    "conditions": summative_evaluation.get("adoption_decision", {}).get("conditions", []),
                    "stakeholder_approval": "Pending approval",
                },
            },
            # [33] E3: 프로그램 개선 및 환류
            "improvement_plan": {
                "feedback_summary": summative_evaluation.get("learner_satisfaction", ""),
                "improvement_areas": summative_evaluation.get("program_improvement", {}).get("improvement_areas", []),
                "action_items": summative_evaluation.get("program_improvement", {}).get("improvement_actions", []),
                "feedback_loop": "Reflect evaluation results in the next iteration of the program",
                "next_iteration_goals": summative_evaluation.get("adoption_decision", {}).get("next_steps", []),
            },
        },
    }


def _extract_storyboards(instructional_materials: dict) -> List[dict]:
    """교수자료에서 스토리보드 추출"""
    storyboards = []
    for material in instructional_materials.get("learner_materials", []):
        if material.get("storyboard"):
            storyboards.extend(material.get("storyboard", []))
    for media in instructional_materials.get("media_list", []):
        if media.get("storyboard"):
            storyboards.extend(media.get("storyboard", []))
    return storyboards
