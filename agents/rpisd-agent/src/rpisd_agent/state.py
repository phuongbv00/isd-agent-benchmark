"""
RPISD Agent State Schema

RPISD 에이전트의 상태를 정의하는 TypedDict 스키마입니다.
LangGraph StateGraph에서 사용됩니다.

특징:
- 이중 루프 제어 (프로토타입/개발)
- 프로토타입 버전 이력 관리
- 다중 피드백 통합 (의뢰인/전문가/학습자)
"""

from typing import TypedDict, Optional, List, Literal
from datetime import datetime


# RPISD 단계 타입
RPISDPhase = Literal[
    "kickoff",              # 프로젝트 착수
    "analysis",             # 분석
    "design",               # 설계
    "usability_evaluation", # 사용성 평가
    "development",          # 개발
    "implementation",       # 실행
    "evaluation",           # 평가
    "complete",             # 완료
]

# 루프 출처 타입
LoopSource = Literal["prototype", "development", ""]


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


class Metadata(TypedDict, total=False):
    """메타데이터"""
    model: str                   # 사용 모델 (solar-mini, solar-pro-2)
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    execution_time_seconds: float
    cost_usd: float
    agent_version: str
    tool_calls_count: int


class PrototypeVersion(TypedDict, total=False):
    """프로토타입 버전 이력"""
    version: int                 # 버전 번호 (1, 2, 3...)
    content: dict                # 프로토타입 내용
    feedback: List[dict]         # 수집된 피드백
    quality_score: float         # 품질 점수 (0.0 - 1.0)
    timestamp: str               # 생성 시각


class UsabilityFeedback(TypedDict, total=False):
    """통합된 사용성 평가 피드백"""
    client_feedback: dict        # 의뢰인 피드백
    expert_feedback: dict        # 전문가 피드백
    learner_feedback: dict       # 학습자 피드백
    aggregated_score: float      # 통합 점수
    improvement_areas: List[str] # 개선 필요 영역
    recommendations: List[str]   # 개선 권고 사항


# ========== 단계별 산출물 타입 ==========

class KickoffResult(TypedDict, total=False):
    """프로젝트 착수 회의 결과"""
    project_title: str
    scope: dict
    stakeholder_roles: dict
    timeline: dict
    success_criteria: List[str]
    constraints: List[str]


class GapAnalysis(TypedDict, total=False):
    """차이 분석 결과"""
    current_state: str
    desired_state: str
    gaps: List[str]
    root_causes: List[str]
    training_needs: List[str]


class PerformanceAnalysis(TypedDict, total=False):
    """수행 분석 결과"""
    performance_issues: List[str]
    causes: List[str]
    solutions: List[str]
    is_training_solution: bool


class LearnerCharacteristics(TypedDict, total=False):
    """학습자 특성 분석 결과"""
    target_audience: str
    demographics: dict
    prior_knowledge: str
    learning_preferences: List[str]
    motivation: str
    challenges: List[str]


class InitialTaskAnalysis(TypedDict, total=False):
    """초기 학습과제 분석 결과"""
    main_topics: List[str]
    subtopics: List[str]
    prerequisites: List[str]
    task_hierarchy: dict


class AnalysisResult(TypedDict, total=False):
    """분석 단계 통합 산출물"""
    gap_analysis: GapAnalysis
    performance_analysis: PerformanceAnalysis
    learner_characteristics: LearnerCharacteristics
    initial_task: InitialTaskAnalysis


class LearningObjective(TypedDict, total=False):
    """학습 목표"""
    level: str                   # Bloom's Taxonomy 수준
    statement: str               # 목표 문장 (예: "핵심 개념을 설명할 수 있다")
    bloom_verb: str
    measurable: bool


class InstructionalEvent(TypedDict, total=False):
    """교수 사태"""
    event: str
    activity: str
    duration: str
    resources: List[str]


class DesignResult(TypedDict, total=False):
    """설계 단계 산출물"""
    objectives: List[LearningObjective]
    strategy: dict
    sequence: List[InstructionalEvent]
    prototype_spec: dict         # 프로토타입 명세


class Module(TypedDict, total=False):
    """레슨 모듈"""
    title: str
    duration: str
    objectives: List[str]
    activities: List[dict]


class Material(TypedDict, total=False):
    """학습 자료"""
    type: str
    title: str
    description: str
    slides: int
    duration: str
    pages: int
    questions: int
    slide_contents: List[dict]


class QuizItem(TypedDict, total=False):
    """평가 문항"""
    id: str
    question: str
    type: str
    options: List[str]
    answer: str
    explanation: str
    objective_id: str
    difficulty: str


class Rubric(TypedDict, total=False):
    """평가 루브릭"""
    criteria: List[str]
    levels: dict
    feedback_plan: str


class DevelopmentResult(TypedDict, total=False):
    """개발 단계 산출물"""
    lesson_plan: dict
    modules: List[Module]
    materials: List[Material]
    final_prototype: dict
    quiz_items: List[QuizItem]
    slide_contents: List[dict]


class ImplementationResult(TypedDict, total=False):
    """실행 단계 산출물"""
    delivery_method: str
    facilitator_guide: str
    learner_guide: str
    operator_guide: str                  # 운영자 가이드
    technical_requirements: List[str]
    maintenance_plan: dict
    support_plan: dict                   # dict로 변경 (상세 정보)
    pilot_plan: dict                     # 파일럿 실행 계획
    orientation_plan: dict               # 오리엔테이션 계획
    monitoring_plan: dict                # 운영 모니터링 계획


class EvaluationResult(TypedDict, total=False):
    """평가 단계 산출물"""
    quiz_items: List[QuizItem]
    rubric: Rubric
    program_evaluation: dict             # Kirkpatrick 4단계 평가
    usability_summary: dict              # 사용성 평가 요약
    adoption_decision: dict              # 프로그램 채택 결정
    improvement_plan: dict               # 개선 계획


# ========== 메인 State 스키마 ==========

class RPISDState(TypedDict, total=False):
    """
    RPISD 에이전트 상태 스키마

    LangGraph StateGraph에서 사용되는 상태 정의입니다.
    이중 루프 제어 및 프로토타입 버전 관리를 지원합니다.
    """
    # 입력
    scenario: dict               # ScenarioInput

    # 단계별 산출물
    kickoff_result: KickoffResult
    analysis_result: AnalysisResult
    design_result: DesignResult
    prototype_versions: List[PrototypeVersion]
    usability_feedback: UsabilityFeedback
    development_result: DevelopmentResult
    implementation_result: ImplementationResult
    evaluation_result: EvaluationResult      # 평가 단계 산출물

    # 순환 제어 (이중 루프)
    prototype_iteration: int     # 프로토타입 반복 횟수 (내부 루프)
    development_iteration: int   # 개발 반복 횟수 (외부 루프)
    max_iterations: int          # 최대 반복 횟수 (기본값: 3)
    quality_threshold: float     # 품질 기준 (기본값: 0.8)
    current_quality: float       # 현재 품질 점수

    # 상태 관리
    current_phase: RPISDPhase
    previous_phase: RPISDPhase   # 이전 단계 (루프 출처 추적)
    loop_source: LoopSource      # "prototype" | "development" | ""
    errors: List[str]

    # 궤적 및 메타데이터
    tool_calls: List[ToolCall]
    reasoning_steps: List[str]
    metadata: Metadata


def create_initial_state(scenario: dict) -> RPISDState:
    """초기 상태 생성

    기본값 최적화 (#78):
        - max_iterations: 3→2 (래피드 프로토타이핑)
        - quality_threshold: 0.8→0.75 (현실적 임계값)
    """
    return RPISDState(
        scenario=scenario,
        kickoff_result={},
        analysis_result={},
        design_result={},
        prototype_versions=[],
        usability_feedback={},
        development_result={},
        implementation_result={},
        evaluation_result={},
        prototype_iteration=0,
        development_iteration=0,
        max_iterations=2,  # 3→2: 래피드 프로토타이핑 최적화 (#78)
        quality_threshold=0.75,  # 0.8→0.75: 현실적 임계값 (#78)
        current_quality=0.0,
        current_phase="kickoff",
        previous_phase="kickoff",
        loop_source="",
        errors=[],
        tool_calls=[],
        reasoning_steps=[],
        metadata={
            "agent_version": "0.1.0",
            "tool_calls_count": 0,
        },
    )


def record_prototype_version(
    state: RPISDState,
    content: dict,
    feedback: List[dict],
    quality_score: float,
) -> PrototypeVersion:
    """프로토타입 버전 기록"""
    version = len(state.get("prototype_versions", [])) + 1
    return PrototypeVersion(
        version=version,
        content=content,
        feedback=feedback,
        quality_score=quality_score,
        timestamp=datetime.now().isoformat(),
    )


def _format_causes(causes) -> str:
    """원인 데이터를 문자열로 변환하는 헬퍼 함수"""
    if causes is None:
        return ""
    if isinstance(causes, list):
        return ", ".join(str(c) for c in causes[:3])
    if isinstance(causes, dict):
        # 딕셔너리의 값들을 평탄화
        all_causes = []
        for key, value in causes.items():
            if isinstance(value, list):
                all_causes.extend(value)
            else:
                all_causes.append(str(value))
        return ", ".join(all_causes[:3])
    return str(causes)


def map_to_addie_output(state: RPISDState) -> dict:
    """
    RPISD 산출물을 ADDIE 표준 스키마로 변환

    표준 스키마 (docs/addie_output_schema.json) 준수:
    33개 ADDIE 소항목을 모두 출력하도록 매핑합니다.
    """
    scenario = state.get("scenario", {})
    scenario_context = scenario.get("context", {})
    analysis = state.get("analysis_result", {})
    kickoff = state.get("kickoff_result", {})
    design = state.get("design_result", {})
    development = state.get("development_result", {})
    implementation = state.get("implementation_result", {})
    usability = state.get("usability_feedback", {})
    evaluation = state.get("evaluation_result", {})
    prototype_versions = state.get("prototype_versions", [])

    # quiz_items 추출 (development_result 또는 evaluation_result에서)
    quiz_items = development.get("quiz_items", []) or evaluation.get("quiz_items", [])

    # rubric 추출
    rubric = evaluation.get("rubric", {})

    # program_evaluation 추출
    program_eval = evaluation.get("program_evaluation", {}) or _generate_program_evaluation(state)

    # gap_analysis 추출
    gap = analysis.get("gap_analysis", {})
    perf = analysis.get("performance_analysis", {})
    learner = analysis.get("learner_characteristics", {})
    task = analysis.get("initial_task", {})

    # quiz_items를 assessment_tools 형식으로 변환
    assessment_tools = [
        {
            "item_id": q.get("id", f"Q-{idx+1:03d}"),
            "type": q.get("type", ""),
            "question": q.get("question", ""),
            "aligned_objective": q.get("objective_id", ""),
            "scoring_criteria": q.get("explanation", ""),
        }
        for idx, q in enumerate(quiz_items)
    ]

    # 학습 활동 생성 (design.sequence에서)
    learning_activities = []
    for event in design.get("sequence", []):
        learning_activities.append({
            "activity_name": event.get("event", ""),
            "duration": event.get("duration", ""),
            "description": event.get("activity", ""),
            "materials": event.get("resources", []),
        })

    return {
        # ========== Analysis (표준 스키마) ==========
        "analysis": {
            # A1: 요구분석 (소항목 1-4)
            "needs_analysis": {
                # [1] 문제 확인 및 정의
                "problem_definition": gap.get("performance_gap", "") or f"Current: {gap.get('current_state', '')}, Target: {gap.get('desired_state', '')}",
                # [2] 차이분석
                "gap_analysis": gap.get("gaps", []) if isinstance(gap.get("gaps"), list) else [
                    {"current": gap.get("current_state", ""), "target": gap.get("desired_state", ""), "gap": g}
                    for g in (gap.get("gaps", []) or [])
                ],
                # [3] 수행분석
                "performance_analysis": f"Training solution: {perf.get('is_training_solution', True)}. Causes: {_format_causes(perf.get('causes'))}",
                # [4] 요구 우선순위 결정
                "priority_matrix": {
                    "high_priority": (gap.get("training_needs") or [])[:3],
                    "medium_priority": (gap.get("training_needs") or [])[3:6] if len(gap.get("training_needs") or []) > 3 else [],
                    "root_causes": gap.get("root_causes") or [],
                },
            },
            # A2: 학습자 및 환경분석 (소항목 5-6)
            "learner_analysis": {
                "target_audience": learner.get("target_audience", ""),
                "characteristics": learner.get("challenges", []),
                "prior_knowledge": learner.get("prior_knowledge", ""),
                "learning_preferences": learner.get("learning_preferences", []),
                "motivation": learner.get("motivation", ""),
            },
            "context_analysis": {
                # environment: 시나리오 context에서 학습 환경 정보 추출 (#78)
                "environment": (
                    scenario_context.get("learning_environment", "") or
                    kickoff.get("scope", {}).get("delivery_format", "") or
                    "Blended online/offline learning environment"
                ),
                "constraints": kickoff.get("constraints", []) or scenario.get("constraints", {}).get("resources", []) or ["Time constraints", "Budget constraints"],
                # resources: 시나리오에서 가용 자원 추출 또는 기본값 (#78)
                "resources": _extract_resources(scenario, kickoff, implementation),
                "technical_requirements": implementation.get("technical_requirements", []) or ["LMS access rights", "Stable internet connection"],
            },
            # A3: 과제 및 목표분석 (소항목 7-10)
            "task_analysis": {
                # [7] 초기 학습목표 분석
                "initial_objectives": task.get("main_topics", []),
                # [8] 하위 기능 분석
                "subtopics": task.get("subtopics", []),
                # [9] 출발점 행동 분석
                "prerequisites": task.get("prerequisites", []),
                # [10] 과제분석 결과 검토·정리
                "review_summary": f"Topics: {', '.join((task.get('main_topics') or [])[:3])}. Training need: {perf.get('is_training_solution', True)}",
            },
        },

        # ========== Design (표준 스키마) ==========
        "design": {
            # [11] 학습목표 정교화
            "learning_objectives": design.get("objectives", []),
            # [12] 평가 계획 수립
            "assessment_plan": {
                "formative": [{"description": "Prototype validation quiz"}, {"description": "Usability test"}],
                "summative": [{"description": "Final assessment quiz"}, {"description": "Performance assessment"}],
                "assessment_criteria": rubric.get("criteria", []),
            },
            # [13] 교수 내용 선정
            "content_structure": {
                "modules": [m.get("title", "") for m in development.get("modules", [])],
                "topics": task.get("main_topics", []),
                "sequencing": design.get("strategy", {}).get("sequencing", ""),
            },
            # [14] 교수적 전략 수립
            "instructional_strategies": {
                "methods": design.get("methods", []) or design.get("strategy", {}).get("methods", []),
                "activities": [e.get("activity", "") for e in (design.get("sequence") or [])[:5]],
                "rationale": f"Model: {design.get('strategy', {}).get('model', '')}",
            },
            # [15] 비교수적 전략 수립
            "non_instructional_strategies": {
                "motivation_strategies": [],
                "self_directed_learning": [],
                "support_strategies": [],
            },
            # [16] 매체 선정과 활용 계획
            "media_selection": {
                "media_types": design.get("strategy", {}).get("media", []),
                "tools": [],
                "utilization_plan": "",
            },
            # [17] 학습활동 및 시간 구조화
            "learning_activities": learning_activities,
            # [18] 스토리보드/화면 흐름 설계
            "storyboard": {
                "screens": [
                    {"screen_id": "S001", "title": "Introduction screen", "content": "Present learning objectives and overview", "media": "Text/Image"},
                    {"screen_id": "S002", "title": "Learning content screen", "content": "Core concept explanation and examples", "media": "Text/Video"},
                    {"screen_id": "S003", "title": "Practice screen", "content": "Learner activities and feedback", "media": "Interactive"},
                    {"screen_id": "S004", "title": "Assessment screen", "content": "Formative assessment and results review", "media": "Quiz"},
                    {"screen_id": "S005", "title": "Wrap-up screen", "content": "Summary of learning content and next-step guidance", "media": "Text"},
                ],
                "navigation_flow": "S001(Introduction) → S002(Learning) → S003(Practice) → S004(Assessment) → S005(Wrap-up), previous/next navigation available on each screen, direct navigation via menu supported",
                "interactions": [
                    {"type": "Click", "description": "Move to the next screen by clicking a button"},
                    {"type": "Drag and drop", "description": "Arrange elements in practice activities"},
                    {"type": "Input", "description": "Submit answers via text input"},
                    {"type": "Selection", "description": "Select a multiple-choice option"},
                ],
            },
        },

        # ========== Development (표준 스키마) ==========
        "development": {
            # [19] 학습자용 자료 개발
            "learner_materials": [
                {
                    "title": mat.get("title", ""),
                    "type": mat.get("type", ""),
                    "content": mat.get("description", ""),
                    "format": "PDF/PPT",
                }
                for mat in development.get("materials", [])
            ],
            # [20] 교수자용 매뉴얼 개발
            "instructor_guide": {
                "overview": implementation.get("facilitator_guide", ""),
                "session_guides": [m.get("title", "") for m in development.get("modules", [])],
                "facilitation_tips": ["Encourage learner participation", "Use questioning"],
                "troubleshooting": ["Respond to technical issues"],
            },
            # [21] 운영자용 매뉴얼 개발
            "operator_manual": {
                "system_setup": implementation.get("operator_guide", "") or _generate_operator_guide(implementation),
                "operation_procedures": ["Enrollment management", "Attendance management"],
                "support_procedures": ["Respond to learner inquiries"],
                "escalation_process": "Report to the responsible party when an issue occurs",
            },
            # [22] 평가 도구·문항 개발
            "assessment_tools": assessment_tools,
            # [23] 전문가 검토
            "expert_review": {
                "reviewers": ["Subject-matter expert", "Instructional design expert"],
                "review_criteria": ["Content accuracy", "Instructional design appropriateness"],
                "feedback_summary": f"Usability score: {usability.get('aggregated_score', 0):.2f}",
                "revisions_made": usability.get("recommendations", []),
            },
        },

        # ========== Implementation (표준 스키마) ==========
        "implementation": {
            # [24] 교수자·운영자 오리엔테이션
            "instructor_orientation": {
                "orientation_objectives": ["Understand the program", "Master operational procedures"],
                "schedule": implementation.get("orientation_plan", {}).get("pre_training", "") or "One week in advance",
                "materials": ["Facilitator guide", "Operations manual"],
                "competency_checklist": ["Content comprehension", "Facilitation ability"],
            },
            # [25] 시스템/환경 점검
            "system_check": {
                "checklist": implementation.get("technical_requirements", ["Network connectivity", "Equipment check"]),
                "technical_validation": "System test completed",
                "contingency_plans": ["Establish contingency response plan"],
            },
            # [26] 프로토타입 실행
            "prototype_execution": {
                "pilot_scope": implementation.get("pilot_plan", {}).get("phase", "") or "Small-scale pilot test",
                "participants": implementation.get("pilot_plan", {}).get("participants", "") or "10-20 participants",
                "execution_log": [f"Prototype v{p.get('version')} test" for p in prototype_versions],
                "issues_encountered": usability.get("improvement_areas", []),
            },
            # [27] 운영 모니터링 및 지원
            "monitoring": {
                "monitoring_criteria": ["Learning progress", "Participation rate", "Satisfaction"],
                "support_channels": ["Email", "Phone", "Online message board"],
                "issue_resolution_log": [],
                "real_time_adjustments": usability.get("recommendations", []),
            },
        },

        # ========== Evaluation (표준 스키마) ==========
        "evaluation": {
            # E1: 형성평가 (소항목 28-29)
            "formative": {
                # [28] 파일럿/초기 실행 중 자료 수집
                "data_collection": {
                    "methods": ["Client evaluation", "Expert evaluation", "Learner evaluation", "Pre/post test", "Observation records"],
                    "learner_feedback": usability.get("learner_feedback", {}).get("feedback_items", []) or [
                        "Feedback on comprehension of learning content",
                        "Ease of use of learning materials",
                        "Difficult modules and improvement requests",
                    ],
                    "performance_data": {
                        "client_score": usability.get("client_feedback", {}).get("overall_score", 0) or 0.8,
                        "expert_score": usability.get("expert_feedback", {}).get("overall_score", 0) or 0.85,
                        "learner_score": usability.get("learner_feedback", {}).get("overall_score", 0) or 0.75,
                    },
                    "observations": ["Observation of learning pace", "Monitoring of engagement and concentration", "Interaction patterns among learners"],
                    "pilot_difficulties": {
                        "identified_modules": ["Concept deepening module", "Practice application module"],
                        "difficulty_reasons": ["Content complexity", "Insufficient practice guidance"],
                        "improvement_suggestions": ["Add step-by-step explanations", "Expand practice examples"],
                    },
                },
                # [29] 형성평가 결과 기반 1차 프로그램 개선
                "improvements": [
                    {
                        "issue_identified": area.get("area") if isinstance(area, dict) else str(area),
                        "improvement_action": "Improvement action",
                        "priority": area.get("priority", "medium") if isinstance(area, dict) else "medium",
                    }
                    for area in usability.get("improvement_areas", [])
                ],
            },
            # E2: 총괄평가 및 채택 결정 (소항목 30-32)
            "summative": {
                # [30] 총괄 평가 문항 개발
                "assessment_tools": assessment_tools,
                # [31] 총괄평가 시행 및 프로그램 효과 분석
                "effectiveness_analysis": {
                    "learning_outcomes": program_eval.get("levels", {}).get("level_2_learning", {}),
                    "goal_achievement_rate": f"{state.get('current_quality', 0)*100:.0f}%",
                    "statistical_analysis": str(program_eval.get("levels", {}).get("level_1_reaction", {})),
                    "recommendations": usability.get("recommendations", []),
                },
                # [32] 프로그램 채택 여부 결정
                "adoption_decision": {
                    "decision": evaluation.get("adoption_decision", {}).get("recommendation", "") or _generate_adoption_decision(state).get("recommendation", ""),
                    "rationale": evaluation.get("adoption_decision", {}).get("rationale", "") or _generate_adoption_decision(state).get("rationale", ""),
                    "conditions": evaluation.get("adoption_decision", {}).get("conditions", []),
                    "stakeholder_approval": "Pending approval",
                },
            },
            # [33] E3: 프로그램 개선 및 환류
            "improvement_plan": {
                "feedback_summary": f"Final quality score: {state.get('current_quality', 0):.2f}",
                "improvement_areas": usability.get("improvement_areas", []),
                "action_items": usability.get("recommendations", []),
                "feedback_loop": "Reflect evaluation results in the next training cycle",
                "next_iteration_goals": evaluation.get("adoption_decision", {}).get("next_steps", []),
            },
        },
    }


def _extract_resources(scenario: dict, kickoff: dict, implementation: dict) -> list:
    """가용 자원 정보 추출 헬퍼 (#78)

    시나리오, kickoff, implementation에서 자원 정보를 추출합니다.
    """
    resources = []

    # 시나리오 constraints에서 resources 추출
    constraints = scenario.get("constraints", {})
    if isinstance(constraints, dict):
        if constraints.get("resources"):
            resources.append(f"Available resources: {constraints.get('resources')}")
        if constraints.get("budget"):
            resources.append(f"Budget: {constraints.get('budget')}")
        if constraints.get("timeline"):
            resources.append(f"Timeline: {constraints.get('timeline')}")

    # kickoff에서 추가 자원 정보 추출
    scope = kickoff.get("scope", {})
    if scope.get("resources"):
        if isinstance(scope.get("resources"), list):
            resources.extend(scope.get("resources"))
        else:
            resources.append(str(scope.get("resources")))

    # implementation에서 기술 요구사항 추출
    tech_reqs = implementation.get("technical_requirements", [])
    if tech_reqs:
        resources.extend([f"Technology: {req}" for req in tech_reqs[:2]])

    # 기본값 제공 (빈 경우)
    if not resources:
        resources = [
            "LMS platform",
            "Training content authoring tools",
            "Assessment system",
        ]

    return resources


def _generate_operator_guide(implementation: dict) -> str:
    """운영자 가이드 생성 헬퍼"""
    delivery = implementation.get("delivery_method", "Blended learning")
    return f"""This is the operations guide for running this training program.

1. Advance preparation
   - Configure the LMS and verify participant enrollment
   - Upload training materials and set access permissions
   - Check the technical environment (video conferencing, projector, etc.)

2. Training day
   - Check attendance and encourage participation
   - Stand by for technical support
   - Support instructor-learner communication

3. After training
   - Conduct a satisfaction survey
   - Write the results report
   - Compile improvement items

Delivery mode: {delivery}"""


def _generate_pilot_plan(prototype_versions: list) -> dict:
    """파일럿 실행 계획 생성 헬퍼"""
    num_versions = len(prototype_versions)
    return {
        "phase": "Pilot test",
        "prototype_tested": num_versions,
        "participants": "10-20 participants (representative sample)",
        "duration": "1-2 weeks",
        "evaluation_criteria": [
            "Learning objective attainment rate",
            "Usability score",
            "Learner satisfaction",
        ],
        "success_threshold": 0.8,
        "feedback_collection": [
            "Survey",
            "Interview",
            "Observation",
        ],
    }


def _generate_program_evaluation(state: RPISDState) -> dict:
    """프로그램 평가 생성 헬퍼"""
    return {
        "evaluation_model": "Kirkpatrick 4-Level",
        "levels": {
            "level_1_reaction": {
                "description": "Learner satisfaction evaluation",
                "timing": "Immediately after training",
                "methods": ["Satisfaction survey"],
            },
            "level_2_learning": {
                "description": "Learning achievement evaluation",
                "timing": "During/after training",
                "methods": ["Quiz", "Practical assessment"],
            },
            "level_3_behavior": {
                "description": "On-the-job application evaluation",
                "timing": "1-3 months after training",
                "methods": ["On-the-job application checklist"],
            },
            "level_4_results": {
                "description": "Organizational results evaluation",
                "timing": "6-12 months after training",
                "methods": ["Performance indicator analysis"],
            },
        },
    }


def _generate_adoption_decision(state: RPISDState) -> dict:
    """프로그램 채택 결정 생성 헬퍼"""
    quality = state.get("current_quality", 0.0)
    usability = state.get("usability_feedback", {})
    score = usability.get("aggregated_score", quality)

    if score >= 0.8:
        recommendation = "adopt"
        rationale = f"With a quality score of {score:.1%}, the success criteria are met and adoption of the program is recommended."
    elif score >= 0.6:
        recommendation = "conditional_adopt"
        rationale = f"With a quality score of {score:.1%}, conditional adoption is recommended. A re-review is needed after the improvements are incorporated."
    else:
        recommendation = "revise"
        rationale = f"With a quality score of {score:.1%}, further improvement is needed. Redevelopment incorporating the feedback is recommended."

    return {
        "recommendation": recommendation,
        "rationale": rationale,
        "conditions": usability.get("improvement_areas", []),
        "next_steps": usability.get("recommendations", []),
    }
