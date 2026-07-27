"""EduPlanner 입출력 스키마 정의"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class ContextInfo(BaseModel):
    """학습 맥락 정보"""
    # 기존 필드 (optional로 변경)
    target_audience: Optional[str] = Field(None, description="Target learners")
    prior_knowledge: Optional[str] = Field(None, description="Prior knowledge level")
    duration: Optional[str] = Field(None, description="Learning time")
    learning_environment: Optional[str] = Field(None, description="Learning environment")
    class_size: Optional[str] = Field(None, description="Number of learners (string)")
    additional_context: Optional[str] = Field(None, description="Additional context")
    # IDLD 시나리오 추가 필드
    institution_type: Optional[str] = Field(None, description="Institution type")
    learner_age: Optional[str] = Field(None, description="Learner age")
    learner_education: Optional[str] = Field(None, description="Learner education level")
    learner_role: Optional[str] = Field(None, description="Learner role")
    domain_expertise: Optional[str] = Field(None, description="Domain expertise")


class Constraints(BaseModel):
    """제약 조건"""
    budget: Optional[str] = Field(None, description="Budget level")
    resources: Optional[list[str]] = Field(None, description="Available materials")
    accessibility: Optional[Any] = Field(None, description="Accessibility requirements")
    language: str = Field(default="ko", description="Content language")
    # IDLD 시나리오 추가 필드
    tech_requirements: Optional[str] = Field(None, description="Technical requirements")
    assessment_type: Optional[str] = Field(None, description="Assessment type")


class ScenarioInput(BaseModel):
    """교수설계 시나리오 입력"""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    variant_type: Optional[str] = Field(None, description="Scenario type")
    title: str = Field(..., description="Scenario title")
    context: ContextInfo = Field(..., description="Learning context")
    learning_goals: list[str] = Field(..., description="Learning objectives")
    constraints: Optional[Constraints] = Field(None, description="Constraints")
    difficulty: Optional[str] = Field(None, description="Difficulty")
    domain: Optional[str] = Field(None, description="Instructional domain")


class LearnerAnalysis(BaseModel):
    """학습자 분석 (Item 5: 학습자 분석)"""
    target_audience: str
    characteristics: list[str] = Field(default_factory=list)
    prior_knowledge: Optional[str] = None
    learning_preferences: list[str] = Field(default_factory=list)
    motivation: Optional[str] = None
    challenges: list[str] = Field(default_factory=list)


class ContextAnalysis(BaseModel):
    """환경 분석 (Item 6: 환경 분석 - 물리/조직/기술 환경)"""
    environment: str
    duration: str
    constraints: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    technical_requirements: list[str] = Field(default_factory=list)
    # Item 6 확장: 물리적, 조직적, 기술적 환경 상세
    physical_environment: Optional[str] = Field(default=None, description="Physical environment analysis")
    organizational_environment: Optional[str] = Field(default=None, description="Organizational environment analysis")
    technology_environment: Optional[str] = Field(default=None, description="Technology environment analysis")


class TaskAnalysis(BaseModel):
    """과제 분석 (Item 7-10: 과제 및 목표분석)"""
    main_topics: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    # Item 7: 초기 학습목표 분석
    initial_learning_objectives: Optional[str] = Field(default=None, description="Initial learning objective analysis")
    # Item 8: 하위 기능 분석
    sub_skills: list[str] = Field(default_factory=list, description="Sub-skill/sub-function analysis")
    # Item 9: 출발점 행동 분석
    entry_behaviors: Optional[str] = Field(default=None, description="Entry behavior analysis")
    # Item 10: 과제분석 결과 검토·정리
    task_analysis_review: Optional[str] = Field(default=None, description="Task analysis result review and summary")


class NeedsAnalysis(BaseModel):
    """요구분석 (Item 1-4)"""
    # Item 1: 문제 확인 및 정의
    problem_definition: Optional[str] = Field(default=None, description="Problem identification and definition")
    # Item 2: 차이분석 (현재-목표 상태 격차)
    gap_analysis: Optional[str] = Field(default=None, description="Gap analysis between current and target performance")
    # Item 3: 수행분석
    performance_analysis: Optional[str] = Field(default=None, description="Performance analysis results")
    # Item 4: 요구 우선순위 결정
    needs_prioritization: Optional[str] = Field(default=None, description="Needs prioritization")


class Analysis(BaseModel):
    """분석 단계 산출물 (Item 1-10)"""
    learner_analysis: LearnerAnalysis
    context_analysis: ContextAnalysis
    task_analysis: TaskAnalysis
    # 요구분석 (Item 1-4) - 신규 추가
    needs_analysis: Optional[NeedsAnalysis] = Field(default=None, description="Needs analysis (Items 1-4)")


class LearningObjective(BaseModel):
    """학습 목표"""
    id: str
    level: str = Field(..., description="Bloom's Taxonomy level")
    statement: str
    bloom_verb: str
    measurable: bool = True


class AssessmentPlan(BaseModel):
    """평가 계획"""
    formative: list[str] = Field(default_factory=list)
    summative: list[str] = Field(default_factory=list)
    diagnostic: list[str] = Field(default_factory=list)


class InstructionalEvent(BaseModel):
    """교수 사태 (Gagné's 9 Events)"""
    event: str
    activity: str
    duration: Optional[str] = None
    resources: list[str] = Field(default_factory=list)


class InstructionalStrategy(BaseModel):
    """교수 전략 (Item 13-17)"""
    model: str = Field(default="Gagné's 9 Events")
    sequence: list[InstructionalEvent] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    # Item 13: 교수 내용 선정 (기존 methods로 커버)
    # Item 14: 교수적 전략 수립
    instructional_strategies: Optional[str] = Field(default=None, description="Instructional strategy development")
    # Item 15: 비교수적 전략 수립 (신규)
    non_instructional_strategies: Optional[str] = Field(default=None, description="Non-instructional strategies (motivation, self-directed learning promotion, etc.)")
    # Item 16: 매체 선정과 활용 계획 (신규)
    media_selection: list[str] = Field(default_factory=list, description="Media selection and utilization plan")
    # Item 17: 학습활동 및 시간 구조화 (기존 sequence로 커버)


class PrototypeDesign(BaseModel):
    """프로토타입 구조 설계 (Item 18)"""
    # Item 18: 스토리보드/화면 흐름 설계
    storyboard: Optional[str] = Field(default=None, description="Storyboard design")
    screen_flow: list[str] = Field(default_factory=list, description="Screen flow design")
    navigation_structure: Optional[str] = Field(default=None, description="Navigation structure")


class Design(BaseModel):
    """설계 단계 산출물 (Item 11-18)"""
    learning_objectives: list[LearningObjective]
    assessment_plan: AssessmentPlan
    instructional_strategy: InstructionalStrategy
    # Item 18: 프로토타입 구조 설계 (신규)
    prototype_design: Optional[PrototypeDesign] = Field(default=None, description="Prototype structure design (storyboard/screen flow)")


class Activity(BaseModel):
    """학습 활동"""
    time: str
    activity: str
    description: Optional[str] = None
    resources: list[str] = Field(default_factory=list)


class Module(BaseModel):
    """학습 모듈"""
    title: str
    duration: str
    objectives: list[str] = Field(default_factory=list)
    activities: list[Activity] = Field(default_factory=list)


class LessonPlan(BaseModel):
    """레슨 플랜"""
    total_duration: str
    modules: list[Module] = Field(default_factory=list)


class SlideContent(BaseModel):
    """개별 슬라이드 콘텐츠"""
    slide_number: int = Field(..., description="Slide number")
    title: str = Field(..., description="Slide title")
    bullet_points: list[str] = Field(default_factory=list, description="Key content (3-5 items)")
    speaker_notes: Optional[str] = Field(default=None, description="Speaker notes")
    visual_suggestion: Optional[str] = Field(default=None, description="Recommended visual materials")


class Material(BaseModel):
    """학습 자료"""
    type: str
    title: str
    description: Optional[str] = None
    slides: Optional[int] = None
    duration: Optional[str] = None
    questions: Optional[int] = None
    pages: Optional[int] = None
    content: Optional[str] = Field(default=None, description="Actual learning material content (handout text, slide outline, etc.)")
    slide_contents: Optional[list[SlideContent]] = Field(default=None, description="Detailed content per slide (for presentations)")


class Development(BaseModel):
    """개발 단계 산출물 (Item 19-23)"""
    lesson_plan: LessonPlan
    # Item 19: 학습자용 자료 개발 (기존 materials)
    materials: list[Material] = Field(default_factory=list)
    # Item 20: 교수자용 매뉴얼 개발 (신규)
    facilitator_manual: Optional[str] = Field(default=None, description="Instructor manual")
    # Item 21: 운영자용 매뉴얼 개발 (신규)
    operator_manual: Optional[str] = Field(default=None, description="Operator manual")
    # Item 22: 평가 도구·문항 개발 (evaluation 단계로 이동, 여기서는 평가 도구 명세)
    assessment_tools: list[str] = Field(default_factory=list, description="Assessment tool specification")
    # Item 23: 전문가 검토 (신규)
    expert_review_plan: Optional[str] = Field(default=None, description="Expert review plan and feedback implementation approach")


class Implementation(BaseModel):
    """실행 단계 산출물 (Item 24-27)"""
    delivery_method: str
    facilitator_guide: Optional[str] = None
    learner_guide: Optional[str] = None
    technical_requirements: list[str] = Field(default_factory=list)
    support_plan: Optional[str] = None
    # Item 24: 교수자·운영자 오리엔테이션 (신규)
    orientation_plan: Optional[str] = Field(default=None, description="Instructor/operator orientation plan")
    # Item 25: 시스템/환경 점검 (신규)
    system_check_plan: Optional[str] = Field(default=None, description="System/environment check plan")
    # Item 26: 프로토타입 실행 (신규)
    pilot_execution_plan: Optional[str] = Field(default=None, description="Prototype/pilot execution plan")
    # Item 27: 운영 모니터링 및 지원 (신규)
    monitoring_plan: Optional[str] = Field(default=None, description="Operation monitoring and support plan")


class QuizItem(BaseModel):
    """퀴즈 문항"""
    id: str
    question: str
    type: str
    options: list[str] = Field(default_factory=list)
    answer: str
    explanation: Optional[str] = None
    objective_id: Optional[str] = None
    difficulty: Optional[str] = None


class Rubric(BaseModel):
    """평가 루브릭"""
    criteria: list[str] = Field(default_factory=list)
    levels: dict[str, str] = Field(default_factory=dict)


class Evaluation(BaseModel):
    """평가 단계 산출물 (Item 28-33)"""
    quiz_items: list[QuizItem] = Field(default_factory=list)
    rubric: Optional[Rubric] = None
    feedback_plan: Optional[str] = None
    # Item 28: 파일럿/초기 실행 중 자료 수집 (신규)
    pilot_data_collection: Optional[str] = Field(default=None, description="Data collection plan during pilot/initial execution")
    # Item 29: 형성평가 결과 기반 1차 프로그램 개선 (신규)
    formative_improvement: Optional[str] = Field(default=None, description="First program improvement plan based on formative evaluation results")
    # Item 30: 총괄 평가 문항 개발 (quiz_items로 커버)
    # Item 31: 총괄평가 시행 및 프로그램 효과 분석 (신규)
    summative_evaluation_plan: Optional[str] = Field(default=None, description="Summative evaluation implementation and program effectiveness analysis plan")
    # Item 32: 프로그램 채택 여부 결정 (신규)
    adoption_decision_criteria: Optional[str] = Field(default=None, description="Program adoption decision criteria")
    # Item 33: 프로그램 개선 (신규)
    program_improvement: Optional[str] = Field(default=None, description="Program improvement and feedback loop plan")


class ADDIEOutput(BaseModel):
    """ADDIE 5단계 산출물"""
    analysis: Analysis
    design: Design
    development: Development
    implementation: Implementation
    evaluation: Evaluation

    def to_standard_dict(self) -> dict:
        """
        표준 ADDIE 33개 소항목 스키마로 변환 (docs/addie_output_schema.json 준수)

        평가기(evaluator)가 기대하는 필드 경로에 맞게 변환합니다.
        """
        # Analysis 단계 변환
        la = self.analysis.learner_analysis
        ca = self.analysis.context_analysis
        ta = self.analysis.task_analysis
        na = self.analysis.needs_analysis

        analysis_dict = {
            # A1: 요구분석 (소항목 1-4)
            "needs_analysis": {
                # [1] 문제 확인 및 정의
                "problem_definition": (na.problem_definition if na else "") or ta.task_analysis_review or "",
                # [2] 차이분석
                "gap_analysis": [{"description": na.gap_analysis}] if na and na.gap_analysis else [],
                # [3] 수행분석
                "performance_analysis": na.performance_analysis if na else "",
                # [4] 요구 우선순위 결정
                "priority_matrix": {
                    "prioritization": na.needs_prioritization if na else "",
                    "high_priority": ta.main_topics[:2] if ta.main_topics else [],
                },
            },
            # A2: 학습자 및 환경분석 (소항목 5-6)
            "learner_analysis": {
                "target_audience": la.target_audience,
                "characteristics": la.characteristics,
                "prior_knowledge": la.prior_knowledge or "",
                "learning_preferences": la.learning_preferences,
                "motivation": la.motivation or "",
            },
            "context_analysis": {
                "environment": ca.environment,
                "constraints": ca.constraints,
                "resources": ca.resources,
                "technical_requirements": ca.technical_requirements,
            },
            # A3: 과제 및 목표분석 (소항목 7-10)
            "task_analysis": {
                # [7] 초기 학습목표 분석
                "initial_objectives": [ta.initial_learning_objectives] if ta.initial_learning_objectives else ta.main_topics,
                # [8] 하위 기능 분석
                "subtopics": ta.sub_skills if ta.sub_skills else ta.subtopics,
                # [9] 출발점 행동 분석
                "prerequisites": ta.prerequisites,
                # [10] 과제분석 결과 검토·정리
                "review_summary": ta.task_analysis_review or f"Main topics: {', '.join(ta.main_topics[:3])}",
            },
        }

        # Design 단계 변환
        d = self.design
        ist = d.instructional_strategy
        pd = d.prototype_design

        # 학습 활동 생성
        learning_activities = []
        for event in ist.sequence:
            learning_activities.append({
                "activity_name": event.event,
                "duration": event.duration or "",
                "description": event.activity,
                "materials": event.resources,
            })

        design_dict = {
            # [11] 학습목표 정교화
            "learning_objectives": [
                {
                    "id": obj.id,
                    "level": obj.level,
                    "statement": obj.statement,
                    "bloom_verb": obj.bloom_verb,
                    "measurable": obj.measurable,
                }
                for obj in d.learning_objectives
            ],
            # [12] 평가 계획 수립
            "assessment_plan": {
                "formative": [{"description": f} for f in d.assessment_plan.formative],
                "summative": [{"description": s} for s in d.assessment_plan.summative],
                "assessment_criteria": d.assessment_plan.diagnostic,
            },
            # [13] 교수 내용 선정
            "content_structure": {
                "modules": [m for m in ist.methods],
                "topics": [],
                "sequencing": ist.model,
            },
            # [14] 교수적 전략 수립
            "instructional_strategies": {
                "methods": ist.methods,
                "activities": [e.activity for e in ist.sequence[:5]],
                "rationale": ist.instructional_strategies or f"Model: {ist.model}",
            },
            # [15] 비교수적 전략 수립
            "non_instructional_strategies": {
                "motivation_strategies": [ist.non_instructional_strategies] if ist.non_instructional_strategies else [],
                "self_directed_learning": [],
                "support_strategies": [],
            },
            # [16] 매체 선정과 활용 계획
            "media_selection": {
                "media_types": ist.media_selection,
                "tools": [],
                "utilization_plan": "",
            },
            # [17] 학습활동 및 시간 구조화
            "learning_activities": learning_activities,
            # [18] 스토리보드/화면 흐름 설계
            "storyboard": {
                "screens": pd.screen_flow if pd else [],
                "navigation_flow": pd.navigation_structure if pd else "",
                "interactions": [],
            } if pd else {"screens": [], "navigation_flow": "", "interactions": []},
        }

        # Development 단계 변환
        dev = self.development

        # learner_materials 변환
        learner_materials = []
        for mat in dev.materials:
            learner_materials.append({
                "title": mat.title,
                "type": mat.type,
                "content": mat.description or "",
                "format": "PDF/PPT" if mat.slides or mat.pages else "Other",
            })

        development_dict = {
            # [19] 학습자용 자료 개발
            "learner_materials": learner_materials,
            # [20] 교수자용 매뉴얼 개발
            "instructor_guide": {
                "overview": dev.facilitator_manual or "",
                "session_guides": [mod.title for mod in dev.lesson_plan.modules],
                "facilitation_tips": ["Encourage learner participation", "Use questioning"],
                "troubleshooting": ["Responding to technical problems"],
            },
            # [21] 운영자용 매뉴얼 개발
            "operator_manual": {
                "system_setup": dev.operator_manual or "System setup guide",
                "operation_procedures": ["Registration management", "Attendance management"],
                "support_procedures": ["Responding to learner inquiries"],
                "escalation_process": "Report to the person in charge when an issue occurs",
            },
            # [22] 평가 도구·문항 개발
            "assessment_tools": [
                {
                    "item_id": tool,
                    "type": "Assessment tool",
                    "question": tool,
                    "aligned_objective": "",
                    "scoring_criteria": "",
                }
                for tool in dev.assessment_tools
            ],
            # [23] 전문가 검토
            "expert_review": {
                "reviewers": ["Content expert"],
                "review_criteria": ["Content accuracy", "Instructional design appropriateness"],
                "feedback_summary": dev.expert_review_plan or "",
                "revisions_made": [],
            },
        }

        # Implementation 단계 변환
        impl = self.implementation

        implementation_dict = {
            # [24] 교수자·운영자 오리엔테이션
            "instructor_orientation": {
                "orientation_objectives": ["Understanding the program", "Familiarity with operation procedures"],
                "schedule": impl.orientation_plan or "One week in advance",
                "materials": ["Facilitator guide", "Operation manual"],
                "competency_checklist": ["Content comprehension", "Facilitation ability"],
            },
            # [25] 시스템/환경 점검
            "system_check": {
                "checklist": impl.technical_requirements or ["Network connection", "Equipment check"],
                "technical_validation": impl.system_check_plan or "System testing completed",
                "contingency_plans": ["Establish contingency response plan"],
            },
            # [26] 프로토타입 실행
            "prototype_execution": {
                "pilot_scope": impl.pilot_execution_plan or "Small-scale pilot test",
                "participants": "Around 10 people",
                "execution_log": [],
                "issues_encountered": [],
            },
            # [27] 운영 모니터링 및 지원
            "monitoring": {
                "monitoring_criteria": [impl.monitoring_plan] if impl.monitoring_plan else ["Learning progress", "Participation rate"],
                "support_channels": ["Email", "Phone"],
                "issue_resolution_log": [],
                "real_time_adjustments": [],
            },
        }

        # Evaluation 단계 변환
        ev = self.evaluation

        # 퀴즈 아이템 변환
        quiz_tools = [
            {
                "item_id": q.id,
                "type": q.type,
                "question": q.question,
                "scoring_rubric": q.explanation or "",
            }
            for q in ev.quiz_items
        ]

        evaluation_dict = {
            # E1: 형성평가 (소항목 28-29)
            "formative": {
                # [28] 파일럿/초기 실행 중 자료 수집
                "data_collection": {
                    "methods": ["Survey", "Observation", "Interview"],
                    "learner_feedback": [],
                    "performance_data": {},
                    "observations": [ev.pilot_data_collection] if ev.pilot_data_collection else [],
                },
                # [29] 형성평가 결과 기반 1차 프로그램 개선
                "improvements": [
                    {
                        "issue_identified": ev.formative_improvement or "Identify items requiring improvement",
                        "improvement_action": "Execute improvement actions",
                        "priority": "High",
                    }
                ] if ev.formative_improvement else [],
            },
            # E2: 총괄평가 및 채택 결정 (소항목 30-32)
            "summative": {
                # [30] 총괄 평가 문항 개발
                "assessment_tools": quiz_tools,
                # [31] 총괄평가 시행 및 프로그램 효과 분석
                "effectiveness_analysis": {
                    "learning_outcomes": {},
                    "goal_achievement_rate": "",
                    "statistical_analysis": ev.summative_evaluation_plan or "",
                    "recommendations": [],
                },
                # [32] 프로그램 채택 여부 결정
                "adoption_decision": {
                    "decision": "",
                    "rationale": ev.adoption_decision_criteria or "",
                    "conditions": [],
                    "stakeholder_approval": "Awaiting approval",
                },
            },
            # [33] E3: 프로그램 개선 및 환류
            "improvement_plan": {
                "feedback_summary": ev.feedback_plan or "",
                "improvement_areas": [ev.program_improvement] if ev.program_improvement else [],
                "action_items": [],
                "feedback_loop": "Reflect evaluation results in the next iteration of the course",
                "next_iteration_goals": [],
            },
        }

        return {
            "analysis": analysis_dict,
            "design": design_dict,
            "development": development_dict,
            "implementation": implementation_dict,
            "evaluation": evaluation_dict,
        }


class ToolCall(BaseModel):
    """도구 호출 기록"""
    step: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: Optional[int] = None
    success: bool = True
    output_data: Optional[dict[str, Any]] = None
    feedback: Optional[dict[str, Any]] = None


class Trajectory(BaseModel):
    """궤적 기록"""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)


class Metadata(BaseModel):
    """메타데이터"""
    model: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    execution_time_seconds: float = 0.0
    cost_usd: float = 0.0
    agent_version: str = "0.1.0"
    iterations: int = 0


class AgentResult(BaseModel):
    """Agent 최종 출력"""
    scenario_id: str
    agent_id: str = "eduplanner"
    timestamp: datetime = Field(default_factory=datetime.now)
    addie_output: ADDIEOutput
    trajectory: Trajectory = Field(default_factory=Trajectory)
    metadata: Metadata


class EvaluationFeedback(BaseModel):
    """평가 에이전트 피드백 (ADDIE Rubric 13항목 기반)"""
    score: float = Field(..., ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    addie_scores: dict[str, float] = Field(
        default_factory=dict,
        description="ADDIE Rubric 13-item scores (A1-A3, D1-D3, Dev1-Dev2, I1-I2, E1-E3)"
    )
    weighted_score: Optional[float] = Field(
        default=None,
        description="Weighted score (0-100)"
    )
