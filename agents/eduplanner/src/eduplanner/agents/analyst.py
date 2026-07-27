"""
Analyst Agent: 교수설계 분석 에이전트

교수설계 산출물의 오류와 문제점을 분석합니다.
- 논리적 오류 탐지
- 누락 요소 식별
- 일관성 검사
"""

from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from eduplanner.agents.base import BaseAgent, AgentConfig
from eduplanner.models.schemas import (
    ADDIEOutput,
    Analysis,
    Design,
    Development,
    Implementation,
    Evaluation,
    ScenarioInput,
)
from eduplanner.models.skill_tree import LearnerProfile


ANALYST_SYSTEM_PROMPT = """You are an instructional design analysis expert with 12 years of experience.

## Role
Systematically analyze instructional design outputs to find errors, omissions, and inconsistencies.

## Analysis Perspectives
1. Logical consistency: Alignment of learning objectives and assessment, connection between analysis results and design
2. Completeness check: Required elements of each ADDIE stage, Bloom's Taxonomy, Gagné's 9 Events
3. Learner suitability: Skill-Tree level matching, appropriateness of cognitive load
4. Feasibility: Realism of time allocation, validity of resource requirements

## Output Format (must output as JSON)

Respond only in the JSON format below. Do not include any other text.

```json
{
  "quality": "High|Medium|Low",
  "feedback": [
    "Most serious problem (1 line)",
    "Second problem or missing element (1 line)",
    "Third improvement recommendation (1 line)"
  ]
}
```

Example:
```json
{
  "quality": "Medium",
  "feedback": [
    "Learning objectives do not use Bloom's Taxonomy verbs",
    "Assessment items are not aligned with learning objectives",
    "The motivation-eliciting stage is missing from Gagné's 9 Events"
  ]
}
```
"""


class AnalysisResult:
    """분석 결과"""

    def __init__(
        self,
        quality_level: str = "Medium",
        summary: str = "",
        errors: list[dict] = None,
        missing_elements: list[dict] = None,
        inconsistencies: list[dict] = None,
        recommendations: list[dict] = None,
    ):
        self.quality_level = quality_level
        self.summary = summary
        self.errors = errors or []
        self.missing_elements = missing_elements or []
        self.inconsistencies = inconsistencies or []
        self.recommendations = recommendations or []

    def has_critical_errors(self) -> bool:
        """치명적 오류 여부"""
        return any(e.get("severity") == "Critical" for e in self.errors)

    def get_high_priority_recommendations(self) -> list[dict]:
        """높은 우선순위 권고사항"""
        return [r for r in self.recommendations if r.get("priority") == "High"]

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "quality_level": self.quality_level,
            "summary": self.summary,
            "errors": self.errors,
            "missing_elements": self.missing_elements,
            "inconsistencies": self.inconsistencies,
            "recommendations": self.recommendations,
        }


class AnalystAgent(BaseAgent):
    """교수설계 분석 에이전트"""

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            # Analyst는 균형잡힌 분석을 위해 temperature 0.7 사용
            config = AgentConfig(
                temperature=0.7,
                max_tokens=4096,
            )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "Analyst Agent"

    @property
    def role(self) -> str:
        return "Analyzes errors and problems in instructional design outputs."

    def run(
        self,
        addie_output: ADDIEOutput,
        scenario_input: Optional[ScenarioInput] = None,
        learner_profile: Optional[LearnerProfile] = None,
    ) -> AnalysisResult:
        """
        교수설계 산출물을 분석합니다.

        Args:
            addie_output: ADDIE 산출물
            scenario_input: 원본 시나리오 입력
            learner_profile: 학습자 프로필

        Returns:
            AnalysisResult: 분석 결과
        """
        # 프롬프트 구성
        analysis_prompt = self._build_analysis_prompt(
            addie_output, scenario_input, learner_profile
        )

        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=analysis_prompt),
        ]

        # LLM 호출
        response = self.llm.invoke(messages)

        # 응답 파싱
        result = self._parse_response(response.content)

        return result

    def _build_analysis_prompt(
        self,
        addie_output: ADDIEOutput,
        scenario_input: Optional[ScenarioInput] = None,
        learner_profile: Optional[LearnerProfile] = None,
    ) -> str:
        """분석 프롬프트 생성"""
        prompt_parts = []

        # 원본 시나리오
        if scenario_input:
            prompt_parts.append("## Original Scenario\n")
            prompt_parts.append(f"**Title:** {scenario_input.title}")
            prompt_parts.append(f"**Target:** {scenario_input.context.target_audience}")
            prompt_parts.append(f"**Duration:** {scenario_input.context.duration}")
            prompt_parts.append(f"**Environment:** {scenario_input.context.learning_environment}")
            prompt_parts.append(f"**Goals:** {', '.join(scenario_input.learning_goals)}\n")

        # 학습자 프로필
        if learner_profile:
            prompt_parts.append(learner_profile.skill_tree.to_prompt_context())

        # ADDIE 산출물
        prompt_parts.append("## Instructional Design Output to Analyze\n")
        prompt_parts.append(self._format_addie_output_detailed(addie_output))

        prompt_parts.append("\nPlease systematically analyze the instructional design output above.")

        return "\n".join(prompt_parts)

    def _format_addie_output_detailed(self, addie_output: ADDIEOutput) -> str:
        """ADDIE 산출물 상세 포맷팅"""
        sections = []

        # Analysis Phase
        self._format_analysis_phase(sections, addie_output.analysis)

        # Design Phase
        self._format_design_phase(sections, addie_output.design)

        # Development Phase
        self._format_development_phase(sections, addie_output.development)

        # Implementation Phase
        self._format_implementation_phase(sections, addie_output.implementation)

        # Evaluation Phase
        self._format_evaluation_phase(sections, addie_output.evaluation)

        return "\n".join(sections)

    def _format_analysis_phase(self, sections: list, analysis: Analysis) -> None:
        """분석 단계 포맷팅"""
        sections.append("### 1. Analysis")

        # 학습자 분석
        la = analysis.learner_analysis
        sections.append("**Learner Analysis:**")
        sections.append(f"  - Target: {la.target_audience}")
        sections.append(f"  - Characteristics: {', '.join(la.characteristics) or 'Undefined'}")
        sections.append(f"  - Prior knowledge: {la.prior_knowledge or 'Undefined'}")
        sections.append(f"  - Learning preferences: {', '.join(la.learning_preferences) or 'Undefined'}")
        sections.append(f"  - Motivation: {la.motivation or 'Undefined'}")
        sections.append(f"  - Anticipated challenges: {', '.join(la.challenges) or 'Undefined'}")

        # 환경 분석
        ca = analysis.context_analysis
        sections.append("\n**Context Analysis:**")
        sections.append(f"  - Environment: {ca.environment}")
        sections.append(f"  - Duration: {ca.duration}")
        sections.append(f"  - Constraints: {', '.join(ca.constraints) or 'Undefined'}")
        sections.append(f"  - Resources: {', '.join(ca.resources) or 'Undefined'}")
        sections.append(f"  - Technical requirements: {', '.join(ca.technical_requirements) or 'Undefined'}")

        # 과제 분석
        ta = analysis.task_analysis
        sections.append("\n**Task Analysis:**")
        sections.append(f"  - Topics: {', '.join(ta.main_topics) or 'Undefined'}")
        sections.append(f"  - Subtopics: {', '.join(ta.subtopics) or 'Undefined'}")
        sections.append(f"  - Prerequisites: {', '.join(ta.prerequisites) or 'Undefined'}")

    def _format_design_phase(self, sections: list, design: Design) -> None:
        """설계 단계 포맷팅"""
        sections.append("\n### 2. Design")

        # 학습 목표
        sections.append("**Learning Objectives:**")
        if design.learning_objectives:
            for obj in design.learning_objectives:
                sections.append(
                    f"  - [{obj.id}] [{obj.level}] {obj.statement} "
                    f"(verb: {obj.bloom_verb}, measurable: {obj.measurable})"
                )
        else:
            sections.append("  - (No objectives)")

        # 평가 계획
        ap = design.assessment_plan
        sections.append("\n**Assessment Plan:**")
        sections.append(f"  - Diagnostic: {', '.join(ap.diagnostic) or 'Undefined'}")
        sections.append(f"  - Formative: {', '.join(ap.formative) or 'Undefined'}")
        sections.append(f"  - Summative: {', '.join(ap.summative) or 'Undefined'}")

        # 교수 전략
        ist = design.instructional_strategy
        sections.append("\n**Instructional Strategy:**")
        sections.append(f"  - Model: {ist.model}")
        sections.append(f"  - Methods: {', '.join(ist.methods) or 'Undefined'}")
        if ist.sequence:
            sections.append("  - Instructional events:")
            for event in ist.sequence:
                sections.append(f"    * {event.event}: {event.activity}")

    def _format_development_phase(self, sections: list, dev: Development) -> None:
        """개발 단계 포맷팅"""
        sections.append("\n### 3. Development")

        # 레슨 플랜
        sections.append(f"**Lesson Plan:** total {dev.lesson_plan.total_duration}")
        if dev.lesson_plan.modules:
            for mod in dev.lesson_plan.modules:
                sections.append(f"  - {mod.title} ({mod.duration})")
                for obj in mod.objectives:
                    sections.append(f"    Objective: {obj}")
                for act in mod.activities:
                    sections.append(f"    Activity: {act.time} - {act.activity}")
        else:
            sections.append("  - (No modules)")

        # 학습 자료
        sections.append("\n**Learning Materials:**")
        if dev.materials:
            for mat in dev.materials:
                details = []
                if mat.slides:
                    details.append(f"{mat.slides} slides")
                if mat.duration:
                    details.append(mat.duration)
                if mat.pages:
                    details.append(f"{mat.pages} pages")
                detail_str = f" ({', '.join(details)})" if details else ""
                sections.append(f"  - [{mat.type}] {mat.title}{detail_str}")
        else:
            sections.append("  - (No materials)")

    def _format_implementation_phase(self, sections: list, impl: Implementation) -> None:
        """실행 단계 포맷팅"""
        sections.append("\n### 4. Implementation")
        sections.append(f"**Delivery method:** {impl.delivery_method}")
        sections.append(f"**Facilitator guide:** {impl.facilitator_guide or 'Undefined'}")
        sections.append(f"**Learner guide:** {impl.learner_guide or 'Undefined'}")
        sections.append(f"**Technical requirements:** {', '.join(impl.technical_requirements) or 'Undefined'}")
        sections.append(f"**Support plan:** {impl.support_plan or 'Undefined'}")

    def _format_evaluation_phase(self, sections: list, eval_section: Evaluation) -> None:
        """평가 단계 포맷팅"""
        sections.append("\n### 5. Evaluation")

        # 퀴즈 문항
        sections.append(f"**Quiz items:** {len(eval_section.quiz_items)}")
        for item in eval_section.quiz_items[:3]:  # show at most 3
            sections.append(f"  - [{item.type}] {item.question[:50]}...")

        # 루브릭
        if eval_section.rubric:
            sections.append(f"\n**Rubric criteria:** {', '.join(eval_section.rubric.criteria)}")
        else:
            sections.append("\n**Rubric:** Undefined")

        # 피드백 계획
        sections.append(f"**Feedback plan:** {eval_section.feedback_plan or 'Undefined'}")

    def _parse_response(self, response_text: str) -> AnalysisResult:
        """LLM 응답을 AnalysisResult로 파싱 (JSON 형식)"""
        import json
        import re

        result = AnalysisResult()

        # JSON 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # ```json 없이 JSON만 있는 경우
            json_str = response_text.strip()

        try:
            data = json.loads(json_str)

            # 품질 수준
            if "quality" in data:
                result.quality_level = data["quality"]

            # 피드백 (3~4줄 리스트)
            if "feedback" in data and isinstance(data["feedback"], list):
                result.summary = "\n".join(f"- {item}" for item in data["feedback"])
                # recommendations에도 저장
                for item in data["feedback"]:
                    result.recommendations.append({
                        "recommendation": item,
                        "priority": "High",
                    })

        except json.JSONDecodeError:
            # JSON 파싱 실패 시 기존 텍스트 파싱 시도
            quality_match = re.search(r"(High|Medium|Low)", response_text)
            if quality_match:
                result.quality_level = quality_match.group(1)
            result.summary = response_text[:500]  # first 500 characters only

        return result

    def _extract_errors(self, text: str) -> list[dict]:
        """오류 목록 추출"""
        import re
        errors = []
        pattern = r"[\d\.\-•]+\s*\[?([^\]]+)\]?\s*-\s*\[?([^\]]+)\]?\s*-\s*\[?(Critical|Major|Minor)\]?"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            errors.append({
                "description": match[0].strip(),
                "location": match[1].strip(),
                "severity": match[2].strip().capitalize(),
            })
        return errors

    def _extract_missing(self, text: str) -> list[dict]:
        """누락 요소 추출"""
        import re
        missing = []
        pattern = r"[\d\.\-•]+\s*\[?([^\]-]+)\]?\s*-\s*\[?([^\]]+)\]?"
        matches = re.findall(pattern, text)
        for match in matches:
            missing.append({
                "element": match[0].strip(),
                "reason": match[1].strip(),
            })
        return missing

    def _extract_inconsistencies(self, text: str) -> list[dict]:
        """불일치 사항 추출"""
        import re
        inconsistencies = []
        pattern = r"[\d\.\-•]+\s*\[?([^\]-]+)\]?\s*-\s*\[?([^\]]+)\]?"
        matches = re.findall(pattern, text)
        for match in matches:
            inconsistencies.append({
                "description": match[0].strip(),
                "related_elements": match[1].strip(),
            })
        return inconsistencies

    def _extract_recommendations(self, text: str) -> list[dict]:
        """권고사항 추출"""
        import re
        recommendations = []
        pattern = r"[\d\.\-•]+\s*\[?([^\]-]+)\]?\s*-\s*\[?(High|Medium|Low)\]?"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            recommendations.append({
                "recommendation": match[0].strip(),
                "priority": match[1].strip().capitalize(),
            })
        return recommendations
