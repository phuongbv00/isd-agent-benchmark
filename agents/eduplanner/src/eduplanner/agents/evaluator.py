"""
Evaluator Agent: 교수설계 평가 에이전트

ADDIE Rubric 13항목 평가 시스템을 사용하여 교수설계의 품질을 평가합니다.
- Analysis (A1-A3): 학습자/환경/요구 분석
- Design (D1-D3): 목표/평가/전략 설계
- Development (Dev1-Dev2): 자료 개발
- Implementation (I1-I2): 실행 계획
- Evaluation (E1-E3): 평가 도구
"""

import sys
from pathlib import Path
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from eduplanner.agents.base import BaseAgent, AgentConfig
from eduplanner.models.schemas import ADDIEOutput, EvaluationFeedback
from eduplanner.models.skill_tree import LearnerProfile

# ADDIE Rubric 정의 임포트
_project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(_project_root / "evaluator" / "src"))
from isd_evaluator.rubrics.addie_definitions import (
    ADDIE_RUBRIC_DEFINITIONS, DEFAULT_PHASE_WEIGHTS, MAX_SCORE_PER_ITEM
)
from isd_evaluator.models import ADDIEPhase


EVALUATOR_SYSTEM_PROMPT = """You are a veteran instructional design expert with 20 years of experience.

## Role
Systematically evaluate instructional design outputs according to the 13-item ADDIE Rubric evaluation system.

## ADDIE Rubric Evaluation Criteria (0-10 points per item)

### Analysis - weight 25%
- **A1. Appropriateness of learner analysis**: Identify learners' level, prior knowledge, characteristics, motivation, needs, etc.
- **A2. Validity of performance context and environment analysis**: Analyze conditions such as facilities, devices, time, and technology environment
- **A3. Clarity of needs analysis and performance gap definition**: Analyze the gap between the current state and the target state

### Design - weight 25%
- **D1. Alignment between learning objectives and needs analysis**: State behavior-centered objectives clearly
- **D2. Validity and coherence of assessment design**: Assessment methods for determining achievement of learning objectives
- **D3. Theoretical appropriateness of instructional strategy and learning experience design**: Appropriate instructional methods, activities, and media use

### Development - weight 20%
- **Dev1. Prototype development**: Development of learner materials, instructor/operator manuals, and assessment tools
- **Dev2. Review and revision of development results**: Reflecting and incorporating feedback through expert review

### Implementation - weight 15%
- **I1. Program execution preparation**: Orientation and system/environment check
- **I2. Program execution**: Prototype execution and operation monitoring

### Evaluation - weight 15%
- **E1. Formative evaluation**: Data collection during pilot/initial execution and first-round improvement
- **E2. Summative evaluation and adoption decision**: Conducting summative evaluation, effectiveness analysis, and adoption decision
- **E3. Program improvement and feedback loop**: Final program improvement and feedback system

## Scoring Criteria (apply strictly)
- **9-10 points (Excellent)**: Theory and practice are very appropriately reflected and immediately applicable
- **7-8 points (Good)**: Theory and practice are appropriately reflected and applicable
- **5-6 points (Average)**: Basic elements are met, but specificity/alignment is partly lacking
- **3-4 points (Insufficient)**: Core elements are partly missing or feasibility is low
- **1-2 points (Absent)**: The element is barely presented

## Output Format
**Output only in the JSON format below. Output only JSON with no other text.**

```json
{
  "addie_scores": {
    "A1": <0.0-10.0>, "A2": <0.0-10.0>, "A3": <0.0-10.0>,
    "D1": <0.0-10.0>, "D2": <0.0-10.0>, "D3": <0.0-10.0>,
    "Dev1": <0.0-10.0>, "Dev2": <0.0-10.0>,
    "I1": <0.0-10.0>, "I2": <0.0-10.0>,
    "E1": <0.0-10.0>, "E2": <0.0-10.0>, "E3": <0.0-10.0>
  },
  "strengths": [
    "<specific description of a strength>",
    "<specific description of a strength>"
  ],
  "weaknesses": [
    "<specific description of a weakness>",
    "<specific description of a weakness>"
  ],
  "suggestions": [
    "<specific improvement suggestion>",
    "<specific improvement suggestion>"
  ]
}
```

**Important:**
- Each item score must be **between 0.0 and 10.0 with 1 decimal place** (e.g., 7.5, 8.2, 6.8)
- **Always use decimals** for fine-grained evaluation
- strengths, weaknesses, and suggestions must each have a minimum of 2 and a maximum of 5 entries
- Do not include any explanation or text other than the JSON
"""


class EvaluatorAgent(BaseAgent):
    """교수설계 평가 에이전트"""

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            # Evaluator: temperature 0.7로 변경하여 다양한 평가 허용
            config = AgentConfig(
                temperature=0.7,
                max_tokens=4096,
            )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "Evaluator Agent"

    @property
    def role(self) -> str:
        return "Evaluates instructional design quality using the 13-item ADDIE Rubric evaluation system."

    def run(
        self,
        addie_output: ADDIEOutput,
        learner_profile: Optional[LearnerProfile] = None,
        scenario_context: Optional[str] = None,
    ) -> EvaluationFeedback:
        """
        교수설계 산출물을 평가합니다.

        Args:
            addie_output: ADDIE 5단계 산출물
            learner_profile: 학습자 프로필 (Skill-Tree)
            scenario_context: 시나리오 맥락 정보

        Returns:
            EvaluationFeedback: 평가 결과
        """
        # 프롬프트 구성
        evaluation_prompt = self._build_evaluation_prompt(
            addie_output, learner_profile, scenario_context
        )

        messages = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=evaluation_prompt),
        ]

        # LLM 호출
        response = self.llm.invoke(messages)

        # 응답 파싱
        feedback = self._parse_response(response.content)

        return feedback

    def _build_evaluation_prompt(
        self,
        addie_output: ADDIEOutput,
        learner_profile: Optional[LearnerProfile] = None,
        scenario_context: Optional[str] = None,
    ) -> str:
        """평가 프롬프트 생성"""
        prompt_parts = []

        # 시나리오 맥락
        if scenario_context:
            prompt_parts.append(f"## Scenario Context\n{scenario_context}\n")

        # 학습자 프로필
        if learner_profile:
            prompt_parts.append(learner_profile.skill_tree.to_prompt_context())

        # ADDIE 산출물
        prompt_parts.append("## Instructional Design Output to Evaluate\n")
        prompt_parts.append(self._format_addie_output(addie_output))

        prompt_parts.append("\nPlease evaluate the instructional design output above against the 13-item ADDIE Rubric.")

        return "\n".join(prompt_parts)

    def _format_addie_output(self, addie_output: ADDIEOutput) -> str:
        """ADDIE 산출물을 상세 포맷팅 (개선 사항을 정확히 평가하기 위해)"""
        sections = []

        # Analysis
        analysis = addie_output.analysis
        sections.append("### 1. Analysis")
        sections.append("**Learner Analysis:**")
        sections.append(f"- Target: {analysis.learner_analysis.target_audience}")
        sections.append(f"- Characteristics: {', '.join(analysis.learner_analysis.characteristics)}")
        sections.append(f"- Prior knowledge: {analysis.learner_analysis.prior_knowledge}")
        sections.append(f"- Learning preferences: {', '.join(analysis.learner_analysis.learning_preferences)}")
        if analysis.learner_analysis.motivation:
            sections.append(f"- Motivation: {analysis.learner_analysis.motivation}")
        sections.append(f"- Anticipated challenges: {', '.join(analysis.learner_analysis.challenges)}")

        sections.append("\n**Context Analysis:**")
        sections.append(f"- Environment: {analysis.context_analysis.environment}")
        sections.append(f"- Duration: {analysis.context_analysis.duration}")
        sections.append(f"- Constraints: {', '.join(analysis.context_analysis.constraints)}")
        sections.append(f"- Resources: {', '.join(analysis.context_analysis.resources)}")

        sections.append("\n**Task Analysis:**")
        sections.append(f"- Main topics: {', '.join(analysis.task_analysis.main_topics)}")
        sections.append(f"- Subtopics: {', '.join(analysis.task_analysis.subtopics)}")
        sections.append(f"- Prerequisites: {', '.join(analysis.task_analysis.prerequisites)}")

        # Design
        design = addie_output.design
        sections.append("\n### 2. Design")
        sections.append("**Learning Objectives:**")
        for obj in design.learning_objectives:
            sections.append(f"- [{obj.id}] [{obj.level}] {obj.statement} (verb: {obj.bloom_verb})")

        sections.append("\n**Assessment Plan:**")
        sections.append(f"- Diagnostic: {', '.join(design.assessment_plan.diagnostic)}")
        sections.append(f"- Formative: {', '.join(design.assessment_plan.formative)}")
        sections.append(f"- Summative: {', '.join(design.assessment_plan.summative)}")

        sections.append("\n**Instructional Strategy:**")
        sections.append(f"- Model: {design.instructional_strategy.model}")
        sections.append(f"- Methods: {', '.join(design.instructional_strategy.methods)}")
        sections.append("- Gagné's 9 Events:")
        for event in design.instructional_strategy.sequence:
            duration_str = f" ({event.duration})" if event.duration else ""
            sections.append(f"  - {event.event}: {event.activity}{duration_str}")

        # Development
        dev = addie_output.development
        sections.append("\n### 3. Development")
        sections.append(f"**Lesson Plan:** {dev.lesson_plan.total_duration}")
        for module in dev.lesson_plan.modules:
            sections.append(f"\n**[Module] {module.title}** ({module.duration})")
            sections.append(f"  - Objectives: {', '.join(module.objectives)}")
            for act in module.activities:
                sections.append(f"  - [{act.time}] {act.activity}: {act.description or ''}")

        sections.append("\n**Learning Materials:**")
        for mat in dev.materials:
            sections.append(f"- {mat.type}: {mat.title} - {mat.description or ''}")

        # Implementation
        impl = addie_output.implementation
        sections.append("\n### 4. Implementation")
        sections.append(f"- Delivery method: {impl.delivery_method}")
        sections.append(f"- Technical requirements: {', '.join(impl.technical_requirements)}")
        if impl.facilitator_guide:
            sections.append(f"- Facilitator guide: {impl.facilitator_guide[:200]}...")
        if impl.learner_guide:
            sections.append(f"- Learner guide: {impl.learner_guide[:200]}...")

        # Evaluation
        eval_section = addie_output.evaluation
        sections.append("\n### 5. Evaluation")
        sections.append(f"**Quiz Items ({len(eval_section.quiz_items)}):**")
        for item in eval_section.quiz_items[:5]:  # show first 5 only
            sections.append(f"- [{item.id}] [{item.difficulty or 'N/A'}] {item.question[:100]}")
        if len(eval_section.quiz_items) > 5:
            sections.append(f"  ... and {len(eval_section.quiz_items) - 5} more")

        if eval_section.rubric:
            sections.append(f"\n**Rubric criteria:** {', '.join(eval_section.rubric.criteria)}")

        if eval_section.feedback_plan:
            sections.append(f"\n**Feedback plan:** {eval_section.feedback_plan}")

        return "\n".join(sections)

    def _parse_response(self, response_text: str) -> EvaluationFeedback:
        """LLM 응답을 EvaluationFeedback으로 파싱 (ADDIE Rubric 13항목)"""
        import json
        import re

        # ADDIE 13항목 기본값 (각 항목 5.0점)
        addie_items = ["A1", "A2", "A3", "D1", "D2", "D3", "Dev1", "Dev2", "I1", "I2", "E1", "E2", "E3"]
        addie_scores = {item: 5.0 for item in addie_items}
        strengths = []
        weaknesses = []
        suggestions = []

        # 1. JSON 파싱 시도
        json_parsed = False
        try:
            # JSON 블록 추출 (```json ... ``` 또는 직접 JSON)
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            json_str = json_match.group(1) if json_match else response_text.strip()

            # JSON 부분만 추출 (앞뒤 텍스트 제거)
            json_start = json_str.find('{')
            json_end = json_str.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = json_str[json_start:json_end]

            data = json.loads(json_str)

            # addie_scores 추출
            if "addie_scores" in data:
                for item in addie_items:
                    if item in data["addie_scores"]:
                        score_val = float(data["addie_scores"][item])
                        addie_scores[item] = min(10.0, max(0.0, score_val))
                json_parsed = True

            if "strengths" in data and isinstance(data["strengths"], list):
                strengths = data["strengths"][:5]

            if "weaknesses" in data and isinstance(data["weaknesses"], list):
                weaknesses = data["weaknesses"][:5]

            if "suggestions" in data and isinstance(data["suggestions"], list):
                suggestions = data["suggestions"][:5]

        except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
            pass  # fall back to regex when JSON parsing fails

        # 2. JSON 파싱 실패 시 정규식 폴백
        if not json_parsed:
            for item in addie_items:
                # 패턴: "A1": 7.5 또는 "A1: 7.5"
                item_match = re.search(
                    rf'"{item}"[:\s]*(\d+(?:\.\d+)?)',
                    response_text,
                    re.IGNORECASE
                )
                if not item_match:
                    item_match = re.search(
                        rf'{item}[:\s]*(\d+(?:\.\d+)?)',
                        response_text,
                        re.IGNORECASE
                    )
                if item_match:
                    parsed_score = float(item_match.group(1))
                    if 0 <= parsed_score <= 10:
                        addie_scores[item] = parsed_score

            # 강점/약점/제안 파싱
            strengths = self._extract_list_items(response_text, ["Strengths", "Strengths", "strengths"])
            weaknesses = self._extract_list_items(response_text, ["Weaknesses", "Weaknesses", "weaknesses"])
            suggestions = self._extract_list_items(response_text, ["Suggestions", "Suggestions", "suggestions"])

        # 가중치 적용 점수 계산
        weighted_score = self._calculate_weighted_score(addie_scores)

        # 총점 계산 (0-100 스케일로 정규화)
        raw_sum = sum(addie_scores.values())  # max 130 points
        normalized_score = (raw_sum / 130.0) * 100.0

        return EvaluationFeedback(
            score=round(normalized_score, 1),
            strengths=strengths if strengths else ["No evaluation strength information"],
            weaknesses=weaknesses if weaknesses else ["No evaluation weakness information"],
            suggestions=suggestions if suggestions else ["No improvement suggestion information"],
            addie_scores=addie_scores,
            weighted_score=round(weighted_score, 1),
        )

    def _calculate_weighted_score(self, addie_scores: dict) -> float:
        """ADDIE 단계별 가중치를 적용한 점수 계산"""
        # 단계별 점수 합산
        phase_scores = {
            ADDIEPhase.ANALYSIS: sum(addie_scores.get(k, 0) for k in ["A1", "A2", "A3"]),
            ADDIEPhase.DESIGN: sum(addie_scores.get(k, 0) for k in ["D1", "D2", "D3"]),
            ADDIEPhase.DEVELOPMENT: sum(addie_scores.get(k, 0) for k in ["Dev1", "Dev2"]),
            ADDIEPhase.IMPLEMENTATION: sum(addie_scores.get(k, 0) for k in ["I1", "I2"]),
            ADDIEPhase.EVALUATION: sum(addie_scores.get(k, 0) for k in ["E1", "E2", "E3"]),
        }

        # 단계별 최대 점수
        phase_max = {
            ADDIEPhase.ANALYSIS: 30.0,  # 3 items * 10 points
            ADDIEPhase.DESIGN: 30.0,
            ADDIEPhase.DEVELOPMENT: 20.0,  # 2 items * 10 points
            ADDIEPhase.IMPLEMENTATION: 20.0,  # 2 items * 10 points
            ADDIEPhase.EVALUATION: 30.0,  # 3 items * 10 points
        }

        # 가중치 적용 점수 계산 (0-100 스케일)
        weighted_sum = 0.0
        for phase, raw_score in phase_scores.items():
            normalized = (raw_score / phase_max[phase]) * 100.0
            weighted_sum += normalized * DEFAULT_PHASE_WEIGHTS[phase]

        return weighted_sum

    def _extract_list_items(self, text: str, section_names: list[str]) -> list[str]:
        """텍스트에서 특정 섹션의 리스트 항목 추출"""
        import re
        items = []

        # 섹션 찾기
        for section_name in section_names:
            pattern = rf"{section_name}.*?[:：]\s*\n((?:[-\d\.\s]+[^\n]+\n?)+)"
            section_match = re.search(pattern, text, re.IGNORECASE)
            if section_match:
                section_text = section_match.group(1)
                for line in section_text.strip().split("\n"):
                    # 번호나 대시로 시작하는 항목 추출
                    match = re.match(r"^[\d\.\-\*]+\s*(.+)$", line.strip())
                    if match:
                        items.append(match.group(1).strip())
                break

        return items[:5]  # at most 5
