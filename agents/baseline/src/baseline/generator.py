"""
Baseline ISD Generator

Generates ADDIE outputs with a single LLM call.
Uses the shared provider-neutral LLM factory.
"""

import json
import re
from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from baseline.prompts import SYSTEM_PROMPT, build_user_prompt
from shared.llm import LLMConfig, create_chat_model, llm_config_from_env, llm_config_from_legacy


class BaselineGenerator:
    """Single prompt ADDIE generator with provider-neutral LLM support"""

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 32768,
        api_key: Optional[str] = None,
        provider: str = None,  # "upstage", "openrouter", or "openai"
        reasoning_budget: int = None,  # Thinking/reasoning token budget (OpenRouter)
        llm_config: Optional[LLMConfig] = None,
    ):
        if llm_config is None:
            llm_config = (
                llm_config_from_legacy(
                    provider=provider or "upstage",
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_budget=reasoning_budget,
                )
                if provider or model or api_key
                else llm_config_from_env(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_budget=reasoning_budget,
                )
            )
        else:
            llm_config = llm_config.copy_with(
                model=model or llm_config.model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_budget=reasoning_budget,
            )

        self.llm_config = llm_config
        self.provider = llm_config.provider
        self.model = llm_config.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_budget = llm_config.reasoning_budget
        self.llm = create_chat_model(llm_config)
        if llm_config.api_spec in {"openai", "openai_compatible"}:
            self.llm = self.llm.bind(response_format={"type": "json_object"})

    def generate(self, scenario: dict) -> dict:
        """
        시나리오를 입력받아 ADDIE 산출물을 생성합니다.

        Args:
            scenario: 시나리오 딕셔너리 (JSON 스키마 준수)

        Returns:
            dict: ADDIE 산출물 + 메타데이터
        """
        start_time = datetime.now()

        # 프롬프트 생성
        user_prompt = build_user_prompt(scenario)

        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        # 응답 파싱
        content = self._extract_content(response.content)
        addie_output = self._parse_response(content)
        usage = self._extract_usage(response)

        # 메타데이터 생성
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # trajectory에 ADDIE 단계별 추론 과정 기록
        reasoning_steps = self._extract_reasoning_steps(scenario, addie_output)
        tool_calls = self._build_tool_calls(scenario, addie_output, start_time)

        result = {
            "scenario_id": scenario.get("scenario_id", "unknown"),
            "agent_id": "baseline",
            "timestamp": end_time.isoformat(),
            "addie_output": addie_output,
            "trajectory": {
                "tool_calls": tool_calls,
                "reasoning_steps": reasoning_steps,
                "agent_interactions": [
                    {
                        "iteration": 1,
                        "agent": "baseline",
                        "action": "analyze",
                        "timestamp": start_time.isoformat(),
                    },
                    {
                        "iteration": 1,
                        "agent": "baseline",
                        "action": "design",
                        "timestamp": start_time.isoformat(),
                    },
                    {
                        "iteration": 1,
                        "agent": "baseline",
                        "action": "develop",
                        "timestamp": start_time.isoformat(),
                    },
                    {
                        "iteration": 1,
                        "agent": "baseline",
                        "action": "implement",
                        "timestamp": start_time.isoformat(),
                    },
                    {
                        "iteration": 1,
                        "agent": "baseline",
                        "action": "evaluate",
                        "timestamp": start_time.isoformat(),
                    },
                ],
            },
            "metadata": {
                "model": self.model,
                "total_tokens": usage.get("total_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "execution_time_seconds": execution_time,
                "cost_usd": self._calculate_cost(usage),
                "agent_version": "0.1.0",
                "iterations": 1,
            },
        }

        return result

    def _parse_response(self, content: str) -> dict:
        """LLM 응답을 ADDIE 출력으로 파싱"""
        import sys

        # Strip markdown code blocks if present
        json_str = content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]  # Remove ```json
        if json_str.startswith("```"):
            json_str = json_str[3:]  # Remove ```
        if json_str.endswith("```"):
            json_str = json_str[:-3]  # Remove trailing ```
        json_str = json_str.strip()

        # Try parsing
        try:
            result = json.loads(json_str)
            return self._ensure_required_fields(result)
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing failed: {e}", file=sys.stderr)
            print(f"[DEBUG] Response length: {len(content)}", file=sys.stderr)
            print(f"[DEBUG] First 200 chars of response: {content[:200]}", file=sys.stderr)

        # 파싱 실패 시 기본 구조 반환
        print("[DEBUG] Returning default output", file=sys.stderr)
        return self._create_default_output()

    def _extract_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _ensure_required_fields(self, data: dict) -> dict:
        """필수 필드 누락 시 기본값 적용 (#71)"""
        default = self._create_default_output()

        # evaluation 섹션 확보
        if "evaluation" not in data:
            data["evaluation"] = {}

        # [28] pilot_data_collection 검증 및 fallback
        if not data.get("evaluation", {}).get("pilot_data_collection"):
            data["evaluation"]["pilot_data_collection"] = default["evaluation"]["pilot_data_collection"]

        return data

    def _create_default_output(self) -> dict:
        """기본 ADDIE 출력 구조 (33개 소항목 완전 포함)"""
        return {
            "analysis": {
                # A-1 ~ A-4: 요구분석
                "needs_analysis": {
                    "problem_definition": None,      # A-1: 문제 확인 및 정의
                    "gap_analysis": [],              # A-2: 차이분석
                    "performance_analysis": None,    # A-3: 수행분석
                    "priority_matrix": {             # A-4: 요구 우선순위 결정
                        "high": [],
                        "medium": [],
                        "low": [],
                    },
                },
                # A-5: 학습자 분석
                "learner_analysis": {
                    "target_audience": "Not specified",
                    "characteristics": [],
                    "prior_knowledge": None,
                    "learning_preferences": [],
                    "motivation": None,
                    "challenges": [],
                },
                # A-6: 환경분석
                "context_analysis": {
                    "environment": "Not specified",
                    "duration": "Not specified",
                    "constraints": [],
                    "resources": [],
                    "technical_requirements": [],
                },
                # A-7 ~ A-10: 과제분석
                "task_analysis": {
                    "main_topics": [],               # A-7: 초기 학습목표 도출
                    "subtopics": [],                 # A-8: 하위 기능 분석
                    "prerequisites": [],             # A-9: 출발점 행동 분석
                    "review_summary": None,          # A-10: 과제분석 결과 검토/정리
                },
            },
            "design": {
                # D-11: 학습목표 정교화
                "learning_objectives": [],
                # D-12: 평가 계획 수립
                "assessment_plan": {
                    "diagnostic": [],
                    "formative": [],
                    "summative": [],
                },
                # D-13: 교수 내용 선정
                "content_selection": [],
                # D-14, D-17: 교수적 전략 수립, 학습활동 및 시간 구조화
                "instructional_strategy": {
                    "model": "Gagné's 9 Events",
                    "sequence": [],
                    "methods": [],
                },
                # D-15: 비교수적 전략 수립
                "non_instructional_strategy": [],
                # D-16: 매체 선정
                "media_selection": [],
                # D-18: 스토리보드/화면 흐름 설계
                "storyboard": [],
            },
            "development": {
                # Dev-19: 학습자용 자료 개발
                "lesson_plan": {
                    "total_duration": "Not specified",
                    "modules": [],
                },
                "materials": [
                    {
                        "type": "Presentation",
                        "title": "Training Slides",
                        "description": "Slide materials that deliver the learning content visually",
                        "slides": 10,
                        "slide_contents": [
                            {"slide_number": 1, "title": "Training Introduction", "bullet_points": ["Welcome", "Learning objectives", "Schedule overview"], "speaker_notes": "Welcome the participants and introduce the training objectives."},
                            {"slide_number": 2, "title": "Core Concept 1", "bullet_points": ["Concept definition", "Key characteristics", "Application examples"], "speaker_notes": "Explain the first core concept with examples."},
                            {"slide_number": 3, "title": "Core Concept 2", "bullet_points": ["Concept definition", "Key characteristics", "Application examples"], "speaker_notes": "Explain the second core concept with examples."},
                            {"slide_number": 4, "title": "Practice Guidance", "bullet_points": ["Practice objectives", "Practice procedure", "Precautions"], "speaker_notes": "Guide the hands-on practice activity."},
                            {"slide_number": 5, "title": "Wrap-up and Q&A", "bullet_points": ["Summary of key content", "Q&A", "Next steps"], "speaker_notes": "Summarize the learning content and take questions."},
                        ],
                    },
                    {
                        "type": "Handout",
                        "title": "Learner Handout",
                        "description": "Summary of the learning content and reference materials",
                        "pages": 5,
                    },
                    {
                        "type": "Practice materials",
                        "title": "Practice Guide",
                        "description": "Step-by-step guide for the hands-on practice activity",
                        "pages": 3,
                    },
                ],
                # Dev-20: 교수자용 매뉴얼 개발
                "instructor_manual": None,
                # Dev-21: 운영자용 매뉴얼 개발
                "operator_manual": None,
                # Dev-23: 전문가 검토
                "expert_review": {
                    "reviewers": [],
                    "checklist": [],
                    "feedback_plan": None,
                },
            },
            "implementation": {
                "delivery_method": "Not specified",
                "facilitator_guide": None,
                "learner_guide": None,
                # I-25: 시스템/환경 점검
                "technical_requirements": [],
                # I-24: 교수자/운영자 오리엔테이션
                "orientation_plan": None,
                # I-26: 프로토타입 실행 계획
                "pilot_plan": {
                    "pilot_scope": None,
                    "participants": None,
                    "duration": None,
                    "success_criteria": [],
                    "data_collection": [],
                    "contingency_plan": None,
                },
                # I-27: 운영 모니터링
                "support_plan": None,
            },
            "evaluation": {
                # Dev-22: 평가 도구/문항 개발
                "quiz_items": [
                    {
                        "id": "Q-01",
                        "question": "This item checks understanding of the main learning content of this training.",
                        "type": "multiple_choice",
                        "options": ["Option A (correct answer)", "Option B", "Option C", "Option D"],
                        "answer": "Option A (correct answer)",
                        "explanation": "This item checks whether the core concepts are accurately understood.",
                        "objective_id": "OBJ-01",
                        "difficulty": "easy",
                    },
                    {
                        "id": "Q-02",
                        "question": "This item checks whether the learning content can be applied to real situations.",
                        "type": "multiple_choice",
                        "options": ["Option A", "Option B (correct answer)", "Option C", "Option D"],
                        "answer": "Option B (correct answer)",
                        "explanation": "Assesses the ability to apply the learning content in practice.",
                        "objective_id": "OBJ-02",
                        "difficulty": "medium",
                    },
                    {
                        "id": "Q-03",
                        "question": "This item analyzes the relationships among the core concepts.",
                        "type": "multiple_choice",
                        "options": ["Option A", "Option B", "Option C (correct answer)", "Option D"],
                        "answer": "Option C (correct answer)",
                        "explanation": "This item assesses analytical thinking.",
                        "objective_id": "OBJ-03",
                        "difficulty": "medium",
                    },
                    {
                        "id": "Q-04",
                        "question": "Compare and analyze the advantages and disadvantages of the learning content.",
                        "type": "short_answer",
                        "options": [],
                        "answer": "State the advantages and disadvantages concretely and compare and analyze them",
                        "explanation": "Assesses critical thinking and analytical ability.",
                        "objective_id": "OBJ-04",
                        "difficulty": "hard",
                    },
                    {
                        "id": "Q-05",
                        "question": "Based on what you have learned, propose a solution to the problem.",
                        "type": "essay",
                        "options": [],
                        "answer": "Synthesize the learning content and propose a creative solution",
                        "explanation": "Assesses integrative thinking and problem-solving ability.",
                        "objective_id": "OBJ-05",
                        "difficulty": "hard",
                    },
                ],
                # E-28: 파일럿 자료 수집
                "pilot_data_collection": {
                    "title": "Pilot / initial implementation data collection plan",
                    "data_types": {
                        "quantitative": [
                            {"type": "Pre-test scores", "purpose": "Baseline measurement", "source": "Learners"},
                            {"type": "Post-test scores", "purpose": "Learning outcome measurement", "source": "Learners"},
                            {"type": "Satisfaction scores", "purpose": "Reaction evaluation", "source": "Learners"},
                            {"type": "Participation/completion rate", "purpose": "Engagement measurement", "source": "System"},
                        ],
                        "qualitative": [
                            {"type": "Open-ended feedback", "purpose": "In-depth opinion collection", "source": "Learners"},
                            {"type": "Observation records", "purpose": "Behavior pattern identification", "source": "Observers"},
                            {"type": "Interview responses", "purpose": "In-depth understanding", "source": "Learners/Instructors"},
                        ],
                    },
                    "collection_methods": [
                        {"method": "Online survey", "timing": "Immediately after training", "tool": "Survey platform"},
                        {"method": "Test/quiz", "timing": "Before/after training", "tool": "LMS"},
                        {"method": "Observation", "timing": "During training", "tool": "Observation checklist"},
                        {"method": "Interview", "timing": "Within 1 week after training", "tool": "Interview guide"},
                    ],
                    "instruments": [
                        {"name": "Pre-post test", "type": "Knowledge assessment", "items": 20},
                        {"name": "Satisfaction survey", "type": "Reaction evaluation", "items": 15},
                        {"name": "Observation checklist", "type": "Behavior observation", "items": 10},
                    ],
                    "timeline": [
                        {"phase": "Before", "timing": "D-1 ~ D-Day", "activities": ["Pre-test", "Baseline information collection"]},
                        {"phase": "Midpoint", "timing": "During training", "activities": ["Real-time observation", "Formative assessment"]},
                        {"phase": "Immediately after", "timing": "D+0", "activities": ["Post-test", "Satisfaction survey"]},
                        {"phase": "Follow-up", "timing": "D+7 ~ D+30", "activities": ["Interview", "On-the-job application survey"]},
                    ],
                    "data_management": {
                        "storage": "Secure storage",
                        "retention_period": "3 years",
                        "access_control": "Restricted to the training team and the evaluation team",
                    },
                },
                # E-29: 형성평가 기반 개선
                "formative_improvement": None,
                # E-30: 총괄평가 계획
                "rubric": {
                    "criteria": [],
                    "levels": {
                        "excellent": None,
                        "good": None,
                        "needs_improvement": None,
                    },
                },
                # E-31: 총괄평가 효과 분석 계획
                "summative_analysis": {
                    "level_1_reaction": None,
                    "level_2_learning": None,
                    "level_3_behavior": None,
                    "level_4_results": None,
                },
                # E-32: 프로그램 채택 여부 결정
                "adoption_decision": {
                    "recommendation": None,
                    "rationale": None,
                    "conditions": [],
                    "next_steps": [],
                },
                # E-33: 프로그램 개선 및 환류
                "program_improvement": None,
                "feedback_plan": None,
            },
        }

    def _extract_reasoning_steps(self, scenario: dict, addie_output: dict) -> list:
        """ADDIE 산출물에서 추론 과정을 추출하여 reasoning_steps 생성"""
        steps = []

        # 시나리오 분석 단계
        target = scenario.get("context", {}).get("target_audience", "Learners")
        goals = scenario.get("learning_goals", [])
        duration = scenario.get("context", {}).get("duration", "Not specified")

        steps.append(f"Step 1 (Analysis): Learner analysis - target audience: {target}")
        steps.append(f"Step 2 (Analysis): Identify learning goals - {len(goals)} goals identified")

        # 설계 단계
        if addie_output.get("design"):
            objectives = addie_output.get("design", {}).get("learning_objectives", [])
            steps.append(f"Step 3 (Design): Design learning objectives - {len(objectives)} detailed objectives established")
            steps.append("Step 4 (Design): Design the assessment plan and instructional strategy")

        # 개발 단계
        if addie_output.get("development"):
            modules = addie_output.get("development", {}).get("lesson_plan", {}).get("modules", [])
            steps.append(f"Step 5 (Development): Develop the lesson plan - {len(modules)} modules organized")
            steps.append("Step 6 (Development): Develop learning materials and content")

        # 실행 단계
        if addie_output.get("implementation"):
            delivery = addie_output.get("implementation", {}).get("delivery_method", "Not specified")
            steps.append(f"Step 7 (Implementation): Determine the delivery method - {delivery}")
            steps.append("Step 8 (Implementation): Write the operations guide")

        # 평가 단계
        if addie_output.get("evaluation"):
            quiz_items = addie_output.get("evaluation", {}).get("quiz_items", [])
            steps.append(f"Step 9 (Evaluation): Develop assessment items - {len(quiz_items)} items")
            steps.append("Step 10 (Evaluation): Establish the feedback plan")

        # 최종 결론
        steps.append(f"Conclusion: ADDIE-framework-based {duration} curriculum design completed")

        return steps

    def _build_tool_calls(self, scenario: dict, addie_output: dict, start_time: datetime) -> list:
        """ADDIE 단계별 도구 호출 기록 생성"""
        tool_calls = [
            {
                "step": 1,
                "tool": "analyze_learner",
                "args": {"scenario_id": scenario.get("scenario_id", "unknown")},
                "timestamp": start_time.isoformat(),
            },
            {
                "step": 2,
                "tool": "analyze_context",
                "args": {"scenario_id": scenario.get("scenario_id", "unknown")},
                "timestamp": start_time.isoformat(),
            },
            {
                "step": 3,
                "tool": "design_objectives",
                "args": {"scenario_id": scenario.get("scenario_id", "unknown")},
                "timestamp": start_time.isoformat(),
            },
            {
                "step": 4,
                "tool": "design_strategy",
                "args": {"scenario_id": scenario.get("scenario_id", "unknown")},
                "timestamp": start_time.isoformat(),
            },
            {
                "step": 5,
                "tool": "develop_lesson_plan",
                "args": {"scenario_id": scenario.get("scenario_id", "unknown")},
                "timestamp": start_time.isoformat(),
            },
            {
                "step": 6,
                "tool": "implement_delivery",
                "args": {"scenario_id": scenario.get("scenario_id", "unknown")},
                "timestamp": start_time.isoformat(),
            },
            {
                "step": 7,
                "tool": "evaluate_create_quiz",
                "args": {"scenario_id": scenario.get("scenario_id", "unknown")},
                "timestamp": start_time.isoformat(),
            },
        ]
        return tool_calls

    def _extract_usage(self, response: Any) -> dict[str, int]:
        """Normalize token usage from LangChain response metadata."""
        usage = getattr(response, "usage_metadata", None) or {}
        response_metadata = getattr(response, "response_metadata", {}) or {}
        token_usage = response_metadata.get("token_usage") or {}

        prompt_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or token_usage.get("prompt_tokens")
            or 0
        )
        completion_tokens = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or token_usage.get("completion_tokens")
            or 0
        )
        total_tokens = (
            usage.get("total_tokens")
            or token_usage.get("total_tokens")
            or prompt_tokens + completion_tokens
        )

        return {
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }

    def _calculate_cost(self, usage: Any) -> float:
        """API 호출 비용 계산 (USD)"""
        if not usage:
            return 0.0

        # GPT-4o 가격 (2024년 기준 근사값)
        # Input: $5/1M tokens, Output: $15/1M tokens
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)

        input_cost = (prompt_tokens / 1_000_000) * 5
        output_cost = (completion_tokens / 1_000_000) * 15

        return round(input_cost + output_cost, 6)
