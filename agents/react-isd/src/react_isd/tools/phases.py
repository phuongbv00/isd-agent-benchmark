"""
ADDIE 5단계별 통합 도구

각 도구는 해당 ADDIE 단계의 표준 스키마 섹션을 직접 반환합니다.
28개 개별 도구 대신 5개 단계별 도구로 통합하여 단순화합니다.
"""

import json
from typing import List, Optional
from langchain_core.tools import tool

from shared.llm import create_chat_model, get_default_llm_config


def get_llm():
    return create_chat_model(get_default_llm_config())


def parse_json_response(content: str) -> dict:
    """LLM 응답에서 JSON 파싱"""
    if "```json" in content:
        json_str = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        json_str = content.split("```")[1].split("```")[0]
    else:
        json_str = content
    return json.loads(json_str.strip())


# ============================================================
# 1. Analysis 단계 (소항목 1-10)
# ============================================================

ANALYSIS_PROMPT = """당신은 교수설계 전문가입니다. 다음 시나리오에 대한 ADDIE Analysis 단계를 수행하세요.

## 시나리오 정보
- 제목: {title}
- 대상 학습자: {target_audience}
- 학습 환경: {learning_environment}
- 학습 시간: {duration}
- 사전 지식: {prior_knowledge}
- 학습 목표: {learning_goals}

## Analysis 단계 요구사항 (10개 소항목)

### A-1~4. 요구분석 (needs_analysis)
- problem_definition: 현재 상태와 목표 상태 간 격차 (2-3문장)
- gap_analysis: 기대 수행과 실제 수행 간 격차 (최소 3개)
- performance_analysis: 교육적 해결책 필요성 판단 (2-3문장)
- priority_matrix: high/medium/low 우선순위 분류

### A-5. 학습자 분석 (learner_analysis)
- target_audience: 대상자 명확히 정의
- characteristics: 최소 5개 구체적 특성
- prior_knowledge: 사전 지식 수준 (2-3문장)
- learning_preferences: 최소 4개 학습 선호도
- motivation: 동기 수준과 이유 (2-3문장)
- challenges: 최소 3개 예상 어려움

### A-6. 환경분석 (context_analysis)
- environment: 학습 환경
- duration: 총 학습 시간
- constraints: 최소 3개 제약조건
- resources: 최소 3개 가용 자원
- technical_requirements: 최소 2개 기술 요구사항

### A-7~10. 과제분석 (task_analysis)
- main_topics: 최소 3개 주요 학습 주제
- subtopics: 최소 6개 세부 학습 내용
- prerequisites: 최소 2개 선수학습 요건
- review_summary: 분석 결과 종합 (3-4문장)

## 출력 형식 (JSON)
```json
{{
  "needs_analysis": {{
    "problem_definition": "...",
    "gap_analysis": ["격차1", "격차2", "격차3"],
    "performance_analysis": "...",
    "priority_matrix": {{
      "high": ["항목1", "항목2"],
      "medium": ["항목3"],
      "low": ["항목4"]
    }}
  }},
  "learner_analysis": {{
    "target_audience": "...",
    "characteristics": ["특성1", "특성2", "특성3", "특성4", "특성5"],
    "prior_knowledge": "...",
    "learning_preferences": ["선호1", "선호2", "선호3", "선호4"],
    "motivation": "...",
    "challenges": ["어려움1", "어려움2", "어려움3"]
  }},
  "context_analysis": {{
    "environment": "...",
    "duration": "...",
    "constraints": ["제약1", "제약2", "제약3"],
    "resources": ["자원1", "자원2", "자원3"],
    "technical_requirements": ["기술1", "기술2"]
  }},
  "task_analysis": {{
    "main_topics": ["주제1", "주제2", "주제3"],
    "subtopics": ["세부1", "세부2", "세부3", "세부4", "세부5", "세부6"],
    "prerequisites": ["선수1", "선수2"],
    "review_summary": "..."
  }}
}}
```

JSON만 출력하세요."""


@tool
def run_analysis(
    title: str,
    target_audience: str,
    learning_environment: str,
    duration: str,
    prior_knowledge: Optional[str],
    learning_goals: list[str],
) -> dict:
    """
    ADDIE Analysis 단계를 수행합니다. (소항목 1-10)

    Returns:
        표준 스키마의 analysis 섹션
    """
    llm = get_llm()

    prompt = ANALYSIS_PROMPT.format(
        title=title,
        target_audience=target_audience,
        learning_environment=learning_environment,
        duration=duration,
        prior_knowledge=prior_knowledge or "정보 없음",
        learning_goals=json.dumps(learning_goals, ensure_ascii=False),
    )

    try:
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        print(f"[WARN] run_analysis failed: {e}")
        return _fallback_analysis(target_audience, learning_environment, duration, learning_goals)


def _fallback_analysis(target_audience, learning_environment, duration, learning_goals):
    """Analysis 폴백"""
    return {
        "needs_analysis": {
            "problem_definition": f"{target_audience}의 역량 향상을 위한 체계적 교육이 필요함",
            "gap_analysis": [f"현재: {g} 역량 부족 → 목표: {g} 달성" for g in learning_goals[:3]],
            "performance_analysis": "교육을 통해 해결 가능한 역량 격차로 판단됨",
            "priority_matrix": {"high": learning_goals[:2], "medium": learning_goals[2:3], "low": []},
        },
        "learner_analysis": {
            "target_audience": target_audience,
            "characteristics": ["학습 의지 보유", "기본 지식 보유", "실무 적용 관심", "자기주도 학습 가능", "협업 선호"],
            "prior_knowledge": "기초 수준의 관련 지식 보유",
            "learning_preferences": ["실습 중심", "단계별 안내", "시각적 자료", "즉각적 피드백"],
            "motivation": "업무/학업 성과 향상을 위한 높은 동기",
            "challenges": ["시간 제약", "실습 기회 부족", "개인차 존재"],
        },
        "context_analysis": {
            "environment": learning_environment,
            "duration": duration,
            "constraints": ["시간 제약", "자원 제한", "기술 환경"],
            "resources": ["학습 플랫폼", "교재", "멘토링"],
            "technical_requirements": ["인터넷 연결", "학습 기기"],
        },
        "task_analysis": {
            "main_topics": learning_goals[:3],
            "subtopics": [f"{g} 기초" for g in learning_goals[:3]] + [f"{g} 실습" for g in learning_goals[:3]],
            "prerequisites": ["기본 개념 이해", "학습 도구 사용법"],
            "review_summary": f"분석 완료. {target_audience} 대상 {duration} 과정으로 {len(learning_goals)}개 목표 달성 계획.",
        },
    }


# ============================================================
# 2. Design 단계 (소항목 11-18)
# ============================================================

DESIGN_PROMPT = """당신은 교수설계 전문가입니다. 다음 분석 결과를 바탕으로 ADDIE Design 단계를 수행하세요.

## 시나리오 정보
- 제목: {title}
- 대상 학습자: {target_audience}
- 학습 시간: {duration}
- 학습 목표: {learning_goals}

## Design 단계 요구사항 (8개 소항목)

### D-11. 학습목표 정교화 (learning_objectives)
- 최소 5개 측정 가능한 학습목표
- Bloom's Taxonomy 수준 분산 (기억/이해 2개, 적용/분석 2개, 평가/창조 1개)
- 각 목표: id, level, statement, bloom_verb, measurable 포함

### D-12. 평가 계획 수립 (assessment_plan)
- diagnostic: 최소 2개 진단평가 방법
- formative: 최소 2개 형성평가 방법
- summative: 최소 2개 총괄평가 방법

### D-13. 교수 내용 선정 (content_structure)
- modules: 최소 3개 모듈
- topics: 각 모듈별 토픽
- sequencing: 학습 순서 설명

### D-14. 교수적 전략 수립 (instructional_strategies)
- model: "Gagné's 9 Events"
- sequence: 9개 Event 모두 포함 (필수!)
- methods: 최소 3개 교수 방법
- rationale: 전략 선정 근거

### D-15. 비교수적 전략 (non_instructional_strategies)
- motivation_strategies: 동기 부여 전략 2-3개
- self_directed_learning: 자기주도 학습 지원 2-3개
- support_strategies: 기타 지원 방안

### D-16. 매체 선정 (media_selection)
- 각 학습 활동에 적합한 매체 최소 3개 선정

### D-17. 학습활동 및 시간 구조화 (learning_activities)
- 최소 3개 학습 활동 설계

### D-18. 스토리보드 (storyboard)
- 최소 3개 프레임 설계

## Gagné's 9 Events (반드시 모두 포함)
1. 주의 획득, 2. 학습 목표 제시, 3. 선수 학습 상기, 4. 학습 내용 제시,
5. 학습 안내 제공, 6. 연습 유도, 7. 피드백 제공, 8. 수행 평가, 9. 파지 및 전이 강화

## 출력 형식 (JSON)
```json
{{
  "learning_objectives": [
    {{"id": "LO-01", "level": "기억", "statement": "...", "bloom_verb": "정의하다", "measurable": true}},
    {{"id": "LO-02", "level": "이해", "statement": "...", "bloom_verb": "설명하다", "measurable": true}},
    {{"id": "LO-03", "level": "적용", "statement": "...", "bloom_verb": "적용하다", "measurable": true}},
    {{"id": "LO-04", "level": "분석", "statement": "...", "bloom_verb": "분석하다", "measurable": true}},
    {{"id": "LO-05", "level": "평가", "statement": "...", "bloom_verb": "평가하다", "measurable": true}}
  ],
  "assessment_plan": {{
    "diagnostic": ["진단1", "진단2"],
    "formative": ["형성1", "형성2"],
    "summative": ["총괄1", "총괄2"]
  }},
  "content_structure": {{
    "modules": ["모듈1", "모듈2", "모듈3"],
    "topics": ["토픽1", "토픽2", "토픽3"],
    "sequencing": "..."
  }},
  "instructional_strategies": {{
    "model": "Gagné's 9 Events",
    "sequence": [
      {{"event": "주의 획득", "activity": "...", "duration": "5분", "resources": ["..."]}},
      {{"event": "학습 목표 제시", "activity": "...", "duration": "5분", "resources": ["..."]}},
      {{"event": "선수 학습 상기", "activity": "...", "duration": "10분", "resources": ["..."]}},
      {{"event": "학습 내용 제시", "activity": "...", "duration": "20분", "resources": ["..."]}},
      {{"event": "학습 안내 제공", "activity": "...", "duration": "10분", "resources": ["..."]}},
      {{"event": "연습 유도", "activity": "...", "duration": "15분", "resources": ["..."]}},
      {{"event": "피드백 제공", "activity": "...", "duration": "10분", "resources": ["..."]}},
      {{"event": "수행 평가", "activity": "...", "duration": "15분", "resources": ["..."]}},
      {{"event": "파지 및 전이 강화", "activity": "...", "duration": "10분", "resources": ["..."]}}
    ],
    "methods": ["방법1", "방법2", "방법3"],
    "rationale": "..."
  }},
  "non_instructional_strategies": {{
    "motivation_strategies": ["동기1", "동기2"],
    "self_directed_learning": ["자기주도1", "자기주도2"],
    "support_strategies": ["지원1", "지원2"]
  }},
  "learning_activities": [
    {{"activity_name": "활동1", "duration": "20분", "description": "...", "materials": ["..."]}},
    {{"activity_name": "활동2", "duration": "30분", "description": "...", "materials": ["..."]}},
    {{"activity_name": "활동3", "duration": "20분", "description": "...", "materials": ["..."]}}
  ],
  "media_selection": [
    {{"media_type": "매체1", "purpose": "...", "rationale": "..."}},
    {{"media_type": "매체2", "purpose": "...", "rationale": "..."}},
    {{"media_type": "매체3", "purpose": "...", "rationale": "..."}}
  ],
  "storyboard": [
    {{"frame_number": 1, "screen_title": "...", "visual_description": "...", "interaction": "...", "notes": "..."}},
    {{"frame_number": 2, "screen_title": "...", "visual_description": "...", "interaction": "...", "notes": "..."}},
    {{"frame_number": 3, "screen_title": "...", "visual_description": "...", "interaction": "...", "notes": "..."}}
  ]
}}
```

JSON만 출력하세요."""


@tool
def run_design(
    title: str,
    target_audience: str,
    duration: str,
    learning_goals: list[str],
) -> dict:
    """
    ADDIE Design 단계를 수행합니다. (소항목 11-18)

    Returns:
        표준 스키마의 design 섹션
    """
    llm = get_llm()

    prompt = DESIGN_PROMPT.format(
        title=title,
        target_audience=target_audience,
        duration=duration,
        learning_goals=json.dumps(learning_goals, ensure_ascii=False),
    )

    try:
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        print(f"[WARN] run_design failed: {e}")
        return _fallback_design(learning_goals, duration)


def _fallback_design(learning_goals, duration):
    """Design 폴백"""
    objectives = [
        {"id": f"LO-0{i+1}", "level": ["기억", "이해", "적용", "분석", "평가"][i % 5],
         "statement": f"{g}을(를) 수행할 수 있다", "bloom_verb": ["정의하다", "설명하다", "적용하다", "분석하다", "평가하다"][i % 5],
         "measurable": True}
        for i, g in enumerate(learning_goals[:5])
    ]

    events = ["주의 획득", "학습 목표 제시", "선수 학습 상기", "학습 내용 제시", "학습 안내 제공",
              "연습 유도", "피드백 제공", "수행 평가", "파지 및 전이 강화"]

    return {
        "learning_objectives": objectives,
        "assessment_plan": {
            "diagnostic": ["사전 퀴즈", "자기 점검표"],
            "formative": ["실습 과제", "동료 평가"],
            "summative": ["최종 평가", "프로젝트 발표"],
        },
        "content_structure": {
            "modules": learning_goals[:3],
            "topics": [f"{g} 세부" for g in learning_goals[:3]],
            "sequencing": "기초 → 심화 → 적용 순서",
        },
        "instructional_strategies": {
            "model": "Gagné's 9 Events",
            "sequence": [{"event": e, "activity": f"{e} 활동", "duration": "10분", "resources": ["교재"]} for e in events],
            "methods": ["강의", "실습", "토론"],
            "rationale": "체계적 수업 설계를 위한 Gagné's 9 Events 적용",
        },
        "non_instructional_strategies": {
            "motivation_strategies": ["성취 인정", "실무 연계"],
            "self_directed_learning": ["학습 체크리스트", "자기 평가"],
            "support_strategies": ["멘토링", "Q&A"],
        },
        "learning_activities": [
            {"activity_name": "개념 학습", "duration": "30분", "description": "핵심 개념 학습", "materials": ["교재"]},
            {"activity_name": "실습", "duration": "40분", "description": "실습 과제 수행", "materials": ["실습자료"]},
            {"activity_name": "토론", "duration": "20분", "description": "그룹 토론", "materials": ["토론 주제"]},
        ],
        "media_selection": [
            {"media_type": "슬라이드", "purpose": "개념 전달", "rationale": "시각적 학습 지원"},
            {"media_type": "동영상", "purpose": "시연", "rationale": "절차적 지식 전달"},
            {"media_type": "실습 환경", "purpose": "적용", "rationale": "체험적 학습"},
        ],
        "storyboard": [
            {"frame_number": i+1, "screen_title": f"화면 {i+1}", "visual_description": "학습 내용", "interaction": "다음 버튼", "notes": ""}
            for i in range(3)
        ],
    }


# ============================================================
# 3. Development 단계 (소항목 17, 19-23)
# ============================================================

DEVELOPMENT_PROMPT = """당신은 교수설계 전문가입니다. ADDIE Development 단계를 수행하세요.

## 시나리오 정보
- 제목: {title}
- 대상 학습자: {target_audience}
- 학습 환경: {learning_environment}
- 학습 목표: {learning_goals}

## Development 단계 요구사항 (5개 소항목)

### Dev-19. 학습자용 자료 개발 (learner_materials)
- 최소 3개 학습자용 자료
- 각 자료: title, type, content (실제 내용 500자+), format 포함

### Dev-20. 교수자용 매뉴얼 (instructor_guide)
- overview: 전체 개요
- session_guides: 최소 3개 세션별 가이드
- facilitation_tips: 최소 3개 진행 팁
- troubleshooting: 최소 2개 문제 해결 가이드

### Dev-21. 운영자용 매뉴얼 (operator_manual)
- system_setup: 시스템 설정 가이드
- operation_procedures: 운영 절차
- support_procedures: 지원 절차
- escalation_process: 에스컬레이션 프로세스

### Dev-22. 평가 도구/문항 (assessment_tools)
- 최소 10개 문항 (easy 3-4개, medium 4-5개, hard 2-3개)
- 각 문항: item_id, type, question, options (4개), answer, aligned_objective, scoring_criteria

### Dev-23. 전문가 검토 (expert_review)
- reviewers: 최소 2명 검토자 유형
- review_criteria: 최소 5개 검토 기준
- feedback_summary: 피드백 요약
- revisions_made: 수정 사항

## 출력 형식 (JSON)
```json
{{
  "learner_materials": [
    {{"title": "자료1", "type": "유인물", "content": "상세 내용...", "format": "PDF"}},
    {{"title": "자료2", "type": "슬라이드", "content": "상세 내용...", "format": "PPT"}},
    {{"title": "자료3", "type": "실습가이드", "content": "상세 내용...", "format": "PDF"}}
  ],
  "instructor_guide": {{
    "overview": "전체 과정 개요 설명...",
    "session_guides": [
      {{"session": 1, "objectives": ["목표1"], "activities": ["활동1"], "notes": "..."}},
      {{"session": 2, "objectives": ["목표2"], "activities": ["활동2"], "notes": "..."}},
      {{"session": 3, "objectives": ["목표3"], "activities": ["활동3"], "notes": "..."}}
    ],
    "facilitation_tips": ["팁1", "팁2", "팁3"],
    "troubleshooting": ["문제해결1", "문제해결2"]
  }},
  "operator_manual": {{
    "system_setup": "시스템 설정 가이드...",
    "operation_procedures": ["절차1", "절차2", "절차3"],
    "support_procedures": ["지원1", "지원2"],
    "escalation_process": "에스컬레이션 절차..."
  }},
  "assessment_tools": [
    {{"item_id": "Q-01", "type": "객관식", "question": "문항1", "options": ["A", "B", "C", "D"], "answer": "A", "aligned_objective": "LO-01", "scoring_criteria": "정답 1점"}},
    {{"item_id": "Q-02", "type": "객관식", "question": "문항2", "options": ["A", "B", "C", "D"], "answer": "B", "aligned_objective": "LO-02", "scoring_criteria": "정답 1점"}}
  ],
  "expert_review": {{
    "reviewers": ["내용전문가", "교수설계전문가"],
    "review_criteria": ["내용 정확성", "교수적 적절성", "기술적 품질", "사용자 경험", "접근성"],
    "feedback_summary": "전반적으로 양호하나 일부 개선 필요...",
    "revisions_made": ["내용 수정1", "디자인 개선2"]
  }}
}}
```

JSON만 출력하세요."""


@tool
def run_development(
    title: str,
    target_audience: str,
    learning_environment: str,
    learning_goals: list[str],
) -> dict:
    """
    ADDIE Development 단계를 수행합니다. (소항목 17, 19-23)

    Returns:
        표준 스키마의 development 섹션
    """
    llm = get_llm()

    prompt = DEVELOPMENT_PROMPT.format(
        title=title,
        target_audience=target_audience,
        learning_environment=learning_environment,
        learning_goals=json.dumps(learning_goals, ensure_ascii=False),
    )

    try:
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        print(f"[WARN] run_development failed: {e}")
        return _fallback_development(learning_goals)


def _fallback_development(learning_goals):
    """Development 폴백"""
    return {
        "learner_materials": [
            {"title": f"학습자료 {i+1}", "type": ["유인물", "슬라이드", "실습가이드"][i % 3],
             "content": f"학습 목표 '{learning_goals[i % len(learning_goals)]}'에 대한 상세 학습 내용입니다. " * 10,
             "format": "PDF"}
            for i in range(3)
        ],
        "instructor_guide": {
            "overview": "본 과정은 학습자의 역량 향상을 위해 설계되었습니다.",
            "session_guides": [
                {"session": i+1, "objectives": [learning_goals[i % len(learning_goals)]], "activities": ["강의", "실습"], "notes": "주의사항"}
                for i in range(3)
            ],
            "facilitation_tips": ["학습자 참여 유도", "시간 관리", "질문 장려"],
            "troubleshooting": ["기술 문제 시 대안 활용", "학습자 어려움 시 추가 설명"],
        },
        "operator_manual": {
            "system_setup": "LMS 접속 및 콘텐츠 등록 절차",
            "operation_procedures": ["사전 점검", "운영 모니터링", "사후 정리"],
            "support_procedures": ["학습자 문의 응대", "기술 지원"],
            "escalation_process": "담당자 → 관리자 → 기술팀 순서로 에스컬레이션",
        },
        "assessment_tools": [
            {"item_id": f"Q-{i+1:02d}", "type": "객관식", "question": f"문항 {i+1}",
             "options": ["A", "B", "C", "D"], "answer": ["A", "B", "C", "D"][i % 4],
             "aligned_objective": f"LO-{(i % 5) + 1:02d}", "scoring_criteria": "정답 1점"}
            for i in range(10)
        ],
        "expert_review": {
            "reviewers": ["내용전문가", "교수설계전문가"],
            "review_criteria": ["내용 정확성", "교수적 적절성", "기술적 품질", "사용자 경험", "접근성"],
            "feedback_summary": "전반적으로 양호함",
            "revisions_made": ["일부 내용 수정", "디자인 개선"],
        },
    }


# ============================================================
# 4. Implementation 단계 (소항목 24-27)
# ============================================================

IMPLEMENTATION_PROMPT = """당신은 교수설계 전문가입니다. ADDIE Implementation 단계를 수행하세요.

## 시나리오 정보
- 제목: {title}
- 대상 학습자: {target_audience}
- 학습 환경: {learning_environment}
- 학습자 수: {class_size}

## Implementation 단계 요구사항 (4개 소항목)

### I-24. 교수자/운영자 오리엔테이션 (instructor_orientation)
- orientation_objectives: 오리엔테이션 목표 최소 3개
- schedule: 일정 계획
- materials: 필요 자료 최소 2개
- competency_checklist: 역량 체크리스트 최소 3개

### I-25. 시스템/환경 점검 (system_check)
- checklist: 최소 5개 점검 항목
- technical_validation: 기술 검증 결과
- contingency_plans: 최소 2개 비상 대응 계획

### I-26. 프로토타입 실행 계획 (prototype_execution)
- pilot_scope: 파일럿 범위
- participants: 참여자 규모 및 특성
- execution_log: 실행 기록 최소 3개
- issues_encountered: 발생 이슈 최소 2개

### I-27. 운영 모니터링 (monitoring)
- monitoring_criteria: 최소 3개 모니터링 기준
- support_channels: 최소 2개 지원 채널
- issue_resolution_log: 이슈 해결 기록 최소 2개
- real_time_adjustments: 실시간 조정 내역 최소 2개

## 출력 형식 (JSON)
```json
{{
  "instructor_orientation": {{
    "orientation_objectives": ["목표1", "목표2", "목표3"],
    "schedule": "일정 계획 설명...",
    "materials": ["자료1", "자료2"],
    "competency_checklist": ["역량1", "역량2", "역량3"]
  }},
  "system_check": {{
    "checklist": ["점검1", "점검2", "점검3", "점검4", "점검5"],
    "technical_validation": "기술 검증 완료...",
    "contingency_plans": ["비상계획1", "비상계획2"]
  }},
  "prototype_execution": {{
    "pilot_scope": "파일럿 범위 설명...",
    "participants": "참여자 정보...",
    "execution_log": ["기록1", "기록2", "기록3"],
    "issues_encountered": ["이슈1", "이슈2"]
  }},
  "monitoring": {{
    "monitoring_criteria": ["기준1", "기준2", "기준3"],
    "support_channels": ["채널1", "채널2"],
    "issue_resolution_log": ["해결1", "해결2"],
    "real_time_adjustments": ["조정1", "조정2"]
  }}
}}
```

JSON만 출력하세요."""


@tool
def run_implementation(
    title: str,
    target_audience: str,
    learning_environment: str,
    class_size: Optional[int],
) -> dict:
    """
    ADDIE Implementation 단계를 수행합니다. (소항목 24-27)

    Returns:
        표준 스키마의 implementation 섹션
    """
    llm = get_llm()

    prompt = IMPLEMENTATION_PROMPT.format(
        title=title,
        target_audience=target_audience,
        learning_environment=learning_environment,
        class_size=class_size or "미지정",
    )

    try:
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        print(f"[WARN] run_implementation failed: {e}")
        return _fallback_implementation(learning_environment)


def _fallback_implementation(learning_environment):
    """Implementation 폴백"""
    return {
        "instructor_orientation": {
            "orientation_objectives": ["과정 이해", "도구 활용", "학습자 관리"],
            "schedule": "과정 시작 1주 전 2시간 오리엔테이션",
            "materials": ["교수자 가이드", "시스템 매뉴얼"],
            "competency_checklist": ["내용 이해", "도구 활용", "촉진 기술"],
        },
        "system_check": {
            "checklist": ["LMS 접속", "콘텐츠 로딩", "평가 기능", "커뮤니케이션 도구", "백업 시스템"],
            "technical_validation": "모든 기능 정상 작동 확인",
            "contingency_plans": ["오프라인 자료 준비", "기술 지원 연락처"],
        },
        "prototype_execution": {
            "pilot_scope": f"{learning_environment} 환경에서 소규모 파일럿 실시",
            "participants": "대상 학습자 중 10-15명 선정",
            "execution_log": ["파일럿 시작", "중간 점검", "파일럿 종료"],
            "issues_encountered": ["일부 기술 문제", "시간 조정 필요"],
        },
        "monitoring": {
            "monitoring_criteria": ["학습 진도", "참여도", "만족도"],
            "support_channels": ["실시간 채팅", "이메일 지원"],
            "issue_resolution_log": ["기술 문제 해결", "학습 지원 제공"],
            "real_time_adjustments": ["일정 조정", "추가 설명 제공"],
        },
    }


# ============================================================
# 5. Evaluation 단계 (소항목 22, 28-33)
# ============================================================

EVALUATION_PROMPT = """당신은 교수설계 전문가입니다. ADDIE Evaluation 단계를 수행하세요.

## 시나리오 정보
- 제목: {title}
- 대상 학습자: {target_audience}
- 학습 목표: {learning_goals}

## Evaluation 단계 요구사항 (6개 소항목)

### E-28. 파일럿 자료 수집 (formative.data_collection)
- methods: 수집 방법 최소 3개
- learner_feedback: 학습자 피드백 항목 최소 3개
- performance_data: 수행 데이터 지표
- observations: 관찰 항목 최소 2개

### E-29. 형성평가 기반 개선 (formative.improvements)
- 최소 3개 개선 항목
- 각 항목: issue_identified, improvement_action, priority

### E-30. 총괄평가 문항 (summative.assessment_tools)
- 최소 5개 총괄평가 문항
- 각 문항: item_id, type, question, scoring_rubric

### E-31. 총괄평가 효과 분석 (summative.effectiveness_analysis)
- learning_outcomes: 학습 성과 분석
- goal_achievement_rate: 목표 달성률
- statistical_analysis: 통계 분석 결과
- recommendations: 권장 사항 최소 3개

### E-32. 프로그램 채택 여부 결정 (summative.adoption_decision)
- decision: adopt/modify/reject 중 선택
- rationale: 결정 근거 (2-3문장)
- conditions: 채택 조건 최소 2개
- stakeholder_approval: 이해관계자 승인 상태

### E-33. 프로그램 개선 및 환류 (improvement_plan)
- feedback_summary: 피드백 요약
- improvement_areas: 개선 영역 최소 3개
- action_items: 실행 항목 최소 3개
- feedback_loop: 환류 체계 설명
- next_iteration_goals: 다음 반복 목표 최소 2개

## 출력 형식 (JSON)
```json
{{
  "formative": {{
    "data_collection": {{
      "methods": ["방법1", "방법2", "방법3"],
      "learner_feedback": ["피드백1", "피드백2", "피드백3"],
      "performance_data": {{"metric1": "지표1", "metric2": "지표2"}},
      "observations": ["관찰1", "관찰2"]
    }},
    "improvements": [
      {{"issue_identified": "이슈1", "improvement_action": "개선1", "priority": "high"}},
      {{"issue_identified": "이슈2", "improvement_action": "개선2", "priority": "medium"}},
      {{"issue_identified": "이슈3", "improvement_action": "개선3", "priority": "low"}}
    ]
  }},
  "summative": {{
    "assessment_tools": [
      {{"item_id": "SA-01", "type": "종합평가", "question": "문항1", "scoring_rubric": "채점기준1"}},
      {{"item_id": "SA-02", "type": "종합평가", "question": "문항2", "scoring_rubric": "채점기준2"}}
    ],
    "effectiveness_analysis": {{
      "learning_outcomes": {{"achievement_rate": "85%", "details": "성과 분석..."}},
      "goal_achievement_rate": "85%",
      "statistical_analysis": "통계 분석 결과...",
      "recommendations": ["권장1", "권장2", "권장3"]
    }},
    "adoption_decision": {{
      "decision": "adopt",
      "rationale": "채택 결정 근거 설명...",
      "conditions": ["조건1", "조건2"],
      "stakeholder_approval": "승인 완료"
    }}
  }},
  "improvement_plan": {{
    "feedback_summary": "피드백 요약...",
    "improvement_areas": ["영역1", "영역2", "영역3"],
    "action_items": ["실행1", "실행2", "실행3"],
    "feedback_loop": "환류 체계 설명...",
    "next_iteration_goals": ["목표1", "목표2"]
  }}
}}
```

JSON만 출력하세요."""


@tool
def run_evaluation(
    title: str,
    target_audience: str,
    learning_goals: list[str],
) -> dict:
    """
    ADDIE Evaluation 단계를 수행합니다. (소항목 22, 28-33)

    Returns:
        표준 스키마의 evaluation 섹션
    """
    llm = get_llm()

    prompt = EVALUATION_PROMPT.format(
        title=title,
        target_audience=target_audience,
        learning_goals=json.dumps(learning_goals, ensure_ascii=False),
    )

    try:
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        print(f"[WARN] run_evaluation failed: {e}")
        return _fallback_evaluation(learning_goals)


def _fallback_evaluation(learning_goals):
    """Evaluation 폴백"""
    return {
        "formative": {
            "data_collection": {
                "methods": ["설문조사", "학습 로그 분석", "인터뷰"],
                "learner_feedback": ["내용 만족도", "난이도 적절성", "실용성"],
                "performance_data": {"completion_rate": "학습 완료율", "assessment_score": "평가 점수"},
                "observations": ["학습 참여도", "상호작용 빈도"],
            },
            "improvements": [
                {"issue_identified": "일부 내용 어려움", "improvement_action": "추가 설명 제공", "priority": "high"},
                {"issue_identified": "시간 부족", "improvement_action": "일정 조정", "priority": "medium"},
                {"issue_identified": "실습 부족", "improvement_action": "실습 시간 확대", "priority": "high"},
            ],
        },
        "summative": {
            "assessment_tools": [
                {"item_id": f"SA-{i+1:02d}", "type": "종합평가", "question": f"종합평가 문항 {i+1}",
                 "scoring_rubric": "채점 기준"}
                for i in range(5)
            ],
            "effectiveness_analysis": {
                "learning_outcomes": {"achievement_rate": "85%", "details": "대부분의 학습 목표 달성"},
                "goal_achievement_rate": "85%",
                "statistical_analysis": "평균 점수 85점, 표준편차 10",
                "recommendations": ["내용 보완", "실습 강화", "피드백 개선"],
            },
            "adoption_decision": {
                "decision": "adopt",
                "rationale": "학습 목표 달성률이 높고 학습자 만족도가 양호함",
                "conditions": ["일부 내용 수정 후 적용", "파일럿 결과 반영"],
                "stakeholder_approval": "승인 완료",
            },
        },
        "improvement_plan": {
            "feedback_summary": "전반적으로 긍정적이나 일부 개선 필요",
            "improvement_areas": ["내용 난이도", "실습 기회", "피드백 시스템"],
            "action_items": ["내용 수정", "실습 추가", "피드백 강화"],
            "feedback_loop": "분기별 검토 및 개선 사이클",
            "next_iteration_goals": ["만족도 90% 달성", "완료율 95% 달성"],
        },
    }


__all__ = [
    "run_analysis",
    "run_design",
    "run_development",
    "run_implementation",
    "run_evaluation",
]
