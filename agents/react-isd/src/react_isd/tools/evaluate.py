"""
Evaluation 단계 도구

ADDIE의 다섯 번째 단계: 평가 도구 및 루브릭 개발
LLM을 활용하여 맥락에 맞는 깊이 있는 평가 문항을 생성합니다.
"""

import json
from typing import Optional
from langchain_core.tools import tool

from shared.llm import create_chat_model, get_default_llm_config


def get_llm():
    return create_chat_model(get_default_llm_config())


@tool
def create_quiz_items(
    objectives: list[dict],
    main_topics: list[str],
    difficulty: Optional[str] = None,
    num_items: int = 10,
) -> list[dict]:
    """
    퀴즈 문항을 생성합니다.

    Args:
        objectives: 학습 목표 목록
        main_topics: 주요 주제 목록
        difficulty: 난이도 (선택)
        num_items: 생성할 문항 수

    Returns:
        퀴즈 문항 목록
    """
    llm = get_llm()

    prompt = f"""당신은 교육 평가 전문가입니다. 다음 학습 목표와 주제에 맞는 퀴즈 문항을 생성해주세요.

## 입력 정보
- 학습 목표: {json.dumps(objectives, ensure_ascii=False)}
- 주요 주제: {json.dumps(main_topics, ensure_ascii=False)}
- 난이도: {difficulty or "medium"}
- 생성할 문항 수: {num_items}

## Bloom's Taxonomy 기반 문항 유형
- 기억/이해 수준: 객관식 문항 (개념 확인)
- 적용/분석 수준: 객관식 또는 단답형 (상황 적용)
- 평가/창조 수준: 서술형 문항 (비판적 사고)

## 요구사항
1. 각 문항은 특정 학습 목표와 연결
2. 난이도 분배: easy(30%), medium(40%), hard(30%)
3. 객관식 문항은 4개의 선택지와 정답, 해설 포함
4. 단답형/서술형은 모범 답안과 채점 기준 포함
5. 문항이 실제 학습 내용을 평가할 수 있도록 구체적으로 작성

## 출력 형식 (JSON 배열)
```json
[
  {{
    "id": "Q-01",
    "question": "신입사원 온보딩 프로그램에서 조직문화 적응을 위해 가장 효과적인 방법은?",
    "type": "multiple_choice",
    "options": [
      "멘토링 프로그램을 통한 1:1 지원",
      "문서 자료만 제공",
      "개인적으로 적응하도록 방치",
      "업무만 집중적으로 교육"
    ],
    "answer": "멘토링 프로그램을 통한 1:1 지원",
    "explanation": "멘토링은 신입사원이 조직문화를 자연스럽게 습득하고 소속감을 형성하는 데 효과적입니다.",
    "objective_id": "OBJ-01",
    "difficulty": "medium"
  }},
  ...
]
```

JSON 배열만 출력하세요."""

    try:
        response = llm.invoke(prompt)
        content = response.content

        # JSON 파싱
        json_match = content
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0]

        quiz_items = json.loads(json_match.strip())
        return quiz_items

    except Exception as e:
        # 폴백: 템플릿 기반 생성
        return _fallback_create_quiz_items(objectives, main_topics, difficulty, num_items)


def _fallback_create_quiz_items(
    objectives: list[dict],
    main_topics: list[str],
    difficulty: Optional[str] = None,
    num_items: int = 10,
) -> list[dict]:
    """LLM 실패 시 폴백 함수"""
    quiz_items = []
    diff = difficulty.lower() if difficulty else "medium"

    # 난이도별 분배
    if "easy" in diff:
        distribution = {"easy": 0.6, "medium": 0.3, "hard": 0.1}
    elif "hard" in diff:
        distribution = {"easy": 0.1, "medium": 0.3, "hard": 0.6}
    else:
        distribution = {"easy": 0.3, "medium": 0.4, "hard": 0.3}

    # 목표별 문항 생성
    for i in range(num_items):
        obj_idx = i % len(objectives) if objectives else 0
        obj = objectives[obj_idx] if objectives else {"id": "OBJ-01", "level": "이해"}
        topic_idx = i % len(main_topics) if main_topics else 0
        topic = main_topics[topic_idx] if main_topics else "학습 내용"

        # 난이도 결정
        rand_val = (i * 7 + 3) % 10 / 10
        if rand_val < distribution["easy"]:
            item_diff = "easy"
        elif rand_val < distribution["easy"] + distribution["medium"]:
            item_diff = "medium"
        else:
            item_diff = "hard"

        # 문항 유형 결정 (Bloom 수준 기반)
        bloom_level = obj.get("level", "이해")
        if bloom_level in ["기억", "이해"]:
            q_type = "multiple_choice"
            question = f"{topic}에 관한 다음 설명 중 옳은 것은?"
            options = [
                f"{topic}의 핵심 개념에 대한 올바른 설명",
                f"{topic}에 대한 일반적인 오해",
                f"{topic}와 관련 없는 내용",
                f"{topic}의 부분적으로 맞는 설명",
            ]
            answer = options[0]
            explanation = f"{topic}의 핵심 개념을 정확히 이해하고 있는지 확인하는 문항입니다."

        elif bloom_level in ["적용", "분석"]:
            q_type = "multiple_choice" if i % 2 == 0 else "short_answer"
            if q_type == "multiple_choice":
                question = f"{topic}를 실제 상황에 적용할 때 가장 적절한 접근 방식은?"
                options = [
                    "상황을 분석하고 적절한 방법 선택",
                    "무조건 기본 방법 사용",
                    "다른 사람의 방법 그대로 따라하기",
                    "직감에 따라 결정하기",
                ]
                answer = options[0]
            else:
                question = f"{topic}를 실제 업무/학습에 어떻게 적용할 수 있는지 설명하시오."
                options = []
                answer = f"{topic}의 개념을 이해하고 구체적인 적용 방안을 제시"
            explanation = f"{topic}의 실제 적용 능력을 평가하는 문항입니다."

        else:  # 평가, 창조
            q_type = "essay"
            question = f"{topic}의 장단점을 비교 분석하고, 개선 방안을 제시하시오."
            options = []
            answer = f"{topic}에 대한 비판적 분석과 창의적 대안 제시"
            explanation = f"고차원적 사고력을 평가하는 문항입니다."

        quiz_items.append({
            "id": f"Q-{i+1:02d}",
            "question": question,
            "type": q_type,
            "options": options,
            "answer": answer,
            "explanation": explanation,
            "objective_id": obj.get("id", f"OBJ-{obj_idx+1:02d}"),
            "difficulty": item_diff,
        })

    return quiz_items


@tool
def create_rubric(
    objectives: list[dict],
    assessment_type: str = "종합 평가",
) -> dict:
    """
    평가 루브릭을 생성합니다.

    Args:
        objectives: 학습 목표 목록
        assessment_type: 평가 유형

    Returns:
        평가 루브릭 (기준, 수준별 설명)
    """
    llm = get_llm()

    prompt = f"""당신은 교육 평가 전문가입니다. 다음 학습 목표에 맞는 평가 루브릭을 생성해주세요.

## 입력 정보
- 학습 목표: {json.dumps(objectives, ensure_ascii=False)}
- 평가 유형: {assessment_type}

## 루브릭 구성 요소
1. 평가 기준 (Criteria): 학습 목표에서 도출된 구체적인 평가 항목
2. 수행 수준 (Levels): 각 기준별 달성 수준 설명

## 요구사항
1. 학습 목표의 Bloom 수준을 고려한 평가 기준 설정
2. 각 기준별로 4단계 수행 수준 (excellent, good, satisfactory, needs_improvement)
3. 각 수준별로 구체적이고 관찰 가능한 행동 지표 포함
4. 평가 가중치 포함

## 출력 형식 (JSON)
```json
{{
  "assessment_type": "종합 평가",
  "criteria": [
    {{
      "name": "개념 이해도",
      "weight": 30,
      "description": "핵심 개념과 원리에 대한 이해 정도",
      "objective_ids": ["OBJ-01", "OBJ-02"]
    }},
    {{
      "name": "적용 능력",
      "weight": 25,
      "description": "학습 내용을 실제 상황에 적용하는 능력",
      "objective_ids": ["OBJ-03"]
    }},
    {{
      "name": "분석 및 비판적 사고",
      "weight": 25,
      "description": "정보를 분석하고 비판적으로 평가하는 능력",
      "objective_ids": ["OBJ-04"]
    }},
    {{
      "name": "과제 완성도",
      "weight": 20,
      "description": "과제의 완성도와 성실성",
      "objective_ids": []
    }}
  ],
  "levels": {{
    "excellent": {{
      "score_range": "90-100%",
      "description": "학습 목표를 탁월하게 달성. 핵심 개념을 완벽히 이해하고 창의적으로 적용할 수 있음. 심층적인 분석과 통찰을 보여줌."
    }},
    "good": {{
      "score_range": "70-89%",
      "description": "학습 목표를 충분히 달성. 핵심 개념을 정확히 이해하고 적절히 적용할 수 있음. 논리적인 분석이 가능함."
    }},
    "satisfactory": {{
      "score_range": "50-69%",
      "description": "학습 목표를 부분적으로 달성. 기본 개념은 이해하나 적용과 분석에 한계가 있음. 추가 연습이 필요함."
    }},
    "needs_improvement": {{
      "score_range": "50% 미만",
      "description": "학습 목표 달성이 미흡. 기본 개념 이해가 부족하며 재학습이 필요함."
    }}
  }}
}}
```

JSON만 출력하세요."""

    try:
        response = llm.invoke(prompt)
        content = response.content

        # JSON 파싱
        json_match = content
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0]

        rubric = json.loads(json_match.strip())
        return rubric

    except Exception as e:
        # 폴백: 템플릿 기반 생성
        return _fallback_create_rubric(objectives, assessment_type)


def _fallback_create_rubric(
    objectives: list[dict],
    assessment_type: str = "종합 평가",
) -> dict:
    """LLM 실패 시 폴백 함수"""
    criteria = []

    # 목표 기반 기준 생성
    for obj in objectives[:4]:
        level = obj.get("level", "이해")
        statement = obj.get("statement", "")

        if "이해" in level or "기억" in level:
            criterion = "개념 이해도"
        elif "적용" in level:
            criterion = "적용 능력"
        elif "분석" in level:
            criterion = "분석 능력"
        elif "평가" in level:
            criterion = "비판적 사고"
        else:
            criterion = "창의적 문제해결"

        if criterion not in criteria:
            criteria.append(criterion)

    # 기본 기준 추가
    if "참여도" not in criteria:
        criteria.append("학습 참여도")
    if "완성도" not in criteria:
        criteria.append("과제 완성도")

    # 수준별 설명
    levels = {
        "excellent": "학습 목표를 완벽히 달성함. 핵심 개념을 정확히 이해하고 창의적으로 적용할 수 있음. 90% 이상 달성.",
        "good": "학습 목표를 대부분 달성함. 핵심 개념을 이해하고 기본적인 적용이 가능함. 70-89% 달성.",
        "satisfactory": "학습 목표를 부분적으로 달성함. 기본 개념은 이해하나 적용에 어려움이 있음. 50-69% 달성.",
        "needs_improvement": "학습 목표 달성이 미흡함. 기본 개념 이해가 부족하며 추가 학습이 필요함. 50% 미만 달성.",
    }

    return {
        "criteria": criteria,
        "levels": levels,
    }


@tool
def create_data_collection_plan(
    pilot_plan: dict,
    learning_objectives: list[dict],
) -> dict:
    """
    자료 수집 계획을 수립합니다.

    ADDIE 소항목: 28. 자료 수집

    Args:
        pilot_plan: 파일럿 계획
        learning_objectives: 학습 목표 목록

    Returns:
        자료 수집 계획 (data_types, collection_methods, instruments, timeline)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 교육 평가를 위한 자료 수집 계획을 수립해주세요.

## 입력 정보
- 파일럿 계획: {json.dumps(pilot_plan, ensure_ascii=False)}
- 학습 목표: {json.dumps(learning_objectives, ensure_ascii=False)}

## 자료 수집 계획 구성 요소
1. **data_types**: 수집할 자료 유형
2. **collection_methods**: 자료 수집 방법
3. **instruments**: 측정 도구
4. **timeline**: 자료 수집 일정

## 출력 형식 (JSON)
```json
{{
  "title": "자료 수집 계획",
  "data_types": {{
    "quantitative": [
      {{"type": "사전 테스트 점수", "purpose": "기초선 측정", "source": "학습자"}},
      {{"type": "사후 테스트 점수", "purpose": "학습 성과 측정", "source": "학습자"}},
      {{"type": "만족도 점수", "purpose": "반응 평가", "source": "학습자"}},
      {{"type": "참여율/완주율", "purpose": "참여도 측정", "source": "시스템/관찰"}}
    ],
    "qualitative": [
      {{"type": "개방형 피드백", "purpose": "심층 의견 수집", "source": "학습자"}},
      {{"type": "관찰 기록", "purpose": "행동 패턴 파악", "source": "관찰자"}},
      {{"type": "인터뷰 응답", "purpose": "심층 이해", "source": "학습자/강사"}}
    ]
  }},
  "collection_methods": [
    {{
      "method": "온라인 설문",
      "data_type": ["만족도", "자기 평가"],
      "timing": "교육 직후",
      "tool": "Google Forms / SurveyMonkey"
    }},
    {{
      "method": "시험/퀴즈",
      "data_type": ["사전/사후 테스트"],
      "timing": "교육 전/후",
      "tool": "LMS 퀴즈 기능"
    }},
    {{
      "method": "관찰",
      "data_type": ["참여도", "행동 패턴"],
      "timing": "교육 중",
      "tool": "관찰 체크리스트"
    }},
    {{
      "method": "인터뷰",
      "data_type": ["심층 피드백"],
      "timing": "교육 후 1주 내",
      "tool": "인터뷰 가이드"
    }}
  ],
  "instruments": [
    {{
      "name": "사전-사후 테스트",
      "type": "지식 평가",
      "items": 20,
      "format": "객관식/단답형",
      "validity": "내용 타당도 검증 완료"
    }},
    {{
      "name": "만족도 설문",
      "type": "반응 평가",
      "items": 15,
      "format": "5점 리커트 척도 + 개방형",
      "dimensions": ["내용", "전달", "환경", "전반적 만족도"]
    }},
    {{
      "name": "관찰 체크리스트",
      "type": "행동 관찰",
      "items": 10,
      "format": "체크리스트 + 메모",
      "focus": ["참여도", "질문 빈도", "협력 수준"]
    }}
  ],
  "timeline": [
    {{"phase": "사전", "timing": "D-1 ~ D-Day", "activities": ["사전 테스트 실시", "기초 정보 수집"]}},
    {{"phase": "중간", "timing": "교육 중", "activities": ["실시간 관찰", "형성 평가"]}},
    {{"phase": "직후", "timing": "D+0", "activities": ["사후 테스트", "만족도 설문"]}},
    {{"phase": "추적", "timing": "D+7 ~ D+30", "activities": ["인터뷰", "현업 적용도 조사"]}}
  ],
  "data_management": {{
    "storage": "보안 클라우드 저장소",
    "anonymization": "개인 식별 정보 분리 저장",
    "retention_period": "3년",
    "access_control": "교육팀 및 평가팀으로 제한"
  }}
}}
```

JSON만 출력하세요."""

    try:
        response = llm.invoke(prompt)
        content = response.content

        json_match = content
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0]

        result = json.loads(json_match.strip())
        return result

    except Exception as e:
        return _fallback_create_data_collection_plan(pilot_plan, learning_objectives)


def _fallback_create_data_collection_plan(
    pilot_plan: dict,
    learning_objectives: list[dict],
) -> dict:
    """LLM 실패 시 폴백 함수

    ADDIE 소항목 28: 파일럿/초기 실행 중 자료 수집
    - 양적/질적 데이터 유형
    - 수집 방법 및 도구
    - 수집 일정 및 데이터 관리 계획
    """
    # pilot_plan에서 정보 추출 (있는 경우)
    pilot_scope = pilot_plan.get("pilot_scope") if pilot_plan else None
    pilot_participants = pilot_plan.get("participants") if pilot_plan else None
    pilot_duration = pilot_plan.get("duration") if pilot_plan else None

    # 학습 목표에서 평가 대상 추출
    objective_statements = []
    for obj in (learning_objectives or [])[:5]:
        if isinstance(obj, dict):
            stmt = obj.get("statement", obj.get("description", ""))
            if stmt:
                objective_statements.append(stmt)

    return {
        "title": "파일럿/초기 실행 자료 수집 계획",
        "pilot_info": {
            "scope": pilot_scope or "소규모 파일럿 (대상 인원의 10-20%)",
            "participants": pilot_participants or "20-30명",
            "duration": pilot_duration or "1-2주",
        },
        "data_types": {
            "quantitative": [
                {"type": "사전 테스트 점수", "purpose": "기초선 측정", "source": "학습자"},
                {"type": "사후 테스트 점수", "purpose": "학습 성과 측정", "source": "학습자"},
                {"type": "만족도 점수", "purpose": "반응 평가 (Kirkpatrick L1)", "source": "학습자"},
                {"type": "참여율/완주율", "purpose": "참여도 및 이탈 분석", "source": "시스템/관찰"},
                {"type": "퀴즈 정답률", "purpose": "형성평가 결과 분석", "source": "LMS"},
            ],
            "qualitative": [
                {"type": "개방형 피드백", "purpose": "개선점 및 심층 의견 수집", "source": "학습자"},
                {"type": "관찰 기록", "purpose": "학습 행동 패턴 파악", "source": "관찰자/강사"},
                {"type": "인터뷰 응답", "purpose": "심층 이해 및 맥락 파악", "source": "학습자/강사"},
                {"type": "질문/토론 기록", "purpose": "참여도 및 이해도 분석", "source": "강사"},
            ],
        },
        "collection_methods": [
            {
                "method": "온라인 설문",
                "data_type": ["만족도", "자기 평가", "개방형 피드백"],
                "timing": "교육 직후 (D+0)",
                "tool": "Google Forms / SurveyMonkey / LMS 설문",
                "response_rate_target": "80% 이상",
            },
            {
                "method": "사전-사후 테스트",
                "data_type": ["지식/기술 평가"],
                "timing": "교육 전/후",
                "tool": "LMS 퀴즈 기능 / 별도 평가 플랫폼",
                "response_rate_target": "95% 이상",
            },
            {
                "method": "관찰",
                "data_type": ["참여도", "학습 행동", "상호작용 패턴"],
                "timing": "교육 중 실시간",
                "tool": "관찰 체크리스트 / 비디오 녹화 (동의 시)",
                "response_rate_target": "100%",
            },
            {
                "method": "포커스 그룹 인터뷰",
                "data_type": ["심층 피드백", "개선 제안"],
                "timing": "교육 후 1주 내",
                "tool": "인터뷰 가이드 / 녹음",
                "response_rate_target": "참가자의 20%",
            },
        ],
        "instruments": [
            {
                "name": "사전-사후 테스트",
                "type": "지식/기술 평가",
                "items": 20,
                "format": "객관식 15문항 + 단답형 5문항",
                "validity": "내용 타당도 검증 (SME 검토)",
                "linked_objectives": objective_statements[:3] if objective_statements else ["학습 목표 1", "학습 목표 2"],
            },
            {
                "name": "만족도 설문지",
                "type": "반응 평가 (Kirkpatrick Level 1)",
                "items": 15,
                "format": "5점 리커트 척도 12문항 + 개방형 3문항",
                "dimensions": ["교육 내용", "교수 방법", "학습 환경", "전반적 만족도"],
            },
            {
                "name": "관찰 체크리스트",
                "type": "행동 관찰",
                "items": 10,
                "format": "체크리스트 + 메모란",
                "focus_areas": ["적극적 참여", "질문 빈도", "동료 협력", "과제 수행"],
            },
            {
                "name": "인터뷰 가이드",
                "type": "질적 데이터 수집",
                "items": 8,
                "format": "반구조화 인터뷰 (30분)",
                "topics": ["학습 경험", "어려웠던 점", "개선 제안", "현업 적용 계획"],
            },
        ],
        "timeline": [
            {
                "phase": "사전 (D-1 ~ D-Day)",
                "timing": "교육 시작 전",
                "activities": ["사전 테스트 실시", "기초 정보 수집", "참가자 배경 조사"],
                "responsible": "평가팀/교육운영팀",
            },
            {
                "phase": "중간 (교육 중)",
                "timing": "교육 진행 시",
                "activities": ["실시간 관찰 기록", "형성평가 실시", "참여도 모니터링"],
                "responsible": "강사/관찰자",
            },
            {
                "phase": "직후 (D+0)",
                "timing": "교육 종료 직후",
                "activities": ["사후 테스트 실시", "만족도 설문 배포", "즉각 피드백 수집"],
                "responsible": "평가팀",
            },
            {
                "phase": "추적 (D+7 ~ D+30)",
                "timing": "교육 후 1주~1개월",
                "activities": ["포커스 그룹 인터뷰", "현업 적용도 조사", "지연 효과 측정"],
                "responsible": "평가팀/현업 관리자",
            },
        ],
        "data_management": {
            "storage": "보안 클라우드 저장소 (암호화)",
            "anonymization": "개인 식별 정보 분리 저장, 익명화 처리",
            "retention_period": "3년 (법적 요건에 따름)",
            "access_control": "교육팀 및 평가팀으로 제한, 접근 로그 기록",
            "backup": "주간 자동 백업, 이중화 저장",
        },
        "analysis_plan": {
            "quantitative": "기술 통계, 대응표본 t-검정 (사전-사후 비교)",
            "qualitative": "주제 분석 (thematic analysis), 코딩 및 범주화",
            "integration": "혼합 방법 설계 - 양적 결과를 질적 데이터로 보완 설명",
        },
    }


@tool
def create_formative_improvement(
    pilot_results: dict,
    learning_objectives: list[dict],
) -> dict:
    """
    형성평가 기반 개선 계획을 수립합니다.

    ADDIE 소항목: 29. 형성평가 기반 개선

    Args:
        pilot_results: 파일럿 결과
        learning_objectives: 학습 목표 목록

    Returns:
        개선 계획 (analysis_results, improvement_areas, action_plan)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 형성평가 결과를 바탕으로 개선 계획을 수립해주세요.

## 입력 정보
- 파일럿 결과: {json.dumps(pilot_results, ensure_ascii=False)}
- 학습 목표: {json.dumps(learning_objectives, ensure_ascii=False)}

## 개선 계획 구성 요소
1. **analysis_results**: 형성평가 분석 결과
2. **improvement_areas**: 개선이 필요한 영역
3. **action_plan**: 구체적인 개선 실행 계획
4. **validation_plan**: 개선 효과 검증 계획

## 출력 형식 (JSON)
```json
{{
  "title": "형성평가 기반 개선 계획",
  "analysis_results": {{
    "strengths": [
      {{"area": "콘텐츠 품질", "finding": "학습 자료의 명확성과 구조화에 대한 긍정적 피드백", "evidence": "만족도 4.5/5.0"}},
      {{"area": "강사 전달력", "finding": "사례 중심 설명이 효과적", "evidence": "참가자 피드백"}}
    ],
    "weaknesses": [
      {{"area": "시간 배분", "finding": "실습 시간 부족으로 일부 학습자 어려움 호소", "evidence": "관찰 기록, 피드백"}},
      {{"area": "상호작용", "finding": "대규모 그룹에서 개별 질문 기회 제한", "evidence": "참여도 데이터"}}
    ],
    "quantitative_summary": {{
      "pre_test_avg": 62,
      "post_test_avg": 78,
      "improvement_rate": "25.8%",
      "satisfaction_score": 4.2,
      "completion_rate": "94%"
    }}
  }},
  "improvement_areas": [
    {{
      "priority": "높음",
      "area": "실습 시간 확보",
      "current_state": "실습 20분, 이론 60분",
      "target_state": "실습 35분, 이론 45분",
      "rationale": "실습 시간 부족으로 실제 적용 능력 발달 제한"
    }},
    {{
      "priority": "중간",
      "area": "상호작용 강화",
      "current_state": "전체 그룹 Q&A 위주",
      "target_state": "소그룹 토론 + 개별 피드백 추가",
      "rationale": "개별 학습자 참여 기회 확대 필요"
    }},
    {{
      "priority": "낮음",
      "area": "보조 자료 보완",
      "current_state": "슬라이드 중심",
      "target_state": "퀵 레퍼런스 카드 추가 제공",
      "rationale": "학습 후 참고 자료 요청"
    }}
  ],
  "action_plan": [
    {{
      "item": "실습 활동 재설계",
      "responsible": "교수설계팀",
      "timeline": "1주",
      "deliverable": "수정된 레슨 플랜",
      "resources_needed": "추가 실습 자료"
    }},
    {{
      "item": "소그룹 활동 스크립트 개발",
      "responsible": "교수설계팀",
      "timeline": "3일",
      "deliverable": "소그룹 토론 가이드",
      "resources_needed": "없음"
    }},
    {{
      "item": "퀵 레퍼런스 카드 제작",
      "responsible": "교수설계팀",
      "timeline": "3일",
      "deliverable": "2페이지 요약 카드",
      "resources_needed": "인쇄비"
    }}
  ],
  "validation_plan": {{
    "method": "2차 파일럿 또는 A/B 테스트",
    "metrics": ["실습 완료율", "적용 능력 평가", "만족도 변화"],
    "success_criteria": "개선 영역 만족도 4.0 이상, 실습 완료율 95% 이상",
    "timeline": "개선 완료 후 2주 내"
  }},
  "lessons_learned": [
    "실습 시간은 이론 시간의 최소 50%를 확보해야 효과적",
    "대규모 그룹에서는 소그룹 활동을 2회 이상 포함",
    "학습 후 참고 자료에 대한 니즈가 높음"
  ]
}}
```

JSON만 출력하세요."""

    try:
        response = llm.invoke(prompt)
        content = response.content

        json_match = content
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0]

        result = json.loads(json_match.strip())
        return result

    except Exception as e:
        return _fallback_create_formative_improvement(pilot_results, learning_objectives)


def _fallback_create_formative_improvement(
    pilot_results: dict,
    learning_objectives: list[dict],
) -> dict:
    """LLM 실패 시 폴백 함수"""
    return {
        "title": "형성평가 기반 개선 계획",
        "analysis_results": {
            "strengths": [{"area": "콘텐츠 품질", "finding": "긍정적 피드백"}],
            "weaknesses": [{"area": "시간 배분", "finding": "실습 시간 부족"}],
            "quantitative_summary": {
                "improvement_rate": "20%+",
                "satisfaction_score": 4.0,
            },
        },
        "improvement_areas": [
            {"priority": "높음", "area": "실습 시간 확보"},
            {"priority": "중간", "area": "상호작용 강화"},
        ],
        "action_plan": [
            {"item": "레슨 플랜 수정", "timeline": "1주"},
            {"item": "보조 자료 보완", "timeline": "3일"},
        ],
        "validation_plan": {
            "method": "2차 파일럿",
            "success_criteria": "만족도 4.0 이상",
        },
        "lessons_learned": ["실습 시간 확보 중요", "소그룹 활동 효과적"],
    }


@tool
def create_program_evaluation(
    program_title: str,
    objectives: list[dict],
    target_audience: Optional[str] = None,
) -> dict:
    """
    교육 프로그램의 성과평가 계획을 생성합니다. (Kirkpatrick 4단계 모델 기반)

    ADDIE 소항목: 30-33. 총괄평가, 채택결정, 효과분석, 유지관리

    Args:
        program_title: 교육 프로그램 제목
        objectives: 학습 목표 목록
        target_audience: 대상 학습자 (선택)

    Returns:
        성과평가 계획 (Kirkpatrick 4단계: 반응, 학습, 행동, 결과)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. Kirkpatrick 4단계 평가 모델에 기반한 성과평가 계획을 수립해주세요.

## 입력 정보
- 프로그램명: {program_title}
- 학습 목표: {json.dumps(objectives, ensure_ascii=False)}
- 대상 학습자: {target_audience or "일반 성인 학습자"}

## Kirkpatrick 4단계 평가 모델
1. **Level 1 - 반응(Reaction)**: 학습자 만족도 및 참여도
2. **Level 2 - 학습(Learning)**: 지식, 기술, 태도 습득 정도
3. **Level 3 - 행동(Behavior)**: 업무 현장에서의 적용 및 행동 변화
4. **Level 4 - 결과(Results)**: 조직 성과에 미치는 영향

## 요구사항
1. 각 단계별 평가 방법과 도구 제시
2. 평가 시점 및 담당자 명시
3. 측정 지표(KPI) 포함
4. ROI 계산 방식 제안

## 출력 형식 (JSON)
```json
{{
  "program_title": "{program_title}",
  "evaluation_model": "Kirkpatrick 4-Level",
  "levels": {{
    "level_1_reaction": {{
      "description": "학습자의 교육 만족도 및 참여도 평가",
      "timing": "교육 직후",
      "methods": ["만족도 설문조사", "참여도 관찰", "즉각적 피드백 수집"],
      "tools": ["Likert 척도 설문지", "참여율 체크리스트"],
      "kpis": ["전체 만족도 점수 (목표: 4.0/5.0 이상)", "참여율 (목표: 90% 이상)"],
      "responsible": "교육 운영팀"
    }},
    "level_2_learning": {{
      "description": "학습 목표 달성도 및 지식/기술 습득 평가",
      "timing": "교육 중/직후",
      "methods": ["사전-사후 테스트", "실습 평가", "역량 체크리스트"],
      "tools": ["지식 평가 문항", "스킬 체크리스트", "시뮬레이션 과제"],
      "kpis": ["사전-사후 점수 향상률 (목표: 30% 이상)", "학습 목표 달성률 (목표: 80% 이상)"],
      "responsible": "강사 및 평가팀"
    }},
    "level_3_behavior": {{
      "description": "업무 현장에서의 학습 내용 적용 및 행동 변화 평가",
      "timing": "교육 후 1-3개월",
      "methods": ["현업 적용도 조사", "상사/동료 평가", "행동 관찰"],
      "tools": ["현업 적용 체크리스트", "360도 피드백", "업무 일지"],
      "kpis": ["현업 적용률 (목표: 70% 이상)", "행동 변화 점수 향상"],
      "responsible": "현업 관리자 및 HR팀"
    }},
    "level_4_results": {{
      "description": "조직 성과에 미치는 영향 및 비즈니스 결과 평가",
      "timing": "교육 후 6-12개월",
      "methods": ["성과 지표 분석", "비용-효과 분석", "ROI 계산"],
      "tools": ["성과 대시보드", "재무 분석 도구"],
      "kpis": ["생산성 향상률", "오류/사고 감소율", "매출/이익 기여도"],
      "responsible": "경영진 및 재무팀"
    }}
  }},
  "roi_calculation": {{
    "formula": "ROI = ((교육으로 인한 이익 - 교육 비용) / 교육 비용) × 100",
    "benefit_factors": ["생산성 향상", "오류 감소", "이직률 감소", "고객 만족도 향상"],
    "cost_factors": ["강사비", "교재비", "시설비", "참가자 인건비", "기회비용"]
  }},
  "evaluation_schedule": [
    {{"phase": "Level 1", "timing": "D+0", "duration": "교육 종료 직후"}},
    {{"phase": "Level 2", "timing": "D+0 ~ D+7", "duration": "교육 중 및 종료 1주 내"}},
    {{"phase": "Level 3", "timing": "D+30 ~ D+90", "duration": "교육 후 1-3개월"}},
    {{"phase": "Level 4", "timing": "D+180 ~ D+365", "duration": "교육 후 6-12개월"}}
  ],
  "success_criteria": {{
    "short_term": "Level 1-2 목표 달성",
    "mid_term": "Level 3 현업 적용률 70% 이상",
    "long_term": "Level 4 ROI 100% 이상"
  }},
  "adoption_decision": {{
    "recommendation": "adopt",
    "rationale": "평가 결과 성공 기준을 충족하여 프로그램 채택을 권고함",
    "conditions": ["파일럿 피드백 반영", "운영 매뉴얼 보완"],
    "next_steps": ["개선 사항 반영", "전사 론칭 준비"]
  }},
  "maintenance_plan": {{
    "review_cycle": "분기별 콘텐츠 검토",
    "update_triggers": ["법규 변경", "기술 변화", "피드백 반영"],
    "version_control": "주요 변경 시 버전 업데이트",
    "responsible": "교수설계팀"
  }}
}}
```

JSON만 출력하세요."""

    try:
        response = llm.invoke(prompt)
        content = response.content

        json_match = content
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0]

        result = json.loads(json_match.strip())
        return result

    except Exception as e:
        return _fallback_create_program_evaluation(program_title, objectives, target_audience)


def _fallback_create_program_evaluation(
    program_title: str,
    objectives: list[dict],
    target_audience: Optional[str] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    return {
        "program_title": program_title,
        "evaluation_model": "Kirkpatrick 4-Level",
        "levels": {
            "level_1_reaction": {
                "description": "학습자 만족도 및 참여도 평가",
                "timing": "교육 직후",
                "methods": ["만족도 설문", "참여도 관찰"],
                "kpis": ["만족도 4.0/5.0 이상", "참여율 90% 이상"],
            },
            "level_2_learning": {
                "description": "학습 목표 달성도 평가",
                "timing": "교육 중/직후",
                "methods": ["사전-사후 테스트", "실습 평가"],
                "kpis": ["향상률 30% 이상", "목표 달성률 80% 이상"],
            },
            "level_3_behavior": {
                "description": "현업 적용 및 행동 변화 평가",
                "timing": "교육 후 1-3개월",
                "methods": ["현업 적용도 조사", "상사 평가"],
                "kpis": ["현업 적용률 70% 이상"],
            },
            "level_4_results": {
                "description": "조직 성과 영향 평가",
                "timing": "교육 후 6-12개월",
                "methods": ["성과 지표 분석", "ROI 계산"],
                "kpis": ["생산성 향상", "ROI 100% 이상"],
            },
        },
        "roi_calculation": {
            "formula": "ROI = ((이익 - 비용) / 비용) × 100",
            "benefit_factors": ["생산성 향상", "오류 감소"],
            "cost_factors": ["강사비", "교재비", "시설비"],
        },
        "evaluation_schedule": [
            {"phase": "Level 1-2", "timing": "교육 직후"},
            {"phase": "Level 3", "timing": "1-3개월 후"},
            {"phase": "Level 4", "timing": "6-12개월 후"},
        ],
        "success_criteria": {
            "short_term": "Level 1-2 목표 달성",
            "mid_term": "현업 적용률 70%",
            "long_term": "ROI 100% 이상",
        },
        "adoption_decision": {
            "recommendation": "adopt",
            "conditions": ["피드백 반영", "매뉴얼 보완"],
        },
        "maintenance_plan": {
            "review_cycle": "분기별",
            "update_triggers": ["법규 변경", "피드백 반영"],
        },
    }
