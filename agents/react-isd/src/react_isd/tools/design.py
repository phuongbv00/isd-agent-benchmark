"""
Design 단계 도구

ADDIE의 두 번째 단계: 학습 목표, 평가 계획, 교수 전략 설계
LLM을 활용하여 맥락에 맞는 깊이 있는 콘텐츠를 생성합니다.
"""

import json
from typing import Optional
from langchain_core.tools import tool

from shared.llm import create_chat_model, get_default_llm_config


def get_llm():
    return create_chat_model(get_default_llm_config())


@tool
def design_objectives(
    learning_goals: list[str],
    target_audience: str,
    difficulty: Optional[str] = None,
) -> list[dict]:
    """
    Bloom's Taxonomy 기반 학습 목표를 설계합니다.

    Args:
        learning_goals: 원본 학습 목표 목록
        target_audience: 학습 대상자
        difficulty: 난이도 (선택)

    Returns:
        학습 목표 목록 (수준, 진술문, 동사 포함)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 다음 학습 목표들을 Bloom's Taxonomy 기반으로 재구성하여 측정 가능한 학습 목표로 변환해주세요.

## 입력 정보
- 원본 학습 목표: {json.dumps(learning_goals, ensure_ascii=False)}
- 학습 대상자: {target_audience}
- 난이도: {difficulty or "medium"}

## Bloom's Taxonomy 동사 참고
- 기억: 정의하다, 나열하다, 명명하다, 인식하다, 회상하다
- 이해: 설명하다, 요약하다, 해석하다, 분류하다, 비교하다
- 적용: 적용하다, 시연하다, 사용하다, 실행하다, 구현하다
- 분석: 분석하다, 구별하다, 조직하다, 비판하다, 검토하다
- 평가: 평가하다, 판단하다, 정당화하다, 비평하다, 추천하다
- 창조: 설계하다, 개발하다, 생성하다, 구성하다, 창작하다

## 요구사항
1. 각 원본 목표에 대해 1-2개의 측정 가능한 학습 목표를 생성
2. 대상자와 난이도에 맞는 Bloom 수준 선택
3. "학습자는 ~을/를 ~할 수 있다" 형식의 자연스러운 한국어 문장
4. 최소 3개 이상의 학습 목표 생성

## 출력 형식 (JSON 배열)
```json
[
  {{
    "id": "OBJ-01",
    "level": "이해",
    "statement": "학습자는 회사의 조직 구조를 설명하고 각 부서의 역할을 비교할 수 있다.",
    "bloom_verb": "설명하다",
    "measurable": true
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

        objectives = json.loads(json_match.strip())
        return objectives

    except Exception as e:
        # 폴백: 템플릿 기반 생성
        return _fallback_design_objectives(learning_goals, target_audience, difficulty)


def _fallback_design_objectives(
    learning_goals: list[str],
    target_audience: str,
    difficulty: Optional[str] = None,
) -> list[dict]:
    """LLM 실패 시 폴백 함수"""
    objectives = []
    audience_lower = target_audience.lower()

    if "초등" in audience_lower or "초보" in audience_lower:
        base_levels = ["기억", "이해", "적용"]
    elif "전문가" in audience_lower or "고급" in audience_lower:
        base_levels = ["적용", "분석", "평가", "창조"]
    else:
        base_levels = ["이해", "적용", "분석"]

    if difficulty:
        diff_lower = difficulty.lower()
        if "hard" in diff_lower:
            base_levels = ["분석", "평가", "창조"]
        elif "easy" in diff_lower:
            base_levels = ["기억", "이해"]

    for i, goal in enumerate(learning_goals):
        level = base_levels[i % len(base_levels)]
        verbs = BLOOM_VERBS.get(level, ["이해하다"])
        verb = verbs[i % len(verbs)]
        statement = f"학습자는 {goal}을/를 {verb} 수 있다."

        objectives.append({
            "id": f"OBJ-{i+1:02d}",
            "level": level,
            "statement": statement,
            "bloom_verb": verb,
            "measurable": True,
        })

    while len(objectives) < 3:
        idx = len(objectives)
        level = base_levels[idx % len(base_levels)]
        verbs = BLOOM_VERBS.get(level, ["이해하다"])
        objectives.append({
            "id": f"OBJ-{idx+1:02d}",
            "level": level,
            "statement": f"학습자는 학습 내용을 {verbs[0]} 수 있다.",
            "bloom_verb": verbs[0],
            "measurable": True,
        })

    return objectives


@tool
def design_assessment(
    objectives: list[dict],
    duration: str,
    learning_environment: str,
) -> dict:
    """
    평가 계획을 수립합니다.

    Args:
        objectives: 학습 목표 목록
        duration: 학습 시간
        learning_environment: 학습 환경

    Returns:
        평가 계획 (진단/형성/총괄 평가)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 다음 학습 목표들에 대한 평가 계획을 수립해주세요.

## 입력 정보
- 학습 목표: {json.dumps(objectives, ensure_ascii=False)}
- 학습 시간: {duration}
- 학습 환경: {learning_environment}

## 평가 유형
1. 진단 평가 (diagnostic): 학습 전 사전 지식 파악
2. 형성 평가 (formative): 학습 중 이해도 확인
3. 총괄 평가 (summative): 학습 후 최종 성취도 평가

## 요구사항
1. 각 유형별 최소 2개 이상의 평가 방법 제시
2. 학습 환경(온라인/대면)에 적합한 평가 방법 선택
3. 각 학습 목표와 연결되는 구체적인 평가 방법

## 출력 형식 (JSON)
```json
{{
  "diagnostic": ["사전 지식 설문 (10문항)", "자기 평가 체크리스트"],
  "formative": ["각 모듈별 퀴즈", "그룹 토론 참여도 평가"],
  "summative": ["종합 실기 테스트", "프로젝트 발표"]
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

        assessment = json.loads(json_match.strip())
        return assessment

    except Exception as e:
        # 폴백
        return _fallback_design_assessment(objectives, duration, learning_environment)


def _fallback_design_assessment(
    objectives: list[dict],
    duration: str,
    learning_environment: str,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    diagnostic = []
    formative = []
    summative = []

    env_lower = learning_environment.lower()
    is_online = "온라인" in env_lower

    diagnostic.append("사전 지식 퀴즈 (5문항)")
    if len(objectives) > 2:
        diagnostic.append("학습 준비도 자가 체크리스트")

    for obj in objectives[:3]:
        level = obj.get("level", "이해")
        if level in ["기억", "이해"]:
            formative.append(f"개념 확인 퀴즈 - {obj['id']}")
        elif level in ["적용", "분석"]:
            if is_online:
                formative.append(f"온라인 실습 과제 - {obj['id']}")
            else:
                formative.append(f"그룹 토론/실습 - {obj['id']}")
        else:
            formative.append(f"프로젝트 중간 점검 - {obj['id']}")

    if "시간" in duration or "분" in duration:
        summative.append("종합 퀴즈 (10문항)")
    else:
        summative.append("종합 평가 (20문항)")
        summative.append("실습 과제 평가")

    if len(objectives) > 3:
        summative.append("포트폴리오 평가")

    return {
        "diagnostic": diagnostic,
        "formative": formative,
        "summative": summative,
    }


GAGNE_EVENTS = [
    {"event": "주의 획득", "template": "{topic} 관련 흥미로운 사례/질문 제시"},
    {"event": "학습 목표 제시", "template": "오늘 학습할 내용과 달성 목표 안내"},
    {"event": "선수 학습 상기", "template": "관련 기존 지식 복습 및 연결"},
    {"event": "학습 내용 제시", "template": "{topic} 핵심 개념 설명"},
    {"event": "학습 안내 제공", "template": "예시와 시연을 통한 이해 촉진"},
    {"event": "연습 유도", "template": "개인/그룹 실습 활동"},
    {"event": "피드백 제공", "template": "실습 결과에 대한 즉각적 피드백"},
    {"event": "수행 평가", "template": "학습 목표 달성 확인 평가"},
    {"event": "파지 및 전이 강화", "template": "실제 상황 적용 방안 논의"},
]


@tool
def design_strategy(
    main_topics: list[str],
    target_audience: str,
    duration: str,
    learning_environment: str,
) -> dict:
    """
    Gagné's 9 Events 기반 교수 전략을 설계합니다.

    Args:
        main_topics: 주요 학습 주제
        target_audience: 학습 대상자
        duration: 학습 시간
        learning_environment: 학습 환경

    Returns:
        교수 전략 (모델, 교수사태, 교수 방법)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. Gagné's 9 Events of Instruction 모델을 기반으로 교수 전략을 설계해주세요.

## 입력 정보
- 주요 학습 주제: {json.dumps(main_topics, ensure_ascii=False)}
- 학습 대상자: {target_audience}
- 학습 시간: {duration}
- 학습 환경: {learning_environment}

## Gagné's 9 Events
1. 주의 획득 (Gain attention)
2. 학습 목표 제시 (Inform learners of objectives)
3. 선수 학습 상기 (Stimulate recall of prior learning)
4. 학습 내용 제시 (Present content)
5. 학습 안내 제공 (Provide learning guidance)
6. 연습 유도 (Elicit performance)
7. 피드백 제공 (Provide feedback)
8. 수행 평가 (Assess performance)
9. 파지 및 전이 강화 (Enhance retention and transfer)

## 요구사항
1. 9개 Event 모두 포함
2. 각 Event에 대해 대상자와 환경에 맞는 구체적인 활동 설명
3. 각 활동에 예상 소요 시간과 필요 자원 포함
4. 학습 주제의 특성을 반영한 맞춤형 활동

## 출력 형식 (JSON)
```json
{{
  "model": "Gagné's 9 Events",
  "sequence": [
    {{
      "event": "주의 획득",
      "activity": "실제 회사 조직도 사례를 보여주며 '여러분이 속한 팀은 어디일까요?' 질문으로 시작",
      "duration": "5분",
      "resources": ["조직도 슬라이드", "인터랙티브 설문"]
    }},
    ...
  ],
  "methods": ["강의", "토론", "실습", "시뮬레이션"]
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

        strategy = json.loads(json_match.strip())
        return strategy

    except Exception as e:
        # 폴백
        return _fallback_design_strategy(main_topics, target_audience, duration, learning_environment)


def _fallback_design_strategy(
    main_topics: list[str],
    target_audience: str,
    duration: str,
    learning_environment: str,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    topic = main_topics[0] if main_topics else "학습 주제"
    audience_lower = target_audience.lower()
    env_lower = learning_environment.lower()

    sequence = []
    for event_info in GAGNE_EVENTS:
        activity = event_info["template"].format(topic=topic)

        if "초등" in audience_lower:
            if event_info["event"] == "주의 획득":
                activity = "재미있는 영상/게임으로 시작"
            elif event_info["event"] == "연습 유도":
                activity = "놀이 기반 활동"

        if "온라인" in env_lower:
            if event_info["event"] == "연습 유도":
                activity = "온라인 실습 환경에서 개별 연습"

        sequence.append({
            "event": event_info["event"],
            "activity": activity,
            "duration": None,
            "resources": [],
        })

    methods = []
    if "온라인" in env_lower:
        methods.extend(["동영상 강의", "화면 공유 시연", "온라인 토론"])
    else:
        methods.extend(["강의", "시연", "그룹 토론", "실습"])

    if "초등" in audience_lower:
        methods.append("게임 기반 학습")
    elif "직장인" in audience_lower:
        methods.append("사례 기반 학습")

    return {
        "model": "Gagné's 9 Events",
        "sequence": sequence,
        "methods": methods,
    }


@tool
def design_content(
    learning_objectives: list[dict],
    main_topics: list[str],
    duration: str,
) -> dict:
    """
    교수 내용을 선정합니다.

    ADDIE 소항목: 13. 교수 내용 선정

    Args:
        learning_objectives: 학습 목표 목록
        main_topics: 주요 학습 주제
        duration: 총 학습 시간

    Returns:
        교수 내용 선정 결과 (content_structure, scope_sequence, time_allocation)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 다음 정보를 바탕으로 교수 내용을 선정해주세요.

## 입력 정보
- 학습 목표: {json.dumps(learning_objectives, ensure_ascii=False)}
- 주요 주제: {json.dumps(main_topics, ensure_ascii=False)}
- 총 학습 시간: {duration}

## 분석 항목
1. **content_structure**: 교수 내용의 계층적 구조 (단원/주제/세부내용)
2. **scope_sequence**: 내용의 범위와 순서 (나선형/계열형 등)
3. **time_allocation**: 주제별 시간 배분
4. **content_depth**: 각 주제별 학습 깊이 (기초/중급/심화)

## 출력 형식 (JSON)
```json
{{
  "content_structure": [
    {{
      "unit": "단원 1: 조직 문화 이해",
      "topics": [
        {{"name": "비전과 미션", "subtopics": ["비전의 정의", "미션의 역할"], "depth": "기초"}},
        {{"name": "핵심 가치", "subtopics": ["가치 체계", "행동 강령"], "depth": "중급"}}
      ]
    }}
  ],
  "scope_sequence": {{
    "approach": "나선형 접근",
    "rationale": "기초 개념부터 시작하여 점진적으로 심화",
    "sequence": ["개념 이해", "사례 분석", "적용 연습", "종합 평가"]
  }},
  "time_allocation": [
    {{"topic": "조직 문화 이해", "duration": "30분", "percentage": 25}},
    {{"topic": "업무 프로세스", "duration": "45분", "percentage": 37}},
    {{"topic": "협업과 커뮤니케이션", "duration": "30분", "percentage": 25}},
    {{"topic": "종합 정리 및 평가", "duration": "15분", "percentage": 13}}
  ],
  "content_selection_criteria": [
    "학습 목표와의 직접적 연관성",
    "학습자 수준 적합성",
    "실무 적용 가능성",
    "시간 대비 학습 효과"
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
        return _fallback_design_content(learning_objectives, main_topics, duration)


def _fallback_design_content(
    learning_objectives: list[dict],
    main_topics: list[str],
    duration: str,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    content_structure = []
    for i, topic in enumerate(main_topics[:3]):
        content_structure.append({
            "unit": f"단원 {i+1}: {topic}",
            "topics": [
                {"name": f"{topic} 기초", "subtopics": ["핵심 개념", "기본 원리"], "depth": "기초"},
                {"name": f"{topic} 적용", "subtopics": ["사례 분석", "실습"], "depth": "중급"},
            ]
        })

    time_allocation = []
    num_topics = len(main_topics)
    for i, topic in enumerate(main_topics):
        pct = 100 // num_topics if num_topics > 0 else 100
        time_allocation.append({
            "topic": topic,
            "duration": f"{pct}%",
            "percentage": pct,
        })

    return {
        "content_structure": content_structure,
        "scope_sequence": {
            "approach": "계열형 접근",
            "rationale": "기초부터 심화까지 단계별 학습",
            "sequence": ["개념 이해", "사례 분석", "적용 연습", "종합 평가"],
        },
        "time_allocation": time_allocation,
        "content_selection_criteria": [
            "학습 목표와의 직접적 연관성",
            "학습자 수준 적합성",
            "실무 적용 가능성",
        ],
    }


@tool
def design_non_instructional(
    learning_goals: list[str],
    constraints: Optional[list[str]] = None,
) -> dict:
    """
    비교수적 전략을 설계합니다.

    ADDIE 소항목: 15. 비교수적 전략

    Args:
        learning_goals: 학습 목표 목록
        constraints: 제약 조건 목록 (선택)

    Returns:
        비교수적 전략 (job_aids, performance_support, environmental_changes)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 교육 외의 비교수적 해결책을 설계해주세요.

## 입력 정보
- 학습 목표: {json.dumps(learning_goals, ensure_ascii=False)}
- 제약 조건: {json.dumps(constraints, ensure_ascii=False) if constraints else "정보 없음"}

## 비교수적 전략 영역
1. **job_aids**: 직무 보조 도구 (체크리스트, 매뉴얼, 퀵 레퍼런스 등)
2. **performance_support**: 수행 지원 시스템 (헬프데스크, FAQ, 검색 시스템 등)
3. **environmental_changes**: 환경 개선 (도구/장비, 작업 환경, 프로세스 개선 등)
4. **motivation_incentives**: 동기 부여 요소 (인센티브, 인정, 경력 개발 등)

## 출력 형식 (JSON)
```json
{{
  "job_aids": [
    {{"type": "체크리스트", "name": "업무 진행 체크리스트", "description": "일상 업무 수행 시 확인해야 할 항목 목록", "format": "인쇄물/디지털"}},
    {{"type": "퀵 레퍼런스", "name": "시스템 단축키 카드", "description": "자주 사용하는 기능 빠른 참조용", "format": "데스크 카드"}},
    {{"type": "매뉴얼", "name": "업무 프로세스 가이드", "description": "상세 업무 절차 안내서", "format": "PDF/온라인"}}
  ],
  "performance_support": [
    {{"type": "헬프데스크", "description": "실시간 질의응답 지원", "availability": "업무 시간 중"}},
    {{"type": "FAQ 시스템", "description": "자주 묻는 질문 데이터베이스", "access": "인트라넷"}},
    {{"type": "멘토링", "description": "선배 직원 1:1 멘토링", "duration": "3개월"}}
  ],
  "environmental_changes": [
    {{"area": "도구/장비", "change": "최신 소프트웨어 버전 업그레이드", "impact": "업무 효율성 향상"}},
    {{"area": "프로세스", "change": "불필요한 승인 단계 간소화", "impact": "업무 속도 개선"}}
  ],
  "motivation_incentives": [
    {{"type": "인정", "description": "우수 수행자 월간 표창", "frequency": "월 1회"}},
    {{"type": "경력", "description": "역량 향상 시 승진 기회", "criteria": "목표 달성률 기준"}}
  ],
  "integration_with_training": "교육과 비교수적 전략을 병행하여 즉각적 수행 지원과 장기적 역량 개발 동시 추구"
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
        return _fallback_design_non_instructional(learning_goals, constraints)


def _fallback_design_non_instructional(
    learning_goals: list[str],
    constraints: Optional[list[str]] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    return {
        "job_aids": [
            {"type": "체크리스트", "name": "업무 수행 체크리스트", "description": "일상 업무 확인 항목", "format": "인쇄물/디지털"},
            {"type": "퀵 레퍼런스", "name": "핵심 내용 요약 카드", "description": "주요 개념 빠른 참조용", "format": "데스크 카드"},
            {"type": "매뉴얼", "name": "업무 가이드북", "description": "상세 절차 안내서", "format": "PDF"},
        ],
        "performance_support": [
            {"type": "헬프데스크", "description": "질의응답 지원", "availability": "업무 시간"},
            {"type": "FAQ", "description": "자주 묻는 질문", "access": "인트라넷"},
            {"type": "멘토링", "description": "선배 멘토 지원", "duration": "3개월"},
        ],
        "environmental_changes": [
            {"area": "도구", "change": "필요 도구 제공", "impact": "업무 효율성"},
            {"area": "프로세스", "change": "절차 간소화", "impact": "업무 속도"},
        ],
        "motivation_incentives": [
            {"type": "인정", "description": "우수 수행자 표창", "frequency": "월 1회"},
            {"type": "경력", "description": "역량 개발 기회", "criteria": "성과 기준"},
        ],
        "integration_with_training": "교육과 비교수적 전략 병행으로 효과 극대화",
    }


@tool
def design_media(
    learning_environment: str,
    target_audience: str,
    content_types: Optional[list[str]] = None,
) -> dict:
    """
    매체를 선정합니다.

    ADDIE 소항목: 16. 매체 선정

    Args:
        learning_environment: 학습 환경 (온라인/대면/블렌디드)
        target_audience: 학습 대상자
        content_types: 콘텐츠 유형 목록 (선택)

    Returns:
        매체 선정 결과 (primary_media, supporting_media, rationale)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 학습 환경과 대상에 최적화된 매체를 선정해주세요.

## 입력 정보
- 학습 환경: {learning_environment}
- 학습 대상자: {target_audience}
- 콘텐츠 유형: {json.dumps(content_types, ensure_ascii=False) if content_types else "일반적인 유형"}

## 매체 선정 기준
1. 학습 환경 적합성
2. 학습자 특성 및 선호도
3. 학습 목표 달성 효과
4. 비용 대비 효과
5. 기술적 실현 가능성

## 출력 형식 (JSON)
```json
{{
  "primary_media": [
    {{
      "type": "동영상",
      "usage": "핵심 개념 설명",
      "format": "MP4 (5-10분 단위)",
      "advantages": ["시각적 이해 촉진", "반복 학습 가능"],
      "considerations": ["자막 필수", "모바일 최적화"]
    }},
    {{
      "type": "프레젠테이션",
      "usage": "강의 진행",
      "format": "PPT/PDF",
      "advantages": ["구조화된 정보 전달", "인쇄 가능"],
      "considerations": ["슬라이드당 핵심 내용 1개"]
    }}
  ],
  "supporting_media": [
    {{"type": "인포그래픽", "usage": "프로세스 시각화"}},
    {{"type": "인터랙티브 퀴즈", "usage": "형성 평가"}},
    {{"type": "워크시트", "usage": "실습 및 정리"}}
  ],
  "delivery_platform": {{
    "primary": "LMS (학습관리시스템)",
    "backup": "클라우드 저장소 (Google Drive/OneDrive)",
    "live_session": "Zoom/Teams"
  }},
  "media_production_plan": [
    {{"media": "동영상", "quantity": "5개", "duration": "총 30분", "timeline": "2주"}},
    {{"media": "PPT", "quantity": "30슬라이드", "duration": "-", "timeline": "1주"}}
  ],
  "accessibility_requirements": [
    "동영상 자막 제공",
    "색상 대비 준수",
    "스크린 리더 호환"
  ],
  "rationale": "온라인 환경과 성인 학습자 특성을 고려하여 동영상 중심의 비동기 학습과 실시간 토론 병행 전략 수립"
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
        return _fallback_design_media(learning_environment, target_audience, content_types)


def _fallback_design_media(
    learning_environment: str,
    target_audience: str,
    content_types: Optional[list[str]] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    env_lower = learning_environment.lower()
    is_online = "온라인" in env_lower

    if is_online:
        primary_media = [
            {"type": "동영상", "usage": "핵심 개념 설명", "format": "MP4", "advantages": ["시각적 이해", "반복 학습"], "considerations": ["자막 필수"]},
            {"type": "웹 콘텐츠", "usage": "인터랙티브 학습", "format": "HTML5", "advantages": ["접근성", "상호작용"], "considerations": ["브라우저 호환"]},
        ]
        delivery_platform = {"primary": "LMS", "live_session": "Zoom/Teams"}
    else:
        primary_media = [
            {"type": "프레젠테이션", "usage": "강의 진행", "format": "PPT", "advantages": ["구조화", "인쇄 가능"], "considerations": ["폰트 크기"]},
            {"type": "인쇄물", "usage": "학습자료 배포", "format": "PDF/종이", "advantages": ["필기 가능", "휴대성"], "considerations": ["컬러 인쇄"]},
        ]
        delivery_platform = {"primary": "교실", "equipment": "프로젝터/스크린"}

    return {
        "primary_media": primary_media,
        "supporting_media": [
            {"type": "인포그래픽", "usage": "시각화"},
            {"type": "퀴즈", "usage": "형성 평가"},
            {"type": "워크시트", "usage": "실습"},
        ],
        "delivery_platform": delivery_platform,
        "media_production_plan": [
            {"media": "주요 자료", "quantity": "5개", "timeline": "1-2주"},
        ],
        "accessibility_requirements": [
            "자막/캡션 제공",
            "색상 대비 준수",
        ],
        "rationale": f"{learning_environment} 환경과 {target_audience} 특성을 고려한 매체 선정",
    }


@tool
def design_storyboard(
    lesson_plan: dict,
    media_selection: Optional[dict] = None,
) -> dict:
    """
    스토리보드/화면 흐름을 설계합니다.

    ADDIE 소항목: 18. 스토리보드/화면 흐름

    Args:
        lesson_plan: 레슨 플랜
        media_selection: 매체 선정 결과 (선택)

    Returns:
        스토리보드 (screens, navigation, interactions)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 레슨 플랜에 따른 스토리보드를 설계해주세요.

## 입력 정보
- 레슨 플랜: {json.dumps(lesson_plan, ensure_ascii=False)}
- 매체 선정: {json.dumps(media_selection, ensure_ascii=False) if media_selection else "일반적인 매체"}

## 스토리보드 구성 요소
1. 화면별 구성 (제목, 내용, 시각 요소, 내레이션)
2. 내비게이션 흐름
3. 상호작용 요소
4. 학습 진행 표시

## 출력 형식 (JSON)
```json
{{
  "title": "교육 프로그램 스토리보드",
  "total_screens": 15,
  "screens": [
    {{
      "screen_number": 1,
      "screen_type": "도입",
      "title": "교육 소개",
      "content": {{
        "text": "환영합니다. 본 교육의 목표와 일정을 안내합니다.",
        "visual": "회사 로고, 교육 제목 애니메이션",
        "audio": "환영 인사 내레이션"
      }},
      "interaction": "시작 버튼 클릭",
      "duration": "30초",
      "notes": "배경음악 페이드인"
    }},
    {{
      "screen_number": 2,
      "screen_type": "목표 제시",
      "title": "학습 목표",
      "content": {{
        "text": "본 교육을 통해 달성할 목표입니다.",
        "visual": "목표 목록 순차 표시",
        "audio": "각 목표 읽기"
      }},
      "interaction": "다음 버튼 또는 자동 진행",
      "duration": "45초",
      "notes": "목표별 아이콘 표시"
    }}
  ],
  "navigation": {{
    "type": "선형 + 메뉴 접근",
    "controls": ["이전", "다음", "메뉴", "도움말"],
    "progress_indicator": "진행률 바 상단 표시"
  }},
  "interaction_types": [
    {{"type": "클릭", "purpose": "정보 확장", "frequency": "화면당 1-2회"}},
    {{"type": "드래그앤드롭", "purpose": "매칭 활동", "frequency": "퀴즈 화면"}},
    {{"type": "입력", "purpose": "성찰 활동", "frequency": "정리 화면"}}
  ],
  "visual_guidelines": {{
    "color_scheme": "기업 CI 색상 기반",
    "font": "맑은 고딕, 본문 14pt 이상",
    "layout": "상단 제목, 중앙 내용, 하단 내비게이션"
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
        return _fallback_design_storyboard(lesson_plan, media_selection)


def _fallback_design_storyboard(
    lesson_plan: dict,
    media_selection: Optional[dict] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    modules = lesson_plan.get("modules", [])
    screens = []

    # 도입 화면
    screens.append({
        "screen_number": 1,
        "screen_type": "도입",
        "title": "교육 소개",
        "content": {
            "text": "교육 목표와 일정 안내",
            "visual": "타이틀 화면",
            "audio": "환영 인사",
        },
        "interaction": "시작 버튼",
        "duration": "30초",
    })

    # 목표 화면
    screens.append({
        "screen_number": 2,
        "screen_type": "목표 제시",
        "title": "학습 목표",
        "content": {
            "text": "학습 목표 목록",
            "visual": "목표 리스트",
            "audio": "목표 설명",
        },
        "interaction": "다음",
        "duration": "45초",
    })

    # 모듈별 화면
    screen_num = 3
    for module in modules:
        screens.append({
            "screen_number": screen_num,
            "screen_type": "학습 내용",
            "title": module.get("title", f"모듈 {screen_num-2}"),
            "content": {
                "text": "핵심 내용",
                "visual": "내용 시각화",
                "audio": "설명 내레이션",
            },
            "interaction": "다음",
            "duration": "5분",
        })
        screen_num += 1

    # 마무리 화면
    screens.append({
        "screen_number": screen_num,
        "screen_type": "정리",
        "title": "학습 정리",
        "content": {
            "text": "핵심 내용 요약",
            "visual": "요약 인포그래픽",
            "audio": "마무리 멘트",
        },
        "interaction": "완료",
        "duration": "1분",
    })

    return {
        "title": "교육 프로그램 스토리보드",
        "total_screens": len(screens),
        "screens": screens,
        "navigation": {
            "type": "선형",
            "controls": ["이전", "다음", "메뉴"],
            "progress_indicator": "진행률 바",
        },
        "interaction_types": [
            {"type": "클릭", "purpose": "진행", "frequency": "매 화면"},
            {"type": "퀴즈", "purpose": "평가", "frequency": "모듈 후"},
        ],
        "visual_guidelines": {
            "color_scheme": "기업 CI 색상",
            "font": "맑은 고딕 14pt",
            "layout": "표준 레이아웃",
        },
    }
