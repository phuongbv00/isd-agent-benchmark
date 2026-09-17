"""
Analysis 단계 도구

ADDIE의 첫 번째 단계: 학습자, 환경, 과제 분석
LLM을 활용하여 맥락에 맞는 깊이 있는 분석을 수행합니다.
"""

import json
from typing import Optional
from langchain_core.tools import tool

from shared.llm import create_chat_model, get_default_llm_config


def get_llm():
    return create_chat_model(get_default_llm_config())


@tool
def analyze_learners(
    target_audience: str,
    prior_knowledge: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> dict:
    """
    학습자 분석을 수행합니다.

    Args:
        target_audience: 학습 대상자 (예: "신입사원", "초등학교 5학년")
        prior_knowledge: 사전 지식 수준 (선택)
        additional_context: 추가 맥락 정보 (선택)

    Returns:
        학습자 분석 결과 (특성, 동기, 예상 어려움 등)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 다음 학습 대상자에 대한 심층 분석을 수행해주세요.

## 입력 정보
- 학습 대상자: {target_audience}
- 사전 지식 수준: {prior_knowledge or "정보 없음"}
- 추가 맥락: {additional_context or "정보 없음"}

## 분석 항목
1. 학습자 특성: 인지적, 정의적, 사회적 특성
2. 학습 선호도: 선호하는 학습 방식과 매체
3. 동기 수준: 내적/외적 동기 요인
4. 예상 어려움: 학습 과정에서 예상되는 도전 과제

## 요구사항
1. 대상자의 발달 단계와 경험을 고려한 분석
2. 실제 교육 설계에 활용 가능한 구체적인 인사이트 제공
3. 개인차를 고려한 다양한 특성 포함

## 출력 형식 (JSON)
```json
{{
  "target_audience": "신입사원",
  "characteristics": [
    "새로운 환경에 적응 중이며 조직 문화 학습 필요",
    "실무 적용에 대한 높은 관심과 빠른 성장 욕구",
    "이론보다 실제 업무 사례를 통한 학습 선호",
    "동료와의 관계 형성에 대한 욕구"
  ],
  "prior_knowledge": "관련 학과 전공자로 기초 이론은 있으나 실무 경험 부족",
  "learning_preferences": [
    "실습 중심의 체험적 학습",
    "단계별로 구조화된 안내",
    "동료와 함께하는 협력 학습",
    "즉각적인 피드백"
  ],
  "motivation": "높음 - 조직 적응과 역량 개발에 대한 내적 동기 강함",
  "challenges": [
    "정보 과부하로 인한 혼란 가능성",
    "이론과 실무의 갭 인식",
    "조직 내 기대 수준 파악 어려움",
    "자기 효능감 형성 과정 필요"
  ]
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

        result = json.loads(json_match.strip())
        return result

    except Exception as e:
        # 폴백: 템플릿 기반 분석
        return _fallback_analyze_learners(target_audience, prior_knowledge, additional_context)


def _fallback_analyze_learners(
    target_audience: str,
    prior_knowledge: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    audience_lower = target_audience.lower()

    characteristics = []
    learning_preferences = []
    motivation = "보통"
    challenges = []

    if "신입" in audience_lower or "초보" in audience_lower:
        characteristics.extend(["새로운 환경 적응 중", "빠른 성장 욕구", "실무 적용 중시"])
        learning_preferences.extend(["실습 중심", "단계별 안내"])
        motivation = "높음 (성장 욕구)"
        challenges.extend(["기초 개념 부족 가능", "정보 과부하 위험"])

    elif "초등" in audience_lower:
        characteristics.extend(["호기심 왕성", "짧은 집중 시간", "놀이 기반 학습 선호"])
        learning_preferences.extend(["시각적 자료", "게임 기반", "그룹 활동"])
        motivation = "외적 동기 의존"
        challenges.extend(["추상적 개념 이해 어려움", "장시간 집중 어려움"])

    elif "직장인" in audience_lower or "성인" in audience_lower:
        characteristics.extend(["시간 제약", "실무 적용 중시", "자기주도적"])
        learning_preferences.extend(["효율적 학습", "사례 기반", "즉시 적용 가능"])
        motivation = "높음 (업무 성과)"
        challenges.extend(["학습 시간 확보", "기존 습관 변화"])

    elif "대학생" in audience_lower:
        characteristics.extend(["자기주도 학습 가능", "비판적 사고", "다양한 배경"])
        learning_preferences.extend(["토론 중심", "연구 기반", "협업"])
        motivation = "중상 (학점/취업)"
        challenges.extend(["동기 유지", "깊이 있는 학습"])

    else:
        characteristics.extend(["일반 학습자"])
        learning_preferences.extend(["다양한 방식"])
        challenges.extend(["개인차 고려 필요"])

    return {
        "target_audience": target_audience,
        "characteristics": characteristics,
        "prior_knowledge": prior_knowledge or "기초 수준",
        "learning_preferences": learning_preferences,
        "motivation": motivation,
        "challenges": challenges,
    }


@tool
def analyze_context(
    learning_environment: str,
    duration: str,
    class_size: Optional[int] = None,
    budget: Optional[str] = None,
    resources: Optional[list[str]] = None,
) -> dict:
    """
    학습 환경을 분석합니다.

    Args:
        learning_environment: 학습 환경 (예: "온라인", "대면 교실", "블렌디드")
        duration: 학습 시간 (예: "2시간", "1일", "4주")
        class_size: 학습자 수 (선택)
        budget: 예산 수준 (선택)
        resources: 사용 가능한 자원 목록 (선택)

    Returns:
        환경 분석 결과 (제약 조건, 자원, 기술 요구사항)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 다음 학습 환경에 대한 심층 분석을 수행해주세요.

## 입력 정보
- 학습 환경: {learning_environment}
- 학습 시간: {duration}
- 학습자 수: {class_size or "정보 없음"}
- 예산: {budget or "정보 없음"}
- 사용 가능 자원: {json.dumps(resources, ensure_ascii=False) if resources else "정보 없음"}

## 분석 항목
1. 환경 유형: 환경의 특성과 장단점
2. 제약 조건: 시간, 공간, 기술적 제약
3. 사용 가능 자원: 활용 가능한 학습 도구와 자료
4. 기술 요구사항: 필요한 기술 인프라

## 요구사항
1. 환경의 특성을 고려한 실현 가능한 분석
2. 제약 조건을 극복할 수 있는 대안 제시
3. 효과적인 학습을 위한 최적화 방안 포함

## 출력 형식 (JSON)
```json
{{
  "environment": "온라인 실시간 교육",
  "duration": "2시간",
  "constraints": [
    "인터넷 연결 안정성 필요",
    "학습자의 자기 관리 능력 요구",
    "비언어적 커뮤니케이션 제한",
    "기술적 문제 발생 가능성"
  ],
  "resources": [
    "화면 공유 기능",
    "실시간 채팅",
    "녹화 기능",
    "소그룹 세션 (Breakout Room)",
    "온라인 화이트보드"
  ],
  "technical_requirements": [
    "화상회의 플랫폼 (Zoom/Teams/Meet)",
    "안정적인 인터넷 연결 (최소 10Mbps)",
    "웹캠 및 마이크",
    "LMS (학습관리시스템) 접근"
  ]
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

        result = json.loads(json_match.strip())
        return result

    except Exception as e:
        # 폴백
        return _fallback_analyze_context(learning_environment, duration, class_size, budget, resources)


def _fallback_analyze_context(
    learning_environment: str,
    duration: str,
    class_size: Optional[int] = None,
    budget: Optional[str] = None,
    resources: Optional[list[str]] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    env_lower = learning_environment.lower()

    constraints = []
    technical_requirements = []
    available_resources = resources or []

    if "온라인" in env_lower:
        constraints.extend(["인터넷 연결 필요", "자기 관리 필요"])
        technical_requirements.extend(["화상회의 도구", "LMS", "안정적인 인터넷"])
        if not available_resources:
            available_resources = ["화면 공유", "채팅", "녹화"]

    elif "대면" in env_lower or "교실" in env_lower:
        constraints.extend(["물리적 공간 필요", "이동 시간"])
        technical_requirements.extend(["프로젝터/스크린", "화이트보드"])
        if not available_resources:
            available_resources = ["교재", "프레젠테이션", "실습 자료"]

    elif "블렌디드" in env_lower or "하이브리드" in env_lower:
        constraints.extend(["양쪽 환경 조율 필요", "기술 격차"])
        technical_requirements.extend(["화상회의", "LMS", "교실 장비"])
        if not available_resources:
            available_resources = ["온라인 콘텐츠", "대면 자료"]

    if "시간" in duration or "분" in duration:
        constraints.append("단시간 집중 학습")
    elif "일" in duration or "주" in duration:
        constraints.append("장기 학습 계획 필요")

    if class_size:
        if class_size > 30:
            constraints.append("대규모 그룹 관리")
        elif class_size < 10:
            constraints.append("소그룹 상호작용 가능")

    return {
        "environment": learning_environment,
        "duration": duration,
        "constraints": constraints,
        "resources": available_resources,
        "technical_requirements": technical_requirements,
    }


@tool
def analyze_task(
    learning_goals: list[str],
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> dict:
    """
    학습 과제를 분석합니다.

    Args:
        learning_goals: 학습 목표 목록
        domain: 교육 도메인 (예: "IT", "비즈니스", "언어")
        difficulty: 난이도 (예: "easy", "medium", "hard")

    Returns:
        과제 분석 결과 (주요 주제, 세부 주제, 선수 학습)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 다음 학습 목표에 대한 과제 분석을 수행해주세요.

## 입력 정보
- 학습 목표: {json.dumps(learning_goals, ensure_ascii=False)}
- 교육 도메인: {domain or "일반"}
- 난이도: {difficulty or "medium"}

## 분석 항목
1. 주요 주제: 학습 목표에서 도출된 핵심 주제
2. 세부 주제: 각 주요 주제를 세분화한 학습 내용
3. 선수 학습: 본 학습 전에 필요한 사전 지식/기술

## 요구사항
1. 학습 목표를 체계적으로 분해
2. 논리적 학습 순서 고려
3. 실제 학습에서 활용 가능한 구체적 주제

## 출력 형식 (JSON)
```json
{{
  "main_topics": [
    "조직 문화와 가치 이해",
    "업무 프로세스 및 시스템",
    "팀 협업과 커뮤니케이션"
  ],
  "subtopics": [
    "회사 비전과 미션 이해",
    "조직 구조와 각 부서 역할",
    "핵심 업무 절차 파악",
    "사내 시스템 활용법",
    "효과적인 업무 커뮤니케이션",
    "팀 내 협업 방식과 도구"
  ],
  "prerequisites": [
    "기본적인 컴퓨터 활용 능력",
    "비즈니스 기초 용어 이해",
    "자기주도적 학습 태도"
  ]
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

        result = json.loads(json_match.strip())
        return result

    except Exception as e:
        # 폴백
        return _fallback_analyze_task(learning_goals, domain, difficulty)


def _fallback_analyze_task(
    learning_goals: list[str],
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    main_topics = learning_goals.copy()
    subtopics = []
    prerequisites = []

    if domain:
        domain_lower = domain.lower()
        if "it" in domain_lower or "프로그래밍" in domain_lower:
            prerequisites.extend(["컴퓨터 기초", "논리적 사고"])
            subtopics.extend(["기본 개념", "실습", "응용"])

        elif "비즈니스" in domain_lower or "경영" in domain_lower:
            prerequisites.extend(["기초 비즈니스 용어"])
            subtopics.extend(["이론", "사례 분석", "실무 적용"])

        elif "언어" in domain_lower or "영어" in domain_lower:
            prerequisites.extend(["기초 어휘", "기본 문법"])
            subtopics.extend(["듣기", "말하기", "읽기", "쓰기"])

    for goal in learning_goals:
        subtopics.append(f"{goal} - 개념 이해")
        subtopics.append(f"{goal} - 실습/적용")

    if difficulty:
        diff_lower = difficulty.lower()
        if "hard" in diff_lower or "어려" in diff_lower:
            prerequisites.append("관련 분야 기초 지식")
        elif "easy" in diff_lower or "쉬" in diff_lower:
            prerequisites = ["특별한 선수 학습 불필요"]

    return {
        "main_topics": main_topics,
        "subtopics": subtopics[:6],
        "prerequisites": prerequisites or ["기초 지식"],
    }


@tool
def analyze_needs(
    learning_goals: list[str],
    current_state: Optional[str] = None,
    desired_state: Optional[str] = None,
    performance_gap: Optional[str] = None,
) -> dict:
    """
    요구분석을 수행합니다. (Gap Analysis)

    ADDIE 소항목: 1. 문제 확인, 2. 차이 분석, 3. 수행 분석, 4. 우선순위 결정

    Args:
        learning_goals: 학습 목표 목록
        current_state: 현재 상태 설명 (선택)
        desired_state: 목표 상태 설명 (선택)
        performance_gap: 수행 격차 설명 (선택)

    Returns:
        요구분석 결과 (gap_analysis, root_causes, training_needs, non_training_solutions, priority_matrix)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 다음 정보를 바탕으로 체계적인 요구분석(Needs Analysis)을 수행해주세요.

## 입력 정보
- 학습 목표: {json.dumps(learning_goals, ensure_ascii=False)}
- 현재 상태: {current_state or "정보 없음"}
- 목표 상태: {desired_state or "정보 없음"}
- 수행 격차: {performance_gap or "정보 없음"}

## 분석 항목 (ADDIE 소항목 1-4 반영)
1. **gap_analysis**: 현재 상태와 목표 상태의 차이 분석 (3개 이상)
2. **root_causes**: 수행 문제의 근본 원인 (3개 이상)
3. **training_needs**: 교육을 통해 해결 가능한 요구 (3개 이상)
4. **non_training_solutions**: 교육 외 해결책 (2개 이상)
5. **priority_matrix**: 긴급성/중요도 기반 우선순위 매트릭스
6. **recommendation**: 교육적 해결책 적절성 판단 및 권고사항

## 출력 형식 (JSON)
```json
{{
  "gap_analysis": [
    "현재: 기본적인 업무 지식 부족 → 목표: 독립적 업무 수행 가능",
    "현재: 조직 문화 이해 부족 → 목표: 조직 가치와 문화 내재화",
    "현재: 협업 도구 미숙 → 목표: 원활한 협업 도구 활용"
  ],
  "root_causes": [
    "체계적인 온보딩 교육 부재",
    "실무 중심 학습 기회 부족",
    "멘토링 시스템 미흡"
  ],
  "training_needs": [
    "조직 문화 및 비전 교육",
    "업무 프로세스 실습 교육",
    "협업 도구 활용 교육"
  ],
  "non_training_solutions": [
    "멘토링 프로그램 운영",
    "업무 매뉴얼 및 가이드 제공"
  ],
  "priority_matrix": {{
    "high_urgency_high_impact": ["조직 문화 및 비전 교육", "업무 프로세스 실습"],
    "high_urgency_low_impact": ["협업 도구 기초 교육"],
    "low_urgency_high_impact": ["심화 업무 역량 개발"],
    "low_urgency_low_impact": ["선택적 자기계발 과정"]
  }},
  "recommendation": "신입사원의 빠른 적응과 생산성 향상을 위해 체계적인 온보딩 교육이 필수적임. 교육과 함께 멘토링 프로그램을 병행하여 실무 적응을 지원할 것을 권고함."
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

        result = json.loads(json_match.strip())
        return result

    except Exception as e:
        return _fallback_analyze_needs(learning_goals, current_state, desired_state, performance_gap)


def _fallback_analyze_needs(
    learning_goals: list[str],
    current_state: Optional[str] = None,
    desired_state: Optional[str] = None,
    performance_gap: Optional[str] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    gap_analysis = []
    for goal in learning_goals[:3]:
        gap_analysis.append(f"현재: {goal} 관련 역량 부족 → 목표: {goal} 달성")

    if len(gap_analysis) < 3:
        gap_analysis.extend([
            "현재: 기본 지식 부족 → 목표: 핵심 개념 이해",
            "현재: 실무 적용 어려움 → 목표: 실제 상황에 적용 가능",
        ][:3 - len(gap_analysis)])

    root_causes = [
        "체계적인 교육 프로그램 부재",
        "실습 및 적용 기회 부족",
        "피드백 및 코칭 시스템 미흡",
    ]

    training_needs = [f"{goal} 관련 교육" for goal in learning_goals[:3]]
    if len(training_needs) < 3:
        training_needs.extend([
            "기초 역량 강화 교육",
            "실무 적용 워크숍",
        ][:3 - len(training_needs)])

    non_training_solutions = [
        "직무 가이드 및 매뉴얼 제공",
        "멘토링/코칭 프로그램 운영",
    ]

    priority_matrix = {
        "high_urgency_high_impact": training_needs[:2] if len(training_needs) >= 2 else training_needs,
        "high_urgency_low_impact": [training_needs[2]] if len(training_needs) > 2 else [],
        "low_urgency_high_impact": ["심화 역량 개발"],
        "low_urgency_low_impact": ["선택적 자기계발 과정"],
    }

    return {
        "gap_analysis": gap_analysis,
        "root_causes": root_causes,
        "training_needs": training_needs,
        "non_training_solutions": non_training_solutions,
        "priority_matrix": priority_matrix,
        "recommendation": "학습 목표 달성을 위해 체계적인 교육 프로그램이 필요함. 교육과 함께 실무 적용 기회와 지속적인 피드백을 제공할 것을 권고함.",
    }


@tool
def analyze_entry_behavior(
    target_audience: str,
    learning_goals: list[str],
    prior_knowledge: Optional[str] = None,
) -> dict:
    """
    출발점 행동 분석을 수행합니다.

    ADDIE 소항목: 9. 출발점 행동 분석

    Args:
        target_audience: 학습 대상자
        learning_goals: 학습 목표 목록
        prior_knowledge: 사전 지식 수준 (선택)

    Returns:
        출발점 행동 분석 결과 (entry_behaviors, prerequisite_skills, assessment_strategy)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 다음 정보를 바탕으로 출발점 행동 분석을 수행해주세요.

## 입력 정보
- 학습 대상자: {target_audience}
- 학습 목표: {json.dumps(learning_goals, ensure_ascii=False)}
- 사전 지식 수준: {prior_knowledge or "정보 없음"}

## 분석 항목
1. **entry_behaviors**: 학습 시작 시 학습자가 갖추어야 할 행동/능력
2. **prerequisite_skills**: 선수 학습 요건 (필수/권장 구분)
3. **knowledge_gaps**: 예상되는 지식 격차
4. **assessment_strategy**: 출발점 행동 진단 전략

## 출력 형식 (JSON)
```json
{{
  "target_audience": "{target_audience}",
  "entry_behaviors": [
    "기본적인 컴퓨터 조작 능력",
    "비즈니스 문서 작성 경험",
    "기초적인 커뮤니케이션 스킬"
  ],
  "prerequisite_skills": {{
    "required": [
      "기본 컴퓨터 활용 능력",
      "한글/영어 읽기 능력"
    ],
    "recommended": [
      "관련 분야 기초 지식",
      "협업 도구 사용 경험"
    ]
  }},
  "knowledge_gaps": [
    "조직 특화 시스템 사용 경험 부족",
    "실무 적용 사례 이해 부족"
  ],
  "assessment_strategy": {{
    "diagnostic_test": "사전 지식 점검 퀴즈 (10문항)",
    "self_assessment": "역량 자가 체크리스트",
    "interview": "필요시 사전 인터뷰 실시"
  }},
  "remediation_plan": "선수 학습 미달 학습자를 위한 사전 학습 모듈 제공"
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
        return _fallback_analyze_entry_behavior(target_audience, learning_goals, prior_knowledge)


def _fallback_analyze_entry_behavior(
    target_audience: str,
    learning_goals: list[str],
    prior_knowledge: Optional[str] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    audience_lower = target_audience.lower()

    entry_behaviors = ["기본적인 학습 능력", "관련 분야 관심"]
    prerequisite_required = ["기본 읽기/쓰기 능력"]
    prerequisite_recommended = ["관련 분야 기초 지식"]

    if "신입" in audience_lower or "초보" in audience_lower:
        entry_behaviors = [
            "기본적인 컴퓨터 조작 능력",
            "비즈니스 문서 작성 경험",
            "기초적인 커뮤니케이션 스킬",
        ]
        prerequisite_required = ["기본 컴퓨터 활용", "한글/영어 읽기"]
        prerequisite_recommended = ["협업 도구 경험", "프레젠테이션 경험"]

    elif "초등" in audience_lower:
        entry_behaviors = [
            "기본적인 읽기/쓰기 능력",
            "지시 따르기 능력",
            "기초 수 개념 이해",
        ]
        prerequisite_required = ["한글 읽기/쓰기"]
        prerequisite_recommended = ["기초 컴퓨터 사용"]

    elif "직장인" in audience_lower or "성인" in audience_lower:
        entry_behaviors = [
            "업무 관련 기본 지식",
            "자기주도 학습 능력",
            "디지털 도구 활용 능력",
        ]
        prerequisite_required = ["해당 직무 기본 경험"]
        prerequisite_recommended = ["관련 자격증", "심화 학습 경험"]

    knowledge_gaps = [f"{goal} 관련 실무 경험 부족" for goal in learning_goals[:2]]

    return {
        "target_audience": target_audience,
        "entry_behaviors": entry_behaviors,
        "prerequisite_skills": {
            "required": prerequisite_required,
            "recommended": prerequisite_recommended,
        },
        "knowledge_gaps": knowledge_gaps or ["실무 적용 경험 부족"],
        "assessment_strategy": {
            "diagnostic_test": "사전 지식 점검 퀴즈 (10문항)",
            "self_assessment": "역량 자가 체크리스트",
            "interview": "필요시 사전 인터뷰 실시",
        },
        "remediation_plan": "선수 학습 미달 학습자를 위한 사전 학습 모듈 제공",
    }


@tool
def review_task_analysis(
    task_analysis: dict,
    learning_objectives: list[dict],
    target_audience: str,
) -> dict:
    """
    과제분석 결과를 검토하고 정리합니다.

    ADDIE 소항목: 10. 과제분석 검토·정리

    Args:
        task_analysis: 과제분석 결과
        learning_objectives: 학습 목표 목록
        target_audience: 학습 대상자

    Returns:
        검토 결과 (validation_results, alignment_check, refinements, final_task_structure)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 과제분석 결과를 검토하고 정리해주세요.

## 입력 정보
- 과제분석 결과: {json.dumps(task_analysis, ensure_ascii=False)}
- 학습 목표: {json.dumps(learning_objectives, ensure_ascii=False)}
- 학습 대상자: {target_audience}

## 검토 항목
1. **validation_results**: 과제분석 결과의 타당성 검토
2. **alignment_check**: 학습 목표와의 정합성 확인
3. **refinements**: 수정/보완이 필요한 사항
4. **final_task_structure**: 최종 정리된 과제 구조

## 출력 형식 (JSON)
```json
{{
  "review_summary": "과제분석 검토 결과 요약",
  "validation_results": {{
    "completeness": {{
      "status": "충족",
      "details": "모든 주요 주제와 세부 주제가 포함됨"
    }},
    "accuracy": {{
      "status": "충족",
      "details": "과제 분해가 논리적이고 체계적임"
    }},
    "relevance": {{
      "status": "충족",
      "details": "학습 대상자 수준에 적합한 과제 구성"
    }}
  }},
  "alignment_check": [
    {{"objective_id": "OBJ-01", "covered_tasks": ["주제1", "주제2"], "gap": null}},
    {{"objective_id": "OBJ-02", "covered_tasks": ["주제3"], "gap": "세부 실습 과제 보완 필요"}}
  ],
  "refinements": [
    {{"area": "하위 기능 분석", "issue": "일부 세부 과제 누락", "recommendation": "실습 관련 하위 과제 추가"}},
    {{"area": "선수 학습", "issue": "기초 요건 불명확", "recommendation": "구체적인 사전 지식 명시"}}
  ],
  "final_task_structure": {{
    "main_tasks": [
      {{
        "id": "T-01",
        "name": "주요 과제 1",
        "subtasks": ["하위 과제 1.1", "하위 과제 1.2"],
        "prerequisites": ["선수 요건"],
        "complexity": "중",
        "estimated_time": "30분"
      }}
    ],
    "learning_sequence": ["T-01", "T-02", "T-03"],
    "dependencies": [
      {{"from": "T-01", "to": "T-02", "type": "선행 필수"}}
    ]
  }},
  "recommendations": [
    "선수 학습 요건을 명확히 하여 학습 진입 장벽 최소화",
    "복잡한 과제는 단계별로 분해하여 학습 부담 감소",
    "실습 과제와 이론 학습의 균형 유지"
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
        return _fallback_review_task_analysis(task_analysis, learning_objectives, target_audience)


def _fallback_review_task_analysis(
    task_analysis: dict,
    learning_objectives: list[dict],
    target_audience: str,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    main_topics = task_analysis.get("main_topics", [])
    subtopics = task_analysis.get("subtopics", [])

    # 학습 목표와의 정합성 확인
    alignment_check = []
    for obj in learning_objectives[:3]:
        obj_id = obj.get("id", "OBJ-01")
        alignment_check.append({
            "objective_id": obj_id,
            "covered_tasks": main_topics[:2] if main_topics else ["일반 과제"],
            "gap": None,
        })

    # 최종 과제 구조
    final_tasks = []
    for i, topic in enumerate(main_topics[:3]):
        related_subtopics = [st for st in subtopics if topic in st] if subtopics else [f"{topic} 세부 과제"]
        final_tasks.append({
            "id": f"T-{i+1:02d}",
            "name": topic,
            "subtasks": related_subtopics[:2] if related_subtopics else [f"{topic} 하위 과제"],
            "prerequisites": task_analysis.get("prerequisites", ["기초 지식"])[:1],
            "complexity": "중",
            "estimated_time": "30분",
        })

    return {
        "review_summary": "과제분석 결과 검토 완료. 전반적으로 학습 목표와 정합하며, 일부 세부 사항 보완 권고.",
        "validation_results": {
            "completeness": {"status": "충족", "details": "주요 과제가 포함됨"},
            "accuracy": {"status": "충족", "details": "논리적 구조"},
            "relevance": {"status": "충족", "details": "대상자 수준 적합"},
        },
        "alignment_check": alignment_check,
        "refinements": [
            {"area": "세부 과제", "issue": "일부 보완 필요", "recommendation": "실습 과제 추가"},
        ],
        "final_task_structure": {
            "main_tasks": final_tasks,
            "learning_sequence": [f"T-{i+1:02d}" for i in range(len(final_tasks))],
            "dependencies": [],
        },
        "recommendations": [
            "선수 학습 요건 명확화",
            "단계별 과제 분해",
            "이론-실습 균형 유지",
        ],
    }
