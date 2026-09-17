"""
Development 단계 도구

ADDIE의 세 번째 단계: 레슨 플랜 및 학습 자료 개발
LLM을 활용하여 맥락에 맞는 깊이 있는 콘텐츠를 생성합니다.
"""

import json
from typing import Optional
from langchain_core.tools import tool

from shared.llm import create_chat_model, get_default_llm_config


def get_llm():
    return create_chat_model(get_default_llm_config())


@tool
def create_lesson_plan(
    objectives: list[dict],
    instructional_strategy: dict,
    duration: str,
    main_topics: list[str],
) -> dict:
    """
    레슨 플랜을 생성합니다.

    Args:
        objectives: 학습 목표 목록
        instructional_strategy: 교수 전략
        duration: 총 학습 시간
        main_topics: 주요 주제 목록

    Returns:
        레슨 플랜 (총 시간, 모듈별 구성)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 다음 정보를 바탕으로 상세한 레슨 플랜을 생성해주세요.

## 입력 정보
- 학습 목표: {json.dumps(objectives, ensure_ascii=False)}
- 교수 전략: {json.dumps(instructional_strategy, ensure_ascii=False)}
- 총 학습 시간: {duration}
- 주요 주제: {json.dumps(main_topics, ensure_ascii=False)}

## 레슨 플랜 구성 원칙
1. 도입(10%): 주의 집중, 학습 목표 제시, 선수 학습 환기
2. 전개(60%): 핵심 내용 제시, 예시 및 시연, 학습 안내
3. 실습(20%): 연습 활동, 피드백 제공
4. 정리(10%): 요약, 형성 평가, 전이 강화

## 요구사항
1. 각 모듈별로 구체적인 활동과 소요 시간 배분
2. 학습 목표와 활동의 연계성 확보
3. 필요한 자원과 준비물 명시
4. Gagné의 9가지 교수사태 반영

## 출력 형식 (JSON)
```json
{{
  "total_duration": "2시간",
  "modules": [
    {{
      "title": "모듈 1: 조직 문화 이해",
      "duration": "40분",
      "objectives": ["OBJ-01", "OBJ-02"],
      "activities": [
        {{
          "time": "5분",
          "activity": "도입 - 주의 획득",
          "description": "실제 회사 상황 영상을 보여주며 '여러분이 이 상황이라면 어떻게 하시겠습니까?' 질문으로 시작. 학습자의 관심과 호기심 유발",
          "resources": ["동영상 클립", "프레젠테이션"]
        }},
        {{
          "time": "5분",
          "activity": "학습 목표 제시",
          "description": "오늘 학습을 통해 달성할 목표와 기대 성과를 명확히 안내. 학습자가 무엇을 배우고 할 수 있게 되는지 설명",
          "resources": ["슬라이드"]
        }},
        {{
          "time": "20분",
          "activity": "핵심 내용 학습",
          "description": "조직 구조, 부서별 역할, 회사 비전과 미션에 대한 체계적 설명. 실제 사례와 스토리텔링 활용",
          "resources": ["슬라이드", "조직도", "사례 자료"]
        }},
        {{
          "time": "8분",
          "activity": "실습 및 토론",
          "description": "소그룹으로 나누어 주어진 상황에서 조직 문화를 어떻게 적용할지 토론. 발표 및 피드백",
          "resources": ["워크시트", "토론 가이드"]
        }},
        {{
          "time": "2분",
          "activity": "정리 및 요약",
          "description": "핵심 개념 요약 및 다음 모듈 연결. 형성 평가 퀴즈 예고",
          "resources": ["요약 슬라이드"]
        }}
      ]
    }}
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
        return _fallback_create_lesson_plan(objectives, instructional_strategy, duration, main_topics)


def _fallback_create_lesson_plan(
    objectives: list[dict],
    instructional_strategy: dict,
    duration: str,
    main_topics: list[str],
) -> dict:
    """LLM 실패 시 폴백 함수"""
    modules = []

    total_minutes = _parse_duration(duration)
    num_modules = min(len(main_topics), 4)
    module_duration = total_minutes // num_modules if num_modules > 0 else total_minutes

    for i, topic in enumerate(main_topics[:num_modules]):
        related_objectives = [
            obj["id"] for obj in objectives
            if i < len(objectives) and obj == objectives[i]
        ] or [objectives[i % len(objectives)]["id"]] if objectives else []

        activities = []

        intro_time = max(5, module_duration // 10)
        activities.append({
            "time": f"{intro_time}분",
            "activity": "도입",
            "description": f"{topic} 학습 동기 유발 및 목표 제시",
            "resources": ["슬라이드"],
        })

        main_time = module_duration * 6 // 10
        activities.append({
            "time": f"{main_time}분",
            "activity": "핵심 내용 학습",
            "description": f"{topic} 개념 설명 및 예시",
            "resources": ["슬라이드", "예제"],
        })

        practice_time = module_duration * 2 // 10
        activities.append({
            "time": f"{practice_time}분",
            "activity": "실습/활동",
            "description": f"{topic} 관련 실습 수행",
            "resources": ["실습 자료", "워크시트"],
        })

        wrap_time = module_duration - intro_time - main_time - practice_time
        activities.append({
            "time": f"{wrap_time}분",
            "activity": "정리 및 평가",
            "description": "핵심 내용 요약 및 형성 평가",
            "resources": ["퀴즈"],
        })

        modules.append({
            "title": f"모듈 {i+1}: {topic}",
            "duration": f"{module_duration}분",
            "objectives": related_objectives,
            "activities": activities,
        })

    return {
        "total_duration": duration,
        "modules": modules,
    }


@tool
def create_materials(
    lesson_plan: dict,
    learning_environment: str,
    target_audience: str,
) -> list[dict]:
    """
    학습 자료 명세를 생성합니다.

    Args:
        lesson_plan: 레슨 플랜
        learning_environment: 학습 환경
        target_audience: 학습 대상자

    Returns:
        학습 자료 목록
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 다음 레슨 플랜에 맞는 학습 자료 명세를 생성해주세요.

## 입력 정보
- 레슨 플랜: {json.dumps(lesson_plan, ensure_ascii=False)}
- 학습 환경: {learning_environment}
- 학습 대상자: {target_audience}

## 자료 유형
1. 프레젠테이션: 강의용 슬라이드
2. 워크북: 학습자용 학습 자료
3. 영상/온라인 자료: 온라인 환경용 콘텐츠
4. 인쇄물/유인물: 대면 환경용 자료
5. 활동 자료: 실습, 게임 등 활동용 자료
6. 진행자 가이드: 교수자용 상세 가이드

## 요구사항
1. 학습 환경에 적합한 자료 유형 선택
2. 대상자 특성을 고려한 자료 설계
3. 각 자료의 구체적인 내용과 분량 명시
4. 제작 우선순위 및 소요 시간 고려

## 출력 형식 (JSON 배열)
```json
[
  {{
    "type": "프레젠테이션",
    "title": "신입사원 온보딩 과정 강의 슬라이드",
    "description": "조직 문화, 업무 프로세스, 협업 방식을 다루는 강의용 프레젠테이션",
    "slides": 10,
    "duration": null,
    "pages": null,
    "slide_contents": [
      {{
        "slide_number": 1,
        "title": "신입사원 온보딩 교육",
        "bullet_points": [
          "환영합니다! 우리 회사의 새로운 가족이 되신 것을 축하드립니다",
          "오늘 교육 일정: 2시간 소요",
          "교육 목표: 조직 문화 이해 및 업무 프로세스 습득"
        ],
        "speaker_notes": "참가자들을 환영하며 편안한 분위기를 조성합니다. 자기소개 시간을 갖고 교육 목표를 명확히 안내합니다.",
        "visual_suggestion": "회사 로고와 환영 이미지"
      }},
      {{
        "slide_number": 2,
        "title": "오늘의 학습 목표",
        "bullet_points": [
          "회사 조직 구조와 부서별 역할을 설명할 수 있다",
          "사내 시스템 사용법을 시연할 수 있다",
          "회사 문화와 핵심 가치를 이해하고 적용할 수 있다"
        ],
        "speaker_notes": "학습 목표를 명확히 제시하여 참가자들이 무엇을 배울지 기대하도록 합니다.",
        "visual_suggestion": "목표 체크리스트 아이콘"
      }}
    ]
  }},
  {{
    "type": "워크북",
    "title": "신입사원 학습 워크북",
    "description": "자기주도 학습과 실습을 위한 학습자용 워크북",
    "slides": null,
    "duration": null,
    "pages": 20
  }}
]
```

## ⚠️ 중요: 프레젠테이션 자료의 slide_contents 필수
프레젠테이션/슬라이드 자료의 경우 **반드시** slide_contents를 포함해야 합니다:
- 각 슬라이드별 slide_number, title, bullet_points 필수
- bullet_points는 3-5개의 핵심 내용
- speaker_notes에 발표자를 위한 상세 설명 포함

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

        result = json.loads(json_match.strip())
        return result

    except Exception as e:
        # 폴백
        return _fallback_create_materials(lesson_plan, learning_environment, target_audience)


def _fallback_create_materials(
    lesson_plan: dict,
    learning_environment: str,
    target_audience: str,
) -> list[dict]:
    """LLM 실패 시 폴백 함수"""
    materials = []
    env_lower = learning_environment.lower()
    audience_lower = target_audience.lower()

    modules = lesson_plan.get("modules", [])
    num_modules = len(modules)

    slides_per_module = 15 if "초등" in audience_lower else 20

    # 폴백용 기본 슬라이드 콘텐츠 생성
    slide_contents = []
    slide_num = 1

    # 도입 슬라이드
    slide_contents.append({
        "slide_number": slide_num,
        "title": "교육 소개",
        "bullet_points": ["환영합니다", "오늘의 학습 목표", "전체 일정 안내"],
        "speaker_notes": "참가자들을 환영하며 교육 개요를 설명합니다.",
    })
    slide_num += 1

    # 모듈별 슬라이드 생성
    for module in modules:
        module_title = module.get("title", "학습 모듈")

        # 모듈 시작 슬라이드
        slide_contents.append({
            "slide_number": slide_num,
            "title": module_title,
            "bullet_points": [f"{module_title} 학습 목표", "주요 학습 내용", "예상 소요 시간"],
            "speaker_notes": f"{module_title}의 학습 목표와 개요를 설명합니다.",
        })
        slide_num += 1

        # 활동별 슬라이드
        activities = module.get("activities", [])
        for activity in activities[:3]:  # 주요 활동 3개만
            activity_name = activity.get("activity", "학습 활동")
            description = activity.get("description", "")
            slide_contents.append({
                "slide_number": slide_num,
                "title": activity_name,
                "bullet_points": [description] if description else ["활동 설명"],
                "speaker_notes": f"{activity_name} 진행 방법을 안내합니다.",
            })
            slide_num += 1

    # 마무리 슬라이드
    slide_contents.append({
        "slide_number": slide_num,
        "title": "정리 및 Q&A",
        "bullet_points": ["오늘 학습 내용 요약", "핵심 포인트 정리", "질의응답"],
        "speaker_notes": "핵심 내용을 요약하고 질문을 받습니다.",
    })

    materials.append({
        "type": "프레젠테이션",
        "title": "강의 슬라이드",
        "description": "전체 모듈 강의용 슬라이드",
        "slides": len(slide_contents),
        "duration": None,
        "pages": None,
        "slide_contents": slide_contents,
    })

    materials.append({
        "type": "워크북",
        "title": "학습자 워크북",
        "description": "핵심 개념 정리 및 실습 문제",
        "slides": None,
        "duration": None,
        "pages": 5 * num_modules,
    })

    if "온라인" in env_lower:
        materials.append({
            "type": "영상",
            "title": "강의 영상",
            "description": "모듈별 강의 녹화 영상",
            "slides": None,
            "duration": lesson_plan.get("total_duration", "60분"),
            "pages": None,
        })
        materials.append({
            "type": "온라인 자료",
            "title": "LMS 콘텐츠",
            "description": "학습관리시스템 업로드용 자료",
            "slides": None,
            "duration": None,
            "pages": None,
        })
    else:
        materials.append({
            "type": "인쇄물",
            "title": "유인물",
            "description": "핵심 개념 요약 유인물",
            "slides": None,
            "duration": None,
            "pages": 2 * num_modules,
        })

    if "초등" in audience_lower:
        materials.append({
            "type": "활동 자료",
            "title": "게임/활동 키트",
            "description": "놀이 기반 학습 활동 자료",
            "slides": None,
            "duration": None,
            "pages": None,
        })

    materials.append({
        "type": "가이드",
        "title": "진행자 가이드",
        "description": "교수자용 상세 진행 가이드",
        "slides": None,
        "duration": None,
        "pages": 10,
    })

    return materials


def _parse_duration(duration: str) -> int:
    """시간 문자열을 분으로 변환"""
    duration_lower = duration.lower()

    min_match = re.search(r"(\d+)\s*분", duration_lower)
    if min_match:
        return int(min_match.group(1))

    hour_match = re.search(r"(\d+)\s*시간", duration_lower)
    if hour_match:
        hours = int(hour_match.group(1))
        extra_min = re.search(r"(\d+)\s*분", duration_lower)
        return hours * 60 + (int(extra_min.group(1)) if extra_min else 0)

    day_match = re.search(r"(\d+)\s*일", duration_lower)
    if day_match:
        return int(day_match.group(1)) * 8 * 60

    week_match = re.search(r"(\d+)\s*주", duration_lower)
    if week_match:
        return int(week_match.group(1)) * 5 * 8 * 60

    return 60


@tool
def create_facilitator_manual(
    lesson_plan: dict,
    instructional_strategy: dict,
    target_audience: str,
) -> dict:
    """
    교수자용 매뉴얼을 생성합니다.

    ADDIE 소항목: 20. 교수자용 매뉴얼

    Args:
        lesson_plan: 레슨 플랜
        instructional_strategy: 교수 전략
        target_audience: 학습 대상자

    Returns:
        교수자용 매뉴얼 (preparation, delivery_guide, facilitation_tips, troubleshooting)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 교수자(강사/진행자)를 위한 상세 매뉴얼을 생성해주세요.

## 입력 정보
- 레슨 플랜: {json.dumps(lesson_plan, ensure_ascii=False)}
- 교수 전략: {json.dumps(instructional_strategy, ensure_ascii=False)}
- 학습 대상자: {target_audience}

## 매뉴얼 구성 요소
1. **preparation**: 사전 준비 체크리스트
2. **delivery_guide**: 진행 가이드 (시간별 상세 안내)
3. **facilitation_tips**: 진행 팁 및 유의사항
4. **troubleshooting**: 문제 상황 대응
5. **assessment_guide**: 평가 실시 안내

## 출력 형식 (JSON)
```json
{{
  "title": "교수자용 매뉴얼",
  "version": "1.0",
  "preparation": {{
    "d_minus_7": ["교육 자료 최종 검토", "참가자 명단 확인", "사전 과제 발송"],
    "d_minus_3": ["강의실/플랫폼 예약 확인", "장비 테스트 예약", "다과/간식 준비"],
    "d_minus_1": ["자료 인쇄/업로드 완료", "리허설 실시", "비상 연락망 확인"],
    "d_day": ["30분 전 도착", "장비 최종 점검", "참가자 영접 준비"]
  }},
  "delivery_guide": [
    {{
      "time": "0:00-0:05",
      "activity": "오프닝",
      "facilitator_action": "참가자 환영, 자기소개, 교육 목표 안내",
      "key_points": ["편안한 분위기 조성", "학습 목표 명확히 전달"],
      "materials": ["오프닝 슬라이드"],
      "script": "안녕하세요, 여러분. 오늘 교육에 참여해주셔서 감사합니다. 저는 오늘 교육을 진행할 [이름]입니다."
    }}
  ],
  "facilitation_tips": [
    "15-20분마다 상호작용 활동 배치하여 집중력 유지",
    "질문 시 답변까지 5초 이상 기다리기 (사고 시간 제공)",
    "비언어적 신호 주시하여 이해도 파악",
    "예시는 학습자의 업무 상황과 연결",
    "실습 중 각 테이블/소그룹 순회하며 지원"
  ],
  "troubleshooting": [
    {{"situation": "참가자 무응답", "response": "소그룹 토론으로 전환 후 대표 발표 유도"}},
    {{"situation": "시간 초과 우려", "response": "선택적 활동 생략, 핵심 내용 중심 진행"}},
    {{"situation": "기술적 문제", "response": "백업 자료 활용, 기술 지원 요청"}}
  ],
  "assessment_guide": {{
    "formative": "각 모듈 종료 시 이해도 확인 질문 3개씩",
    "summative": "교육 종료 전 종합 퀴즈 실시 (10문항, 15분)",
    "feedback": "즉각적 피드백 제공, 오답에 대한 해설"
  }},
  "post_session": {{
    "immediately_after": ["참가자 질문 응대", "만족도 설문 안내", "추가 자료 공유"],
    "within_24_hours": ["참가자 감사 메일", "추가 질문 응대", "출석/평가 결과 정리"],
    "within_1_week": ["개선 사항 정리", "운영팀 피드백 공유", "자료 업데이트"]
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
        return _fallback_create_facilitator_manual(lesson_plan, instructional_strategy, target_audience)


def _fallback_create_facilitator_manual(
    lesson_plan: dict,
    instructional_strategy: dict,
    target_audience: str,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    modules = lesson_plan.get("modules", [])

    delivery_guide = []
    for i, module in enumerate(modules):
        delivery_guide.append({
            "time": f"모듈 {i+1}",
            "activity": module.get("title", f"학습 모듈 {i+1}"),
            "facilitator_action": "핵심 내용 설명 및 실습 진행",
            "key_points": ["학습 목표 달성 확인", "참가자 참여 유도"],
            "materials": ["슬라이드", "실습 자료"],
        })

    return {
        "title": "교수자용 매뉴얼",
        "version": "1.0",
        "preparation": {
            "d_minus_7": ["교육 자료 검토", "참가자 명단 확인"],
            "d_minus_3": ["장비 테스트", "자료 준비"],
            "d_minus_1": ["리허설 실시", "최종 점검"],
            "d_day": ["30분 전 도착", "장비 점검"],
        },
        "delivery_guide": delivery_guide,
        "facilitation_tips": [
            "15-20분마다 상호작용 활동 배치",
            "질문 후 충분한 사고 시간 제공",
            "학습자 참여 적극 유도",
            "실습 중 순회 지원",
        ],
        "troubleshooting": [
            {"situation": "참가자 무응답", "response": "소그룹 토론으로 전환"},
            {"situation": "시간 초과", "response": "핵심 내용 중심 진행"},
            {"situation": "기술 문제", "response": "백업 자료 활용"},
        ],
        "assessment_guide": {
            "formative": "모듈별 이해도 확인",
            "summative": "종합 평가 실시",
            "feedback": "즉각적 피드백 제공",
        },
        "post_session": {
            "immediately_after": ["질문 응대", "설문 안내"],
            "within_24_hours": ["감사 메일", "결과 정리"],
            "within_1_week": ["개선 사항 정리"],
        },
    }


@tool
def create_operator_manual(
    lesson_plan: dict,
    learning_environment: str,
    class_size: Optional[int] = None,
) -> dict:
    """
    운영자용 매뉴얼을 생성합니다.

    ADDIE 소항목: 21. 운영자용 매뉴얼

    Args:
        lesson_plan: 레슨 플랜
        learning_environment: 학습 환경
        class_size: 학습자 수 (선택)

    Returns:
        운영자용 매뉴얼 (logistics, participant_management, support_procedures)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 교육 운영자(코디네이터)를 위한 상세 매뉴얼을 생성해주세요.

## 입력 정보
- 레슨 플랜: {json.dumps(lesson_plan, ensure_ascii=False)}
- 학습 환경: {learning_environment}
- 학습자 수: {class_size or "미정"}

## 매뉴얼 구성 요소
1. **logistics**: 물리적/기술적 준비 사항
2. **participant_management**: 참가자 관리
3. **support_procedures**: 운영 지원 절차
4. **emergency_procedures**: 비상 상황 대응

## 출력 형식 (JSON)
```json
{{
  "title": "운영자용 매뉴얼",
  "version": "1.0",
  "logistics": {{
    "venue_setup": {{
      "room_arrangement": "U자형 배치 (토론 용이)",
      "equipment_list": ["프로젝터", "스크린", "마이크 2개", "화이트보드"],
      "materials_checklist": ["참가자 명찰", "바인더", "필기구", "간식"]
    }},
    "technical_setup": {{
      "hardware": ["강사용 노트북", "보조 스피커"],
      "software": ["프레젠테이션 소프트웨어", "화상회의 도구 (백업)"],
      "network": ["Wi-Fi 접속 정보 게시", "유선 백업"]
    }},
    "timeline": [
      {{"time": "D-7", "task": "장비 예약 및 자료 인쇄 의뢰"}},
      {{"time": "D-3", "task": "참가 확정자 명단 확정, 명찰 제작"}},
      {{"time": "D-1", "task": "강의실 사전 점검, 자료 배치"}},
      {{"time": "D-Day 1시간 전", "task": "최종 장비 테스트, 다과 세팅"}}
    ]
  }},
  "participant_management": {{
    "registration": {{
      "process": "접수 데스크에서 명찰 배부 및 자료 전달",
      "late_arrival": "별도 안내 후 빈자리 착석 유도",
      "no_show_handling": "교육 시작 15분 후 불참 처리"
    }},
    "attendance_tracking": {{
      "method": "QR 코드 또는 서명",
      "report_to": "교육 담당자 (당일 17시까지)"
    }},
    "special_needs": [
      "휠체어 접근성 확인",
      "특이 식이 요청 확인",
      "통역/자막 필요 시 사전 조율"
    ]
  }},
  "support_procedures": {{
    "during_session": [
      "강의실 뒷자리에서 진행 상황 모니터링",
      "휴식 시간 안내 (5분 전 알림)",
      "다과 보충 및 정리",
      "강사 요청 사항 즉시 지원"
    ],
    "breaks": {{
      "duration": "10분",
      "announcement": "진행자 신호에 따라 안내",
      "activities": ["다과 보충", "화장실 안내", "다음 세션 준비"]
    }}
  }},
  "emergency_procedures": {{
    "fire_evacuation": "비상구 위치 사전 안내, 대피 시 인원 파악",
    "medical_emergency": "응급 연락처 게시, 구급함 위치 숙지",
    "technical_failure": "IT 지원 연락처, 백업 장비 위치"
  }},
  "post_event": {{
    "immediately_after": ["강의실 정리", "장비 반납", "분실물 확인"],
    "within_24_hours": ["설문 결과 수합", "출석 보고서 작성"],
    "within_1_week": ["비용 정산", "운영 보고서 제출"]
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
        return _fallback_create_operator_manual(lesson_plan, learning_environment, class_size)


def _fallback_create_operator_manual(
    lesson_plan: dict,
    learning_environment: str,
    class_size: Optional[int] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    env_lower = learning_environment.lower()
    is_online = "온라인" in env_lower

    if is_online:
        logistics = {
            "venue_setup": {"platform": "화상회의 플랫폼", "backup": "전화 회의"},
            "technical_setup": {"software": ["화상회의", "LMS"], "network": "안정적인 인터넷"},
        }
    else:
        logistics = {
            "venue_setup": {"room_arrangement": "U자형/원탁", "equipment_list": ["프로젝터", "마이크"]},
            "technical_setup": {"hardware": ["노트북", "스피커"], "network": "Wi-Fi"},
        }

    return {
        "title": "운영자용 매뉴얼",
        "version": "1.0",
        "logistics": logistics,
        "participant_management": {
            "registration": {"process": "명찰 배부 및 자료 전달"},
            "attendance_tracking": {"method": "서명/QR"},
            "special_needs": ["접근성 확인", "특이사항 대응"],
        },
        "support_procedures": {
            "during_session": ["진행 모니터링", "휴식 안내", "강사 지원"],
            "breaks": {"duration": "10분", "activities": ["다과 보충", "환기"]},
        },
        "emergency_procedures": {
            "fire_evacuation": "비상구 안내",
            "medical_emergency": "응급 연락처",
            "technical_failure": "IT 지원 요청",
        },
        "post_event": {
            "immediately_after": ["정리", "장비 반납"],
            "within_24_hours": ["출석 보고"],
            "within_1_week": ["운영 보고서"],
        },
    }


@tool
def create_expert_review(
    lesson_plan: dict,
    learning_objectives: list[dict],
    materials: list[dict],
) -> dict:
    """
    전문가 검토를 수행합니다.

    ADDIE 소항목: 23. 전문가 검토

    Args:
        lesson_plan: 레슨 플랜
        learning_objectives: 학습 목표 목록
        materials: 학습 자료 목록

    Returns:
        전문가 검토 결과 (sme_review, id_review, usability_review, recommendations)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 교육 프로그램에 대한 전문가 검토 프레임워크를 생성해주세요.

## 입력 정보
- 레슨 플랜: {json.dumps(lesson_plan, ensure_ascii=False)}
- 학습 목표: {json.dumps(learning_objectives, ensure_ascii=False)}
- 학습 자료: {json.dumps(materials, ensure_ascii=False)}

## 검토 영역
1. **sme_review**: 내용 전문가(SME) 검토 - 내용 정확성, 최신성
2. **id_review**: 교수설계자 검토 - 교수 전략 적절성
3. **usability_review**: 사용성 검토 - 학습자 관점
4. **technical_review**: 기술 검토 - 자료 품질

## 출력 형식 (JSON)
```json
{{
  "title": "전문가 검토 보고서",
  "review_date": "검토 일자",
  "sme_review": {{
    "reviewer_role": "현업 전문가 / 주제 전문가",
    "checklist": [
      {{"item": "내용 정확성", "criteria": "모든 정보가 사실에 기반하고 오류가 없음", "status": "검토 필요"}},
      {{"item": "최신성", "criteria": "최신 규정, 트렌드, 사례가 반영됨", "status": "검토 필요"}},
      {{"item": "실무 적합성", "criteria": "실제 업무 상황에 적용 가능함", "status": "검토 필요"}},
      {{"item": "범위 적절성", "criteria": "학습 목표 달성에 필요한 내용이 포함됨", "status": "검토 필요"}}
    ],
    "feedback_form": "SME 피드백 양식 제공",
    "review_timeline": "3-5 영업일"
  }},
  "id_review": {{
    "reviewer_role": "교수설계 전문가",
    "checklist": [
      {{"item": "목표-내용 정합성", "criteria": "모든 학습 목표에 대응하는 내용이 있음", "status": "검토 필요"}},
      {{"item": "교수 전략 적절성", "criteria": "대상 학습자와 환경에 적합한 전략", "status": "검토 필요"}},
      {{"item": "평가 도구 타당성", "criteria": "평가가 학습 목표를 측정함", "status": "검토 필요"}},
      {{"item": "시간 배분 적절성", "criteria": "각 활동에 충분한 시간이 배정됨", "status": "검토 필요"}}
    ],
    "review_timeline": "3-5 영업일"
  }},
  "usability_review": {{
    "reviewer_role": "대표 학습자 그룹",
    "method": "파일럿 테스트 또는 워크스루",
    "checklist": [
      {{"item": "이해 용이성", "criteria": "내용이 쉽게 이해됨", "status": "테스트 필요"}},
      {{"item": "참여도", "criteria": "활동이 흥미롭고 참여를 유도함", "status": "테스트 필요"}},
      {{"item": "난이도 적절성", "criteria": "너무 쉽거나 어렵지 않음", "status": "테스트 필요"}}
    ],
    "sample_size": "5-10명",
    "review_timeline": "파일럿 교육 후"
  }},
  "technical_review": {{
    "reviewer_role": "기술/미디어 전문가",
    "checklist": [
      {{"item": "자료 품질", "criteria": "시각적 품질, 오디오 품질 양호", "status": "검토 필요"}},
      {{"item": "접근성", "criteria": "장애인 접근성 기준 충족", "status": "검토 필요"}},
      {{"item": "플랫폼 호환", "criteria": "다양한 기기/브라우저에서 작동", "status": "테스트 필요"}}
    ],
    "review_timeline": "2-3 영업일"
  }},
  "recommendations": {{
    "priority_high": [],
    "priority_medium": [],
    "priority_low": []
  }},
  "sign_off_process": {{
    "reviewers": ["SME", "교수설계자", "교육 담당자"],
    "approval_criteria": "모든 필수 검토 항목 통과",
    "documentation": "검토 결과 문서화 및 버전 관리"
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
        return _fallback_create_expert_review(lesson_plan, learning_objectives, materials)


def _fallback_create_expert_review(
    lesson_plan: dict,
    learning_objectives: list[dict],
    materials: list[dict],
) -> dict:
    """LLM 실패 시 폴백 함수"""
    return {
        "title": "전문가 검토 보고서",
        "review_date": "검토 예정",
        "sme_review": {
            "reviewer_role": "현업/주제 전문가",
            "checklist": [
                {"item": "내용 정확성", "criteria": "사실 기반, 오류 없음", "status": "검토 필요"},
                {"item": "최신성", "criteria": "최신 정보 반영", "status": "검토 필요"},
                {"item": "실무 적합성", "criteria": "실제 적용 가능", "status": "검토 필요"},
            ],
            "review_timeline": "3-5일",
        },
        "id_review": {
            "reviewer_role": "교수설계 전문가",
            "checklist": [
                {"item": "목표-내용 정합성", "criteria": "목표와 내용 일치", "status": "검토 필요"},
                {"item": "교수 전략", "criteria": "전략 적절성", "status": "검토 필요"},
                {"item": "평가 타당성", "criteria": "평가 도구 적합", "status": "검토 필요"},
            ],
            "review_timeline": "3-5일",
        },
        "usability_review": {
            "reviewer_role": "대표 학습자",
            "method": "파일럿 테스트",
            "checklist": [
                {"item": "이해 용이성", "status": "테스트 필요"},
                {"item": "참여도", "status": "테스트 필요"},
            ],
            "sample_size": "5-10명",
        },
        "technical_review": {
            "reviewer_role": "기술 전문가",
            "checklist": [
                {"item": "자료 품질", "status": "검토 필요"},
                {"item": "접근성", "status": "검토 필요"},
            ],
        },
        "recommendations": {
            "priority_high": [],
            "priority_medium": [],
            "priority_low": [],
        },
        "sign_off_process": {
            "reviewers": ["SME", "교수설계자", "담당자"],
            "approval_criteria": "필수 항목 통과",
        },
    }
