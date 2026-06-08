"""
Implementation 단계 도구

ADDIE의 네 번째 단계: 실행 계획 수립
LLM을 활용하여 맥락에 맞는 깊이 있는 실행 계획을 생성합니다.
"""

import json
from typing import Optional
from langchain_core.tools import tool

from shared.llm import create_chat_model, get_default_llm_config


def get_llm():
    return create_chat_model(get_default_llm_config())


@tool
def create_implementation_plan(
    lesson_plan: dict,
    learning_environment: str,
    target_audience: str,
    class_size: Optional[int] = None,
) -> dict:
    """
    실행 계획을 수립합니다.

    Args:
        lesson_plan: 레슨 플랜
        learning_environment: 학습 환경
        target_audience: 학습 대상자
        class_size: 학습자 수 (선택)

    Returns:
        실행 계획 (전달 방식, 가이드, 기술 요구사항, 지원 계획)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 다음 레슨 플랜을 효과적으로 실행하기 위한 상세 계획을 수립해주세요.

## 입력 정보
- 레슨 플랜: {json.dumps(lesson_plan, ensure_ascii=False)}
- 학습 환경: {learning_environment}
- 학습 대상자: {target_audience}
- 학습자 수: {class_size or "정보 없음"}

## 실행 계획 구성 요소
1. 전달 방식 (Delivery Method): 학습 환경에 최적화된 교수 전달 방식
2. 진행자 가이드: 교수자를 위한 상세 진행 안내
3. 학습자 가이드: 학습자를 위한 사전/중/사후 안내
4. 기술 요구사항: 필요한 도구 및 인프라
5. 지원 계획: 학습 지원 및 트러블슈팅 계획

## 요구사항
1. 학습 환경의 특성을 최대한 활용하는 전달 방식
2. 구체적이고 실행 가능한 가이드라인
3. 예상되는 문제 상황에 대한 대비책
4. 학습자 특성을 고려한 맞춤형 지원

## 출력 형식 (JSON)
```json
{{
  "delivery_method": "실시간 온라인 화상 강의 + 비동기 자료 학습",
  "facilitator_guide": "1. 사전 준비\\n   - 화상회의 링크 및 접속 안내 발송 (D-3)\\n   - 학습 자료 LMS 업로드 완료 (D-2)\\n   - 기술 테스트 및 백업 계획 확인 (D-1)\\n\\n2. 진행 시 유의사항\\n   - 시작 5분 전 접속하여 기술 점검\\n   - 학습 목표를 명확히 제시하고 학습자 동기 유발\\n   - 15-20분마다 상호작용 활동 배치\\n   - 채팅창을 통한 실시간 질문 수용\\n   - 소그룹 활동 시 Breakout Room 활용\\n\\n3. 평가 및 피드백\\n   - 각 모듈 종료 시 형성 평가 실시\\n   - 학습자 참여도 모니터링 및 기록\\n   - 개별 피드백은 24시간 내 제공",
  "learner_guide": "1. 학습 전 준비\\n   - 학습 일정 및 목표 확인\\n   - 화상회의 도구 설치 및 테스트\\n   - 인터넷 연결 안정성 확인\\n   - 학습 환경(조용한 공간, 헤드셋) 준비\\n\\n2. 학습 중 활동\\n   - 적극적인 참여 및 질문\\n   - 노트 필기 및 핵심 내용 정리\\n   - 실습 활동 성실히 수행\\n   - 동료와의 협력 활동 참여\\n\\n3. 학습 후 활동\\n   - 핵심 내용 복습 및 정리\\n   - 자기 평가 체크리스트 작성\\n   - 질문 사항 정리 후 Q&A 채널 활용\\n   - 적용 계획 수립",
  "technical_requirements": [
    "화상회의 플랫폼 (Zoom/Teams) - 소그룹 세션 기능 필수",
    "안정적인 인터넷 연결 (최소 10Mbps 권장)",
    "웹캠 및 마이크 (헤드셋 권장)",
    "LMS 접근 권한 및 로그인 정보",
    "공유 문서 도구 (Google Docs/MS Teams)",
    "온라인 화이트보드 (Miro/Jamboard)"
  ],
  "support_plan": "학습 지원 계획:\\n\\n1. 기술 지원\\n   - 접속 문제 시 대체 연락처 제공 (전화/카카오톡)\\n   - 기술 지원 헬프데스크 운영 (교육 시작 30분 전부터)\\n   - 녹화본 제공으로 접속 장애 시 학습 기회 보장\\n\\n2. 학습 지원\\n   - 학습 자료 사전 배포 및 예습 안내\\n   - Q&A 채널 상시 운영 (응답 24시간 이내)\\n   - 개별 멘토링 신청 가능\\n\\n3. 진도 관리\\n   - 주간 학습 진도 모니터링\\n   - 참여율 저조 학습자 개별 연락\\n   - 보충 학습 자료 및 기회 제공"
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
        return _fallback_create_implementation_plan(lesson_plan, learning_environment, target_audience, class_size)


def _fallback_create_implementation_plan(
    lesson_plan: dict,
    learning_environment: str,
    target_audience: str,
    class_size: Optional[int] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    env_lower = learning_environment.lower()
    audience_lower = target_audience.lower()

    # 전달 방식 결정
    if "온라인" in env_lower:
        delivery_method = "실시간 온라인 화상 강의"
        if "자기주도" in env_lower:
            delivery_method = "비동기 온라인 자기주도학습"
    elif "블렌디드" in env_lower:
        delivery_method = "블렌디드 러닝 (온라인 + 대면)"
    else:
        delivery_method = "대면 강의 및 실습"

    # 진행자 가이드
    facilitator_guide_parts = [
        "1. 사전 준비 사항",
        "   - 학습 자료 준비 및 점검",
        "   - 학습 환경 설정",
    ]

    if "온라인" in env_lower:
        facilitator_guide_parts.extend([
            "   - 화상회의 도구 테스트",
            "   - 화면 공유 자료 준비",
        ])
    else:
        facilitator_guide_parts.extend([
            "   - 교실 장비 점검",
            "   - 유인물 인쇄",
        ])

    facilitator_guide_parts.extend([
        "",
        "2. 진행 시 유의사항",
        "   - 학습 목표 명확히 제시",
        "   - 학습자 참여 유도",
        "   - 형성 평가 실시",
    ])

    if "초등" in audience_lower:
        facilitator_guide_parts.append("   - 집중력 유지를 위한 활동 전환")

    facilitator_guide_parts.extend([
        "",
        "3. 평가 및 피드백",
        "   - 학습 활동 관찰",
        "   - 즉각적 피드백 제공",
        "   - 학습 결과 기록",
    ])

    facilitator_guide = "\n".join(facilitator_guide_parts)

    # 학습자 가이드
    learner_guide_parts = [
        "1. 학습 전 준비",
        "   - 학습 환경 점검",
    ]

    if "온라인" in env_lower:
        learner_guide_parts.extend([
            "   - 인터넷 연결 확인",
            "   - 화상회의 도구 접속 테스트",
        ])

    learner_guide_parts.extend([
        "",
        "2. 학습 중 활동",
        "   - 적극적인 참여",
        "   - 질문 및 토론 참여",
        "   - 실습 활동 수행",
        "",
        "3. 학습 후 활동",
        "   - 복습 및 정리",
        "   - 과제 제출 (해당 시)",
        "   - 자기 평가 실시",
    ])

    learner_guide = "\n".join(learner_guide_parts)

    # 기술 요구사항
    technical_requirements = []
    if "온라인" in env_lower:
        technical_requirements.extend([
            "안정적인 인터넷 연결",
            "화상회의 도구 (Zoom/Teams 등)",
            "웹캠 및 마이크",
            "LMS 접근 권한",
        ])
    else:
        technical_requirements.extend([
            "프로젝터/스크린",
            "스피커",
            "화이트보드",
        ])

    # 지원 계획
    support_parts = ["학습 지원 계획:"]

    if class_size and class_size > 20:
        support_parts.append("- 보조 진행자 배치")

    support_parts.extend([
        "- 학습 자료 사전 배포",
        "- 질의응답 채널 운영",
    ])

    if "온라인" in env_lower:
        support_parts.append("- 기술 지원 헬프데스크 안내")

    support_parts.extend([
        "- 학습 진도 모니터링",
        "- 개별 피드백 제공",
    ])

    support_plan = "\n".join(support_parts)

    return {
        "delivery_method": delivery_method,
        "facilitator_guide": facilitator_guide,
        "learner_guide": learner_guide,
        "technical_requirements": technical_requirements,
        "support_plan": support_plan,
    }


@tool
def create_orientation_plan(
    lesson_plan: dict,
    learning_environment: str,
    target_audience: str,
) -> dict:
    """
    교수자 오리엔테이션 계획을 수립합니다.

    ADDIE 소항목: 24. 교수자 오리엔테이션

    Args:
        lesson_plan: 레슨 플랜
        learning_environment: 학습 환경
        target_audience: 학습 대상자

    Returns:
        오리엔테이션 계획 (facilitator_orientation, operator_orientation, rehearsal_plan)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 교수자/운영자 오리엔테이션 계획을 수립해주세요.

## 입력 정보
- 레슨 플랜: {json.dumps(lesson_plan, ensure_ascii=False)}
- 학습 환경: {learning_environment}
- 학습 대상자: {target_audience}

## 오리엔테이션 구성 요소
1. **facilitator_orientation**: 강사/진행자 대상 오리엔테이션
2. **operator_orientation**: 운영자/코디네이터 대상 오리엔테이션
3. **rehearsal_plan**: 리허설 계획

## 출력 형식 (JSON)
```json
{{
  "title": "교수자 오리엔테이션 계획",
  "facilitator_orientation": {{
    "timing": "교육 1주 전",
    "duration": "2시간",
    "agenda": [
      {{"time": "00:00-00:30", "topic": "교육 프로그램 개요 및 학습 목표 설명"}},
      {{"time": "00:30-01:00", "topic": "교수 자료 검토 및 활용법 안내"}},
      {{"time": "01:00-01:30", "topic": "교수 전략 및 진행 방식 협의"}},
      {{"time": "01:30-02:00", "topic": "Q&A 및 역할 확정"}}
    ],
    "materials": ["강사 가이드", "슬라이드", "교수 전략 요약", "FAQ 문서"],
    "key_points": [
      "학습 목표와 기대 성과 명확히 이해",
      "교수 자료 활용법 숙지",
      "학습자 특성 및 예상 질문 파악",
      "평가 방법 및 피드백 방식 확인"
    ]
  }},
  "operator_orientation": {{
    "timing": "교육 3일 전",
    "duration": "1시간",
    "agenda": [
      {{"time": "00:00-00:20", "topic": "운영 역할 및 책임 설명"}},
      {{"time": "00:20-00:40", "topic": "체크리스트 및 운영 절차 안내"}},
      {{"time": "00:40-01:00", "topic": "비상 연락망 공유 및 문제 대응 훈련"}}
    ],
    "materials": ["운영자 매뉴얼", "체크리스트", "비상 연락망"],
    "key_points": [
      "장비 및 환경 점검 절차 숙지",
      "참가자 관리 및 출석 체크 방법",
      "돌발 상황 대응 요령"
    ]
  }},
  "rehearsal_plan": {{
    "timing": "교육 1일 전",
    "duration": "1-2시간",
    "activities": [
      "전체 플로우 점검 (도입-전개-마무리)",
      "장비 및 기술 테스트",
      "시간 배분 확인",
      "역할 분담 최종 확인"
    ],
    "participants": ["강사", "운영자", "기술 지원"],
    "checklist": [
      "프로젝터/화면 공유 정상 작동",
      "음향 시스템 테스트 완료",
      "학습 자료 준비 상태 확인",
      "비상 연락망 테스트"
    ]
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
        return _fallback_create_orientation_plan(lesson_plan, learning_environment, target_audience)


def _fallback_create_orientation_plan(
    lesson_plan: dict,
    learning_environment: str,
    target_audience: str,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    env_lower = learning_environment.lower()
    is_online = "온라인" in env_lower

    if is_online:
        facilitator_materials = ["강사 가이드", "슬라이드", "화상회의 매뉴얼"]
        operator_materials = ["운영자 매뉴얼", "기술 지원 가이드"]
        rehearsal_checklist = ["화상회의 플랫폼 테스트", "화면 공유 테스트", "소그룹 세션 테스트"]
    else:
        facilitator_materials = ["강사 가이드", "슬라이드", "교재"]
        operator_materials = ["운영자 매뉴얼", "체크리스트"]
        rehearsal_checklist = ["프로젝터 테스트", "음향 테스트", "자료 배치 확인"]

    return {
        "title": "교수자 오리엔테이션 계획",
        "facilitator_orientation": {
            "timing": "교육 1주 전",
            "duration": "2시간",
            "agenda": [
                {"time": "00:00-00:30", "topic": "교육 프로그램 개요"},
                {"time": "00:30-01:00", "topic": "교수 자료 검토"},
                {"time": "01:00-01:30", "topic": "교수 전략 협의"},
                {"time": "01:30-02:00", "topic": "Q&A"},
            ],
            "materials": facilitator_materials,
            "key_points": ["학습 목표 이해", "자료 활용법 숙지", "학습자 특성 파악"],
        },
        "operator_orientation": {
            "timing": "교육 3일 전",
            "duration": "1시간",
            "agenda": [
                {"time": "00:00-00:20", "topic": "운영 역할 설명"},
                {"time": "00:20-00:40", "topic": "체크리스트 안내"},
                {"time": "00:40-01:00", "topic": "비상 대응 훈련"},
            ],
            "materials": operator_materials,
            "key_points": ["환경 점검", "참가자 관리", "문제 대응"],
        },
        "rehearsal_plan": {
            "timing": "교육 1일 전",
            "duration": "1-2시간",
            "activities": ["플로우 점검", "장비 테스트", "시간 배분 확인"],
            "participants": ["강사", "운영자"],
            "checklist": rehearsal_checklist,
        },
    }


@tool
def create_system_checklist(
    learning_environment: str,
    technical_requirements: list[str],
    class_size: Optional[int] = None,
) -> dict:
    """
    시스템/환경 점검 체크리스트를 생성합니다.

    ADDIE 소항목: 25. 시스템/환경 점검

    Args:
        learning_environment: 학습 환경
        technical_requirements: 기술 요구사항 목록
        class_size: 학습자 수 (선택)

    Returns:
        시스템 점검 체크리스트 (pre_event, day_of, technical_checklist)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 교육 실행을 위한 시스템/환경 점검 체크리스트를 생성해주세요.

## 입력 정보
- 학습 환경: {learning_environment}
- 기술 요구사항: {json.dumps(technical_requirements, ensure_ascii=False)}
- 학습자 수: {class_size or "미정"}

## 점검 영역
1. **pre_event**: 교육 전 사전 점검 (D-7, D-3, D-1)
2. **day_of**: 당일 점검
3. **technical_checklist**: 기술적 점검 항목
4. **contingency**: 비상 대응 계획

## 출력 형식 (JSON)
```json
{{
  "title": "시스템/환경 점검 체크리스트",
  "pre_event": {{
    "d_minus_7": [
      {{"item": "장비 예약 확인", "responsible": "운영팀", "status": "미완료"}},
      {{"item": "학습 자료 인쇄 의뢰", "responsible": "교수설계팀", "status": "미완료"}},
      {{"item": "참가자 명단 확정", "responsible": "HR팀", "status": "미완료"}}
    ],
    "d_minus_3": [
      {{"item": "강의실/플랫폼 예약 최종 확인", "responsible": "운영팀", "status": "미완료"}},
      {{"item": "장비 테스트 실시", "responsible": "기술팀", "status": "미완료"}},
      {{"item": "참가자 안내 발송", "responsible": "운영팀", "status": "미완료"}}
    ],
    "d_minus_1": [
      {{"item": "자료 배치/업로드 완료", "responsible": "운영팀", "status": "미완료"}},
      {{"item": "최종 리허설 완료", "responsible": "강사/운영팀", "status": "미완료"}},
      {{"item": "비상 연락망 확인", "responsible": "운영팀", "status": "미완료"}}
    ]
  }},
  "day_of": [
    {{"time": "2시간 전", "item": "강의실 오픈/플랫폼 접속", "responsible": "운영팀"}},
    {{"time": "1시간 전", "item": "장비 최종 점검", "responsible": "기술팀"}},
    {{"time": "30분 전", "item": "자료 최종 확인", "responsible": "강사"}},
    {{"time": "15분 전", "item": "참가자 영접 시작", "responsible": "운영팀"}}
  ],
  "technical_checklist": [
    {{"category": "영상/음향", "items": ["프로젝터 정상 작동", "마이크 음량 적정", "스피커 정상 출력"]}},
    {{"category": "네트워크", "items": ["인터넷 연결 안정", "Wi-Fi 접속 정보 게시", "백업 연결 준비"]}},
    {{"category": "소프트웨어", "items": ["프레젠테이션 SW 작동", "화상회의 플랫폼 테스트", "LMS 접속 확인"]}}
  ],
  "contingency": {{
    "power_failure": "비상 전원 위치 확인, 백업 배터리 준비",
    "network_failure": "모바일 핫스팟 준비, 오프라인 자료 백업",
    "equipment_failure": "예비 장비 준비, 기술 지원 연락처 확보",
    "software_issue": "대체 소프트웨어 준비, 수동 진행 가능 자료 준비"
  }},
  "sign_off": {{
    "final_approval": "모든 점검 항목 완료 시 교육 책임자 승인",
    "go_no_go_criteria": "필수 항목 100% 완료, 권장 항목 80% 이상 완료"
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
        return _fallback_create_system_checklist(learning_environment, technical_requirements, class_size)


def _fallback_create_system_checklist(
    learning_environment: str,
    technical_requirements: list[str],
    class_size: Optional[int] = None,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    env_lower = learning_environment.lower()
    is_online = "온라인" in env_lower

    if is_online:
        tech_items = [
            {"category": "네트워크", "items": ["인터넷 연결 안정", "백업 연결 준비"]},
            {"category": "화상회의", "items": ["플랫폼 접속 테스트", "소그룹 세션 테스트"]},
            {"category": "녹화", "items": ["녹화 설정 확인", "저장 공간 확보"]},
        ]
        contingency = {
            "network_failure": "모바일 핫스팟 준비",
            "platform_issue": "대체 플랫폼 준비",
            "audio_issue": "전화 회의 백업",
        }
    else:
        tech_items = [
            {"category": "영상/음향", "items": ["프로젝터 작동", "마이크 테스트"]},
            {"category": "네트워크", "items": ["Wi-Fi 확인", "유선 연결 백업"]},
            {"category": "전원", "items": ["콘센트 확인", "멀티탭 준비"]},
        ]
        contingency = {
            "power_failure": "비상 전원 위치 확인",
            "equipment_failure": "예비 장비 준비",
            "hvac_issue": "환기 및 온도 조절",
        }

    return {
        "title": "시스템/환경 점검 체크리스트",
        "pre_event": {
            "d_minus_7": [
                {"item": "장비 예약", "responsible": "운영팀", "status": "미완료"},
                {"item": "자료 준비", "responsible": "교수설계팀", "status": "미완료"},
            ],
            "d_minus_3": [
                {"item": "장비 테스트", "responsible": "기술팀", "status": "미완료"},
                {"item": "참가자 안내", "responsible": "운영팀", "status": "미완료"},
            ],
            "d_minus_1": [
                {"item": "리허설", "responsible": "강사/운영팀", "status": "미완료"},
                {"item": "최종 점검", "responsible": "운영팀", "status": "미완료"},
            ],
        },
        "day_of": [
            {"time": "1시간 전", "item": "장비 최종 점검", "responsible": "기술팀"},
            {"time": "30분 전", "item": "참가자 영접", "responsible": "운영팀"},
        ],
        "technical_checklist": tech_items,
        "contingency": contingency,
        "sign_off": {
            "final_approval": "교육 책임자 승인",
            "go_no_go_criteria": "필수 항목 100% 완료",
        },
    }


@tool
def create_pilot_plan(
    lesson_plan: dict,
    target_audience: str,
    learning_objectives: list[dict],
) -> dict:
    """
    파일럿(시범 운영) 계획을 수립합니다.

    ADDIE 소항목: 26. 프로토타입 실행

    Args:
        lesson_plan: 레슨 플랜
        target_audience: 학습 대상자
        learning_objectives: 학습 목표 목록

    Returns:
        파일럿 계획 (pilot_scope, participants, success_criteria, data_collection)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 파일럿(시범 운영) 계획을 수립해주세요.

## 입력 정보
- 레슨 플랜: {json.dumps(lesson_plan, ensure_ascii=False)}
- 학습 대상자: {target_audience}
- 학습 목표: {json.dumps(learning_objectives, ensure_ascii=False)}

## 파일럿 계획 구성 요소
1. **pilot_scope**: 파일럿 범위 및 목적
2. **participants**: 참가자 선정 기준
3. **success_criteria**: 성공 기준
4. **data_collection**: 데이터 수집 계획
5. **iteration_plan**: 개선 및 반복 계획

## 출력 형식 (JSON)
```json
{{
  "title": "파일럿 운영 계획",
  "pilot_scope": {{
    "purpose": "본 교육 전 소규모 그룹 대상 시범 운영으로 문제점 파악 및 개선",
    "coverage": "전체 교육과정 (모든 모듈 포함)",
    "duration": "본 교육과 동일한 시간",
    "environment": "실제 교육 환경과 동일하게 구성"
  }},
  "participants": {{
    "sample_size": "10-15명",
    "selection_criteria": [
      "실제 대상 학습자 그룹의 대표성 확보",
      "다양한 경험 수준 포함",
      "피드백 제공 의지가 있는 참가자"
    ],
    "roles": [
      {{"role": "파일럿 학습자", "count": "10-15명", "responsibility": "교육 참여 및 피드백 제공"}},
      {{"role": "관찰자", "count": "2-3명", "responsibility": "진행 상황 관찰 및 기록"}},
      {{"role": "평가자", "count": "1-2명", "responsibility": "평가 실시 및 결과 분석"}}
    ]
  }},
  "success_criteria": [
    {{"metric": "학습 목표 달성률", "target": "80% 이상", "measurement": "사후 평가 점수"}},
    {{"metric": "만족도", "target": "4.0/5.0 이상", "measurement": "만족도 설문"}},
    {{"metric": "완주율", "target": "95% 이상", "measurement": "출석/완료 기록"}},
    {{"metric": "운영 이슈", "target": "심각한 이슈 0건", "measurement": "관찰 기록"}}
  ],
  "data_collection": {{
    "quantitative": [
      "사전-사후 테스트 점수",
      "만족도 설문 결과 (Likert 척도)",
      "소요 시간 기록",
      "참여율/완주율"
    ],
    "qualitative": [
      "참가자 인터뷰 (교육 직후)",
      "관찰자 노트",
      "서술형 피드백",
      "강사 소감"
    ],
    "observation_focus": [
      "학습자 참여도 및 반응",
      "시간 배분 적절성",
      "자료의 명확성",
      "기술적 문제 발생 여부"
    ]
  }},
  "iteration_plan": {{
    "feedback_analysis": "파일럿 종료 후 3일 내 피드백 분석 완료",
    "improvement_priorities": [
      "학습자 이해도가 낮은 영역 보완",
      "시간 초과/부족 구간 조정",
      "기술적 문제 해결",
      "자료 명확성 개선"
    ],
    "revision_timeline": "분석 완료 후 1주 내 수정 완료",
    "second_pilot": "대규모 수정 시 2차 파일럿 고려"
  }},
  "contingency_plan": {{
    "technical_issues": "백업 자료 및 대체 플랫폼 준비",
    "low_attendance": "추가 모집 또는 일정 조정",
    "time_overrun": "선택적 모듈 축소 또는 다음 세션 연계"
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
        return _fallback_create_pilot_plan(lesson_plan, target_audience, learning_objectives)


def _fallback_create_pilot_plan(
    lesson_plan: dict,
    target_audience: str,
    learning_objectives: list[dict],
) -> dict:
    """LLM 실패 시 폴백 함수"""
    return {
        "title": "파일럿 운영 계획",
        "pilot_scope": {
            "purpose": "본 교육 전 시범 운영으로 문제점 파악 및 개선",
            "coverage": "전체 교육과정",
            "duration": lesson_plan.get("total_duration", "2시간"),
            "environment": "실제 교육 환경과 동일",
        },
        "participants": {
            "sample_size": "10-15명",
            "selection_criteria": ["대표성 확보", "다양한 경험 수준", "피드백 의지"],
            "roles": [
                {"role": "파일럿 학습자", "count": "10-15명"},
                {"role": "관찰자", "count": "2-3명"},
            ],
        },
        "success_criteria": [
            {"metric": "학습 목표 달성률", "target": "80% 이상"},
            {"metric": "만족도", "target": "4.0/5.0 이상"},
            {"metric": "완주율", "target": "95% 이상"},
        ],
        "data_collection": {
            "quantitative": ["사전-사후 테스트", "만족도 설문", "참여율"],
            "qualitative": ["참가자 인터뷰", "관찰자 노트", "서술형 피드백"],
            "observation_focus": ["참여도", "시간 배분", "자료 명확성"],
        },
        "iteration_plan": {
            "feedback_analysis": "파일럿 후 3일 내 완료",
            "improvement_priorities": ["이해도 낮은 영역 보완", "시간 조정"],
            "revision_timeline": "1주 내 수정 완료",
        },
        "contingency_plan": {
            "technical_issues": "백업 자료 준비",
            "low_attendance": "일정 조정",
        },
    }


@tool
def create_monitoring_plan(
    lesson_plan: dict,
    implementation_plan: dict,
    learning_environment: str,
) -> dict:
    """
    운영 모니터링 계획을 수립합니다.

    ADDIE 소항목: 27. 운영 모니터링

    Args:
        lesson_plan: 레슨 플랜
        implementation_plan: 실행 계획
        learning_environment: 학습 환경

    Returns:
        모니터링 계획 (real_time_monitoring, quality_metrics, issue_tracking, reporting)
    """
    llm = get_llm()

    prompt = f"""당신은 교수설계 전문가입니다. 교육 운영 모니터링 계획을 수립해주세요.

## 입력 정보
- 레슨 플랜: {json.dumps(lesson_plan, ensure_ascii=False)}
- 실행 계획: {json.dumps(implementation_plan, ensure_ascii=False)}
- 학습 환경: {learning_environment}

## 모니터링 계획 구성 요소
1. **real_time_monitoring**: 실시간 모니터링 항목
2. **quality_metrics**: 품질 지표
3. **issue_tracking**: 이슈 추적 및 대응
4. **reporting**: 보고 체계

## 출력 형식 (JSON)
```json
{{
  "title": "운영 모니터링 계획",
  "real_time_monitoring": {{
    "facilitator_observation": [
      "학습자 참여도 및 집중도",
      "질문 빈도 및 내용",
      "비언어적 반응 (표정, 자세)",
      "활동 참여 수준"
    ],
    "technical_monitoring": [
      "시스템/플랫폼 안정성",
      "음향/영상 품질",
      "자료 표시 정상 여부",
      "네트워크 연결 상태"
    ],
    "time_tracking": [
      "모듈별 소요 시간",
      "활동별 진행 상황",
      "휴식 시간 준수",
      "전체 일정 대비 진행률"
    ],
    "monitoring_tools": [
      "관찰 체크리스트",
      "시간 기록표",
      "이슈 로그"
    ]
  }},
  "quality_metrics": {{
    "engagement_indicators": [
      {{"metric": "질문 수", "target": "모듈당 3개 이상", "tracking": "실시간 기록"}},
      {{"metric": "활동 참여율", "target": "90% 이상", "tracking": "활동 후 확인"}},
      {{"metric": "과제 완료율", "target": "85% 이상", "tracking": "실습 후 확인"}}
    ],
    "delivery_indicators": [
      {{"metric": "시간 준수율", "target": "±5분 이내", "tracking": "모듈 종료 시"}},
      {{"metric": "기술 문제 발생", "target": "0건", "tracking": "실시간"}},
      {{"metric": "자료 오류", "target": "0건", "tracking": "실시간"}}
    ]
  }},
  "issue_tracking": {{
    "severity_levels": [
      {{"level": "Critical", "description": "교육 진행 불가", "response_time": "즉시", "escalation": "교육 책임자"}},
      {{"level": "Major", "description": "품질에 심각한 영향", "response_time": "5분 내", "escalation": "운영팀장"}},
      {{"level": "Minor", "description": "경미한 문제", "response_time": "휴식 시간 내", "escalation": "현장 담당자"}}
    ],
    "issue_log_template": {{
      "fields": ["시간", "카테고리", "설명", "심각도", "담당자", "조치 내용", "해결 상태"]
    }},
    "common_issues_response": [
      {{"issue": "기술적 문제", "response": "백업 자료 활용, 기술팀 호출"}},
      {{"issue": "시간 초과", "response": "선택 활동 축소, 핵심 내용 집중"}},
      {{"issue": "학습자 이탈", "response": "휴식 후 재참여 유도, 개별 지원"}}
    ]
  }},
  "reporting": {{
    "during_session": {{
      "frequency": "모듈 종료 시마다",
      "recipient": "운영팀장",
      "format": "간략 현황 (구두/메시지)"
    }},
    "end_of_session": {{
      "frequency": "교육 종료 직후",
      "recipient": "교육 책임자",
      "format": "운영 요약 보고서",
      "contents": ["참석 현황", "주요 이슈", "특이사항", "개선 제안"]
    }},
    "post_session": {{
      "frequency": "교육 후 24시간 내",
      "recipient": "이해관계자",
      "format": "상세 운영 보고서",
      "contents": ["전체 현황", "품질 지표 달성도", "이슈 및 해결 내역", "참가자 피드백", "개선 권고사항"]
    }}
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
        return _fallback_create_monitoring_plan(lesson_plan, implementation_plan, learning_environment)


def _fallback_create_monitoring_plan(
    lesson_plan: dict,
    implementation_plan: dict,
    learning_environment: str,
) -> dict:
    """LLM 실패 시 폴백 함수"""
    env_lower = learning_environment.lower()
    is_online = "온라인" in env_lower

    if is_online:
        tech_monitoring = ["플랫폼 안정성", "음향/영상 품질", "네트워크 상태", "채팅 활성도"]
    else:
        tech_monitoring = ["음향 품질", "프로젝터 상태", "환경 (온도, 조명)", "자료 가시성"]

    return {
        "title": "운영 모니터링 계획",
        "real_time_monitoring": {
            "facilitator_observation": ["참여도", "질문 빈도", "비언어적 반응", "활동 참여"],
            "technical_monitoring": tech_monitoring,
            "time_tracking": ["모듈별 소요 시간", "진행 상황", "일정 준수"],
            "monitoring_tools": ["관찰 체크리스트", "시간 기록표", "이슈 로그"],
        },
        "quality_metrics": {
            "engagement_indicators": [
                {"metric": "질문 수", "target": "모듈당 3개 이상"},
                {"metric": "활동 참여율", "target": "90% 이상"},
            ],
            "delivery_indicators": [
                {"metric": "시간 준수율", "target": "±5분 이내"},
                {"metric": "기술 문제", "target": "0건"},
            ],
        },
        "issue_tracking": {
            "severity_levels": [
                {"level": "Critical", "response_time": "즉시"},
                {"level": "Major", "response_time": "5분 내"},
                {"level": "Minor", "response_time": "휴식 시간 내"},
            ],
            "issue_log_template": {"fields": ["시간", "설명", "심각도", "조치 내용"]},
        },
        "reporting": {
            "during_session": {"frequency": "모듈 종료 시", "format": "간략 현황"},
            "end_of_session": {"frequency": "교육 종료 직후", "format": "운영 요약"},
            "post_session": {"frequency": "24시간 내", "format": "상세 보고서"},
        },
    }
