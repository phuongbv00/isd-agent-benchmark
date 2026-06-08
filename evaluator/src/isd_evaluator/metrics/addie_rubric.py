"""
ADDIE 단계별 루브릭 평가 메트릭

평가 프로세스:
1. 33개 소단계를 각각 개별 평가 (DETAILED_RUBRIC_DEFINITIONS 기준)
2. 평가 결과를 13개 항목으로 통합/집계 (ADDIE_RUBRIC_DEFINITIONS 기준)
"""

import os
import re
import json
from typing import Optional, Dict, List

from openai import OpenAI

from isd_evaluator.models import (
    ADDIEPhase, RubricItem, PhaseScore, ADDIEScore
)
from isd_evaluator.rubrics.addie_definitions import (
    ADDIE_RUBRIC_DEFINITIONS, DEFAULT_PHASE_WEIGHTS, MAX_SCORE_PER_ITEM,
    DETAILED_RUBRIC_DEFINITIONS, ITEM_MAPPING, BENCHMARK_EXAMPLES,
    get_item_ids_for_phase, get_rubric_definition,
    get_detailed_rubric, get_sub_items_for_item, get_phase_sub_items
)




def _build_sub_item_criteria_text(
    sub_item_ids: List[int],
    include_benchmarks: bool = False,
    benchmark_data: Optional[Dict[int, Dict[str, str]]] = None,
    max_example_chars: int = 200,
) -> str:
    """
    33개 소단계 루브릭을 프롬프트용 텍스트로 변환

    Args:
        sub_item_ids: 평가할 소단계 ID 목록
        include_benchmarks: Benchmark Examples 포함 여부
        benchmark_data: {sid: {"question": "...", "answer": "..."}, ...}
        max_example_chars: 예시 텍스트 최대 길이 (토큰 절약)

    Returns:
        프롬프트용 루브릭 텍스트
    """
    lines = []
    for sid in sub_item_ids:
        rubric = get_detailed_rubric(sid)
        if not rubric:
            continue
        name = rubric.get("name", f"항목 {sid}")
        criteria = rubric.get("criteria", {})

        lines.append(f"\n### [{sid}] {name}")

        # Benchmark Examples 추가 (Few-shot)
        if include_benchmarks and benchmark_data and sid in benchmark_data:
            bench = benchmark_data[sid]
            question = bench.get("question", "")
            answer = bench.get("answer", "")

            # 텍스트 길이 제한 (토큰 절약)
            if len(question) > max_example_chars:
                question = question[:max_example_chars] + "..."
            if len(answer) > max_example_chars:
                answer = answer[:max_example_chars] + "..."

            lines.append(f"\n**평가 예시:**")
            lines.append(f"- 질문: {question}")
            lines.append(f"- 우수 답변 요소: {answer}")
            lines.append("")

        # 점수 기준
        lines.append(f"- 9-10점(매우우수): {criteria.get('excellent', '')}")
        lines.append(f"- 7-8점(우수): {criteria.get('good', '')}")
        lines.append(f"- 5-6점(보통): {criteria.get('satisfactory', '')}")
        lines.append(f"- 3-4점(미흡): {criteria.get('poor', '')}")
        lines.append(f"- 1-2점(부재): {criteria.get('absent', '')}")

    return "\n".join(lines)


# 상태별 점수 범위 정의 (연속적 범위)
STATUS_SCORE_RANGES = {
    "absent": (0.0, 0.0),      # 완전 누락
    "weak": (1.0, 3.9),        # 피상적/형식적
    "moderate": (4.0, 6.9),    # 일부 요소 있으나 부족
    "good": (7.0, 8.9),        # 대부분 충족
    "excellent": (9.0, 10.0),  # 완벽
}

# 1단계: 상태 평가 프롬프트 (존재 여부 및 품질 수준 판단)
STATUS_EVALUATION_PROMPT = """당신은 20년 경력의 베테랑 교수설계 전문가입니다.
주어진 ADDIE 산출물에서 각 소단계의 **존재 여부와 품질 수준**을 판단해주세요.

## ⚠️ 핵심 원칙
- **현재 제시된 산출물 내에서만** 해당 내용이 있는지 판단
- 관대한 판단은 금지. 없으면 "absent", 빈약하면 "weak"로 엄격히 분류

## 상태 분류 기준
| 상태 | 기준 | 점수 범위 |
|------|------|-----------|
| **absent** | 해당 요소가 **완전히 존재하지 않음** (언급조차 없음) | 0점 |
| **weak** | 용어만 언급되거나 1-2문장의 피상적 서술 | 1~3.9점 |
| **moderate** | 일부 필수요소는 있으나 구체성 부족, 일반적 서술 | 4~6.9점 |
| **good** | 대부분 필수요소 제시, 약간의 보완만 필요 | 7~8.9점 |
| **excellent** | 모든 필수요소가 구체적으로 제시, 즉시 실행 가능 | 9~10점 |

## 🔍 absent vs weak 판단 기준 (중요!)

**absent (0점):**
- 해당 항목과 관련된 내용이 산출물에 **전혀 없음**
- 관련 용어, 개념, 설명이 단 한 번도 등장하지 않음

**weak (1~3.9점):**
- 관련 용어가 **1회 이상 등장**하거나
- 해당 개념이 **1-2문장으로 간략히 언급**됨
- 구체적인 내용 없이 형식적으로만 존재

**판단 시 주의:**
- 동일한 산출물은 동일한 상태로 판단되어야 함
- 불확실하면 산출물의 텍스트를 다시 확인 후 판단

## 시나리오 컨텍스트
{scenario_context}

## 평가 대상 산출물
{phase_output}

## 평가할 소단계 및 필수요소
{sub_item_criteria}

## 출력 형식 (JSON)
각 소단계 ID에 대해 상태를 판단하세요. reasoning 없이 상태만 출력.
```json
{{
  "sub_status": {{
    {status_keys}
  }}
}}
```
"""

# 2단계: 점수 평가 프롬프트 (상태 범위 내 세부 점수)
SCORE_EVALUATION_PROMPT = """당신은 20년 경력의 베테랑 교수설계 전문가입니다.
각 소단계의 상태가 이미 판단되었습니다. 이제 **상태별 점수 범위 내에서** 세부 점수를 부여하세요.

## ⚠️ 핵심 원칙
- 각 소단계는 **지정된 상태의 점수 범위 내에서만** 점수 부여 가능
- 범위를 벗어난 점수는 절대 불가
- **소수점 첫째 자리까지 다양하게** 점수 부여 (예: 7.2, 7.4, 7.7 등)
- 모든 항목에 동일한 점수를 주지 말 것. **각 항목의 품질 차이를 반영**하여 차등 점수 부여

## 상태별 점수 범위 및 세부 기준
| 상태 | 점수 범위 | 세부 기준 |
|------|-----------|----------|
| absent | **0.0** | 고정값 |
| weak | **1.0 ~ 3.9** | 1.0=용어만 언급, 2.5=1-2문장, 3.5=형식적 틀은 있음 |
| moderate | **4.0 ~ 6.9** | 4.5=핵심 누락, 5.5=일부 있음, 6.5=대부분 있으나 피상적 |
| good | **7.0 ~ 8.9** | 7.2=보완 필요 다수, 8.0=약간 보완, 8.7=거의 완성 |
| excellent | **9.0 ~ 10.0** | 9.2=매우 우수, 9.6=거의 완벽, 10.0=완벽 |

## 시나리오 컨텍스트
{scenario_context}

## 평가 대상 산출물
{phase_output}

## 각 소단계의 판정된 상태
{status_result}

## 평가할 소단계 및 세부 기준
{sub_item_criteria}

## 출력 형식 (JSON)
reasoning 없이 점수만 출력.
```json
{{
  "sub_scores": {{
    {score_keys}
  }}
}}
```
"""

# 단계별 평가 프롬프트 (33개 소단계 개별 평가) - 레거시 호환용
PHASE_EVALUATION_PROMPTS = {
    "analysis": """당신은 20년 경력의 베테랑 교수설계 전문가입니다.
주어진 ADDIE 산출물 중 **Analysis(분석) 단계**의 10개 소단계를 각각 평가해주세요.

## ⚠️ 핵심 평가 원칙 (반드시 준수)
1. 각 소단계를 **0.0 ~ 10.0 점 사이**로 평가 (소수점 첫째 자리)
2. **해당 소단계 내용이 완전히 누락된 경우 → 0점 부여** (예외 없음)
3. **피상적/형식적 언급만 있는 경우 → 1-3점 부여**
4. 관대한 점수 부여는 교육 품질 저하로 이어짐을 인식

## 점수 기준 (엄격 적용)
| 점수 | 기준 |
|------|------|
| **0점** | 해당 요소가 **완전히 존재하지 않음** (섹션 자체가 없음) |
| **1-2점** | 용어만 언급, 실질적 내용 없음 |
| **3-4점** | 일부 요소 있으나 대부분 누락, 형식적/피상적 수준 |
| **5-6점** | 필수요소 언급되나 구체성 부족, 일반적 서술 |
| **7-8점** | 대부분 필수요소 제시, 약간의 보완 필요 |
| **9-10점** | 모든 필수요소가 구체적으로 제시, 즉시 실행 가능 |

## 시나리오 컨텍스트
{scenario_context}

## 평가 대상 산출물
{phase_output}

## 평가할 10개 소단계 및 세부 기준
{sub_item_criteria}

## 출력 형식 (JSON)
```json
{{
  "sub_scores": {{
    "1": <0.0-10.0>,
    "2": <0.0-10.0>,
    "3": <0.0-10.0>,
    "4": <0.0-10.0>,
    "5": <0.0-10.0>,
    "6": <0.0-10.0>,
    "7": <0.0-10.0>,
    "8": <0.0-10.0>,
    "9": <0.0-10.0>,
    "10": <0.0-10.0>
  }},
  "sub_reasoning": {{
    "1": "<문제 확인 및 정의 평가>",
    "2": "<차이분석 평가>",
    "3": "<수행분석 평가>",
    "4": "<요구 우선순위 결정 평가>",
    "5": "<학습자 분석 평가>",
    "6": "<환경 분석 평가>",
    "7": "<초기 학습목표 분석 평가>",
    "8": "<하위 기능 분석 평가>",
    "9": "<출발점 행동 분석 평가>",
    "10": "<과제분석 결과 검토·정리 평가>"
  }},
  "present_elements": ["<존재하는 필수요소들>"],
  "missing_elements": ["<누락된 필수요소들>"],
  "phase_assessment": "<Analysis 단계 종합 평가>"
}}
```
""",

    "design": """당신은 20년 경력의 베테랑 교수설계 전문가입니다.
주어진 ADDIE 산출물 중 **Design(설계) 단계**의 8개 소단계를 각각 평가해주세요.

## ⚠️ 핵심 평가 원칙 (반드시 준수)
1. 각 소단계를 **0.0 ~ 10.0 점 사이**로 평가 (소수점 첫째 자리)
2. **해당 소단계 내용이 완전히 누락된 경우 → 0점 부여** (예외 없음)
3. **피상적/형식적 언급만 있는 경우 → 1-3점 부여**
4. Analysis 결과와의 **논리적 연결**을 검증

## 점수 기준 (엄격 적용)
| 점수 | 기준 |
|------|------|
| **0점** | 해당 요소가 **완전히 존재하지 않음** (섹션 자체가 없음) |
| **1-2점** | 용어만 언급, 실질적 내용 없음 |
| **3-4점** | 일부 요소 있으나 대부분 누락, 형식적/피상적 수준 |
| **5-6점** | 필수요소 언급되나 구체성 부족, 일반적 서술 |
| **7-8점** | 대부분 필수요소 제시, 약간의 보완 필요 |
| **9-10점** | 모든 필수요소가 구체적으로 제시, 즉시 실행 가능 |

## 시나리오 컨텍스트
{scenario_context}

## 평가 대상 산출물
{phase_output}

## 평가할 8개 소단계 및 세부 기준
{sub_item_criteria}

## 출력 형식 (JSON)
```json
{{
  "sub_scores": {{
    "11": <0.0-10.0>,
    "12": <0.0-10.0>,
    "13": <0.0-10.0>,
    "14": <0.0-10.0>,
    "15": <0.0-10.0>,
    "16": <0.0-10.0>,
    "17": <0.0-10.0>,
    "18": <0.0-10.0>
  }},
  "sub_reasoning": {{
    "11": "<학습목표 정교화 평가>",
    "12": "<평가 계획 수립 평가>",
    "13": "<교수 내용 선정 평가>",
    "14": "<교수적 전략 수립 평가>",
    "15": "<비교수적 전략 수립 평가>",
    "16": "<매체 선정과 활용 계획 평가>",
    "17": "<학습활동 및 시간 구조화 평가>",
    "18": "<스토리보드/화면 흐름 설계 평가>"
  }},
  "present_elements": ["<존재하는 필수요소들>"],
  "missing_elements": ["<누락된 필수요소들>"],
  "phase_assessment": "<Design 단계 종합 평가>"
}}
```
""",

    "development": """당신은 20년 경력의 베테랑 교수설계 전문가입니다.
주어진 ADDIE 산출물 중 **Development(개발) 단계**의 5개 소단계를 각각 평가해주세요.

## ⚠️ 핵심 평가 원칙 (반드시 준수)
1. 각 소단계를 **0.0 ~ 10.0 점 사이**로 평가 (소수점 첫째 자리)
2. **해당 소단계 내용이 완전히 누락된 경우 → 0점 부여** (예외 없음)
3. **피상적/형식적 언급만 있는 경우 → 1-3점 부여**
4. Design 단계와의 **일관성**을 검증

## 점수 기준 (엄격 적용)
| 점수 | 기준 |
|------|------|
| **0점** | 해당 요소가 **완전히 존재하지 않음** (섹션 자체가 없음) |
| **1-2점** | 용어만 언급, 실질적 내용 없음 |
| **3-4점** | 일부 요소 있으나 대부분 누락, 형식적/피상적 수준 |
| **5-6점** | 필수요소 언급되나 구체성 부족, 일반적 서술 |
| **7-8점** | 대부분 필수요소 제시, 약간의 보완 필요 |
| **9-10점** | 모든 필수요소가 구체적으로 제시, 즉시 실행 가능 |

## 시나리오 컨텍스트
{scenario_context}

## 평가 대상 산출물
{phase_output}

## 평가할 5개 소단계 및 세부 기준
{sub_item_criteria}

## 출력 형식 (JSON)
```json
{{
  "sub_scores": {{
    "19": <0.0-10.0>,
    "20": <0.0-10.0>,
    "21": <0.0-10.0>,
    "22": <0.0-10.0>,
    "23": <0.0-10.0>
  }},
  "sub_reasoning": {{
    "19": "<학습자용 자료 개발 평가>",
    "20": "<교수자용 매뉴얼 개발 평가>",
    "21": "<운영자용 매뉴얼 개발 평가>",
    "22": "<평가 도구·문항 개발 평가>",
    "23": "<전문가 검토 평가>"
  }},
  "present_elements": ["<존재하는 필수요소들>"],
  "missing_elements": ["<누락된 필수요소들>"],
  "phase_assessment": "<Development 단계 종합 평가>"
}}
```
""",

    "implementation": """당신은 20년 경력의 베테랑 교수설계 전문가입니다.
주어진 ADDIE 산출물 중 **Implementation(실행) 단계**의 4개 소단계를 각각 평가해주세요.

## ⚠️ 핵심 평가 원칙 (반드시 준수)
1. 각 소단계를 **0.0 ~ 10.0 점 사이**로 평가 (소수점 첫째 자리)
2. **해당 소단계 내용이 완전히 누락된 경우 → 0점 부여** (예외 없음)
3. **피상적/형식적 언급만 있는 경우 → 1-3점 부여**
4. **현실적 실행 가능성**을 중점 검증

## 점수 기준 (엄격 적용)
| 점수 | 기준 |
|------|------|
| **0점** | 해당 요소가 **완전히 존재하지 않음** (섹션 자체가 없음) |
| **1-2점** | 용어만 언급, 실질적 내용 없음 |
| **3-4점** | 일부 요소 있으나 대부분 누락, 형식적/피상적 수준 |
| **5-6점** | 필수요소 언급되나 구체성 부족, 일반적 서술 |
| **7-8점** | 대부분 필수요소 제시, 약간의 보완 필요 |
| **9-10점** | 모든 필수요소가 구체적으로 제시, 즉시 실행 가능 |

## 시나리오 컨텍스트
{scenario_context}

## 평가 대상 산출물
{phase_output}

## 평가할 4개 소단계 및 세부 기준
{sub_item_criteria}

## 출력 형식 (JSON)
```json
{{
  "sub_scores": {{
    "24": <0.0-10.0>,
    "25": <0.0-10.0>,
    "26": <0.0-10.0>,
    "27": <0.0-10.0>
  }},
  "sub_reasoning": {{
    "24": "<교수자·운영자 오리엔테이션 평가>",
    "25": "<시스템/환경 점검 평가>",
    "26": "<프로토타입 실행 평가>",
    "27": "<운영 모니터링 및 지원 평가>"
  }},
  "present_elements": ["<존재하는 필수요소들>"],
  "missing_elements": ["<누락된 필수요소들>"],
  "phase_assessment": "<Implementation 단계 종합 평가>"
}}
```
""",

    "evaluation": """당신은 20년 경력의 베테랑 교수설계 전문가입니다.
주어진 ADDIE 산출물 중 **Evaluation(평가) 단계**의 6개 소단계를 각각 평가해주세요.

## ⚠️ 핵심 평가 원칙 (반드시 준수)
1. 각 소단계를 **0.0 ~ 10.0 점 사이**로 평가 (소수점 첫째 자리)
2. **해당 소단계 내용이 완전히 누락된 경우 → 0점 부여** (예외 없음)
3. **피상적/형식적 언급만 있는 경우 → 1-3점 부여**
4. Design 단계 학습목표와의 **정렬성**을 검증

## 점수 기준 (엄격 적용)
| 점수 | 기준 |
|------|------|
| **0점** | 해당 요소가 **완전히 존재하지 않음** (섹션 자체가 없음) |
| **1-2점** | 용어만 언급, 실질적 내용 없음 |
| **3-4점** | 일부 요소 있으나 대부분 누락, 형식적/피상적 수준 |
| **5-6점** | 필수요소 언급되나 구체성 부족, 일반적 서술 |
| **7-8점** | 대부분 필수요소 제시, 약간의 보완 필요 |
| **9-10점** | 모든 필수요소가 구체적으로 제시, 즉시 실행 가능 |

## 시나리오 컨텍스트
{scenario_context}

## 평가 대상 산출물
{phase_output}

## 평가할 6개 소단계 및 세부 기준
{sub_item_criteria}

## 출력 형식 (JSON)
```json
{{
  "sub_scores": {{
    "28": <0.0-10.0>,
    "29": <0.0-10.0>,
    "30": <0.0-10.0>,
    "31": <0.0-10.0>,
    "32": <0.0-10.0>,
    "33": <0.0-10.0>
  }},
  "sub_reasoning": {{
    "28": "<파일럿/초기 실행 중 자료 수집 평가>",
    "29": "<형성평가 결과 기반 1차 프로그램 개선 평가>",
    "30": "<총괄 평가 문항 개발 평가>",
    "31": "<총괄평가 시행 및 프로그램 효과 분석 평가>",
    "32": "<프로그램 채택 여부 결정 평가>",
    "33": "<프로그램 개선 평가>"
  }},
  "present_elements": ["<존재하는 필수요소들>"],
  "missing_elements": ["<누락된 필수요소들>"],
  "phase_assessment": "<Evaluation 단계 종합 평가>"
}}
```
"""
}


class ADDIERubricEvaluator:
    """
    ADDIE 단계별 루브릭 평가기

    평가 프로세스:
    1. 33개 소단계를 각각 개별 평가
    2. 평가 결과를 13개 항목으로 통합/집계 (평균)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        phase_weights: Optional[Dict[ADDIEPhase, float]] = None,
        include_benchmarks: bool = True,  # Benchmark Examples 포함 여부 (기본 ON)
        temperature: float = 0.0,  # LLM temperature (0.0=결정적, 0.7=일반적)
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
    ):
        self.provider = provider or os.getenv("ADDIE_EVAL_PROVIDER") or os.getenv("JUDGE_MODEL_PROVIDER", "upstage")
        self.phase_weights = phase_weights or DEFAULT_PHASE_WEIGHTS.copy()
        self.temperature = temperature

        # API client configuration. Judges may override base_url/key via JUDGE_MODEL_*.
        OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
        direct_api_key = api_key or os.getenv("ADDIE_EVAL_API_KEY") or os.getenv("JUDGE_MODEL_API_KEY")
        api_key_env = api_key_env or os.getenv("ADDIE_EVAL_API_KEY_ENV") or os.getenv("JUDGE_MODEL_API_KEY_ENV")
        base_url = base_url or os.getenv("ADDIE_EVAL_BASE_URL") or os.getenv("JUDGE_MODEL_BASE_URL")

        if self.provider == "upstage":
            self.model = model or os.getenv("ADDIE_EVAL_MODEL", "solar-pro3")
            api_key_env = api_key_env or "UPSTAGE_API_KEY"
            base_url = base_url or "https://api.upstage.ai/v1/solar"
        elif self.provider == "google":
            self.model = model or os.getenv("ADDIE_EVAL_MODEL", "google/gemini-3-pro-preview")
            api_key_env = api_key_env or "OPENROUTER_API_KEY"
            base_url = base_url or OPENROUTER_BASE_URL
        elif self.provider == "deepseek":
            self.model = model or os.getenv("ADDIE_EVAL_MODEL", "deepseek/deepseek-v3.2")
            api_key_env = api_key_env or "OPENROUTER_API_KEY"
            base_url = base_url or OPENROUTER_BASE_URL
        elif self.provider == "anthropic":
            self.model = model or os.getenv("ADDIE_EVAL_MODEL", "anthropic/claude-opus-4.5")
            api_key_env = api_key_env or "OPENROUTER_API_KEY"
            base_url = base_url or OPENROUTER_BASE_URL
        else:  # openai/openrouter/local/custom
            self.model = model or os.getenv("ADDIE_EVAL_MODEL", "openai/gpt-5.2")
            api_key_env = api_key_env or "OPENROUTER_API_KEY"
            base_url = base_url or OPENROUTER_BASE_URL

        resolved_api_key = direct_api_key or os.getenv(api_key_env or "")
        if not resolved_api_key and base_url:
            from urllib.parse import urlparse
            host = urlparse(base_url).hostname or ""
            if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
                resolved_api_key = "not-needed"

        client_kwargs = {"api_key": resolved_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        # Benchmark Examples 설정 (Few-shot 프롬프트용)
        self.include_benchmarks = include_benchmarks
        self._benchmark_data: Optional[Dict[int, Dict[str, str]]] = None
        if include_benchmarks:
            self._benchmark_data = BENCHMARK_EXAMPLES
            print(f"[ADDIERubricEvaluator] Benchmark Examples 활성화: {len(self._benchmark_data)}개 항목")

    def evaluate(
        self,
        addie_output: dict,
        scenario: Optional[dict] = None,
    ) -> ADDIEScore:
        """
        ADDIE 산출물을 투-스텝으로 평가합니다.
        1단계: 각 소단계의 상태(존재 여부/품질 수준) 판단
        2단계: 상태별 점수 범위 내에서 세부 점수 부여

        Args:
            addie_output: ADDIE 산출물 딕셔너리
            scenario: 원본 시나리오 (선택)

        Returns:
            ADDIEScore: 평가 점수
        """
        scenario_context = self._format_scenario(scenario) if scenario else "제공되지 않음"

        # 단계별 산출물 추출
        phase_outputs = self._extract_phase_outputs(addie_output)

        # 33개 소단계별 점수 저장
        all_sub_scores = {}  # {1: 7.5, 2: 8.0, ...}
        all_sub_reasoning = {}  # {1: "...", 2: "...", ...}
        all_sub_status = {}  # {1: "good", 2: "absent", ...}
        all_missing = []
        all_weak = []

        # 5개 단계별로 개별 평가 (각 단계의 소단계들을 평가)
        phase_configs = [
            ("analysis", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            ("design", [11, 12, 13, 14, 15, 16, 17, 18]),
            ("development", [19, 20, 21, 22, 23]),
            ("implementation", [24, 25, 26, 27]),
            ("evaluation", [28, 29, 30, 31, 32, 33]),
        ]

        for phase_name, sub_item_ids in phase_configs:
            phase_output = phase_outputs.get(phase_name, {})
            phase_output_text = json.dumps(phase_output, ensure_ascii=False, indent=2)

            # 해당 단계의 소단계 루브릭 텍스트 생성
            sub_item_criteria = _build_sub_item_criteria_text(
                sub_item_ids,
                include_benchmarks=self.include_benchmarks,
                benchmark_data=self._benchmark_data,
                max_example_chars=200,
            )

            # === 1단계: 상태 평가 ===
            status_keys = ", ".join([f'"{sid}": "<absent|weak|moderate|good|excellent>"' for sid in sub_item_ids])
            reasoning_keys = ", ".join([f'"{sid}": "<판단 근거>"' for sid in sub_item_ids])

            status_prompt = STATUS_EVALUATION_PROMPT.format(
                scenario_context=scenario_context,
                phase_output=phase_output_text,
                sub_item_criteria=sub_item_criteria,
                status_keys=status_keys,
                reasoning_keys=reasoning_keys,
            )

            status_result = self._call_llm(status_prompt)
            sub_status, status_reasoning = self._parse_status_result(status_result, sub_item_ids)
            all_sub_status.update(sub_status)

            # === 2단계: 점수 평가 (상태 범위 내) ===
            status_result_text = json.dumps(
                {str(sid): sub_status.get(sid, "moderate") for sid in sub_item_ids},
                ensure_ascii=False, indent=2
            )
            score_keys = ", ".join([
                f'"{sid}": <{STATUS_SCORE_RANGES.get(sub_status.get(sid, "moderate"), (4, 6))[0]}-{STATUS_SCORE_RANGES.get(sub_status.get(sid, "moderate"), (4, 6))[1]}>'
                for sid in sub_item_ids
            ])

            score_prompt = SCORE_EVALUATION_PROMPT.format(
                scenario_context=scenario_context,
                phase_output=phase_output_text,
                status_result=status_result_text,
                sub_item_criteria=sub_item_criteria,
                score_keys=score_keys,
                reasoning_keys=reasoning_keys,
            )

            score_result = self._call_llm(score_prompt)
            sub_scores, sub_reasoning, missing, weak = self._parse_score_result(
                score_result, sub_item_ids, sub_status
            )

            all_sub_scores.update(sub_scores)
            all_sub_reasoning.update(sub_reasoning)
            all_missing.extend(missing)
            all_weak.extend(weak)

        # 33개 → 13개 통합 및 최종 점수 계산
        return self._build_final_score_from_sub_items(
            all_sub_scores, all_sub_reasoning, all_missing, all_weak
        )

    def _extract_phase_outputs(self, addie_output: dict) -> dict:
        """ADDIE 산출물에서 단계별 내용 추출"""
        phase_mapping = {
            "analysis": ["analysis", "Analysis", "분석", "learner_analysis", "context_analysis"],
            "design": ["design", "Design", "설계", "learning_objectives", "assessment_design"],
            "development": ["development", "Development", "개발", "content", "materials"],
            "implementation": ["implementation", "Implementation", "실행", "delivery_plan", "instructor_guide"],
            "evaluation": ["evaluation", "Evaluation", "평가", "assessment", "improvement_plan"],
        }

        result = {}
        for phase, keys in phase_mapping.items():
            phase_content = {}
            for key in keys:
                if key in addie_output:
                    if isinstance(addie_output[key], dict):
                        phase_content.update(addie_output[key])
                    else:
                        phase_content[key] = addie_output[key]

            for k, v in addie_output.items():
                k_lower = k.lower()
                if any(pk in k_lower for pk in phase_mapping[phase]):
                    if isinstance(v, dict):
                        phase_content.update(v)
                    else:
                        phase_content[k] = v

            result[phase] = phase_content if phase_content else addie_output

        return result

    def _call_llm(self, prompt: str) -> str:
        """LLM API 호출 (nothink mode: reasoning 비활성화)"""
        api_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.model.startswith("gpt-4") or self.model.startswith("gpt-5") or "gpt-4" in self.model or "gpt-5" in self.model:
            api_params["max_completion_tokens"] = 3000
        else:
            api_params["max_tokens"] = 3000

        # OpenRouter nothink mode: reasoning 비활성화 (속도 향상)
        if self.provider != "upstage":
            api_params["extra_body"] = {"reasoning": {"enabled": False}}

        response = self.client.chat.completions.create(**api_params)
        return response.choices[0].message.content

    def _parse_sub_item_result(
        self, content: str, sub_item_ids: List[int]
    ) -> tuple:
        """33개 소단계별 LLM 응답 파싱"""
        default_score = 5.0
        sub_scores = {}
        sub_reasoning = {}
        missing = []
        weak = []

        try:
            # JSON 블록 추출
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
                json_str = json_str.replace("}}", "}")
                json_str = re.sub(r'<0\.0-10\.0[|>]?', '5.0', json_str)
            else:
                json_match = re.search(r'\{[\s\S]*\}', content)
                json_str = json_match.group(0) if json_match else None

            if json_str:
                data = json.loads(json_str)
                scores_data = data.get("sub_scores", {})
                reasoning_data = data.get("sub_reasoning", {})

                for sid in sub_item_ids:
                    sid_str = str(sid)
                    score_val = scores_data.get(sid_str, scores_data.get(sid, default_score))
                    sub_scores[sid] = min(10.0, max(0.0, float(score_val)))
                    sub_reasoning[sid] = reasoning_data.get(sid_str, reasoning_data.get(str(sid), ""))

                missing = data.get("missing_elements", [])
                weak = data.get("weak_areas", [])
            else:
                for sid in sub_item_ids:
                    sub_scores[sid] = default_score
                    sub_reasoning[sid] = "파싱 실패"

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError) as e:
            print(f"[ADDIERubricEvaluator] 소단계별 파싱 오류: {e}")
            for sid in sub_item_ids:
                sub_scores[sid] = default_score
                sub_reasoning[sid] = f"파싱 오류: {e}"

        return sub_scores, sub_reasoning, missing, weak

    def _parse_status_result(
        self, content: str, sub_item_ids: List[int]
    ) -> tuple:
        """1단계: 상태 평가 결과 파싱"""
        default_status = "moderate"
        sub_status = {}
        status_reasoning = {}

        try:
            # JSON 블록 추출
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{[\s\S]*\}', content)
                json_str = json_match.group(0) if json_match else None

            if json_str:
                data = json.loads(json_str)
                status_data = data.get("sub_status", {})
                reasoning_data = data.get("status_reasoning", {})

                valid_statuses = {"absent", "weak", "moderate", "good", "excellent"}

                for sid in sub_item_ids:
                    sid_str = str(sid)
                    status_val = status_data.get(sid_str, status_data.get(sid, default_status))
                    # 유효한 상태값인지 확인
                    if status_val.lower() in valid_statuses:
                        sub_status[sid] = status_val.lower()
                    else:
                        sub_status[sid] = default_status
                    status_reasoning[sid] = reasoning_data.get(sid_str, reasoning_data.get(str(sid), ""))
            else:
                for sid in sub_item_ids:
                    sub_status[sid] = default_status
                    status_reasoning[sid] = "파싱 실패"

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError) as e:
            print(f"[ADDIERubricEvaluator] 상태 파싱 오류: {e}")
            for sid in sub_item_ids:
                sub_status[sid] = default_status
                status_reasoning[sid] = f"파싱 오류: {e}"

        return sub_status, status_reasoning

    def _parse_score_result(
        self, content: str, sub_item_ids: List[int], sub_status: Dict[int, str]
    ) -> tuple:
        """2단계: 점수 평가 결과 파싱 (상태 범위 강제 적용)"""
        sub_scores = {}
        sub_reasoning = {}
        missing = []
        weak = []

        try:
            # JSON 블록 추출
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
                json_str = json_str.replace("}}", "}")
                json_str = re.sub(r'<[\d\.\-]+>', '5.0', json_str)
            else:
                json_match = re.search(r'\{[\s\S]*\}', content)
                json_str = json_match.group(0) if json_match else None

            if json_str:
                data = json.loads(json_str)
                scores_data = data.get("sub_scores", {})
                reasoning_data = data.get("sub_reasoning", {})

                for sid in sub_item_ids:
                    sid_str = str(sid)
                    status = sub_status.get(sid, "moderate")
                    score_range = STATUS_SCORE_RANGES.get(status, (4.0, 6.0))

                    # LLM이 반환한 점수
                    raw_score = scores_data.get(sid_str, scores_data.get(sid, sum(score_range) / 2))
                    try:
                        raw_score = float(raw_score)
                    except (ValueError, TypeError):
                        raw_score = sum(score_range) / 2

                    # ★ 핵심: 상태 범위 내로 강제 조정 ★
                    min_score, max_score = score_range
                    enforced_score = min(max_score, max(min_score, raw_score))
                    sub_scores[sid] = enforced_score

                    # 상태에 따라 missing/weak 분류
                    rubric = get_detailed_rubric(sid)
                    item_name = rubric.get("name", f"항목 {sid}") if rubric else f"항목 {sid}"

                    if status == "absent":
                        missing.append(item_name)
                    elif status == "weak":
                        weak.append(item_name)

                    # reasoning에 상태 정보 포함
                    reasoning = reasoning_data.get(sid_str, reasoning_data.get(str(sid), ""))
                    sub_reasoning[sid] = f"[{status.upper()}] {reasoning}"
            else:
                # 파싱 실패 시 상태 기반 기본값
                for sid in sub_item_ids:
                    status = sub_status.get(sid, "moderate")
                    score_range = STATUS_SCORE_RANGES.get(status, (4.0, 6.0))
                    sub_scores[sid] = sum(score_range) / 2
                    sub_reasoning[sid] = f"[{status.upper()}] 파싱 실패, 상태 기반 기본값"

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError) as e:
            print(f"[ADDIERubricEvaluator] 점수 파싱 오류: {e}")
            for sid in sub_item_ids:
                status = sub_status.get(sid, "moderate")
                score_range = STATUS_SCORE_RANGES.get(status, (4.0, 6.0))
                sub_scores[sid] = sum(score_range) / 2
                sub_reasoning[sid] = f"[{status.upper()}] 파싱 오류: {e}"

        return sub_scores, sub_reasoning, missing, weak

    def _aggregate_sub_scores_to_item(
        self, item_id: str, all_sub_scores: dict
    ) -> float:
        """33개 소단계 점수를 13개 항목으로 집계 (평균)"""
        sub_item_ids = get_sub_items_for_item(item_id)
        if not sub_item_ids:
            return 5.0

        scores = [all_sub_scores.get(sid, 5.0) for sid in sub_item_ids]
        return sum(scores) / len(scores) if scores else 5.0

    def _build_final_score_from_sub_items(
        self,
        all_sub_scores: dict,
        all_sub_reasoning: dict,
        all_missing: list,
        all_weak: list,
    ) -> ADDIEScore:
        """33개 소단계 점수를 13개 항목으로 통합하여 최종 점수 생성"""

        # 13개 항목별 점수 계산 (소단계 평균)
        item_scores = {}
        for item_id in ITEM_MAPPING.keys():
            item_scores[item_id] = self._aggregate_sub_scores_to_item(item_id, all_sub_scores)

        # 13개 항목별 reasoning 생성 (소단계 reasoning 통합)
        item_reasoning = {}
        for item_id, sub_ids in ITEM_MAPPING.items():
            reasoning_parts = []
            for sid in sub_ids:
                rubric = get_detailed_rubric(sid)
                name = rubric.get("name", f"항목 {sid}")
                reason = all_sub_reasoning.get(sid, "")
                score = all_sub_scores.get(sid, 5.0)
                reasoning_parts.append(f"[{name}({score:.1f}점)] {reason}")
            item_reasoning[item_id] = " | ".join(reasoning_parts)

        # 단계별 PhaseScore 구성 (get_item_ids_for_phase 사용으로 중앙화)
        phase_scores = {}

        for phase in ADDIEPhase:
            phase_scores[phase] = self._build_phase_score(
                phase,
                get_item_ids_for_phase(phase),
                item_scores,
                item_reasoning,
            )

        # 종합 점수 계산
        total_raw = sum(ps.raw_score for ps in phase_scores.values())
        total_weighted = sum(
            ps.raw_score * self.phase_weights[phase]
            for phase, ps in phase_scores.items()
        )

        # 정규화 (0-100)
        max_possible = sum(
            ps.max_score * self.phase_weights[phase]
            for phase, ps in phase_scores.items()
        )
        normalized = (total_weighted / max_possible) * 100 if max_possible > 0 else 0

        # 33개 소단계별 상세 점수 정보 생성
        sub_item_details = []
        for sid in sorted(all_sub_scores.keys()):
            rubric = get_detailed_rubric(sid)
            sub_item_details.append(
                f"[{sid}] {rubric.get('name', '')}: {all_sub_scores[sid]:.1f}점"
            )

        return ADDIEScore(
            analysis=phase_scores[ADDIEPhase.ANALYSIS],
            design=phase_scores[ADDIEPhase.DESIGN],
            development=phase_scores[ADDIEPhase.DEVELOPMENT],
            implementation=phase_scores[ADDIEPhase.IMPLEMENTATION],
            evaluation=phase_scores[ADDIEPhase.EVALUATION],
            total_raw=total_raw,
            total_weighted=total_weighted,
            normalized_score=normalized,
            strengths=[],
            improvements=all_missing[:5] + all_weak[:5],
            overall_assessment=f"33개 소단계 평가 완료. 누락 요소: {len(all_missing)}개, 빈약한 영역: {len(all_weak)}개\n\n소단계별 점수:\n" + "\n".join(sub_item_details),
        )

    def _build_phase_score(
        self,
        phase: ADDIEPhase,
        item_ids: List[str],
        item_scores: dict,
        item_reasoning: dict,
    ) -> PhaseScore:
        """단계별 점수 구성"""
        items = []
        for item_id in item_ids:
            score = item_scores.get(item_id, 5.0)
            item_def = get_rubric_definition(phase.value, item_id)
            reasoning = item_reasoning.get(item_id, "")

            items.append(RubricItem(
                item_id=item_id,
                phase=phase,
                name=item_def.get("name", item_id),
                description=item_def.get("description", ""),
                score=score,
                reasoning=reasoning,
            ))

        raw_score = sum(item.score for item in items)
        max_score = len(items) * MAX_SCORE_PER_ITEM

        return PhaseScore(
            phase=phase,
            items=items,
            raw_score=raw_score,
            weighted_score=raw_score * self.phase_weights[phase],
            max_score=max_score,
        )

    def _generate_context_guidelines(self, scenario: dict) -> str:
        """학습자 맥락 기반 동적 평가 지침 생성 (Dual-Track)"""
        context = scenario.get("context", {})
        target = context.get("target_audience", "")
        prior = context.get("prior_knowledge", "")
        duration = context.get("duration", "")
        environment = context.get("learning_environment", "")

        guidelines = []

        # 연령별 지침 (Context Matrix 1-4번)
        target_lower = target.lower()
        if any(kw in target_lower for kw in ["초등", "어린이", "아동"]):
            guidelines.append(
                "- 학습자가 아동이므로 시각적 요소, 게임화, 흥미 유발 전략의 적절성을 중점적으로 평가하라"
            )
            guidelines.append(
                "- 추상적 개념보다 구체적 예시와 활동 중심 설계가 충분한지 검토하라"
            )
        elif any(kw in target_lower for kw in ["중학", "고등", "청소년"]):
            guidelines.append(
                "- 청소년 학습자의 자기주도 학습 역량 개발이 설계에 반영되었는지 평가하라"
            )
            guidelines.append(
                "- 또래 협력 및 토론 활동이 적절히 포함되었는지 검토하라"
            )
        elif any(kw in target_lower for kw in ["성인", "직장인", "신입", "경력"]):
            guidelines.append(
                "- 성인 학습자의 실무 적용성과 자기주도 학습 원칙(Andragogy)이 반영되었는지 평가하라"
            )
            guidelines.append(
                "- 업무 현장과의 연계성, 즉각적 활용 가능성을 중점적으로 검토하라"
            )

        # 도메인 전문성별 지침 (Context Matrix 10-12번)
        prior_lower = prior.lower()
        if any(kw in prior_lower for kw in ["없음", "초보", "입문", "처음"]):
            guidelines.append(
                "- 사전 지식이 부족한 학습자를 위한 기초 개념 설명과 스캐폴딩이 충분한지 평가하라"
            )
        elif any(kw in prior_lower for kw in ["고급", "전문", "경력", "풍부"]):
            guidelines.append(
                "- 고급 학습자를 위한 심화 내용과 도전적 과제가 포함되었는지 평가하라"
            )

        # 환경별 지침 (Context Matrix 33-39번)
        env_lower = environment.lower()
        if any(kw in env_lower for kw in ["오프라인", "대면", "교실", "회의실"]):
            guidelines.append(
                "- 대면 환경에서의 학습자 상호작용과 즉각적 피드백 전략이 적절한지 평가하라"
            )
            guidelines.append(
                "- 교수자의 현장 진행 역량과 활동 기반 학습 설계가 충분한지 검토하라"
            )
        elif any(kw in env_lower for kw in ["실시간", "zoom", "화상"]):
            guidelines.append(
                "- 실시간 온라인 환경에서의 참여 유도와 집중 유지 전략이 적절한지 평가하라"
            )
            guidelines.append(
                "- 기술적 장애 대응과 비언어적 소통 보완 전략이 설계되었는지 검토하라"
            )
        elif any(kw in env_lower for kw in ["온라인", "lms", "비실시간"]):
            guidelines.append(
                "- 비대면 환경에서의 학습자 참여와 상호작용 촉진 전략이 적절한지 평가하라"
            )
        elif any(kw in env_lower for kw in ["모바일", "마이크로러닝"]):
            guidelines.append(
                "- 모바일 환경에 적합한 짧고 집중적인 학습 단위 설계가 적절한지 평가하라"
            )
            guidelines.append(
                "- 이동 중 학습 상황을 고려한 인터페이스와 콘텐츠 구성이 적절한지 검토하라"
            )
        elif any(kw in env_lower for kw in ["시뮬레이션", "vr"]):
            guidelines.append(
                "- 시뮬레이션/VR 환경의 기술적 적합성과 학습 효과 연계가 적절한지 평가하라"
            )
        elif any(kw in env_lower for kw in ["pbl", "프로젝트"]):
            guidelines.append(
                "- 프로젝트 기반 학습의 구조화된 단계와 산출물 설계가 적절한지 평가하라"
            )
            guidelines.append(
                "- 협력 학습과 과정 중심 평가 전략이 충분히 반영되었는지 검토하라"
            )

        # 교육 도메인별 지침 (Context Matrix 23-32번)
        topic = context.get("topic", "").lower()
        subject = context.get("subject", "").lower()
        objectives = " ".join(context.get("objectives", [])).lower() if isinstance(context.get("objectives"), list) else ""
        domain_text = f"{topic} {subject} {objectives}"

        # 23: 언어
        if any(kw in domain_text for kw in ["영어", "회화", "작문", "문법", "한국어", "외국어", "언어"]):
            guidelines.append(
                "- 언어 교육에 적합한 말하기/듣기/읽기/쓰기 통합 활동 설계가 적절한지 평가하라"
            )
        # 24: 수학
        elif any(kw in domain_text for kw in ["수학", "통계", "확률", "기하", "대수"]):
            guidelines.append(
                "- 수학 개념의 단계적 구조화와 시각화 자료 활용이 적절한지 평가하라"
            )
        # 25: 과학
        elif any(kw in domain_text for kw in ["물리", "화학", "생물", "과학", "실험"]):
            guidelines.append(
                "- 과학 탐구 활동과 실험 설계가 학습 목표에 부합하는지 평가하라"
            )
        # 26: 사회
        elif any(kw in domain_text for kw in ["역사", "사회", "경제학", "정치", "지리"]):
            guidelines.append(
                "- 사회과 교육에서 비판적 사고와 토론/협력 활동 설계가 적절한지 평가하라"
            )
        # 27: 개발(Software/IT)
        elif any(kw in domain_text for kw in ["코딩", "프로그래밍", "파이썬", "개발", "소프트웨어", "python"]):
            guidelines.append(
                "- IT/개발 교육에서 실습 환경 구성과 hands-on 활동 설계가 적절한지 평가하라"
            )
        # 28: AI
        elif any(kw in domain_text for kw in ["인공지능", "ai", "머신러닝", "딥러닝"]):
            guidelines.append(
                "- AI 교육에서 실습 환경과 이론-실습 연계 설계가 적절한지 평가하라"
            )
        # 29: 의료/간호
        elif any(kw in domain_text for kw in ["의료", "간호", "환자", "병원", "의학"]):
            guidelines.append(
                "- 의료 교육에서 절차적 정확성과 안전 프로토콜 반영이 적절한지 평가하라"
            )
        # 30: 경영/HR/경영지원
        elif any(kw in domain_text for kw in ["리더십", "마케팅", "경영", "인사", "hr"]):
            guidelines.append(
                "- 경영 교육에서 조직 요구분석 반영과 성과 평가 체계가 적절한지 평가하라"
            )
        # 31: 교육(교수·학습)
        elif any(kw in domain_text for kw in ["교수법", "교사", "수업", "교육학"]):
            guidelines.append(
                "- 교육학 교육에서 교수법 설계와 학습 평가 체계가 적절한지 평가하라"
            )
        # 32: 서비스/고객응대
        elif any(kw in domain_text for kw in ["cs", "고객응대", "서비스", "상담"]):
            guidelines.append(
                "- 서비스 교육에서 롤플레이 실습과 피드백 체계가 적절한지 평가하라"
            )

        # 시간 제약별 지침
        if any(kw in duration for kw in ["분", "시간"]) and not any(kw in duration for kw in ["개월", "주"]):
            guidelines.append(
                "- 단시간 수업에서 핵심 학습목표 달성을 위한 효율적 설계인지 평가하라"
            )
        elif any(kw in duration for kw in ["개월", "학기"]):
            guidelines.append(
                "- 장기 과정에 적합한 단계별 학습 구조와 지속적 평가 체계가 설계되었는지 평가하라"
            )

        if guidelines:
            return "\n\n## 학습자 맥락 기반 평가 지침\n" + "\n".join(guidelines)
        return ""

    def _format_scenario(self, scenario: dict) -> str:
        """시나리오 포맷팅 (학습자 맥락 지침 포함)"""
        context = scenario.get("context", {})
        parts = [
            f"제목: {scenario.get('title', '미지정')}",
            f"대상: {context.get('target_audience', '미지정')}",
            f"사전지식: {context.get('prior_knowledge', '미지정')}",
            f"시간: {context.get('duration', '미지정')}",
            f"환경: {context.get('learning_environment', '미지정')}",
            f"학습 목표: {', '.join(scenario.get('learning_goals', []))}",
            f"제약조건: {scenario.get('constraints', {})}",
        ]

        # 학습자 맥락 지침 추가 (Dual-Track)
        context_guidelines = self._generate_context_guidelines(scenario)

        return "\n".join(parts) + context_guidelines

    def _create_default_score(self, default: float = 5.0) -> ADDIEScore:
        """기본 점수 생성 (파싱 실패 시)"""
        phase_scores = {}

        for phase in ADDIEPhase:
            item_ids = get_item_ids_for_phase(phase)
            items = []
            for item_id in item_ids:
                item_def = get_rubric_definition(phase.value, item_id)
                items.append(RubricItem(
                    item_id=item_id,
                    phase=phase,
                    name=item_def.get("name", item_id),
                    description=item_def.get("description", ""),
                    score=default,
                    reasoning="기본값 (평가 실패)",
                ))

            raw_score = sum(item.score for item in items)
            max_score = len(items) * MAX_SCORE_PER_ITEM

            phase_scores[phase] = PhaseScore(
                phase=phase,
                items=items,
                raw_score=raw_score,
                weighted_score=raw_score * self.phase_weights[phase],
                max_score=max_score,
            )

        total_raw = sum(ps.raw_score for ps in phase_scores.values())
        total_weighted = sum(
            ps.raw_score * self.phase_weights[phase]
            for phase, ps in phase_scores.items()
        )
        max_possible = sum(
            ps.max_score * self.phase_weights[phase]
            for phase, ps in phase_scores.items()
        )
        normalized = (total_weighted / max_possible) * 100 if max_possible > 0 else 0

        return ADDIEScore(
            analysis=phase_scores[ADDIEPhase.ANALYSIS],
            design=phase_scores[ADDIEPhase.DESIGN],
            development=phase_scores[ADDIEPhase.DEVELOPMENT],
            implementation=phase_scores[ADDIEPhase.IMPLEMENTATION],
            evaluation=phase_scores[ADDIEPhase.EVALUATION],
            total_raw=total_raw,
            total_weighted=total_weighted,
            normalized_score=normalized,
            strengths=[],
            improvements=["평가 실패로 인한 기본값 적용"],
            overall_assessment="평가를 완료할 수 없습니다.",
        )
