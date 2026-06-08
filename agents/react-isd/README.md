# ReAct-ISD Agent

LangGraph 기반 ReAct 패턴 교수설계 Agent

## 개요

ReAct-ISD는 LangGraph의 ReAct(Reasoning + Acting) 패턴을 사용하여
11개의 전문 도구를 통해 ADDIE 교수설계 산출물을 생성합니다.

## 도구 목록

### Analysis (분석)
- `analyze_learners`: 학습자 분석
- `analyze_context`: 학습 환경 분석
- `analyze_task`: 과제 분석

### Design (설계)
- `design_objectives`: 학습 목표 설계
- `design_assessment`: 평가 전략 설계
- `design_strategy`: 교수 전략 설계

### Development (개발)
- `create_lesson_plan`: 차시별 수업 계획
- `create_materials`: 학습 자료 명세

### Implementation (실행)
- `create_implementation_plan`: 실행 계획

### Evaluation (평가)
- `create_quiz_items`: 퀴즈 문항 생성
- `create_rubric`: 평가 루브릭 생성

## 설치

```bash
pip install -e .
```

## 사용법

```bash
# 기본 실행
react-isd run --input scenario.json --output result.json

# 궤적 저장
react-isd run --input scenario.json --output result.json --trajectory traj.json

# 정보 출력
react-isd info
```

## CLI 옵션

| 옵션 | 설명 |
|------|------|
| `--input, -i` | 입력 시나리오 JSON 파일 |
| `--output, -o` | 출력 ADDIE 산출물 파일 |
| `--trajectory, -t` | 궤적 저장 파일 (선택) |
| `--verbose, -v` | 상세 출력 |

## LLM 설정

공통 모델 설정은 `--agent-model-*` CLI flags 또는 저장소 루트의 optional `.env`로 지정합니다. ReAct 도구들은
`shared/llm` factory를 통해 같은 provider-neutral 설정을 사용합니다.

```bash
# OpenRouter 예시
AGENT_MODEL_PROVIDER=openrouter
AGENT_MODEL_API_SPEC=openai_compatible
AGENT_MODEL_NAME=anthropic/claude-opus-4.5
AGENT_MODEL_API_KEY_ENV=OPENROUTER_API_KEY

# 로컬 vLLM 예시
# AGENT_MODEL_PROVIDER=local-vllm
# AGENT_MODEL_BASE_URL=http://localhost:8000/v1
# AGENT_MODEL_NAME=local-model
# AGENT_MODEL_API_KEY=not-needed
```

## 구현 검증 (Verification & Alignment)

본 구현체는 **ReAct (Reasoning and Acting)** 프레임워크 (Yao et al., 2022)를 기반으로 하며, 이를 **ISD Agent Benchmark**의 목적에 맞춰 교수설계 도메인에 적용하였습니다.

### 원본 방법론과의 공통점 (Aligned)
- **ReAct 패턴**: LLM이 추론(Reasoning)과 행동(Acting)을 교차하며 작업을 수행하는 핵심 매커니즘 구현
- **도구 활용**: 외부 도구(Tools)를 자율적으로 호출하여 정보를 수집하고 결과를 생성하는 방식 유지
- **LangGraph 활용**: LangChain/LangGraph의 검증된 ReAct 에이전트 구조 사용

### 도메인 맞춤형 변형 (Adapted)
- **순차적 프로세스 강제**: 일반적인 ReAct 에이전트와 달리, ADDIE 모델의 5단계(분석-설계-개발-실행-평가) 순서를 지키도록 프롬프트로 유도
- **도구 전문화**: 검색 도구(Search) 대신 교수설계 각 단계에 특화된 11개의 전문 도구(analyze_learners, design_objectives 등)를 제공하여 품질 확보
