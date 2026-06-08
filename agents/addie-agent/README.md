# ADDIE Agent

ADDIE 모형(Analysis-Design-Development-Implementation-Evaluation) 기반의 순차적 교수설계 에이전트입니다.

## 특징

- **순차적 프로세스**: LangGraph StateGraph를 사용하여 ADDIE 5단계를 엄격하게 순차 실행
- **단계별 완결성**: 각 단계의 산출물이 다음 단계의 입력으로 활용
- **11개 전문 도구**: 각 단계별 특화된 도구로 고품질 산출물 생성
- **output_schema.json 준수**: 기존 벤치마크 시스템과 호환

## 아키텍처

```
[START] → [Analysis] → [Design] → [Development] → [Implementation] → [Evaluation] → [END]
```

### 단계별 도구

| 단계 | 도구 | 역할 |
|------|------|------|
| **Analysis** | `analyze_learner` | 학습자 특성, 동기, 어려움 |
| | `analyze_context` | 환경, 제약, 자원 |
| | `analyze_task` | 주제, 선수학습, 지식구조 |
| **Design** | `design_objectives` | Bloom's Taxonomy 학습목표 |
| | `design_assessment` | 진단/형성/총괄 평가계획 |
| | `design_strategy` | Gagné's 9 Events 교수전략 |
| **Development** | `create_lesson_plan` | 모듈별 레슨 플랜 |
| | `create_materials` | 학습자료 (PPT, 동영상 등) |
| **Implementation** | `create_implementation_plan` | 실행가이드, 기술요구사항 |
| **Evaluation** | `create_quiz_items` | 평가문항 10개+ |
| | `create_rubric` | 평가 루브릭 |

## 설치

```bash
cd agents/addie-agent
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

## 환경 설정

공통 모델 설정은 `--agent-model-*` CLI flags 또는 저장소 루트의 optional `.env`로 지정합니다. 모든 ADDIE 도구는
`shared/llm` factory를 통해 같은 설정을 사용합니다.

```bash
# OpenRouter 예시
AGENT_MODEL_PROVIDER=openrouter
AGENT_MODEL_API_SPEC=openai_compatible
AGENT_MODEL_NAME=openai/gpt-4o-mini
AGENT_MODEL_API_KEY_ENV=OPENROUTER_API_KEY

# 로컬 LM Studio 예시
# AGENT_MODEL_PROVIDER=local-lmstudio
# AGENT_MODEL_BASE_URL=http://localhost:1234/v1
# AGENT_MODEL_NAME=local-model
# AGENT_MODEL_API_KEY=not-needed
```

## 사용법

### CLI

```bash
# 시나리오 실행
addie-agent run scenario.json -o output.json

# 디버그 모드
addie-agent run scenario.json --debug

# 시나리오 검증
addie-agent validate scenario.json

# 버전 확인
addie-agent version
```

### Python API

```python
from addie_agent import ADDIEAgent

# 에이전트 생성
agent = ADDIEAgent(model="solar-mini", debug=True)

# 시나리오 정의
scenario = {
    "scenario_id": "TEST-001",
    "title": "신입사원 온보딩 교육",
    "context": {
        "target_audience": "신입사원",
        "duration": "2시간",
        "learning_environment": "대면 교육",
        "prior_knowledge": "기초 수준",
    },
    "learning_goals": [
        "조직 문화 이해",
        "업무 프로세스 파악",
        "협업 도구 활용",
    ],
}

# 실행
result = agent.run(scenario)

# 결과 확인
print(f"실행 시간: {result['metadata']['execution_time_seconds']:.2f}초")
print(f"도구 호출: {result['metadata']['tool_calls_count']}회")
```

## 테스트

```bash
# 모든 테스트 실행
pytest tests/

# 특정 테스트 실행
pytest tests/test_tools.py -v
```

## 프로젝트 구조

```
addie-agent/
├── src/
│   └── addie_agent/
│       ├── __init__.py
│       ├── agent.py          # StateGraph 정의
│       ├── state.py          # State 스키마
│       ├── prompts.py        # 단계별 프롬프트
│       ├── cli.py            # CLI 엔트리포인트
│       └── tools/            # 11개 도구
│           ├── __init__.py
│           ├── analysis.py   # 3개
│           ├── design.py     # 3개
│           ├── development.py # 2개
│           ├── implementation.py # 1개
│           └── evaluation.py # 2개
├── tests/
│   ├── __init__.py
│   ├── test_state.py
│   └── test_tools.py
├── pyproject.toml
└── README.md
```

## 기존 에이전트와 비교

| 특성 | EduPlanner | ReAct-ISD | ADDIE Agent |
|------|-----------|-----------|-------------|
| 단계 전이 | 에이전트별 반복 | 자율 호출 | **엄격한 순차 진행** |
| StateGraph | 없음 | create_react_agent | **선형 워크플로우** |
| 피드백 루프 | 있음 (반복) | 없음 | **없음 (단방향)** |
| State 관리 | Pydantic 모델 | dict | **TypedDict** |

## 참고 자료

- Branch, R. M. (2009). *Instructional design: The ADDIE approach*. Springer.
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
