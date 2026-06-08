# Dick & Carey Agent

Dick & Carey 모형의 **체제적 교수설계(Systems Approach)** 10단계 프로세스를 구현한 교수설계 에이전트입니다.

## 특징

### Dick & Carey 모형 10단계

| 단계 | 주요 활동 | 도구 |
|------|----------|------|
| 1 | 교수목적 설정 | `set_instructional_goal` |
| 2 | 교수분석 | `analyze_instruction` |
| 3 | 학습자/환경 분석 | `analyze_entry_behaviors`, `analyze_context` |
| 4 | 수행목표 진술 | `write_performance_objectives` |
| 5 | 평가도구 개발 | `develop_assessment_instruments` |
| 6 | 교수전략 개발 | `develop_instructional_strategy` |
| 7 | 교수자료 개발 | `develop_instructional_materials` |
| 8 | 형성평가 실시 | `conduct_formative_evaluation` |
| 9 | 교수프로그램 수정 | `revise_instruction` |
| 10 | 총괄평가 실시 | `conduct_summative_evaluation` |

### 피드백 루프

형성평가-수정 피드백 루프를 통해 품질 기준 달성까지 반복적으로 개선합니다.

```
[형성평가(8)] → [품질판정] → (점수 < 7.0 AND 반복 < 3) → [수정(9)] → [형성평가(8)]
                          ↘ (점수 >= 7.0 OR 반복 >= 3) → [총괄평가(10)] → [END]
```

- **품질 기준**: 7.0점 (기본값, 조정 가능)
- **최대 반복**: 3회 (기본값, 조정 가능)

## 설치

```bash
cd agents/dick-carey-agent
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

## 환경 설정

공통 모델 설정은 `--agent-model-*` CLI flags 또는 저장소 루트의 optional `.env`로 지정합니다. Dick & Carey
도구들은 `shared/llm` factory를 통해 같은 cloud/local 설정을 사용합니다.

```bash
# Upstage Solar round-robin 예시
AGENT_MODEL_PROVIDER=upstage
AGENT_MODEL_API_SPEC=openai_compatible
AGENT_MODEL_NAME=solar-pro3
AGENT_MODEL_BASE_URL=https://api.upstage.ai/v1/solar
AGENT_MODEL_API_KEY_ENVS=UPSTAGE_API_KEY,UPSTAGE_API_KEY2,UPSTAGE_API_KEY3
AGENT_MODEL_CREDENTIAL_STRATEGY=round_robin

# 로컬 Ollama 예시
# AGENT_MODEL_PROVIDER=local-ollama
# AGENT_MODEL_BASE_URL=http://localhost:11434/v1
# AGENT_MODEL_NAME=qwen2.5:14b
# AGENT_MODEL_API_KEY=not-needed
```

## 사용법

### CLI

```bash
# 기본 실행
dick-carey-agent run --input scenario.json --output result.json

# 상세 출력
dick-carey-agent run -i scenario.json -o result.json -v

# 피드백 루프 설정 조정
dick-carey-agent run -i scenario.json -o result.json \
    --max-iterations 5 \
    --quality-threshold 8.0

# Trajectory 저장
dick-carey-agent run -i scenario.json -o result.json -t trajectory.json

# 시나리오 검증
dick-carey-agent validate scenario.json

# 버전 정보
dick-carey-agent version
```

### Python API

```python
from dick_carey_agent import DickCareyAgent

agent = DickCareyAgent(
    model="solar-mini",
    max_iterations=3,
    quality_threshold=7.0,
    debug=True,
)

scenario = {
    "scenario_id": "TEST-001",
    "title": "신입사원 온보딩 교육",
    "context": {
        "target_audience": "신입사원",
        "duration": "4시간",
        "learning_environment": "온라인",
    },
    "learning_goals": [
        "회사 문화와 가치를 이해한다",
        "업무 프로세스를 파악한다",
    ],
}

result = agent.run(scenario)

# 결과 확인
print(f"품질 점수: {result['metadata']['final_quality_score']}")
print(f"반복 횟수: {result['metadata']['iteration_count']}")
print(f"최종 결정: {result['metadata']['final_decision']}")
```

## 출력 형식

### ADDIE 호환 출력 (`addie_output`)

기존 ADDIE 기반 평가 시스템과 호환되도록 Dick & Carey 산출물을 ADDIE 형식으로 매핑합니다.

| Dick & Carey | ADDIE |
|--------------|-------|
| 1-3단계 | Analysis |
| 4-5단계 | Design |
| 6-7단계 | Development |
| 8-9단계 | Implementation |
| 10단계 | Evaluation |

### Dick & Carey 고유 출력 (`dick_carey_output`)

10단계 각각의 상세 산출물을 포함합니다.

## 테스트

```bash
cd agents/dick-carey-agent
pytest tests/ -v
```

## 디렉토리 구조

```
dick-carey-agent/
├── src/dick_carey_agent/
│   ├── __init__.py
│   ├── agent.py          # StateGraph 정의 (피드백 루프)
│   ├── state.py          # DickCareyState 스키마
│   ├── prompts.py        # 단계별 프롬프트
│   ├── cli.py            # Typer CLI
│   └── tools/
│       ├── __init__.py
│       ├── goal_analysis.py      # 1-3단계 (4개)
│       ├── objective_assessment.py # 4-5단계 (2개)
│       ├── strategy_materials.py # 6-7단계 (2개)
│       └── evaluation.py         # 8-10단계 (3개)
├── tests/
│   ├── test_state.py
│   └── test_tools.py
├── pyproject.toml
└── README.md
```

## 참고 자료

- Dick, W., Carey, L., & Carey, J. O. (2009). *The Systematic Design of Instruction* (7th ed.). Pearson.

## 라이센스

MIT
