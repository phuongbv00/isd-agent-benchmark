# 교수설계 Agent CLI 인터페이스 명세

## 1. 개요

모든 교수설계 Agent는 동일한 CLI 인터페이스를 구현해야 합니다. 이를 통해 평가 Agent가 일관된 방식으로 각 Agent를 호출하고 결과를 수집할 수 있습니다.

## 2. 공통 인터페이스

### 2.1 기본 명령어

```bash
# 교수설계 생성
<agent-name> generate --input <scenario.json> --output <result.json>

# 버전 확인
<agent-name> --version

# 도움말
<agent-name> --help
```

### 2.2 Agent 이름

| Agent | CLI 명령어 | 설계 모형 |
|-------|-----------|----------|
| EduPlanner | `eduplanner` | 3-Agent 협업 구조 |
| Baseline | `baseline` | 단일 LLM 기준선 |
| ReAct-ISD | `react-isd` | LangGraph ReAct 패턴 |
| ADDIE-Agent | `addie-agent` | ADDIE 5단계 순차 실행 |
| Dick-Carey-Agent | `dick-carey-agent` | Dick & Carey 10단계 체제적 교수설계 |
| RPISD-Agent | `rpisd-agent` | 래피드 프로토타이핑 교수설계 |
| 평가 Agent | `isd-evaluator` | 평가 시스템 |

## 3. 교수설계 Agent 인터페이스

### 3.1 `generate` 명령어

교수설계 산출물을 생성합니다.

```bash
<agent-name> generate --input <scenario.json> --output <result.json> [OPTIONS]
```

**필수 인자:**

| 인자 | 설명 |
|------|------|
| `--input`, `-i` | 입력 시나리오 JSON 파일 경로 |
| `--output`, `-o` | 출력 결과 JSON 파일 경로 |

**선택 인자:**

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--model` | 사용할 LLM 모델 | `solar-pro2` |
| `--language` | 출력 언어 | `ko` |
| `--verbose`, `-v` | 상세 로그 출력 | `false` |
| `--iterations` | 반복 횟수 (EduPlanner) | `3` |
| `--timeout` | 타임아웃 (초) | `300` |

**예시:**

```bash
# EduPlanner (3-Agent 협업)
eduplanner generate -i scenarios/idld_aligned/scenario_idld_0001.json -o results/eduplanner/result.json

# ReAct-ISD (상세 로그)
react-isd generate -i scenario.json -o result.json --verbose

# Baseline (다른 모델)
baseline generate -i scenario.json -o result.json --model local-model

# ADDIE-Agent (디버그 모드)
addie-agent run scenario.json -o output.json --debug

# Dick-Carey-Agent (피드백 루프 설정)
dick-carey-agent run --input scenario.json --output result.json --max-iterations 5 --quality-threshold 8.0

# RPISD-Agent (품질 임계값 설정)
rpisd-agent run scenario.json --output ./output --threshold 0.8
```

### 3.2 입력 형식

입력 파일은 `shared/schemas/input_schema.json` 스키마를 따릅니다.

```json
{
  "scenario_id": "IDLD-0001",
  "title": "신입사원 온보딩 교육",
  "context": {
    "target_audience": "신입사원 (대졸, IT 기업)",
    "prior_knowledge": "기본 컴퓨터 활용 능력",
    "duration": "2시간",
    "learning_environment": "온라인 비동기 학습"
  },
  "learning_goals": [
    "회사 문화와 가치를 이해한다",
    "업무 프로세스를 설명할 수 있다"
  ],
  "constraints": {
    "budget": "low",
    "resources": ["PPT", "동영상", "퀴즈"]
  }
}
```

### 3.3 출력 형식

출력 파일은 `shared/schemas/output_schema.json` 스키마를 따릅니다.

```json
{
  "scenario_id": "IDLD-0001",
  "agent_id": "eduplanner",
  "timestamp": "2024-12-09T10:00:00Z",
  "addie_output": {
    "analysis": { ... },
    "design": { ... },
    "development": { ... },
    "implementation": { ... },
    "evaluation": { ... }
  },
  "trajectory": {
    "tool_calls": [ ... ],
    "reasoning_steps": [ ... ]
  },
  "metadata": {
    "model": "solar-pro2",
    "total_tokens": 5000,
    "execution_time_seconds": 30
  }
}
```

### 3.4 종료 코드

| 코드 | 의미 |
|------|------|
| 0 | 성공 |
| 1 | 일반 오류 |
| 2 | 입력 파일 오류 |
| 3 | 출력 파일 오류 |
| 4 | API 오류 (LLM 호출 실패) |
| 5 | 타임아웃 |
| 6 | 스키마 검증 오류 |

### 3.5 로그 출력

`--verbose` 옵션 사용 시 표준 오류(stderr)로 로그를 출력합니다.

```
[2024-12-09 10:00:00] INFO: 시나리오 로드 완료: IDLD-0001
[2024-12-09 10:00:01] INFO: 학습자 분석 시작...
[2024-12-09 10:00:05] INFO: 학습자 분석 완료
[2024-12-09 10:00:06] INFO: 환경 분석 시작...
...
```

## 4. 평가 Agent 인터페이스

### 4.1 `evaluate` 명령어

Agent를 실행하고 결과를 평가합니다.

```bash
isd-evaluator evaluate --agent <agent-id> --scenario <scenario.json> [OPTIONS]
```

또는 기존 결과 파일을 평가합니다.

```bash
isd-evaluator evaluate --result <result.json> [OPTIONS]
```

**인자:**

| 인자 | 설명 |
|------|------|
| `--agent`, `-a` | 평가할 Agent ID |
| `--scenario`, `-s` | 시나리오 파일 경로 |
| `--result`, `-r` | 기존 결과 파일 경로 |
| `--metrics`, `-m` | 평가 메트릭 (기본: all) |
| `--output`, `-o` | 평가 결과 저장 경로 |

**예시:**

```bash
# Agent 실행 후 평가
isd-evaluator evaluate -a eduplanner -s scenarios/idld_aligned/scenario_idld_0001.json

# 기존 결과 평가
isd-evaluator evaluate -r results/eduplanner/result.json

# 특정 메트릭만 평가
isd-evaluator evaluate -r result.json -m design_quality,trajectory_quality
```

### 4.2 `compare` 명령어

여러 Agent를 비교 평가합니다.

```bash
isd-evaluator compare --agents <agent1,agent2,...> --scenarios <path> [OPTIONS]
```

**예시:**

```bash
# 3종 Agent 비교
isd-evaluator compare \
  --agents eduplanner,baseline,react-isd \
  --scenarios scenarios/idld_aligned/

# 특정 시나리오만
isd-evaluator compare \
  --agents eduplanner,baseline \
  --scenarios scenarios/idld_aligned/scenario_idld_0001.json
```

### 4.3 `report` 명령어

평가 결과 리포트를 생성합니다.

```bash
isd-evaluator report --results-dir <path> --output <report.md> [OPTIONS]
```

**예시:**

```bash
# Markdown 리포트 생성
isd-evaluator report --results-dir results/ --output report.md

# HTML 리포트 생성
isd-evaluator report --results-dir results/ --output report.html --format html
```

## 5. 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `AGENT_MODEL_PROVIDER` | Agent model provider preset (`openrouter`, `upstage`, `local-lmstudio`, etc.) | `openrouter` |
| `AGENT_MODEL_API_SPEC` | Agent model API spec (`openai_compatible`, `anthropic`) | `openai_compatible` |
| `AGENT_MODEL_NAME` | Agent model name sent to the backend | provider default |
| `AGENT_MODEL_BASE_URL` | Agent model OpenAI-compatible endpoint | provider default |
| `AGENT_MODEL_API_KEY_ENV` | Env var containing one agent model API key | - |
| `AGENT_MODEL_API_KEY_ENVS` | Comma-separated agent key env vars for round-robin | - |
| `AGENT_MODEL_API_KEY` | Direct agent model API key; use `not-needed` for local endpoints | - |
| `JUDGE_MODEL_PROVIDER` | Default judge model provider preset/label | `openrouter` |
| `JUDGE_MODEL_NAMES` | Comma-separated judge model names | evaluator default |
| `JUDGE_MODEL_PROVIDERS` | Comma-separated judge provider labels, aligned with model names | model prefixes |
| `JUDGE_MODEL_BASE_URL` | Default judge model OpenAI-compatible endpoint | provider default |
| `JUDGE_MODEL_BASE_URLS` | Comma-separated judge model endpoints, aligned with model names | - |
| `JUDGE_MODEL_API_KEY_ENV` | Env var containing one judge model API key | - |
| `JUDGE_MODEL_API_KEY_ENVS` | Comma-separated judge model key env groups, `|` within a group for multiple keys | - |
| `JUDGE_MODEL_API_KEY` | Direct judge model API key; use `not-needed` for local endpoints | - |
| `JUDGE_MODEL_API_KEYS` | Comma-separated direct judge model API key groups, `|` within a group for multiple keys | - |
| `JUDGE_MODEL_CREDENTIAL_STRATEGIES` | Comma-separated credential strategies (`first`, `round_robin`), aligned with model names | inferred |
| `ISD_LOG_LEVEL` | 로그 레벨 | `INFO` |
| `ISD_TIMEOUT` | 기본 타임아웃 (초) | `300` |

## 6. Python API

CLI 외에도 Python API를 통해 직접 호출할 수 있습니다.

### 6.1 교수설계 Agent

```python
from eduplanner import EduPlannerAgent

agent = EduPlannerAgent(model="gpt-4o")
result = agent.generate(scenario_path="scenario.json")
result.save("result.json")
```

### 6.2 평가 Agent

```python
from isd_evaluator import ISDEvaluator

evaluator = ISDEvaluator()

# 단일 평가
result = evaluator.evaluate_agent(
    agent_id="eduplanner",
    scenario_path="scenario.json"
)

# 비교 평가
report = evaluator.compare_agents(
    agent_ids=["eduplanner", "baseline", "react-isd"],
    scenario_paths=["scenarios/idld_aligned/"]
)
```

## 7. 에러 처리

### 7.1 입력 검증 오류

```json
{
  "error": "ValidationError",
  "message": "Invalid scenario format",
  "details": {
    "field": "context.duration",
    "expected": "string matching pattern ^[0-9]+(시간|분)$",
    "received": "2 hours"
  }
}
```

### 7.2 API 오류

```json
{
  "error": "APIError",
  "message": "OpenAI API call failed",
  "details": {
    "status_code": 429,
    "message": "Rate limit exceeded"
  }
}
```

## 8. 테스트

각 Agent는 다음 테스트를 통과해야 합니다.

```bash
# 단위 테스트
pytest tests/

# 통합 테스트 (실제 API 호출)
pytest tests/integration/ --run-integration

# 스키마 검증 테스트
pytest tests/test_schema_validation.py
```

## 9. 추가 Agent 상세 명세

### 9.1 ADDIE-Agent

ADDIE 모형(Analysis-Design-Development-Implementation-Evaluation) 기반의 순차적 교수설계 에이전트.

**특징:**
- LangGraph StateGraph를 사용하여 ADDIE 5단계 엄격한 순차 실행
- 11개 전문 도구로 고품질 산출물 생성
- 피드백 루프 없음 (단방향 진행)

**CLI 명령어:**

```bash
# 실행
addie-agent run <scenario.json> -o <output.json>

# 디버그 모드
addie-agent run <scenario.json> --debug

# 검증
addie-agent validate <scenario.json>

# 버전
addie-agent version
```

**도구 (11개):**

| 단계 | 도구 | 역할 |
|------|------|------|
| Analysis | `analyze_learner`, `analyze_context`, `analyze_task` | 학습자/환경/과제 분석 |
| Design | `design_objectives`, `design_assessment`, `design_strategy` | 목표/평가/전략 설계 |
| Development | `create_lesson_plan`, `create_materials` | 레슨 플랜, 학습 자료 |
| Implementation | `create_implementation_plan` | 실행 가이드 |
| Evaluation | `create_quiz_items`, `create_rubric` | 평가문항, 루브릭 |

---

### 9.2 Dick-Carey-Agent

Dick & Carey 모형의 체제적 교수설계(Systems Approach) 10단계 프로세스 구현.

**특징:**
- 형성평가-수정 피드백 루프를 통한 반복적 개선
- 품질 기준 달성까지 최대 3회 반복 (기본값)
- ADDIE 호환 출력 + Dick & Carey 고유 출력

**CLI 명령어:**

```bash
# 기본 실행
dick-carey-agent run --input <scenario.json> --output <result.json>

# 피드백 루프 설정 조정
dick-carey-agent run -i <scenario.json> -o <result.json> \
    --max-iterations 5 \
    --quality-threshold 8.0

# Trajectory 저장
dick-carey-agent run -i <scenario.json> -o <result.json> -t trajectory.json

# 검증
dick-carey-agent validate <scenario.json>

# 버전
dick-carey-agent version
```

**10단계 프로세스:**

| 단계 | 활동 | 도구 |
|------|------|------|
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

---

### 9.3 RPISD-Agent

RPISD(Rapid Prototyping Instructional System Design) 래피드 프로토타이핑 교수설계 에이전트.

**특징:**
- 이중 루프 구조: 프로토타입 루프 + 개발 루프
- 프로토타입 버전 관리 및 반복 개선 이력 추적
- 의뢰인/전문가/학습자 3종 피드백 통합

**CLI 명령어:**

```bash
# 실행
rpisd-agent run <scenario.json> --output ./output --threshold 0.8

# 검증
rpisd-agent validate <output/result.json>

# 정보
rpisd-agent info
```

**도구 (14개):**

| 단계 | 도구 | 설명 |
|------|------|------|
| 착수 | `kickoff_meeting` | 프로젝트 범위, 역할 정의 |
| 분석 | `analyze_gap`, `analyze_performance`, `analyze_learner_characteristics`, `analyze_initial_task` | 차이/수행/학습자/과제 분석 |
| 설계 | `design_instruction`, `develop_prototype`, `analyze_task_detailed` | 교수설계, 프로토타입, 상세 과제 |
| 평가 | `evaluate_with_client`, `evaluate_with_expert`, `evaluate_with_learner`, `aggregate_feedback` | 3종 피드백 및 통합 |
| 개발 | `develop_final_program` | 최종 프로그램 개발 |
| 실행 | `implement_program` | 실행 계획 |

**출력 형식:**

ADDIE 호환 출력 (`addie_output`) + RPISD 고유 출력 (`rpisd_output`)

```json
{
  "scenario_id": "EASY-001",
  "agent_id": "rpisd-agent",
  "addie_output": { ... },
  "rpisd_output": {
    "kickoff": { ... },
    "analysis": { ... },
    "design": { ... },
    "prototype_versions": [ ... ],
    "usability_feedback": { ... },
    "development": { ... },
    "implementation": { ... }
  },
  "trajectory": { ... },
  "metadata": { ... }
}
```
