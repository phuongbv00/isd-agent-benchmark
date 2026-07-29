# ISD Evaluator

교수설계 Agent 평가 시스템

## 개요

ISD Evaluator는 교수설계 Agent의 산출물과 생성 과정을 종합적으로 평가합니다.
Context Matrix 참고문서를 기반으로 33개 소단계 루브릭과 컨텍스트 인식형 가중치 조정을 지원합니다.

## 평가 메트릭

### ADDIE 루브릭 평가 (산출물 품질) - 총 100점

#### 33개 소단계 → 13개 통합 항목

Context Matrix 참고문서의 33개 소단계가 13개 상위 항목으로 매핑됩니다:

| 단계 | 통합 항목 | 소단계 (Context Matrix) | 가중치 |
|------|----------|------------------------|--------|
| **Analysis** | A1: 요구분석 | 1-4번 | 25% |
| | A2: 학습자 및 환경분석 | 5-6번 | |
| | A3: 과제 및 목표분석 | 7-10번 | |
| **Design** | D1: 평가 및 목표 정렬 설계 | 11-12번 | 25% |
| | D2: 교수전략 및 학습경험 설계 | 13-17번 | |
| | D3: 프로토타입 구조 설계 | 18번 | |
| **Development** | Dev1: 프로토타입 개발 | 19-22번 | 20% |
| | Dev2: 개발 결과 검토 및 수정 | 23번 | |
| **Implementation** | I1: 프로그램 실행 준비 | 24-25번 | 15% |
| | I2: 프로그램 실행 | 26-27번 | |
| **Evaluation** | E1: 형성평가 | 28-29번 | 15% |
| | E2: 총괄평가 및 채택 결정 | 30-32번 | |
| | E3: 프로그램 개선 및 환류 | 33번 | |

#### 점수 척도 (0-10점, 5단계)

| 등급 | 점수 | 설명 |
|------|------|------|
| 매우우수 | 9-10 | 이론과 실제 적절히 반영, 즉시 적용 가능 |
| 우수 | 7-8 | 이론과 실제 적절히 반영 |
| 보통 | 5-6 | 기본 요소 충족, 일부 부족 |
| 미흡 | 3-4 | 핵심 요소 결여 |
| 부재 | 1-2 | 해당 요소 거의 없음 |

#### Benchmark Examples (Few-shot 프롬프트)

각 소단계별 Benchmark Examples가 평가 프롬프트에 자동으로 포함되어
LLM이 일관된 평가 기준을 적용할 수 있도록 지원합니다.

### Trajectory 평가 (생성 과정) - BFCL 기반 100점

| 항목 | 점수 | 설명 |
|------|------|------|
| Tool Correctness | 25 | 도구 선택 정확성 |
| Argument Accuracy | 25 | 인자 전달 정확성 |
| Redundancy Avoidance | 25 | 중복 호출 회피 |
| Result Utilization | 25 | 결과 활용도 |

### 종합 점수

```
총점 = ADDIE × 0.7 + Trajectory × 0.3
```

### Constructive Alignment 평가 (RQ2, LLM 미사용)

`isd_evaluator.metrics.alignment`의 `AlignmentEvaluator`는 ADDIE 산출물의
**constructive alignment**(학습목표 ↔ 평가 ↔ 활동 ↔ 총괄평가 정합성)를
결정론적 공식으로 측정합니다 — 집계 단계에서 LLM 호출이 전혀 없고,
텍스트만 읽습니다(에이전트가 자기 신고한 `objective_id`/`level` 필드는
채점에 쓰지 않음). 구성 요소는 모두 [0, 1]:

**매칭 임계값(threshold)은 어디에도 없습니다** — 모든 지표는 연속형이며,
합성 스칼라(composite)도 없습니다. 각 지표를 따로 보고합니다.

| 구성 요소 | 설명 |
|-----------|------|
| `objective_assessment_similarity` | objective별 임의 평가항목까지의 max cosine 평균 (연속형) — `aligned(o)`의 *measures* 축 |
| `objective_activity_similarity` / `objective_evaluation_similarity` | 같은 공식, 활동 / 총괄평가 텍스트 대상 |
| `assessment_objective_similarity` | 역방향 (평가항목 → objective). **방향성 없음**: 높다는 것이 항상 좋다는 뜻은 아니므로 진단 지표로만 읽음 |
| `objective_cognitive_congruence` | best 평가항목(argmax)의 Bloom level ≥ objective level 비율 |
| Porter Alignment Index | topic × Bloom-level 분포 행렬 일치도 (Porter 2002) |
| Webb Bloom-Consistency | 매칭 항목의 Bloom level ≥ objective level 비율 (item-centric) |

**주 지표는 없습니다** (2026-07-28 변경). 위 7개는 평평한 **패널**이며 하나를 "주
지표"로 부르는 것은 이미 제거한 composite를 암묵적으로 되살리는 일입니다. 앞의 4개는
encoder가 재는 *textual correspondence*, 뒤의 3개는 Bloom classifier가 재는
*cognitive demand* — 서로 독립된 두 계열이고 각각 자체 sensitivity 축을 가집니다.
이름이 `*_alignment`에서 `*_similarity`로 바뀐 이유도 같습니다: 지표 이름은 그림 축과
슬라이드에서 읽히는데, 거기서 `..._alignment = 0.86`은 "86% 정렬됨"으로 들리고,
그것은 validity evidence만이 할 수 있는 주장을 이름이 대신 하는 것입니다.

cosine은 **rectified** 입니다 — `max(0, cos)`, 상한 1.0으로 clip. 이는 임계값이
아니라 *척도* 선택입니다: 음의 유사도는 정렬(alignment)로서 의미가 없어 0 바닥으로
모으고, 모든 지표를 공통 [0, 1] 범위에 둡니다. 두 텍스트가 "매칭되는지"를 정하는
상수는 존재하지 않습니다.

프로토콜의 기본 텍스트 인코더는 OpenAI 호환 `/v1/embeddings` 엔드포인트를 쓰는
`OpenAIAPIEncoder`(`nvidia/llama-embed-nemotron-8b`)이며, 순수 파이썬 character
n-gram TF-IDF(`TfidfCharNgramEncoder`)는 네트워크가 필요 없는 어휘적 하한선
(sensitivity 축)입니다. `SentenceTransformerEncoder`로도 교체 가능합니다.
Bloom 분류는 한/영 동사 lexicon 기반입니다(측정 과정에 LLM 호출 없음).

벤치마크 레벨 실행: 상위 저장소의 `scripts/alignmentgraph-isd-bench/06_score_alignment.py`가 run dir
전체를 채점해 시나리오별 `alignment_scores.json`을 남깁니다. 지표 정의
전문은 모노레포의 `docs/benchmark_guides.md` 부록 A.3 참조.

## 컨텍스트 기반 가중치 조정

시나리오의 맥락에 따라 ADDIE 단계별 가중치가 **동적으로 조정**됩니다.

### 지원되는 컨텍스트 요소

#### 학습자 특성 (Context Matrix Items 1-16)

| 요소 | 항목 | 가중치 조정 예시 |
|------|------|----------------|
| 연령대 | 1-4 | 10대: Development↑, 40대+: Analysis↑ |
| 학력수준 | 5-9 | 초등: Development↑, 성인: Analysis↑ |
| 도메인 전문성 | 10-12 | 초급: Development↑, 고급: Evaluation↑ |
| 직업/역할 | 13-16 | 학생: Design↑, 직장인: Implementation↑ |

#### 교육 도메인 (Context Matrix Items 23-32)

| 도메인 | 항목 | 가중치 조정 |
|--------|------|------------|
| 언어 | 23 | Implementation↑ (말하기/듣기 실습) |
| 수학 | 24 | Design↑ (개념 구조화) |
| 과학 | 25 | Development↑ (실험/시각화) |
| 사회 | 26 | Analysis↑, Design↑ (토론/협력) |
| 개발(Software/IT) | 27 | Implementation↑ (실습 환경) |
| AI | 28 | Implementation↑ (실습 환경) |
| 의료/간호 | 29 | Evaluation↑ (절차 정확성) |
| 경영/HR | 30 | Analysis↑, Evaluation↑ (성과 평가) |
| 교육(교수·학습) | 31 | Design↑ (교수법 설계) |
| 서비스/고객응대 | 32 | Implementation↑ (롤플레이) |

#### 전달 방식 (Context Matrix Items 33-39)

| 전달 방식 | 항목 | 가중치 조정 |
|----------|------|------------|
| 오프라인(교실) | 33 | Implementation↑ |
| 온라인 실시간 | 34 | Implementation↑ |
| 온라인 비실시간(LMS) | 35 | Development↑ |
| 블렌디드(혼합형) | 36 | Design↑ |
| 모바일 마이크로러닝 | 37 | Development↑ |
| 시뮬레이션/VR | 38 | Development↑↑ |
| 프로젝트 기반(PBL) | 39 | Design↑, Evaluation↑ |

#### 환경 요소 (Context Matrix Items 40-48)

| 요소 | 항목 | 가중치 조정 |
|------|------|------------|
| 소규모(1-10명) | 40 | Evaluation↑ (개별 피드백) |
| 중규모(10-30명) | 41 | Implementation↑ (소집단 활동) |
| 대규모(30명+) | 42 | Development↑ (표준화 자료) |
| 디지털 기기 제공 | 46 | Development↑ (멀티미디어) |
| BYOD | 47 | Design↑ (접근성 설계) |
| 제한적 기술 환경 | 48 | Development↑, Implementation↑ |

### 동적 평가 지침 생성

시나리오의 컨텍스트를 분석하여 맞춤형 평가 가이드라인이 자동 생성됩니다.
이를 통해 LLM 평가자가 시나리오 맥락에 적합한 평가를 수행합니다.

## 설치

```bash
pip install -e .
```

## 사용법

### 단일 평가

```bash
isd-evaluator evaluate \
  --output result.json \
  --scenario scenarios/idld_aligned/scenario_idld_0001.json \
  --trajectory traj.json
```

### 비교 평가

```bash
# 기존 결과 비교
isd-evaluator compare \
  --scenario scenario.json \
  --output-dir results/

# Agent 실행 후 비교
isd-evaluator compare \
  --scenario scenario.json \
  --output-dir results/ \
  --run
```

### 정보 출력

```bash
isd-evaluator info
```

## CLI 옵션

### evaluate 명령

| 옵션 | 설명 |
|------|------|
| `--output, -o` | 평가할 ADDIE 산출물 |
| `--scenario, -s` | 원본 시나리오 (선택) |
| `--trajectory, -t` | 궤적 파일 (선택) |
| `--result, -r` | 평가 결과 저장 경로 |
| `--use-llm/--no-llm` | LLM 평가 사용 여부 |

### compare 명령

| 옵션 | 설명 |
|------|------|
| `--scenario, -s` | 시나리오 파일 |
| `--output-dir, -d` | 결과 디렉토리 |
| `--agents, -a` | 평가할 Agent 목록 |
| `--run/--no-run` | Agent 실행 여부 |

## 환경 변수

Judge 모델 설정은 `JUDGE_MODEL_*` 환경 변수로 해석됩니다(멀티 judge 기본,
agent 모델과 분리 — self-preference bias 방지). 전체 매트릭스는 벤치마크
루트 `README.md` 참조:

```bash
export JUDGE_MODEL_PROVIDER=openrouter
export JUDGE_MODEL_NAMES=openai/gpt-4o-mini,google/gemini-2.5-flash-lite
export JUDGE_MODEL_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL_API_KEY_ENV=OPENROUTER_API_KEY
export OPENROUTER_API_KEY="your-api-key"
```

## 참조

- 지표 정의 전문 및 e2e 파이프라인: 모노레포 `docs/benchmark_guides.md`
  (부록 A: metrics reference)
- Bloom 분류기 학습: [`scripts/README.md`](scripts/README.md)
