"""Tests for the deterministic constructive-alignment metric.

Uses only the offline pieces: the TF-IDF char-ngram encoder and the
Korean/English Bloom verb lexicon. No network, no LLM.
"""

from isd_evaluator.metrics.alignment import (
    AlignmentEvaluator,
    LexiconBloomClassifier,
    TfidfCharNgramEncoder,
    extract_activities,
    extract_assessment_items,
    extract_objectives,
    normalize_declared_level,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

SCENARIO = {
    "scenario_id": "TEST-001",
    "title": "파이썬 프로그래밍 기초",
    "domain": "Programming",
    "learning_goals": [
        "변수와 자료형의 개념을 설명할 수 있다",
        "조건문과 반복문을 활용하여 프로그램을 구현할 수 있다",
        "함수 설계 원칙을 적용하여 모듈화된 코드를 작성할 수 있다",
    ],
}

# Perfectly aligned: every objective has a same-topic assessment item at the
# same-or-higher Bloom level, plus matching activities and evaluation.
ALIGNED_OUTPUT = {
    "design": {
        "learning_objectives": [
            {"id": "LO-001", "level": "이해",
             "statement": "변수와 자료형의 개념과 차이를 설명한다"},
            {"id": "LO-002", "level": "적용",
             "statement": "조건문과 반복문을 활용하여 간단한 프로그램을 구현한다"},
            {"id": "LO-003", "level": "창조",
             "statement": "함수 설계 원칙을 적용하여 모듈화된 코드를 작성한다"},
        ],
        "instructional_strategy": {
            "sequence": [
                {"event": "개념 학습", "activity": "변수와 자료형 개념을 설명하는 강의"},
                {"event": "실습", "activity": "조건문과 반복문을 활용한 프로그램 구현 실습"},
                {"event": "프로젝트", "activity": "함수 설계 원칙으로 모듈화된 코드를 작성하는 프로젝트"},
            ]
        },
    },
    "evaluation": {
        "quiz_items": [
            {"id": "Q-001", "type": "서술형",
             "question": "변수와 자료형의 개념과 차이를 설명하시오"},
            {"id": "Q-002", "type": "주관식",
             "question": "조건문과 반복문을 활용하여 짝수 합을 구하는 프로그램을 구현하시오"},
            {"id": "Q-003", "type": "서술형",
             "question": "함수 설계 원칙을 적용하여 모듈화된 코드를 작성하시오"},
        ],
        "rubric": {
            "criteria": [
                "변수와 자료형 개념 설명의 정확성",
                "조건문과 반복문 구현의 완성도",
                "함수 설계와 모듈화 코드 작성 수준",
            ]
        },
    },
}

# Misaligned: items drift to an unrelated topic and sit at a lower Bloom level
# than the objectives; one objective is never assessed.
MISALIGNED_OUTPUT = {
    "design": {
        "learning_objectives": [
            {"id": "LO-001", "level": "이해",
             "statement": "변수와 자료형의 개념과 차이를 설명한다"},
            {"id": "LO-002", "level": "적용",
             "statement": "조건문과 반복문을 활용하여 간단한 프로그램을 구현한다"},
            {"id": "LO-003", "level": "창조",
             "statement": "함수 설계 원칙을 적용하여 모듈화된 코드를 작성한다"},
        ],
        "instructional_strategy": {
            "sequence": [
                {"event": "강의", "activity": "세계 커피 문화의 역사를 나열하는 강의"},
            ]
        },
    },
    "evaluation": {
        "quiz_items": [
            {"id": "Q-001", "type": "단답형",
             "question": "커피 원두 품종의 이름을 나열하시오"},
            {"id": "Q-002", "type": "OX",
             "question": "에스프레소 추출 온도를 기억하여 진술하시오"},
        ],
        "rubric": {"criteria": ["커피 지식 암기의 정확성"]},
    },
}


def evaluate(output, scenario=SCENARIO):
    return AlignmentEvaluator().evaluate(output, scenario)


def assert_unit_interval(score):
    values = [
        score.porter_mean,
        score.webb_range,
        score.webb_bloom_consistency,
        score.embedding_coverage,
        score.embedding_precision,
        score.composite,
        *score.porter.values(),
    ]
    for value in values:
        if value is not None:
            assert 0.0 <= value <= 1.0, f"out of [0,1]: {value}"


# ---------------------------------------------------------------------------
# End-to-end contrast: aligned must clearly beat misaligned
# ---------------------------------------------------------------------------


class TestAlignedVsMisaligned:
    def test_aligned_scores_higher(self):
        aligned = evaluate(ALIGNED_OUTPUT)
        misaligned = evaluate(MISALIGNED_OUTPUT)
        assert aligned.composite > misaligned.composite + 0.15

    def test_all_values_in_unit_interval(self):
        assert_unit_interval(evaluate(ALIGNED_OUTPUT))
        assert_unit_interval(evaluate(MISALIGNED_OUTPUT))

    def test_webb_range_full_when_every_objective_assessed(self):
        aligned = evaluate(ALIGNED_OUTPUT)
        assert aligned.webb_range == 1.0

    def test_webb_range_drops_for_off_topic_items(self):
        misaligned = evaluate(MISALIGNED_OUTPUT)
        assert misaligned.webb_range < 1.0

    def test_bloom_consistency_contrast(self):
        aligned = evaluate(ALIGNED_OUTPUT)
        misaligned = evaluate(MISALIGNED_OUTPUT)
        # Aligned items match or exceed objective levels; misaligned items are
        # Remember-level against Understand/Apply/Create objectives.
        assert aligned.webb_bloom_consistency == 1.0
        assert misaligned.webb_bloom_consistency is not None
        assert misaligned.webb_bloom_consistency < aligned.webb_bloom_consistency

    def test_coverage_contrast(self):
        aligned = evaluate(ALIGNED_OUTPUT)
        misaligned = evaluate(MISALIGNED_OUTPUT)
        assert aligned.embedding_coverage > misaligned.embedding_coverage

    def test_porter_contrast(self):
        aligned = evaluate(ALIGNED_OUTPUT)
        misaligned = evaluate(MISALIGNED_OUTPUT)
        assert aligned.porter_mean is not None
        assert misaligned.porter_mean is not None
        assert aligned.porter_mean > misaligned.porter_mean

    def test_details_are_explainable(self):
        aligned = evaluate(ALIGNED_OUTPUT)
        per_objective = aligned.details["objectives"]
        assert len(per_objective) == 3
        for entry in per_objective:
            assert entry["best_item"] is not None
            assert entry["matched"] is True
            assert entry["best_similarity"] > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_output(self):
        score = evaluate({})
        assert score.composite == 0.0
        assert score.counts["objectives"] == 0
        assert any("no learning objectives" in note for note in score.notes)

    def test_no_objectives(self):
        score = evaluate({"evaluation": ALIGNED_OUTPUT["evaluation"]})
        assert score.composite == 0.0
        assert_unit_interval(score)

    def test_objectives_but_no_items(self):
        output = {"design": {"learning_objectives":
                             ALIGNED_OUTPUT["design"]["learning_objectives"]}}
        score = evaluate(output)
        assert score.webb_range == 0.0
        assert score.embedding_coverage == 0.0
        assert score.porter["assessment"] == 0.0
        assert_unit_interval(score)

    def test_no_scenario_falls_back_to_single_topic(self):
        score = AlignmentEvaluator().evaluate(ALIGNED_OUTPUT, None)
        assert score.counts["topics"] == 1
        assert_unit_interval(score)

    def test_wrapped_output_is_unwrapped(self):
        wrapped = {"addie_output": ALIGNED_OUTPUT}
        direct = evaluate(ALIGNED_OUTPUT)
        via_wrapper = evaluate(wrapped)
        assert via_wrapper.composite == direct.composite


# ---------------------------------------------------------------------------
# Bloom lexicon classifier
# ---------------------------------------------------------------------------


class TestLexiconBloomClassifier:
    def setup_method(self):
        self.clf = LexiconBloomClassifier()

    def test_korean_levels(self):
        assert self.clf.classify("주요 개념을 나열할 수 있다") == 1
        assert self.clf.classify("개념의 차이를 설명한다") == 2
        assert self.clf.classify("배운 내용을 실무에 적용한다") == 3
        assert self.clf.classify("두 사례를 비교하고 분석한다") == 4
        assert self.clf.classify("결과의 타당성을 평가한다") == 5
        assert self.clf.classify("새로운 수업 자료를 설계한다") == 6

    def test_english_levels(self):
        assert self.clf.classify("List the main components of a computer") == 1
        assert self.clf.classify("Explain how photosynthesis works") == 2
        assert self.clf.classify("Apply the formula to solve the problem") == 3
        assert self.clf.classify("Compare and contrast the two approaches") == 4
        assert self.clf.classify("Evaluate the validity of the results") == 5
        assert self.clf.classify("Design a new lesson module") == 6

    def test_highest_level_wins(self):
        # Contains both 설명 (2) and 설계 (6): highest cognitive demand wins.
        assert self.clf.classify("설계 원칙을 설명하고 새 모듈을 설계한다") == 6

    def test_noun_usage_does_not_fire(self):
        # 교수설계 / 사용자 are noun usages; no verbal conjugation follows.
        assert self.clf.classify("교수설계 문서") is None
        assert self.clf.classify("사용자 목록") is None

    def test_unclassifiable_returns_none(self):
        assert self.clf.classify("") is None
        assert self.clf.classify("커피와 원두") is None

    def test_recall_question_fallback(self):
        assert self.clf.classify("다음 중 올바른 것은 무엇인가?") == 1

    def test_declared_level_normalization(self):
        assert normalize_declared_level("이해") == 2
        assert normalize_declared_level("Understand") == 2
        assert normalize_declared_level("Remember") == 1
        assert normalize_declared_level("nonsense") is None
        assert normalize_declared_level(None) is None


# ---------------------------------------------------------------------------
# TF-IDF fallback encoder
# ---------------------------------------------------------------------------


class TestTfidfEncoder:
    def test_identical_texts_similar(self):
        from isd_evaluator.metrics.alignment import _cosine

        vectors = TfidfCharNgramEncoder().encode(
            ["변수와 자료형을 설명한다", "변수와 자료형을 설명한다", "커피 원두의 역사"]
        )
        assert _cosine(vectors[0], vectors[1]) > 0.99
        assert _cosine(vectors[0], vectors[2]) < 0.2

    def test_empty_text_yields_zero_vector(self):
        vectors = TfidfCharNgramEncoder().encode(["", "무언가"])
        assert all(v == 0.0 for v in vectors[0])


# ---------------------------------------------------------------------------
# Extraction robustness (schema layout AND observed real-run layout)
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_real_run_layout(self):
        # Layout observed in actual benchmark runs: assessment items in
        # development.assessment_tools / evaluation.summative.assessment_tools,
        # activities in design.instructional_strategies + learning_activities.
        output = {
            "design": {
                "learning_objectives": [
                    {"id": "LO-01", "level": "Apply",
                     "statement": "Apply diagnostic survey tools to a cohort."}
                ],
                "instructional_strategies": {
                    "sequence": [{"event": "Warm-up", "activity": "Group discussion"}]
                },
                "learning_activities": [
                    {"activity_name": "Survey lab", "description": "Hands-on survey design"}
                ],
            },
            "development": {
                "assessment_tools": [
                    {"item_id": "A-01", "question": "Apply the survey tool to the sample."}
                ]
            },
            "evaluation": {
                "summative": {
                    "assessment_tools": [
                        {"item_id": "S-01", "question": "Analyze the collected results."}
                    ]
                }
            },
        }
        objectives = extract_objectives(output)
        items = extract_assessment_items(output)
        activities = extract_activities(output)
        assert len(objectives) == 1
        assert len(items) == 2
        assert len(activities) == 2
        score = AlignmentEvaluator().evaluate(output, SCENARIO)
        assert_unit_interval(score)
        assert score.counts["assessment"] == 2

    def test_assessment_plan_fallback_only_without_items(self):
        output = {
            "design": {
                "learning_objectives": [{"id": "LO-01", "statement": "x를 설명한다"}],
                "assessment_plan": {"formative": ["퀴즈로 x 개념 확인"]},
            }
        }
        assert extract_assessment_items(output) == ["퀴즈로 x 개념 확인"]

    def test_sanity_report_agreement(self):
        score = evaluate(ALIGNED_OUTPUT)
        sanity = score.details["sanity"]
        assert sanity["n_objectives"] == 3
        assert sanity["n_declared"] == 3
        # Statements were written so lexicon level == declared level.
        assert sanity["lexicon_vs_declared_agreement"] == 1.0
