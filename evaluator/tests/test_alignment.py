"""Tests for the deterministic constructive-alignment metric.

Uses only the offline pieces: the TF-IDF char-ngram encoder and the
Korean/English Bloom verb lexicon. No network, no LLM.
"""

import hashlib

import pytest

from isd_evaluator.metrics.alignment import (
    AlignmentEvaluator,
    LexiconBloomClassifier,
    OpenAIAPIEncoder,
    TfidfCharNgramEncoder,
    _EmbeddingCache,
    extract_activities,
    extract_assessment_items,
    extract_declared_links,
    extract_evaluation_texts,
    extract_objectives,
    normalize_declared_level,
    reaggregate,
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
        score.objective_assessment_alignment,
        score.objective_activity_alignment,
        score.objective_evaluation_alignment,
        score.objective_cognitive_congruence,
        score.porter_mean,
        score.webb_bloom_consistency,
        score.assessment_precision,
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
        # Primary endpoint: continuous assessment alignment (mean-max cosine).
        aligned = evaluate(ALIGNED_OUTPUT)
        misaligned = evaluate(MISALIGNED_OUTPUT)
        assert (aligned.objective_assessment_alignment
                > misaligned.objective_assessment_alignment)

    def test_all_values_in_unit_interval(self):
        assert_unit_interval(evaluate(ALIGNED_OUTPUT))
        assert_unit_interval(evaluate(MISALIGNED_OUTPUT))

    def test_bloom_consistency_contrast(self):
        aligned = evaluate(ALIGNED_OUTPUT)
        misaligned = evaluate(MISALIGNED_OUTPUT)
        # Aligned items match or exceed objective levels; misaligned items are
        # Remember-level against Understand/Apply/Create objectives.
        assert aligned.webb_bloom_consistency == 1.0
        assert misaligned.webb_bloom_consistency is not None
        assert misaligned.webb_bloom_consistency < aligned.webb_bloom_consistency

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
            assert entry["best_similarity"] > 0
            assert entry["best_activity"] is not None
            assert entry["best_activity_similarity"] > 0
            assert entry["best_evaluation"] is not None
            assert entry["cognitively_congruent"] is True
            # No threshold-based booleans anymore.
            assert "matched" not in entry
            assert "fully_aligned" not in entry


# ---------------------------------------------------------------------------
# Objective-level endpoints (continuous, threshold-free)
# ---------------------------------------------------------------------------


class TestObjectiveEndpoints:
    def test_no_threshold_attribute(self):
        # The evaluator must carry no threshold at all.
        assert not hasattr(AlignmentEvaluator(), "match_threshold")

    def test_reaggregate_reproduces_endpoints(self):
        # Re-aggregating a saved score from its stored per-objective signals
        # must reproduce the endpoints exactly — no re-encoding needed for an
        # objective-level metric-definition change.
        score = evaluate(ALIGNED_OUTPUT).to_dict()
        expected = {k: score[k] for k in (
            "objective_assessment_alignment", "objective_activity_alignment",
            "objective_evaluation_alignment", "objective_cognitive_congruence")}
        # Corrupt the top-level endpoints; reaggregate must restore them.
        for k in expected:
            score[k] = -1.0
        score["objective_measurability"] = 0.5  # a since-removed criterion
        reaggregate(score)
        for k, v in expected.items():
            assert score[k] == v
        assert "objective_measurability" not in score  # stale key dropped

    def test_alignment_endpoints_are_continuous_means(self):
        # Each endpoint equals the mean over objectives of the recorded
        # best-match similarity — no thresholding.
        score = evaluate(ALIGNED_OUTPUT)
        objs = score.details["objectives"]
        exp_asm = sum(o["best_similarity"] for o in objs) / len(objs)
        exp_act = sum(o["best_activity_similarity"] for o in objs) / len(objs)
        assert abs(score.objective_assessment_alignment - exp_asm) < 1e-9
        assert abs(score.objective_activity_alignment - exp_act) < 1e-9
        # Continuous: not pinned to 0/1.
        assert 0.0 < score.objective_assessment_alignment < 1.0

    def test_assessment_alignment_contrast(self):
        aligned = evaluate(ALIGNED_OUTPUT)
        misaligned = evaluate(MISALIGNED_OUTPUT)
        assert (aligned.objective_assessment_alignment
                > misaligned.objective_assessment_alignment)
        assert (aligned.objective_activity_alignment
                > misaligned.objective_activity_alignment)

    def test_cognitive_congruence_contrast(self):
        aligned = evaluate(ALIGNED_OUTPUT)
        misaligned = evaluate(MISALIGNED_OUTPUT)
        # Misaligned items sit at Remember level against higher objectives.
        assert aligned.objective_cognitive_congruence == 1.0
        congruence = misaligned.objective_cognitive_congruence
        assert congruence is None or congruence < 1.0

    def test_no_activities_zeroes_activity_alignment(self):
        output = {
            "design": {
                "learning_objectives":
                    ALIGNED_OUTPUT["design"]["learning_objectives"]
            },
            "evaluation": ALIGNED_OUTPUT["evaluation"],
        }
        score = evaluate(output)
        assert score.objective_activity_alignment == 0.0
        assert any("no activities" in note for note in score.notes)

    def test_no_evaluation_zeroes_evaluation_alignment(self):
        output = {
            "design": ALIGNED_OUTPUT["design"],
            "development": {
                "assessment_tools": [
                    {"item_id": f"A-{i}", "question": q["question"]}
                    for i, q in enumerate(ALIGNED_OUTPUT["evaluation"]["quiz_items"])
                ]
            },
        }
        score = evaluate(output)
        assert score.objective_assessment_alignment > 0.0
        assert score.objective_evaluation_alignment == 0.0
        assert any("no evaluation texts" in note for note in score.notes)


# ---------------------------------------------------------------------------
# Language and validation-evidence layers
# ---------------------------------------------------------------------------


class TestLanguageAndValidation:
    def test_declared_link_agreement(self):
        output = {
            "design": ALIGNED_OUTPUT["design"],
            "development": {
                "assessment_tools": [
                    {"item_id": "A-001", "aligned_objective": "LO-001",
                     "question": "변수와 자료형의 개념과 차이를 설명하시오"},
                    {"item_id": "A-002", "aligned_objective": "LO-002",
                     "question": "조건문과 반복문을 활용하여 짝수 합을 구하는 프로그램을 구현하시오"},
                ]
            },
            "evaluation": ALIGNED_OUTPUT["evaluation"],
        }
        assert len(extract_declared_links(output)) == 2
        score = evaluate(output)
        validation = score.details["validation"]
        assert validation["declared_links_checked"] == 2
        assert validation["declared_link_agreement"] == 1.0

    def test_no_declared_links_is_undefined(self):
        score = evaluate(ALIGNED_OUTPUT)
        assert score.details["validation"]["declared_link_agreement"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_output(self):
        score = evaluate({})
        assert score.objective_assessment_alignment == 0.0
        assert score.counts["objectives"] == 0
        assert any("no learning objectives" in note for note in score.notes)

    def test_no_objectives(self):
        score = evaluate({"evaluation": ALIGNED_OUTPUT["evaluation"]})
        assert score.objective_assessment_alignment == 0.0
        assert score.objective_activity_alignment == 0.0
        assert score.objective_evaluation_alignment == 0.0
        assert_unit_interval(score)

    def test_objectives_but_no_items(self):
        output = {"design": {"learning_objectives":
                             ALIGNED_OUTPUT["design"]["learning_objectives"]}}
        score = evaluate(output)
        assert score.objective_assessment_alignment == 0.0
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
        assert (via_wrapper.objective_assessment_alignment
                == direct.objective_assessment_alignment)


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

    def test_assessment_plan_object_layout_observed_in_baselines(self):
        output = {
            "design": {
                "assessment_plan": {
                    "diagnostic": [
                        {
                            "method": "Pre-course EMR evaluation",
                            "description": "Assess prior navigation knowledge.",
                        }
                    ],
                    "formative": [
                        {
                            "description": "Review the draft protocol with peers.",
                        }
                    ],
                    "summative": [
                        {
                            "summative1": "Present the final learning portfolio.",
                            "summative2": "Defend the selected design decisions.",
                        }
                    ],
                    "assessment_criteria": [
                        "Accuracy of the protocol and supporting rationale."
                    ],
                }
            }
        }
        assert extract_assessment_items(output) == [
            "Pre-course EMR evaluation Assess prior navigation knowledge.",
            "Review the draft protocol with peers.",
            "Present the final learning portfolio. "
            "Defend the selected design decisions.",
            "Accuracy of the protocol and supporting rationale.",
        ]

    def test_strategy_activities_layout_observed_in_agents(self):
        output = {
            "design": {
                "instructional_strategies": {
                    "methods": ["Case study", "Peer review"],
                    "activities": [
                        "Analyze a failed project.",
                        "Review a peer's protocol.",
                    ],
                    "rationale": "Gagné's 9 Events",
                }
            }
        }
        assert extract_activities(output) == [
            "Analyze a failed project.",
            "Review a peer's protocol.",
        ]

    def test_pilot_data_collection_layout_observed_in_baselines(self):
        output = {
            "evaluation": {
                "pilot_data_collection": {
                    "collection_methods": [
                        {
                            "method": "Test/quiz",
                            "timing": "Before and after training",
                            "tool": "LMS",
                        }
                    ],
                    "instruments": [
                        {"name": "Pre-post test", "type": "Knowledge assessment"}
                    ],
                    "data_types": {
                        "quantitative": [
                            {
                                "type": "Post-test scores",
                                "purpose": "Learning outcome measurement",
                                "source": "Learners",
                            }
                        ]
                    },
                    "timeline": [
                        {
                            "phase": "Immediately after",
                            "activities": ["Post-test", "Satisfaction survey"],
                        }
                    ],
                    "data_management": {
                        "storage": "Secure storage",
                        "retention_period": "3 years",
                    },
                }
            }
        }
        assert extract_evaluation_texts(output) == [
            "Test/quiz Before and after training LMS",
            "Pre-post test Knowledge assessment",
            "Post-test scores Learning outcome measurement Learners",
            "Post-test",
            "Satisfaction survey",
        ]

    def test_sanity_report_agreement(self):
        score = evaluate(ALIGNED_OUTPUT)
        sanity = score.details["sanity"]
        assert sanity["n_objectives"] == 3
        assert sanity["n_declared"] == 3
        # Statements were written so lexicon level == declared level.
        assert sanity["lexicon_vs_declared_agreement"] == 1.0


# ---------------------------------------------------------------------------
# Default-configuration guards (library vs scorer script)
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_library_defaults_are_offline(self):
        evaluator = AlignmentEvaluator()
        assert isinstance(evaluator.encoder, TfidfCharNgramEncoder)
        assert isinstance(evaluator.bloom, LexiconBloomClassifier)
        assert not hasattr(evaluator, "match_threshold")  # threshold-free

    def test_scorer_defaults_match_protocol(self):
        import importlib.util
        import sys
        from pathlib import Path

        script = (Path(__file__).resolve().parents[2] / "scripts"
                  / "alignmentgraph-isd-bench" / "06_score_alignment.py")
        spec = importlib.util.spec_from_file_location("score_alignment", script)
        module = importlib.util.module_from_spec(spec)
        # Register before exec: @dataclass resolves annotations through
        # sys.modules[cls.__module__], which is None for an unregistered module.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        assert module.DEFAULT_EMBED_MODEL == "nvidia/llama-embed-nemotron-8b"
        assert "composite" not in module.COMPONENTS
        assert module.COMPONENTS[0] == "objective_assessment_alignment"
        assert "objective_evaluation_alignment" in module.COMPONENTS
        # the primary encoder owns the unsuffixed artifact pooling reads
        assert module.ENCODER_PRESETS[module.PRIMARY_ENCODER][2] == ""


# ---------------------------------------------------------------------------
# OpenAI-compatible API encoder (fake transport — no network)
# ---------------------------------------------------------------------------


def _fake_api_encoder(tmp_path, calls, base_urls="http://localhost:1/v1",
                      revision=None):
    encoder = OpenAIAPIEncoder(
        base_urls=base_urls,
        model="fake-embed",
        cache_path=tmp_path / "embed_cache.sqlite",
        batch_size=2,
        revision=revision,
    )

    def fake_embed(texts, client=None):
        calls.append((client, list(texts)))
        return [[float(len(t)), 1.0, 0.5] for t in texts]

    encoder._embed_batch = fake_embed
    return encoder


class TestOpenAIAPIEncoder:
    def test_batching_and_shapes(self, tmp_path):
        calls = []
        encoder = _fake_api_encoder(tmp_path, calls)
        vectors = encoder.encode(["a", "bb", "ccc"])
        assert [len(v) for v in vectors] == [3, 3, 3]
        assert vectors[1][0] == 2.0
        assert len(calls) == 2  # batch_size=2 -> two API batches

    def test_empty_text_zero_vector_without_call(self, tmp_path):
        calls = []
        encoder = _fake_api_encoder(tmp_path, calls)
        vectors = encoder.encode(["hello", "   "])
        assert all(x == 0.0 for x in vectors[1])
        assert [texts for _, texts in calls] == [["hello"]]

    def test_cache_hit_skips_api(self, tmp_path):
        calls = []
        encoder = _fake_api_encoder(tmp_path, calls)
        first = encoder.encode(["same text", "other"])
        assert len(calls) == 1
        second = encoder.encode(["same text", "other"])
        assert len(calls) == 1  # everything served from cache
        assert first == second

    def test_multi_endpoint_load_balances(self, tmp_path):
        calls = []
        encoder = _fake_api_encoder(
            tmp_path, calls,
            base_urls=["http://pod-a/v1", "http://pod-b/v1"],
        )
        # 6 texts / batch_size 2 = 3 batches across 2 endpoints (round-robin).
        vectors = encoder.encode(["aa", "bb", "cc", "dd", "ee", "ff"])
        assert [len(v) for v in vectors] == [3] * 6
        assert len(encoder._clients) == 2
        clients_used = {id(c) for c, _ in calls}
        assert len(clients_used) == 2  # both pods received batches
        # every text embedded exactly once, order preserved
        assert sorted(t for _, ts in calls for t in ts) == \
            ["aa", "bb", "cc", "dd", "ee", "ff"]

    def test_single_url_string_still_accepted(self, tmp_path):
        encoder = _fake_api_encoder(tmp_path, [], base_urls="http://one/v1")
        assert len(encoder._clients) == 1

    def test_cache_persists_across_instances(self, tmp_path):
        calls = []
        encoder = _fake_api_encoder(tmp_path, calls)
        first = encoder.encode(["persisted"])
        fresh = OpenAIAPIEncoder(
            base_urls="http://localhost:1/v1",
            model="fake-embed",
            cache_path=tmp_path / "embed_cache.sqlite",
        )

        def boom(texts, client=None):  # pragma: no cover - must not be reached
            raise AssertionError("cache miss hit the API")

        fresh._embed_batch = boom
        assert fresh.encode(["persisted"]) == first

    def test_revision_is_part_of_the_cache_key(self, tmp_path):
        """A revision bump must re-embed, not serve the old weights' vectors.

        The model id does not pin weights, so without the revision in the key
        a re-score after an upstream update would silently reuse the previous
        vectors while the score file claims the new revision.
        """
        old_calls, new_calls = [], []
        old = _fake_api_encoder(tmp_path, old_calls, revision="rev-a")
        old.encode(["shared text"])
        assert len(old_calls) == 1

        new = _fake_api_encoder(tmp_path, new_calls, revision="rev-b")
        new.encode(["shared text"])
        assert len(new_calls) == 1, "rev-b wrongly served rev-a's cached vector"

        again = _fake_api_encoder(tmp_path, [], revision="rev-a")

        def boom(texts, client=None):  # pragma: no cover - must not be reached
            raise AssertionError("rev-a lost its own cache entry")

        again._embed_batch = boom
        again.encode(["shared text"])

    def test_unpinned_revision_keeps_the_legacy_cache_key(self, tmp_path):
        """Existing caches (written before revisions existed) stay valid."""
        legacy_key = hashlib.sha1(b"fake-embed\x00shared text").hexdigest()
        assert _EmbeddingCache.key("fake-embed", "shared text") == legacy_key
        assert _EmbeddingCache.key("fake-embed", "shared text", None) == legacy_key

    def test_cache_only_refuses_to_call_the_endpoint(self, tmp_path):
        calls = []
        warm = _fake_api_encoder(tmp_path, calls)
        warm.encode(["already cached"])

        offline = OpenAIAPIEncoder(
            base_urls="http://embed-cache-only.invalid/v1",
            model="fake-embed",
            cache_path=tmp_path / "embed_cache.sqlite",
            cache_only=True,
        )

        def boom(texts, client=None):  # pragma: no cover - must not be reached
            raise AssertionError("cache_only still called the API")

        offline._embed_batch = boom
        assert offline.encode(["already cached"]) == warm.encode(["already cached"])
        with pytest.raises(RuntimeError, match="not in the embedding cache"):
            offline.encode(["never seen before"])
