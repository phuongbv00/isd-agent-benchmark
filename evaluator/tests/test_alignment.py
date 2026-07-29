"""Tests for the deterministic constructive-alignment metric.

Uses only the offline pieces: the TF-IDF char-ngram encoder and the English
Bloom verb lexicon. No network, no LLM. Fixtures are English because the
scored corpus is (measured 100% across agents and model sizes).
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
    "title": "Python Programming Basics",
    "domain": "Programming",
    "learning_goals": [
        "Explain the concepts of variables and data types",
        "Implement programs using conditionals and loops",
        "Compose modular code applying function design principles",
    ],
}

# Perfectly aligned: every objective has a same-topic assessment item at the
# same-or-higher Bloom level, plus matching activities and evaluation.
# Statements are worded so the lexicon level equals the declared level.
ALIGNED_OUTPUT = {
    "design": {
        "learning_objectives": [
            {"id": "LO-001", "level": "Understand",
             "statement": "Explain the concepts and differences of variables and data types"},
            {"id": "LO-002", "level": "Apply",
             "statement": "Implement a simple program using conditionals and loops"},
            {"id": "LO-003", "level": "Create",
             "statement": "Compose modular code applying function design principles"},
        ],
        "instructional_strategy": {
            "sequence": [
                {"event": "Concept lesson",
                 "activity": "Lecture explaining the concepts of variables and data types"},
                {"event": "Lab",
                 "activity": "Hands-on practice implementing programs with conditionals and loops"},
                {"event": "Project",
                 "activity": "Project composing modular code with function design principles"},
            ]
        },
    },
    "evaluation": {
        "quiz_items": [
            {"id": "Q-001", "type": "short_answer",
             "question": "Explain the concepts and differences of variables and data types"},
            {"id": "Q-002", "type": "coding",
             "question": "Implement a program summing even numbers using conditionals and loops"},
            {"id": "Q-003", "type": "project",
             "question": "Compose modular code applying function design principles"},
        ],
        "rubric": {
            "criteria": [
                "Accuracy of explaining variable and data type concepts",
                "Completeness of the conditional and loop implementation",
                "Quality of function design and modular code composition",
            ]
        },
    },
}

# Misaligned: items drift to an unrelated topic and sit at a lower Bloom level
# than the objectives; one objective is never assessed.
MISALIGNED_OUTPUT = {
    "design": {
        "learning_objectives": ALIGNED_OUTPUT["design"]["learning_objectives"],
        "instructional_strategy": {
            "sequence": [
                {"event": "Lecture",
                 "activity": "Lecture listing the history of world coffee culture"},
            ]
        },
    },
    "evaluation": {
        "quiz_items": [
            {"id": "Q-001", "type": "short_answer",
             "question": "List the names of coffee bean varieties"},
            {"id": "Q-002", "type": "true_false",
             "question": "Recall and state the espresso extraction temperature"},
        ],
        "rubric": {"criteria": ["Accuracy of memorizing coffee knowledge"]},
    },
}


def evaluate(output, scenario=SCENARIO):
    return AlignmentEvaluator().evaluate(output, scenario)


def assert_unit_interval(score):
    values = [
        score.objective_assessment_similarity,
        score.objective_activity_similarity,
        score.objective_evaluation_similarity,
        score.objective_cognitive_congruence,
        score.porter_mean,
        score.webb_bloom_consistency,
        score.assessment_objective_similarity,
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
        assert (aligned.objective_assessment_similarity
                > misaligned.objective_assessment_similarity)

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
            "objective_assessment_similarity", "objective_activity_similarity",
            "objective_evaluation_similarity", "objective_cognitive_congruence")}
        # Corrupt the top-level signals; reaggregate must restore them.
        for k in expected:
            score[k] = -1.0
        reaggregate(score)
        for k, v in expected.items():
            assert score[k] == v

    def test_alignment_endpoints_are_continuous_means(self):
        # Each endpoint equals the mean over objectives of the recorded
        # best-match similarity — no thresholding.
        score = evaluate(ALIGNED_OUTPUT)
        objs = score.details["objectives"]
        exp_asm = sum(o["best_similarity"] for o in objs) / len(objs)
        exp_act = sum(o["best_activity_similarity"] for o in objs) / len(objs)
        assert abs(score.objective_assessment_similarity - exp_asm) < 1e-9
        assert abs(score.objective_activity_similarity - exp_act) < 1e-9
        # Continuous: not pinned to 0/1.
        assert 0.0 < score.objective_assessment_similarity < 1.0

    def test_assessment_alignment_contrast(self):
        aligned = evaluate(ALIGNED_OUTPUT)
        misaligned = evaluate(MISALIGNED_OUTPUT)
        assert (aligned.objective_assessment_similarity
                > misaligned.objective_assessment_similarity)
        assert (aligned.objective_activity_similarity
                > misaligned.objective_activity_similarity)

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
        assert score.objective_activity_similarity == 0.0
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
        assert score.objective_assessment_similarity > 0.0
        assert score.objective_evaluation_similarity == 0.0
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
                     "question": "Explain the concepts and differences of variables and data types"},
                    {"item_id": "A-002", "aligned_objective": "LO-002",
                     "question": "Implement a program summing even numbers using conditionals and loops"},
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
        assert score.objective_assessment_similarity == 0.0
        assert score.counts["objectives"] == 0
        assert any("no learning objectives" in note for note in score.notes)

    def test_no_objectives(self):
        score = evaluate({"evaluation": ALIGNED_OUTPUT["evaluation"]})
        assert score.objective_assessment_similarity == 0.0
        assert score.objective_activity_similarity == 0.0
        assert score.objective_evaluation_similarity == 0.0
        assert_unit_interval(score)

    def test_objectives_but_no_items(self):
        output = {"design": {"learning_objectives":
                             ALIGNED_OUTPUT["design"]["learning_objectives"]}}
        score = evaluate(output)
        assert score.objective_assessment_similarity == 0.0
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
        assert (via_wrapper.objective_assessment_similarity
                == direct.objective_assessment_similarity)


# ---------------------------------------------------------------------------
# Bloom lexicon classifier
# ---------------------------------------------------------------------------


class TestLexiconBloomClassifier:
    def setup_method(self):
        self.clf = LexiconBloomClassifier()

    def test_non_english_returns_none(self):
        # English-only by design: other scripts are unclassifiable, not guessed.
        assert self.clf.classify("개념의 차이를 설명한다") is None
        assert self.clf.classify("Giải thích sự khác biệt giữa các khái niệm") is None

    def test_english_levels(self):
        assert self.clf.classify("List the main components of a computer") == 1
        assert self.clf.classify("Explain how photosynthesis works") == 2
        assert self.clf.classify("Apply the formula to solve the problem") == 3
        assert self.clf.classify("Differentiate relevant from irrelevant data") == 4
        assert self.clf.classify("Evaluate the validity of the results") == 5
        assert self.clf.classify("Design a new lesson module") == 6

    def test_level_follows_anderson_krathwohl_not_folk_verb_lists(self):
        # Comparing/contrasting/matching and categorizing are Understand
        # processes in A&K Table 5.1, though folk verb handouts put them at
        # Analyze; integrating/outlining/structuring are Analyze (alternative
        # names of organizing), not Create.
        assert self.clf.classify("Compare and contrast the two approaches") == 2
        assert self.clf.classify("Categorize the samples by type") == 2
        assert self.clf.classify("Match each term to its definition") == 2
        assert self.clf.classify("Integrate the findings into an outline") == 4
        assert self.clf.classify("Check the solution for errors") == 5

    def test_every_verb_has_exactly_one_level(self):
        # "Highest level wins" must be decided by the text, never by a verb
        # accidentally listed at two levels.
        from isd_evaluator.metrics.alignment import _EN_VERB_LEXICON

        seen: dict[str, int] = {}
        for level, verbs in _EN_VERB_LEXICON.items():
            for verb in verbs:
                assert verb not in seen, f"{verb} at {seen.get(verb)} and {level}"
                seen[verb] = level

    def test_noun_usage_does_not_fire(self):
        # Determiner/possessive immediately before => noun or participial
        # adjective, not the performed action.
        assert self.clf.classify("The proposed solution must be reviewed") is None
        assert self.clf.classify("Learners submit a plan") is None
        assert self.clf.classify("Review of the design") is None
        # ... but a genuine verb after a determiner+subject still fires.
        assert self.clf.classify("The learners design a rubric") == 6

    def test_leading_verb_decides_not_the_most_demanding_one(self):
        # Webb: the level follows the CENTRAL performance. A trailing verb names
        # the purpose or the medium, not the demand being assessed.
        assert self.clf.classify(
            "Explain the design principles and design a new module") == 2
        assert self.clf.classify(
            "Apply metaverse tools to create a 3D model") == 3
        assert self.clf.classify(
            "Evaluate the effectiveness of the lesson plan") == 5
        # ... and a genuine Create objective still reads as Create.
        assert self.clf.classify("Design a rubric for peer assessment") == 6

    def test_word_boundary_is_respected(self):
        # Substrings of longer words must not fire the verb pattern.
        assert self.clf.classify("The username field is required") is None
        assert self.clf.classify("Listless prose about nothing") is None

    def test_unclassifiable_returns_none(self):
        assert self.clf.classify("") is None
        assert self.clf.classify("Coffee and roasted beans") is None

    def test_recall_question_fallback(self):
        assert self.clf.classify("Which of the following is correct: what is a variable?") == 1
        assert self.clf.classify("True or false: the sky is green") == 1

    def test_declared_level_normalization(self):
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
        assert sanity["bloom_vs_declared_agreement"] == 1.0


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
        # Encoders are preset-only; the primary preset pins the model name.
        assert (module.ENCODER_PRESETS[module.PRIMARY_ENCODER][1]
                == "nvidia/llama-embed-nemotron-8b")
        assert "composite" not in module.COMPONENTS
        assert module.COMPONENTS[0] == "objective_assessment_similarity"
        assert "objective_evaluation_similarity" in module.COMPONENTS
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
