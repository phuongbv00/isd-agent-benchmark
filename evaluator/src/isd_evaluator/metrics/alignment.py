"""Quantitative constructive-alignment metrics for ADDIE outputs.

Deterministic, LLM-free alignment scoring of an agent's ADDIE output against
its own stated learning objectives and the scenario's learning goals.

Components (all in [0, 1], reported separately and combined):

1. **Porter Alignment Index** (Porter 2002; Fulmer 2011)
   ``P = 1 - 0.5 * sum(|X_ij - Y_ij|)`` between two normalized
   content-topic x Bloom-level distribution matrices (each sums to 1).
   Computed for the pairs objectives<->assessment, objectives<->activities,
   objectives<->evaluation.

2. **Webb criteria** (Webb 1997/1999), reported continuous:
   - *Range-of-Knowledge*: fraction of objectives with at least one
     corresponding assessment item (similarity >= threshold).
   - *Bloom-Consistency*: fraction of assessment items whose cognitive level
     is >= the level of the objective they correspond to (argmax similarity).

3. **Embedding coverage / precision** (SE traceability framing,
   Cleland-Huang): coverage = mean over objectives of the max cosine
   similarity to any assessment item; precision = the reverse direction.

Composite score (see :meth:`AlignmentEvaluator.evaluate`):
``composite = mean of the defined values among [porter_mean, webb_range,
webb_bloom_consistency, embedding_coverage, embedding_precision]``, or 0.0 if
none is defined (e.g. no objectives at all).

Design principles enforced here:

(a) Aggregation is a pure deterministic formula — zero LLM calls.
(b) Scoring reads **text only** (objective statements, questions, activity
    descriptions). Structural links (``objective_id`` / ``aligned_objective``)
    and self-declared ``level`` / ``bloom_verb`` fields are NEVER used in
    scoring; they only feed the auxiliary sanity report.
(c) Correspondence between items and objectives is established via text
    similarity matching, never via ids.

Pluggable pieces:

- :class:`BloomClassifier` protocol; default :class:`LexiconBloomClassifier`
  (Korean + English Bloom verb lexicon, rule-based, offline).
  :class:`TransformerBloomClassifier` loads a local fine-tuned checkpoint if
  one exists (it never downloads).
- :class:`TextEncoder` protocol; default :class:`TfidfCharNgramEncoder`
  (character n-gram TF-IDF, pure Python, deterministic, suits unspaced
  Korean). :class:`SentenceTransformerEncoder` (LaBSE / BGE-M3) is an
  optional, lazily-imported alternative for real scoring runs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

# ---------------------------------------------------------------------------
# Bloom levels
# ---------------------------------------------------------------------------

#: Canonical Bloom levels (revised taxonomy), index 1..6.
BLOOM_LEVELS = ["기억", "이해", "적용", "분석", "평가", "창조"]

_LEVEL_ALIASES = {
    # Korean canonical
    "기억": 1, "이해": 2, "적용": 3, "분석": 4, "평가": 5, "창조": 6, "창안": 6,
    # English (revised taxonomy)
    "remember": 1, "understand": 2, "apply": 3, "analyze": 4, "analyse": 4,
    "evaluate": 5, "create": 6,
    # English (original taxonomy, occasionally emitted by agents)
    "knowledge": 1, "comprehension": 2, "application": 3, "analysis": 4,
    "synthesis": 6, "evaluation": 5,
}


def normalize_declared_level(value: Any) -> Optional[int]:
    """Map a self-declared level field (Korean or English) to 1..6, else None."""
    if not isinstance(value, str):
        return None
    return _LEVEL_ALIASES.get(value.strip().lower())


class BloomClassifier(Protocol):
    """Text -> Bloom level (1..6), or None when undecidable."""

    def classify(self, text: str) -> Optional[int]:  # pragma: no cover - protocol
        ...


# Korean instructional verb stems per level. Matching requires a verbal
# conjugation right after the stem (하/할/한/함/해/했/합/히 or explicit native
# endings) so that bare noun usages ("교수설계", "사용자") do not fire.
_KO_VERB_LEXICON: dict[int, list[str]] = {
    1: ["기억", "암기", "나열", "열거", "정의", "명명", "회상", "식별", "확인",
        "인지", "진술", "명시", "지칭", "상기"],
    2: ["설명", "요약", "해석", "분류", "예시", "번역", "이해", "기술", "논의",
        "표현", "부연", "추론", "예측", "환언"],
    3: ["적용", "활용", "사용", "실행", "실습", "구현", "수행", "시연", "연습",
        "계산", "조작", "해결", "응용", "시행"],
    4: ["분석", "비교", "대조", "구별", "구분", "분해", "조사", "진단", "규명",
        "검사", "탐색", "범주화", "차별화", "도식화"],
    5: ["평가", "판단", "비평", "비판", "심사", "검증", "검토", "논증", "정당화",
        "타당화", "추천", "옹호", "반박", "판정"],
    6: ["창조", "창작", "설계", "개발", "제작", "작성", "고안", "생성", "구축",
        "발명", "계획", "종합", "통합", "제안", "구성"],
}

# Native (non-하다) Korean verb stems: matched as bare stems.
_KO_NATIVE_STEMS: dict[int, list[str]] = {
    1: ["찾"],
    6: ["만들", "세우"],
}

# English instructional verbs per level (inflections generated automatically).
_EN_VERB_LEXICON: dict[int, list[str]] = {
    1: ["recall", "list", "define", "name", "identify", "label", "state",
        "recognize", "memorize", "recite", "locate", "repeat", "retrieve",
        "match"],
    2: ["explain", "summarize", "summarise", "describe", "interpret",
        "classify", "paraphrase", "discuss", "illustrate", "translate",
        "infer", "predict", "exemplify", "restate"],
    3: ["apply", "use", "implement", "execute", "demonstrate", "solve",
        "operate", "practice", "calculate", "employ", "perform", "utilize",
        "conduct"],
    4: ["analyze", "analyse", "compare", "contrast", "differentiate",
        "distinguish", "examine", "categorize", "diagnose", "deconstruct",
        "attribute", "investigate", "dissect"],
    5: ["evaluate", "judge", "critique", "justify", "assess", "validate",
        "verify", "defend", "appraise", "argue", "recommend", "prioritize",
        "criticize"],
    6: ["create", "design", "develop", "construct", "produce", "compose",
        "formulate", "devise", "generate", "build", "invent", "plan",
        "propose", "synthesize", "prototype", "integrate"],
}

# Question-form fallbacks (used only when no verb matched): plain
# recall-style interrogatives are treated as Remember-level items.
_QUESTION_FALLBACKS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"무엇인가|무엇입니까|무엇일까"), 1),
    (re.compile(r"\bwhat\s+(is|are)\b", re.IGNORECASE), 1),
    (re.compile(r"\btrue\s+or\s+false\b|\(?OX\)?\s*퀴즈|OX\s*문제"), 1),
]


def _en_inflections(verb: str) -> str:
    """Regex alternation of simple inflections of an English verb."""
    if verb.endswith("y"):
        base = re.escape(verb[:-1])
        return f"{base}(?:y|ies|ied|ying)"
    if verb.endswith("e"):
        base = re.escape(verb[:-1])
        return f"{base}(?:e|es|ed|ing)"
    base = re.escape(verb)
    return f"{base}(?:s|es|ed|ing)?"


class LexiconBloomClassifier:
    """Rule-based Bloom classifier over a Korean + English verb lexicon.

    When verbs of several levels match, the **highest** level wins — the
    conventional "highest cognitive demand" coding rule used in DOK /
    alignment studies (a task that requires any higher-order process is coded
    at that level).
    """

    def __init__(self) -> None:
        self._patterns: list[tuple[re.Pattern[str], int]] = []
        for level, stems in _KO_VERB_LEXICON.items():
            alternation = "|".join(re.escape(s) for s in stems)
            pattern = re.compile(f"(?:{alternation})[하할한함해했합히]")
            self._patterns.append((pattern, level))
        for level, stems in _KO_NATIVE_STEMS.items():
            alternation = "|".join(re.escape(s) for s in stems)
            self._patterns.append((re.compile(f"(?:{alternation})"), level))
        for level, verbs in _EN_VERB_LEXICON.items():
            alternation = "|".join(_en_inflections(v) for v in verbs)
            pattern = re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)
            self._patterns.append((pattern, level))

    def classify(self, text: str) -> Optional[int]:
        if not text:
            return None
        best: Optional[int] = None
        for pattern, level in self._patterns:
            if (best is None or level > best) and pattern.search(text):
                best = level
        if best is None:
            for pattern, level in _QUESTION_FALLBACKS:
                if pattern.search(text):
                    return level
        return best


class TransformerBloomClassifier:
    """Bloom classifier backed by a locally fine-tuned sequence classifier.

    Loads a checkpoint from ``model_path`` only if it already exists on disk;
    it never downloads. Train one with
    ``evaluator/scripts/train_bloom_classifier.py``.
    """

    def __init__(self, model_path: str, max_length: int = 128) -> None:
        import os

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Bloom classifier checkpoint not found at '{model_path}'. "
                "Fine-tune one with evaluator/scripts/train_bloom_classifier.py "
                "or fall back to LexiconBloomClassifier."
            )
        from transformers import (  # noqa: PLC0415 - optional heavy dep
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.eval()
        self._max_length = max_length

    def classify(self, text: str) -> Optional[int]:
        if not text:
            return None
        import torch  # noqa: PLC0415 - optional heavy dep

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self._max_length
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return int(logits.argmax(dim=-1).item()) + 1


# ---------------------------------------------------------------------------
# Text encoders
# ---------------------------------------------------------------------------


class TextEncoder(Protocol):
    """Batch text encoder; vectors from one call are mutually comparable."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        ...


class TfidfCharNgramEncoder:
    """Character n-gram TF-IDF encoder (pure Python, deterministic).

    Character 2-3 grams suit Korean, which segments poorly on whitespace.
    The IDF is fitted on the batch passed to :meth:`encode`, so all texts of
    one ADDIE output must be encoded in a single call (the evaluator does).
    """

    def __init__(self, ngram_range: tuple[int, int] = (2, 3)) -> None:
        self.ngram_range = ngram_range

    def _ngrams(self, text: str) -> dict[str, int]:
        text = re.sub(r"\s+", " ", text.lower()).strip()
        counts: dict[str, int] = {}
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(max(len(text) - n + 1, 0)):
                gram = text[i:i + n]
                counts[gram] = counts.get(gram, 0) + 1
        return counts

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        docs = [self._ngrams(t or "") for t in texts]
        df: dict[str, int] = {}
        for doc in docs:
            for gram in doc:
                df[gram] = df.get(gram, 0) + 1
        vocab = {gram: idx for idx, gram in enumerate(sorted(df))}
        n_docs = len(docs)
        idf = {
            gram: math.log((1 + n_docs) / (1 + count)) + 1.0
            for gram, count in df.items()
        }
        vectors: list[list[float]] = []
        for doc in docs:
            vec = [0.0] * len(vocab)
            for gram, tf in doc.items():
                vec[vocab[gram]] = tf * idf[gram]
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


class SentenceTransformerEncoder:
    """Multilingual sentence-embedding encoder (lazy-loaded, optional dep).

    Requires the ``alignment`` extra (``pip install isd-evaluator[alignment]``).
    Default model is LaBSE; ``BAAI/bge-m3`` is a good alternative.
    """

    def __init__(self, model_name: str = "sentence-transformers/LaBSE") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        embeddings = model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, row)) for row in embeddings]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


# ---------------------------------------------------------------------------
# Text extraction from ADDIE outputs (schema + observed real-run variants)
# ---------------------------------------------------------------------------


def _texts_of(items: Any, *keys: str) -> list[str]:
    """Join the given keys of each dict item (or the item itself if str)."""
    out: list[str] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, str):
            if item.strip():
                out.append(item.strip())
        elif isinstance(item, dict):
            parts = [str(item[k]).strip() for k in keys if item.get(k)]
            if parts:
                out.append(" ".join(parts))
    return out


def extract_objectives(addie_output: dict) -> list[dict[str, Any]]:
    """Learning objectives as [{text, declared_level}] (text-only scoring)."""
    objectives = []
    raw = (addie_output.get("design") or {}).get("learning_objectives") or []
    if not isinstance(raw, list):
        return objectives
    for entry in raw:
        if isinstance(entry, str):
            text, declared = entry.strip(), None
        elif isinstance(entry, dict):
            text = str(entry.get("statement") or "").strip()
            declared = normalize_declared_level(entry.get("level"))
        else:
            continue
        if text:
            objectives.append({"text": text, "declared_level": declared})
    return objectives


def extract_assessment_items(addie_output: dict) -> list[str]:
    """Assessment item texts, union over schema and observed layouts.

    Item-level questions come from ``evaluation.quiz_items``,
    ``development.assessment_tools`` and ``evaluation.summative
    .assessment_tools``. Only when no item-level question exists do the
    free-text ``design.assessment_plan`` strings serve as fallback.
    """
    evaluation = addie_output.get("evaluation") or {}
    development = addie_output.get("development") or {}
    texts: list[str] = []
    texts += _texts_of(evaluation.get("quiz_items"), "question")
    texts += _texts_of(development.get("assessment_tools"), "question")
    summative = evaluation.get("summative")
    if isinstance(summative, dict):
        texts += _texts_of(summative.get("assessment_tools"), "question")
    if not texts:
        plan = (addie_output.get("design") or {}).get("assessment_plan") or {}
        if isinstance(plan, dict):
            for kind in ("diagnostic", "formative", "summative"):
                texts += _texts_of(plan.get(kind))
    seen: set[str] = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def extract_activities(addie_output: dict) -> list[str]:
    """Learning-activity texts, union over schema and observed layouts."""
    design = addie_output.get("design") or {}
    development = addie_output.get("development") or {}
    texts: list[str] = []
    lesson_plan = development.get("lesson_plan") or {}
    for module in lesson_plan.get("modules") or []:
        if isinstance(module, dict):
            texts += _texts_of(module.get("activities"), "activity", "description")
    for strategy_key in ("instructional_strategy", "instructional_strategies"):
        strategy = design.get(strategy_key)
        if isinstance(strategy, dict):
            texts += _texts_of(strategy.get("sequence"), "event", "activity")
    texts += _texts_of(design.get("learning_activities"), "activity_name", "description")
    seen: set[str] = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def extract_evaluation_texts(addie_output: dict) -> list[str]:
    """Evaluation-phase instrument texts (rubric, feedback, Kirkpatrick...)."""
    evaluation = addie_output.get("evaluation") or {}
    texts: list[str] = []
    texts += _texts_of(evaluation.get("quiz_items"), "question")
    rubric = evaluation.get("rubric")
    if isinstance(rubric, dict):
        texts += _texts_of(rubric.get("criteria"))
    feedback = evaluation.get("feedback_plan")
    if isinstance(feedback, str) and feedback.strip():
        texts.append(feedback.strip())
    program = evaluation.get("program_evaluation")
    if isinstance(program, dict):
        texts += _texts_of(program.get("kirkpatrick_levels"), "name", "method")
    formative = evaluation.get("formative")
    if isinstance(formative, dict):
        data_collection = formative.get("data_collection")
        if isinstance(data_collection, dict):
            texts += _texts_of(data_collection.get("methods"))
        elif isinstance(data_collection, list):
            texts += _texts_of(data_collection)
    summative = evaluation.get("summative")
    if isinstance(summative, dict):
        texts += _texts_of(summative.get("assessment_tools"), "question")
    seen: set[str] = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def extract_topics(scenario: Optional[dict]) -> list[str]:
    """Brief-anchored content-topic taxonomy from the scenario itself.

    Uses the scenario's ``learning_goals`` (one topic per goal). Falls back to
    title+domain as a single topic, keeping the Porter matrix well-defined
    (it then degenerates to a Bloom-marginal comparison).
    """
    if isinstance(scenario, dict):
        goals = scenario.get("learning_goals")
        if isinstance(goals, list):
            topics = [str(g).strip() for g in goals if str(g).strip()]
            if topics:
                return topics
        title = str(scenario.get("title") or "").strip()
        domain = str(scenario.get("domain") or "").strip()
        combined = " ".join(p for p in (title, domain) if p)
        if combined:
            return [combined]
    return ["__all__"]


# ---------------------------------------------------------------------------
# Score container
# ---------------------------------------------------------------------------


@dataclass
class AlignmentScore:
    """Quantitative alignment score; every numeric component lies in [0, 1]."""

    porter: dict[str, Optional[float]] = field(default_factory=dict)
    porter_mean: Optional[float] = None
    webb_range: Optional[float] = None
    webb_bloom_consistency: Optional[float] = None
    embedding_coverage: Optional[float] = None
    embedding_precision: Optional[float] = None
    composite: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite": self.composite,
            "porter": self.porter,
            "porter_mean": self.porter_mean,
            "webb_range": self.webb_range,
            "webb_bloom_consistency": self.webb_bloom_consistency,
            "embedding_coverage": self.embedding_coverage,
            "embedding_precision": self.embedding_precision,
            "counts": self.counts,
            "details": self.details,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class AlignmentEvaluator:
    """Deterministic constructive-alignment evaluator for one ADDIE output.

    Conventions for degenerate inputs (documented, all keep values in [0,1]):

    - No objectives: every component is undefined and ``composite = 0.0``.
    - Objectives present but a counterpart set empty: the Porter index for
      that pair, ``webb_range`` and ``embedding_coverage`` are 0.0 (nothing
      addresses the objectives); ``embedding_precision`` is undefined.
    - Bloom-dependent components use only texts the classifier can label;
      if no assessment item (or no objective) is classifiable while items
      exist, ``webb_bloom_consistency`` is undefined and excluded from the
      composite (a note records this).
    """

    PAIRS = ("assessment", "activities", "evaluation")

    def __init__(
        self,
        encoder: Optional[TextEncoder] = None,
        bloom_classifier: Optional[BloomClassifier] = None,
        match_threshold: float = 0.25,
    ) -> None:
        self.encoder = encoder or TfidfCharNgramEncoder()
        self.bloom = bloom_classifier or LexiconBloomClassifier()
        self.match_threshold = match_threshold

    # -- public API ---------------------------------------------------------

    def evaluate(self, addie_output: dict, scenario: Optional[dict] = None) -> AlignmentScore:
        if isinstance(addie_output, dict) and "addie_output" in addie_output:
            addie_output = addie_output["addie_output"] or {}
        score = AlignmentScore()

        objectives = extract_objectives(addie_output)
        assessments = extract_assessment_items(addie_output)
        activities = extract_activities(addie_output)
        evaluations = extract_evaluation_texts(addie_output)
        topics = extract_topics(scenario)

        score.counts = {
            "objectives": len(objectives),
            "assessment": len(assessments),
            "activities": len(activities),
            "evaluation": len(evaluations),
            "topics": len(topics),
        }

        if not objectives:
            score.notes.append("no learning objectives found; composite = 0.0")
            score.composite = 0.0
            return score

        objective_texts = [o["text"] for o in objectives]
        sets: dict[str, list[str]] = {
            "objectives": objective_texts,
            "assessment": assessments,
            "activities": activities,
            "evaluation": evaluations,
        }

        # One joint encoding call so all vectors share a space (required by
        # the TF-IDF fallback, harmless for neural encoders).
        all_texts = list(topics)
        offsets: dict[str, tuple[int, int]] = {}
        cursor = len(topics)
        for name, texts in sets.items():
            offsets[name] = (cursor, cursor + len(texts))
            all_texts += texts
            cursor += len(texts)
        vectors = self.encoder.encode(all_texts)
        topic_vecs = vectors[: len(topics)]
        set_vecs = {
            name: vectors[start:end] for name, (start, end) in offsets.items()
        }

        levels = {
            name: [self.bloom.classify(t) for t in texts]
            for name, texts in sets.items()
        }
        topic_ids = {
            name: [self._argmax_topic(vec, topic_vecs) for vec in vecs]
            for name, vecs in set_vecs.items()
        }

        # Porter alignment index per pair.
        obj_matrix = self._porter_matrix(
            topic_ids["objectives"], levels["objectives"], len(topics)
        )
        for pair in self.PAIRS:
            matrix = self._porter_matrix(topic_ids[pair], levels[pair], len(topics))
            if obj_matrix is None or matrix is None:
                if sets[pair]:
                    score.porter[pair] = None
                    score.notes.append(
                        f"porter[{pair}] undefined: no Bloom-classifiable text"
                    )
                else:
                    score.porter[pair] = 0.0
                    score.notes.append(f"porter[{pair}] = 0.0: '{pair}' set is empty")
                continue
            score.porter[pair] = self._porter_index(obj_matrix, matrix)
        defined = [v for v in score.porter.values() if v is not None]
        score.porter_mean = sum(defined) / len(defined) if defined else None

        # Similarity between objectives and assessment items.
        sim = [
            [_cosine(ov, av) for av in set_vecs["assessment"]]
            for ov in set_vecs["objectives"]
        ]

        # Webb range-of-knowledge.
        if assessments:
            matched = [
                1 if row and max(row) >= self.match_threshold else 0 for row in sim
            ]
            score.webb_range = sum(matched) / len(matched)
        else:
            score.webb_range = 0.0
            score.notes.append("webb_range = 0.0: no assessment items found")

        # Webb Bloom-consistency (levels from text, correspondence by argmax
        # similarity to a classifiable objective).
        score.webb_bloom_consistency = self._bloom_consistency(
            sim, levels["objectives"], levels["assessment"], score.notes
        )

        # Embedding coverage / precision.
        if assessments:
            score.embedding_coverage = sum(max(row) for row in sim) / len(sim)
            per_item_max = [
                max(sim[i][j] for i in range(len(sim))) for j in range(len(assessments))
            ]
            score.embedding_precision = sum(per_item_max) / len(per_item_max)
        else:
            score.embedding_coverage = 0.0
            score.notes.append("embedding_coverage = 0.0: no assessment items found")

        score.details["objectives"] = self._objective_details(
            objectives, assessments, sim, levels, topic_ids["objectives"], topics
        )
        score.details["sanity"] = self._sanity_report(objectives, levels["objectives"])

        components = [
            score.porter_mean,
            score.webb_range,
            score.webb_bloom_consistency,
            score.embedding_coverage,
            score.embedding_precision,
        ]
        defined = [c for c in components if c is not None]
        score.composite = sum(defined) / len(defined) if defined else 0.0
        return score

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _argmax_topic(vec: Sequence[float], topic_vecs: list[list[float]]) -> int:
        sims = [_cosine(vec, tv) for tv in topic_vecs]
        return max(range(len(sims)), key=lambda i: sims[i]) if sims else 0

    @staticmethod
    def _porter_matrix(
        topic_ids: list[int], levels: list[Optional[int]], n_topics: int
    ) -> Optional[list[list[float]]]:
        """Normalized topic x Bloom distribution; None if nothing classifiable."""
        cells = [
            (topic, level)
            for topic, level in zip(topic_ids, levels)
            if level is not None
        ]
        if not cells:
            return None
        matrix = [[0.0] * 6 for _ in range(n_topics)]
        weight = 1.0 / len(cells)
        for topic, level in cells:
            matrix[topic][level - 1] += weight
        return matrix

    @staticmethod
    def _porter_index(x: list[list[float]], y: list[list[float]]) -> float:
        diff = sum(
            abs(a - b) for row_x, row_y in zip(x, y) for a, b in zip(row_x, row_y)
        )
        return max(0.0, min(1.0, 1.0 - 0.5 * diff))

    @staticmethod
    def _bloom_consistency(
        sim: list[list[float]],
        objective_levels: list[Optional[int]],
        item_levels: list[Optional[int]],
        notes: list[str],
    ) -> Optional[float]:
        if not item_levels:
            notes.append("webb_bloom_consistency = 0.0: no assessment items found")
            return 0.0
        classifiable_objs = [
            i for i, level in enumerate(objective_levels) if level is not None
        ]
        if not classifiable_objs:
            notes.append(
                "webb_bloom_consistency undefined: no Bloom-classifiable objective"
            )
            return None
        consistent = 0
        counted = 0
        for j, item_level in enumerate(item_levels):
            if item_level is None:
                continue
            target = max(classifiable_objs, key=lambda i: sim[i][j])
            counted += 1
            if item_level >= (objective_levels[target] or 0):
                consistent += 1
        if counted == 0:
            notes.append(
                "webb_bloom_consistency undefined: no Bloom-classifiable item"
            )
            return None
        return consistent / counted

    def _objective_details(
        self,
        objectives: list[dict[str, Any]],
        assessments: list[str],
        sim: list[list[float]],
        levels: dict[str, list[Optional[int]]],
        objective_topics: list[int],
        topics: list[str],
    ) -> list[dict[str, Any]]:
        details = []
        for i, objective in enumerate(objectives):
            row = sim[i] if i < len(sim) else []
            best_j = max(range(len(row)), key=lambda j: row[j]) if row else None
            details.append(
                {
                    "text": objective["text"],
                    "topic": topics[objective_topics[i]],
                    "lexicon_level": levels["objectives"][i],
                    "declared_level": objective["declared_level"],
                    "best_item": assessments[best_j] if best_j is not None else None,
                    "best_similarity": round(row[best_j], 4) if best_j is not None else None,
                    "best_item_level": (
                        levels["assessment"][best_j] if best_j is not None else None
                    ),
                    "matched": bool(row) and max(row) >= self.match_threshold,
                }
            )
        return details

    @staticmethod
    def _sanity_report(
        objectives: list[dict[str, Any]], lexicon_levels: list[Optional[int]]
    ) -> dict[str, Any]:
        """Report-only agreement between the lexicon classifier and the
        agent's self-declared ``level`` fields. Never enters the composite."""
        both = [
            (obj["declared_level"], lex)
            for obj, lex in zip(objectives, lexicon_levels)
            if obj["declared_level"] is not None and lex is not None
        ]
        agreement = (
            sum(1 for declared, lex in both if declared == lex) / len(both)
            if both
            else None
        )
        return {
            "n_objectives": len(objectives),
            "n_declared": sum(
                1 for o in objectives if o["declared_level"] is not None
            ),
            "n_lexicon_classified": sum(
                1 for level in lexicon_levels if level is not None
            ),
            "n_comparable": len(both),
            "lexicon_vs_declared_agreement": agreement,
        }
