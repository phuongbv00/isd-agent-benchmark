"""Automated constructive-alignment evaluation protocol for ADDIE outputs.

Deterministic scoring (no LLM call anywhere in the measurement) of an agent's
ADDIE output against its own stated learning objectives and the scenario's
learning goals, operationalizing established alignment criteria (Biggs & Tang
constructive alignment; QM-style review criteria; Porter/Webb indices).

**No similarity threshold anywhere** — objective<->item relations are scored
as continuous mean-max cosine, not thresholded binary rates. This avoids the
arbitrary-cutoff critique and keeps the full similarity signal. The one
non-identity applied to a similarity is a *rectification*: cosine enters as
``max(0, cos)`` (see :func:`_cosine`) so every endpoint lives on a common
[0, 1] scale. That is a range choice, not a matching cutoff — no comparison
against any constant decides whether two texts count as related.

Primary endpoint — **objective_assessment_alignment**: the mean over learning
objectives of the maximum rectified cosine to any assessment item. This is
the objective<->assessment relation at the heart of constructive alignment
(Biggs), continuous in [0, 1]. Reported alongside it (see
:meth:`AlignmentEvaluator.evaluate`):

- *objective_activity_alignment* / *objective_evaluation_alignment* — the same
  continuous mean-max cosine against the activities / evaluation-phase texts
  (the ``supports`` / ``evaluates`` legs of the paper's aligned(o) predicate);
- *objective_cognitive_congruence* — share of objectives whose argmax-closest
  assessment item has Bloom level >= the objective's level (argmax, no
  threshold), over objectives with both levels classifiable.

Descriptive indices (also threshold-free, reported separately — NO composite):

- **Porter Alignment Index** (Porter 2002; Fulmer 2011)
  ``P = 1 - 0.5 * sum(|X_ij - Y_ij|)`` between two normalized
  content-topic x Bloom-level distribution matrices, for the pairs
  objectives<->assessment, <->activities, <->evaluation.
- **Webb Bloom-Consistency** (Webb 1997/1999), item-centric.
- **assessment_precision** — reverse of the primary (mean over assessment
  items of the max cosine to any objective; SE traceability framing).

Design principles enforced here:

(a) The whole protocol is deterministic — zero LLM calls, no threshold.
(b) Scoring reads **text only** (objective statements, questions, activity
    descriptions). Structural links (``objective_id`` / ``aligned_objective``)
    and self-declared ``level`` fields are NEVER used in scoring; they feed
    only the validation-evidence report (declared-link agreement,
    lexicon-vs-declared agreement).
(c) Correspondence between items and objectives is established via text
    similarity matching, never via ids.

Pluggable pieces:

- :class:`BloomClassifier` protocol; default :class:`LexiconBloomClassifier`
  (Korean + English Bloom verb lexicon, rule-based, offline).
  :class:`TransformerBloomClassifier` loads a local fine-tuned checkpoint if
  one exists (it never downloads).
- :class:`TextEncoder` protocol; default :class:`TfidfCharNgramEncoder`
  (character n-gram TF-IDF, pure Python, deterministic, suits unspaced
  Korean). :class:`OpenAIAPIEncoder` (OpenAI-compatible ``/v1/embeddings``
  endpoint + persistent SQLite vector cache) is the intended encoder for real
  scoring runs; :class:`SentenceTransformerEncoder` runs the same models
  locally.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
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

    Character 2-3 grams are robust to morphology and tokenisation, which is
    what makes this a fair lexical floor rather than a word-overlap counter.
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


class _EmbeddingCache:
    """SQLite-backed float32 vector cache keyed by sha1(model[@revision] + text).

    Makes real scoring runs resumable and later sensitivity passes free:
    each distinct text is embedded exactly once per model *revision*.

    The revision belongs in the key, not just in the artifact metadata: a
    model id like ``nvidia/llama-embed-nemotron-8b`` does not pin weights, so
    without it a re-score after an upstream weight update would silently serve
    the OLD vectors out of the cache while the score file claims the new
    revision. Keys stay byte-identical to the pre-revision format when no
    revision is given, so existing caches remain valid.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), timeout=60.0)
        # WAL + busy timeout: several scorer processes may share one cache.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=60000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)"
        )
        self._conn.commit()

    @staticmethod
    def key(model: str, text: str, revision: Optional[str] = None) -> str:
        ident = f"{model}@{revision}" if revision else model
        return hashlib.sha1(f"{ident}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        chunk = 500  # stay under SQLite's variable limit
        for start in range(0, len(keys), chunk):
            batch = list(keys[start:start + chunk])
            marks = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT key, vec FROM vectors WHERE key IN ({marks})", batch
            )
            for key, blob in rows:
                out[key] = list(array("f", blob))
        return out

    def put_many(self, items: dict[str, list[float]]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)",
            [(k, array("f", v).tobytes()) for k, v in items.items()],
        )
        self._conn.commit()


class OpenAIAPIEncoder:
    """OpenAI-compatible ``/v1/embeddings`` client with a persistent cache.

    Intended for one or more vLLM pods serving the embedding model
    (``vllm serve <model> --task embed``). Deterministic given a pinned
    model: identical text always maps to the identical cached vector.
    Empty/whitespace texts become zero vectors without an API call
    (mirroring :class:`TfidfCharNgramEncoder`).

    Multiple endpoints (``base_urls`` as a list) are load-balanced: the
    uncached batches of a single ``encode`` call are dispatched round-robin
    and run concurrently, one thread per endpoint. All endpoints must serve
    the same model (same vectors), so which pod answers a batch is
    irrelevant to the result. The SQLite cache is touched only from the
    calling thread (lookup before, write after) — the HTTP calls are the
    only thing parallelized.
    """

    def __init__(
        self,
        base_urls: str | Sequence[str],
        model: str,
        api_key: Optional[str] = None,
        batch_size: int = 64,
        cache_path: Optional[Path] = None,
        timeout: float = 120.0,
        cache_only: bool = False,
        revision: Optional[str] = None,
    ) -> None:
        from openai import OpenAI  # noqa: PLC0415 - lazy, optional at import

        if isinstance(base_urls, str):
            base_urls = [base_urls]
        urls = [u.strip() for u in base_urls if u and u.strip()]
        if not urls:
            raise ValueError("OpenAIAPIEncoder needs at least one base URL")
        self.model = model
        self.revision = revision or None
        self.batch_size = batch_size
        self.cache_only = cache_only
        self._clients = [
            OpenAI(base_url=u, api_key=api_key or "EMPTY",
                   timeout=timeout, max_retries=3)
            for u in urls
        ]
        self._cache = _EmbeddingCache(cache_path) if cache_path else None
        self._dim: Optional[int] = None

    def _embed_batch(self, texts: list[str], client=None) -> list[list[float]]:
        client = client or self._clients[0]
        response = client.embeddings.create(model=self.model, input=texts)
        rows = sorted(response.data, key=lambda item: item.index)
        return [list(map(float, row.embedding)) for row in rows]

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        texts = [str(t) for t in texts]
        vectors: list[Optional[list[float]]] = [None] * len(texts)
        pending: list[int] = []
        keys = [
            self._cache.key(self.model, t, self.revision) if self._cache else ""
            for t in texts
        ]
        cached = self._cache.get_many(
            [keys[i] for i, t in enumerate(texts) if t.strip()]
        ) if self._cache else {}
        for i, text in enumerate(texts):
            if not text.strip():
                continue  # zero vector, filled once the dimension is known
            hit = cached.get(keys[i])
            if hit is not None:
                vectors[i] = hit
                self._dim = self._dim or len(hit)
            else:
                pending.append(i)

        if pending and self.cache_only:
            raise RuntimeError(
                f"cache_only=True but {len(pending)} of {len(texts)} texts are "
                f"not in the embedding cache for model {self.model!r}. Re-run "
                "the primary scoring pass against a live endpoint first, or "
                "drop cache_only."
            )
        batches = [
            pending[start:start + self.batch_size]
            for start in range(0, len(pending), self.batch_size)
        ]
        n_clients = len(self._clients)
        if n_clients > 1 and len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

            def run(job):
                bi, batch_idx = job
                client = self._clients[bi % n_clients]
                return batch_idx, self._embed_batch(
                    [texts[i] for i in batch_idx], client=client)

            with ThreadPoolExecutor(max_workers=n_clients) as pool:
                results = list(pool.map(run, enumerate(batches)))
        else:
            results = [
                (batch_idx, self._embed_batch([texts[i] for i in batch_idx]))
                for batch_idx in batches
            ]

        for batch_idx, embedded in results:
            for i, vec in zip(batch_idx, embedded):
                vectors[i] = vec
                self._dim = self._dim or len(vec)
            if self._cache:
                self._cache.put_many(
                    {keys[i]: vec for i, vec in zip(batch_idx, embedded)}
                )
        dim = self._dim or 1
        return [vec if vec is not None else [0.0] * dim for vec in vectors]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """RECTIFIED cosine: ``max(0, cos)``, clipped to [0, 1].

    Not a matching threshold — there is no cutoff anywhere in this module, and
    every endpoint is continuous. The rectification only fixes the *range*: an
    anti-correlated pair is no more aligned than an orthogonal one, so negative
    similarity is meaningless as "alignment" and collapses to the 0 floor,
    which keeps every endpoint on a common [0, 1] scale. The upper clip guards
    floating-point overshoot past 1.0 only. Describe the metric as *rectified
    cosine on [0, 1]*, never as raw cosine.
    """
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


def _texts_of_flexible(items: Any, *keys: str) -> list[str]:
    """Like :func:`_texts_of`, with a string-leaf fallback for free-form dicts.

    Some baseline assessment plans use generated keys such as ``diagnostic1``
    or ``summative_method``. Because the containing field is already scoped to
    an assessment plan, joining its string leaves is safer than discarding the
    assessment solely because its keys differ from the canonical schema.
    """

    def leaves(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [text for item in value for text in leaves(item)]
        if isinstance(value, dict):
            return [text for item in value.values() for text in leaves(item)]
        return []

    out: list[str] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, str):
            if item.strip():
                out.append(item.strip())
        elif isinstance(item, dict):
            parts = [str(item[k]).strip() for k in keys if item.get(k)]
            if not parts:
                parts = leaves(item)
            if parts:
                out.append(" ".join(parts))
    return out


def extract_objectives(addie_output: dict) -> list[dict[str, Any]]:
    """Learning objectives as [{text, declared_level, id}].

    ``id`` is the output's self-declared objective id — used ONLY by the
    validation-evidence report (declared-link agreement), never in scoring.
    """
    objectives = []
    raw = (addie_output.get("design") or {}).get("learning_objectives") or []
    if not isinstance(raw, list):
        return objectives
    for entry in raw:
        if isinstance(entry, str):
            text, declared, obj_id = entry.strip(), None, None
        elif isinstance(entry, dict):
            text = str(entry.get("statement") or "").strip()
            declared = normalize_declared_level(entry.get("level"))
            obj_id = str(entry.get("id") or "").strip() or None
        else:
            continue
        if text:
            objectives.append({"text": text, "declared_level": declared, "id": obj_id})
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
                texts += _texts_of_flexible(
                    plan.get(kind),
                    "method",
                    "description",
                    "question",
                    "assessment",
                    "name",
                )
            texts += _texts_of_flexible(
                plan.get("assessment_criteria"),
                "criterion",
                "description",
                "name",
            )
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
            # Observed EduPlanner / ADDIE-Agent layout: the strategy is a
            # compact {methods, activities, rationale} object rather than a
            # Gagné-style sequence.
            texts += _texts_of(
                strategy.get("activities"),
                "activity",
                "activity_name",
                "name",
                "description",
            )
    texts += _texts_of(
        design.get("learning_activities"),
        "activity",
        "activity_name",
        "name",
        "description",
    )
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
    pilot = evaluation.get("pilot_data_collection")
    if isinstance(pilot, dict):
        # Observed baseline layout: evaluation may consist solely of a pilot
        # data-collection plan. Keep instrument/method text (the evidence
        # actually used to evaluate learning) and avoid administrative fields
        # such as storage/retention metadata.
        texts += _texts_of(
            pilot.get("collection_methods"),
            "method",
            "timing",
            "tool",
        )
        texts += _texts_of(pilot.get("instruments"), "name", "type")
        data_types = pilot.get("data_types")
        if isinstance(data_types, dict):
            for items in data_types.values():
                texts += _texts_of(items, "type", "purpose", "source")
        timeline = pilot.get("timeline")
        if isinstance(timeline, list):
            for stage in timeline:
                if not isinstance(stage, dict):
                    continue
                texts += _texts_of(stage.get("activities"))
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


def extract_declared_links(addie_output: dict) -> list[tuple[str, str]]:
    """(assessment-item text, self-declared objective id) pairs.

    Read from the output's own ``aligned_objective`` / ``objective_id``
    fields. Validation-evidence only — these links are NEVER used in scoring
    (principle (b) in the module docstring); they serve as noisy labels to
    check whether similarity matching reproduces the links the agent itself
    declared.
    """
    evaluation = addie_output.get("evaluation") or {}
    development = addie_output.get("development") or {}
    sources = [evaluation.get("quiz_items"), development.get("assessment_tools")]
    summative = evaluation.get("summative")
    if isinstance(summative, dict):
        sources.append(summative.get("assessment_tools"))
    pairs: list[tuple[str, str]] = []
    for items in sources:
        if not isinstance(items, list):
            continue
        for entry in items:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("question") or "").strip()
            target = str(
                entry.get("aligned_objective") or entry.get("objective_id") or ""
            ).strip()
            if text and target:
                pairs.append((text, target))
    return pairs


# ---------------------------------------------------------------------------
# Score container
# ---------------------------------------------------------------------------


@dataclass
class AlignmentScore:
    """Alignment endpoints and indices; every numeric value lies in [0, 1].

    ``objective_assessment_alignment`` (continuous, threshold-free) is the
    primary endpoint. The protocol uses **no similarity threshold anywhere** —
    coverage of each objective by assessments/activities/evaluation is the
    mean-max cosine, not a thresholded rate. There is no composite scalar; the
    criteria are reported separately.
    """

    # Primary endpoint: mean over objectives of the max cosine to any
    # assessment item (continuous, threshold-free) -- the objective<->
    # assessment relation at the heart of constructive alignment (Biggs).
    objective_assessment_alignment: Optional[float] = None
    # The other two legs of aligned(o), same continuous form.
    objective_activity_alignment: Optional[float] = None
    objective_evaluation_alignment: Optional[float] = None
    # Categorical criterion (no similarity threshold anywhere).
    objective_cognitive_congruence: Optional[float] = None
    # Descriptive indices.
    porter: dict[str, Optional[float]] = field(default_factory=dict)
    porter_mean: Optional[float] = None
    webb_bloom_consistency: Optional[float] = None
    assessment_precision: Optional[float] = None
    counts: dict[str, int] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_assessment_alignment": self.objective_assessment_alignment,
            "objective_activity_alignment": self.objective_activity_alignment,
            "objective_evaluation_alignment": self.objective_evaluation_alignment,
            "objective_cognitive_congruence": self.objective_cognitive_congruence,
            "porter": self.porter,
            "porter_mean": self.porter_mean,
            "webb_bloom_consistency": self.webb_bloom_consistency,
            "assessment_precision": self.assessment_precision,
            "counts": self.counts,
            "details": self.details,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Objective-level endpoint aggregation (shared by scoring and re-aggregation)
# ---------------------------------------------------------------------------

#: The four objective-level endpoints, derivable purely from the per-objective
#: signals stored in ``details["objectives"]`` + the ``counts`` dict.
OBJECTIVE_ENDPOINTS = (
    "objective_assessment_alignment",
    "objective_activity_alignment",
    "objective_evaluation_alignment",
    "objective_cognitive_congruence",
)


def objective_endpoints(
    obj_details: list[dict[str, Any]], counts: dict[str, int]
) -> tuple[dict[str, Optional[float]], list[str]]:
    """Aggregate the objective-level endpoints from per-objective signals.

    Pure function of the already-recorded per-objective best-match
    similarities and Bloom congruence flags (plus ``counts`` to distinguish
    "empty counterpart set" from "objectives all missed it"). This is what
    lets a metric-DEFINITION change (threshold removal, dropping a criterion,
    a different aggregation) be applied by re-aggregating a saved score
    instead of re-encoding — see :func:`reaggregate`. Only a change to the
    stored SIGNALS themselves (encoder, Bloom classifier, what gets extracted)
    requires a real re-score.
    """
    notes: list[str] = []
    n = len(obj_details)
    out: dict[str, Optional[float]] = dict.fromkeys(OBJECTIVE_ENDPOINTS, None)
    if n == 0:
        out["objective_assessment_alignment"] = 0.0
        out["objective_activity_alignment"] = 0.0
        out["objective_evaluation_alignment"] = 0.0
        return out, notes
    for endpoint, signal, count_key, label in (
        ("objective_assessment_alignment", "best_similarity",
         "assessment", "assessment items"),
        ("objective_activity_alignment", "best_activity_similarity",
         "activities", "activities"),
        ("objective_evaluation_alignment", "best_evaluation_similarity",
         "evaluation", "evaluation texts"),
    ):
        if counts.get(count_key, 0):
            out[endpoint] = sum((d.get(signal) or 0.0) for d in obj_details) / n
        else:
            out[endpoint] = 0.0
            notes.append(f"{endpoint} = 0.0: no {label} found")
    congruent = [
        d["cognitively_congruent"]
        for d in obj_details
        if d.get("cognitively_congruent") is not None
    ]
    if congruent:
        out["objective_cognitive_congruence"] = sum(congruent) / len(congruent)
    else:
        notes.append(
            "objective_cognitive_congruence undefined: no objective with "
            "Bloom-classifiable objective and best-item levels"
        )
    return out, notes


def reaggregate(score_dict: dict[str, Any]) -> dict[str, Any]:
    """Recompute the objective-level endpoints of a saved score in place.

    ``score_dict`` is one agent's :meth:`AlignmentScore.to_dict` payload from
    an ``alignment_scores.json`` file. Reads its ``details.objectives`` +
    ``counts`` and overwrites the four :data:`OBJECTIVE_ENDPOINTS` — no
    encoder, no Bloom call, no network. The descriptive indices
    (``porter_mean``, ``webb_bloom_consistency``, ``assessment_precision``)
    are left untouched: they depend on full matrices not stored per objective,
    and are unaffected by objective-level metric-definition changes.
    """
    obj_details = (score_dict.get("details") or {}).get("objectives") or []
    counts = score_dict.get("counts") or {}
    endpoints, _ = objective_endpoints(obj_details, counts)
    # Drop any endpoint keys that no longer exist (e.g. removed criteria).
    for stale in ("objective_measurability", "full_alignment_rate",
                  "objective_assessment_coverage", "objective_activity_coverage",
                  "objective_evaluation_coverage"):
        score_dict.pop(stale, None)
    score_dict.update(endpoints)
    return score_dict


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class AlignmentEvaluator:
    """Deterministic constructive-alignment evaluator for one ADDIE output.

    **No similarity threshold anywhere** — the objective<->item relations are
    scored as continuous mean-max cosine, not thresholded rates.

    Objective-level endpoints:

    - ``objective_assessment_alignment`` (**primary**): mean over objectives
      of the max cosine similarity to any assessment item — the
      objective<->assessment relation at the heart of constructive alignment
      (Biggs). Continuous, threshold-free.
    - ``objective_activity_alignment`` / ``objective_evaluation_alignment``:
      the same continuous mean-max cosine against activities / evaluation-
      phase texts (the ``supports`` / ``evaluates`` legs of aligned(o)).
    - ``objective_cognitive_congruence``: share of objectives whose
      argmax-closest assessment item has Bloom level >= the objective's
      level, over objectives with both levels classifiable (objective-centric,
      argmax picks the closest item with no threshold); undefined when no
      objective qualifies.

    Descriptive indices (also threshold-free): Porter alignment index,
    item-centric ``webb_bloom_consistency``, ``assessment_precision``
    (reverse direction: mean over assessment items of max cosine to any
    objective).

    Conventions for degenerate inputs (all values stay in [0,1]):

    - No objectives: the alignment endpoints are 0.0; congruence and the
      descriptive indices are undefined.
    - Objectives present but a counterpart set empty: that alignment endpoint
      and the Porter index for the pair are 0.0; ``assessment_precision`` is
      undefined.
    - Bloom-dependent values use only texts the classifier can label; when
      nothing qualifies they are undefined with a note.

    ``details`` additionally carries per-objective best-match info (audit
    trail) and ``validation`` (declared-link agreement — validity evidence).
    There is NO composite scalar and NO threshold.
    """

    PAIRS = ("assessment", "activities", "evaluation")

    def __init__(
        self,
        encoder: Optional[TextEncoder] = None,
        bloom_classifier: Optional[BloomClassifier] = None,
    ) -> None:
        self.encoder = encoder or TfidfCharNgramEncoder()
        self.bloom = bloom_classifier or LexiconBloomClassifier()

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
            score.notes.append(
                "no learning objectives found; objective-level endpoints = 0.0"
            )
            score.objective_assessment_alignment = 0.0
            score.objective_activity_alignment = 0.0
            score.objective_evaluation_alignment = 0.0
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

        # Similarity between objectives and each counterpart set.
        sim = [
            [_cosine(ov, av) for av in set_vecs["assessment"]]
            for ov in set_vecs["objectives"]
        ]
        sim_act = [
            [_cosine(ov, av) for av in set_vecs["activities"]]
            for ov in set_vecs["objectives"]
        ]
        sim_eval = [
            [_cosine(ov, ev) for ev in set_vecs["evaluation"]]
            for ov in set_vecs["objectives"]
        ]

        # Webb Bloom-consistency (levels from text, correspondence by argmax
        # similarity to a classifiable objective).
        score.webb_bloom_consistency = self._bloom_consistency(
            sim, levels["objectives"], levels["assessment"], score.notes
        )

        # Assessment precision (reverse direction: does each assessment item
        # bind to some objective?). Threshold-free.
        if assessments:
            per_item_max = [
                max(sim[i][j] for i in range(len(sim))) for j in range(len(assessments))
            ]
            score.assessment_precision = sum(per_item_max) / len(per_item_max)

        obj_details = self._objective_details(
            objectives, assessments, activities, evaluations, sim, sim_act,
            sim_eval, levels, topic_ids["objectives"], topics,
        )
        score.details["objectives"] = obj_details
        score.details["sanity"] = self._sanity_report(objectives, levels["objectives"])

        # Continuous alignment endpoints, aggregated from the per-objective
        # signals already recorded in ``details["objectives"]`` (single source
        # of truth). Split out so a metric-definition change can re-aggregate
        # from a saved score without re-encoding (see :func:`reaggregate`).
        endpoints, notes = objective_endpoints(obj_details, score.counts)
        for name, value in endpoints.items():
            setattr(score, name, value)
        score.notes += notes

        score.details["validation"] = self._declared_link_validation(
            addie_output, objectives, assessments, sim
        )
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
        activities: list[str],
        evaluations: list[str],
        sim: list[list[float]],
        sim_act: list[list[float]],
        sim_eval: list[list[float]],
        levels: dict[str, list[Optional[int]]],
        objective_topics: list[int],
        topics: list[str],
    ) -> list[dict[str, Any]]:
        details = []
        for i, objective in enumerate(objectives):
            row = sim[i] if i < len(sim) else []
            best_j = max(range(len(row)), key=lambda j: row[j]) if row else None
            row_act = sim_act[i] if i < len(sim_act) else []
            best_a = (
                max(range(len(row_act)), key=lambda j: row_act[j]) if row_act else None
            )
            row_ev = sim_eval[i] if i < len(sim_eval) else []
            best_e = (
                max(range(len(row_ev)), key=lambda j: row_ev[j]) if row_ev else None
            )
            obj_level = levels["objectives"][i]
            best_item_level = (
                levels["assessment"][best_j] if best_j is not None else None
            )
            # Congruence uses the argmax-closest assessment item (no
            # threshold); undefined when either level is unclassifiable.
            congruent = (
                best_item_level >= obj_level
                if obj_level is not None and best_item_level is not None
                else None
            )
            details.append(
                {
                    "text": objective["text"],
                    "topic": topics[objective_topics[i]],
                    "lexicon_level": obj_level,
                    "declared_level": objective["declared_level"],
                    "best_item": assessments[best_j] if best_j is not None else None,
                    "best_similarity": round(row[best_j], 4) if best_j is not None else None,
                    "best_item_level": best_item_level,
                    "best_activity": (
                        activities[best_a] if best_a is not None else None
                    ),
                    "best_activity_similarity": (
                        round(row_act[best_a], 4) if best_a is not None else None
                    ),
                    "best_evaluation": (
                        evaluations[best_e] if best_e is not None else None
                    ),
                    "best_evaluation_similarity": (
                        round(row_ev[best_e], 4) if best_e is not None else None
                    ),
                    "cognitively_congruent": congruent,
                }
            )
        return details

    @staticmethod
    def _declared_link_validation(
        addie_output: dict,
        objectives: list[dict[str, Any]],
        assessments: list[str],
        sim: list[list[float]],
    ) -> dict[str, Any]:
        """Validation evidence: does similarity matching reproduce the
        objective links the output itself declared? (Links are never used in
        scoring — they serve as noisy convergent-validity labels.)"""
        links = extract_declared_links(addie_output)
        obj_index = {
            o["id"]: i for i, o in enumerate(objectives) if o.get("id")
        }
        asm_index = {text: j for j, text in enumerate(assessments)}
        checked = agree = 0
        for text, target in links:
            j = asm_index.get(text)
            i_decl = obj_index.get(target)
            if j is None or i_decl is None or not objectives:
                continue
            col = [sim[i][j] for i in range(len(objectives))]
            checked += 1
            agree += max(range(len(col)), key=col.__getitem__) == i_decl
        return {
            "declared_links": len(links),
            "declared_links_checked": checked,
            "declared_link_agreement": agree / checked if checked else None,
        }

    @staticmethod
    def _sanity_report(
        objectives: list[dict[str, Any]], lexicon_levels: list[Optional[int]]
    ) -> dict[str, Any]:
        """Validation evidence: agreement between the lexicon classifier and
        the agent's self-declared ``level`` fields. Never enters scoring."""
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
