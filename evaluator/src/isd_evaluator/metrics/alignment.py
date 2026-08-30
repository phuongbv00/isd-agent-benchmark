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

**A panel of signals, with NO primary endpoint.** Alignment is a
multi-faceted construct; collapsing it into one number is the mistake that
dropping the composite already fixed, and labelling one facet "primary" is
just an implicit composite. Every signal below is reported for every
comparison — win or lose — so there is no selective reporting. The panel
spans **two mechanically independent instrument families**, each with its own
sensitivity axis, so that no single family carries the construct on its own:

The panel is :data:`PANEL_SIGNALS`, which is the single source of truth for
its own size — prose that repeats a count goes stale the moment a signal moves,
so nothing here states one.

Family A — *textual correspondence* (measured by the encoder). Exactly the
three edges of the constructive-alignment triad (Biggs), each mirroring a core
relation of the harness graph:

- **objective_assessment_similarity** — mean over learning objectives of the
  maximum rectified cosine to any assessment item; the ``measures`` leg of the
  paper's aligned(o) predicate, the relation at the heart of constructive
  alignment (C2 ``assessed_by``).
- **objective_activity_similarity** — the same mean-max form against
  activities: the ``supports`` leg (C3 ``practiced_by``).
- **activity_assessment_similarity** — the third edge, mean over activities of
  the max rectified cosine to any assessment item (C4 ``prepares_for``). It
  does NOT involve objectives, so it stays defined on outputs whose objectives
  failed to parse.

Family B — *cognitive demand* (measured by the Bloom classifier, no
similarity involved):

- **objective_cognitive_congruence** — share of objectives whose
  argmax-closest assessment item has Bloom level >= the objective's level
  (argmax, no threshold), over objectives with both levels classifiable.
- **Porter Alignment Index** (Porter 2002; Fulmer 2011)
  ``P = 1 - 0.5 * sum(|X_ij - Y_ij|)`` between two normalized
  content-topic x Bloom-level distribution matrices, over the SAME three
  triad edges family A measures (see :data:`PORTER_PAIRS`), so the two
  families differ by instrument and by nothing else.
- **webb_bloom_consistency** (Webb 1997/1999), item-centric.

Naming rule: a signal is named after what it computes. A metric name is never
read inside the paragraph that defines it — it is read on a figure axis, a
table header, a slide bullet — so a name that asserts more than the operation
does the work only the validity evidence is entitled to do. "Alignment" is
therefore the name of the protocol and of the aligned(o) predicate legs, never
of a metric; and nothing here is called "precision", which would imply a
retrieval frame that needs a threshold this protocol does not have.

Design principles enforced here:

(a) The whole protocol is deterministic — zero LLM calls, no threshold.
(b) Scoring reads **text only** (objective statements, questions, activity
    descriptions). Structural links (``objective_id`` / ``aligned_objective``)
    and self-declared ``level`` fields are NEVER used in scoring; they feed
    only the validation-evidence report (declared-link agreement,
    Bloom-vs-declared agreement).
(c) Correspondence between items and objectives is established via text
    similarity matching, never via ids.

Pluggable pieces:

- :class:`BloomClassifier` protocol. The protocol's **primary** arm is
  :class:`TransformerBloomClassifier` — a 6-way BERT fine-tuned on the
  EDM2022CLO corpus of Li et al. (2022), loaded from a local checkpoint (it
  never downloads), agreeing with the released expert labels at kappa 0.926 on
  a held-out split. :class:`LexiconBloomClassifier` (English Bloom verb
  lexicon, rule-based, offline, every level assignment traceable to Anderson &
  Krathwohl, kappa 0.650) is the **sensitivity arm**. Family B feeds three of the
  panel signals, so it needs an independence check exactly as family A does,
  and a rule instrument derived from the taxonomy is the strongest possible
  check on a model trained from labelled data: the two agree at only kappa
  0.665, so neither is a mirror of the other. They also differ by construction
  in coverage — the lexicon abstains (83%), the transformer always argmaxes —
  which is declared, not hidden.

  NOTE: the zero-argument default of :class:`AlignmentEvaluator` is still the
  lexicon, because the transformer needs a checkpoint that cannot be a safe
  default. "Code default" and "protocol primary" are therefore different
  things here; the scorer always passes the arm explicitly.
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
BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

_LEVEL_ALIASES = {
    # Revised taxonomy
    "remember": 1, "understand": 2, "apply": 3, "analyze": 4, "analyse": 4,
    "evaluate": 5, "create": 6,
    # Original taxonomy (occasionally emitted by agents)
    "knowledge": 1, "comprehension": 2, "application": 3, "analysis": 4,
    "synthesis": 6, "evaluation": 5,
}


def normalize_declared_level(value: Any) -> Optional[int]:
    """Map a self-declared level field to 1..6, else None.

    Self-declared levels are validation evidence only — they never enter
    scoring — so an unmappable value is reported as unclassifiable rather
    than guessed at.
    """
    if not isinstance(value, str):
        return None
    return _LEVEL_ALIASES.get(value.strip().lower())


class BloomClassifier(Protocol):
    """Text -> Bloom level (1..6), or None when undecidable.

    Implementations expose ``classify_many`` as well; the evaluator calls that
    so a batching arm is not throttled to one forward pass per text.

    ``name`` identifies which arm this is and is recorded in the score
    artifact, so a scores file always states which instrument produced its
    family-B signals instead of leaving the reader to assume the default.
    """

    name: str

    def classify(self, text: str) -> Optional[int]:  # pragma: no cover - protocol
        ...

    def classify_many(  # pragma: no cover - protocol
        self, texts: Sequence[str]
    ) -> list[Optional[int]]:
        ...


# --- Bloom verb lexicon -----------------------------------------------------
#
# TIER 1 — the citable core. The 19 cognitive processes of the Cognitive
# Process Dimension (Anderson & Krathwohl 2001, Table 5.1) plus the
# "alternative names" that table lists for each process. Level membership here
# is NOT ours to choose: it is whatever A&K assign. Two consequences that
# differ from the folk verb lists circulated as teaching handouts:
#   * comparing/contrasting/matching are UNDERSTAND (2), not Analyze;
#     categorizing is Understand too (alternative name of classifying).
#   * integrating/outlining/structuring are ANALYZE (4) (alternative names of
#     organizing), not Create.
#
# A&K alternative names deliberately NOT included, because they are polysemous
# in ordinary prose and would fire on non-instructional usage: focusing,
# selecting, mapping, representing, finding coherence, parsing, carrying out,
# constructing models, subsuming, instantiating, abstracting, interpolating,
# coordinating. Excluding them costs recall, never provenance.
_AK_PROCESSES: dict[int, list[str]] = {
    1: ["recognize", "identify", "recall", "retrieve"],
    2: ["interpret", "clarify", "paraphrase", "translate",
        "exemplify", "illustrate",
        "classify", "categorize", "categorise",
        "summarize", "summarise", "generalize", "generalise",
        "infer", "conclude", "extrapolate", "predict",
        "compare", "contrast", "match",
        "explain"],
    3: ["execute", "implement", "use"],
    4: ["differentiate", "discriminate", "distinguish",
        "organize", "organise", "integrate", "outline", "structure",
        "attribute", "deconstruct"],
    5: ["check", "detect", "monitor", "test",
        "critique", "judge"],
    6: ["generate", "hypothesize", "hypothesise",
        "plan", "design",
        "produce", "construct"],
}

# TIER 2 — conventional instructional verbs that are NOT A&K processes or
# alternative names. They are widely used in objective-writing practice and in
# the rule-based Bloom-classification literature (Omar et al. 2012 and the work
# that follows it), and are kept for recall. Provenance is explicitly weaker
# than tier 1: when a tier-2 verb conflicts with a tier-1 placement, tier 1
# wins by construction (a verb appears in exactly one level below).
_CONVENTIONAL_VERBS: dict[int, list[str]] = {
    1: ["list", "define", "name", "label", "state", "memorize", "memorise",
        "recite", "locate", "repeat"],
    2: ["describe", "discuss", "restate"],
    3: ["apply", "demonstrate", "solve", "operate", "practice", "calculate",
        "employ", "perform", "utilize", "utilise", "conduct"],
    4: ["analyze", "analyse", "examine", "diagnose", "investigate", "dissect"],
    5: ["evaluate", "justify", "assess", "validate", "verify", "defend",
        "appraise", "argue", "recommend", "prioritize", "prioritise",
        "criticize", "criticise"],
    6: ["create", "develop", "compose", "formulate", "devise", "build",
        "invent", "propose", "synthesize", "synthesise", "prototype"],
}

#: Merged lexicon actually compiled into patterns.
_EN_VERB_LEXICON: dict[int, list[str]] = {
    level: _AK_PROCESSES[level] + _CONVENTIONAL_VERBS[level]
    for level in sorted(_AK_PROCESSES)
}

# A verb must not appear at two levels: the "highest level wins" rule would
# then be decided by lexicon duplication rather than by the text.
_seen_verbs: dict[str, int] = {}
for _level, _verbs in _EN_VERB_LEXICON.items():
    for _verb in _verbs:
        if _verb in _seen_verbs:  # pragma: no cover - construction-time guard
            raise ValueError(
                f"verb {_verb!r} is at level {_seen_verbs[_verb]} and {_level}"
            )
        _seen_verbs[_verb] = _level
del _seen_verbs, _level, _verbs, _verb

# A determiner, possessive or "of" immediately before a match means the word is
# being used as a noun or a participial adjective ("the design", "their plan",
# "a test", "the proposed solution"), not as the performed action. Measured on
# the ladder corpus, this accounted for >=4.3% of all objective-level
# assignments before the guard existed. Only the IMMEDIATE left context is
# checked: allowing a gap would also suppress genuine verbs ("the learners
# create ...").
_NOUN_CONTEXT = re.compile(
    r"\b(?:the|a|an|this|that|these|those|their|its|his|her|our|your|my|"
    r"each|every|any|some|no|of)\s+$",
    re.IGNORECASE,
)

# ``of`` is the one determiner-position word that also precedes a GERUND, where
# the verb is the performed action rather than a noun: "be able/capable of
# analyzing ...", "consists of evaluating ...". Suppressing those cost the
# objective its level entirely, and "be able to"/"capable of" phrasing is
# idiomatic in objective writing, so the loss was systematic rather than
# incidental. An -ing form after ``of`` is verbal; any other form after ``of``
# ("a set of tests", "the results of the design") stays nominal and suppressed.
_OF_CONTEXT = re.compile(r"\bof\s+$", re.IGNORECASE)

# Question-form fallbacks (used only when no verb matched): plain
# recall-style interrogatives are treated as Remember-level items.
_QUESTION_FALLBACKS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\bwhat\s+(is|are)\b", re.IGNORECASE), 1),
    (re.compile(r"\btrue\s+or\s+false\b", re.IGNORECASE), 1),
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
    """Rule-based Bloom classifier over an English instructional-verb lexicon.

    English-only by design: the scored corpus is English (measured 100% across
    all agents and model sizes), so a second-language lexicon would be dead
    weight that the protocol description still has to account for. Text in
    another language simply returns ``None`` — unclassifiable, and therefore
    excluded from the Bloom-dependent quantities rather than guessed at.

    Level membership comes from :data:`_AK_PROCESSES` (Anderson & Krathwohl
    2001, Table 5.1) wherever the taxonomy defines it, extended by
    :data:`_CONVENTIONAL_VERBS` for recall.

    When verbs of several levels match, the level comes from the **leading**
    verb — the first one in the text, which in an objective ("Learners will
    <verb> ...") or an exam prompt ("Compare X and Y") is the central
    performance. This follows Webb's alignment methodology, which assigns a
    level from the central performance rather than from the most demanding verb
    appearing anywhere; a trailing verb usually names the purpose or the medium
    ("apply X to design Y"), not the demand being assessed.

    The rule was chosen on evidence, not taste. Against the highest-wins rule
    this lexicon previously used, on two independent labelled corpora:

    ======================================  ==============  ==============
    corpus                                  highest wins    leading verb
    ======================================  ==============  ==============
    EDM2022 learning objectives (n=18773)   kappa 0.591     kappa 0.657
    Public exam-question set (n=8767)       kappa 0.386     kappa 0.434
    ======================================  ==============  ==============

    On the EDM2022 objectives — the corpus matching what this protocol scores —
    kappa 0.657 is "substantial" agreement and exceeds the kappa 0.63 reported
    between two trained human coders on that same dataset.

    Matches in noun position are rejected (see :data:`_NOUN_CONTEXT`).

    This is the **sensitivity** arm of the Bloom axis; the primary arm is
    :class:`TransformerBloomClassifier` (the roles were swapped on 2026-07-29,
    when the transformer measured kappa 0.926 against this one's 0.650 on a
    held-out split). What this arm buys, and why it is kept: its level
    assignment stays traceable to Anderson & Krathwohl and is bit-wise
    deterministic, neither of which holds for a softmax argmax. Note the
    coverage asymmetry between them: this one abstains (``None``) on text with
    no lexicon verb, ~84% coverage on EDM2022CLO, while the transformer always
    emits a level.
    """

    name = "lexicon"

    def classify_many(self, texts: Sequence[str]) -> list[Optional[int]]:
        """Same entry point as the transformer arm; no batching to gain here."""
        return [self.classify(t) for t in texts]

    def __init__(self) -> None:
        self._patterns: list[tuple[re.Pattern[str], int]] = []
        for level, verbs in _EN_VERB_LEXICON.items():
            alternation = "|".join(_en_inflections(v) for v in verbs)
            pattern = re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)
            self._patterns.append((pattern, level))

    @staticmethod
    def _in_noun_position(text: str, match: re.Match[str]) -> bool:
        # Only the immediate left context matters, and the patterns are
        # ``$``-anchored, so bound the search rather than slicing a prefix
        # copy per match.
        start = match.start()
        left_from = max(0, start - 24)
        if not _NOUN_CONTEXT.search(text, left_from, start):
            return False
        if _OF_CONTEXT.search(text, left_from, start):
            return not match.group(0).lower().endswith("ing")
        return True

    def _verbal_matches(self, text: str) -> list[tuple[int, int]]:
        """(offset, level) of every match in non-noun position, in text order."""
        found = [
            (m.start(), level)
            for pattern, level in self._patterns
            for m in pattern.finditer(text)
            if not self._in_noun_position(text, m)
        ]
        return sorted(found)

    def classify(self, text: str) -> Optional[int]:
        if not text:
            return None
        matches = self._verbal_matches(text)
        if matches:
            return matches[0][1]
        for pattern, level in _QUESTION_FALLBACKS:
            if pattern.search(text):
                return level
        return None


class TransformerBloomClassifier:
    """Bloom classifier backed by a locally fine-tuned sequence classifier.

    The **primary arm** of the Bloom axis since 2026-07-29 (kappa 0.926 vs the
    lexicon's 0.650 on a held-out split), but it does not replace the lexicon:
    the sweep's question is not "which classifier is more accurate" but "does
    the agent ranking depend on the choice of classifier at all". Two
    instruments derived from completely separate sources — the lexicon from
    Anderson & Krathwohl's Table 5.1, this one from 21,380 human labels —
    agreeing on the ranking is stronger evidence than either instrument alone at
    any kappa. The cost of leading with this one, which must be declared: a
    per-case level is no longer traceable to Anderson & Krathwohl, and family B
    stops being bit-wise deterministic (a discrete argmax, so floating-point
    noise can flip a label rather than nudge a number).

    Train one with ``evaluator/scripts/train_bloom_6way_classifier.py`` (default
    recipe: ``bert-base-uncased`` on the single-label subset of EDM2022CLO,
    after Li et al. 2022). Loads a checkpoint from ``model_path`` only if it
    already exists on disk; it never downloads.

    Unlike the lexicon this classifier **never abstains** — argmax always
    returns a level, so family-B denominators differ between the two arms by
    construction. That is declared rather than patched: a ranking that
    survives two instruments with 84% and 100% coverage is a stronger result
    than one tuned to make the arms comparable.
    """

    name = "transformer"

    def __init__(
        self,
        model_path: str,
        max_length: int = 128,
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        import os

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Bloom classifier checkpoint not found at '{model_path}'. "
                "Fine-tune one with evaluator/scripts/train_bloom_6way_classifier.py "
                "or fall back to LexiconBloomClassifier."
            )
        from transformers import (  # noqa: PLC0415 - optional heavy dep
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        import torch  # noqa: PLC0415 - optional heavy dep

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.eval()
        self._max_length = max_length
        self._batch_size = batch_size
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self._model.to(device)
        # Scoring calls this once per text per scenario, and the same objective
        # text recurs across scenarios and arms, so a plain memo removes ~30% of
        # the forward passes within one invocation. Levels are a pure function
        # of the text for a fixed checkpoint, so memoising cannot skew a score.
        self._memo: dict[str, Optional[int]] = {}

    def classify(self, text: str) -> Optional[int]:
        return self.classify_many([text])[0]

    def classify_many(self, texts: Sequence[str]) -> list[Optional[int]]:
        """Batched classification — the form the evaluator should call.

        One text at a time on CPU costs hours over a full ladder (measured: 58
        texts/s vs 464k calls per arm). Batching on the local GPU is what makes
        the transformer arm affordable to run on every sweep arm.
        """
        import torch  # noqa: PLC0415 - optional heavy dep

        out: list[Optional[int]] = [None] * len(texts)
        pending = [
            i for i, t in enumerate(texts)
            if t and t not in self._memo
        ]
        for i, t in enumerate(texts):
            if t and t in self._memo:
                out[i] = self._memo[t]
        for start in range(0, len(pending), self._batch_size):
            idx = pending[start:start + self._batch_size]
            batch = [texts[i] for i in idx]
            inputs = self._tokenizer(
                batch, return_tensors="pt", truncation=True, padding=True,
                max_length=self._max_length,
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            levels = (logits.argmax(dim=-1) + 1).tolist()
            for i, level in zip(idx, levels):
                out[i] = int(level)
                self._memo[texts[i]] = int(level)
        return out


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
    """Sentence-embedding encoder running locally, with the same vector cache.

    Requires the ``alignment`` extra (``pip install isd-evaluator[alignment]``).
    Same cache contract as :class:`OpenAIAPIEncoder` — cache key is
    ``sha1(model[@revision] + text)`` — so a locally-embedded arm is resumable
    and re-analysable exactly like a served one.

    **One backend per model.** The cache key does NOT record whether a vector
    came from a pod or from here, so embedding one model both ways would mix
    two numerically different sources under one key, invisibly. Each preset in
    the scorer therefore pins its backend: the primary encoder is served, the
    smaller sweep arms run here.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        cache_path: Optional[Path] = None,
        cache_only: bool = False,
        revision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model_name
        self.model_name = model_name        # kept: older callers use this name
        self.revision = revision or None
        self.batch_size = batch_size
        self.cache_only = cache_only
        self.device = device
        self._cache = _EmbeddingCache(cache_path) if cache_path else None
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            kwargs = {}
            if self.revision:
                kwargs["revision"] = self.revision
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        def compute(pending: list[int]):
            model = self._load()
            for start in range(0, len(pending), self.batch_size):
                idx = pending[start:start + self.batch_size]
                vecs = model.encode([texts[i] for i in idx],
                                    normalize_embeddings=True)
                yield idx, [list(map(float, row)) for row in vecs]

        return _encode_with_cache(
            texts, cache=self._cache, model=self.model, revision=self.revision,
            cache_only=self.cache_only, compute_chunks=compute,
        )


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


def _encode_with_cache(
    texts: Sequence[str],
    *,
    cache: Optional[_EmbeddingCache],
    model: str,
    revision: Optional[str],
    cache_only: bool,
    compute_chunks,
) -> list[list[float]]:
    """Cache lookup -> compute the misses -> write back, shared by both encoders.

    ``compute_chunks(pending_indices)`` yields ``(indices, vectors)`` chunks.
    Chunking is the caller's business (HTTP batches dispatched round-robin for
    the served encoder, forward-pass batches for the local one); everything
    about the cache is here, so the two backends cannot drift apart in how a
    key is formed or when a vector is persisted.

    Vectors are written per chunk, not at the end: a long run that dies part
    way keeps what it already embedded.

    Empty/whitespace texts never reach the model — they become zero vectors of
    whatever dimension the run establishes.
    """
    vectors: list[Optional[list[float]]] = [None] * len(texts)
    pending: list[int] = []
    dim: Optional[int] = None
    keys = [cache.key(model, t, revision) if cache else "" for t in texts]
    cached = cache.get_many(
        [keys[i] for i, t in enumerate(texts) if t.strip()]
    ) if cache else {}
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        hit = cached.get(keys[i])
        if hit is not None:
            vectors[i] = hit
            dim = dim or len(hit)
        else:
            pending.append(i)

    if pending and cache_only:
        raise RuntimeError(
            f"cache_only=True but {len(pending)} of {len(texts)} texts are "
            f"not in the embedding cache for model {model!r}. Re-run the "
            "primary scoring pass with the encoder available first, or drop "
            "cache_only."
        )

    for idx, embedded in compute_chunks(pending):
        for i, vec in zip(idx, embedded):
            vectors[i] = vec
            dim = dim or len(vec)
        if cache:
            cache.put_many({keys[i]: vec for i, vec in zip(idx, embedded)})

    width = dim or 1
    return [vec if vec is not None else [0.0] * width for vec in vectors]


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

        def compute(pending: list[int]):
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
                    yield from pool.map(run, enumerate(batches))
            else:
                for batch_idx in batches:
                    yield batch_idx, self._embed_batch(
                        [texts[i] for i in batch_idx])

        return _encode_with_cache(
            texts, cache=self._cache, model=self.model, revision=self.revision,
            cache_only=self.cache_only, compute_chunks=compute,
        )


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


def _l2_normalize(vectors: list[list[float]]) -> list[list[float]]:
    """Unit-length copies, so cosine reduces to a dot product.

    Every similarity matrix here compares one set against another, so an
    unnormalized :func:`_cosine` re-derives each vector's norm once per element
    of the opposing set — with 4096-dim vectors and four matrices per scenario
    that dominated scoring time. Normalizing once up front is the same
    arithmetic with the redundant passes removed. A zero vector stays zero, so
    :func:`_cosine_unit` still floors it at 0.0.
    """
    out: list[list[float]] = []
    for vec in vectors:
        norm = math.sqrt(sum(x * x for x in vec))
        out.append([x / norm for x in vec] if norm else list(vec))
    return out


def _cosine_unit(a: Sequence[float], b: Sequence[float]) -> float:
    """Rectified cosine for vectors already unit-normalized by
    :func:`_l2_normalize` — identical semantics to :func:`_cosine`, one pass."""
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


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
    """The panel of alignment signals; every numeric value lies in [0, 1].

    **No primary endpoint** — :data:`PANEL_SIGNALS` is a flat panel, each
    signal answering a different question, all reported for every comparison. The
    protocol uses **no similarity threshold anywhere**: the correspondence
    between an objective and the assessments/activities is a
    mean-max cosine, not a thresholded rate. There is no composite scalar.
    """

    # --- Family A: textual correspondence (encoder) ---
    # The three edges of the constructive-alignment triad (Biggs), each the
    # mean over the left set of the max rectified cosine into the right set.
    objective_assessment_similarity: Optional[float] = None   # C2 assessed_by
    objective_activity_similarity: Optional[float] = None     # C3 practiced_by
    activity_assessment_similarity: Optional[float] = None    # C4 prepares_for
    # --- Family B: cognitive demand (Bloom classifier, no similarity) ---
    objective_cognitive_congruence: Optional[float] = None
    porter: dict[str, Optional[float]] = field(default_factory=dict)
    porter_mean: Optional[float] = None
    webb_bloom_consistency: Optional[float] = None
    counts: dict[str, int] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_assessment_similarity": self.objective_assessment_similarity,
            "objective_activity_similarity": self.objective_activity_similarity,
            "activity_assessment_similarity": self.activity_assessment_similarity,
            "objective_cognitive_congruence": self.objective_cognitive_congruence,
            "porter": self.porter,
            "porter_mean": self.porter_mean,
            "webb_bloom_consistency": self.webb_bloom_consistency,
            "counts": self.counts,
            "details": self.details,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Objective-level endpoint aggregation (shared by scoring and re-aggregation)
# ---------------------------------------------------------------------------

#: The four objective-level signals, derivable purely from the per-objective
#: signals stored in ``details["objectives"]`` + the ``counts`` dict.
OBJECTIVE_ENDPOINTS = (
    "objective_assessment_similarity",
    "objective_activity_similarity",
    "objective_cognitive_congruence",
)

#: The reported panel, in reporting order, as (signal, instrument family).
#: The single source of truth for "what gets reported" -- scoring, pooling,
#: tables and figures all read this so a signal can never be reported by one
#: stage and silently dropped by another. Flat by construction: there is no
#: primary entry and nothing here is a "secondary" index.
PANEL_SIGNALS: tuple[tuple[str, str], ...] = (
    # Family A is exactly the three edges of the constructive-alignment triad
    # (Biggs): ILO-AT, ILO-TLA, TLA-AT. Each one mirrors a core relation of the
    # harness graph (C2, C3, C4), so no panel signal measures something the
    # ontology does not model.
    ("objective_assessment_similarity", "correspondence"),   # C2 assessed_by
    ("objective_activity_similarity", "correspondence"),     # C3 practiced_by
    ("activity_assessment_similarity", "correspondence"),    # C4 prepares_for
    ("objective_cognitive_congruence", "cognitive"),
    ("porter_mean", "cognitive"),
    ("webb_bloom_consistency", "cognitive"),
)

#: The three edges of the constructive-alignment triad, as
#: ``(key, left set, right set)``. Porter compares the topic x Bloom
#: distribution of the left set against the right one.
#:
#: Family B measures the SAME three edges family A does. It used to hold every
#: set against ``objectives``, which predated C4 joining the panel and left the
#: activity<->assessment edge measured by textual correspondence alone while
#: the other two edges were measured by both instrument families. Mirroring the
#: edges makes the two families differ by INSTRUMENT and by nothing else, which
#: is the whole point of having two of them.
PORTER_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("objective_assessment", "objectives", "assessment"),   # C2 assessed_by
    ("objective_activity", "objectives", "activities"),     # C3 practiced_by
    ("activity_assessment", "activities", "assessment"),    # C4 prepares_for
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
        # Every objective-side endpoint floors at 0.0, congruence included.
        # Congruence is "share of objectives whose best item is at or above
        # their level" over an empty set of objectives -- an agent failure,
        # not a classifier limitation, so it scores like the similarity legs
        # rather than dropping out of the pooled denominator. The undefined
        # case below (objectives exist, none Bloom-classifiable on both sides)
        # is the instrument limitation and stays None.
        for endpoint in OBJECTIVE_ENDPOINTS:
            out[endpoint] = 0.0
        notes.append("objective-level endpoints = 0.0: no objectives found")
        return out, notes
    for endpoint, signal, count_key, label in (
        ("objective_assessment_similarity", "best_similarity",
         "assessment", "assessment items"),
        ("objective_activity_similarity", "best_activity_similarity",
         "activities", "activities"),
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
    ``counts`` and overwrites the :data:`OBJECTIVE_ENDPOINTS` — no encoder, no
    Bloom call, no network.

    The other three panel signals — ``activity_assessment_similarity``,
    ``porter_mean`` and ``webb_bloom_consistency`` — are left untouched,
    because they depend on full matrices that are not stored per objective.
    Note the consequence for ``activity_assessment_similarity`` in particular:
    it IS a panel signal, so an objective-level definition change applied
    through here leaves one panel signal on the old definition. When a change
    touches the panel rather than only the objective-level endpoints, re-score
    instead of re-aggregating.
    """
    obj_details = (score_dict.get("details") or {}).get("objectives") or []
    counts = score_dict.get("counts") or {}
    endpoints, _ = objective_endpoints(obj_details, counts)
    score_dict.update(endpoints)
    return score_dict


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class AlignmentEvaluator:
    """Deterministic constructive-alignment evaluator for one ADDIE output.

    **No similarity threshold anywhere** — the objective<->item relations are
    scored as continuous mean-max cosine, not thresholded rates. Emits the
    flat panel of :data:`PANEL_SIGNALS`; there is no primary endpoint.

    Family A — textual correspondence (the encoder):

    - ``objective_assessment_similarity``: mean over objectives of the max
      cosine similarity to any assessment item — the ``measures`` leg of
      aligned(o), the relation at the heart of constructive alignment
      (Biggs). Continuous, threshold-free.
    - ``objective_activity_similarity``: the same mean-max form against
      activities — the ``supports`` leg.
    - ``activity_assessment_similarity``: the third triad edge, mean over
      activities of the max cosine to any assessment item. Independent of
      objectives, so it survives an output whose objectives failed to parse.

    Family B — cognitive demand (the Bloom classifier, no similarity):

    - ``objective_cognitive_congruence``: share of objectives whose
      argmax-closest assessment item has Bloom level >= the objective's
      level, over objectives with both levels classifiable (objective-centric,
      argmax picks the closest item with no threshold); undefined when no
      objective qualifies.
    - Porter alignment index (per pair + mean) and the item-centric
      ``webb_bloom_consistency``.

    Conventions for degenerate inputs (all values stay in [0,1]):

    An ABSENT set and an UNCLASSIFIABLE set are different events and never
    share a value:

    - A set the agent did not produce is an AGENT failure and scores the 0.0
      floor, with a note. That covers no objectives, no assessment items and
      no activities, and it applies to every signal referencing the missing
      set — the Bloom-derived ones included. Leaving them undefined let the
      worst outputs drop out of the pooled denominator instead of counting
      against the agent that produced them.
    - A set that exists but carries no Bloom-classifiable text is an
      INSTRUMENT limitation and stays undefined, with a note. This is the
      only case that legitimately leaves a denominator.
    - ``activity_assessment_similarity`` never depends on objectives, so it
      is scored whenever activities and assessment items both exist.

    ``details`` additionally carries per-objective best-match info (audit
    trail) and ``validation`` (declared-link agreement — validity evidence).
    There is NO composite scalar and NO threshold.
    """

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
        topics = extract_topics(scenario)

        score.counts = {
            "objectives": len(objectives),
            "assessment": len(assessments),
            "activities": len(activities),
            "topics": len(topics),
        }

        if not objectives:
            # NOT an early return. C4 (activity<->assessment) does not involve
            # objectives at all, and Porter/Webb still describe the counterpart
            # sets, so bailing out here used to leave four of the six panel
            # signals as None on exactly the outputs that failed hardest --
            # which then dropped out of the pooled means instead of scoring
            # against the agent that produced them. Fall through; every block
            # below is guarded for an empty objective set.
            score.notes.append(
                "no learning objectives found; objective-level signals = 0.0"
            )

        objective_texts = [o["text"] for o in objectives]
        sets: dict[str, list[str]] = {
            "objectives": objective_texts,
            "assessment": assessments,
            "activities": activities,
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
        # Normalize once here so every similarity below is a bare dot product;
        # see :func:`_l2_normalize`.
        vectors = _l2_normalize(self.encoder.encode(all_texts))
        topic_vecs = vectors[: len(topics)]
        set_vecs = {
            name: vectors[start:end] for name, (start, end) in offsets.items()
        }

        levels = {
            name: self.bloom.classify_many(texts)
            for name, texts in sets.items()
        }
        topic_ids = {
            name: [self._argmax_topic(vec, topic_vecs) for vec in vecs]
            for name, vecs in set_vecs.items()
        }

        # Porter alignment index over the same three triad edges family A
        # measures (see PORTER_PAIRS).
        matrices = {
            name: self._porter_matrix(topic_ids[name], levels[name], len(topics))
            for name in sets
        }
        for key, left, right in PORTER_PAIRS:
            # An ABSENT set and an UNCLASSIFIABLE set are different events and
            # must not collapse to the same value. A set the agent never
            # produced is an agent failure and scores the 0.0 floor, exactly as
            # the family-A legs do for the same input; a set that exists but
            # carries no Bloom-classifiable verb is an instrument limitation
            # and is genuinely undefined. Conflating them let a total parse
            # failure leave the pooled mean instead of scoring against it.
            missing = [name for name in (left, right) if not sets[name]]
            if missing:
                score.porter[key] = 0.0
                score.notes.append(
                    f"porter[{key}] = 0.0: no {' and no '.join(missing)} found")
            elif matrices[left] is None or matrices[right] is None:
                score.porter[key] = None
                score.notes.append(
                    f"porter[{key}] undefined: no Bloom-classifiable text"
                )
            else:
                score.porter[key] = self._porter_index(
                    matrices[left], matrices[right])
        defined = [v for v in score.porter.values() if v is not None]
        score.porter_mean = sum(defined) / len(defined) if defined else None

        # Similarity between objectives and each counterpart set.
        sim = [
            [_cosine_unit(ov, av) for av in set_vecs["assessment"]]
            for ov in set_vecs["objectives"]
        ]
        sim_act = [
            [_cosine_unit(ov, av) for av in set_vecs["activities"]]
            for ov in set_vecs["objectives"]
        ]

        # Webb Bloom-consistency (levels from text, correspondence by argmax
        # similarity to a classifiable objective).
        score.webb_bloom_consistency = self._bloom_consistency(
            sim, levels["objectives"], levels["assessment"], score.notes
        )

        # C4, activity prepares_for assessment: the third edge of the
        # constructive-alignment triad (Biggs). Mean over activities of the max
        # rectified cosine to any assessment item -- same form as the two
        # objective-side legs, so all three edges are measured the same way.
        if activities and assessments:
            sim_act_asm = [
                [_cosine_unit(av, mv) for mv in set_vecs["assessment"]]
                for av in set_vecs["activities"]
            ]
            per_activity_max = [max(row) for row in sim_act_asm if row]
            if per_activity_max:
                score.activity_assessment_similarity = (
                    sum(per_activity_max) / len(per_activity_max))
        else:
            # Either side empty -- including BOTH empty, which used to fall
            # through every branch and leave the signal undefined, so the
            # emptiest possible output was the one excluded from the mean
            # rather than scored 0.0. Same convention as porter below.
            score.activity_assessment_similarity = 0.0
            missing = " and ".join(
                name for name, present in
                (("activities", activities), ("assessment items", assessments))
                if not present
            )
            score.notes.append(
                f"activity_assessment_similarity = 0.0: no {missing} found")

        obj_details = self._objective_details(
            objectives, assessments, activities, sim, sim_act,
            levels, topic_ids["objectives"], topics,
        )
        score.details["objectives"] = obj_details
        score.details["sanity"] = self._sanity_report(
            objectives, levels["objectives"], self.bloom.name
        )

        # Continuous objective-level signals, aggregated from the per-objective
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
        sims = [_cosine_unit(vec, tv) for tv in topic_vecs]
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
        if not objective_levels:
            # No objectives at all: an agent failure, scored at the floor like
            # every other absent set. Distinct from the next branch, where
            # objectives exist but the classifier cannot label any of them --
            # that one is an instrument limitation and stays undefined.
            notes.append("webb_bloom_consistency = 0.0: no objectives found")
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
            # Argmax over CLASSIFIABLE objectives only — the opposite
            # convention to the congruence leg, which argmaxes over everything
            # and then abstains. Both are deliberate; see the comment in
            # _objective_details and docs/rq2_protocol.md section 4.
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
        sim: list[list[float]],
        sim_act: list[list[float]],
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
            obj_level = levels["objectives"][i]
            best_item_level = (
                levels["assessment"][best_j] if best_j is not None else None
            )
            # Congruence uses the argmax-closest assessment item (no
            # threshold); undefined when either level is unclassifiable.
            #
            # Note the argmax runs over ALL items and then abstains if the
            # winner is unclassifiable, whereas _bloom_consistency restricts
            # its argmax to classifiable candidates. That asymmetry is
            # deliberate, not an oversight: the closest item is the one that
            # REPRESENTS this objective, so failing to grade it means we
            # cannot judge the objective — substituting the next-closest
            # gradeable item would score a different item than the one the
            # objective is actually assessed by. The cost is that this
            # denominator shrinks with the arm's coverage (never for the
            # transformer, 12.0% of objectives for the lexicon, measured over
            # the r1 rung). Declared in docs/rq2_protocol.md section 4, same
            # as the coverage gap between the two arms.
            congruent = (
                best_item_level >= obj_level
                if obj_level is not None and best_item_level is not None
                else None
            )
            details.append(
                {
                    "text": objective["text"],
                    "topic": topics[objective_topics[i]],
                    "bloom_level": obj_level,
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
        objectives: list[dict[str, Any]],
        bloom_levels: list[Optional[int]],
        classifier: str,
    ) -> dict[str, Any]:
        """Validation evidence: agreement between the Bloom classifier of the
        arm being scored and the agent's self-declared ``level`` fields. Never
        enters scoring.

        Keys say "bloom", not "lexicon": with two arms on the Bloom axis the
        old ``lexicon_*`` names would state a falsehood in the transformer arm.
        ``classifier`` records which arm produced these numbers.
        """
        both = [
            (obj["declared_level"], level)
            for obj, level in zip(objectives, bloom_levels)
            if obj["declared_level"] is not None and level is not None
        ]
        agreement = (
            sum(1 for declared, level in both if declared == level) / len(both)
            if both
            else None
        )
        return {
            "classifier": classifier,
            "n_objectives": len(objectives),
            "n_declared": sum(
                1 for o in objectives if o["declared_level"] is not None
            ),
            "n_bloom_classified": sum(
                1 for level in bloom_levels if level is not None
            ),
            "n_comparable": len(both),
            "bloom_vs_declared_agreement": agreement,
        }
