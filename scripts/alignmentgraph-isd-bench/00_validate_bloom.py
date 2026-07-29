#!/usr/bin/env python3
"""Validity evidence for the Bloom axis of the alignment protocol.

Always benchmarks the rule-based lexicon
(:class:`isd_evaluator.metrics.alignment.LexiconBloomClassifier`), which is the
axis's **sensitivity arm**, against a learning-outcome dataset carrying expert
Bloom labels, reporting coverage, accuracy, adjacent accuracy, Cohen's kappa
and the confusion matrix. With ``--bloom-checkpoint`` it additionally
benchmarks the **primary** arm — a single 6-way softmax head fine-tuned on the
EDM2022CLO corpus — on the same rows, and the two arms against each other.
The scored corpus is English (measured 100% across all agents and model sizes)
and the lexicon is English-only, so an English benchmark is the representative
one; no network access is needed here — you download the CSV yourself.

Input: a CSV with one text column and one label column (names configurable).
Labels may be level names of either taxonomy generation
("remember"/"knowledge", "understand"/"comprehension", ...) or integers 1-6.

Public datasets that fit (download manually, any of):
  * Yahya et al. exam-question dataset (~600 questions, 6 levels);
  * the "Bloom's Taxonomy questions" datasets on Kaggle/GitHub
    (search: bloom taxonomy question classification dataset CSV).

The two invocations that produce the shipped artifacts (run from the benchmark
root; `--bloom-checkpoint` needs `transformers`, so use the venv interpreter):

  # bloom_validation.md — lexicon only, full EDM2022CLO corpus (wide multi-label)
  python scripts/alignmentgraph-isd-bench/00_validate_bloom.py \
      ../EDM2022CLO/data/sample_full.csv --text-col Learning_outcome \
      --label-cols Remember,Understand,Apply,Analyze,Evaluate,Create \
      --out results/generated/bloom_validation.md

  # bloom_arms_validation.md — both arms, held-out split next to the checkpoint
  python scripts/alignmentgraph-isd-bench/00_validate_bloom.py \
      models/bloom-bert-edm2022/eval_split.csv --text-col text --label-col label \
      --bloom-checkpoint models/bloom-bert-edm2022 \
      --out results/generated/bloom_arms_validation.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluator" / "src"))

from isd_evaluator.metrics.alignment import (  # noqa: E402
    BLOOM_LEVELS,
    LexiconBloomClassifier,
    normalize_declared_level,
)

LEVEL_NAMES = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

#: Bloom-1956 level names (some public datasets predate the revision).
_LEGACY_LEVELS = {
    "knowledge": 1,
    "comprehension": 2,
    "application": 3,
    "analysis": 4,
    "synthesis": 6,  # revised taxonomy folds synthesis into Create
    "evaluation": 5,
}


#: Datasets that encode the level as a short prefixed code: BT1..BT6, L1..L6,
#: "level 3", "C4" (cognitive level 4). The digit is the revised-taxonomy level.
_CODED_LABEL = re.compile(r"^(?:bt|bl|l|c|level|lvl|cat|class)\s*[-_]?\s*([1-6])$")


def parse_label(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if not text:
        return None
    if text.isdigit() and 1 <= int(text) <= 6:
        return int(text)
    coded = _CODED_LABEL.match(text)
    if coded:
        return int(coded.group(1))
    if text in _LEGACY_LEVELS:
        return _LEGACY_LEVELS[text]
    return normalize_declared_level(text)


def _first_token(text: str) -> str | None:
    """First alphabetic token, lowercased. Usually the leading verb, not always."""
    match = re.match(r"[^A-Za-z]*([A-Za-z]+)", text)
    return match.group(1).lower() if match else None


def _first_token_purity(text: str, by_first_token: dict[str, Counter]) -> float:
    """Share of rows sharing this text's leading token that carry its modal label."""
    token = _first_token(text)
    counts = by_first_token.get(token) if token else None
    return max(counts.values()) / sum(counts.values()) if counts else 0.0


def _repo_relative(path: Path) -> str:
    """Path relative to the monorepo root, so reports carry no local home dir."""
    resolved = path.resolve()
    root = REPO_ROOT.parent
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        rel = os.path.relpath(resolved, root)
        # Outside the monorepo: a "../../.." chain is less readable than the path.
        return rel if not rel.startswith("..") else str(resolved)


def _is_checkpoint_eval_split(csv_path: Path, checkpoint: Path | None) -> bool:
    """True when the input CSV is the held-out split written next to CHECKPOINT."""
    if checkpoint is None:
        return False
    try:
        return (
            csv_path.name == "eval_split.csv"
            and csv_path.resolve().parent == checkpoint.resolve()
        )
    except OSError:
        return False


def _provenance_line(args, n_rows: int, n_single: int, n_multi: int, held_out: bool) -> str:
    """Corpus + split + row counts. The backbone of the validity claim."""
    rel = _repo_relative(args.csv_path)
    known_corpus = "edm2022" in rel.lower().replace("-", "")
    corpus = (
        "EDM2022CLO, the 21,380 expert-labelled learning objectives released by "
        "Li et al. (EDM 2022)"
        if known_corpus
        else f"the labelled set supplied as `{args.csv_path.name}`"
    )
    if held_out:
        split = (
            "held-out eval split written next to the transformer checkpoint by "
            "`evaluator/scripts/train_bloom_6way_classifier.py`; it is a strict subset "
            "of the corpus's single-label rows, so the lexicon figures here are nested "
            "inside the full-corpus report, not independent of it"
        )
    else:
        split = "the full labelled set as supplied, with no split held back"
    return (
        f"- Provenance: corpus = {corpus}; split = {split}. Measured file: `{rel}` "
        f"({n_rows} labelled rows: {n_single} single-label, {n_multi} multi-label)."
    )


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    n = len(pairs)
    observed = sum(1 for gold, pred in pairs if gold == pred) / n
    gold_freq = Counter(gold for gold, _ in pairs)
    pred_freq = Counter(pred for _, pred in pairs)
    expected = sum(
        (gold_freq[level] / n) * (pred_freq[level] / n) for level in range(1, 7)
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def _transformer_arm_section(
    args, labeled, lexicon, lex_pairs, lex_kappa, oracle, held_out
) -> list[str]:
    """Benchmark the PRIMARY transformer arm on the SAME rows, and the arms against each other.

    Two numbers carry this section, and they say different things. The
    transformer's own agreement with the gold labels is what makes it the
    primary arm. The arm-vs-arm agreement is the *sensitivity* statement: it
    quantifies how little the sensitivity arm mirrors the primary one, so a
    family-B result that survives both instruments is not an artifact of either.

    Coverage differs by construction: the lexicon abstains, the transformer
    always argmaxes. Reported side by side rather than equalised — a ranking
    that survives instruments with different coverage is the stronger result.
    """
    from isd_evaluator.metrics.alignment import (  # noqa: PLC0415 - optional heavy dep
        TransformerBloomClassifier,
    )

    bert = TransformerBloomClassifier(str(args.bloom_checkpoint))
    bert_pairs, both = [], []
    for text, gold in labeled:
        pred_b = bert.classify(str(text))
        pred_l = lexicon.classify(str(text))
        if pred_b is not None:
            bert_pairs.append((gold, pred_b))
        if pred_b is not None and pred_l is not None:
            both.append((pred_l, pred_b))
    b_cov = len(bert_pairs) / len(labeled) if labeled else float("nan")
    b_acc = (sum(1 for g, p in bert_pairs if g == p) / len(bert_pairs)
             if bert_pairs else float("nan"))
    b_kappa = cohen_kappa(bert_pairs) if bert_pairs else float("nan")
    arm_kappa = cohen_kappa(both) if both else float("nan")
    arm_agree = (sum(1 for a, b in both if a == b) / len(both)
                 if both else float("nan"))
    lex_acc = (sum(1 for g, p in lex_pairs if g == p) / len(lex_pairs)
               if lex_pairs else float("nan"))
    lines = [
        "",
        "## Bloom axis — transformer (primary arm) vs lexicon (sensitivity arm)",
        "",
        f"Checkpoint under test: `{_repo_relative(args.bloom_checkpoint)}`. Both arms "
        f"scored on the same {len(labeled)} single-label rows.",
        "",
        "| arm | coverage | accuracy | kappa vs gold |",
        "|---|---|---|---|",
        f"| lexicon (Anderson & Krathwohl verb rules, sensitivity arm) | "
        f"{len(lex_pairs) / len(labeled):.1%} | {lex_acc:.1%} | {lex_kappa:.3f} |",
        f"| transformer (BERT-base head fine-tuned on the EDM2022CLO corpus, PRIMARY) | "
        f"{b_cov:.1%} | {b_acc:.1%} | {b_kappa:.3f} |",
        "",
        f"- Arm-vs-arm agreement (rows both classify, n={len(both)}): "
        f"{arm_agree:.1%} raw, kappa {arm_kappa:.3f}. The arms are far from "
        "interchangeable, which is what makes the sensitivity axis informative.",
        "- The transformer never abstains, so its coverage is 100% by "
        "construction; the two arms are NOT equalised.",
        "- The transformer reuses Li et al.'s *corpus*, not their classifier design: "
        "their notebooks yield six independent binary heads or one multi-label sigmoid "
        "head, whereas this arm is a single 6-way softmax head, so every text gets "
        "exactly one level as the panel requires.",
        f"- The first-token oracle reported above ({oracle:.1%}) bounds rules keyed on "
        f"the leading token, such as the lexicon; it does not bound this arm, which "
        f"reads the whole text and scores above it ({b_acc:.1%}).",
    ]
    if held_out:
        recorded = _recorded_held_out_metrics(args.bloom_checkpoint)
        line = (
            "- Held-out status: this CSV **is** the checkpoint's own `eval_split.csv` "
            f"({len(labeled)} rows), never seen during fine-tuning, so the transformer "
            "numbers above are OUT-OF-SAMPLE."
        )
        if recorded:
            line += (
                f" Cross-check against the training run's own record "
                f"(`held_out_metrics.json`): n={recorded['n_held_out']}, accuracy "
                f"{recorded['accuracy']:.1%}, kappa {recorded['cohen_kappa']:.3f}."
            )
        lines.append(line)
    else:
        lines.append(
            "- WARNING: this CSV is not the checkpoint's `eval_split.csv`. If it "
            "overlaps the fine-tuning corpus, the transformer numbers are IN-SAMPLE "
            "and not comparable — re-run against the `eval_split.csv` written next "
            "to the checkpoint."
        )
    return lines


def _recorded_held_out_metrics(checkpoint: Path) -> dict | None:
    """Held-out metrics recorded by the training run, for cross-checking. None if absent."""
    path = checkpoint / "held_out_metrics.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    keys = ("n_held_out", "accuracy", "cohen_kappa")
    return data if all(k in data for k in keys) else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the Bloom verb lexicon against a labeled CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Labeled dataset (CSV).")
    parser.add_argument("--text-col", default="question",
                        help="Column holding the question/outcome text.")
    parser.add_argument("--label-col", default="label",
                        help="Column holding the Bloom label (name or 1-6). "
                             "Ignored in wide multi-label mode.")
    parser.add_argument(
        "--label-cols", default=None, metavar="C1,C2,...",
        help="Wide multi-label mode: six columns, one per level in ascending "
             "order, each truthy when the level applies (the EDM2022 learning-"
             "objective dataset ships in this shape). An objective may carry "
             "several levels; kappa is then computed on the single-label "
             "subset, and the multi-label rows are reported as a hit rate.",
    )
    parser.add_argument(
        "--bloom-checkpoint", type=Path, default=None, metavar="DIR",
        help="Also benchmark the PRIMARY (transformer) arm of the Bloom axis from "
             "this checkpoint (evaluator/scripts/train_bloom_6way_classifier.py) "
             "and report lexicon-vs-transformer agreement. Point the CSV at the "
             "checkpoint's eval_split.csv so the transformer's numbers are "
             "out-of-sample; the report states which case it detected.",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="Also write the markdown report to this file.")
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise SystemExit(f"Not found: {args.csv_path} (download a labeled "
                         "Bloom dataset first — see the module docstring).")
    with args.csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if args.text_col not in fields:
            raise SystemExit(
                f"Column {args.text_col!r} not in CSV (has: {fields}). "
                "Use --text-col."
            )
        if args.label_cols:
            cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]
            if len(cols) != 6:
                raise SystemExit(
                    f"--label-cols needs exactly 6 columns (levels 1..6), got {len(cols)}"
                )
            missing = [c for c in cols if c not in fields]
            if missing:
                raise SystemExit(f"--label-cols not in CSV: {missing} (has: {fields})")
            multi = [
                (r[args.text_col],
                 frozenset(i for i, c in enumerate(cols, 1) if str(r[c]).strip()))
                for r in reader
            ]
        else:
            if args.label_col not in fields:
                raise SystemExit(
                    f"Column {args.label_col!r} not in CSV (has: {fields}). "
                    "Use --label-col, or --label-cols for wide multi-label data."
                )
            multi = [
                (r[args.text_col], frozenset(g for g in [parse_label(r[args.label_col])] if g))
                for r in reader
            ]

    multi = [(t, g) for t, g in multi if g and str(t).strip()]
    if not multi:
        raise SystemExit("No usable labeled rows found.")
    # Single-label rows carry an unambiguous gold level; kappa and the confusion
    # matrix are defined only there. Multi-label rows are scored separately as a
    # hit rate, since a single-level prediction cannot be "wrong" about an
    # objective the annotators themselves placed at two levels.
    labeled = [(t, next(iter(g))) for t, g in multi if len(g) == 1]
    n_multi = len(multi) - len(labeled)

    clf = LexiconBloomClassifier()
    pairs: list[tuple[int, int]] = []
    unclassified = 0
    for text, gold in labeled:
        pred = clf.classify(str(text))
        if pred is None:
            unclassified += 1
        else:
            pairs.append((gold, pred))

    coverage = len(pairs) / len(labeled)
    accuracy = sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0.0
    adjacent = (
        sum(1 for g, p in pairs if abs(g - p) <= 1) / len(pairs) if pairs else 0.0
    )
    kappa = cohen_kappa(pairs) if pairs else float("nan")

    # The lexicon's accuracy is uninterpretable without the two reference points
    # below. Floor: always predict the most frequent level. First-token oracle: a
    # cheat that sees the gold labels and assigns each leading token its single
    # best level — the best attainable by any rule keyed on that token, which is
    # what the lexicon is. It is NOT an upper bound on classifiers that read the
    # whole text (the transformer arm exceeds it). The gap between the oracle and
    # 100% is gold-label ambiguity, not classifier error: the same leading token
    # carries different labels across rows. The key is the first alphabetic token,
    # which is usually but not always a verb — hence "first-token", not "verb".
    majority = max(Counter(gold for _, gold in labeled).values()) / len(labeled)
    by_first_token: dict[str, Counter] = {}
    for text, gold in labeled:
        token = _first_token(str(text))
        if token:
            by_first_token.setdefault(token, Counter())[gold] += 1
    oracle = (
        sum(max(c.values()) for c in by_first_token.values()) / len(labeled)
        if by_first_token else float("nan")
    )

    # Multi-label rows: a prediction counts as a hit when it names any level the
    # annotators assigned. Reported apart from the single-label statistics so the
    # two are never silently averaged.
    multi_rows = [(t, g) for t, g in multi if len(g) > 1]
    multi_pred = [(g, clf.classify(str(t))) for t, g in multi_rows]
    multi_scored = [(g, p) for g, p in multi_pred if p is not None]
    multi_hit = (
        sum(1 for g, p in multi_scored if p in g) / len(multi_scored)
        if multi_scored else float("nan")
    )

    held_out = _is_checkpoint_eval_split(args.csv_path, args.bloom_checkpoint)
    title = (
        "# Bloom axis validation — transformer (primary arm) vs lexicon (sensitivity arm)"
        if args.bloom_checkpoint
        else "# Bloom axis validation — lexicon (sensitivity arm), rule-based"
    )
    lines = [
        title,
        "",
        _provenance_line(args, len(multi), len(labeled), n_multi, held_out),
        "",
        "## Lexicon arm (Anderson & Krathwohl verb rules) vs gold labels",
        "",
        f"- Coverage (classifiable): {coverage:.1%} "
        f"({len(pairs)} classified, {unclassified} unclassified)",
        f"- Accuracy (on classifiable): {accuracy:.1%}",
        f"- Adjacent accuracy (|pred-gold| <= 1): {adjacent:.1%}",
        f"- Cohen's kappa (on classifiable): {kappa:.3f}",
        "",
        f"- Majority-class floor: {majority:.1%}",
        f"- First-token oracle: {oracle:.1%} — the best attainable by mapping each "
        f"leading token to one fixed level, i.e. a ceiling for token-keyed rules such "
        f"as this lexicon. It is NOT a bound on classifiers that read the whole text.",
    ]
    if oracle > majority:
        lines.append(
            f"- Lexicon accuracy within the floor-to-oracle range: "
            f"{(accuracy - majority) / (oracle - majority):.0%} "
            f"(this share is meaningful for the lexicon only, because the oracle "
            f"bounds token-keyed rules only)"
        )
    if multi_scored:
        lines.append(
            f"- Multi-label rows ({len(multi_scored)} scored): "
            f"hit rate {multi_hit:.1%} (prediction names one of the assigned levels)"
        )
    lines += [
        "",
        "## Lexicon arm stratified by gold-label consistency",
        "",
        "Public Bloom datasets label the same construction at different levels "
        "(e.g. \"Compare X and Y\" appearing as both Understand and Analyze). "
        "*Purity* = the share of rows sharing a leading token that carry that "
        "token's most common label; low purity is gold-label noise, which no "
        "classifier can fit. The mid-purity strata (60-90%) are the honest read of "
        "the instrument; the >= 100% stratum is a small, skewed slice and is read "
        "separately — see the note under the table. `rows` is the size of the "
        "stratum, `scored` the subset the lexicon does not abstain on; accuracy and "
        "kappa are computed over `scored` only.",
        "",
        "| min purity | rows | scored | accuracy | kappa |",
        "|---|---|---|---|---|",
    ]
    for threshold in (0.0, 0.6, 0.7, 0.8, 0.9, 1.0):
        subset = [
            (gold, clf.classify(str(text)))
            for text, gold in labeled
            if _first_token_purity(str(text), by_first_token) >= threshold
        ]
        scored = [(g, p) for g, p in subset if p is not None]
        if len(scored) < 50:
            continue
        sub_acc = sum(1 for g, p in scored if g == p) / len(scored)
        lines.append(
            f"| >= {threshold:.0%} | {len(subset)} | {len(scored)} | {sub_acc:.1%} | "
            f"{cohen_kappa(scored):.3f} |"
        )
    lines += [
        "",
        "Note: the >= 100% stratum does not continue the trend, and is not expected "
        "to. Most of its leading-token groups occur only once or twice, so their "
        "purity is 1.0 by construction rather than by annotator consistency, while "
        "its remaining mass sits on a handful of idiosyncratic verbs. It is therefore "
        "not a cleaner subsample of the corpus but a differently composed one, with a "
        "shifted gold-level mix, and its accuracy and kappa are not comparable with "
        "the rows above. Read the 60-90% rows.",
        "",
        "Lexicon confusion matrix (gold rows x lexicon prediction columns, "
        "classifiable rows only):",
        "",
        "| gold \\ pred | " + " | ".join(LEVEL_NAMES) + " |",
        "|---|" + "---|" * 6,
    ]
    confusion = Counter(pairs)
    for gold in range(1, 7):
        cells = " | ".join(str(confusion[(gold, pred)]) for pred in range(1, 7))
        lines.append(f"| {LEVEL_NAMES[gold - 1]} | {cells} |")
    lines.append("")
    lines.append(f"Levels (revised taxonomy): {', '.join(BLOOM_LEVELS)} = 1..6.")

    if args.bloom_checkpoint:
        lines += _transformer_arm_section(
            args, labeled, clf, pairs, kappa, oracle, held_out
        )

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
