#!/usr/bin/env python3
"""Validate the rule-based Bloom verb lexicon against a labeled dataset.

Validity evidence for the weakest link of the alignment protocol: runs
:class:`isd_evaluator.metrics.alignment.LexiconBloomClassifier` over a public
question/learning-outcome dataset with expert Bloom labels and reports
coverage, accuracy, adjacent accuracy, Cohen's kappa and the confusion
matrix. The corpus scored by the protocol is ~91% English-dominant, so an
English benchmark is representative; no network access is needed here — you
download the CSV yourself.

Input: a CSV with one text column and one label column (names configurable).
Labels may be level names of either taxonomy generation
("remember"/"knowledge", "understand"/"comprehension", ...) or integers 1-6.

Public datasets that fit (download manually, any of):
  * Yahya et al. exam-question dataset (~600 questions, 6 levels);
  * the "Bloom's Taxonomy questions" datasets on Kaggle/GitHub
    (search: bloom taxonomy question classification dataset CSV).

Example:
  python scripts/alignmentgraph-isd-bench/13_validate_bloom.py blooms_questions.csv \
      --text-col Questions --label-col Category --out results/generated/bloom_validation.md
"""

from __future__ import annotations

import argparse
import csv
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


def parse_label(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if not text:
        return None
    if text.isdigit() and 1 <= int(text) <= 6:
        return int(text)
    if text in _LEGACY_LEVELS:
        return _LEGACY_LEVELS[text]
    return normalize_declared_level(text)


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the Bloom verb lexicon against a labeled CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Labeled dataset (CSV).")
    parser.add_argument("--text-col", default="question",
                        help="Column holding the question/outcome text.")
    parser.add_argument("--label-col", default="label",
                        help="Column holding the Bloom label (name or 1-6).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Also write the markdown report to this file.")
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise SystemExit(f"Not found: {args.csv_path} (download a labeled "
                         "Bloom dataset first — see the module docstring).")
    with args.csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or args.text_col not in reader.fieldnames \
                or args.label_col not in reader.fieldnames:
            raise SystemExit(
                f"Columns {args.text_col!r}/{args.label_col!r} not in CSV "
                f"(has: {reader.fieldnames}). Use --text-col/--label-col."
            )
        rows = [(r[args.text_col], parse_label(r[args.label_col])) for r in reader]

    labeled = [(t, gold) for t, gold in rows if gold is not None and str(t).strip()]
    if not labeled:
        raise SystemExit("No usable labeled rows found.")

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

    lines = [
        "# Bloom lexicon validation",
        "",
        f"- Dataset: `{args.csv_path.name}` ({len(labeled)} labeled rows)",
        f"- Coverage (classifiable): {coverage:.1%} "
        f"({len(pairs)} classified, {unclassified} unclassified)",
        f"- Accuracy (on classifiable): {accuracy:.1%}",
        f"- Adjacent accuracy (|pred-gold| <= 1): {adjacent:.1%}",
        f"- Cohen's kappa (on classifiable): {kappa:.3f}",
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

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
