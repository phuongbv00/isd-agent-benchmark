#!/usr/bin/env python3
"""Fine-tune the Bloom classifier that serves as the alignment protocol's
Bloom-axis **sensitivity arm**.

Trains a 6-way sequence classifier (labels = revised Bloom taxonomy levels
1..6). The intended corpus is EDM2022CLO (Li et al. 2022; 21,380 expert-
labelled learning objectives, wide multi-label layout) — its CSV is read
directly, no reshaping needed. Any CSV/JSONL with ``text`` and ``label``
columns also works.

The point of this arm is NOT to replace the rule-based lexicon, which stays
primary because every level assignment traces to Anderson & Krathwohl. It is
to answer a different question: does the agent ranking depend on the choice of
Bloom classifier at all? Two instruments derived from entirely separate sources
— one from the taxonomy, one from 21,380 human labels — agreeing on the ranking
is stronger evidence than either instrument alone at any kappa.

This follows ``EDM2022 CLO.ipynb``'s ``createBERT`` deliberately closely, so
that the arm is Li et al.'s fine-tuning configuration with ONE substitution
rather than a recipe of our own:

===================  ==========================  ==========================
                     createBERT (notebook)       here
===================  ==========================  ==========================
base model           bert-base-uncased           same
max_length           55                          same
batch size           64                          same
epochs               3                           same
warmup_steps         5                           same
weight_decay         0.05                        same
learning rate        transformers default        same (``--lr`` unset)
eval/save strategy   steps, every 10             same
best model           load_best, metric=f1        same
early stopping       patience 10                 same
metric               macro F1 (argmax)           same
split                20% test, then 20% val,     same, and the outer split is
                     random_state 666            stratified too (see below)
**head**             **num_labels=2, x6 levels** **num_labels=6, one model**
===================  ==========================  ==========================

The head is the whole point of the departure. Six binary heads answer "does
level L apply", and the panel needs "which level" — congruence compares levels,
Porter needs one topic x level cell per text, Webb one level per item. Getting
one level out of six independently trained heads (positive rates ~4% to ~27%)
means inventing a decision rule the notebook never validated; a 6-way softmax
gives it directly. ``Multi-Label.ipynb`` (one sigmoid head, 0.5 per level) has
the same problem from the other side: it can emit several levels or none.

Because the head is single-label, rows the annotators marked at several levels
cannot be used as-is and are dropped (2,607 of 21,380); the notebook kept them
because each of its binary tasks only asked about one level at a time. Its
outer split could not be stratified for the same reason — it served six targets
at once — so ours stratifies on the 6-way label.

The resulting checkpoint directory plugs into the evaluator via::

    from isd_evaluator.metrics.alignment import (
        AlignmentEvaluator, TransformerBloomClassifier,
    )
    evaluator = AlignmentEvaluator(
        bloom_classifier=TransformerBloomClassifier("path/to/checkpoint")
    )

This script is offline-friendly except for the initial download of the base
model; it is NOT run as part of the benchmark and the default lexicon
classifier works without it. Requires the ``bloom-train`` extra::

    pip install -e evaluator[bloom-train]

Usage (the recipe the protocol spec pins — run from isd-agent-benchmark/):
    python evaluator/scripts/train_bloom_6way_classifier.py \
        --train ../EDM2022CLO/data/sample_full.csv \
        --output-dir models/bloom-bert-edm2022

Then score the sensitivity arm — no pod needed, Bloom runs on CPU and the
vectors come from the embedding cache:
    python scripts/alignmentgraph-isd-bench/06_score_alignment.py <run_dirs> \
        --bloom-presets bertbloom --embed-cache-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Revised-taxonomy level names, index 0..5 -> level 1..6. These land in the
#: checkpoint's id2label, so they must match BLOOM_LEVELS in alignment.py.
LEVEL_NAMES = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
LEVEL_ALIASES = {
    "remember": 1, "understand": 2, "apply": 3, "analyze": 4, "analyse": 4,
    "evaluate": 5, "create": 6,
    "knowledge": 1, "comprehension": 2, "application": 3, "analysis": 4,
    "synthesis": 6, "evaluation": 5,
    "기억": 1, "이해": 2, "적용": 3, "분석": 4, "평가": 5, "창조": 6,
}


def torch_dataset_base():
    """``torch.utils.data.Dataset``, imported late (heavy, optional dep)."""
    import torch.utils.data

    return torch.utils.data.Dataset


def normalize_label(raw) -> int:
    """Accept 1..6, 0..5, or a level name (En/Ko); return a 0-based id."""
    if isinstance(raw, str) and not raw.strip().isdigit():
        level = LEVEL_ALIASES.get(raw.strip().lower())
        if level is None:
            raise ValueError(f"Unknown Bloom label: {raw!r}")
        return level - 1
    value = int(raw)
    if 1 <= value <= 6:
        return value - 1
    if 0 <= value <= 5:
        return value
    raise ValueError(f"Bloom label out of range: {raw!r}")


#: EDM2022CLO's wide layout: one text column + six 0/1 level columns.
EDM_TEXT_COL = "Learning_outcome"
EDM_LEVEL_COLS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


def _edm_rows(rows: list[dict], multilabel: str) -> tuple[list[dict], dict[str, int]]:
    """Rows of the EDM2022CLO wide CSV -> (text, label) pairs + a drop tally.

    ``multilabel`` decides what happens to a row with more than one level
    marked: ``drop`` (default) keeps only the unambiguous rows, since assigning
    such a row a single level would invent a label the annotators did not give.
    ``max`` keeps it at its highest marked level, for a recall-oriented
    variant.
    """
    out: list[dict] = []
    stats = {"total": 0, "no_label": 0, "multi_label": 0, "kept": 0}
    for row in rows:
        text = (row.get(EDM_TEXT_COL) or "").strip()
        if not text:
            continue
        stats["total"] += 1
        marked = [i + 1 for i, col in enumerate(EDM_LEVEL_COLS)
                  if str(row.get(col) or "").strip() not in ("", "0")]
        if not marked:
            stats["no_label"] += 1
            continue
        if len(marked) > 1:
            stats["multi_label"] += 1
            if multilabel == "drop":
                continue
            marked = [max(marked)]
        out.append({"text": text, "label": marked[0] - 1})
        stats["kept"] += 1
    return out, stats


def load_rows(path: Path, multilabel: str = "drop") -> list[dict]:
    """Load (text, label) rows from a CSV or JSONL file.

    Accepts both the generic ``text``/``label`` layout and the EDM2022CLO wide
    layout (auto-detected from the header).
    """
    rows: list[dict] = []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        import csv

        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if rows and EDM_TEXT_COL in rows[0] and EDM_LEVEL_COLS[0] in rows[0]:
        out, stats = _edm_rows(rows, multilabel)
        print(f"EDM2022CLO layout: {stats['total']} rows, kept {stats['kept']}, "
              f"dropped {stats['no_label']} unlabelled + {stats['multi_label']} "
              f"multi-label (policy={multilabel})")
        if not out:
            raise SystemExit(f"No usable rows in {path}")
        return out
    out = []
    for row in rows:
        text = (row.get("text") or row.get("objective") or "").strip()
        if not text:
            continue
        out.append({"text": text, "label": normalize_label(row.get("label"))})
    if not out:
        raise SystemExit(f"No usable (text,label) rows in {path}")
    return out


def _training_arguments(TrainingArguments, args):
    """``createBERT``'s TrainingArguments, verbatim except the output dir.

    ``evaluation_strategy`` was renamed ``eval_strategy`` in transformers 4.41;
    the notebook predates the rename, so try the new name and fall back rather
    than pinning the reader to one library version.
    """
    common = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=10,
        metric_for_best_model="f1",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        save_total_limit=3,
        seed=args.seed,
        report_to=[],
    )
    if args.lr is not None:            # the notebook leaves the HF default
        common["learning_rate"] = args.lr
    try:
        return TrainingArguments(eval_strategy="steps", save_strategy="steps",
                                 eval_steps=args.save_steps, **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="steps", save_strategy="steps",
                                 eval_steps=args.save_steps, **common)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True,
                        help="EDM2022CLO-style CSV, or any CSV/JSONL with "
                             "'text' and 'label' columns.")
    parser.add_argument("--eval", type=Path, default=None,
                        help="Optional held-out CSV/JSONL. Without it the "
                             "notebook's own 80/20 hold-out is reproduced.")
    parser.add_argument("--base-model", default="bert-base-uncased",
                        help="HF base model — the notebook's.")
    parser.add_argument("--multilabel", choices=["drop", "max"], default="drop",
                        help="EDM2022CLO rows with several levels marked: drop "
                             "them (the unambiguous subset) or keep them at the "
                             "highest marked level.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("models/bloom-bert-edm2022"))
    # Defaults below are createBERT's, not ours. Change one and the "same
    # configuration as Li et al." claim stops being true.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=55)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10,
                        help="EarlyStoppingCallback patience.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate. Default: unset, i.e. the "
                             "transformers default the notebook also relies on.")
    parser.add_argument("--seed", type=int, default=666,
                        help="Split/training seed. The notebook's random_state.")
    args = parser.parse_args()

    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        cohen_kappa_score,
        f1_score,
    )
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    class EncodeDataset(torch_dataset_base()):
        """The notebook's dataset wrapper (same shape, same tensor keys)."""

        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            import torch

            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    rows = load_rows(args.train, args.multilabel)
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    # Same two-stage split as the notebook: 20% held out first, then 20% of the
    # remainder as validation (stratified). Ours stratifies the outer split too
    # — the notebook could not, because its outer split served six different
    # binary targets at once.
    if args.eval:
        eval_rows = load_rows(args.eval, args.multilabel)
        train_texts, train_labels = texts, labels
        test_texts = [r["text"] for r in eval_rows]
        test_labels = [r["label"] for r in eval_rows]
    else:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=args.seed, stratify=labels)
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=args.seed,
        stratify=train_labels)
    print(f"split: train {len(tr_texts)} / val {len(val_texts)} / "
          f"held-out {len(test_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_cache=False)
    enc = lambda xs: tokenizer(xs, truncation=True, padding=True,  # noqa: E731
                               max_length=args.max_length)
    train_set = EncodeDataset(enc(tr_texts), tr_labels)
    val_set = EncodeDataset(enc(val_texts), val_labels)
    test_set = EncodeDataset(enc(test_texts), test_labels)

    # The one deliberate departure: a 6-way head instead of the notebook's
    # num_labels=2. The panel needs one level per text, and six binary heads do
    # not give one without an extra decision rule.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=6,
        id2label=dict(enumerate(LEVEL_NAMES)),
        label2id={name: i for i, name in enumerate(LEVEL_NAMES)},
        use_cache=False,
    )

    def compute_metrics(eval_pred):
        logits, labels_ = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels_, preds, average="macro"),
                "accuracy": accuracy_score(labels_, preds)}

    trainer = Trainer(
        model=model,
        args=_training_arguments(TrainingArguments, args),
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    predicted = np.argmax(trainer.predict(test_set).predictions, axis=-1)
    report = {
        "n_held_out": len(test_labels),
        "accuracy": accuracy_score(test_labels, predicted),
        "cohen_kappa": cohen_kappa_score(test_labels, predicted),
        "macro_f1": f1_score(test_labels, predicted, average="macro"),
    }
    print(json.dumps(report, indent=2))
    print(classification_report(test_labels, predicted, target_names=LEVEL_NAMES))

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "held_out_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    # Persist the held-out split so the two Bloom arms can be benchmarked on
    # IDENTICAL rows that this model never saw. Without it the only shared
    # corpus is the training set, where the transformer's numbers are in-sample
    # and any comparison with the lexicon is rigged.
    import csv as _csv

    split_path = args.output_dir / "eval_split.csv"
    with split_path.open("w", encoding="utf-8", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in zip(test_texts, test_labels):
            writer.writerow([text, int(label) + 1])
    print(f"Wrote held-out split ({len(test_texts)} rows) to {split_path} — "
          "benchmark both Bloom arms on it with "
          "scripts/alignmentgraph-isd-bench/00_validate_bloom.py")

    print(f"Saved checkpoint to {args.output_dir} — point "
          "TransformerBloomClassifier(model_path=...) at this directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
