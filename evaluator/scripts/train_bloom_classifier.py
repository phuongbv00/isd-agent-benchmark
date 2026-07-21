#!/usr/bin/env python3
"""Fine-tune a multilingual Bloom-level classifier for the alignment metric.

Trains a 6-way sequence classifier (labels = revised Bloom taxonomy levels
1..6) on a public learning-objective dataset, e.g. the EDM 2022 style
learning-objective corpora, or any CSV/JSONL you assemble with columns
``text`` and ``label`` (label = 1..6 or a level name in English/Korean).

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

Usage:
    python evaluator/scripts/train_bloom_classifier.py \
        --train data/bloom_objectives.csv \
        --base-model xlm-roberta-base \
        --output-dir models/bloom-xlmr \
        --epochs 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LEVEL_NAMES = ["기억", "이해", "적용", "분석", "평가", "창조"]
LEVEL_ALIASES = {
    "remember": 1, "understand": 2, "apply": 3, "analyze": 4, "analyse": 4,
    "evaluate": 5, "create": 6,
    "knowledge": 1, "comprehension": 2, "application": 3, "analysis": 4,
    "synthesis": 6, "evaluation": 5,
    "기억": 1, "이해": 2, "적용": 3, "분석": 4, "평가": 5, "창조": 6,
}


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


def load_rows(path: Path) -> list[dict]:
    """Load (text, label) rows from a CSV or JSONL file."""
    rows: list[dict] = []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        import csv

        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        text = (row.get("text") or row.get("objective") or "").strip()
        if not text:
            continue
        out.append({"text": text, "label": normalize_label(row.get("label"))})
    if not out:
        raise SystemExit(f"No usable (text,label) rows in {path}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True,
                        help="CSV/JSONL with 'text' and 'label' columns.")
    parser.add_argument("--eval", type=Path, default=None,
                        help="Optional held-out CSV/JSONL; if absent, 10%% of train is split off.")
    parser.add_argument("--base-model", default="xlm-roberta-base",
                        help="HF base model (xlm-roberta-base or microsoft/mdeberta-v3-base).")
    parser.add_argument("--output-dir", type=Path, default=Path("models/bloom-classifier"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    train_rows = load_rows(args.train)
    if args.eval:
        eval_rows = load_rows(args.eval)
        train_ds = Dataset.from_list(train_rows)
        eval_ds = Dataset.from_list(eval_rows)
    else:
        split = Dataset.from_list(train_rows).train_test_split(
            test_size=0.1, seed=args.seed, stratify_by_column=None
        )
        train_ds, eval_ds = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=6,
        id2label={i: name for i, name in enumerate(LEVEL_NAMES)},
        label2id={name: i for i, name in enumerate(LEVEL_NAMES)},
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            seed=args.seed,
            report_to=[],
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved checkpoint to {args.output_dir} — point "
          "TransformerBloomClassifier(model_path=...) at this directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
