# Evaluator scripts

## `train_bloom_classifier.py`

Fine-tunes a multilingual 6-way Bloom-level classifier (XLM-R or mDeBERTa)
that can replace the default rule-based verb lexicon in the alignment metric
(`isd_evaluator.metrics.alignment`).

### Data

Provide a CSV or JSONL file with columns `text` (a learning objective or
assessment item, Korean or English) and `label` (1..6, 0..5, or a level name
such as `Apply` / `적용`). Public sources you can convert to this format:

- EDM 2022-style learning-objective datasets labeled with Bloom levels
  (e.g. "Automated Classification of Learning Objectives" corpora).
- Course-catalog learning-outcome datasets with Bloom annotations on
  Hugging Face / Kaggle.

Korean coverage matters for this benchmark: if your corpus is English-only,
the multilingual base model still transfers reasonably, but adding
machine-translated or manually labeled Korean objectives improves accuracy.

### Setup and run

```bash
pip install -e evaluator[bloom-train]

python evaluator/scripts/train_bloom_classifier.py \
    --train data/bloom_objectives.csv \
    --base-model xlm-roberta-base \
    --output-dir models/bloom-xlmr \
    --epochs 4
```

### Use the checkpoint

```python
from isd_evaluator.metrics.alignment import (
    AlignmentEvaluator, TransformerBloomClassifier,
)

evaluator = AlignmentEvaluator(
    bloom_classifier=TransformerBloomClassifier("models/bloom-xlmr"),
)
```

`TransformerBloomClassifier` only loads an existing local checkpoint — it
never downloads. Everything else in the alignment metric (default lexicon
classifier, TF-IDF fallback encoder) runs fully offline with no extra
dependencies.
