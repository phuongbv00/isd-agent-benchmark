# Evaluator scripts

## `train_bloom_6way_classifier.py`

Fine-tunes the **sensitivity arm** of the alignment protocol's Bloom axis: one
6-way (single-label) BERT classifier over the revised Bloom levels. The primary
arm stays the rule-based verb lexicon in `isd_evaluator.metrics.alignment`;
this one exists to show whether the agent ranking depends on the choice of
Bloom classifier at all.

The configuration is `createBERT` from `EDM2022CLO/EDM2022 CLO.ipynb`, with one
substitution: `num_labels=6` in place of the notebook's six `num_labels=2`
heads, because the panel needs one level per text rather than six independent
"does this level apply" answers. See the module docstring for the full
side-by-side.

### Data

The intended input is `EDM2022CLO/data/sample_full.csv` — the wide layout
(one text column + six 0/1 level columns) is read directly, no reshaping. Rows
the annotators marked at several levels are dropped (2,607 of 21,380) because a
single-label head cannot represent them; `--multilabel max` keeps them at their
highest marked level instead.

Any CSV/JSONL with `text` and `label` columns (1..6, 0..5, or a level name)
also works.

### Setup and run

```bash
pip install -e evaluator[bloom-train]

python evaluator/scripts/train_bloom_6way_classifier.py \
    --train ../EDM2022CLO/data/sample_full.csv \
    --output-dir models/bloom-bert-edm2022
```

Every hyperparameter defaults to the notebook's value — overriding one means
the "same configuration as Li et al." claim no longer holds.

The run writes three things next to the checkpoint: the model, the held-out
20% as `eval_split.csv`, and `held_out_metrics.json` (accuracy, Cohen's kappa,
macro F1 on that split). Report the arm's numbers from there, never from the
training corpus.

### Use the checkpoint

```bash
# benchmark both Bloom arms on identical, unseen rows
python scripts/alignmentgraph-isd-bench/00_validate_bloom.py \
    models/bloom-bert-edm2022/eval_split.csv \
    --text-col text --label-col label \
    --bloom-checkpoint models/bloom-bert-edm2022

# score the sensitivity arm (no pod: Bloom is CPU, vectors come from cache)
python scripts/alignmentgraph-isd-bench/06_score_alignment.py <run_dirs> \
    --bloom-presets bertbloom --embed-cache-only
```

`TransformerBloomClassifier` only loads an existing local checkpoint — it never
downloads. Everything else in the alignment metric (lexicon classifier, TF-IDF
encoder) runs fully offline with no extra dependencies.
