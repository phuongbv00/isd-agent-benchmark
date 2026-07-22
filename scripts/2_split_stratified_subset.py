#!/usr/bin/env python3
"""Create a stratified scenario subset (domain × difficulty, N per cell).

Reproduces how scenarios/test_30 and test_90 were built, as a repeatable script
rather than an ad-hoc step. Use it to carve a small TUNING/dev set out of the
TRAIN split so hyper-parameter/prompt tuning never touches the test set.

Default: 1 scenario per (domain × difficulty) cell = 10 domains × 3 levels = 30.

See also (different jobs, both also called "stratified"):
  - split_train_test.py: one-time 95/5 split of the raw dataset, stratified on
    domain|difficulty|learning_environment.
  - sampling_strategy.py (StratifiedScenarioSampler): runtime n-sample selection
    inside run_benchmark.py, stratified on 4 separate imbalance axes and may
    duplicate items (oversample strategy).
This script instead copies a fixed N-per-cell subset to disk, no duplicates.

Examples:
  # 30-scenario tuning set from train (proper ML hygiene: tune on train, report on test)
  python scripts/2_split_stratified_subset.py --source scenarios/train --target scenarios/train_30
  # 90-scenario test set, 3 per cell
  python scripts/2_split_stratified_subset.py --source scenarios/test --target scenarios/test_90 --per-cell 3
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# The stratification cells, matching scenarios/test_30_metadata.json exactly.
DOMAINS = [
    "AI", "Business/HR/Admin Support", "Education (Teaching & Learning)", "Language",
    "Mathematics", "Medical/Nursing", "Science", "Service/Customer Support",
    "Social Studies", "Software Development/IT",
]
DIFFICULTY_ORDER = [
    "Easy - Simple structure with minimal constraints",
    "Moderate - Standard complexity with some constraints",
    "Hard - Complex requirements with multiple constraints",
]


def _index_by_cell(source: Path) -> dict[tuple[str, str], list[str]]:
    """(domain, difficulty) -> sorted list of file names present in source."""
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in sorted(source.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cells[(data.get("domain"), data.get("difficulty"))].append(path.name)
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("scenarios/train"), help="Source scenario dir.")
    parser.add_argument("--target", type=Path, default=Path("scenarios/train_30"), help="Output subset dir.")
    parser.add_argument("--per-cell", type=int, default=1, help="Scenarios per (domain × difficulty) cell.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (reproducibility).")
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        print(f"Source not found: {args.source}")
        return 1
    print(f"Indexing {args.source} by (domain × difficulty) ...")
    cells = _index_by_cell(args.source)

    rng = random.Random(args.seed)
    selected: list[str] = []
    matrix: dict[str, dict[str, int]] = {}
    missing: list[str] = []
    for domain in DOMAINS:
        matrix[domain] = {}
        for difficulty in DIFFICULTY_ORDER:
            pool = cells.get((domain, difficulty), [])
            take = min(args.per_cell, len(pool))
            picked = rng.sample(pool, take) if take else []
            matrix[domain][difficulty] = len(picked)
            selected.extend(picked)
            if len(picked) < args.per_cell:
                missing.append(f"{domain} / {difficulty} (had {len(pool)})")

    args.target.mkdir(parents=True, exist_ok=True)
    # Start clean so re-running is deterministic, not additive.
    for old in args.target.glob("*.json"):
        old.unlink()
    for name in selected:
        shutil.copy2(args.source / name, args.target / name)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(args.source),
        "source_note": f"Stratified subset ({args.per_cell} per domain×difficulty cell) for tuning.",
        "target_dir": str(args.target),
        "source_count": sum(len(v) for v in cells.values()),
        "selected_count": len(selected),
        "seed": args.seed,
        "sampling_unit": "domain × difficulty",
        "samples_per_cell": args.per_cell,
        "domains": DOMAINS,
        "difficulty_order": DIFFICULTY_ORDER,
        "matrix": matrix,
        "selected_files": sorted(selected),
    }
    meta_path = args.target.parent / f"{args.target.name}_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(selected)} scenarios to {args.target}")
    print(f"Metadata: {meta_path}")
    if missing:
        print(f"WARNING: {len(missing)} cell(s) under-filled:")
        for m in missing:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
