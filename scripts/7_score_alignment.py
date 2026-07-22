#!/usr/bin/env python3
"""Post-hoc offline constructive-alignment scoring over a benchmark run dir.

Walks a results run directory (flat single-run or nested dataset layout),
scores every ``<agent_id>_output.json`` with the deterministic, LLM-free
:class:`isd_evaluator.metrics.alignment.AlignmentEvaluator`, writes an
``alignment_scores.json`` next to the outputs of each scenario, and prints a
per-agent summary table (optionally also written as CSV).

No LLM calls, no network. The default encoder is the pure-Python TF-IDF
character-n-gram fallback; pass ``--encoder st`` to use a local
sentence-transformers model for real scoring runs.

Examples:
  python scripts/7_score_alignment.py results/test_90_benchmark_..._104536
  python scripts/7_score_alignment.py results/test_90_... --limit 5
  python scripts/7_score_alignment.py results/test_90_... --encoder st \
      --st-model sentence-transformers/LaBSE
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluator" / "src"))

from isd_evaluator.metrics.alignment import (  # noqa: E402
    AlignmentEvaluator,
    SentenceTransformerEncoder,
    TfidfCharNgramEncoder,
)

COMPONENTS = [
    "composite",
    "porter_mean",
    "webb_range",
    "webb_bloom_consistency",
    "embedding_coverage",
    "embedding_precision",
]


def find_scenario_dirs(run_dir: Path) -> list[Path]:
    """Directories that contain agent outputs (flat run dir or nested)."""
    if list(run_dir.glob("*_output.json")):
        return [run_dir]
    dirs = sorted(
        {path.parent for path in run_dir.glob("*/*/*_output.json")}
        | {path.parent for path in run_dir.glob("*/*_output.json")}
    )
    return dirs


def load_scenario(scenario_dir: Path, run_dir: Path, scenario_root: Path) -> dict | None:
    """Scenario dict for a scenario dir.

    Prefers the copy embedded in ``comparison_report.json`` (always matches
    the run), then falls back to ``<scenario_root>/<variant>/<dir_name>.json``.
    """
    report_path = scenario_dir / "comparison_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            scenario = report.get("scenario")
            if isinstance(scenario, dict) and scenario:
                return scenario
        except (json.JSONDecodeError, OSError):
            pass
    variant = scenario_dir.parent.name if scenario_dir != run_dir else None
    candidates = [scenario_root / f"{scenario_dir.name}.json"]
    if variant:
        candidates.insert(0, scenario_root / variant / f"{scenario_dir.name}.json")
    candidates += sorted(scenario_root.glob(f"*/{scenario_dir.name}.json"))
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
    return None


def score_scenario_dir(
    scenario_dir: Path,
    evaluator: AlignmentEvaluator,
    scenario: dict | None,
    agents_filter: set[str] | None,
) -> dict[str, dict]:
    scores: dict[str, dict] = {}
    for output_path in sorted(scenario_dir.glob("*_output.json")):
        agent_id = output_path.stem[: -len("_output")]
        if agents_filter and agent_id not in agents_filter:
            continue
        try:
            addie_output = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"    ! {agent_id}: unreadable output ({exc})", file=sys.stderr)
            continue
        score = evaluator.evaluate(addie_output, scenario)
        scores[agent_id] = score.to_dict()
    return scores


def summarize(per_agent: dict[str, list[dict]]) -> list[tuple[str, int, dict[str, float | None]]]:
    rows = []
    for agent_id in sorted(per_agent):
        entries = per_agent[agent_id]
        means: dict[str, float | None] = {}
        for component in COMPONENTS:
            values = [e[component] for e in entries if e.get(component) is not None]
            means[component] = statistics.fmean(values) if values else None
        sanity_values = [
            e["details"]["sanity"]["lexicon_vs_declared_agreement"]
            for e in entries
            if e.get("details", {}).get("sanity", {}).get("lexicon_vs_declared_agreement")
            is not None
        ]
        means["sanity_agreement"] = (
            statistics.fmean(sanity_values) if sanity_values else None
        )
        rows.append((agent_id, len(entries), means))
    rows.sort(key=lambda r: -(r[2]["composite"] if r[2]["composite"] is not None else -1))
    return rows


def print_table(rows: list[tuple[str, int, dict[str, float | None]]]) -> None:
    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "  -  "

    header = (
        f"{'agent':30s} {'n':>3s} {'Align':>6s} {'Porter':>6s} {'Range':>6s} "
        f"{'BloomC':>6s} {'Cover':>6s} {'Prec':>6s} {'Sanity':>6s}"
    )
    print(header)
    print("-" * len(header))
    for agent_id, n, means in rows:
        print(
            f"{agent_id:30s} {n:3d} {fmt(means['composite']):>6s}"
            f" {fmt(means['porter_mean']):>6s} {fmt(means['webb_range']):>6s}"
            f" {fmt(means['webb_bloom_consistency']):>6s}"
            f" {fmt(means['embedding_coverage']):>6s}"
            f" {fmt(means['embedding_precision']):>6s}"
            f" {fmt(means['sanity_agreement']):>6s}"
        )


def write_csv(path: Path, rows: list[tuple[str, int, dict[str, float | None]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "n"] + COMPONENTS + ["sanity_agreement"])
        for agent_id, n, means in rows:
            writer.writerow(
                [agent_id, n]
                + [
                    f"{means[c]:.4f}" if means[c] is not None else ""
                    for c in COMPONENTS + ["sanity_agreement"]
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score constructive alignment (deterministic, LLM-free) over a run dir."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--scenario-root", type=Path, default=REPO_ROOT / "scenarios",
        help="Scenario directory root (default: <repo>/scenarios).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Score at most N scenario dirs.")
    parser.add_argument(
        "--agents", default=None,
        help="Comma-separated agent ids to score (default: every *_output.json).",
    )
    parser.add_argument(
        "--encoder", choices=["tfidf", "st"], default="tfidf",
        help="Text encoder: 'tfidf' (pure-Python fallback, default) or 'st' "
             "(sentence-transformers, requires the [alignment] extra).",
    )
    parser.add_argument(
        "--st-model", default="sentence-transformers/LaBSE",
        help="sentence-transformers model name/path for --encoder st.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.25,
        help="Similarity threshold for objective<->item correspondence (default 0.25).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-score scenario dirs that already have alignment_scores.json.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    encoder = (
        SentenceTransformerEncoder(args.st_model)
        if args.encoder == "st"
        else TfidfCharNgramEncoder()
    )
    evaluator = AlignmentEvaluator(encoder=encoder, match_threshold=args.threshold)
    agents_filter = (
        {a.strip() for a in args.agents.split(",") if a.strip()} if args.agents else None
    )

    scenario_dirs = find_scenario_dirs(run_dir)
    if args.limit is not None:
        scenario_dirs = scenario_dirs[: args.limit]
    if not scenario_dirs:
        raise SystemExit(f"No *_output.json found under {run_dir}")

    per_agent: dict[str, list[dict]] = defaultdict(list)
    for scenario_dir in scenario_dirs:
        scores_path = scenario_dir / "alignment_scores.json"
        if scores_path.exists() and not args.overwrite:
            payload = json.loads(scores_path.read_text(encoding="utf-8"))
            print(f"  = {scenario_dir.name} (cached, {len(payload.get('agents', {}))} agents)")
        else:
            scenario = load_scenario(scenario_dir, run_dir, args.scenario_root.resolve())
            if scenario is None:
                print(f"  ! {scenario_dir.name}: scenario JSON not found, "
                      "topics fall back to a single bucket", file=sys.stderr)
            scores = score_scenario_dir(scenario_dir, evaluator, scenario, agents_filter)
            payload = {
                "scenario_dir": scenario_dir.name,
                "scenario_id": (scenario or {}).get("scenario_id"),
                "encoder": args.st_model if args.encoder == "st" else "tfidf-char-ngram",
                "match_threshold": args.threshold,
                "agents": scores,
            }
            scores_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  + {scenario_dir.name} ({len(scores)} agents)")
        for agent_id, score in payload.get("agents", {}).items():
            per_agent[agent_id].append(score)

    rows = summarize(per_agent)
    print(f"\nAlignment summary over {len(scenario_dirs)} scenario dir(s) "
          f"(encoder: {payload['encoder']}):")
    print_table(rows)
    csv_path = run_dir / "alignment_summary.csv"
    write_csv(csv_path, rows)
    print(f"\nWrote {csv_path}")
    print("Note: 'Sanity' is the report-only lexicon-vs-declared Bloom agreement; "
          "it is not part of the Align composite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
