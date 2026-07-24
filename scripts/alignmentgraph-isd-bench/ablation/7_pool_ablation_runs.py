#!/usr/bin/env python3
"""Pool the ablation-study runs into per-arm stats (defense: component contribution).

Companion of ../7_pool_ladder_runs.py and a thin orchestrator over its machinery
(pool_model / compare_model / pool_alignment / pool_tokens are imported and
reused verbatim). The comparison axis differs from the ladder: instead of the
proposed agent vs the 6 baseline agents ACROSS runs, this compares the ablation
arms of alignmentgraph-isd WITHIN one run per model size — all arms share each
scenario's judge session, so the deltas carry no judge drift.

Arms (agent ids registered in run_benchmark.py):
  A0 alignmentgraph-isd                (full pipeline)
  A1 alignmentgraph-isd-no-verifier    (QC verify+repair off)
  A2 alignmentgraph-isd-no-graph-ctx   (graph-context prompt injection off)
  A3 alignmentgraph-isd-skeleton       (both off)

Direction convention: reusing compare_model with proposed=A0 and baselines=arms
means every mean_diff / CI is  A0 − arm  = the CONTRIBUTION of the removed
component(s) (positive = removing it hurts).

Besides the arm comparisons this script adds two ablation-specific layers:
  * flag sanity check — each arm's *_trajectory.json metadata.run_config must
    match what its agent id claims (hard error on mismatch; the field only
    exists for runs made after the ablation flags landed);
  * A0 consistency check — A0's in-session mean vs the pooled ladder r1–r3 mean
    of the same size (from --ladder-pooled), reported against the ladder's
    run-to-run SD as an empirical judge/session-drift gauge.

Usage:
  python scripts/alignmentgraph-isd-bench/ablation/7_pool_ablation_runs.py \
      --auto-glob 'results/test_90_benchmark_*_abl1_*' \
      --ladder-pooled results/pooled_ladder.json
  # -> results/pooled_ablation.json (input of 8_gen_ablation_tables/figures)

Run scripts/alignmentgraph-isd-bench/7_score_alignment.py on each run dir first
if you want the RQ2 alignment layer pooled too (graceful skip otherwise).
"""
from __future__ import annotations

import argparse
import glob as globmod
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_ladder_pool_module():
    """Import ../7_pool_ladder_runs.py (digit-leading name -> spec loader)."""
    path = _HERE.parent / "7_pool_ladder_runs.py"
    spec = importlib.util.spec_from_file_location("pool_ladder_runs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LP = _load_ladder_pool_module()

A0_DEFAULT = "alignmentgraph-isd"
ARMS_DEFAULT = [
    "alignmentgraph-isd-no-verifier",
    "alignmentgraph-isd-no-graph-ctx",
    "alignmentgraph-isd-skeleton",
]
# What metadata.run_config must say per arm (kwargs pinned by the
# run_benchmark.py registry; see agents/alignmentgraph-isd adapter).
EXPECTED_FLAGS = {
    "alignmentgraph-isd": {"enable_verifier": True, "enable_graph_context": True},
    "alignmentgraph-isd-no-verifier": {"enable_verifier": False, "enable_graph_context": True},
    "alignmentgraph-isd-no-graph-ctx": {"enable_verifier": True, "enable_graph_context": False},
    "alignmentgraph-isd-skeleton": {"enable_verifier": False, "enable_graph_context": False},
}


def group_by_glob(pattern: str, run_tag: str) -> dict[str, list[Path]]:
    """Group ablation run dirs by model label (the tag is fixed, not r\\d+)."""
    run_dir_re = re.compile(
        r"^(?P<dataset>.+)_benchmark_(?P<model>.+?)_" + re.escape(run_tag)
        + r"_(?P<ts>\d{8}_\d{6})$"
    )
    groups: dict[str, list[Path]] = {}
    for path_str in sorted(globmod.glob(pattern)):
        path = Path(path_str)
        if not path.is_dir():
            continue
        m = run_dir_re.match(path.name)
        if not m:
            print(f"  ! dir does not match ablation naming, skipped: {path.name}",
                  file=sys.stderr)
            continue
        groups.setdefault(m.group("model"), []).append(path)
    return dict(sorted(groups.items(),
                       key=lambda kv: LP.parse_size_key(kv[0])))


def check_arm_flags(run_dirs_by_model: dict[str, list[Path]],
                    agents: list[str]) -> dict:
    """Verify each arm ran the config its agent id claims. Hard error on mismatch."""
    report: dict[str, dict[str, str]] = {}
    mismatches: list[str] = []
    for label, run_dirs in run_dirs_by_model.items():
        report[label] = {}
        for run_dir in run_dirs:
            for agent in agents:
                expected = EXPECTED_FLAGS.get(agent)
                if expected is None:
                    report[label][agent] = "no expectation registered — skipped"
                    continue
                run_config = None
                for _sid, scen_dir in LP.iter_scenario_dirs(run_dir):
                    traj = scen_dir / f"{agent}_trajectory.json"
                    if not traj.exists():
                        continue
                    try:
                        md = json.loads(traj.read_text(encoding="utf-8")).get("metadata") or {}
                    except (json.JSONDecodeError, OSError):
                        continue
                    run_config = md.get("run_config")
                    break  # one scenario suffices: flags are fixed per process
                if run_config is None:
                    report[label][agent] = (
                        "WARN: no metadata.run_config found (run predates the "
                        "ablation flags, or no trajectory written)"
                    )
                    continue
                got = {k: run_config.get(k) for k in expected}
                if got == expected:
                    report[label][agent] = "ok"
                else:
                    report[label][agent] = f"MISMATCH: expected {expected}, got {got}"
                    mismatches.append(f"{label}/{agent}: expected {expected}, got {got}")
    if mismatches:
        print("\nERROR: arm flag mismatch — an arm did not run the config its id claims:",
              file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)
    return report


def a0_consistency(models: list[dict], a0: str, ladder_pooled_path: Path) -> dict:
    """A0 in-session mean vs the pooled ladder mean of the same size label."""
    try:
        ladder = json.loads(ladder_pooled_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"error": f"could not read {ladder_pooled_path}: {exc}"}
    ladder_by_label = {m["label"]: m for m in ladder.get("models", [])}
    rows = []
    for model in models:
        label = model["label"]
        abl = model["agents"].get(a0, {})
        lad = ladder_by_label.get(label, {}).get("agents", {}).get(a0)
        if lad is None:
            rows.append({"label": label,
                         "note": f"size '{label}' not found in ladder pooled JSON"})
            continue
        delta = abl.get("mean_total", float("nan")) - lad.get("mean_total", float("nan"))
        rows.append({
            "label": label,
            "a0_mean_total_ablation_session": abl.get("mean_total"),
            "a0_mean_total_ladder_pooled": lad.get("mean_total"),
            "delta": delta,
            "ladder_run_to_run_sd": lad.get("run_to_run_sd"),
            "within_2x_ladder_run_sd": (
                abs(delta) <= 2 * lad["run_to_run_sd"]
                if isinstance(lad.get("run_to_run_sd"), (int, float)) else None
            ),
        })
    return {
        "definition": (
            "A0 (full pipeline) re-ran inside the ablation session; its mean vs "
            "the pooled ladder r-runs of the same size gauges judge/session "
            "drift. |delta| within ~2x the ladder run-to-run SD supports "
            "attributing arm deltas to the ablated mechanisms."
        ),
        "per_model": rows,
    }


def print_report(pooled: dict, a0: str, arms: list[str]) -> None:
    print("\n" + "=" * 78)
    print("ABLATION POOLING — contribution = A0 − arm (positive: component helps)")
    print("=" * 78)
    for model in pooled["models"]:
        print(f"\n[{model['label']}]  runs={model['n_runs']}  "
              f"scenarios={model['n_scenarios_union']}")
        for agent in [a0] + arms:
            s = model["agents"].get(agent, {})
            print(f"  {agent:<38} mean_total={s.get('mean_total', float('nan')):7.2f}  "
                  f"nCC={s.get('n_scenarios_complete', 0):3d}  "
                  f"missing={s.get('n_scenarios_any_missing', 0)}")
        rows = model["comparisons"]["policies"]["complete_case"]
        for r in rows:
            if r.get("n"):
                lo, hi = r["ci95"]
                print(f"    A0 − {r['baseline']:<34} Δ={r['mean_diff']:+6.2f} "
                      f"CI95=[{lo:+.2f},{hi:+.2f}] n={r['n']} "
                      f"p_holm={r.get('p_holm', float('nan')):.2g}")
            else:
                print(f"    A0 − {r['baseline']:<34} (no shared scenarios)")
    cons = pooled.get("a0_consistency", {})
    for row in cons.get("per_model", []):
        if "delta" in row:
            print(f"\n  A0 consistency [{row['label']}]: session−ladder "
                  f"Δ={row['delta']:+.2f} (ladder run SD "
                  f"{row.get('ladder_run_to_run_sd')})")
    if pooled.get("alignment") is None:
        print("\n  (RQ2 alignment layer absent — run "
              "scripts/alignmentgraph-isd-bench/7_score_alignment.py on each "
              "run dir, then re-pool.)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool ablation runs (arms within one run per size) into stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", action="append", default=[], metavar="LABEL=DIR",
        help="Explicit mapping size-label -> run dir. Repeatable, size order kept.",
    )
    parser.add_argument(
        "--auto-glob", default=None, metavar="PATTERN",
        help="Glob of ablation run dirs (naming "
             "<dataset>_benchmark_<model>_<RUN_TAG>_<ts>); groups by <model>.",
    )
    parser.add_argument("--run-tag", default="abl1",
                        help="Tag used by ablation/5_run_ablation.sh.")
    parser.add_argument("--a0-agent", default=A0_DEFAULT)
    parser.add_argument("--arms", default=",".join(ARMS_DEFAULT),
                        help="Comma-separated arm agent ids (Holm family size).")
    parser.add_argument("--ladder-pooled", default="results/pooled_ladder.json",
                        help="pooled_ladder.json for the A0 consistency check "
                             "(skipped with a note if absent).")
    parser.add_argument("--out", default="results/pooled_ablation.json")
    args = parser.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    agents = [args.a0_agent] + arms

    run_dirs_by_model: dict[str, list[Path]] = {}
    for spec in args.model:
        if "=" not in spec:
            parser.error(f"--model expects LABEL=DIR, got: {spec}")
        label, d = spec.split("=", 1)
        run_dirs_by_model[label] = [Path(d)]
    if args.auto_glob:
        for label, dirs in group_by_glob(args.auto_glob, args.run_tag).items():
            run_dirs_by_model.setdefault(label, dirs)
    if not run_dirs_by_model:
        parser.error("no input: give --model and/or --auto-glob")

    for label, dirs in run_dirs_by_model.items():
        missing = [d for d in dirs if not d.is_dir()]
        if missing:
            parser.error(f"size '{label}': run dirs not found: {missing}")
        print(f"size '{label}': {len(dirs)} run(s)")
        for d in dirs:
            print(f"  - {d}")

    flag_check = check_arm_flags(run_dirs_by_model, agents)

    models = []
    for label, dirs in run_dirs_by_model.items():
        model = LP.pool_model(label, dirs, agents)
        model["comparisons"] = LP.compare_model(model, args.a0_agent, arms)
        models.append(model)

    ladder_pooled_path = Path(args.ladder_pooled)
    if ladder_pooled_path.exists():
        consistency = a0_consistency(models, args.a0_agent, ladder_pooled_path)
    else:
        consistency = {"note": f"{ladder_pooled_path} not found — check skipped"}

    pooled = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "a0": args.a0_agent,
            "arms": arms,
            "expected_flags": {a: EXPECTED_FLAGS.get(a) for a in agents},
            "score_field": "total_score (per-scenario composite from comparison_report.json)",
            "direction": "mean_diff = A0 − arm = contribution of the removed component(s)",
            "design": "all arms judged within one run per size (same judge session)",
            "primary_failure_policy": "complete_case",
            "holm_family": f"{len(arms)} arm comparisons within each model size",
            "n_boot": LP.N_BOOT,
            "bootstrap_seed": LP.BOOT_SEED,
            "size_order": [m["label"] for m in models],
        },
        "flag_check": flag_check,
        "models": models,
        "a0_consistency": consistency,
        "alignment": LP.pool_alignment(models, run_dirs_by_model, agents,
                                       args.a0_agent, arms),
        "token_usage": LP.pool_tokens(run_dirs_by_model, agents),
    }

    print_report(pooled, args.a0_agent, arms)

    for model in pooled["models"]:
        model.pop("_totals", None)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pooled, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nPooled ablation JSON written: {out_path}")


if __name__ == "__main__":
    main()
