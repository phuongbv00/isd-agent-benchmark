#!/usr/bin/env python3
"""Pool the ablation-study runs into per-arm stats (defense: component contribution).

Companion of ../07_pool_ladder_runs.py and a thin orchestrator over its machinery
(pool_model / compare_model / pool_alignment / pool_tokens are imported and
reused verbatim). The comparison axis differs from the ladder: instead of the
proposed agent vs the 6 baseline agents, this compares the ablation arms of
alignmentgraph-isd — but out of THE SAME run dirs. Since 2026-07-26 the arms
run inside the ladder (02_run_ladder.sh launches all 10 agents), so every arm
shares each scenario's judge session (no judge drift in the deltas) AND gets
the ladder's 3-run averaging. A0 is not a re-run of the full pipeline any more:
it is literally the ladder's alignmentgraph-isd, which is why no table needs a
caveat reconciling two different A0 numbers.

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
    match what its agent id claims (hard error on mismatch);
  * A0 invariant check — A0 here vs A0 in the pooled ladder. These are the same
    runs now, so the delta must be ~0; a non-zero value means the two poolings
    disagree about identical data and should be debugged, not written up.

Usage:
  python scripts/alignmentgraph-isd-bench/ablation/08_pool_ablation_runs.py \
      --auto-glob 'results/test_90_benchmark_*_r[123]_*' \
      --ladder-pooled results/pooled_ladder.json
  # -> results/pooled_ablation.json (input of 13_gen_ablation_tables / 14_gen_ablation_figures)

Run scripts/alignmentgraph-isd-bench/06_score_alignment.py on each run dir first
if you want the RQ2 alignment layer pooled too (graceful skip otherwise).
"""
from __future__ import annotations

import argparse
import glob as globmod
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_ladder_pool_module():
    """Import ../07_pool_ladder_runs.py (digit-leading name -> spec loader)."""
    path = _HERE.parent / "07_pool_ladder_runs.py"
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


def group_by_glob(pattern: str) -> dict[str, list[Path]]:
    """Group run dirs by model label, pooling every run of that size.

    Uses the ladder's own naming regex rather than pinning one tag: the arms
    ride along in r1/r2/r3, so pinning a tag would silently drop two thirds of
    the data and quietly undo the 3-run averaging.
    """
    run_dir_re = LP.RUN_DIR_RE
    groups: dict[str, list[Path]] = {}
    for path_str in sorted(globmod.glob(pattern)):
        path = Path(path_str)
        if not path.is_dir():
            continue
        m = run_dir_re.match(path.name)
        if not m:
            print(f"  ! dir does not match naming convention, skipped: {path.name}",
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
                        "WARN: no metadata.run_config found (no trajectory "
                        "written for this arm — did it crash?)"
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
    """A0 here vs A0 in the pooled ladder — now an invariant, not a drift gauge.

    The arms used to run in their own session, so this compared two different
    executions of the full pipeline and delta measured judge/session drift.
    Since the arms moved into the ladder runs (2026-07-26) both numbers come
    from the SAME runs, so delta must be ~0. A non-zero delta is therefore a
    bug signal — the two poolings disagreeing about identical data — not
    something to explain away in the write-up.
    """
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
            "A0 (full pipeline) is pooled here from the same ladder runs as the "
            "ladder's own A0, so delta must be ~0 — this is an invariant check "
            "that both poolings read the same data, NOT a drift measurement. "
            "A materially non-zero delta means the two poolings disagree and "
            "should be debugged, not reported."
        ),
        "per_model": rows,
    }


# ── 2x2 factorial layer ──────────────────────────────────────────────────────

#: The four arms ARE a complete 2x2 (verifier x graph context), so the design
#: supports simple effects and an interaction — not just each arm against A0.
#: Reading only "arm vs A0" is a main-effects-only view and it hides the case
#: where one mechanism does nothing *because the other one already did the job*.
FACTORIAL_CELLS = {
    ("on", "on"): "alignmentgraph-isd",
    ("off", "on"): "alignmentgraph-isd-no-verifier",
    ("on", "off"): "alignmentgraph-isd-no-graph-ctx",
    ("off", "off"): "alignmentgraph-isd-skeleton",
}

#: Judge signals (read from comparison_report.json rankings).
FACTORIAL_JUDGE_SIGNALS = ("addie_median", "trajectory_score", "total_score")

#: Every signal the factorial layer is computed on: the three judge signals plus
#: the whole RQ2 panel. Deliberately exhaustive — the panel's defence against
#: selective reporting is that every signal is reported for every comparison,
#: and a factorial computed on a hand-picked subset would break exactly that.
#: The paper's main text leads on one signal; the appendix carries all of them.
FACTORIAL_SIGNALS = FACTORIAL_JUDGE_SIGNALS + tuple(
    sig for sig, _family in LP.PANEL_SIGNALS)


def _paired(diffs: list[float]) -> dict:
    if not diffs:
        return {"n": 0}
    w, p, n_eff = LP.wilcoxon_signed_rank(diffs)
    lo, hi = LP.bootstrap_ci(diffs)
    return {"n": len(diffs), "n_effective_nonzero": n_eff,
            "mean_diff": LP._mean(diffs), "ci95": [lo, hi],
            "wilcoxon_w_plus": w, "p_raw": p,
            "rank_biserial_r": LP.rank_biserial(diffs)}


def per_scenario_maps(run_dirs: list[Path], agents: list[str]) -> dict:
    """signal -> agent -> {scenario: mean over runs}.

    One pass over the run dirs picks up both the judge fields (from
    comparison_report.json) and the alignment panel (from alignment_scores.json),
    so the factorial layer never re-derives a number a different way than the
    rest of the pool does.
    """
    acc: dict[str, dict[str, dict[str, list[float]]]] = {
        sig: {a: {} for a in agents} for sig in FACTORIAL_SIGNALS
    }
    panel_signals = [sig for sig, _f in LP.PANEL_SIGNALS]
    for run_dir in run_dirs:
        for sid, payload in LP.load_run(run_dir).items():
            for a in agents:
                r = (payload.get("rankings") or {}).get(a) or {}
                for sig in FACTORIAL_JUDGE_SIGNALS:
                    v = r.get(sig)
                    if isinstance(v, (int, float)):
                        acc[sig][a].setdefault(sid, []).append(float(v))
            doc = payload.get("alignment") or {}
            for a, sc in (doc.get("agents") or {}).items():
                if a not in agents:
                    continue
                for sig in panel_signals:
                    v = sc.get(sig)
                    if isinstance(v, (int, float)):
                        acc[sig][a].setdefault(sid, []).append(float(v))
    return {sig: {a: {sid: LP._mean(vals) for sid, vals in by_sid.items()}
                  for a, by_sid in by_agent.items()}
            for sig, by_agent in acc.items()}


def factorial_effects(maps: dict[str, dict[str, float]]) -> dict:
    """Simple effects + interaction for one signal at one model size."""
    cells = {k: maps.get(v, {}) for k, v in FACTORIAL_CELLS.items()}
    shared = sorted(set.intersection(*(set(c) for c in cells.values()))) if all(
        cells.values()) else []
    if not shared:
        return {"n": 0, "note": "no scenario scored for all four cells"}
    g = lambda key: [cells[key][s] for s in shared]  # noqa: E731
    on_on, off_on = g(("on", "on")), g(("off", "on"))
    on_off, off_off = g(("on", "off")), g(("off", "off"))
    verif_ctx_on = [a - b for a, b in zip(on_on, off_on)]
    verif_ctx_off = [a - b for a, b in zip(on_off, off_off)]
    ctx_verif_on = [a - b for a, b in zip(on_on, on_off)]
    ctx_verif_off = [a - b for a, b in zip(off_on, off_off)]
    interaction = [a - b for a, b in zip(verif_ctx_on, verif_ctx_off)]
    return {
        "n_scenarios": len(shared),
        "cell_means": {f"verifier_{k[0]}__graphctx_{k[1]}": LP._mean(list(cells[k][s] for s in shared))
                       for k in FACTORIAL_CELLS},
        "verifier_effect_given_graphctx_on": _paired(verif_ctx_on),
        "verifier_effect_given_graphctx_off": _paired(verif_ctx_off),
        "graphctx_effect_given_verifier_on": _paired(ctx_verif_on),
        "graphctx_effect_given_verifier_off": _paired(ctx_verif_off),
        "interaction": _paired(interaction),
    }


def verifier_activity(run_dirs_by_model: dict[str, list[Path]],
                      agents: list[str]) -> list[dict]:
    """How often the verifier fired and repaired, per arm and size.

    Mechanism evidence for the factorial result: it separates "the component
    never runs" from "it runs but has nothing left to fix".
    """
    out = []
    for label, dirs in run_dirs_by_model.items():
        per_arm: dict[str, dict[str, list[float]]] = {
            a: {"verifier_events": [], "repair_events": []} for a in agents}
        for run_dir in dirs:
            for _sid, scen_dir in LP.iter_scenario_dirs(run_dir):
                for a in agents:
                    path = scen_dir / f"{a}_trajectory.json"
                    if not path.exists():
                        continue
                    try:
                        tr = (json.loads(path.read_text(encoding="utf-8"))
                              .get("trajectory") or {})
                    except (json.JSONDecodeError, OSError):
                        continue
                    per_arm[a]["verifier_events"].append(len(tr.get("verifier_events") or []))
                    per_arm[a]["repair_events"].append(len(tr.get("repair_events") or []))
        out.append({
            "label": label,
            "arms": {
                a: {
                    "n_scenario_runs": len(v["verifier_events"]),
                    "verifier_events_mean": LP._mean(v["verifier_events"]),
                    "repair_events_mean": LP._mean(v["repair_events"]),
                    "pct_scenario_runs_with_repair": (
                        100.0 * sum(1 for x in v["repair_events"] if x > 0)
                        / len(v["repair_events"]) if v["repair_events"] else float("nan")),
                }
                for a, v in per_arm.items() if v["verifier_events"]
            },
        })
    return out


def factorial_layer(run_dirs_by_model: dict[str, list[Path]],
                    agents: list[str]) -> dict:
    missing = [a for a in FACTORIAL_CELLS.values() if a not in agents]
    if missing:
        return {"skipped": f"arms missing from this pool: {missing}"}
    signals: dict[str, dict] = {sig: {"per_model": []} for sig in FACTORIAL_SIGNALS}
    for label, dirs in run_dirs_by_model.items():
        maps = per_scenario_maps(dirs, agents)
        for sig in FACTORIAL_SIGNALS:
            signals[sig]["per_model"].append(
                {"label": label, **factorial_effects(maps[sig])})
    return {
        "definition": (
            "The four arms form a complete 2x2 (verifier x graph context). "
            "'verifier_effect_given_graphctx_off' is the verifier's effect when "
            "graph context is absent, 'interaction' is the difference between "
            "the two simple effects — a negative interaction means the two "
            "mechanisms are SUBSTITUTES (each does less when the other is on). "
            "Paired per scenario (mean of runs), Wilcoxon two-sided + bootstrap "
            f"CI ({LP.N_BOOT}, seed {LP.BOOT_SEED}). Reported uncorrected: these "
            "are pre-specified structural contrasts of the design, not a family "
            "of arm-vs-A0 comparisons."),
        "cells": {f"verifier_{k[0]}__graphctx_{k[1]}": v
                  for k, v in FACTORIAL_CELLS.items()},
        "signals": signals,
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
        rows = (model["comparisons"]["by_signal"]["addie_median"]
                ["policies"]["complete_case"])
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
            flag = "" if abs(row["delta"]) < 0.005 else "   <-- SHOULD BE ~0, INVESTIGATE"
            print(f"\n  A0 invariant [{row['label']}]: ablation−ladder "
                  f"Δ={row['delta']:+.2f}{flag}")
    if pooled.get("alignment") is None:
        print("\n  (RQ2 alignment layer absent — run "
              "scripts/alignmentgraph-isd-bench/06_score_alignment.py on each "
              "run dir, then re-pool.)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool the ablation arms out of the ladder run dirs into stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", action="append", default=[], metavar="LABEL=DIR",
        help="Explicit mapping size-label -> run dir. Repeatable, size order kept.",
    )
    parser.add_argument(
        "--auto-glob", default=None, metavar="PATTERN",
        help="Glob of run dirs (naming "
             "<dataset>_benchmark_<model>_<RUN_TAG>_<ts>); groups by <model>. "
             "The arms now live in the ladder runs, so this is normally "
             "'results/test_90_benchmark_*_r[123]_*'.",
    )

    parser.add_argument("--a0-agent", default=A0_DEFAULT)
    parser.add_argument("--arms", default=",".join(ARMS_DEFAULT),
                        help="Comma-separated arm agent ids (Holm family size).")
    parser.add_argument("--ladder-pooled", default="results/pooled_ladder.json",
                        help="pooled_ladder.json for the A0 invariant check "
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
        for label, dirs in group_by_glob(args.auto_glob).items():
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
            # The comparisons layer downstream consumers read is addie_median (the
            # RQ1 lead); total_score is carried alongside, not the lead. The old
            # single-field string advertised only total_score and was stale.
            "score_fields": {
                "lead": "addie_median (ADDIE rubric /100, per-scenario, "
                        "from comparison_report.json)",
                "also_reported": list(FACTORIAL_JUDGE_SIGNALS),
                "a0_consistency_check": "total_score",
            },
            "direction": "mean_diff = A0 − arm = contribution of the removed component(s)",
            "design": "arms run inside the ladder; pooled across its runs per size, and within each run all arms share the scenario's judge session",
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
        "factorial": factorial_layer(run_dirs_by_model, agents),
        "verifier_activity": verifier_activity(run_dirs_by_model, agents),
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
