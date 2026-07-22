#!/usr/bin/env python3
"""Pool model-ladder benchmark runs (RQ1) into per-scenario means + statistics.

Input: N model sizes x M independent runs (results dirs produced by
``run_benchmark.py --run-tag rX``). For every (model, agent, scenario) the
per-scenario composite ``total_score`` from ``comparison_report.json`` is
pooled as mean across runs. This per-scenario composite is the canonical
number for the paper (NOT the judge-mean "Avg" column of aggregate_table.csv,
which diverges when a component is missing).

Outputs ``pooled_ladder.json`` plus a human-readable stdout report:
  * per (model, agent): pooled mean Total (+ADDIE/Traj), scenario n,
    missing/failed per run, run-to-run SD (SD of the 3 run-level means)
  * per model size: paired Wilcoxon signed-rank (proposed vs each baseline,
    paired by scenario on mean-of-runs), Holm correction over the explicit
    family of 6 comparisons per model size, bootstrap 95% CI (10k, seed 42)
    for the mean paired delta, matched-pairs rank-biserial effect size
  * failure policies: complete-case (primary) plus two sensitivity analyses
    (impute failures as 0 / as the min observed score of that model size)
  * method x model-size interaction (reviewer Q11): bootstrap
    difference-in-differences of the paired delta at the largest vs smallest
    size, plus the delta trend across all sizes
  * optional alignment metric pooling if ``alignment_scores.json`` files
    exist in scenario dirs (graceful skip otherwise)

Usage:
  python scripts/7_pool_ladder_runs.py \
      --model qwen0.8b=results/test_90_benchmark_qwen3.5-0.8b-nitro_r1_...,results/..._r2_...,results/..._r3_... \
      --model qwen2b=... --model qwen4b=... --model qwen9b=...

  # or auto-group by model name parsed from the dir naming convention
  python scripts/7_pool_ladder_runs.py --auto-glob 'results/test_90_benchmark_*_r*'

The statistics helpers (wilcoxon_signed_rank, bootstrap_ci, holm_correct)
live here — this file is the single home of the algorithms since
5_aggregate_benchmark_results.py (their original co-owner) was removed;
its per-agent token-usage aggregation was absorbed as pool_tokens().
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_PROPOSED = "alignmentgraph-isd"
DEFAULT_BASELINES = [
    "baseline", "react-isd", "dick-carey-agent", "addie-agent", "rpisd-agent", "eduplanner",
]
N_BOOT = 10000
BOOT_SEED = 42


# ── statistics (single home since 5_aggregate_benchmark_results.py was removed) ──

def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank (zeros discarded, tie-corrected normal
    approximation). Returns (W_plus, p_value, n_effective)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0, 0
    absd = sorted((abs(x), i) for i, x in enumerate(d))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1][0] == absd[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[absd[k][1]] = avg
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    cnt = Counter(abs(x) for x in d)
    var -= sum(t**3 - t for t in cnt.values()) / 48
    if var <= 0:
        return w_plus, 1.0, n
    z = (w_plus - mu) / math.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return w_plus, p, n


def bootstrap_ci(diffs: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    if not diffs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def holm_correct(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted: list[float] = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


# ── extra helpers (this script only) ─────────────────────────────────────────

def rank_biserial(diffs: list[float]) -> float:
    """Matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 0.0
    w_plus, _, _ = wilcoxon_signed_rank(d)
    total = n * (n + 1) / 2
    return (2 * w_plus - total) / total


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else float("nan")


# ── loading ──────────────────────────────────────────────────────────────────

def iter_scenario_dirs(run_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield (scenario_id, scenario_dir) for a run dir.

    Handles both a dataset run dir (results/<run>/<variant>/<scenario>/...)
    and a bare variant dir (results/<run>/<scenario>/...).
    """
    for child in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if (child / "comparison_report.json").exists():
            yield child.name, child          # run_dir IS the variant dir
        else:
            for scen in sorted(p for p in child.iterdir() if p.is_dir()):
                if (scen / "comparison_report.json").exists():
                    yield scen.name, scen


def load_run(run_dir: Path) -> dict[str, dict]:
    """scenario_id -> {"rankings": {agent_id: ranking}, "alignment": dict|None}."""
    out: dict[str, dict] = {}
    for scenario_id, scen_dir in iter_scenario_dirs(run_dir):
        rep = scen_dir / "comparison_report.json"
        try:
            data = json.loads(rep.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skipping unreadable report {rep}: {exc}", file=sys.stderr)
            continue
        rankings = {r["agent_id"]: r for r in data.get("comparison", {}).get("rankings", [])}
        alignment = None
        align_path = scen_dir / "alignment_scores.json"
        if align_path.exists():
            try:
                alignment = json.loads(align_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                alignment = None
        out[scenario_id] = {"rankings": rankings, "alignment": alignment}
    return out


# ── alignment_scores.json (schema tolerant) ──────────────────────────────────

def extract_alignment_metrics(doc, agents: list[str]) -> dict[str, dict[str, float]]:
    """Best-effort per-agent numeric metrics from an alignment_scores.json doc.

    Accepted shapes: {agent: {...}}, {"agents": {agent: {...}}},
    {"scores": {agent: {...}}}. Nested dicts flatten to dotted keys; numeric
    leaves only. Unknown shapes return {} (graceful skip).
    """
    if not isinstance(doc, dict):
        return {}
    for key in ("agents", "scores", "per_agent"):
        if isinstance(doc.get(key), dict):
            doc = doc[key]
            break
    out: dict[str, dict[str, float]] = {}
    for agent, payload in doc.items():
        if agent not in agents or not isinstance(payload, dict):
            continue
        flat: dict[str, float] = {}

        def walk(prefix: str, node) -> None:
            if isinstance(node, bool):
                return
            if isinstance(node, (int, float)):
                flat[prefix] = float(node)
            elif isinstance(node, dict):
                for k, v in node.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), v)

        walk("", payload)
        if flat:
            out[agent] = flat
    return out


# ── auto-glob grouping ───────────────────────────────────────────────────────

RUN_DIR_RE = re.compile(
    r"^(?P<dataset>.+)_benchmark_(?P<model>.+?)(?:_(?P<tag>r\d+))?_(?P<ts>\d{8}_\d{6})$"
)


def parse_size_key(model_safe: str) -> float:
    """Sort key: parameter count parsed from names like qwen3.5-0.8b-nitro."""
    matches = re.findall(r"(\d+(?:\.\d+)?)b\b", model_safe.lower())
    return float(matches[-1]) if matches else float("inf")


def group_by_glob(pattern: str) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path_str in sorted(globmod.glob(pattern)):
        path = Path(path_str)
        if not path.is_dir():
            continue
        m = RUN_DIR_RE.match(path.name)
        if not m:
            print(f"  ! dir does not match naming convention, skipped: {path.name}", file=sys.stderr)
            continue
        groups.setdefault(m.group("model"), []).append(path)
    return dict(sorted(groups.items(), key=lambda kv: parse_size_key(kv[0])))


# ── pooling core ─────────────────────────────────────────────────────────────

def pool_model(label: str, run_dirs: list[Path], agents: list[str]) -> dict:
    """Pool one model size across its runs."""
    runs = [load_run(d) for d in run_dirs]
    all_scenarios = sorted({sid for run in runs for sid in run})

    per_run_info = []
    for d, run in zip(run_dirs, runs):
        missing_by_agent = {
            a: sorted(
                sid for sid in all_scenarios
                if sid not in run or a not in run[sid]["rankings"]
                or run[sid]["rankings"][a].get("total_score") is None
            )
            for a in agents
        }
        per_run_info.append({
            "dir": str(d),
            "n_scenarios_scored": len(run),
            "missing_by_agent": {a: len(v) for a, v in missing_by_agent.items()},
            "missing_scenarios_by_agent": {a: v for a, v in missing_by_agent.items() if v},
        })

    # scores[agent][scenario] = [per-run total_score or None]
    def collect(field: str) -> dict[str, dict[str, list[Optional[float]]]]:
        out: dict[str, dict[str, list[Optional[float]]]] = {a: {} for a in agents}
        for a in agents:
            for sid in all_scenarios:
                vals: list[Optional[float]] = []
                for run in runs:
                    r = run.get(sid, {}).get("rankings", {}).get(a)
                    v = r.get(field) if r else None
                    vals.append(float(v) if isinstance(v, (int, float)) else None)
                out[a][sid] = vals
        return out

    totals = collect("total_score")
    addies = collect("addie_median")
    trajs = collect("trajectory_score")

    agents_summary: dict[str, dict] = {}
    for a in agents:
        complete = {sid: v for sid, v in totals[a].items() if all(x is not None for x in v)}
        scen_means = [_mean([x for x in v]) for v in complete.values()]  # type: ignore[list-item]
        scen_sds = [_sd([x for x in v]) for v in complete.values() if len(v) >= 2]  # type: ignore[arg-type]
        # run-level mean (over scenarios present in that run) -> SD across runs
        run_means = []
        for ri in range(len(runs)):
            vals = [v[ri] for v in totals[a].values() if v[ri] is not None]
            run_means.append(_mean(vals) if vals else float("nan"))
        addie_means = [
            _mean([x for x in addies[a][sid] if x is not None])
            for sid in complete if any(x is not None for x in addies[a][sid])
        ]
        traj_means = [
            _mean([x for x in trajs[a][sid] if x is not None])
            for sid in complete if any(x is not None for x in trajs[a][sid])
        ]
        agents_summary[a] = {
            "n_scenarios_complete": len(complete),
            "n_scenarios_any_missing": len(all_scenarios) - len(complete),
            "mean_total": _mean(scen_means),
            "sd_total_across_scenarios": _sd(scen_means),
            "mean_within_scenario_run_sd": _mean([s for s in scen_sds if not math.isnan(s)]),
            "mean_addie": _mean(addie_means),
            "mean_traj": _mean(traj_means),
            "run_means_total": run_means,
            "run_to_run_sd": _sd([m for m in run_means if not math.isnan(m)]),
        }

    return {
        "label": label,
        "run_dirs": [str(d) for d in run_dirs],
        "n_runs": len(runs),
        "n_scenarios_union": len(all_scenarios),
        "per_run": per_run_info,
        "agents": agents_summary,
        "_totals": totals,  # stripped before JSON dump
    }


def scenario_mean_map(totals: dict[str, dict[str, list[Optional[float]]]],
                     agent: str, policy: str, floor: float) -> dict[str, float]:
    """scenario -> pooled score under a failure policy.

    complete_case: only scenarios with ALL runs present.
    zero / min_score: missing run-values imputed with 0 / the model-size min.
    """
    out: dict[str, float] = {}
    for sid, vals in totals[agent].items():
        if policy == "complete_case":
            if all(x is not None for x in vals):
                out[sid] = _mean(vals)  # type: ignore[arg-type]
        else:
            imput = 0.0 if policy == "zero" else floor
            if any(x is not None for x in vals):  # scenario attempted at least once
                out[sid] = _mean([x if x is not None else imput for x in vals])
            else:
                out[sid] = imput
    return out


def compare_model(model: dict, proposed: str, baselines: list[str]) -> dict:
    totals = model["_totals"]
    observed = [x for a in totals.values() for v in a.values() for x in v if x is not None]
    floor = min(observed) if observed else 0.0

    policies = {}
    for policy in ("complete_case", "zero", "min_score"):
        rows = []
        for b in baselines:
            pmap = scenario_mean_map(totals, proposed, policy, floor)
            bmap = scenario_mean_map(totals, b, policy, floor)
            shared = sorted(set(pmap) & set(bmap))
            diffs = [pmap[s] - bmap[s] for s in shared]
            if not diffs:
                rows.append({"baseline": b, "n": 0})
                continue
            w, p, n_eff = wilcoxon_signed_rank(diffs)
            lo, hi = bootstrap_ci(diffs)
            rows.append({
                "baseline": b,
                "n": len(diffs),
                "n_effective_nonzero": n_eff,
                "mean_diff": _mean(diffs),
                "ci95": [lo, hi],
                "wilcoxon_w_plus": w,
                "p_raw": p,
                "rank_biserial_r": rank_biserial(diffs),
            })
        with_p = [r for r in rows if "p_raw" in r]
        holm = holm_correct([r["p_raw"] for r in with_p])
        for r, ph in zip(with_p, holm):
            r["p_holm"] = ph
        policies[policy] = rows
    return {
        "holm_family": (
            f"{len(baselines)} comparisons ({proposed} vs each baseline) "
            f"within model size '{model['label']}'; Holm applied per policy."
        ),
        "primary_policy": "complete_case",
        "min_score_floor": floor,
        "policies": policies,
    }


def interaction_tests(models: list[dict], proposed: str, baselines: list[str],
                      n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Method x model-size interaction (reviewer Q11).

    DiD = mean_i[ delta_i(largest) - delta_i(smallest) ] where
    delta_i(size) = proposed_i - baseline_i on scenario i (complete-case
    mean-of-runs), paired by scenario across sizes. Bootstrap CI resamples
    scenarios. Also reports the per-size delta trend + OLS slope on size rank.
    """
    if len(models) < 2:
        return {"note": "needs >= 2 model sizes", "per_baseline": []}
    small, large = models[0], models[-1]
    results = []
    for b in baselines:
        deltas_by_size = []
        for m in models:
            pmap = scenario_mean_map(m["_totals"], proposed, "complete_case", 0.0)
            bmap = scenario_mean_map(m["_totals"], b, "complete_case", 0.0)
            deltas_by_size.append({s: pmap[s] - bmap[s] for s in set(pmap) & set(bmap)})
        d_small, d_large = deltas_by_size[0], deltas_by_size[-1]
        shared = sorted(set(d_small) & set(d_large))
        did_vals = [d_large[s] - d_small[s] for s in shared]
        lo, hi = bootstrap_ci(did_vals, n_boot=n_boot, seed=seed)
        _, p, _ = wilcoxon_signed_rank(did_vals)

        # trend: scenarios present at every size; OLS slope of delta on size rank
        common = set(deltas_by_size[0])
        for d in deltas_by_size[1:]:
            common &= set(d)
        common = sorted(common)
        trend_means = [_mean([d[s] for s in common]) for d in deltas_by_size]
        slope, slope_lo, slope_hi = float("nan"), float("nan"), float("nan")
        if common and len(models) >= 2:
            k = len(models)
            xbar = (k - 1) / 2
            denom = sum((x - xbar) ** 2 for x in range(k))

            def ols_slope(scen_subset: list[str]) -> float:
                ms = [_mean([d[s] for s in scen_subset]) for d in deltas_by_size]
                return sum((x - xbar) * m for x, m in zip(range(k), ms)) / denom

            slope = ols_slope(common)
            rng = random.Random(seed)
            boots = sorted(
                ols_slope([common[rng.randrange(len(common))] for _ in common])
                for _ in range(n_boot)
            )
            slope_lo = boots[int(0.025 * n_boot)]
            slope_hi = boots[int(0.975 * n_boot) - 1]

        results.append({
            "baseline": b,
            "smallest": small["label"],
            "largest": large["label"],
            "n_paired_scenarios": len(shared),
            "delta_smallest_mean": _mean(list(d_small.values())),
            "delta_largest_mean": _mean(list(d_large.values())),
            "did_mean": _mean(did_vals),
            "did_ci95": [lo, hi],
            "did_wilcoxon_p": p,
            "trend_delta_by_size": {
                m["label"]: t for m, t in zip(models, trend_means)
            },
            "trend_n_common_scenarios": len(common),
            "trend_ols_slope_per_size_step": slope,
            "trend_slope_ci95": [slope_lo, slope_hi],
        })
    return {
        "definition": (
            "DiD = mean over scenarios of [delta(largest size) - delta(smallest size)], "
            "delta = proposed - baseline per scenario (mean of runs, complete-case); "
            "bootstrap CI (10k, seed 42) resamples scenarios; positive DiD means the "
            "proposed method's advantage GROWS with model size, negative means it "
            "shrinks (i.e. the method helps small models more)."
        ),
        "per_baseline": results,
    }


def pool_alignment(models: list[dict], run_dirs_by_model: dict[str, list[Path]],
                   agents: list[str]) -> Optional[dict]:
    """Pool alignment_scores.json metrics if any exist (RQ2). Graceful skip."""
    out_models = {}
    found_any = False
    for model in models:
        label = model["label"]
        runs = [load_run(d) for d in run_dirs_by_model[label]]
        # metric_vals[agent][metric][scenario] = [per-run values]
        metric_vals: dict[str, dict[str, dict[str, list[float]]]] = {}
        for run in runs:
            for sid, payload in run.items():
                doc = payload.get("alignment")
                if doc is None:
                    continue
                found_any = True
                for agent, metrics in extract_alignment_metrics(doc, agents).items():
                    for metric, val in metrics.items():
                        metric_vals.setdefault(agent, {}).setdefault(metric, {}) \
                            .setdefault(sid, []).append(val)
        if metric_vals:
            out_models[label] = {
                agent: {
                    metric: {
                        "n_scenarios": len(by_scen),
                        "mean": _mean([_mean(v) for v in by_scen.values()]),
                        "sd_across_scenarios": _sd([_mean(v) for v in by_scen.values()]),
                    }
                    for metric, by_scen in sorted(metrics.items())
                }
                for agent, metrics in sorted(metric_vals.items())
            }
    if not found_any:
        return None
    return {
        "source": "alignment_scores.json per scenario dir (pooled mean across runs, then across scenarios)",
        "models": out_models,
    }



def pool_tokens(run_dirs_by_model: dict[str, list[Path]],
                agents: list[str]) -> Optional[dict]:
    """Pool per-agent token usage / runtime from *_trajectory.json metadata.

    (Absorbed from the deleted 5_aggregate_benchmark_results.py "tokens"
    section, in pooled form: mean across runs per scenario, then across
    scenarios.) Operational metadata only -- never part of Total or the
    alignment composite.
    """
    out_models = {}
    found_any = False
    for label, run_dirs in run_dirs_by_model.items():
        # vals[agent][field][scenario] = [per-run values]
        vals: dict[str, dict[str, dict[str, list[float]]]] = {}
        for run_dir in run_dirs:
            for sid, scen_dir in iter_scenario_dirs(run_dir):
                for agent in agents:
                    traj = scen_dir / f"{agent}_trajectory.json"
                    if not traj.exists():
                        continue
                    try:
                        md = json.loads(traj.read_text(encoding="utf-8")).get("metadata") or {}
                    except (json.JSONDecodeError, OSError):
                        continue
                    tu = md.get("token_usage") or {}
                    fields = {
                        "prompt_tokens": tu.get("prompt_tokens"),
                        "completion_tokens": tu.get("completion_tokens"),
                        "total_tokens": tu.get("total_tokens"),
                        "llm_calls": tu.get("llm_calls"),
                        "execution_time_seconds": md.get("execution_time_seconds"),
                    }
                    for field, v in fields.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            found_any = True
                            vals.setdefault(agent, {}).setdefault(field, {}) \
                                .setdefault(sid, []).append(float(v))
        if vals:
            out_models[label] = {
                agent: {
                    field: {
                        "n_scenarios": len(by_scen),
                        "mean": _mean([_mean(v) for v in by_scen.values()]),
                        "sd_across_scenarios": _sd([_mean(v) for v in by_scen.values()]),
                    }
                    for field, by_scen in sorted(fields_map.items())
                }
                for agent, fields_map in sorted(vals.items())
            }
    if not found_any:
        return None
    return {
        "source": "*_trajectory.json metadata (token_usage + execution_time), "
                  "pooled mean across runs then scenarios; operational metadata only",
        "models": out_models,
    }


# ── report printing ──────────────────────────────────────────────────────────

def print_report(pooled: dict, proposed: str, baselines: list[str]) -> None:
    print("\n" + "=" * 78)
    print("POOLED MODEL LADDER — per-scenario composite total_score (paper-canonical)")
    print("=" * 78)
    for model in pooled["models"]:
        print(f"\n## {model['label']}  ({model['n_runs']} runs, "
              f"{model['n_scenarios_union']} scenarios in union)")
        for info in model["per_run"]:
            miss = {a: n for a, n in info["missing_by_agent"].items() if n}
            print(f"  run {Path(info['dir']).name}: {info['n_scenarios_scored']} scenarios scored"
                  + (f", missing {miss}" if miss else ""))
        print(f"  {'agent':20s} {'nCC':>4s} {'miss':>4s} {'Total':>7s} {'±runSD':>7s} "
              f"{'ADDIE':>7s} {'Traj':>7s}")
        for a, s in model["agents"].items():
            print(f"  {a:20s} {s['n_scenarios_complete']:4d} {s['n_scenarios_any_missing']:4d} "
                  f"{s['mean_total']:7.2f} {s['run_to_run_sd']:7.2f} "
                  f"{s['mean_addie']:7.2f} {s['mean_traj']:7.2f}")
        comp = model["comparisons"]
        print(f"\n  Paired stats ({proposed} - baseline), Holm family = "
              f"{len(baselines)} comparisons in this model size:")
        for policy in ("complete_case", "zero", "min_score"):
            label = {"complete_case": "PRIMARY complete-case",
                     "zero": "sensitivity: failure=0",
                     "min_score": f"sensitivity: failure=min ({comp['min_score_floor']:.2f})"}[policy]
            print(f"  [{label}]")
            for r in comp["policies"][policy]:
                if r.get("n", 0) == 0 or "p_raw" not in r:
                    print(f"    {r['baseline']:20s} (no paired scenarios)")
                    continue
                lo, hi = r["ci95"]
                print(f"    {r['baseline']:20s} n={r['n']:3d} dTotal={r['mean_diff']:+7.2f} "
                      f"CI95[{lo:+6.2f},{hi:+6.2f}] r_rb={r['rank_biserial_r']:+.2f} "
                      f"p={r['p_raw']:.2e} p_holm={r['p_holm']:.2e}")

    inter = pooled["interaction"]
    print("\n## Method x model-size interaction (DiD, largest - smallest)")
    print(f"  {inter.get('definition', '')}")
    for r in inter.get("per_baseline", []):
        lo, hi = r["did_ci95"]
        trend = "  ".join(f"{k}:{v:+.2f}" for k, v in r["trend_delta_by_size"].items())
        print(f"  vs {r['baseline']:20s} n={r['n_paired_scenarios']:3d} "
              f"DiD={r['did_mean']:+6.2f} CI95[{lo:+6.2f},{hi:+6.2f}] p={r['did_wilcoxon_p']:.2e}")
        print(f"      trend: {trend}  slope/step={r['trend_ols_slope_per_size_step']:+.2f} "
              f"CI95[{r['trend_slope_ci95'][0]:+.2f},{r['trend_slope_ci95'][1]:+.2f}]")

    if pooled.get("token_usage"):
        print("\n## Token usage / runtime (operational metadata, mean per scenario)")
        for label, agents_tok in pooled["token_usage"]["models"].items():
            print(f"  [{label}]")
            for agent, fields in agents_tok.items():
                tt = fields.get("total_tokens", {}).get("mean", float("nan"))
                calls = fields.get("llm_calls", {}).get("mean", float("nan"))
                secs = fields.get("execution_time_seconds", {}).get("mean", float("nan"))
                print(f"    {agent:20s} total_tokens={tt:10.0f}  llm_calls={calls:6.1f}  exec_s={secs:7.1f}")

    if pooled.get("alignment"):
        print("\n## Alignment metrics (RQ2, pooled from alignment_scores.json)")
        for label, agents_metrics in pooled["alignment"]["models"].items():
            print(f"  {label}:")
            for agent, metrics in agents_metrics.items():
                parts = ", ".join(f"{m}={v['mean']:.3f}(n={v['n_scenarios']})"
                                  for m, v in metrics.items())
                print(f"    {agent:20s} {parts}")
    else:
        print("\n## Alignment metrics: no alignment_scores.json found — skipped (RQ2 metric "
              "not yet computed; re-run pooling after it lands).")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool model-ladder benchmark runs into per-scenario means + stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", action="append", default=[], metavar="LABEL=DIR1,DIR2,...",
        help="Explicit mapping model-label -> comma-separated run dirs. Repeatable; "
             "give models in size order (smallest first) — order is preserved.",
    )
    parser.add_argument(
        "--auto-glob", default=None, metavar="PATTERN",
        help="Glob of run dirs following the naming convention "
             "<dataset>_benchmark_<model>[_rN]_<ts>; groups by <model> and sorts "
             "sizes by the parameter count parsed from the name.",
    )
    parser.add_argument("--proposed", default=DEFAULT_PROPOSED)
    parser.add_argument(
        "--baselines", default=",".join(DEFAULT_BASELINES),
        help="Comma-separated baseline agent ids (defines the Holm family size).",
    )
    parser.add_argument("--out", default="results/pooled_ladder.json",
                        help="Output JSON path.")
    args = parser.parse_args()

    baselines = [b for b in args.baselines.split(",") if b]
    agents = baselines + [args.proposed]

    run_dirs_by_model: dict[str, list[Path]] = {}
    for spec in args.model:
        if "=" not in spec:
            parser.error(f"--model expects LABEL=DIR1,DIR2,... got: {spec}")
        label, dirs = spec.split("=", 1)
        run_dirs_by_model[label] = [Path(d) for d in dirs.split(",") if d]
    if args.auto_glob:
        for label, dirs in group_by_glob(args.auto_glob).items():
            run_dirs_by_model.setdefault(label, dirs)
    if not run_dirs_by_model:
        parser.error("no input: give --model and/or --auto-glob")

    for label, dirs in run_dirs_by_model.items():
        missing = [d for d in dirs if not d.is_dir()]
        if missing:
            parser.error(f"model '{label}': run dirs not found: {missing}")
        print(f"model '{label}': {len(dirs)} run(s)")
        for d in dirs:
            print(f"  - {d}")

    models = []
    for label, dirs in run_dirs_by_model.items():
        model = pool_model(label, dirs, agents)
        model["comparisons"] = compare_model(model, args.proposed, baselines)
        models.append(model)

    pooled = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "proposed": args.proposed,
            "baselines": baselines,
            "score_field": "total_score (per-scenario composite from comparison_report.json)",
            "pooling": "mean across runs per (model, agent, scenario)",
            "primary_failure_policy": "complete_case",
            "n_boot": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "size_order": [m["label"] for m in models],
        },
        "models": models,
        "interaction": interaction_tests(models, args.proposed, baselines),
        "alignment": pool_alignment(models, run_dirs_by_model, agents),
        "token_usage": pool_tokens(run_dirs_by_model, agents),
    }

    print_report(pooled, args.proposed, baselines)

    for model in pooled["models"]:
        model.pop("_totals", None)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pooled, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPooled JSON written: {out_path}")


if __name__ == "__main__":
    main()
