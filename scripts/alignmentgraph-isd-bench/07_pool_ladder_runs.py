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
    exist in scenario dirs (graceful skip otherwise) — with its own
    complete-case-plus-failure=0 layer, since RQ2 has exactly the same
    missing-output problem as RQ1 and must not be reported as unconditional

Usage:
  python scripts/alignmentgraph-isd-bench/07_pool_ladder_runs.py \
      --model qwen0.8b=results/test_90_benchmark_qwen3.5-0.8b-nitro_r1_...,results/..._r2_...,results/..._r3_... \
      --model qwen2b=... --model qwen4b=... --model qwen9b=...

  # or auto-group by model name parsed from the dir naming convention
  python scripts/alignmentgraph-isd-bench/07_pool_ladder_runs.py --auto-glob 'results/test_90_benchmark_*_r*'

The statistics helpers (wilcoxon_signed_rank, bootstrap_ci, holm_correct)
live here — this file is the single home of the algorithms since
5_aggregate_benchmark_results.py (their original co-owner) was removed;
its per-agent token-usage aggregation was absorbed as pool_tokens().
"""

from __future__ import annotations

import argparse
import glob as globmod
import itertools
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

EXACT_WILCOXON_N_MAX = 25  # exact permutation null up to this n_effective


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank (zeros discarded). Returns
    (W_plus, p_value, n_effective).

    For n_effective <= EXACT_WILCOXON_N_MAX the p-value is EXACT: the full
    sign-flip permutation null of W+ is enumerated by dynamic programming
    over (tie-averaged, doubled-to-integer) ranks. The normal approximation
    is only used beyond that — it is anti-conservative at tiny n (e.g. it
    can print p=0.11 at n=3 where the exact two-sided minimum is 0.25),
    which misled the first ladder report."""
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

    if n <= EXACT_WILCOXON_N_MAX:
        # DP over doubled ranks (tie-averaged ranks are multiples of 0.5,
        # so doubling makes them integers). dp[s] = #sign assignments whose
        # positive-rank doubled-sum equals s.
        dranks = [round(2 * r) for r in ranks]
        total = sum(dranks)
        dp = [0] * (total + 1)
        dp[0] = 1
        for r in dranks:
            for s in range(total, r - 1, -1):
                dp[s] += dp[s - r]
        w2 = round(2 * w_plus)
        lo, hi = min(w2, total - w2), max(w2, total - w2)
        count = sum(dp[: lo + 1]) + sum(dp[hi:])
        if lo == hi:
            count -= dp[lo]  # the single central point was counted in both tails
        p = min(1.0, count / (1 << n))
        return w_plus, p, n

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


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test p for the 2x2 table [[a, b], [c, d]]
    (sum of hypergeometric probabilities <= that of the observed table)."""
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2

    def p_table(x: int) -> float:
        return (math.comb(row1, x) * math.comb(row2, col1 - x)) / math.comb(n, col1)

    p_obs = p_table(a)
    lo_x = max(0, col1 - row2)
    hi_x = min(col1, row1)
    return min(1.0, sum(p for x in range(lo_x, hi_x + 1)
                        if (p := p_table(x)) <= p_obs * (1 + 1e-12)))


def holm_correct(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted: list[float] = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def _avg_ranks(vals: list[float]) -> list[float]:
    """Ranks 1..n with ties averaged."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties (stdlib only)."""
    n = len(xs)
    if n < 3:
        return float("nan")
    rx, ry = _avg_ranks(xs), _avg_ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def kendall_tau_b(xs: list[float], ys: list[float]) -> float:
    """Kendall's tau-b (ties corrected in both margins).

    Reported next to Spearman because the cross-encoder agreement is computed
    over the agent vector (n = 7), where Spearman is noisy; tau-b is the
    defensible small-n rank statistic and is the one prose should quote.
    """
    n = len(xs)
    if n < 3:
        return float("nan")
    concordant = discordant = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) / 2
    den = math.sqrt((n0 - tx) * (n0 - ty))
    return (concordant - discordant) / den if den else float("nan")


def spearman_p_exact(xs: list[float], ys: list[float],
                     n_max: int = 8) -> Optional[float]:
    """Two-sided permutation p for Spearman's rho; None when n > n_max.

    Full enumeration of the n! orderings of one vector against the other —
    5040 for the 7-agent ladder, i.e. instant and exact. Same
    exact-null-at-small-n convention as wilcoxon_signed_rank(), and for the
    same reason: an asymptotic approximation is anti-conservative at these n.
    """
    n = len(xs)
    if n < 3 or n > n_max:
        return None
    observed = spearman_rho(xs, ys)
    if observed != observed:  # NaN
        return None
    hits = total = 0
    for perm in itertools.permutations(ys):
        rho = spearman_rho(xs, list(perm))
        total += 1
        if rho == rho and abs(rho) >= abs(observed) - 1e-12:
            hits += 1
    return hits / total if total else None


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


def load_run(run_dir: Path,
             scores_filename: str = "alignment_scores.json") -> dict[str, dict]:
    """scenario_id -> {"rankings": {agent_id: ranking}, "alignment": dict|None}.

    ``scores_filename`` selects which encoder's artifact to read. The default
    is the primary (unsuffixed) file — the only one pooled["alignment"] and
    every downstream generator know about; sweep arms pass
    ``alignment_scores.<suffix>.json``.
    """
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
        align_path = scen_dir / scores_filename
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


# ── encoder sweep arms ───────────────────────────────────────────────────────

_SCORER_CACHE = {}


def _scorer():
    """06_score_alignment.py, for its encoder registry (digit-prefixed name)."""
    if "m" not in _SCORER_CACHE:
        import importlib.util  # noqa: PLC0415 - only needed on this path

        path = Path(__file__).resolve().with_name("06_score_alignment.py")
        spec = importlib.util.spec_from_file_location("_scorer", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # @dataclass resolves via sys.modules
        spec.loader.exec_module(module)
        _SCORER_CACHE["m"] = module
    return _SCORER_CACHE["m"]


def resolve_sensitivity_arms(args, run_dirs_by_model: dict[str, list[Path]]
                             ) -> list[tuple[str, str]]:
    """[(encoder key, artifact suffix)] for the sweep arms to pool.

    The primary encoder is never an arm: its unsuffixed artifact IS
    pooled["alignment"], and comparing it with itself is not a sensitivity
    check.

    --auto-encoders is registry-first, disk-second. The sensitivity design is
    fixed (ENCODER_PRESETS), so every non-primary preset is an arm whether or
    not it has been scored yet — an arm nobody ran then shows up as `skipped`
    instead of being absent, and absence is the one failure mode that makes a
    partial sweep look complete. Suffixed artifacts found on disk are added on
    top, so a one-off encoder outside the registry is still picked up.
    """
    scorer = _scorer()
    suffix_to_key = scorer.SUFFIX_TO_KEY
    arms: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(key: str, suffix: str) -> None:
        if suffix and key not in seen:
            seen.add(key)
            arms.append((key, suffix))

    for raw in (args.encoders or "").split(","):
        key = raw.strip()
        if not key:
            continue
        preset = scorer.ENCODER_PRESETS.get(key)
        add(key, preset[2] if preset else key)
    if args.auto_encoders:
        # The fixed design first: never-scored arms must still be reported.
        for preset_key, (_kind, _model, suffix) in scorer.ENCODER_PRESETS.items():
            add(preset_key, suffix)
        found: set[str] = set()
        for dirs in run_dirs_by_model.values():
            for run_dir in dirs:
                for _sid, scen_dir in iter_scenario_dirs(run_dir):
                    for path in scen_dir.glob("alignment_scores.*.json"):
                        found.add(path.name[len("alignment_scores."):-len(".json")])
                    break       # one scenario dir per run is enough to see the roster
        for suffix in sorted(found):
            add(suffix_to_key.get(suffix, suffix), suffix)
    return arms


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


def failure_tests(models: list[dict], proposed: str, baselines: list[str]) -> dict:
    """Scenario-level output-failure comparison (proposed vs each baseline).

    Unit = scenario (union across the size's runs); "failed" = the agent has no
    scorable output in at least one run (n_scenarios_any_missing). Fisher exact
    two-sided on the 2x2 [failed, ok] x [proposed, baseline]. This is the
    robustness face of the "harness compensates for small models" claim — it
    stays valid where tiny complete-case n makes the score comparison
    untestable. Run-level totals are reported descriptively only (runs of the
    same scenario are not independent)."""
    out = []
    for model in models:
        n_union = model["n_scenarios_union"]
        n_runs = model["n_runs"]
        summ = model["agents"]
        rows = []
        miss_p = summ[proposed]["n_scenarios_any_missing"]
        for b in baselines:
            miss_b = summ[b]["n_scenarios_any_missing"]
            p = fisher_exact_2x2(miss_p, n_union - miss_p, miss_b, n_union - miss_b)
            run_level_b = sum(info["missing_by_agent"].get(b, 0) for info in model["per_run"])
            rows.append({
                "baseline": b,
                "proposed_failed_scenarios": miss_p,
                "baseline_failed_scenarios": miss_b,
                "n_scenarios": n_union,
                "baseline_failed_run_level": f"{run_level_b}/{n_runs * n_union}",
                "fisher_p_two_sided": p,
            })
        holm = holm_correct([r["fisher_p_two_sided"] for r in rows])
        for r, ph in zip(rows, holm):
            r["p_holm"] = ph
        out.append({"label": model["label"], "rows": rows})
    return {
        "definition": (
            "Scenario-level failure = agent missing a scorable output in >=1 of the "
            "size's runs (n_scenarios_any_missing). Fisher exact two-sided, Holm over "
            f"{len(baselines)} comparisons per size. Run-level counts descriptive only."
        ),
        "per_model": out,
    }


def pool_alignment(models: list[dict], run_dirs_by_model: dict[str, list[Path]],
                   agents: list[str], proposed: str, baselines: list[str],
                   scores_filename: str = "alignment_scores.json",
                   return_scenario_maps: bool = False) -> Optional[dict]:
    """Pool alignment_scores.json metrics if any exist (RQ2). Graceful skip.

    Besides the per-metric means, computes for objective_assessment_alignment
    (the primary continuous, threshold-free endpoint — no composite):
      * 'comparisons': paired Wilcoxon (exact at small n) + Holm + bootstrap
        CI, proposed vs each baseline, paired by scenario on mean-of-runs
        (scenarios where both agents were scored) — the RQ2 inferential layer;
      * 'align_descriptive': per-agent mean + bootstrap 95% CI over
        per-scenario values (the absolute descriptive layer);
      * 'conditional_align': restricted to scenarios whose output yielded
        >=1 objective in every scored run — isolates alignment quality from
        parse failure, which the primary layer folds in as 0;
      * 'align_failures' + 'comparisons_failure_zero': the failure-policy
        layer, mirroring RQ1. The primary 'comparisons' are complete-case over
        AVAILABLE outputs — a crashed agent writes no *_output.json, so its
        scenario drops out of the paired set and the crash costs it nothing.
        The zero policy re-adds those cells at 0.0 from the scorer's
        'missing_outputs' roster. Report both; never call the primary layer
        unconditional;
      * 'interaction_align' (top-level under stats): method x model-size DiD on
        the paired delta (largest - smallest size), bootstrap CI +
        Wilcoxon p, mirroring the RQ1 interaction convention.

    Pooling is refused outright if the run dirs carry more than one
    (encoder, encoder_revision) pair — see the SystemExit below.
    """
    PRIMARY = "objective_assessment_alignment"
    out_models = {}
    stats_models = []
    align_maps_by_label: dict[str, dict[str, dict[str, float]]] = {}
    found_any = False
    n_scenario_files = 0
    encoder_label: Optional[str] = None
    encoder_revision: Optional[str] = None
    # (encoder, revision) -> one example file, to hard-fail on a mixed pool.
    configs: dict[tuple[Optional[str], Optional[str]], str] = {}
    # Do the artifacts carry the failure roster at all? Score files written
    # before 'missing_outputs' existed are indistinguishable from a clean run
    # unless we track this — and reporting "no failures" for them would be the
    # same silent overclaim the roster was added to fix.
    roster_available = False
    for model in models:
        label = model["label"]
        runs = [load_run(d, scores_filename) for d in run_dirs_by_model[label]]
        # metric_vals[agent][metric][scenario] = [per-run values]
        metric_vals: dict[str, dict[str, dict[str, list[float]]]] = {}
        # miss_counts[agent][scenario] = #runs where the agent produced no
        # scoreable output (the other half of the roster; see failure policies).
        miss_counts: dict[str, dict[str, int]] = {}
        roster_here = False
        for run in runs:
            for sid, payload in run.items():
                doc = payload.get("alignment")
                if doc is None:
                    continue
                found_any = True
                n_scenario_files += 1
                if isinstance(doc, dict):
                    configs.setdefault(
                        (doc.get("encoder"), doc.get("encoder_revision")),
                        f"{label}/{sid}",
                    )
                    if encoder_label is None:
                        encoder_label = doc.get("encoder")
                        encoder_revision = doc.get("encoder_revision")
                    if "missing_outputs" in doc:
                        roster_here = roster_available = True
                    for agent, reason in (doc.get("missing_outputs") or {}).items():
                        if agent in agents and reason:
                            per = miss_counts.setdefault(agent, {})
                            per[sid] = per.get(sid, 0) + 1
                for agent, metrics in extract_alignment_metrics(doc, agents).items():
                    for metric, val in metrics.items():
                        metric_vals.setdefault(agent, {}).setdefault(metric, {}) \
                            .setdefault(sid, []).append(val)
        if not metric_vals:
            continue
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

        # per-scenario values (mean of runs where scored) for the primary
        # metric that carries the inferential layer.
        def per_scenario(metric: str) -> dict[str, dict[str, float]]:
            return {
                agent: {sid: _mean(v) for sid, v in metrics.get(metric, {}).items()}
                for agent, metrics in metric_vals.items()
            }

        def paired_rows(metric_map: dict[str, dict[str, float]]) -> list[dict]:
            rows = []
            for b in baselines:
                pmap, bmap = metric_map.get(proposed, {}), metric_map.get(b, {})
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
            return rows

        align_map = per_scenario(PRIMARY)
        align_maps_by_label[label] = align_map
        # Conditional-on-parseable-output: the documented criterion is ">=1
        # objective", so read the objective COUNT, not the incidental
        # definedness of porter_mean (which is also None when the objectives
        # carry no Bloom-classifiable verb — a scenario that has objectives and
        # belongs in the conditional set). Require it in every scored run, so a
        # mean-of-runs is never half parse-failure.
        cond_map = {
            agent: {
                sid: val
                for sid, val in align_map[agent].items()
                if min(metric_vals[agent].get("counts.objectives", {}).get(sid, [0]))
                >= 1
            }
            for agent in align_map
        }
        # Failure policy, mirroring RQ1. align_map is complete-case over
        # AVAILABLE outputs: an agent that crashed has no *_output.json, so its
        # scenario silently leaves the paired set and the crash costs it
        # nothing. zero_map re-adds those (agent, scenario, run) cells at 0.0 —
        # the floor of every endpoint here, and the same value a parseable but
        # objective-less output already scores.
        zero_map: dict[str, dict[str, float]] = {}
        for agent in sorted(set(align_map) | set(miss_counts)):
            per_scen = metric_vals.get(agent, {}).get(PRIMARY, {})
            row: dict[str, float] = {}
            for sid in set(per_scen) | set(miss_counts.get(agent, {})):
                vals = list(per_scen.get(sid, []))
                denom = len(vals) + miss_counts.get(agent, {}).get(sid, 0)
                row[sid] = (sum(vals) / denom) if denom else 0.0
            if row:
                zero_map[agent] = row
        stats_models.append({
            "label": label,
            "align_descriptive": {
                agent: {
                    "n": len(vals),
                    "mean": _mean(list(vals.values())),
                    "ci95": list(bootstrap_ci(list(vals.values()))),
                }
                for agent, vals in sorted(align_map.items()) if vals
            },
            "conditional_align": {
                agent: {"n": len(vals), "mean": _mean(list(vals.values()))}
                for agent, vals in sorted(cond_map.items()) if vals
            },
            "align_failures": {
                agent: {
                    "n_scenarios_affected": len(sids),
                    "n_agent_run_cells_missing": sum(sids.values()),
                    "scenarios": sorted(sids),
                }
                for agent, sids in sorted(miss_counts.items()) if sids
            } if roster_here else None,
            "comparisons": paired_rows(align_map),
            "comparisons_failure_zero": (
                paired_rows(zero_map) if roster_here else None),
            "failure_zero_descriptive": {
                agent: {"n": len(vals), "mean": _mean(list(vals.values()))}
                for agent, vals in sorted(zero_map.items()) if vals
            } if roster_here else None,
        })
    if not found_any:
        return None

    # One pool = one protocol configuration. Mixing encoders (or two weight
    # revisions of one encoder) across run dirs produces a mean that measures
    # nothing, and it used to be invisible because only the first file's
    # metadata was kept. Fail loudly instead.
    if len(configs) > 1:
        detail = "; ".join(
            f"{enc!r} @ revision {rev!r} (e.g. {where})"
            for (enc, rev), where in sorted(configs.items(), key=lambda kv: kv[1])
        )
        raise SystemExit(
            f"[{scores_filename}] the pooled run dirs were scored with "
            f"{len(configs)} different encoder configurations: {detail}. "
            "Re-score every run dir with one configuration "
            "(06_score_alignment.py --overwrite) before pooling — a mean over "
            "two encoders is not an alignment measurement."
        )

    # Method x model-size interaction on the primary endpoint (largest -
    # smallest size), mirroring interaction_tests()'s convention for RQ1 Total.
    interaction_align = None
    labels_with = [m["label"] for m in models if m["label"] in align_maps_by_label]
    if len(labels_with) >= 2:
        small, large = labels_with[0], labels_with[-1]
        per_baseline = []
        for b in baselines:
            def delta_map(label: str) -> dict[str, float]:
                pmap = align_maps_by_label[label].get(proposed, {})
                bmap = align_maps_by_label[label].get(b, {})
                return {s: pmap[s] - bmap[s] for s in set(pmap) & set(bmap)}

            d_small, d_large = delta_map(small), delta_map(large)
            shared = sorted(set(d_small) & set(d_large))
            if not shared:
                per_baseline.append({"baseline": b, "n_paired_scenarios": 0})
                continue
            did_vals = [d_large[s] - d_small[s] for s in shared]
            lo, hi = bootstrap_ci(did_vals)
            _, p, _ = wilcoxon_signed_rank(did_vals)
            per_baseline.append({
                "baseline": b,
                "n_paired_scenarios": len(shared),
                "did_mean": _mean(did_vals),
                "did_ci95": [lo, hi],
                "did_wilcoxon_p": p,
            })
        interaction_align = {
            "definition": (
                "DiD on the paired assessment-alignment delta (proposed - "
                f"baseline): value at the largest size ({large}) minus the "
                f"smallest ({small}), per scenario; negative = the gap narrows "
                "as the model grows (i.e. widens toward small models). "
                f"Bootstrap CI ({N_BOOT}, seed {BOOT_SEED}) + Wilcoxon p on the "
                "per-scenario DiD values."
            ),
            "smallest": small,
            "largest": large,
            "per_baseline": per_baseline,
        }

    result = {
        "source": f"{scores_filename} per scenario dir (pooled mean across runs, then across scenarios)",
        "encoder_label": encoder_label,
        "encoder_revision": encoder_revision,
        "encoder_revision_pinned": encoder_revision is not None,
        "failure_roster_available": roster_available,
        "n_scenario_files": n_scenario_files,
        "models": out_models,
        "stats": {
            "definition": (
                "Primary endpoint = objective_assessment_alignment "
                "(continuous mean-max RECTIFIED cosine, max(0,cos) on [0,1]; "
                "no matching threshold, no composite). "
                "'comparisons' = paired diffs (proposed - baseline) per "
                "scenario (mean of runs where scored; scenarios scored for "
                "both agents), Wilcoxon two-sided (exact null for n<=%d), "
                "Holm over %d comparisons per size, bootstrap CI (%d, seed "
                "%d). align_descriptive = per-agent mean + bootstrap 95%% "
                "CI over per-scenario values. Both are COMPLETE-CASE over "
                "available outputs, not unconditional: an agent run that "
                "crashed leaves no output to score. align_failures counts "
                "those cells and comparisons_failure_zero re-runs the paired "
                "test with them imputed at 0.0 (the endpoint's floor; a "
                "min-score policy is degenerate here because the observed "
                "minimum IS 0.0). conditional_align = over scenarios whose "
                "output yielded >=1 objective in every scored run (isolates "
                "alignment quality from parse failure). interaction_align = "
                "method x model-size DiD on the paired delta (complete-case)."
                % (EXACT_WILCOXON_N_MAX, len(baselines), N_BOOT, BOOT_SEED)
            ),
            "per_model": stats_models,
            "interaction_align": interaction_align,
        },
    }
    if return_scenario_maps:
        # Per-scenario primary-metric values, keyed label -> agent -> scenario.
        # Only the sensitivity layer needs them (scenario-level agreement);
        # main() pops the key before the JSON dump.
        result["_align_maps"] = align_maps_by_label
    return result



def _rank_of(agent: str, means: dict[str, float]) -> Optional[int]:
    """1-based rank of ``agent`` by descending mean (1 = best)."""
    if agent not in means:
        return None
    order = sorted(means, key=lambda a: -means[a])
    return order.index(agent) + 1


def _agreement(primary: dict, arm: dict, primary_maps: dict, arm_maps: dict,
               models: list[dict], agents: list[str], proposed: str,
               baselines: list[str], metric: str) -> dict:
    """Cross-encoder agreement between one sweep arm and the primary encoder.

    The headline is a rank correlation over the AGENT VECTOR, per model size:
    that is literally the claim the paper makes ("the ranking of agents is the
    same under a different encoder"). Spearman and Kendall tau-b are both
    reported — at n = 7 agents Spearman is noisy and tau-b is the conservative
    one to quote.

    A high rho can still coexist with a flipped conclusion, so three non-rank
    corroborations travel with it: top-1 agreement, the proposed agent's rank,
    and whether the six paired deltas keep their sign and their Holm verdict.
    A fourth, scenario-level Spearman for the proposed agent (n ~ 90), answers
    the different question of whether the encoders agree about WHICH scenarios
    are well aligned — and unlike the agent vector it is not small-n.
    """
    per_model = []
    for model in models:
        label = model["label"]
        p_agents = (primary.get("models") or {}).get(label) or {}
        a_agents = (arm.get("models") or {}).get(label) or {}
        common = [
            a for a in agents
            if (p_agents.get(a, {}).get(metric, {}).get("mean") is not None
                and a_agents.get(a, {}).get(metric, {}).get("mean") is not None)
        ]
        if len(common) < 3:
            per_model.append({"label": label, "n_agents": len(common),
                              "note": "too few shared agents for a rank correlation"})
            continue
        p_means = {a: p_agents[a][metric]["mean"] for a in common}
        a_means = {a: a_agents[a][metric]["mean"] for a in common}
        xs = [p_means[a] for a in common]
        ys = [a_means[a] for a in common]

        # scenario-level, proposed agent only (n ~ 90, not small-n)
        p_scen = (primary_maps.get(label) or {}).get(proposed, {})
        a_scen = (arm_maps.get(label) or {}).get(proposed, {})
        shared = sorted(set(p_scen) & set(a_scen))
        scen_rho = (spearman_rho([p_scen[s] for s in shared],
                                 [a_scen[s] for s in shared])
                    if len(shared) >= 3 else None)

        # do the paired verdicts survive the encoder swap?
        def rows_of(pooled: dict) -> dict[str, dict]:
            for entry in (pooled.get("stats") or {}).get("per_model") or []:
                if entry.get("label") == label:
                    return {r["baseline"]: r for r in entry.get("comparisons") or []}
            return {}

        p_rows, a_rows = rows_of(primary), rows_of(arm)
        same_sign = same_verdict = n_cmp = 0
        for b in baselines:
            pr, ar = p_rows.get(b), a_rows.get(b)
            if not pr or not ar or "mean_diff" not in pr or "mean_diff" not in ar:
                continue
            n_cmp += 1
            if (pr["mean_diff"] > 0) == (ar["mean_diff"] > 0):
                same_sign += 1
            if (pr.get("p_holm", 1.0) < 0.05) == (ar.get("p_holm", 1.0) < 0.05):
                same_verdict += 1

        per_model.append({
            "label": label,
            "n_agents": len(common),
            "spearman_rho_agent_means": spearman_rho(xs, ys),
            "spearman_p_exact": spearman_p_exact(xs, ys),
            "kendall_tau_b_agent_means": kendall_tau_b(xs, ys),
            "mean_abs_diff_agent_means": _mean([abs(a - b) for a, b in zip(xs, ys)]),
            "top1_agent_primary": max(p_means, key=lambda a: p_means[a]),
            "top1_agent_here": max(a_means, key=lambda a: a_means[a]),
            "proposed_rank_primary": _rank_of(proposed, p_means),
            "proposed_rank_here": _rank_of(proposed, a_means),
            "n_scenarios_paired_proposed": len(shared),
            "spearman_rho_scenario_level_proposed": scen_rho,
            "sign_agreement_vs_baselines": {
                "n": n_cmp, "n_same_sign": same_sign,
                "n_same_holm_verdict_at_05": same_verdict,
            },
        })

    rhos = [r["spearman_rho_agent_means"] for r in per_model
            if isinstance(r.get("spearman_rho_agent_means"), float)
            and r["spearman_rho_agent_means"] == r["spearman_rho_agent_means"]]
    taus = [r["kendall_tau_b_agent_means"] for r in per_model
            if isinstance(r.get("kendall_tau_b_agent_means"), float)
            and r["kendall_tau_b_agent_means"] == r["kendall_tau_b_agent_means"]]
    return {
        "per_model": per_model,
        "pooled_over_sizes": {
            "n_sizes": len(per_model),
            "min_spearman_rho": min(rhos) if rhos else None,
            "mean_spearman_rho": _mean(rhos) if rhos else None,
            "min_kendall_tau_b": min(taus) if taus else None,
            "n_sizes_top1_agrees": sum(
                1 for r in per_model
                if r.get("top1_agent_here") and r["top1_agent_here"] == r.get("top1_agent_primary")),
            "n_sizes_proposed_rank_one": sum(
                1 for r in per_model if r.get("proposed_rank_here") == 1),
        },
    }


def pool_alignment_sensitivity(models: list[dict],
                               run_dirs_by_model: dict[str, list[Path]],
                               agents: list[str], proposed: str,
                               baselines: list[str], arms: list[tuple[str, str]],
                               primary_pooled: Optional[dict], primary_key: str,
                               metric: str) -> Optional[dict]:
    """Pool the sweep arms and score each one's agreement with the primary.

    Encoder is swept exactly like model size: the SAME runs are re-scored by a
    different encoder, so every arm reuses pool_alignment() unchanged and only
    differs in which ``alignment_scores.<suffix>.json`` it reads. An arm with
    no artifacts on disk is recorded as skipped rather than dropped, so a
    half-finished sweep is visible in the JSON instead of looking complete.
    """
    if not arms:
        return None
    primary_maps = (primary_pooled or {}).get("_align_maps") or {}
    encoders = []
    for key, suffix in arms:
        scores_filename = f"alignment_scores.{suffix}.json" if suffix else "alignment_scores.json"
        pooled = pool_alignment(models, run_dirs_by_model, agents, proposed,
                                baselines, scores_filename=scores_filename,
                                return_scenario_maps=True)
        if pooled is None:
            encoders.append({"key": key, "suffix": suffix,
                             "scores_filename": scores_filename,
                             "skipped": f"no {scores_filename} found under any run dir"})
            continue
        arm_maps = pooled.pop("_align_maps", {})
        entry = {
            "key": key,
            "suffix": suffix,
            "scores_filename": scores_filename,
            "encoder_label": pooled.get("encoder_label"),
            "encoder_revision": pooled.get("encoder_revision"),
            "coverage": {"n_scenario_files": pooled.get("n_scenario_files")},
            "models": pooled.get("models"),
            "stats": pooled.get("stats"),
        }
        entry["agreement_with_primary"] = (
            _agreement(primary_pooled, pooled, primary_maps, arm_maps, models,
                       agents, proposed, baselines, metric)
            if primary_pooled else
            {"note": f"no primary ({primary_key}) alignment_scores.json pooled — "
                     "agreement undefined", "per_model": [], "pooled_over_sizes": {}}
        )
        encoders.append(entry)

    scored = [e for e in encoders if "skipped" not in e]
    rhos, taus, cells, first, p_max = [], [], 0, 0, None
    for entry in scored:
        for row in entry["agreement_with_primary"].get("per_model") or []:
            rho, tau = row.get("spearman_rho_agent_means"), row.get("kendall_tau_b_agent_means")
            if isinstance(rho, float) and rho == rho:
                rhos.append(rho)
            if isinstance(tau, float) and tau == tau:
                taus.append(tau)
            cells += 1
            first += 1 if row.get("proposed_rank_here") == 1 else 0
        for entry_model in (entry.get("stats") or {}).get("per_model") or []:
            for row in entry_model.get("comparisons") or []:
                if "p_holm" in row:
                    p_max = row["p_holm"] if p_max is None else max(p_max, row["p_holm"])
    return {
        "definition": (
            "Encoder as a swept dimension: the SAME runs re-scored by each "
            f"encoder. '{primary_key}' owns the unsuffixed artifact and stays in "
            "pooled['alignment']; the arms here are read from "
            "alignment_scores.<suffix>.json. agreement_with_primary compares "
            f"the per-size agent vector on '{metric}' — Spearman rho with an "
            "exact permutation p (n<=8) and Kendall tau-b, plus top-1 / "
            "proposed-rank / sign / Holm-verdict agreement, and a "
            "scenario-level Spearman for the proposed agent."
        ),
        "primary_encoder": primary_key,
        "metric": metric,
        "encoders": encoders,
        "summary": {
            "n_encoders": len(scored),
            "n_skipped": len(encoders) - len(scored),
            "min_spearman_rho_any_encoder_any_size": min(rhos) if rhos else None,
            "min_kendall_tau_b_any_encoder_any_size": min(taus) if taus else None,
            "top1_stable_all_encoders_all_sizes": all(
                row.get("top1_agent_here") == row.get("top1_agent_primary")
                for e in scored
                for row in e["agreement_with_primary"].get("per_model") or []
            ) if cells else None,
            "n_cells_proposed_ranks_first": first,
            "n_cells": cells,
            "max_p_holm_any_encoder_any_size": p_max,
        },
    }


def pool_tokens(run_dirs_by_model: dict[str, list[Path]],
                agents: list[str]) -> Optional[dict]:
    """Pool per-agent token usage / runtime from *_trajectory.json metadata.

    (Absorbed from the deleted 5_aggregate_benchmark_results.py "tokens"
    section, in pooled form: mean across runs per scenario, then across
    scenarios.) Operational metadata only -- never part of Total or the
    alignment endpoints.
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

    if pooled.get("failure_rates"):
        print("\n## Output-failure rates (scenario-level, Fisher exact vs proposed)")
        print(f"  {pooled['failure_rates']['definition']}")
        for entry in pooled["failure_rates"]["per_model"]:
            print(f"  [{entry['label']}]")
            for r in entry["rows"]:
                print(f"    vs {r['baseline']:20s} failed {r['proposed_failed_scenarios']:2d} vs "
                      f"{r['baseline_failed_scenarios']:2d} /{r['n_scenarios']} scenarios "
                      f"(run-level {r['baseline_failed_run_level']:>8s})  "
                      f"p={r['fisher_p_two_sided']:.2e} p_holm={r['p_holm']:.2e}")

    if pooled.get("alignment"):
        print("\n## Alignment metrics (RQ2, pooled from alignment_scores.json)")
        align = pooled["alignment"]
        print(f"  encoder = {align.get('encoder_label')} "
              f"@ revision {align.get('encoder_revision') or 'UNPINNED'}")
        if not align.get("encoder_revision_pinned"):
            print("  NOTE: no encoder_revision in the score files — reproducibility "
                  "rests on the embedding cache only. Set <SLOT>_EMBED_REVISION "
                  "(or --embed-model-revision) and re-score to pin the weights.")
        for label, agents_metrics in pooled["alignment"]["models"].items():
            print(f"  {label}:")
            for agent, metrics in agents_metrics.items():
                parts = ", ".join(f"{m}={v['mean']:.3f}(n={v['n_scenarios']})"
                                  for m, v in metrics.items())
                print(f"    {agent:20s} {parts}")
        stats = pooled["alignment"].get("stats")
        if stats:
            print("\n## Objective–assessment alignment — paired stats + descriptive (RQ2)")
            print(f"  {stats['definition']}")
            for entry in stats["per_model"]:
                print(f"  [{entry['label']}]")
                for r in entry["comparisons"]:
                    if r.get("n", 0) == 0 or "p_raw" not in r:
                        print(f"    vs {r['baseline']:20s} (no paired scenarios)")
                        continue
                    lo, hi = r["ci95"]
                    print(f"    vs {r['baseline']:20s} n={r['n']:3d} dAlign={r['mean_diff']:+7.3f} "
                          f"CI95[{lo:+6.3f},{hi:+6.3f}] r_rb={r['rank_biserial_r']:+.2f} "
                          f"p={r['p_raw']:.2e} p_holm={r['p_holm']:.2e}")
                desc = ", ".join(
                    f"{a}={v['mean']:.3f}[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]"
                    for a, v in entry.get("align_descriptive", {}).items())
                if desc:
                    print(f"    Align mean [CI95]: {desc}")
                cond = ", ".join(f"{a}={v['mean']:.3f}(n={v['n']})"
                                 for a, v in entry.get("conditional_align", {}).items())
                if cond:
                    print(f"    conditional (>=1 objective): {cond}")
                fails = entry.get("align_failures")
                if fails is None:
                    print("    missing outputs: UNKNOWN — these score files predate "
                          "the missing_outputs roster, so a crashed agent is "
                          "invisible here and the numbers above are complete-case "
                          "only. Re-score with 06_score_alignment.py --overwrite "
                          "to get the failure=0 layer.")
                elif fails:
                    fail_desc = ", ".join(
                        f"{a}={v['n_agent_run_cells_missing']} cell(s)/"
                        f"{v['n_scenarios_affected']} scen"
                        for a, v in fails.items())
                    print(f"    missing outputs (excluded above): {fail_desc}")
                    for r in entry.get("comparisons_failure_zero") or []:
                        if r.get("n", 0) == 0 or "p_raw" not in r:
                            continue
                        lo, hi = r["ci95"]
                        print(f"      [failure=0] vs {r['baseline']:20s} n={r['n']:3d} "
                              f"dAlign={r['mean_diff']:+7.3f} "
                              f"CI95[{lo:+6.3f},{hi:+6.3f}] "
                              f"p_holm={r['p_holm']:.2e}")
                else:
                    print("    missing outputs: none — complete-case == "
                          "failure=0 for this size")
            inter = stats.get("interaction_align")
            if inter:
                print(f"  [interaction {inter['largest']} - {inter['smallest']}]")
                for r in inter["per_baseline"]:
                    if not r.get("n_paired_scenarios"):
                        print(f"    vs {r['baseline']:20s} (no paired scenarios)")
                        continue
                    lo, hi = r["did_ci95"]
                    print(f"    vs {r['baseline']:20s} n={r['n_paired_scenarios']:3d} "
                          f"DiD={r['did_mean']:+7.3f} CI95[{lo:+6.3f},{hi:+6.3f}] "
                          f"p={r['did_wilcoxon_p']:.2e}")
    else:
        print("\n## Alignment metrics: no alignment_scores.json found — skipped (RQ2 metric "
              "not yet computed; re-run pooling after it lands).")

    sens = pooled.get("alignment_sensitivity")
    if sens:
        print("\n## Encoder sensitivity (RQ2 robustness — encoder as a swept dimension)")
        print(f"  primary = {sens['primary_encoder']} (alignment_scores.json), "
              f"metric = {sens['metric']}")
        for entry in sens["encoders"]:
            if "skipped" in entry:
                print(f"  [{entry['key']}] SKIPPED — {entry['skipped']}")
                continue
            cov = (entry.get("coverage") or {}).get("n_scenario_files")
            print(f"  [{entry['key']} / {entry.get('encoder_label')}]  "
                  f"coverage {cov} scenario file(s)")
            for row in entry["agreement_with_primary"].get("per_model") or []:
                if "note" in row:
                    print(f"    {row['label']:14s} {row['note']}")
                    continue
                p_exact = row.get("spearman_p_exact")
                worst = max(
                    (r.get("p_holm", 0.0)
                     for m in (entry.get("stats") or {}).get("per_model") or []
                     if m.get("label") == row["label"]
                     for r in m.get("comparisons") or []),
                    default=float("nan"))
                top1 = "=" if row["top1_agent_here"] == row["top1_agent_primary"] else "DIFFERS"
                print(f"    {row['label']:14s} rho={row['spearman_rho_agent_means']:.3f} "
                      f"(p_exact={p_exact if p_exact is None else f'{p_exact:.2e}'}) "
                      f"tau_b={row['kendall_tau_b_agent_means']:.3f}  "
                      f"top1={row['top1_agent_here']} ({top1})  "
                      f"rank_prop={row['proposed_rank_here']}  worst p_holm={worst:.2e}")
        s = sens["summary"]
        rho_min = s.get("min_spearman_rho_any_encoder_any_size")
        print(f"  HEADLINE: min rho over {s['n_encoders']} encoder(s) x sizes = "
              f"{'n/a' if rho_min is None else f'{rho_min:.3f}'} ; "
              f"{proposed} ranks first in {s['n_cells_proposed_ranks_first']}/{s['n_cells']} cells")


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
    parser.add_argument(
        "--encoders", default="", metavar="K1,K2,...",
        help="Also pool these encoder arms of the RQ2 sensitivity sweep into "
             "'alignment_sensitivity' (scored by 06_score_alignment.py "
             "--encoder-presets). pooled['alignment'] always stays the primary "
             "encoder. Empty (default) = output identical to before.",
    )
    parser.add_argument(
        "--auto-encoders", action="store_true",
        help="Discover the arms from disk (any alignment_scores.<suffix>.json "
             "present in the run dirs) instead of naming them.",
    )
    parser.add_argument(
        "--primary-encoder", default=None,
        help="Encoder that owns the unsuffixed alignment_scores.json "
             "(default: 06_score_alignment.py's PRIMARY_ENCODER).",
    )
    parser.add_argument(
        "--sensitivity-metric", default="objective_assessment_alignment",
        help="Endpoint the cross-encoder agreement statistics run on.",
    )
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

    arms = resolve_sensitivity_arms(args, run_dirs_by_model)
    primary_key = args.primary_encoder or _scorer().PRIMARY_ENCODER
    alignment = pool_alignment(models, run_dirs_by_model, agents, args.proposed,
                               baselines, return_scenario_maps=bool(arms))
    sensitivity = pool_alignment_sensitivity(
        models, run_dirs_by_model, agents, args.proposed, baselines, arms,
        alignment, primary_key, args.sensitivity_metric,
    ) if arms else None
    if alignment is not None:
        alignment.pop("_align_maps", None)

    pooled = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "proposed": args.proposed,
            "baselines": baselines,
            "score_field": "total_score (per-scenario composite from comparison_report.json)",
            "pooling": "mean across runs per (model, agent, scenario)",
            "primary_failure_policy": "complete_case",
            "rq2_failure_policies": (
                "complete_case (primary) + zero; RQ2 has no min_score policy "
                "because the alignment endpoints already floor at 0.0"
            ),
            "n_boot": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "size_order": [m["label"] for m in models],
        },
        "models": models,
        "interaction": interaction_tests(models, args.proposed, baselines),
        "failure_rates": failure_tests(models, args.proposed, baselines),
        "alignment": alignment,
        "token_usage": pool_tokens(run_dirs_by_model, agents),
    }
    # Added only when a sweep was requested, so the default invocation keeps
    # producing byte-identical JSON for the existing generators.
    if sensitivity is not None:
        pooled["config"]["primary_encoder"] = primary_key
        pooled["config"]["encoders"] = [k for k, _s in arms]
        pooled["alignment_sensitivity"] = sensitivity

    print_report(pooled, args.proposed, baselines)

    for model in pooled["models"]:
        model.pop("_totals", None)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pooled, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPooled JSON written: {out_path}")


if __name__ == "__main__":
    main()
