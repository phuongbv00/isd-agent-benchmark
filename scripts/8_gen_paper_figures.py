#!/usr/bin/env python3
"""Generate paper figures (RQ1/RQ2 statistical charts) from pooled_ladder.json.

Companion of scripts/8_gen_paper_tables.py — same stage of the pipeline, same
input, but emits figures instead of tables. Writes every figure as both .pdf
(vector, for LaTeX \\includegraphics) and .png (dpi 200, for quick preview)
into docs/paper/acm/vn/generated/figures/ :

  fig_rq1_ladder          mean Total vs model size, one line per agent,
                          error bar = run-to-run SD (headline RQ1 figure)
  fig_rq1_delta           per-baseline paired delta (proposed - baseline) vs
                          size with bootstrap 95% CI band, Holm-significance
                          markers, and the OLS delta trend slope
  fig_rq1_components      ADDIE vs Trajectory component means vs size
  fig_rq2_ladder          alignment composite vs model size per agent
                          (RQ2 headline: does the gap widen as models shrink?)
  fig_rq2_components      heatmap agents x 5 alignment components per size
  fig_rq2_vs_rq1_scatter  per-scenario alignment composite vs Total (judge),
                          Spearman rho per size  [needs --runs-glob]
  fig_failure_rate        % scenarios with no scorable output per agent/size
                          (mean of runs, min-max whiskers)
  fig_cost_quality        mean Total vs mean total tokens / execution time

All figure text is English (Vietnamese captions live in the LaTeX source, as
with the generated tables). Numbers come exclusively from pooled_ladder.json
and the raw run dirs — nothing hand-typed.

Usage:
  python scripts/8_gen_paper_figures.py                       # pooled JSON only
  python scripts/8_gen_paper_figures.py \
      --runs-glob 'results/test_90_benchmark_*_r*'            # + scatter figure
  python scripts/8_gen_paper_figures.py --demo                # synthetic layout test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:  # pragma: no cover
    print(
        "Error: matplotlib is required for this script.\n"
        "Install the benchmark deps from the repo root:  pip install -e .",
        file=sys.stderr,
    )
    sys.exit(1)

BENCH_ROOT = Path(__file__).resolve().parents[1]   # isd-agent-benchmark/
THESIS_ROOT = Path(__file__).resolve().parents[2]  # master-thesis/
DEFAULT_POOLED = BENCH_ROOT / "results" / "pooled_ladder.json"
DEFAULT_OUTDIR = THESIS_ROOT / "docs" / "paper" / "acm" / "vn" / "generated" / "figures"

# Same identities as scripts/8_gen_paper_tables.py (kept in sync by hand).
AGENT_DISPLAY = {
    "baseline": "Baseline",
    "eduplanner": "EduPlanner",
    "addie-agent": "ADDIE-Agent",
    "rpisd-agent": "RPISD-Agent",
    "dick-carey-agent": "Dick-Carey-Agent",
    "react-isd": "ReAct-ADDIE",
    "alignmentgraph-isd": "AlignmentGraph-ISD",
}
AGENT_ORDER = [
    "baseline", "eduplanner", "addie-agent", "rpisd-agent",
    "dick-carey-agent", "react-isd", "alignmentgraph-isd",
]
PROPOSED = "alignmentgraph-isd"

# Okabe-Ito-derived categorical palette, fixed per agent (never cycled),
# validated colorblind-safe as an ordered set; the proposed agent gets the
# most salient slot plus a heavier line + distinct marker (secondary encoding).
AGENT_COLOR = {
    "baseline": "#E69F00",
    "eduplanner": "#56B4E9",
    "addie-agent": "#D55E00",
    "rpisd-agent": "#009E73",
    "dick-carey-agent": "#CC79A7",
    "react-isd": "#882255",
    "alignmentgraph-isd": "#0072B2",
}
AGENT_MARKER = {
    "baseline": "o", "eduplanner": "s", "addie-agent": "^", "rpisd-agent": "v",
    "dick-carey-agent": "P", "react-isd": "X", "alignmentgraph-isd": "D",
}

# The 5 alignment components (order fixed; same set as 7_score_alignment.py
# COMPONENTS minus the composite, which gets its own figure).
RQ2_COMPONENTS = [
    ("porter_mean", "Porter"),
    ("webb_range", "Webb range"),
    ("webb_bloom_consistency", "Webb-Bloom"),
    ("embedding_coverage", "Emb. coverage"),
    ("embedding_precision", "Emb. precision"),
]

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
})


# ── shared helpers ───────────────────────────────────────────────────────────

def size_of_label(label: str) -> float:
    m = re.findall(r"(\d+(?:\.\d+)?)b\b", label.lower())
    return float(m[-1]) if m else float("nan")


def size_display(label: str) -> str:
    s = size_of_label(label)
    return label if math.isnan(s) else f"{s:g}B"


def _nan(v) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")


def agents_in(pooled: dict) -> list[str]:
    present = {a for m in pooled["models"] for a in m.get("agents", {})}
    return [a for a in AGENT_ORDER if a in present]


def line_kw(agent: str) -> dict:
    """Series style: proposed is emphasized, baselines recessive."""
    proposed = agent == PROPOSED
    return {
        "color": AGENT_COLOR.get(agent, "#666666"),
        "marker": AGENT_MARKER.get(agent, "o"),
        "linewidth": 2.0 if proposed else 1.2,
        "markersize": 5.5 if proposed else 4,
        "zorder": 5 if proposed else 3,
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
    }


def agent_legend(fig, agents: list[str], **kw) -> None:
    handles = [
        Line2D([], [], label=AGENT_DISPLAY.get(a, a), linestyle="-", **{
            k: v for k, v in line_kw(a).items() if k != "zorder"})
        for a in agents
    ]
    kw.setdefault("loc", "upper center")
    kw.setdefault("bbox_to_anchor", (0.5, 0.0))
    kw.setdefault("ncol", 4)
    kw.setdefault("frameon", False)
    fig.legend(handles=handles, **kw)


def sized_axis(ax, sizes: list[float], labels: list[str]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([size_display(lb) for lb in labels])
    ax.minorticks_off()
    ax.set_xlabel("Model size (Qwen3.5)")


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties (stdlib only)."""
    def ranks(vals: list[float]) -> list[float]:
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
    n = len(xs)
    if n < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def load_pool_module():
    """Import 7_pool_ladder_runs.py (digit-prefixed name) for its loaders."""
    path = Path(__file__).resolve().with_name("7_pool_ladder_runs.py")
    spec = importlib.util.spec_from_file_location("pool_ladder_runs", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def save(fig, outdir: Path, name: str, written: list[Path]) -> None:
    for ext in ("pdf", "png"):
        path = outdir / f"{name}.{ext}"
        fig.savefig(path, dpi=200)
        written.append(path)
    plt.close(fig)


# ── figure builders ──────────────────────────────────────────────────────────

def fig_rq1_ladder(pooled: dict, outdir: Path, written: list[Path]) -> None:
    models = pooled["models"]
    agents = agents_in(pooled)
    sizes = [size_of_label(m["label"]) for m in models]
    labels = [m["label"] for m in models]

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for a in agents:
        ys = [_nan(m["agents"].get(a, {}).get("mean_total")) for m in models]
        errs = [_nan(m["agents"].get(a, {}).get("run_to_run_sd")) for m in models]
        errs = [0.0 if math.isnan(e) else e for e in errs]
        ax.errorbar(sizes, ys, yerr=errs, capsize=2, elinewidth=0.8, **line_kw(a))
    # direct label on the proposed agent (secondary encoding beside the legend)
    prop_y = _nan(models[-1]["agents"].get(PROPOSED, {}).get("mean_total"))
    if not math.isnan(prop_y):
        ax.annotate(AGENT_DISPLAY[PROPOSED], xy=(sizes[-1], prop_y),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=AGENT_COLOR[PROPOSED],
                    fontweight="bold")
        ax.set_xlim(right=sizes[-1] * 2.4)
    sized_axis(ax, sizes, labels)
    ax.set_ylabel("Total composite (0.7·ADDIE + 0.3·Traj)")
    ax.grid(axis="x", visible=False)
    agent_legend(fig, agents, bbox_to_anchor=(0.5, -0.02))
    save(fig, outdir, "fig_rq1_ladder", written)


def fig_rq1_delta(pooled: dict, outdir: Path, written: list[Path]) -> None:
    models = pooled["models"]
    baselines = pooled["config"]["baselines"]
    sizes = [size_of_label(m["label"]) for m in models]
    labels = [m["label"] for m in models]
    inter = {r["baseline"]: r for r in pooled.get("interaction", {}).get("per_baseline", [])}

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True, sharey=True)
    for ax, b in zip(axes.flat, baselines):
        rows_by_size = []
        for m in models:
            rows = m["comparisons"]["policies"]["complete_case"]
            rows_by_size.append(next((r for r in rows if r.get("baseline") == b), {}))
        ys = [_nan(r.get("mean_diff")) for r in rows_by_size]
        los = [_nan((r.get("ci95") or [None, None])[0]) for r in rows_by_size]
        his = [_nan((r.get("ci95") or [None, None])[1]) for r in rows_by_size]
        sig = [_nan(r.get("p_holm")) < 0.05 for r in rows_by_size]

        ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
        ax.fill_between(sizes, los, his, color="#0072B2", alpha=0.15, linewidth=0, zorder=2)
        ax.plot(sizes, ys, color="#0072B2", linewidth=1.4, zorder=3)
        for x, y, s, r, m in zip(sizes, ys, sig, rows_by_size, models):
            ax.plot([x], [y], marker="D", markersize=4.5,
                    markerfacecolor="#0072B2" if s else "white",
                    markeredgecolor="#0072B2", markeredgewidth=1.0, zorder=4)
            # flag points backed by few complete-case pairs (heavy agent failure)
            n = r.get("n")
            if n is not None and n < 0.5 * (m.get("n_scenarios_union") or n):
                ax.annotate(f"n={n}", xy=(x, y), xytext=(0, -11),
                            textcoords="offset points", ha="center",
                            fontsize=6, color="#666666")
        row = inter.get(b)
        if row:
            ax.set_title(
                f"vs {AGENT_DISPLAY.get(b, b)}\n"
                f"slope/step {row['trend_ols_slope_per_size_step']:+.2f} "
                f"[{row['trend_slope_ci95'][0]:+.2f}, {row['trend_slope_ci95'][1]:+.2f}]",
                fontsize=7.5)
        else:
            ax.set_title(f"vs {AGENT_DISPLAY.get(b, b)}", fontsize=7.5)
        sized_axis(ax, sizes, labels)
        ax.grid(axis="x", visible=False)
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ Total (proposed − baseline)")
    fig.suptitle(
        "Paired per-scenario delta with bootstrap 95% CI "
        "(filled marker: Holm-adjusted p < 0.05)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, outdir, "fig_rq1_delta", written)


def fig_rq1_components(pooled: dict, outdir: Path, written: list[Path]) -> None:
    models = pooled["models"]
    agents = agents_in(pooled)
    sizes = [size_of_label(m["label"]) for m in models]
    labels = [m["label"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    for ax, (field, title) in zip(axes, [("mean_addie", "ADDIE (rubric, /100)"),
                                         ("mean_traj", "Trajectory (/100)")]):
        for a in agents:
            ys = [_nan(m["agents"].get(a, {}).get(field)) for m in models]
            ax.plot(sizes, ys, **line_kw(a))
        sized_axis(ax, sizes, labels)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Component mean")
    agent_legend(fig, agents, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, outdir, "fig_rq1_components", written)


def fig_rq2_ladder(pooled: dict, outdir: Path, written: list[Path]) -> bool:
    align = pooled.get("alignment") or {}
    if not align.get("models"):
        return False
    models = [m for m in pooled["models"] if m["label"] in align["models"]]
    agents = agents_in(pooled)
    sizes = [size_of_label(m["label"]) for m in models]
    labels = [m["label"] for m in models]

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for a in agents:
        ys, errs = [], []
        for m in models:
            cell = align["models"][m["label"]].get(a, {}).get("composite", {})
            ys.append(_nan(cell.get("mean")))
            n = cell.get("n_scenarios") or 0
            sd = _nan(cell.get("sd_across_scenarios"))
            errs.append(sd / math.sqrt(n) if n and not math.isnan(sd) else 0.0)
        ax.errorbar(sizes, ys, yerr=errs, capsize=2, elinewidth=0.8, **line_kw(a))
    sized_axis(ax, sizes, labels)
    ax.set_ylabel("Alignment composite (0–1)")
    ax.grid(axis="x", visible=False)
    ax.text(0.02, 0.02, "error bars: ±SEM across scenarios",
            transform=ax.transAxes, fontsize=6.5, color="#666666")
    agent_legend(fig, agents, bbox_to_anchor=(0.5, -0.02))
    save(fig, outdir, "fig_rq2_ladder", written)
    return True


def fig_rq2_components(pooled: dict, outdir: Path, written: list[Path]) -> bool:
    align = pooled.get("alignment") or {}
    if not align.get("models"):
        return False
    models = [m for m in pooled["models"] if m["label"] in align["models"]]
    agents = agents_in(pooled)
    comp_keys = [k for k, _ in RQ2_COMPONENTS]
    comp_names = [n for _, n in RQ2_COMPONENTS]

    fig, axes = plt.subplots(1, len(models), figsize=(7.0, 2.0 + 0.28 * len(agents)),
                             sharey=True)
    axes = [axes] if len(models) == 1 else list(axes)
    cmap = plt.get_cmap("Blues")
    last_im = None
    for ax, m in zip(axes, models):
        grid = [[_nan(align["models"][m["label"]].get(a, {}).get(k, {}).get("mean"))
                 for k in comp_keys] for a in agents]
        last_im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if math.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else "#333333")
        ax.set_title(f"Qwen3.5-{size_display(m['label'])}")
        ax.set_xticks(range(len(comp_names)))
        ax.set_xticklabels(comp_names, rotation=40, ha="right", fontsize=6.5)
        ax.grid(visible=False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(agents)))
    axes[0].set_yticklabels([AGENT_DISPLAY.get(a, a) for a in agents], fontsize=7)
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("Component mean (0–1)", fontsize=7)
    save(fig, outdir, "fig_rq2_components", written)
    return True


def collect_scenario_pairs(pattern: str, agents: list[str]) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """label -> agent -> [(total_mean, composite_mean)] per scenario (mean of runs)."""
    pool = load_pool_module()
    out: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for label, dirs in pool.group_by_glob(pattern).items():
        acc: dict[str, dict[str, dict[str, list[float]]]] = {}
        for d in dirs:
            for sid, payload in pool.load_run(d).items():
                doc = payload.get("alignment")
                align = pool.extract_alignment_metrics(doc, agents) if doc else {}
                for a in agents:
                    r = payload["rankings"].get(a) or {}
                    t, c = r.get("total_score"), align.get(a, {}).get("composite")
                    if isinstance(t, (int, float)) and isinstance(c, (int, float)):
                        slot = acc.setdefault(a, {}).setdefault(sid, {"t": [], "c": []})
                        slot["t"].append(float(t))
                        slot["c"].append(float(c))
        out[label] = {
            a: [(sum(v["t"]) / len(v["t"]), sum(v["c"]) / len(v["c"]))
                for v in by_sid.values()]
            for a, by_sid in acc.items()
        }
    return out


def fig_rq2_vs_rq1_scatter(pairs: dict, agents: list[str], outdir: Path,
                           written: list[Path]) -> bool:
    labels = sorted(pairs, key=size_of_label)
    labels = [lb for lb in labels if any(pairs[lb].get(a) for a in agents)]
    if not labels:
        return False
    fig, axes = plt.subplots(1, len(labels), figsize=(7.0, 2.4),
                             sharex=True, sharey=True)
    axes = [axes] if len(labels) == 1 else list(axes)
    for ax, lb in zip(axes, labels):
        all_t, all_c = [], []
        for a in agents:
            pts = pairs[lb].get(a, [])
            if not pts:
                continue
            ts, cs = zip(*pts)
            all_t += list(ts)
            all_c += list(cs)
            ax.scatter(ts, cs, s=7, alpha=0.45, linewidths=0,
                       color=AGENT_COLOR.get(a, "#666666"),
                       marker=AGENT_MARKER.get(a, "o"),
                       zorder=5 if a == PROPOSED else 3)
        rho = spearman_rho(all_t, all_c)
        ax.set_title(f"Qwen3.5-{size_display(lb)}", fontsize=8)
        ax.text(0.04, 0.96, f"$\\rho_s$ = {rho:.2f}\nn = {len(all_t)}",
                transform=ax.transAxes, va="top", fontsize=7, color="#333333")
        ax.set_xlabel("Total (judge)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Alignment composite")
    agent_legend(fig, agents, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, outdir, "fig_rq2_vs_rq1_scatter", written)
    return True


def fig_failure_rate(pooled: dict, outdir: Path, written: list[Path]) -> None:
    models = pooled["models"]
    agents = agents_in(pooled)
    n_agents = len(agents)
    width = 0.8 / n_agents

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for ai, a in enumerate(agents):
        xs, ys, err_lo, err_hi = [], [], [], []
        for mi, m in enumerate(models):
            denom = m.get("n_scenarios_union") or 1
            rates = [100.0 * info["missing_by_agent"].get(a, 0) / denom
                     for info in m["per_run"]]
            mean = sum(rates) / len(rates) if rates else 0.0
            xs.append(mi + (ai - (n_agents - 1) / 2) * width)
            ys.append(mean)
            err_lo.append(mean - min(rates, default=0.0))
            err_hi.append(max(rates, default=0.0) - mean)
        bars = ax.bar(xs, ys, width=width * 0.92, color=AGENT_COLOR.get(a, "#666666"),
                      yerr=[err_lo, err_hi], error_kw={"elinewidth": 0.7, "capsize": 1.5},
                      zorder=3)
        for rect, y, hi in zip(bars, ys, err_hi):
            if y > 0:
                ax.annotate(f"{y:.1f}", xy=(rect.get_x() + rect.get_width() / 2, y + hi),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=6, color="#333333")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([f"Qwen3.5-{size_display(m['label'])}" for m in models])
    ax.set_ylabel("Failed scenarios (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.set_title("Scenarios with no scorable output "
                 "(mean of runs; whiskers: min–max across runs)", fontsize=8)
    agent_legend(fig, agents, bbox_to_anchor=(0.5, -0.04))
    save(fig, outdir, "fig_failure_rate", written)


def fig_cost_quality(pooled: dict, outdir: Path, written: list[Path]) -> bool:
    tok = pooled.get("token_usage") or {}
    if not tok.get("models"):
        return False
    models = [m for m in pooled["models"] if m["label"] in tok["models"]]
    agents = agents_in(pooled)
    size_ms = [3.5, 5, 6.5, 8.5]  # marker size grows with model size

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    specs = [("total_tokens", "Mean total tokens per scenario", True),
             ("execution_time_seconds", "Mean execution time per scenario (s)", False)]
    for ax, (field, xlabel, logx) in zip(axes, specs):
        for a in agents:
            xs, ys = [], []
            for m in models:
                xs.append(_nan(tok["models"][m["label"]].get(a, {})
                               .get(field, {}).get("mean")))
                ys.append(_nan(m["agents"].get(a, {}).get("mean_total")))
            kw = line_kw(a)
            ax.plot(xs, ys, alpha=0.9, color=kw["color"], marker="",
                    linewidth=0.9 if a != PROPOSED else 1.6, zorder=kw["zorder"])
            for x, y, ms in zip(xs, ys, size_ms):
                ax.plot([x], [y], marker=kw["marker"], markersize=ms,
                        color=kw["color"], markeredgecolor="white",
                        markeredgewidth=0.6, zorder=kw["zorder"] + 1)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Total composite")
    fig.suptitle("Quality vs cost — marker size grows with model size "
                 f"({', '.join(size_display(m['label']) for m in models)})", fontsize=8)
    agent_legend(fig, agents, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, outdir, "fig_cost_quality", written)
    return True


# ── demo data ────────────────────────────────────────────────────────────────

def demo_pooled() -> dict:
    """Synthetic pooled structure exercising every figure (clearly fake)."""
    labels = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B"]
    base_by_size = [40.0, 55.0, 65.0, 75.0]
    offsets = {"baseline": 0.0, "eduplanner": -8.0, "addie-agent": 1.0,
               "rpisd-agent": -1.0, "dick-carey-agent": 2.0, "react-isd": 1.5,
               "alignmentgraph-isd": 6.0}
    rng = random.Random(42)
    models, align_models, tok_models = [], {}, {}
    for si, (label, base) in enumerate(zip(labels, base_by_size)):
        agents_summary, comp_rows = {}, []
        for a, off in offsets.items():
            total = base + off
            agents_summary[a] = {
                "n_scenarios_complete": 88, "n_scenarios_any_missing": 2,
                "mean_total": total, "sd_total_across_scenarios": 5.0,
                "mean_addie": total - 2.0, "mean_traj": total + 4.0,
                "run_means_total": [total - 0.4, total, total + 0.4],
                "run_to_run_sd": 1.2 + (3 - si) * 0.5,
            }
            if a != PROPOSED:
                d = offsets[PROPOSED] - off + (3 - si) * 0.8
                comp_rows.append({"baseline": a, "n": 88, "mean_diff": d,
                                  "ci95": [d - 1.2, d + 1.2], "p_raw": 1e-9,
                                  "rank_biserial_r": 0.8,
                                  "p_holm": 6e-9 if si else 0.2})
        models.append({
            "label": label, "n_runs": 3, "n_scenarios_union": 90,
            "per_run": [{"dir": "TBD-DEMO", "n_scenarios_scored": 90,
                         "missing_by_agent": {
                             a: rng.choice([0, 0, 0, 1, 2, 5]) if si < 2 else 0
                             for a in offsets}}
                        for _ in range(3)],
            "agents": agents_summary,
            "comparisons": {"policies": {"complete_case": comp_rows}},
        })
        align_models[label] = {
            a: {"composite": {"n_scenarios": 90,
                              "mean": min(0.95, 0.35 + si * 0.08 + off / 60
                                          + (0.15 if a == PROPOSED else 0)),
                              "sd_across_scenarios": 0.12},
                **{k: {"n_scenarios": 90,
                       "mean": min(0.95, 0.3 + si * 0.07 + off / 70 + ki * 0.05
                                   + (0.12 if a == PROPOSED else 0)),
                       "sd_across_scenarios": 0.1}
                   for ki, (k, _) in enumerate(RQ2_COMPONENTS)}}
            for a, off in offsets.items()
        }
        tok_models[label] = {
            a: {"total_tokens": {"mean": 20000 + i * 15000 + si * 5000},
                "execution_time_seconds": {"mean": 60 + i * 40 + si * 20}}
            for i, a in enumerate(offsets)
        }
    inter = {"per_baseline": [
        {"baseline": b, "trend_ols_slope_per_size_step": -0.8,
         "trend_slope_ci95": [-1.3, -0.3]} for b in offsets if b != PROPOSED]}
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"proposed": PROPOSED,
                   "baselines": [a for a in AGENT_ORDER if a != PROPOSED],
                   "size_order": labels},
        "models": models, "interaction": inter,
        "alignment": {"source": "TBD-DEMO", "models": align_models},
        "token_usage": {"source": "TBD-DEMO", "models": tok_models},
        "_demo": True,
    }


def demo_pairs(pooled: dict) -> dict:
    """Synthetic per-scenario (total, composite) pairs for the scatter figure."""
    rng = random.Random(42)
    pairs: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for m in pooled["models"]:
        by_agent = {}
        for a, s in m["agents"].items():
            comp = pooled["alignment"]["models"][m["label"]][a]["composite"]["mean"]
            pts = []
            for _ in range(90):
                t = rng.gauss(s["mean_total"], 8.0)
                c = max(0.0, min(1.0, comp + (t - s["mean_total"]) / 120
                                 + rng.gauss(0, 0.06)))
                pts.append((t, c))
            by_agent[a] = pts
        pairs[m["label"]] = by_agent
    return pairs


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures (RQ1/RQ2) from pooled_ladder.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pooled", type=Path, default=DEFAULT_POOLED,
                        help="pooled_ladder.json from scripts/7_pool_ladder_runs.py")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="output directory for .pdf/.png figures")
    parser.add_argument("--runs-glob", default=None, metavar="PATTERN",
                        help="glob of raw run dirs (same convention as "
                             "7_pool_ladder_runs.py --auto-glob); enables the "
                             "per-scenario RQ2-vs-RQ1 scatter figure")
    parser.add_argument("--demo", action="store_true",
                        help="render every figure from synthetic data "
                             "(layout test, no pooled JSON needed)")
    args = parser.parse_args()

    if args.demo:
        pooled = demo_pooled()
        print("DEMO MODE: rendering synthetic placeholder data.")
    else:
        if not args.pooled.exists():
            print(f"Error: pooled JSON not found: {args.pooled} (use --demo for a layout test)",
                  file=sys.stderr)
            sys.exit(1)
        pooled = json.loads(args.pooled.read_text(encoding="utf-8"))

    args.outdir.mkdir(parents=True, exist_ok=True)
    agents = agents_in(pooled)
    written: list[Path] = []

    fig_rq1_ladder(pooled, args.outdir, written)
    fig_rq1_delta(pooled, args.outdir, written)
    fig_rq1_components(pooled, args.outdir, written)
    if not fig_rq2_ladder(pooled, args.outdir, written):
        print("! no alignment data in pooled JSON — skipped fig_rq2_ladder")
    if not fig_rq2_components(pooled, args.outdir, written):
        print("! no alignment data in pooled JSON — skipped fig_rq2_components")
    if args.demo:
        fig_rq2_vs_rq1_scatter(demo_pairs(pooled), agents, args.outdir, written)
    elif args.runs_glob:
        pairs = collect_scenario_pairs(args.runs_glob, agents)
        if not fig_rq2_vs_rq1_scatter(pairs, agents, args.outdir, written):
            print("! no per-scenario alignment data under --runs-glob — skipped scatter")
    else:
        print("- fig_rq2_vs_rq1_scatter skipped (pass --runs-glob to enable)")
    fig_failure_rate(pooled, args.outdir, written)
    if not fig_cost_quality(pooled, args.outdir, written):
        print("! no token_usage data in pooled JSON — skipped fig_cost_quality")

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
