#!/usr/bin/env python3
"""Generate paper figures (RQ1/RQ2 statistical charts) from pooled_ladder.json.

Companion of scripts/alignmentgraph-isd-bench/09_gen_paper_tables.py — same stage of the pipeline, same
input, but emits figures instead of tables. Writes every figure as both .pdf
(vector, for LaTeX \\includegraphics) and .png (dpi 200, for quick preview)
into results/generated/figures/ (benchmark-local; sync into the thesis paper
tree with docs/scripts/sync_generated.sh from the thesis repo root):

  fig_rq1_ladder          mean ADDIE vs model size, one line per agent,
                          error bar = run-to-run SD *of ADDIE* (headline RQ1
                          figure); thin complete-case support marked
  fig_rq1_delta           per-baseline paired delta (proposed - baseline) vs
                          size with bootstrap 95% CI band, Holm-significance
                          markers, and the OLS delta trend slope
  fig_rq1_components      ADDIE vs Trajectory component means vs size
  fig_rq2_ladder          objective->assessment similarity vs model size per agent
                          (RQ2 headline: does the gap widen as models shrink?)
  fig_rq2_components      heatmap agents x the 7 alignment panel signals per size
  fig_rq2_vs_rq1_scatter  per-scenario objective->assessment similarity vs ADDIE (judge),
                          Spearman rho per size  [needs --runs-glob]
  fig_failure_rate        % scenarios with no scorable output per agent/size
                          (mean of runs, min-max whiskers)
  fig_cost_quality        mean ADDIE vs mean total tokens (execution time
                          omitted — it reflects serving load, not the method)

All figure text is English (Vietnamese captions live in the LaTeX source, as
with the generated tables). Numbers come exclusively from pooled_ladder.json
and the raw run dirs — nothing hand-typed.

Usage:
  python scripts/alignmentgraph-isd-bench/10_gen_paper_figures.py                       # pooled JSON only
  python scripts/alignmentgraph-isd-bench/10_gen_paper_figures.py \
      --runs-glob 'results/test_90_benchmark_*_r*'            # + scatter figure
  python scripts/alignmentgraph-isd-bench/10_gen_paper_figures.py --demo                # synthetic layout test
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
    from matplotlib.patches import Patch
except ImportError:  # pragma: no cover
    print(
        "Error: matplotlib is required for this script.\n"
        "Install the benchmark deps from the repo root:  pip install -e .",
        file=sys.stderr,
    )
    sys.exit(1)

BENCH_ROOT = Path(__file__).resolve().parents[2]   # isd-agent-benchmark/
DEFAULT_POOLED = BENCH_ROOT / "results" / "pooled_ladder.json"
DEFAULT_OUTDIR = BENCH_ROOT / "results" / "generated" / "figures"

# Same identities as scripts/alignmentgraph-isd-bench/09_gen_paper_tables.py (kept in sync by hand).
AGENT_DISPLAY = {
    "baseline": "Baseline",
    "eduplanner": "EduPlanner",
    "addie-agent": "ADDIE-Agent",
    "rpisd-agent": "RPISD-Agent",
    "dick-carey-agent": "Dick-Carey-Agent",
    "react-isd": "ReAct-ISD",
    "alignmentgraph-isd": "AlignmentGraph-ISD",
}
AGENT_ORDER = [
    "baseline", "eduplanner", "addie-agent", "rpisd-agent",
    "dick-carey-agent", "react-isd", "alignmentgraph-isd",
]
PROPOSED = "alignmentgraph-isd"

#: RQ1 lead signal: ADDIE, not the benchmark composite (see 07's RQ1_SIGNALS).
RQ1_LEAD = "addie_median"
RQ1_LEAD_FIELD = "mean_addie"

#: Run-to-run SD field matching each plotted mean. The SD must belong to the
#: signal on the y axis: 07 emits one per signal, and the bare
#: `run_to_run_sd` is the *total_score* value (kept only for backward
#: compatibility) — never use it under an ADDIE or Trajectory mean.
RQ1_SD_FIELD = {
    "mean_addie": "run_to_run_sd_addie",
    "mean_traj": "run_to_run_sd_traj",
    "mean_total": "run_to_run_sd_total",
}

# Rainbow-ordered (spectral) 7-hue categorical palette, fixed per agent
# (never cycled). Spectral ordering means the perceptually-closest hues are
# validated as adjacent pairs; the set passes the lightness band, chroma
# floor and normal-vision floor, with one adjacent pair (olive/orange) in
# the CVD 6-8 band — covered by a secondary encoding (olive is dashed, see
# AGENT_LINESTYLE). Line charts draw baselines as plain lines (no markers);
# only the proposed agent carries a marker + heavier width, so it can never
# be confused with a comparison line.
AGENT_COLOR = {
    "baseline": "#C42D00",            # red
    "eduplanner": "#EF9E0B",          # orange
    "addie-agent": "#8F9900",         # olive   (dashed — CVD pair w/ orange)
    "rpisd-agent": "#007A3D",         # green
    "dick-carey-agent": "#00B0D5",    # cyan
    "alignmentgraph-isd": "#0048B0",  # blue    (proposed — emphasized)
    "react-isd": "#E8438F",           # magenta
}
AGENT_LINESTYLE = {
    "addie-agent": (0, (4, 1.8)),     # secondary encoding for the olive/orange CVD pair
}


def point_marker(agent: str) -> str:
    """Marker for point/scatter figures: uniform circles, diamond = proposed.

    Per-agent marker shapes were dropped everywhere — identity is carried by
    hue (+ the olive dash on lines); shape only singles out the proposed."""
    return "D" if agent == PROPOSED else "o"

# The whole panel, both instrument families. Every signal is plotted because
# the panel's defence against selective reporting is that nothing is left out;
# tab_rq2_panel.tex carries the same seven with exact numbers.
RQ2_COMPONENTS = [
    ("objective_assessment_similarity", "Obj→Asm"),
    ("objective_activity_similarity", "Obj→Act"),
    ("objective_evaluation_similarity", "Obj→Evl"),
    ("assessment_objective_similarity", "Asm→Obj†"),
    ("objective_cognitive_congruence", "Cognitive"),
    ("porter_mean", "Porter"),
    ("webb_bloom_consistency", "Webb"),
]

#: Number of leading RQ2_COMPONENTS entries belonging to instrument family A
#: (textual correspondence). The rest are family B (cognitive demand). A
#: separator is drawn at this boundary so the single colourbar cannot be read
#: as one construct across both families.
RQ2_FAMILY_A_N = 4

#: In-figure disclosure for the panel heatmaps. Figures get read detached from
#: their LaTeX caption, so every caveat a reader needs to not over-read a cell
#: has to live inside the image: the two instrument families, the
#: non-directionality of Asm->Obj, the flat-panel status, and why some cells
#: carry an n.
RQ2_PANEL_FOOTNOTE = (
    "Flat panel of 7 signals, no primary endpoint. Family A (textual correspondence, "
    "rectified cosine max(0, cos) on [0,1]): Obj→Asm, Obj→Act, Obj→Evl, Asm→Obj.  "
    "Family B (cognitive demand, Bloom-derived, no similarity): Cognitive congruence, "
    "Porter (mean of 3 pairwise indices), Webb.\n"
    "† Asm→Obj is non-directional: a high value can mean no orphan assessment items "
    "OR items that merely restate the objectives.\n"
    "Cell values omit the leading zero. A red n marks a cell where fewer than half the "
    "scenarios produced a scorable output — a survivorship mean, not comparable with "
    "full-coverage cells."
)

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
    """Series style: proposed = heavy line + diamond marker; baselines =
    plain lines, no markers (identity carried by hue + the one dash)."""
    proposed = agent == PROPOSED
    return {
        "color": AGENT_COLOR.get(agent, "#666666"),
        "marker": "D" if proposed else "",
        "linestyle": AGENT_LINESTYLE.get(agent, "-"),
        "linewidth": 2.6 if proposed else 1.4,
        "markersize": 5.5,
        "zorder": 5 if proposed else 3,
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
    }


def agent_legend(fig, agents: list[str], **kw) -> None:
    handles = [
        Line2D([], [], label=AGENT_DISPLAY.get(a, a), **{
            k: v for k, v in line_kw(a).items() if k != "zorder"})
        for a in agents
    ]
    kw.setdefault("loc", "upper center")
    kw.setdefault("bbox_to_anchor", (0.5, 0.0))
    kw.setdefault("ncol", 4)
    kw.setdefault("frameon", False)
    fig.legend(handles=handles, **kw)


def bar_legend(fig, agents: list[str], labels: dict[str, str] | None = None, **kw) -> None:
    """Legend for BAR charts: filled patches, matching the drawn encoding.

    agent_legend() builds Line2D handles (dashes, markers) that appear nowhere
    in a bar chart, so bars get their own key."""
    handles = [
        Patch(facecolor=AGENT_COLOR.get(a, "#666666"),
              label=(labels or {}).get(a, AGENT_DISPLAY.get(a, a)))
        for a in agents
    ]
    kw.setdefault("loc", "upper center")
    kw.setdefault("bbox_to_anchor", (0.5, 0.0))
    kw.setdefault("ncol", 4)
    kw.setdefault("frameon", False)
    fig.legend(handles=handles, **kw)


def thin_coverage_n(summary: dict, model: dict) -> int | None:
    """Complete-case n when it covers < half the scenario union, else None.

    A mean over 5 of 90 scenarios is a survivorship mean; drawn identically to
    a 90-scenario mean it reads as comparable, so callers mark it."""
    n = summary.get("n_scenarios_complete")
    union = model.get("n_scenarios_union") or 0
    if isinstance(n, (int, float)) and not isinstance(n, bool) and union and n < 0.5 * union:
        return int(n)
    return None


def edge_annot_kw(x: float, sizes: list[float], dy: int = -11) -> dict:
    """Offset/alignment for an annotation, kept clear of the axis spines.

    At the leftmost x a centred label straddles the y-axis spine and collides
    with the tick labels, so edge points get pushed inward."""
    if x == sizes[0]:
        return {"xytext": (6, dy), "ha": "left"}
    if x == sizes[-1]:
        return {"xytext": (-6, dy), "ha": "right"}
    return {"xytext": (0, dy), "ha": "center"}


def median(xs: list[float]) -> float:
    ys = sorted(v for v in xs if not math.isnan(v))
    if not ys:
        return float("nan")
    mid = len(ys) // 2
    return ys[mid] if len(ys) % 2 else 0.5 * (ys[mid - 1] + ys[mid])


def sized_axis(ax, sizes: list[float], labels: list[str]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([size_display(lb) for lb in labels])
    ax.minorticks_off()
    ax.set_xlabel("Model size (Qwen3.5)")


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rho — delegates to 07_pool_ladder_runs.py, the single home of
    the statistics (which also uses it for the cross-encoder agreement)."""
    return load_pool_module().spearman_rho(xs, ys)


def load_pool_module():
    """Import 07_pool_ladder_runs.py (digit-prefixed name) for its loaders."""
    path = Path(__file__).resolve().with_name("07_pool_ladder_runs.py")
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

    sd_field = RQ1_SD_FIELD[RQ1_LEAD_FIELD]
    union = models[0].get("n_scenarios_union") if models else None

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    thin_seen = False
    for a in agents:
        summaries = [m["agents"].get(a, {}) for m in models]
        ys = [_nan(s.get(RQ1_LEAD_FIELD)) for s in summaries]
        # SD must belong to the plotted signal; if 07 did not emit the
        # per-signal field, draw no error bars rather than a foreign SD.
        errs = [_nan(s.get(sd_field)) for s in summaries]
        have_sd = any(not math.isnan(e) for e in errs)
        errs = [0.0 if math.isnan(e) else e for e in errs]
        ax.errorbar(sizes, ys, yerr=errs if have_sd else None,
                    capsize=2, elinewidth=0.8, **line_kw(a))
        # thin complete-case support: open marker + n, so a survivorship mean
        # cannot be read as comparable with a full-coverage point
        for x, y, s, m in zip(sizes, ys, summaries, models):
            n = thin_coverage_n(s, m)
            if n is None or math.isnan(y):
                continue
            thin_seen = True
            ax.plot([x], [y], marker="o", markersize=6.0, markerfacecolor="white",
                    markeredgecolor=AGENT_COLOR.get(a, "#666666"),
                    markeredgewidth=1.3, linestyle="", zorder=7)
            ax.annotate(f"n={n}", xy=(x, y), textcoords="offset points",
                        fontsize=6, color=AGENT_COLOR.get(a, "#666666"),
                        **edge_annot_kw(x, sizes, dy=-12))
    # direct label on the proposed agent (secondary encoding beside the legend)
    prop_y = _nan(models[-1]["agents"].get(PROPOSED, {}).get(RQ1_LEAD_FIELD))
    if not math.isnan(prop_y):
        ax.annotate(AGENT_DISPLAY[PROPOSED], xy=(sizes[-1], prop_y),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=AGENT_COLOR[PROPOSED],
                    fontweight="bold")
        ax.set_xlim(right=sizes[-1] * 2.4)
    sized_axis(ax, sizes, labels)
    ax.set_ylabel("ADDIE rubric (/100)")
    ax.grid(axis="x", visible=False)
    n_runs = models[0].get("n_runs") if models else None
    note = f"error bars: run-to-run SD of ADDIE ({n_runs} runs)"
    if thin_seen:
        note += ("\nopen marker + n: complete-case mean over\n"
                 f"fewer than half of {union} scenarios (agent failures)")
    ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            multialignment="right", fontsize=6.5, color="#666666", zorder=8)
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
            rows = m["comparisons"]["by_signal"][RQ1_LEAD]["policies"]["complete_case"]
            rows_by_size.append(next((r for r in rows if r.get("baseline") == b), {}))
        ys = [_nan(r.get("mean_diff")) for r in rows_by_size]
        los = [_nan((r.get("ci95") or [None, None])[0]) for r in rows_by_size]
        his = [_nan((r.get("ci95") or [None, None])[1]) for r in rows_by_size]
        sig = [_nan(r.get("p_holm")) < 0.05 for r in rows_by_size]

        ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
        ax.fill_between(sizes, los, his, color=AGENT_COLOR[PROPOSED], alpha=0.15, linewidth=0, zorder=2)
        ax.plot(sizes, ys, color=AGENT_COLOR[PROPOSED], linewidth=1.4, zorder=3)
        for x, y, s, r, m in zip(sizes, ys, sig, rows_by_size, models):
            ax.plot([x], [y], marker="D", markersize=4.5,
                    markerfacecolor=AGENT_COLOR[PROPOSED] if s else "white",
                    markeredgecolor=AGENT_COLOR[PROPOSED], markeredgewidth=1.0, zorder=4)
            # flag points backed by few complete-case pairs (heavy agent failure)
            n = r.get("n")
            if n is not None and n < 0.5 * (m.get("n_scenarios_union") or n):
                ax.annotate(f"n={n}", xy=(x, y), textcoords="offset points",
                            fontsize=6, color="#666666",
                            **edge_annot_kw(x, sizes))
        row = inter.get(b)
        if row:
            # The slope rests on the scenarios common to every size; with a
            # handful of them the bootstrap CI collapses to zero width and
            # would read as infinite precision, so print n and drop the
            # interval when the support is too thin to bound anything.
            n_common = row.get("trend_n_common_scenarios")
            ci = row.get("trend_slope_ci95")
            slope = f"slope/step {row['trend_ols_slope_per_size_step']:+.2f}"
            if isinstance(n_common, int) and n_common < 10:
                stat = f"{slope} (n={n_common} common scen.)"
            elif ci:
                stat = f"{slope} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
                stat += f", n={n_common}" if isinstance(n_common, int) else ""
            else:
                stat = slope
            ax.set_title(f"vs {AGENT_DISPLAY.get(b, b)}\n{stat}", fontsize=7.0)
        else:
            ax.set_title(f"vs {AGENT_DISPLAY.get(b, b)}", fontsize=7.5)
        sized_axis(ax, sizes, labels)
        ax.grid(axis="x", visible=False)
    for ax in axes[0]:
        ax.set_xlabel("")
    # one figure-level y label: the 30-char label is longer than a single row's
    # axis is tall, so per-axes labels butt into each other
    fig.supylabel("Δ ADDIE (proposed − baseline)", fontsize=8)
    fig.suptitle(
        "Paired per-scenario delta with bootstrap 95% CI "
        "(filled marker: Holm-adjusted p < 0.05)", fontsize=8.5)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    save(fig, outdir, "fig_rq1_delta", written)


def fig_rq1_components(pooled: dict, outdir: Path, written: list[Path]) -> None:
    models = pooled["models"]
    agents = agents_in(pooled)
    sizes = [size_of_label(m["label"]) for m in models]
    labels = [m["label"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9), sharex=True)
    for ax, (field, title) in zip(axes, [("mean_addie", "ADDIE (rubric, /100)"),
                                         ("mean_traj", "Trajectory (/100)"),
                                         ("mean_total", "Total (0.7 ADDIE + 0.3 Traj, /100)")]):
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
            cell = align["models"][m["label"]].get(a, {}).get("objective_assessment_similarity", {})
            ys.append(_nan(cell.get("mean")))
            n = cell.get("n_scenarios") or 0
            sd = _nan(cell.get("sd_across_scenarios"))
            errs.append(sd / math.sqrt(n) if n and not math.isnan(sd) else 0.0)
        ax.errorbar(sizes, ys, yerr=errs, capsize=2, elinewidth=0.8, **line_kw(a))
    sized_axis(ax, sizes, labels)
    ax.set_ylabel("Objective$\\to$assessment similarity (0–1)")
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

    # wide enough that a 3-glyph value fits inside a column without touching
    # its neighbour (the 7 signals are narrow by construction)
    fig, axes = plt.subplots(1, len(models),
                             figsize=(3.05 * len(models) + 1.6, 2.4 + 0.34 * len(agents)),
                             sharey=True)
    axes = [axes] if len(models) == 1 else list(axes)
    cmap = plt.get_cmap("Blues")
    last_im = None
    for ax, m in zip(axes, models):
        cells = [[align["models"][m["label"]].get(a, {}).get(k, {}) or {}
                  for k in comp_keys] for a in agents]
        grid = [[_nan(c.get("mean")) for c in row] for row in cells]
        ns = [n for row in cells for c in row
              if isinstance(n := c.get("n_scenarios"), int)]
        n_max = max(ns) if ns else 0
        last_im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if math.isnan(v):
                    continue
                txt = f"{v:.2f}"
                dark = v > 0.55
                ax.text(j, i, txt[1:] if 0 <= v < 1 else txt,
                        ha="center", va="center", fontsize=6.5,
                        color="white" if dark else "#333333")
                n = cells[i][j].get("n_scenarios")
                if isinstance(n, int) and n_max and n < 0.5 * n_max:
                    ax.text(j, i + 0.30, f"n={n}", ha="center", va="center",
                            fontsize=5.5,
                            color="#FFC4C4" if dark else "#8B0000")
        # separator between the two instrument families (A | B)
        ax.axvline(RQ2_FAMILY_A_N - 0.5, color="white", linewidth=2.0, zorder=4)
        ax.set_title(f"Qwen3.5-{size_display(m['label'])}")
        ax.set_xticks(range(len(comp_names)))
        ax.set_xticklabels(comp_names, rotation=35, ha="right", fontsize=7.5)
        ax.grid(visible=False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(agents)))
    axes[0].set_yticklabels([AGENT_DISPLAY.get(a, a) for a in agents], fontsize=8)
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("Panel signal mean (0–1)", fontsize=7)
    fig.text(0.5, -0.05, RQ2_PANEL_FOOTNOTE, ha="center", va="top",
             fontsize=6.5, color="#555555", linespacing=1.5)
    save(fig, outdir, "fig_rq2_components", written)
    return True


#: Baseline the RQ2 forest plot is drawn against: the strongest overall rival,
#: matching the comparison the results narrative leads with.
RQ2_FOREST_BASELINE = "react-isd"


def fig_rq2_forest(pooled: dict, outdir: Path, written: list[Path]) -> bool:
    """Forest plot of paired panel-signal deltas (proposed - RQ2_FOREST_BASELINE).

    Primary failure=0 layer, one facet per size, 7 signals per facet in panel
    order with the family A|B separator; filled marker = within-signal
    Holm-adjusted p < 0.05. This is the significance view the heatmap
    deliberately does not carry.
    """
    align = pooled.get("alignment") or {}
    stats = (align.get("stats") or {}).get("per_model") or []
    if not stats:
        return False
    comp_keys = [k for k, _ in RQ2_COMPONENTS]
    comp_names = [n for _, n in RQ2_COMPONENTS]
    ys = list(range(len(comp_keys)))[::-1]  # top-down in panel order

    fig, axes = plt.subplots(1, len(stats), figsize=(1.85 * len(stats) + 1.2, 2.9),
                             sharey=True, sharex=True)
    axes = [axes] if len(stats) == 1 else list(axes)
    color = AGENT_COLOR[PROPOSED]
    for ax, m in zip(axes, stats):
        panel = m.get("panel") or {}
        for y, key in zip(ys, comp_keys):
            rows = (panel.get(key) or {}).get("comparisons_failure_zero") or []
            r = next((r for r in rows if r.get("baseline") == RQ2_FOREST_BASELINE), {})
            d = _nan(r.get("mean_diff"))
            lo, hi = (r.get("ci95") or [float("nan"), float("nan")])
            if math.isnan(d):
                continue
            sig = _nan(r.get("p_holm")) < 0.05
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.1, zorder=2)
            ax.plot([d], [y], marker="D", markersize=4.0,
                    markerfacecolor=color if sig else "white",
                    markeredgecolor=color, markeredgewidth=1.0, zorder=3)
        ax.axvline(0, color="#999999", linewidth=0.8, zorder=1)
        # family A | B separator, same convention as the heatmap
        ax.axhline(len(comp_keys) - RQ2_FAMILY_A_N - 0.5, color="#CCCCCC",
                   linewidth=0.8, linestyle=":", zorder=1)
        ax.set_title(f"Qwen3.5-{size_display(m['label'])}", fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(comp_names, fontsize=7.5)
    fig.supxlabel(
        f"Δ signal (proposed − {AGENT_DISPLAY.get(RQ2_FOREST_BASELINE, RQ2_FOREST_BASELINE)}), "
        "failure=0 layer", fontsize=8)
    fig.text(0.5, -0.06,
             "Paired per-scenario delta with bootstrap 95% CI; filled marker: "
             "within-signal Holm-adjusted p < 0.05. † Asm→Obj is non-directional "
             "and excluded from the directional count.",
             ha="center", va="top", fontsize=6.5, color="#555555")
    fig.tight_layout()
    save(fig, outdir, "fig_rq2_forest", written)
    return True


def collect_scenario_pairs(pattern: str, agents: list[str]) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """label -> agent -> [(total_mean, align_mean)] per scenario (mean of runs)."""
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
                    t, c = r.get(RQ1_LEAD), align.get(a, {}).get("objective_assessment_similarity")
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
        all_t, all_c, within = [], [], []
        for a in agents:
            pts = pairs[lb].get(a, [])
            if not pts:
                continue
            ts, cs = zip(*pts)
            all_t += list(ts)
            all_c += list(cs)
            if len(pts) >= 10:
                within.append(spearman_rho(list(ts), list(cs)))
            ax.scatter(ts, cs, s=7, alpha=0.45, linewidths=0,
                       color=AGENT_COLOR.get(a, "#666666"),
                       marker=point_marker(a),
                       zorder=5 if a == PROPOSED else 3)
        # The pooled rho mixes within- and between-agent variation and is
        # dominated by the latter (the agent clusters sit apart), so its scope
        # is named and the within-agent view is printed beside it.
        rho = spearman_rho(all_t, all_c)
        med = median(within)
        txt = (f"pooled $\\rho_s$ = {rho:.2f}\n"
               f"n = {len(all_t)} (agent-scen.)")
        if not math.isnan(med):
            txt += f"\nwithin-agent med. = {med:.2f}"
        ax.set_title(f"Qwen3.5-{size_display(lb)}", fontsize=8)
        ax.text(0.04, 0.97, txt, transform=ax.transAxes, va="top",
                fontsize=6.0, color="#333333", zorder=8)
        ax.set_xlabel("ADDIE rubric (judge, /100)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Objective$\\to$assessment similarity")
    agent_legend(fig, agents, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, -0.30,
             "Pooled $\\rho_s$ is a between-agent correlation: every agent forms its own "
             "cluster, so it is not a per-scenario convergent-validity estimate.\n"
             "\"within-agent med.\" = median of the per-agent Spearman "
             f"$\\rho_s$ ({len(agents)} agents), the per-scenario view.",
             ha="center", va="top", fontsize=6.5, color="#555555")
    fig.tight_layout()
    save(fig, outdir, "fig_rq2_vs_rq1_scatter", written)
    return True


def fig_failure_rate(pooled: dict, outdir: Path, written: list[Path]) -> None:
    models = pooled["models"]
    agents = agents_in(pooled)
    n_agents = len(agents)
    width = 0.8 / n_agents

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    zero_everywhere: dict[str, bool] = {}
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
        zero_everywhere[a] = all(y == 0 for y in ys) and all(h == 0 for h in err_hi)
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
    # bar chart ⇒ patch handles: the shared line legend would advertise dashes
    # and markers that appear nowhere in the plot
    bar_legend(fig, agents,
               labels={a: f"{AGENT_DISPLAY.get(a, a)} (0% at every size)"
                       for a, z in zero_everywhere.items() if z},
               bbox_to_anchor=(0.5, -0.04))
    save(fig, outdir, "fig_failure_rate", written)


def fig_cost_quality(pooled: dict, outdir: Path, written: list[Path]) -> bool:
    tok = pooled.get("token_usage") or {}
    if not tok.get("models"):
        return False
    models = [m for m in pooled["models"] if m["label"] in tok["models"]]
    agents = agents_in(pooled)
    size_ms = [3.5, 5, 6.5, 8.5]  # marker size grows with model size

    # Tokens only: execution time reflects serving load/API latency, not the
    # method, so it is deliberately not plotted.
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    partial_seen = False
    for a in agents:
        xs, ys, notes = [], [], []
        for m in models:
            cell = tok["models"][m["label"]].get(a, {}).get("total_tokens", {})
            summary = m["agents"].get(a, {})
            xs.append(_nan(cell.get("mean")))
            ys.append(_nan(summary.get(RQ1_LEAD_FIELD)))
            # the two coordinates are averaged over different scenario sets
            # (tokens over runs with metadata, ADDIE over the complete-case
            # set); flag the markers where they diverge badly
            n_tok, n_add = cell.get("n_scenarios"), summary.get("n_scenarios_complete")
            union = m.get("n_scenarios_union") or 0
            thin = (union and isinstance(n_tok, int) and isinstance(n_add, int)
                    and min(n_tok, n_add) < 0.5 * union)
            notes.append(f"n={n_add}/{n_tok}" if thin else None)
        kw = line_kw(a)
        ax.plot(xs, ys, alpha=0.9, color=kw["color"], marker="",
                linewidth=0.9 if a != PROPOSED else 1.6, zorder=kw["zorder"])
        for x, y, ms, note in zip(xs, ys, size_ms, notes):
            ax.plot([x], [y], marker=point_marker(a), markersize=ms,
                    color=kw["color"] if note is None else "white",
                    markerfacecolor=kw["color"] if note is None else "white",
                    markeredgecolor="white" if note is None else kw["color"],
                    markeredgewidth=0.6 if note is None else 1.2,
                    zorder=kw["zorder"] + 1)
            if note:
                partial_seen = True
                # to the side, not below: these points sit on the axis floor
                ax.annotate(note, xy=(x, y), xytext=(7, -5),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=5.5, color=kw["color"])
    ax.set_xscale("log")
    ax.set_xlabel("Mean total tokens per scenario")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("ADDIE rubric (/100)")
    if partial_seen:
        # below the legend: the data area has no free corner at the small sizes,
        # which is exactly where the flagged markers are
        fig.text(0.5, -0.20,
                 "Hollow marker: the two coordinates are averaged over different "
                 "partial scenario sets\n(n = ADDIE complete-case / scenarios with "
                 "token metadata), so the point is not a property\nof one common "
                 "scenario set.",
                 ha="center", va="top", fontsize=6.5, color="#555555")
    fig.suptitle("Quality vs token cost — marker size grows with model size "
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
            # baselines "fail" at the smallest size so the thin-coverage
            # marking is exercised by the layout test too
            n_complete = 88 if (si or a == PROPOSED) else 6
            agents_summary[a] = {
                "n_scenarios_complete": n_complete,
                "n_scenarios_any_missing": 90 - n_complete,
                "mean_total": total, "sd_total_across_scenarios": 5.0,
                "mean_addie": total - 2.0, "mean_traj": total + 4.0,
                "run_means_total": [total - 0.4, total, total + 0.4],
                "run_to_run_sd": 1.2 + (3 - si) * 0.5,
                "run_to_run_sd_total": 1.2 + (3 - si) * 0.5,
                "run_to_run_sd_addie": 1.5 + (3 - si) * 0.6,
                "run_to_run_sd_traj": 0.8 + (3 - si) * 0.3,
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
            "comparisons": {"by_signal": {
                RQ1_LEAD: {"policies": {"complete_case": comp_rows}}}},
        })
        align_models[label] = {
            a: {"objective_assessment_similarity": {"n_scenarios": nsc,
                              "mean": min(0.95, 0.35 + si * 0.08 + off / 60
                                          + (0.15 if a == PROPOSED else 0)),
                              "sd_across_scenarios": 0.12},
                **{k: {"n_scenarios": nsc if ki % 3 else max(15, nsc // 3),
                       "mean": min(0.95, 0.3 + si * 0.07 + off / 70 + ki * 0.05
                                   + (0.12 if a == PROPOSED else 0)),
                       "sd_across_scenarios": 0.1}
                   for ki, (k, _) in enumerate(RQ2_COMPONENTS)}}
            for a, off in offsets.items()
            for nsc in [90 if (si or a == PROPOSED) else 40]
        }
        tok_models[label] = {
            a: {"total_tokens": {"n_scenarios": 90 if (si or a == PROPOSED) else 70,
                                 "mean": 20000 + i * 15000 + si * 5000},
                "execution_time_seconds": {"mean": 60 + i * 40 + si * 20}}
            for i, a in enumerate(offsets)
        }
    inter = {"per_baseline": [
        {"baseline": b, "trend_ols_slope_per_size_step": -0.8,
         "trend_n_common_scenarios": 88 if b != "baseline" else 3,
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
    """Synthetic per-scenario (ADDIE, similarity) pairs for the scatter figure."""
    rng = random.Random(42)
    pairs: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for m in pooled["models"]:
        by_agent = {}
        for a, s in m["agents"].items():
            comp = pooled["alignment"]["models"][m["label"]][a]["objective_assessment_similarity"]["mean"]
            pts = []
            for _ in range(90):
                t = rng.gauss(s[RQ1_LEAD_FIELD], 8.0)
                c = max(0.0, min(1.0, comp + (t - s[RQ1_LEAD_FIELD]) / 120
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
                        help="pooled_ladder.json from scripts/alignmentgraph-isd-bench/07_pool_ladder_runs.py")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="output directory for .pdf/.png figures")
    parser.add_argument("--runs-glob", default=None, metavar="PATTERN",
                        help="glob of raw run dirs (same convention as "
                             "07_pool_ladder_runs.py --auto-glob); enables the "
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
    if not fig_rq2_forest(pooled, args.outdir, written):
        print("! no alignment stats in pooled JSON — skipped fig_rq2_forest")
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
