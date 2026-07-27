#!/usr/bin/env python3
"""Generate the ablation figure from pooled_ablation.json.

Companion of ../10_gen_paper_figures.py — the shared style (rcParams, save
helper, sized log-x axis) is imported from it, so the ablation figure renders
consistently with the ladder figures. Writes .pdf + .png into
results/generated/figures/ (sync with docs/scripts/sync_generated.sh):

  fig_ablation   paired per-scenario delta (arm − A0) vs model size, one line
                 per arm, error bar = bootstrap 95% CI, Holm-significance
                 markers (filled = p_holm < 0.05). Negative = removing the
                 component hurts.

Usage:
  python scripts/alignmentgraph-isd-bench/ablation/12_gen_ablation_figures.py
  python scripts/alignmentgraph-isd-bench/ablation/12_gen_ablation_figures.py \
      --pooled results/pooled_ablation.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
BENCH_ROOT = _HERE.parents[2]  # isd-agent-benchmark/
DEFAULT_POOLED = BENCH_ROOT / "results" / "pooled_ablation.json"
DEFAULT_OUTDIR = BENCH_ROOT / "results" / "generated" / "figures"


def _load_figures_module():
    """Import ../10_gen_paper_figures.py for the shared style + helpers.

    Importing it applies the shared plt.rcParams and performs the matplotlib
    availability check (it exits with the install hint if missing).
    """
    path = _HERE.parent / "10_gen_paper_figures.py"
    spec = importlib.util.spec_from_file_location("gen_paper_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GF = _load_figures_module()
plt = GF.plt

ARM_DISPLAY = {
    "alignmentgraph-isd-no-verifier": "A1 — No verifier",
    "alignmentgraph-isd-no-graph-ctx": "A2 — No graph context",
    "alignmentgraph-isd-skeleton": "A3 — Skeleton (both off)",
}
ARM_COLOR = {
    "alignmentgraph-isd-no-verifier": "#d62728",
    "alignmentgraph-isd-no-graph-ctx": "#1f77b4",
    "alignmentgraph-isd-skeleton": "#7f7f7f",
}
ARM_LINESTYLE = {
    "alignmentgraph-isd-no-verifier": "-",
    "alignmentgraph-isd-no-graph-ctx": "--",
    "alignmentgraph-isd-skeleton": ":",
}

# All four arms (A0 included) for the alignment level figures.
ARM_ORDER = [
    "alignmentgraph-isd",
    "alignmentgraph-isd-no-verifier",
    "alignmentgraph-isd-no-graph-ctx",
    "alignmentgraph-isd-skeleton",
]
ARM_DISPLAY_FULL = {
    "alignmentgraph-isd": "A0 — Full",
    "alignmentgraph-isd-no-verifier": "A1 — No verifier",
    "alignmentgraph-isd-no-graph-ctx": "A2 — No graph context",
    "alignmentgraph-isd-skeleton": "A3 — Skeleton (both off)",
}
ARM_COLOR_FULL = {
    "alignmentgraph-isd": "#0048B0",           # blue (full — emphasized)
    "alignmentgraph-isd-no-verifier": "#d62728",
    "alignmentgraph-isd-no-graph-ctx": "#1f77b4",
    "alignmentgraph-isd-skeleton": "#7f7f7f",
}
ARM_LS_FULL = {
    "alignmentgraph-isd": "-",
    "alignmentgraph-isd-no-verifier": "-",
    "alignmentgraph-isd-no-graph-ctx": "--",
    "alignmentgraph-isd-skeleton": ":",
}


def _align_models(pooled: dict) -> dict:
    return (pooled.get("alignment") or {}).get("models") or {}


def _ordered_labels(pooled: dict) -> list[str]:
    am = _align_models(pooled)
    return [m["label"] for m in pooled["models"] if m["label"] in am]


def fig_ablation_align_ladder(pooled: dict, outdir: Path, written: list[Path]) -> bool:
    """Primary endpoint (objective-assessment alignment) per arm vs size —
    the ablation analogue of fig_rq2_ladder."""
    am = _align_models(pooled)
    labels = _ordered_labels(pooled)
    if not labels:
        return False
    sizes = [GF.size_of_label(lb) for lb in labels]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for arm in ARM_ORDER:
        ys = [GF._nan(am[lb].get(arm, {})
                     .get("objective_assessment_alignment", {}).get("mean"))
              for lb in labels]
        proposed = arm == "alignmentgraph-isd"
        ax.plot(sizes, ys, color=ARM_COLOR_FULL[arm],
                linestyle=ARM_LS_FULL[arm],
                linewidth=2.4 if proposed else 1.5,
                marker="D" if proposed else "o", markersize=5.5 if proposed else 4.5,
                markeredgecolor="white", markeredgewidth=0.6,
                zorder=5 if proposed else 3, label=ARM_DISPLAY_FULL[arm])
    GF.sized_axis(ax, sizes, labels)
    ax.set_ylabel("Objective–assessment alignment (0–1)")
    ax.set_title("Ablation: alignment level per arm across model sizes")
    ax.legend(loc="best", frameon=False, fontsize=7)
    GF.save(fig, outdir, "fig_ablation_ladder", written)
    return True


def fig_ablation_components(pooled: dict, outdir: Path, written: list[Path]) -> bool:
    """Per-criterion heatmap (arm x 4 criteria) per model size — the ablation
    analogue of fig_rq2_components."""
    am = _align_models(pooled)
    labels = _ordered_labels(pooled)
    if not labels:
        return False
    comp_keys = [k for k, _ in GF.RQ2_COMPONENTS]
    comp_names = [n for _, n in GF.RQ2_COMPONENTS]
    n_arm = len(ARM_ORDER)
    fig, axes = plt.subplots(
        1, len(labels),
        figsize=(2.15 * len(labels) + 1.4, 1.6 + 0.34 * n_arm), sharey=True)
    axes = [axes] if len(labels) == 1 else list(axes)
    cmap = plt.get_cmap("Blues")
    last_im = None
    for ax, lb in zip(axes, labels):
        grid = [[GF._nan(am[lb].get(arm, {}).get(k, {}).get("mean"))
                 for k in comp_keys] for arm in ARM_ORDER]
        last_im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        for i, r in enumerate(grid):
            for j, v in enumerate(r):
                if GF.math.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else "#333333")
        ax.set_title(f"Qwen3.5-{GF.size_display(lb)}")
        ax.set_xticks(range(len(comp_names)))
        ax.set_xticklabels(comp_names, rotation=35, ha="right", fontsize=7.5)
        ax.grid(visible=False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(n_arm))
    axes[0].set_yticklabels(
        [ARM_DISPLAY_FULL[a] for a in ARM_ORDER], fontsize=8)
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("Criterion mean (0–1)", fontsize=7)
    GF.save(fig, outdir, "fig_ablation_components", written)
    return True


def arm_rows(pooled: dict, arm: str) -> dict[str, dict]:
    """label -> complete_case comparison row (stored direction: A0 − arm)."""
    out = {}
    for m in pooled["models"]:
        for row in m.get("comparisons", {}).get("policies", {}).get("complete_case", []):
            if row.get("baseline") == arm and row.get("n"):
                out[m["label"]] = row
    return out


def fig_ablation(pooled: dict, outdir: Path, written: list[Path]) -> None:
    models = pooled["models"]
    arms = pooled["config"]["arms"]
    labels = [m["label"] for m in models]
    sizes = [GF.size_of_label(lb) for lb in labels]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axhline(0.0, color="#444444", linewidth=0.8, zorder=2)
    for arm in arms:
        rows = arm_rows(pooled, arm)
        xs, ys, lo_err, hi_err, sig = [], [], [], [], []
        for lb, size in zip(labels, sizes):
            row = rows.get(lb)
            if row is None:
                continue
            # plotted direction: arm − A0 (negative = removal hurts)
            delta = -row["mean_diff"]
            ci_lo, ci_hi = -row["ci95"][1], -row["ci95"][0]
            xs.append(size)
            ys.append(delta)
            lo_err.append(delta - ci_lo)
            hi_err.append(ci_hi - delta)
            p = row.get("p_holm")
            sig.append(isinstance(p, (int, float)) and p < 0.05)
        if not xs:
            continue
        color = ARM_COLOR.get(arm, "#666666")
        ax.errorbar(
            xs, ys, yerr=[lo_err, hi_err],
            color=color, linestyle=ARM_LINESTYLE.get(arm, "-"),
            linewidth=1.6, capsize=2.5, elinewidth=0.9,
            marker="", zorder=4, label=ARM_DISPLAY.get(arm, arm),
        )
        # markers: filled = significant after Holm, open = not
        for x, y, s in zip(xs, ys, sig):
            ax.plot(
                [x], [y], marker="o", markersize=5.0,
                markerfacecolor=color if s else "white",
                markeredgecolor=color, markeredgewidth=1.1,
                linestyle="", zorder=5,
            )
    GF.sized_axis(ax, sizes, labels)
    ax.set_ylabel("Δ Total (arm − A0), paired per scenario")
    ax.set_title("Ablation: component contribution across model sizes")
    ax.legend(loc="best", frameon=False)
    fig.text(
        0.01, -0.04,
        "Error bar = bootstrap 95% CI. Filled marker = p$_{Holm}$ < 0.05 "
        f"(Wilcoxon two-sided, {len(arms)} arms per size).",
        fontsize=6.5, color="#555555",
    )
    GF.save(fig, outdir, "fig_ablation", written)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fig_ablation.{pdf,png} from pooled_ablation.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pooled", default=str(DEFAULT_POOLED))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    pooled_path = Path(args.pooled)
    if not pooled_path.exists():
        print(f"Error: {pooled_path} not found — run "
              "scripts/alignmentgraph-isd-bench/ablation/08_pool_ablation_runs.py first.",
              file=sys.stderr)
        sys.exit(1)
    pooled = json.loads(pooled_path.read_text(encoding="utf-8"))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    fig_ablation(pooled, outdir, written)
    if not fig_ablation_align_ladder(pooled, outdir, written):
        print("! no alignment data in pooled JSON — skipped fig_ablation_ladder")
    if not fig_ablation_components(pooled, outdir, written):
        print("! no alignment data in pooled JSON — skipped fig_ablation_components")
    for path in written:
        print(f"written: {path}")


if __name__ == "__main__":
    main()
