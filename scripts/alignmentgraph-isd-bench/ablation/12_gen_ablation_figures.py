#!/usr/bin/env python3
"""Generate the ablation figure from pooled_ablation.json.

Companion of ../10_gen_paper_figures.py — the shared style (rcParams, save
helper, sized log-x axis) is imported from it, so the ablation figure renders
consistently with the ladder figures. Writes .pdf + .png into
results/generated/figures/ (sync with docs/scripts/sync_generated.sh):

  fig_ablation            paired per-scenario delta (arm − A0) vs model size,
                          one line per arm, error bar = bootstrap 95% CI,
                          Holm-significance markers (filled = p_holm < 0.05).
                          Negative = removing the component hurts. These are
                          simple contrasts vs A0, not 2x2 main effects.
  fig_ablation_factorial  each mechanism's effect under both states of the
                          other (the interaction view)
  fig_ablation_ladder     objective->assessment similarity per arm vs size
  fig_ablation_components heatmap arm x the 7 alignment panel signals per size

Usage:
  python scripts/alignmentgraph-isd-bench/ablation/12_gen_ablation_figures.py
  python scripts/alignmentgraph-isd-bench/ablation/12_gen_ablation_figures.py \
      --pooled results/pooled_ablation.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
    "alignmentgraph-isd-single-prose": "A1 — Single-agent, prose context",
    "alignmentgraph-isd-single-graph": "A2 — Single-agent, graph context",
    "alignmentgraph-isd-multi-prose": "A3 — Multi-agent, prose context",
}
ARM_COLOR = {
    "alignmentgraph-isd-single-prose": "#7f7f7f",
    "alignmentgraph-isd-single-graph": "#d62728",
    "alignmentgraph-isd-multi-prose": "#1f77b4",
}
ARM_LINESTYLE = {
    "alignmentgraph-isd-single-prose": ":",
    "alignmentgraph-isd-single-graph": "-",
    "alignmentgraph-isd-multi-prose": "--",
}

# All four arms (A0 included) for the alignment level figures.
ARM_ORDER = [
    "alignmentgraph-isd",
    "alignmentgraph-isd-single-graph",
    "alignmentgraph-isd-multi-prose",
    "alignmentgraph-isd-single-prose",
]
#: Same arm names as 11_gen_ablation_tables.py's ARM_DISPLAY — figure legends
#: and table rows come from the same pooled file and must read alike.
ARM_DISPLAY_FULL = {
    "alignmentgraph-isd": "A0 — Multi-agent, graph context",
    "alignmentgraph-isd-single-prose": "A1 — Single-agent, prose context",
    "alignmentgraph-isd-single-graph": "A2 — Single-agent, graph context",
    "alignmentgraph-isd-multi-prose": "A3 — Multi-agent, prose context",
}
ARM_COLOR_FULL = {
    "alignmentgraph-isd": "#0048B0",           # blue (full — emphasized)
    "alignmentgraph-isd-single-graph": "#d62728",
    "alignmentgraph-isd-multi-prose": "#1f77b4",
    "alignmentgraph-isd-single-prose": "#7f7f7f",
}
ARM_LS_FULL = {
    "alignmentgraph-isd": "-",
    "alignmentgraph-isd-single-graph": "-",
    "alignmentgraph-isd-multi-prose": "--",
    "alignmentgraph-isd-single-prose": ":",
}


def _align_models(pooled: dict) -> dict:
    return (pooled.get("alignment") or {}).get("models") or {}


def _ordered_labels(pooled: dict) -> list[str]:
    am = _align_models(pooled)
    return [m["label"] for m in pooled["models"] if m["label"] in am]


def fig_ablation_align_ladder(pooled: dict, outdir: Path, written: list[Path]) -> bool:
    """Objective->assessment similarity per arm vs size —
    the ablation analogue of fig_rq2_ladder."""
    am = _align_models(pooled)
    labels = _ordered_labels(pooled)
    if not labels:
        return False
    sizes = [GF.size_of_label(lb) for lb in labels]
    fig, ax = plt.subplots(figsize=(5.9, 3.2))
    for arm in ARM_ORDER:
        ys = [GF._nan(am[lb].get(arm, {})
                     .get("objective_assessment_similarity", {}).get("mean"))
              for lb in labels]
        proposed = arm == "alignmentgraph-isd"
        ax.plot(sizes, ys, color=ARM_COLOR_FULL[arm],
                linestyle=ARM_LS_FULL[arm],
                linewidth=2.4 if proposed else 1.5,
                marker="D" if proposed else "o", markersize=5.5 if proposed else 4.5,
                markeredgecolor="white", markeredgewidth=0.6,
                zorder=5 if proposed else 3, label=ARM_DISPLAY_FULL[arm])
    GF.sized_axis(ax, sizes, labels)
    ax.set_ylabel("Objective$\\to$assessment similarity (0–1)")
    ax.set_title("Ablation: objective$\\to$assessment similarity per arm across model sizes",
                 fontsize=8.5)
    # outside the data area: loc="best" dropped the key onto the A2/A3 series,
    # which were then drawn through the labels
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7)
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
    # figure width per panel matches fig_rq2_components: 7 narrow columns need
    # the room, otherwise the cell values merge into one digit run
    fig, axes = plt.subplots(
        1, len(labels),
        figsize=(3.05 * len(labels) + 1.6, 1.9 + 0.38 * n_arm), sharey=True)
    axes = [axes] if len(labels) == 1 else list(axes)
    cmap = plt.get_cmap("Blues")
    last_im = None
    for ax, lb in zip(axes, labels):
        cells = [[am[lb].get(arm, {}).get(k, {}) or {} for k in comp_keys]
                 for arm in ARM_ORDER]
        grid = [[GF._nan(c.get("mean")) for c in row] for row in cells]
        ns = [n for row in cells for c in row
              if isinstance(n := c.get("n_scenarios"), int)]
        n_max = max(ns) if ns else 0
        last_im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        for i, r in enumerate(grid):
            for j, v in enumerate(r):
                if GF.math.isnan(v):
                    continue
                txt = f"{v:.2f}"
                dark = v > 0.55
                ax.text(j, i, txt[1:] if 0 <= v < 1 else txt,
                        ha="center", va="center", fontsize=6.5,
                        color="white" if dark else "#333333")
                n = cells[i][j].get("n_scenarios")
                if isinstance(n, int) and n_max and n < 0.5 * n_max:
                    ax.text(j, i + 0.30, f"n={n}", ha="center", va="center",
                            fontsize=5.5, color="#FFC4C4" if dark else "#8B0000")
        ax.axvline(GF.RQ2_FAMILY_A_N - 0.5, color="white", linewidth=2.0, zorder=4)
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
    cbar.set_label("Panel signal mean (0–1)", fontsize=7)
    fig.text(0.5, -0.07, GF.RQ2_PANEL_FOOTNOTE, ha="center", va="top",
             fontsize=6.5, color="#555555", linespacing=1.5)
    GF.save(fig, outdir, "fig_ablation_components", written)
    return True


def arm_rows(pooled: dict, arm: str) -> dict[str, dict]:
    """label -> complete_case comparison row (stored direction: A0 − arm)."""
    out = {}
    for m in pooled["models"]:
        for row in (m.get("comparisons", {}).get("by_signal", {})
                    .get("addie_median", {}).get("policies", {})
                    .get("complete_case", [])):
            if row.get("baseline") == arm and row.get("n"):
                out[m["label"]] = row
    return out


#: Signal the factorial figure leads on (see 11's FACT_LEAD).
FACT_LEAD = "objective_assessment_similarity"

#: (effect key, condition label) pairs per panel — each panel shows ONE
#: mechanism's effect under both states of the other mechanism.
FACT_PANELS = [
    ("Decomposition", [("decomposition_effect_given_context_graph", "context = graph", "#0048B0"),
                       ("decomposition_effect_given_context_prose", "context = prose", "#C42D00")]),
    ("Context representation", [("context_effect_given_decomposition_multi", "decomposition = multi", "#0048B0"),
                                ("context_effect_given_decomposition_single", "decomposition = single", "#C42D00")]),
]


def fig_ablation_factorial(pooled: dict, outdir: Path, written: list[Path]) -> bool:
    """Each mechanism's effect under both states of the other one.

    The arm-vs-A0 figure shows simple contrasts at one fixed level of the
    other factor, so it cannot show that a mechanism looks idle only because
    the other one already prevented the problem. Two lines per panel — if they
    separate, the mechanisms substitute for each other, and the gap between
    them IS the interaction.
    """
    fact = pooled.get("factorial") or {}
    if "skipped" in fact or not fact.get("signals"):
        return False
    per_model = fact["signals"][FACT_LEAD]["per_model"]
    labels = [e["label"] for e in per_model]
    sizes = [GF.size_of_label(lb) for lb in labels]

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.5), sharey=True)
    for ax, (mech, series) in zip(axes, FACT_PANELS):
        ax.axhline(0.0, color="#444444", linewidth=0.8, zorder=2)
        for key, cond, color in series:
            xs, ys, lo_err, hi_err, sig = [], [], [], [], []
            for e, size in zip(per_model, sizes):
                d = e.get(key) or {}
                if not d.get("n"):
                    continue
                xs.append(size)
                ys.append(d["mean_diff"])
                lo_err.append(d["mean_diff"] - d["ci95"][0])
                hi_err.append(d["ci95"][1] - d["mean_diff"])
                sig.append(isinstance(d.get("p_raw"), (int, float)) and d["p_raw"] < 0.05)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=[lo_err, hi_err], color=color,
                        linewidth=1.8, capsize=2.5, elinewidth=0.9,
                        marker="", zorder=4, label=cond)
            for x, y, sg in zip(xs, ys, sig):
                ax.plot([x], [y], marker="o", markersize=5.0,
                        markerfacecolor=color if sg else "white",
                        markeredgecolor=color, markeredgewidth=1.1,
                        linestyle="", zorder=5)
        GF.sized_axis(ax, sizes, labels)
        ax.set_title(f"Effect of {mech}", fontsize=8.5)
        ax.legend(loc="best", frameon=False, fontsize=7)
    axes[0].set_ylabel("Δ objective→assessment similarity\n(paired per scenario)")
    # Short, wrapped, centred: a one-line footnote wider than the axes sets the
    # tight bounding box and halves the panels once the figure is scaled to a
    # paper column. The "separation IS the interaction" reading belongs in the
    # LaTeX caption. n is read from the data, never typed.
    n_per_size = sorted({e.get("n_scenarios") for e in per_model
                         if isinstance(e.get("n_scenarios"), int)})
    n_txt = (f"n = {n_per_size[0]} scenarios per size" if len(n_per_size) == 1
             else f"n = {n_per_size[0]}–{n_per_size[-1]} scenarios per size"
             if n_per_size else "n per size: see pooled JSON")
    fig.text(
        0.5, -0.03,
        "Error bar = bootstrap 95% CI; filled marker = p < 0.05\n"
        f"(Wilcoxon two-sided, uncorrected; {n_txt}).",
        ha="center", va="top", fontsize=6.5, color="#555555",
    )
    fig.tight_layout()
    GF.save(fig, outdir, "fig_ablation_factorial", written)
    return True


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
    ax.set_ylabel("Δ ADDIE (arm − A0), paired per scenario")
    # Simple contrasts, not main effects: each arm differs from A0 in one
    # mechanism while the other stays at one fixed level. The 2x2 main/simple
    # effects under both states live in fig_ablation_factorial.
    ax.set_title("Ablation: effect of removing a component vs A0 "
                 "(simple contrasts)", fontsize=8.5)
    ax.legend(loc="best", frameon=False)
    fig.text(
        0.5, -0.04,
        "Error bar = bootstrap 95% CI. Filled marker = p$_{Holm}$ < 0.05\n"
        f"(Wilcoxon two-sided, {len(arms)} arms per size). Each arm is compared\n"
        "with A0; each mechanism's effect under both states of the other\n"
        "is in fig_ablation_factorial.",
        ha="center", va="top", fontsize=6.5, color="#555555",
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
    if not fig_ablation_factorial(pooled, outdir, written):
        print("! no factorial layer in pooled JSON — skipped fig_ablation_factorial")
    if not fig_ablation_align_ladder(pooled, outdir, written):
        print("! no alignment data in pooled JSON — skipped fig_ablation_ladder")
    if not fig_ablation_components(pooled, outdir, written):
        print("! no alignment data in pooled JSON — skipped fig_ablation_components")
    for path in written:
        print(f"written: {path}")


if __name__ == "__main__":
    main()
