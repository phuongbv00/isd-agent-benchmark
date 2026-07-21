#!/usr/bin/env python3
"""Generate paper tables + number macros from pooled_ladder.json (no hand-typed numbers).

Reads the output of scripts/7_pool_ladder_runs.py and writes into
docs/paper/acm/vn/generated/ :

  tab_rq1.tex           complete booktabs table* — agents x 4 model sizes,
                        cell = mean Total (+- run-to-run SD), n column per size
  tab_rq2_alignment.tex alignment-metric table (Porter/Webb/coverage etc.) if
                        alignment data exists in the pooled JSON; otherwise a
                        placeholder comment file (still valid for \\input)
  stats_macros.tex      \\newcommand macros for every headline number used in
                        prose (deterministic names, documented inline)
  stats_summary.md      human-readable dump of ALL numbers + test statistics +
                        effect sizes + CIs, for the paper writer to cross-check

Numbers are formatted Vietnamese-style with a decimal comma written as ``{,}``
so every macro/cell renders correctly in BOTH text and math mode (matches the
existing hand-written tables, e.g. 87,77).

Without real data, run ``--demo``: it emits the same three .tex files with
clearly marked TBD-DEMO placeholder values so the paper can already compile
with \\input{generated/...}.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # master-thesis/
DEFAULT_OUTDIR = REPO_ROOT / "docs" / "paper" / "acm" / "vn" / "generated"
DEFAULT_POOLED = Path(__file__).resolve().parents[1] / "results" / "pooled_ladder.json"

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

# parsed size (in B params) -> digit-free macro word
SIZE_WORDS = {0.8: "ZeroEightB", 2.0: "TwoB", 4.0: "FourB", 9.0: "NineB"}
FALLBACK_WORDS = ["SizeA", "SizeB", "SizeC", "SizeD", "SizeE", "SizeF"]


# ── formatting ───────────────────────────────────────────────────────────────

def fmt_vn(x: float, nd: int = 2) -> str:
    """87.77 -> '87{,}77' (renders 87,77 in text and math mode)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}".replace(".", "{,}")


def fmt_p(p: float) -> str:
    """p-value in Vietnamese math style: 4{,}4\\times 10^{-16} or <10^{-16}."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p >= 0.001:
        return fmt_vn(p, 3)
    if p <= 0:
        return "<10^{-16}"
    exp = math.floor(math.log10(p))
    mant = p / 10 ** exp
    return f"{fmt_vn(mant, 1)}\\times 10^{{{exp}}}"


def size_of_label(label: str) -> float:
    m = re.findall(r"(\d+(?:\.\d+)?)b\b", label.lower())
    return float(m[-1]) if m else float("nan")


def size_word(label: str, index: int) -> str:
    s = size_of_label(label)
    return SIZE_WORDS.get(s, FALLBACK_WORDS[index % len(FALLBACK_WORDS)])


def size_display(label: str) -> str:
    s = size_of_label(label)
    if math.isnan(s):
        return label
    txt = f"{s:g}".replace(".", ",")
    return f"{txt}B"


# ── demo data ────────────────────────────────────────────────────────────────

def demo_pooled() -> dict:
    """Synthetic pooled structure with clearly fake values (TBD-DEMO)."""
    labels = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"]
    base_by_size = [40.0, 55.0, 65.0, 75.0]
    offsets = {  # fake per-agent offsets from the size base
        "baseline": 0.0, "eduplanner": -8.0, "addie-agent": 1.0,
        "rpisd-agent": -1.0, "dick-carey-agent": 2.0, "react-isd": 1.5,
        "alignmentgraph-isd": 6.0,
    }
    models = []
    for label, base in zip(labels, base_by_size):
        agents = {}
        for a, off in offsets.items():
            agents[a] = {
                "n_scenarios_complete": 90, "n_scenarios_any_missing": 0,
                "mean_total": base + off, "sd_total_across_scenarios": 5.0,
                "mean_within_scenario_run_sd": 1.2,
                "mean_addie": base + off - 2.0, "mean_traj": base + off + 4.0,
                "run_means_total": [base + off - 0.4, base + off, base + off + 0.4],
                "run_to_run_sd": 0.4,
            }
        rows = []
        for b, off in offsets.items():
            if b == "alignmentgraph-isd":
                continue
            d = offsets["alignmentgraph-isd"] - off
            rows.append({
                "baseline": b, "n": 90, "n_effective_nonzero": 90,
                "mean_diff": d, "ci95": [d - 1.0, d + 1.0],
                "wilcoxon_w_plus": 4000.0, "p_raw": 1e-9,
                "rank_biserial_r": 0.9, "p_holm": 6e-9,
            })
        models.append({
            "label": label, "run_dirs": ["TBD-DEMO"] * 3, "n_runs": 3,
            "n_scenarios_union": 90,
            "per_run": [{"dir": "TBD-DEMO", "n_scenarios_scored": 90,
                         "missing_by_agent": {}}] * 3,
            "agents": agents,
            "comparisons": {
                "holm_family": "6 comparisons (proposed vs each baseline) within this model size",
                "primary_policy": "complete_case", "min_score_floor": 0.0,
                "policies": {"complete_case": rows, "zero": rows, "min_score": rows},
            },
        })
    inter = {
        "definition": "TBD-DEMO",
        "per_baseline": [{
            "baseline": b, "smallest": labels[0], "largest": labels[-1],
            "n_paired_scenarios": 90,
            "delta_smallest_mean": offsets["alignmentgraph-isd"] - off + 2.0,
            "delta_largest_mean": offsets["alignmentgraph-isd"] - off,
            "did_mean": -2.0, "did_ci95": [-3.0, -1.0], "did_wilcoxon_p": 0.01,
            "trend_delta_by_size": {lb: offsets["alignmentgraph-isd"] - off + 2.0 - 0.66 * i
                                    for i, lb in enumerate(labels)},
            "trend_n_common_scenarios": 90,
            "trend_ols_slope_per_size_step": -0.66,
            "trend_slope_ci95": [-1.0, -0.3],
        } for b, off in offsets.items() if b != "alignmentgraph-isd"],
    }
    alignment = {
        "source": "TBD-DEMO",
        "models": {
            label: {
                a: {"porter_alignment": {"n_scenarios": 90, "mean": 0.5 + off / 100,
                                         "sd_across_scenarios": 0.1},
                    "webb_consistency": {"n_scenarios": 90, "mean": 0.6 + off / 100,
                                         "sd_across_scenarios": 0.1},
                    "coverage": {"n_scenarios": 90, "mean": 0.7 + off / 100,
                                 "sd_across_scenarios": 0.1}}
                for a, off in offsets.items()
            } for label in labels
        },
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"proposed": "alignmentgraph-isd",
                   "baselines": [a for a in AGENT_ORDER if a != "alignmentgraph-isd"],
                   "primary_failure_policy": "complete_case",
                   "n_boot": 10000, "bootstrap_seed": 42,
                   "size_order": labels},
        "models": models, "interaction": inter, "alignment": alignment,
        "_demo": True,
    }


# ── generators ───────────────────────────────────────────────────────────────

def header_comment(pooled: dict, what: str) -> str:
    demo = pooled.get("_demo", False)
    lines = [
        f"% AUTO-GENERATED by isd-agent-benchmark/scripts/8_gen_paper_tables.py — DO NOT EDIT BY HAND",
        f"% {what}",
        f"% generated_at: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if demo:
        lines += [
            "% " + "!" * 70,
            "% !! TBD-DEMO: ALL VALUES BELOW ARE FAKE PLACEHOLDERS (--demo mode). !!",
            "% !! Regenerate from the real pooled_ladder.json before submission.  !!",
            "% " + "!" * 70,
        ]
    else:
        lines.append(f"% source pooled_ladder.json generated_at: {pooled.get('generated_at')}")
    return "\n".join(lines) + "\n"


def gen_tab_rq1(pooled: dict) -> str:
    models = pooled["models"]
    demo = pooled.get("_demo", False)
    agents_present = [a for a in AGENT_ORDER
                      if any(a in m["agents"] for m in models)]
    proposed = pooled["config"]["proposed"]

    # best (max mean_total) per size column, for bolding
    best_by_model = {}
    for m in models:
        vals = {a: m["agents"][a]["mean_total"] for a in agents_present if a in m["agents"]}
        vals = {a: v for a, v in vals.items() if v is not None and not math.isnan(v)}
        best_by_model[m["label"]] = max(vals, key=vals.get) if vals else None

    ncols = 1 + 2 * len(models)
    colspec = "@{}l" + "r@{\\hskip 3pt}r" * len(models) + "@{}"
    demo_note = " [TBD-DEMO: số liệu giả, chờ ladder runs thật]" if demo else ""
    lines = [header_comment(pooled, "tab_rq1.tex — RQ1 ladder table (agents x model sizes)")]
    lines.append("\\begin{table*}[t]")
    lines.append(
        "  \\caption{RQ1: ISD-Agent-Bench \\texttt{test\\_90} theo thang kích thước mô hình"
        " (Qwen, 3 run độc lập mỗi kích thước). Mỗi ô: trung bình Total composite"
        " per-scenario ($0{,}7\\cdot\\mathrm{ADDIE}+0{,}3\\cdot\\mathrm{Traj}$), gộp"
        " mean-of-runs trên các scenario complete-case; giá trị sau"
        " $\\pm$ là SD giữa 3 run (run-to-run); $n$ = số scenario đủ cả 3 run."
        " \\textbf{Đậm} = tốt nhất theo cột." + demo_note + "}")
    lines.append("  \\label{tab:rq1-ladder}")
    lines.append("  \\small")
    lines.append(f"  \\begin{{tabular}}{{{colspec}}}")
    lines.append("    \\toprule")
    heads = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{Qwen3.5-{size_display(m['label'])}}}" for m in models)
    lines.append(f"    & {heads} \\\\")
    cmids = "".join(
        f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models)))
    lines.append(f"    {cmids}")
    subs = " & ".join(["Total {\\scriptsize$\\pm$SD}", "$n$"] * len(models))
    lines.append(f"    Agent & {subs} \\\\")
    lines.append("    \\midrule")
    for a in agents_present:
        name = AGENT_DISPLAY.get(a, a)
        if a == proposed:
            name = f"\\textbf{{{name}}}"
        cells = []
        for m in models:
            s = m["agents"].get(a)
            if not s or s["mean_total"] is None or math.isnan(s["mean_total"]):
                cells += ["--", "--"]
                continue
            val = fmt_vn(s["mean_total"])
            if best_by_model.get(m["label"]) == a:
                val = f"\\textbf{{{val}}}"
            sd = s.get("run_to_run_sd")
            sd_txt = ("{\\scriptsize$\\pm$" + fmt_vn(sd, 2) + "}"
                      if sd is not None and not math.isnan(sd) else "")
            n = s["n_scenarios_complete"]
            miss = s.get("n_scenarios_any_missing", 0)
            n_txt = f"{n}" if not miss else f"{n}\\textsuperscript{{*}}"
            cells += [val + sd_txt, n_txt]
        lines.append(f"    {name} & " + " & ".join(cells) + " \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append(
        "  \\par\\smallskip\\footnotesize\\textsuperscript{*}$n<$ tổng số scenario:"
        " có scenario lỗi/thiếu ở ít nhất một run (complete-case; xem stats\\_summary.md).")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def gen_tab_rq2(pooled: dict) -> str:
    align = pooled.get("alignment")
    demo = pooled.get("_demo", False)
    if not align or not align.get("models"):
        return (header_comment(pooled, "tab_rq2_alignment.tex — placeholder")
                + "% No alignment_scores.json data was present in the pooled input.\n"
                  "% Re-run scripts/7_pool_ladder_runs.py after the alignment metric\n"
                  "% (RQ2) lands in the scenario dirs, then regenerate this file.\n"
                  "% This file intentionally renders nothing.\n")

    model_labels = [m["label"] for m in pooled["models"] if m["label"] in align["models"]]
    # metric set: union across models/agents, keep stable sorted order
    metrics: list[str] = []
    for lb in model_labels:
        for agent_metrics in align["models"][lb].values():
            for k in agent_metrics:
                if k not in metrics:
                    metrics.append(k)
    metrics = metrics[:3]  # keep the table printable; full dump is in stats_summary.md
    agents_present = [a for a in AGENT_ORDER
                      if any(a in align["models"][lb] for lb in model_labels)]

    def short(metric: str) -> str:
        base = metric.split(".")[-1].replace("_", " ")
        return {"porter alignment": "Porter", "webb consistency": "Webb",
                "coverage": "Cover."}.get(base, base[:7].title())

    demo_note = " [TBD-DEMO: số liệu giả]" if demo else ""
    colspec = "@{}l" + ("r" * len(metrics)) * len(model_labels) + "@{}"
    lines = [header_comment(pooled, "tab_rq2_alignment.tex — RQ2 alignment metrics x model sizes")]
    lines.append("\\begin{table*}[t]")
    lines.append(
        "  \\caption{RQ2: chỉ số alignment (trung bình per-scenario, gộp mean-of-runs)"
        " theo kích thước mô hình." + demo_note + "}")
    lines.append("  \\label{tab:rq2-alignment}")
    lines.append("  \\small")
    lines.append(f"  \\begin{{tabular}}{{{colspec}}}")
    lines.append("    \\toprule")
    k = len(metrics)
    heads = " & ".join(
        f"\\multicolumn{{{k}}}{{c}}{{Qwen3.5-{size_display(lb)}}}" for lb in model_labels)
    lines.append(f"    & {heads} \\\\")
    cmids = "".join(
        f"\\cmidrule(lr){{{2 + k * i}-{1 + k * (i + 1)}}}" for i in range(len(model_labels)))
    lines.append(f"    {cmids}")
    subs = " & ".join(short(mt) for _ in model_labels for mt in metrics)
    lines.append(f"    Agent & {subs} \\\\")
    lines.append("    \\midrule")
    for a in agents_present:
        name = AGENT_DISPLAY.get(a, a)
        if a == pooled["config"]["proposed"]:
            name = f"\\textbf{{{name}}}"
        cells = []
        for lb in model_labels:
            am = align["models"][lb].get(a, {})
            for mt in metrics:
                v = am.get(mt, {}).get("mean")
                cells.append(fmt_vn(v, 3) if v is not None else "--")
        lines.append(f"    {name} & " + " & ".join(cells) + " \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def gen_macros(pooled: dict) -> tuple[str, str]:
    """Returns (stats_macros.tex, stats_summary.md)."""
    models = pooled["models"]
    proposed = pooled["config"]["proposed"]
    demo = pooled.get("_demo", False)
    words = [size_word(m["label"], i) for i, m in enumerate(models)]

    tex: list[str] = [header_comment(pooled, "stats_macros.tex — headline numbers for prose")]
    md: list[str] = [
        "# Ladder statistics summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}"
        + ("  \n**TBD-DEMO: every value below is a fake placeholder.**" if demo else ""),
        "",
        f"- Proposed agent: `{proposed}`",
        f"- Size order: {', '.join(m['label'] for m in models)}",
        f"- Pooling: mean of {models[0]['n_runs']} runs per scenario; primary policy complete-case",
        f"- Bootstrap: {pooled['config'].get('n_boot', 10000)} resamples, seed "
        f"{pooled['config'].get('bootstrap_seed', 42)}",
        "- Holm family: 6 comparisons (proposed vs each baseline) per model size",
        "",
    ]

    def newcmd(name: str, value: str, desc: str) -> None:
        tex.append(f"% {desc}")
        tex.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    newcmd("ladderNumModels", str(len(models)), "number of model sizes in the ladder")
    newcmd("ladderNumRunsPerModel", str(models[0]["n_runs"]),
           "independent runs per model size")
    n_scen = max(m["n_scenarios_union"] for m in models)
    newcmd("ladderNumScenarios", str(n_scen), "scenarios per run (test_90)")
    newcmd("ladderNumAgents", str(len(pooled["config"]["baselines"]) + 1),
           "agents compared (6 baselines + proposed)")

    all_p_holm: list[float] = []
    for m, w in zip(models, words):
        agents = m["agents"]
        rows = m["comparisons"]["policies"]["complete_case"]
        rows_p = [r for r in rows if "p_holm" in r]
        harness = agents.get(proposed, {})
        base_vals = {a: s["mean_total"] for a, s in agents.items()
                     if a != proposed and s["mean_total"] is not None
                     and not math.isnan(s["mean_total"])}
        best_b = max(base_vals, key=base_vals.get) if base_vals else None

        md += [f"## {m['label']} (macro suffix: {w})", ""]
        newcmd(f"rqOneHarnessTotal{w}", fmt_vn(harness.get("mean_total", float("nan"))),
               f"{proposed} pooled mean Total at {m['label']}")
        newcmd(f"rqOneHarnessRunSD{w}", fmt_vn(harness.get("run_to_run_sd", float("nan"))),
               f"{proposed} run-to-run SD (SD of the run-level means) at {m['label']}")
        if best_b:
            newcmd(f"rqOneBestBaselineTotal{w}", fmt_vn(base_vals[best_b]),
                   f"best baseline ({best_b}) pooled mean Total at {m['label']}")
            newcmd(f"rqOneBestBaselineName{w}", AGENT_DISPLAY.get(best_b, best_b),
                   f"name of the best baseline at {m['label']}")
            row = next((r for r in rows_p if r["baseline"] == best_b), None)
            if row:
                newcmd(f"rqOneDeltaBest{w}", fmt_vn(row["mean_diff"]),
                       f"mean paired delta {proposed} - {best_b} at {m['label']}")
                newcmd(f"rqOneDeltaBestCILo{w}", fmt_vn(row["ci95"][0]),
                       "bootstrap 95% CI lower bound of that delta")
                newcmd(f"rqOneDeltaBestCIHi{w}", fmt_vn(row["ci95"][1]),
                       "bootstrap 95% CI upper bound of that delta")
        if rows_p:
            pmax = max(r["p_holm"] for r in rows_p)
            all_p_holm.append(pmax)
            newcmd(f"rqOnePHolmMax{w}", fmt_p(pmax),
                   f"max Holm-adjusted p over the 6 comparisons at {m['label']}")

        md.append(f"- {proposed}: Total={harness.get('mean_total'):.4f}, "
                  f"runSD={harness.get('run_to_run_sd'):.4f}, "
                  f"nCC={harness.get('n_scenarios_complete')}, "
                  f"missing={harness.get('n_scenarios_any_missing')}")
        md.append("")
        md.append("| baseline | n | mean diff | 95% CI | W+ | r_rb | p | p_holm |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if "p_raw" not in r:
                md.append(f"| {r['baseline']} | 0 | -- | -- | -- | -- | -- | -- |")
                continue
            md.append(
                f"| {r['baseline']} | {r['n']} | {r['mean_diff']:+.4f} "
                f"| [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
                f"| {r['wilcoxon_w_plus']:.1f} | {r['rank_biserial_r']:+.3f} "
                f"| {r['p_raw']:.3e} | {r['p_holm']:.3e} |")
        md.append("")
        md.append("Sensitivity (mean diff under imputation policies):")
        for policy in ("zero", "min_score"):
            vals = ", ".join(
                f"{r['baseline']}={r['mean_diff']:+.2f}"
                for r in m["comparisons"]["policies"][policy] if "mean_diff" in r)
            md.append(f"- {policy}: {vals}")
        md.append("")

    if all_p_holm:
        newcmd("rqOnePHolmMaxAll", fmt_p(max(all_p_holm)),
               "max Holm-adjusted p across ALL sizes x baselines (complete-case)")

    # scale drop of the proposed agent: Total(largest) - Total(smallest)
    h_small = models[0]["agents"].get(proposed, {}).get("mean_total", float("nan"))
    h_large = models[-1]["agents"].get(proposed, {}).get("mean_total", float("nan"))
    newcmd("rqOneHarnessDropSmall", fmt_vn(h_large - h_small),
           f"{proposed} Total at largest size minus smallest size "
           f"({models[-1]['label']} - {models[0]['label']}); positive = larger is better")

    inter = pooled.get("interaction", {})
    md += ["## Interaction (method x model size, DiD largest - smallest)", "",
           inter.get("definition", ""), ""]
    row = next((r for r in inter.get("per_baseline", []) if r["baseline"] == "baseline"),
               None)
    if row:
        newcmd("rqOneDiDVsBaseline", fmt_vn(row["did_mean"]),
               "DiD of delta(proposed - baseline): largest minus smallest size")
        newcmd("rqOneDiDVsBaselineCILo", fmt_vn(row["did_ci95"][0]),
               "bootstrap 95% CI lower bound of that DiD")
        newcmd("rqOneDiDVsBaselineCIHi", fmt_vn(row["did_ci95"][1]),
               "bootstrap 95% CI upper bound of that DiD")
        newcmd("rqOneDiDVsBaselineP", fmt_p(row["did_wilcoxon_p"]),
               "Wilcoxon p of the per-scenario DiD values vs 0")
        newcmd("rqOneTrendSlopeVsBaseline", fmt_vn(row["trend_ols_slope_per_size_step"]),
               "OLS slope of delta(proposed - baseline) per size step (4 sizes)")
    for r in inter.get("per_baseline", []):
        md.append(
            f"- vs {r['baseline']}: n={r['n_paired_scenarios']}, "
            f"DiD={r['did_mean']:+.4f} CI[{r['did_ci95'][0]:+.4f}, {r['did_ci95'][1]:+.4f}] "
            f"p={r['did_wilcoxon_p']:.3e}; trend "
            + " ".join(f"{k}:{v:+.2f}" for k, v in r["trend_delta_by_size"].items())
            + f"; slope/step={r['trend_ols_slope_per_size_step']:+.4f} "
              f"CI[{r['trend_slope_ci95'][0]:+.4f}, {r['trend_slope_ci95'][1]:+.4f}]")
    md.append("")

    align = pooled.get("alignment")
    md.append("## Alignment metrics (RQ2)")
    if align and align.get("models"):
        for lb, agents_metrics in align["models"].items():
            md.append(f"### {lb}")
            for agent, metrics in agents_metrics.items():
                parts = ", ".join(f"{mt}={v['mean']:.4f} (SD {v['sd_across_scenarios']:.4f}, "
                                  f"n={v['n_scenarios']})" for mt, v in metrics.items())
                md.append(f"- {agent}: {parts}")
    else:
        md.append("No alignment_scores.json data in the pooled input — skipped.")
    md.append("")

    if demo:
        newcmd("ladderDataIsDemo", "TBD-DEMO",
               "sentinel: nonempty means the generated numbers are placeholders")
    return "\n".join(tex) + "\n", "\n".join(md) + "\n"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper tables + macros from pooled_ladder.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pooled", type=Path, default=DEFAULT_POOLED,
                        help="pooled_ladder.json from scripts/7_pool_ladder_runs.py")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="output directory for generated .tex files")
    parser.add_argument("--demo", action="store_true",
                        help="emit files with clearly marked TBD-DEMO placeholder "
                             "values (no pooled JSON needed)")
    args = parser.parse_args()

    if args.demo:
        pooled = demo_pooled()
        print("DEMO MODE: emitting TBD-DEMO placeholder values.")
    else:
        if not args.pooled.exists():
            print(f"Error: pooled JSON not found: {args.pooled} (use --demo for placeholders)",
                  file=sys.stderr)
            sys.exit(1)
        pooled = json.loads(args.pooled.read_text(encoding="utf-8"))

    args.outdir.mkdir(parents=True, exist_ok=True)
    macros_tex, summary_md = gen_macros(pooled)
    outputs = {
        "tab_rq1.tex": gen_tab_rq1(pooled),
        "tab_rq2_alignment.tex": gen_tab_rq2(pooled),
        "stats_macros.tex": macros_tex,
        "stats_summary.md": summary_md,
    }
    for name, content in outputs.items():
        path = args.outdir / name
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
