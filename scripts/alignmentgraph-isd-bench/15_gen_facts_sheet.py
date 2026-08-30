#!/usr/bin/env python3
"""15_gen_facts_sheet.py — consolidate every aggregate number into ONE file.

Why this exists
---------------
The generated numbers currently live in six files (``stats_summary.md``,
``stats_ablation.md``, ``stats_cost.md``, ``stats_sensitivity.md``,
``bloom_validation.md``, ``bloom_arms_validation.md``) plus four ``*_macros.tex``.
They use DIFFERENT sign conventions and cover DIFFERENT signal families, so a
reader who opens one file and stops has a real chance of quoting a number that
means the opposite of what they think. Two failure modes we actually hit:

1. Reading the ADDIE (judge) ablation table and concluding "graph context does
   not help", without reading the RQ2 panel table lower in the same file, where
   all 28 cells say the opposite. The two families disagree, and that
   disagreement IS the finding — but only if you see both.
2. Hand-rolling a claim over a printed table ("strongest at the small model",
   "skeleton drops deeper at every size") and getting the roll-up wrong.

So this script does two things the per-artifact generators deliberately do not:
it puts every family side by side under one explicit reading contract, and it
PRE-COMPUTES the roll-ups that people otherwise eyeball (counts of significant
cells, delta ranges, which size carries the largest effect, and where the two
families disagree in sign).

Scope
-----
Everything derivable from ``pooled_ladder.json`` + ``pooled_ablation.json``,
plus a full index of the ``\\newcommand`` values the paper actually \\input.
What is NOT here is listed explicitly in the output's section 0 so a reader
knows the boundary rather than assuming completeness.

Local only — no agent calls, no judge calls, no network.

Usage
-----
  python scripts/alignmentgraph-isd-bench/15_gen_facts_sheet.py
  python scripts/alignmentgraph-isd-bench/15_gen_facts_sheet.py --out results/generated/facts.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LADDER = BENCH_ROOT / "results" / "pooled_ladder.json"
DEFAULT_ABLATION = BENCH_ROOT / "results" / "pooled_ablation.json"
DEFAULT_OUTDIR = BENCH_ROOT / "results" / "generated"

PROPOSED = "alignmentgraph-isd"

# Judge signals are rubric points /100; panel signals are on [0,1]. Keeping the
# two lists separate here is what lets every table below label its own family.
JUDGE_SIGNALS = ("addie_median", "trajectory_score", "total_score")
JUDGE_LEAD = "addie_median"

ARM_LABELS = {
    "alignmentgraph-isd-single-prose": "A1 - single-agent, prose context",
    "alignmentgraph-isd-single-graph": "A2 - single-agent, graph context",
    "alignmentgraph-isd-multi-prose": "A3 - multi-agent, prose context",
}

MACRO_FILES = ("stats_macros.tex", "abl_macros.tex", "cost_macros.tex", "sens_macros.tex")
MACRO_RE = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\s*(?:%(.*))?$")

SIG_ALPHA = 0.05


# ── formatting ─────────────────────────────────────────────────────────────────
# Decimal POINT everywhere, English throughout (repo convention: generated
# artifacts are English and render 87.77, never 87,77).

def num(value, digits=3):
    if value is None:
        return "--"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def signed(value, digits=3):
    if value is None:
        return "--"
    return f"{float(value):+.{digits}f}"


def pval(value):
    if value is None:
        return "--"
    v = float(value)
    if v == 0:
        return "0"
    return f"{v:.3g}"


def ci(pair, digits=3):
    if not pair or len(pair) != 2:
        return "--"
    return f"[{float(pair[0]):+.{digits}f}, {float(pair[1]):+.{digits}f}]"


def verdict(p, alpha=SIG_ALPHA):
    if p is None:
        return "n/a"
    return "SIG" if float(p) < alpha else "n.s."


# ── data access helpers ────────────────────────────────────────────────────────

def judge_comparisons(model_block, signal, policy="complete_case"):
    """RQ1 comparison rows for one signal under one failure policy.

    Direction as stored: mean_diff = proposed - baseline (positive favours the
    proposed harness).
    """
    by_signal = (model_block.get("comparisons") or {}).get("by_signal") or {}
    sig = by_signal.get(signal) or {}
    rows = (sig.get("policies") or {}).get(policy) or []
    return rows if isinstance(rows, list) else []


def panel_blocks(pooled):
    """[(size_label, {signal: block})] from the alignment panel."""
    stats = ((pooled.get("alignment") or {}).get("stats") or {})
    out = []
    for per_model in stats.get("per_model") or []:
        out.append((per_model.get("label", "?"), per_model.get("panel") or {}))
    return out


def panel_signal_names(pooled):
    """Signal names in panel order.

    ``panel_signals`` carries dicts ({signal, family, directional}), not bare
    strings — normalise so every caller gets names.
    """
    stats = ((pooled.get("alignment") or {}).get("stats") or {})
    names = []
    for entry in stats.get("panel_signals") or []:
        names.append(entry.get("signal") if isinstance(entry, dict) else str(entry))
    names = [n for n in names if n]
    if names:
        return names
    for _, panel in panel_blocks(pooled):
        return list(panel)
    return []


# ── section 0: the reading contract ────────────────────────────────────────────

def sec_reading_contract(lad, abl):
    L = []
    L.append("## 0. Reading contract - READ BEFORE QUOTING ANY NUMBER\n")
    L.append("**There are two signal families in this file. They are measured by mechanically "
             "different instruments, they are on different scales, and on the ablation they "
             "DISAGREE. Never quote a number without naming its family.**\n")
    L.append("| Family | Signals | Scale | Instrument |")
    L.append("|---|---|---|---|")
    L.append(f"| JUDGE (RQ1) | `{'`, `'.join(JUDGE_SIGNALS)}` | rubric points /100 | LLM judge rubric |")
    L.append(f"| PANEL (RQ2) | `{'`, `'.join(panel_signal_names(lad))}` | [0, 1] | deterministic, no LLM |")
    L.append("")
    L.append(f"`{JUDGE_LEAD}` LEADS the RQ1 narrative because it matches the construct the research "
             "question is about, but it is NOT an endpoint: all three judge signals are reported "
             "side by side. The panel has NO primary endpoint at all - it is flat by design, so "
             "calling any one panel signal 'primary' reintroduces the composite the protocol removes.\n")
    L.append("### Sign conventions\n")
    L.append("Each table below restates its own direction in its header. The two that bite:\n")
    L.append("- **RQ1 / panel comparisons**: `mean_diff = proposed - baseline`. Positive favours "
             "the proposed harness.")
    L.append("- **Ablation**: the pooled JSON stores `A0 - arm` (the contribution of the removed "
             "component). This file prints the column `arm - A0` instead, i.e. **what happens when "
             "you REMOVE the component** - negative means removing it made the score worse. Both "
             "labels appear in the header of every ablation table; do not mix them up.")
    L.append("- **2x2 factorial**: `component ON - component OFF`. This is the OPPOSITE sign of "
             "the `arm - A0` column, by construction.\n")
    L.append("### Multiplicity\n")
    L.append("- `p_holm` = Holm within one signal (RQ1: 6 baselines; ablation: 3 arms).")
    L.append("- `p_holm_panel` = Holm over the whole panel x baseline family for that size. "
             "Panel rows carry both; a panel claim should survive `p_holm_panel`.")
    L.append("- Factorial p-values are UNCORRECTED - they are pre-specified structural contrasts, "
             "not a family of arm-vs-A0 tests.\n")
    L.append("### Failure policies\n")
    L.append("Complete-case is over AVAILABLE outputs - never describe it as unconditional. "
             "A crashed run leaves no output to score. The `zero` policy re-runs the paired test "
             "with those cells imputed at the signal floor.\n")
    L.append("### NOT in this file\n")
    L.append("- Cost ratios and token/quality economics -> `stats_cost.md`, `cost_macros.tex` "
             "(computed from run dirs, not from the pooled JSONs).")
    L.append("- Bloom classifier validity evidence -> `bloom_validation.md`, "
             "`bloom_arms_validation.md`.")
    L.append("- Per-scenario values. Everything here is aggregated to (size, agent, signal).")
    L.append("- Figures. See `results/generated/figures/`.\n")
    return L


# ── section 1: provenance ──────────────────────────────────────────────────────

def sec_provenance(lad, abl):
    L = ["## 1. Provenance and coverage\n"]
    align = lad.get("alignment") or {}
    cfg = lad.get("config") or {}
    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append(f"| pooled_ladder.json generated_at | {lad.get('generated_at', '?')} |")
    L.append(f"| pooled_ablation.json generated_at | {abl.get('generated_at', '?')} |")
    L.append(f"| proposed agent | `{cfg.get('proposed', PROPOSED)}` |")
    L.append(f"| baselines | {', '.join('`%s`' % b for b in cfg.get('baselines', []))} |")
    L.append(f"| model sizes | {', '.join(cfg.get('size_order', []))} |")
    L.append(f"| pooling | {cfg.get('pooling', '?')} |")
    L.append(f"| primary failure policy | {cfg.get('primary_failure_policy', '?')} |")
    L.append(f"| bootstrap | n={cfg.get('n_boot', '?')}, seed={cfg.get('bootstrap_seed', '?')} |")
    L.append(f"| encoder (primary) | {align.get('encoder_label', '?')} |")
    L.append(f"| encoder revision pinned | {align.get('encoder_revision_pinned', '?')} "
             f"({align.get('encoder_revision', '?')}) |")
    L.append(f"| Bloom classifier | {align.get('bloom_classifier', '?')} |")
    L.append(f"| alignment scenario files | {align.get('n_scenario_files', '?')} / "
             f"{align.get('n_scenario_files_expected', '?')} "
             f"(complete: {align.get('coverage_complete', '?')}) |")
    L.append("")
    L.append("Per size: runs and scenario union.\n")
    L.append("| Size | n_runs | n_scenarios_union |")
    L.append("|---|---|---|")
    for m in lad.get("models") or []:
        L.append(f"| {m.get('label','?')} | {m.get('n_runs','?')} | {m.get('n_scenarios_union','?')} |")
    L.append("")
    flag = abl.get("flag_check") or {}
    bad = [f"{size}/{agent}={state}"
           for size, agents in flag.items()
           for agent, state in (agents or {}).items() if state != "ok"]
    L.append(f"**Ablation arm flag check**: {'ALL OK' if not bad else 'PROBLEM -> ' + ', '.join(bad)} "
             "(verifies `metadata.run_config` matches the agent id each arm was registered under).\n")
    cons = (abl.get("a0_consistency") or {}).get("per_model") or []
    if cons:
        L.append("**A0 pooling invariant** (ablation vs ladder, on `total_score`; must be ~0 - this "
                 "checks both poolings read the same data, it is NOT a drift measurement, and "
                 "these A0 numbers are deliberately on a different signal from the ADDIE means "
                 "reported everywhere else):\n")
        L.append("| Size | A0 (ablation) | A0 (ladder) | delta | ladder run-to-run SD | within 2xSD |")
        L.append("|---|---|---|---|---|---|")
        for row in cons:
            L.append(f"| {row.get('label','?')} | "
                     f"{num(row.get('a0_mean_total_ablation_session'), 2)} | "
                     f"{num(row.get('a0_mean_total_ladder_pooled'), 2)} | "
                     f"{signed(row.get('delta'), 4)} | "
                     f"{num(row.get('ladder_run_to_run_sd'))} | "
                     f"{row.get('within_2x_ladder_run_sd', '--')} |")
        L.append("")
    return L


# ── section 2: roll-ups ────────────────────────────────────────────────────────

def rollup_rq1(lad):
    """For each baseline x judge signal: how many sizes the proposed agent wins
    significantly, and the delta range. Pre-computed so nobody counts by eye."""
    L = ["### 2.1 RQ1 judge signals - proposed vs each baseline\n",
         "Direction: `proposed - baseline`, complete-case. "
         "`sizes SIG+` counts sizes with p_holm < 0.05 AND a positive delta; "
         "`sizes SIG-` counts sizes significant in the baseline's favour.\n",
         "| Signal | Baseline | sizes SIG+ | sizes SIG- | sizes n.s. | delta range |",
         "|---|---|---|---|---|---|"]
    models = lad.get("models") or []
    for signal in JUDGE_SIGNALS:
        per_baseline = {}
        for m in models:
            for row in judge_comparisons(m, signal):
                per_baseline.setdefault(row.get("baseline"), []).append(row)
        for baseline, rows in sorted(per_baseline.items()):
            pos = sum(1 for r in rows if r.get("p_holm") is not None
                      and r["p_holm"] < SIG_ALPHA and r.get("mean_diff", 0) > 0)
            neg = sum(1 for r in rows if r.get("p_holm") is not None
                      and r["p_holm"] < SIG_ALPHA and r.get("mean_diff", 0) < 0)
            ns = len(rows) - pos - neg
            diffs = [r.get("mean_diff") for r in rows if r.get("mean_diff") is not None]
            rng = f"{signed(min(diffs), 2)} .. {signed(max(diffs), 2)}" if diffs else "--"
            L.append(f"| `{signal}` | `{baseline}` | {pos}/{len(rows)} | {neg}/{len(rows)} | "
                     f"{ns}/{len(rows)} | {rng} |")
    L.append("")
    return L


def rollup_panel(lad):
    L = ["### 2.2 RQ2 panel - proposed vs each baseline\n",
         "Direction: `proposed - baseline`, complete-case. Verdict uses `p_holm_panel` "
         "(the stricter of the two multiplicity levels).\n",
         "| Signal | Baseline | sizes SIG+ | sizes SIG- | sizes n.s. | delta range |",
         "|---|---|---|---|---|---|"]
    blocks = panel_blocks(lad)
    for signal in panel_signal_names(lad):
        per_baseline = {}
        for _, panel in blocks:
            block = panel.get(signal) or {}
            for row in block.get("comparisons") or []:
                per_baseline.setdefault(row.get("baseline"), []).append(row)
        for baseline, rows in sorted(per_baseline.items()):
            def p_of(r):
                return r.get("p_holm_panel", r.get("p_holm"))
            pos = sum(1 for r in rows if p_of(r) is not None
                      and p_of(r) < SIG_ALPHA and r.get("mean_diff", 0) > 0)
            neg = sum(1 for r in rows if p_of(r) is not None
                      and p_of(r) < SIG_ALPHA and r.get("mean_diff", 0) < 0)
            ns = len(rows) - pos - neg
            diffs = [r.get("mean_diff") for r in rows if r.get("mean_diff") is not None]
            rng = f"{signed(min(diffs))} .. {signed(max(diffs))}" if diffs else "--"
            L.append(f"| `{signal}` | `{baseline}` | {pos}/{len(rows)} | {neg}/{len(rows)} | "
                     f"{ns}/{len(rows)} | {rng} |")
    L.append("")
    return L


def _arm_rows(abl, signal, family):
    """[(size, arm, row)] for one signal, complete-case, with mean_diff stored as A0 - arm."""
    out = []
    if family == "judge":
        for m in abl.get("models") or []:
            for row in judge_comparisons(m, signal):
                out.append((m.get("label", "?"), row.get("baseline"), row))
    else:
        stats = ((abl.get("alignment") or {}).get("stats") or {})
        for per_model in stats.get("per_model") or []:
            block = (per_model.get("panel") or {}).get(signal) or {}
            for row in block.get("comparisons") or []:
                out.append((per_model.get("label", "?"), row.get("baseline"), row))
    return out


def rollup_ablation(abl, lad):
    """The section that would have caught our two bad claims."""
    L = ["### 2.3 Ablation - removing a component, per family\n",
         "**Direction here is `arm - A0` = the effect of REMOVING the component** "
         "(the pooled JSON stores the negation, `A0 - arm`). Negative = removing it made the "
         "score worse = the component was load-bearing.\n",
         "| Family | Signal | Arm | cells neg | cells pos | cells SIG | delta range (arm - A0) |",
         "|---|---|---|---|---|---|---|"]
    panel_names = panel_signal_names(abl) or panel_signal_names(lad)
    plan = [("JUDGE", s) for s in JUDGE_SIGNALS] + [("PANEL", s) for s in panel_names]
    for family_label, signal in plan:
        fam = "judge" if family_label == "JUDGE" else "panel"
        rows = _arm_rows(abl, signal, fam)
        by_arm = {}
        for size, arm, row in rows:
            by_arm.setdefault(arm, []).append((size, row))
        for arm in sorted(by_arm, key=lambda a: ARM_LABELS.get(a, a)):
            entries = by_arm[arm]
            # flip: stored A0 - arm  ->  printed arm - A0
            deltas = [-(r.get("mean_diff") or 0.0) for _, r in entries]
            ps = [r.get("p_holm_panel", r.get("p_holm")) for _, r in entries]
            neg = sum(1 for d in deltas if d < 0)
            pos = sum(1 for d in deltas if d > 0)
            sig = sum(1 for p in ps if p is not None and p < SIG_ALPHA)
            digits = 2 if fam == "judge" else 3
            rng = f"{signed(min(deltas), digits)} .. {signed(max(deltas), digits)}" if deltas else "--"
            L.append(f"| {family_label} | `{signal}` | {ARM_LABELS.get(arm, arm)} | "
                     f"{neg}/{len(deltas)} | {pos}/{len(deltas)} | {sig}/{len(ps)} | {rng} |")
    L.append("")

    # Cross-family aggregate for each arm — the headline roll-up.
    L.append("**Per-arm totals across a whole family** (this is the number to quote, not a "
             "hand-count of the tables in section 5):\n")
    L.append("| Arm | Family | cells | neg | pos | SIG | SIG and neg | delta range (arm - A0) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm in sorted(ARM_LABELS, key=lambda a: ARM_LABELS[a]):
        for family_label, signals, digits in (("JUDGE", JUDGE_SIGNALS, 2),
                                              ("PANEL", tuple(panel_names), 3)):
            fam = "judge" if family_label == "JUDGE" else "panel"
            deltas, ps = [], []
            for signal in signals:
                for size, a, row in _arm_rows(abl, signal, fam):
                    if a != arm:
                        continue
                    deltas.append(-(row.get("mean_diff") or 0.0))
                    ps.append(row.get("p_holm_panel", row.get("p_holm")))
            if not deltas:
                continue
            neg = sum(1 for d in deltas if d < 0)
            pos = sum(1 for d in deltas if d > 0)
            sig = sum(1 for p in ps if p is not None and p < SIG_ALPHA)
            signeg = sum(1 for d, p in zip(deltas, ps)
                         if d < 0 and p is not None and p < SIG_ALPHA)
            L.append(f"| {ARM_LABELS[arm]} | {family_label} | {len(deltas)} | {neg} | {pos} | "
                     f"{sig} | **{signeg}** | {signed(min(deltas), digits)} .. "
                     f"{signed(max(deltas), digits)} |")
    L.append("")
    return L


def rollup_scale_gradient(abl, lad):
    """Which size carries the largest |effect|. Exists to stop 'strongest at the
    small model' style claims that the per-signal argmax does not support."""
    L = ["### 2.4 Scale gradient check - where does the largest effect actually sit?\n",
         "For each arm x signal, the size with the largest |arm - A0|. **If the argmax size is "
         "scattered across sizes, there is NO monotone scale gradient and a sentence like "
         "'strongest at the smallest model' is unsupported.** Count the distribution before "
         "writing any scale claim.\n"]
    panel_names = panel_signal_names(abl) or panel_signal_names(lad)
    for family_label, signals, digits in (("JUDGE", JUDGE_SIGNALS, 2),
                                          ("PANEL", tuple(panel_names), 3)):
        fam = "judge" if family_label == "JUDGE" else "panel"
        L.append(f"**{family_label} family**\n")
        L.append("| Arm | Signal | argmax size | |delta| there | delta by size |")
        L.append("|---|---|---|---|---|")
        tally = {}
        for arm in sorted(ARM_LABELS, key=lambda a: ARM_LABELS[a]):
            for signal in signals:
                entries = [(size, -(row.get("mean_diff") or 0.0))
                           for size, a, row in _arm_rows(abl, signal, fam) if a == arm]
                if not entries:
                    continue
                top = max(entries, key=lambda t: abs(t[1]))
                tally.setdefault(arm, {}).setdefault(top[0], 0)
                tally[arm][top[0]] += 1
                by_size = ", ".join(f"{s}={signed(d, digits)}" for s, d in entries)
                L.append(f"| {ARM_LABELS[arm]} | `{signal}` | **{top[0]}** | "
                         f"{num(abs(top[1]), digits)} | {by_size} |")
        L.append("")
        for arm, counts in tally.items():
            spread = ", ".join(f"{s}: {c}" for s, c in sorted(counts.items()))
            n_sizes = len(counts)
            note = ("CONCENTRATED - a scale claim may be defensible" if n_sizes == 1
                    else "SCATTERED - do NOT claim a scale gradient")
            L.append(f"- {ARM_LABELS[arm]} ({family_label}) argmax distribution: {spread} -> **{note}**")
        L.append("")
    return L


def rollup_family_disagreement(abl, lad):
    """Where the judge and the panel point opposite ways — the finding itself."""
    L = ["### 2.5 Cross-family disagreement (judge vs panel)\n",
         "Same arm, same size, opposite sign between the two families. **Every row here is a "
         "place where quoting one family alone would misrepresent the result.**\n",
         "| Arm | Size | JUDGE lead (`%s`) | PANEL (n signals neg / total) | disagreement |"
         % JUDGE_LEAD,
         "|---|---|---|---|---|"]
    panel_names = panel_signal_names(abl) or panel_signal_names(lad)
    sizes = [m.get("label") for m in abl.get("models") or []]
    for arm in sorted(ARM_LABELS, key=lambda a: ARM_LABELS[a]):
        for size in sizes:
            jrows = [row for s, a, row in _arm_rows(abl, JUDGE_LEAD, "judge")
                     if a == arm and s == size]
            if not jrows:
                continue
            jd = -(jrows[0].get("mean_diff") or 0.0)
            jp = jrows[0].get("p_holm")
            pneg = ptot = 0
            for signal in panel_names:
                for s, a, row in _arm_rows(abl, signal, "panel"):
                    if a != arm or s != size:
                        continue
                    ptot += 1
                    if -(row.get("mean_diff") or 0.0) < 0:
                        pneg += 1
            if not ptot:
                continue
            panel_dir = -1 if pneg > ptot / 2 else (1 if pneg < ptot / 2 else 0)
            judge_dir = -1 if jd < 0 else (1 if jd > 0 else 0)
            flag = "**OPPOSITE**" if judge_dir * panel_dir < 0 else "same direction"
            L.append(f"| {ARM_LABELS[arm]} | {size} | {signed(jd, 2)} ({verdict(jp)}) | "
                     f"{pneg}/{ptot} negative | {flag} |")
    L.append("")
    L.append("Reading rule that follows from this table: a sentence about a mechanism must name "
             "its family. 'Removing graph context hurts alignment' and 'removing graph context "
             "does not lower the judged score' are both true and belong in the same paragraph.\n")
    return L


# ── sections 3-5: full tables ──────────────────────────────────────────────────

def sec_rq1_tables(lad):
    L = ["## 3. RQ1 judge signals - full tables\n",
         "Per-agent descriptive means come from the pooled per-agent block. "
         "**A mean and its SD must come from the same signal** - the columns below are already "
         "paired correctly (`mean_addie` with `run_to_run_sd_addie`, etc.).\n"]
    for m in lad.get("models") or []:
        L.append(f"### {m.get('label','?')}\n")
        L.append("| Agent | n complete | n any missing | ADDIE | +-SD | Traj | +-SD | Total | +-SD |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for agent, a in sorted((m.get("agents") or {}).items()):
            L.append(f"| `{agent}` | {a.get('n_scenarios_complete','--')} | "
                     f"{a.get('n_scenarios_any_missing','--')} | "
                     f"{num(a.get('mean_addie'), 2)} | {num(a.get('run_to_run_sd_addie'), 2)} | "
                     f"{num(a.get('mean_traj'), 2)} | {num(a.get('run_to_run_sd_traj'), 2)} | "
                     f"{num(a.get('mean_total'), 2)} | {num(a.get('run_to_run_sd_total'), 2)} |")
        L.append("")
        for signal in JUDGE_SIGNALS:
            rows = judge_comparisons(m, signal)
            if not rows:
                continue
            L.append(f"Paired vs baselines - `{signal}`, complete-case, "
                     "direction `proposed - baseline`:\n")
            L.append("| Baseline | n | mean_diff | CI95 | p_raw | p_holm | r_rb | verdict |")
            L.append("|---|---|---|---|---|---|---|---|")
            for r in rows:
                L.append(f"| `{r.get('baseline')}` | {r.get('n','--')} | "
                         f"{signed(r.get('mean_diff'), 2)} | {ci(r.get('ci95'), 2)} | "
                         f"{pval(r.get('p_raw'))} | {pval(r.get('p_holm'))} | "
                         f"{num(r.get('rank_biserial_r'))} | {verdict(r.get('p_holm'))} |")
            L.append("")
    return L


def sec_panel_tables(lad):
    L = ["## 4. RQ2 panel - full tables\n",
         "Flat panel, no primary endpoint. Continuous signals are mean-max RECTIFIED cosine "
         "(`max(0, cos)` on [0,1]) - a scale choice, not a matching cutoff. There is no "
         "similarity threshold anywhere in the protocol.\n"]
    for size, panel in panel_blocks(lad):
        L.append(f"### {size}\n")
        for signal, block in panel.items():
            fam = block.get("family", "?")
            directional = block.get("directional", True)
            tag = "" if directional else " - NON-DIRECTIONAL"
            L.append(f"#### `{signal}` [{fam}]{tag}\n")
            desc = block.get("align_descriptive") or {}
            if desc:
                L.append("| Agent | mean | CI95 |")
                L.append("|---|---|---|")
                for agent, d in sorted(desc.items()):
                    if isinstance(d, dict):
                        L.append(f"| `{agent}` | {num(d.get('mean'))} | {ci(d.get('ci95'))} |")
                    else:
                        L.append(f"| `{agent}` | {num(d)} | -- |")
                L.append("")
            rows = block.get("comparisons") or []
            if rows:
                L.append("Paired, complete-case, direction `proposed - baseline`:\n")
                L.append("| Baseline | n | mean_diff | CI95 | p_holm | p_holm_panel | r_rb | verdict (panel) |")
                L.append("|---|---|---|---|---|---|---|---|")
                for r in rows:
                    pp = r.get("p_holm_panel", r.get("p_holm"))
                    L.append(f"| `{r.get('baseline')}` | {r.get('n','--')} | "
                             f"{signed(r.get('mean_diff'))} | {ci(r.get('ci95'))} | "
                             f"{pval(r.get('p_holm'))} | {pval(pp)} | "
                             f"{num(r.get('rank_biserial_r'))} | {verdict(pp)} |")
                L.append("")
    return L


def sec_ablation_tables(abl, lad):
    L = ["## 5. Ablation - full tables\n",
         "**Direction: the `arm - A0` column is the effect of REMOVING the component.** "
         "The pooled JSON stores `A0 - arm`; it is negated here. Negative = removing it "
         "made the score worse.\n"]
    panel_names = panel_signal_names(abl) or panel_signal_names(lad)

    L.append("### 5.1 JUDGE family\n")
    for m in abl.get("models") or []:
        L.append(f"#### {m.get('label','?')}\n")
        L.append("| Agent/arm | ADDIE | Traj | Total |")
        L.append("|---|---|---|---|")
        for agent, a in sorted((m.get("agents") or {}).items()):
            L.append(f"| `{agent}` | {num(a.get('mean_addie'), 2)} | "
                     f"{num(a.get('mean_traj'), 2)} | {num(a.get('mean_total'), 2)} |")
        L.append("")
        for signal in JUDGE_SIGNALS:
            rows = judge_comparisons(m, signal)
            if not rows:
                continue
            L.append(f"`{signal}` - paired vs A0:\n")
            L.append("| Arm | n | arm - A0 | CI95 (arm - A0) | p_holm | r_rb | verdict |")
            L.append("|---|---|---|---|---|---|---|")
            for r in rows:
                d = -(r.get("mean_diff") or 0.0)
                c = r.get("ci95") or [None, None]
                cflip = [-c[1], -c[0]] if None not in c else None
                arm = r.get("baseline")
                L.append(f"| {ARM_LABELS.get(arm, arm)} | {r.get('n','--')} | {signed(d, 2)} | "
                         f"{ci(cflip, 2)} | {pval(r.get('p_holm'))} | "
                         f"{num(r.get('rank_biserial_r'))} | {verdict(r.get('p_holm'))} |")
            L.append("")

    L.append("### 5.2 PANEL family\n")
    stats = ((abl.get("alignment") or {}).get("stats") or {})
    for per_model in stats.get("per_model") or []:
        L.append(f"#### {per_model.get('label','?')}\n")
        L.append("| Signal | Arm | n | arm - A0 | p_holm | p_holm_panel | verdict (panel) |")
        L.append("|---|---|---|---|---|---|---|")
        panel = per_model.get("panel") or {}
        for signal in panel_names:
            block = panel.get(signal) or {}
            for r in block.get("comparisons") or []:
                d = -(r.get("mean_diff") or 0.0)
                pp = r.get("p_holm_panel", r.get("p_holm"))
                arm = r.get("baseline")
                L.append(f"| `{signal}` | {ARM_LABELS.get(arm, arm)} | {r.get('n','--')} | "
                         f"{signed(d)} | {pval(r.get('p_holm'))} | {pval(pp)} | {verdict(pp)} |")
        L.append("")

    fac = abl.get("factorial") or {}
    if fac:
        L.append("### 5.3 2x2 factorial (decomposition x context representation)\n")
        L.append("**Direction: `multi-agent - single-agent` / `graph - prose`** - the OPPOSITE "
                 "sign of the `arm - A0` columns above. Positive = the richer setting scores "
                 "higher. p-values UNCORRECTED (pre-specified structural contrasts).\n")
        L.append("A negative `interaction` means the two mechanisms SUBSTITUTE for each other - "
                 "but only where both simple effects carry the beneficial sign. Where both are "
                 "negative it is a magnitude statement; where the signs differ it is a crossover, "
                 "neither substitution nor complementarity.\n")
        for signal, block in (fac.get("signals") or {}).items():
            L.append(f"#### `{signal}`\n")
            L.append("| Size | n | decomp \\| ctx=graph | decomp \\| ctx=prose | ctx \\| decomp=multi | "
                     "ctx \\| decomp=single | interaction |")
            L.append("|---|---|---|---|---|---|---|")
            digits = 2 if signal in JUDGE_SIGNALS else 3
            for row in block.get("per_model") or []:
                def cell(key):
                    e = row.get(key) or {}
                    if not e:
                        return "--"
                    return f"{signed(e.get('mean_diff'), digits)} (p={pval(e.get('p_raw'))})"
                L.append(f"| {row.get('label','?')} | {row.get('n_scenarios','--')} | "
                         f"{cell('decomposition_effect_given_context_graph')} | "
                         f"{cell('decomposition_effect_given_context_prose')} | "
                         f"{cell('context_effect_given_decomposition_multi')} | "
                         f"{cell('context_effect_given_decomposition_single')} | "
                         f"{cell('interaction')} |")
            L.append("")

    va = abl.get("self_validation_activity") or []
    if va:
        L.append("### 5.4 Self-validation activity (repairs actually performed)\n")
        L.append("Operational counts, not a scored signal. Self-validation is always on (not "
                 "one of the two ablated axes), so this is context for the decomposition/context "
                 "effects above, not mechanism evidence for an on/off switch.\n")
        L.append("| Size | Arm | n scenario-runs | verifier events (mean) | repair events (mean) "
                 "| % runs with a repair |")
        L.append("|---|---|---|---|---|---|")
        for row in va:
            for arm, d in (row.get("arms") or {}).items():
                label = "A0 - full pipeline" if arm == PROPOSED else ARM_LABELS.get(arm, arm)
                L.append(f"| {row.get('label','?')} | {label} | "
                         f"{(d or {}).get('n_scenario_runs','--')} | "
                         f"{num((d or {}).get('verifier_events_mean'), 2)} | "
                         f"{num((d or {}).get('repair_events_mean'), 2)} | "
                         f"{num((d or {}).get('pct_scenario_runs_with_repair'), 1)} |")
        L.append("")
    return L


# ── sections 6-9 ───────────────────────────────────────────────────────────────

def sec_interaction(lad):
    L = ["## 6. Scale interaction (DiD, largest size - smallest size)\n"]
    by_signal = lad.get("interaction_by_signal") or {}
    if not by_signal and lad.get("interaction"):
        by_signal = {"(default)": lad["interaction"]}
    for signal, block in by_signal.items():
        L.append(f"### `{signal}`\n")
        defn = block.get("definition")
        if defn:
            L.append(f"> {defn}\n")
        L.append("| Baseline | DiD | CI95 | p | verdict |")
        L.append("|---|---|---|---|---|")
        for row in block.get("per_baseline") or []:
            p = row.get("p_raw", row.get("p"))
            L.append(f"| `{row.get('baseline')}` | {signed(row.get('did', row.get('mean_diff')), 2)} | "
                     f"{ci(row.get('ci95'), 2)} | {pval(p)} | {verdict(p)} |")
        L.append("")
    ia = ((lad.get("alignment") or {}).get("stats") or {}).get("interaction_align") or {}
    if ia:
        L.append("### Panel DiD (`interaction_align`)\n")
        if ia.get("definition"):
            L.append(f"> {ia['definition']}\n")
        rows = ia.get("per_baseline") or []
        if rows:
            L.append("| Baseline | DiD | CI95 | p |")
            L.append("|---|---|---|---|")
            for row in rows:
                p = row.get("p_raw", row.get("p"))
                L.append(f"| `{row.get('baseline')}` | {signed(row.get('did', row.get('mean_diff')))} | "
                         f"{ci(row.get('ci95'))} | {pval(p)} |")
            L.append("")
    return L


def sec_failures(lad):
    L = ["## 7. Output failure rates\n"]
    fr = lad.get("failure_rates") or {}
    if fr.get("definition"):
        L.append(f"> {fr['definition']}\n")
    L.append("Direction: these compare the proposed agent's failed-scenario count against each "
             "baseline's. Fisher exact two-sided, Holm over the 6 comparisons per size.\n")
    for block in fr.get("per_model") or []:
        rows = block.get("rows") or []
        if not rows:
            continue
        L.append(f"### {block.get('label','?')}\n")
        L.append("| Baseline | proposed failed | baseline failed | n | baseline run-level | "
                 "p_fisher | p_holm | verdict |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            L.append(f"| `{r.get('baseline')}` | {r.get('proposed_failed_scenarios','--')} | "
                     f"{r.get('baseline_failed_scenarios','--')} | {r.get('n_scenarios','--')} | "
                     f"{r.get('baseline_failed_run_level','--')} | "
                     f"{pval(r.get('fisher_p_two_sided'))} | {pval(r.get('p_holm'))} | "
                     f"{verdict(r.get('p_holm'))} |")
        L.append("")
    return L


def sec_tokens(lad):
    L = ["## 8. Token usage (operational metadata)\n"]
    tu = lad.get("token_usage") or {}
    if tu.get("source"):
        L.append(f"> {tu['source']}\n")
    L.append("Cost RATIOS and token/quality economics are NOT here - see `stats_cost.md`. "
             "Two artifact traps that apply to these fields: `metadata.cost_usd` is wrong "
             "(commercial API prices applied to a self-hosted pod), and wall-clock is "
             "QUEUED THROUGHPUT, not latency - never call it latency.\n")
    fields = ("prompt_tokens", "completion_tokens", "total_tokens", "llm_calls",
              "execution_time_seconds")
    for size, block in (tu.get("models") or {}).items():
        L.append(f"### {size}\n")
        L.append("| Agent | " + " | ".join(f"{f} (mean +- SD)" for f in fields) + " |")
        L.append("|---" * (len(fields) + 1) + "|")
        for agent, d in sorted(block.items()):
            cells = []
            for f in fields:
                e = (d or {}).get(f) or {}
                digits = 1 if f in ("llm_calls", "execution_time_seconds") else 0
                cells.append(f"{num(e.get('mean'), digits)} +- "
                             f"{num(e.get('sd_across_scenarios'), digits)}")
            L.append(f"| `{agent}` | " + " | ".join(cells) + " |")
        L.append("")
    return L


def sec_sensitivity(lad):
    L = ["## 9. Instrument sensitivity star\n"]
    sens = lad.get("alignment_sensitivity") or {}
    if not sens:
        L.append("_No `alignment_sensitivity` block in the pooled file._\n")
        return L
    if sens.get("definition"):
        L.append(f"> {sens['definition']}\n")
    summary = sens.get("summary") or {}
    if summary:
        L.append("| Field | Value |")
        L.append("|---|---|")
        for k, v in summary.items():
            L.append(f"| {k} | {num(v, 4) if isinstance(v, float) else v} |")
        L.append("")
    L.append("The star is 3 axes, NOT a grid: each arm deviates from the primary configuration "
             "on exactly ONE axis. An encoder swap cannot move a Bloom level and a Bloom swap "
             "cannot move a cosine, so read each arm's own `axis` and `agreement_metric`.\n")
    return L


def sec_macros(outdir):
    L = ["## 10. Macro index - every value the paper can \\input\n",
         "Parsed from the generated `*_macros.tex`. If a number appears in the paper, it should "
         "appear here; a paper number that is NOT in this table was typed by hand and violates "
         "the repo rule that all reported numbers are generated.\n"]
    total = 0
    for name in MACRO_FILES:
        path = outdir / name
        if not path.exists():
            L.append(f"_`{name}` not found - skipped._\n")
            continue
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            m = MACRO_RE.match(line.strip())
            if m:
                entries.append((m.group(1), m.group(2), (m.group(3) or "").strip()))
        total += len(entries)
        L.append(f"### `{name}` ({len(entries)} macros)\n")
        L.append("| Macro | Value | Comment |")
        L.append("|---|---|---|")
        for macro, value, comment in entries:
            L.append(f"| `\\{macro}` | {value} | {comment} |")
        L.append("")
    L.insert(2, f"**{total} macros total.**\n")
    return L


# ── main ───────────────────────────────────────────────────────────────────────

def build(lad, abl, outdir, brief=False):
    L = ["# Facts sheet - every aggregate number in one file\n"]
    L.append("AUTO-GENERATED by `15_gen_facts_sheet.py`. Do not edit by hand.\n")
    L.append("This file exists so that a reader (human or model) can quote a number without "
             "opening six artifacts with three different sign conventions. Section 0 is the "
             "reading contract; section 2 pre-computes the roll-ups so nobody has to count "
             "cells by eye.\n")
    if brief:
        L.append("**BRIEF MODE** - sections 0-2 only (contract + provenance + roll-ups). "
                 "Every roll-up here is computed from the same pooled files as the full sheet; "
                 "run without `--brief` for the per-cell tables behind these counts.\n")
    L += sec_reading_contract(lad, abl)
    L += sec_provenance(lad, abl)
    L.append("## 2. Roll-ups - pre-computed, do not hand-count\n")
    L += rollup_rq1(lad)
    L += rollup_panel(lad)
    L += rollup_ablation(abl, lad)
    L += rollup_scale_gradient(abl, lad)
    L += rollup_family_disagreement(abl, lad)
    if brief:
        L.append("_Sections 3-10 (full per-cell tables, DiD, failure rates, token usage, "
                 "sensitivity, macro index) omitted in brief mode._\n")
        return "\n".join(L) + "\n"
    L += sec_rq1_tables(lad)
    L += sec_panel_tables(lad)
    L += sec_ablation_tables(abl, lad)
    L += sec_interaction(lad)
    L += sec_failures(lad)
    L += sec_tokens(lad)
    L += sec_sensitivity(lad)
    L += sec_macros(outdir)
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pooled", type=Path, default=DEFAULT_LADDER,
                    help="pooled_ladder.json (default: %(default)s)")
    ap.add_argument("--pooled-ablation", type=Path, default=DEFAULT_ABLATION,
                    help="pooled_ablation.json (default: %(default)s)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                    help="directory holding the *_macros.tex and receiving the output")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: <outdir>/facts{,_brief}.md)")
    ap.add_argument("--brief", action="store_true",
                    help="emit sections 0-2 only (reading contract + provenance + roll-ups) - "
                         "the interpretation-ready core, for when the whole sheet will not fit")
    args = ap.parse_args()

    missing = [p for p in (args.pooled, args.pooled_ablation) if not p.exists()]
    if missing:
        raise SystemExit("Missing pooled file(s): " + ", ".join(str(p) for p in missing) +
                         "\nRun 07_pool_ladder_runs.py and ablation/08_pool_ablation_runs.py first.")

    lad = json.loads(args.pooled.read_text(encoding="utf-8"))
    abl = json.loads(args.pooled_ablation.read_text(encoding="utf-8"))

    out = args.out or (args.outdir / ("facts_brief.md" if args.brief else "facts.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build(lad, abl, args.outdir, brief=args.brief)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}  ({len(text)//1024} KB, {text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
