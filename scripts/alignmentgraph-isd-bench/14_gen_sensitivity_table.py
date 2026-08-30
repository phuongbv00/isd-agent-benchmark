#!/usr/bin/env python3
"""Generate the RQ2 instrument-sensitivity table, macros and stats dump.

Reads ``pooled["alignment_sensitivity"]`` (written by
07_pool_ladder_runs.py --encoders/--auto-encoders) and writes into
``results/generated/``:

  tab_sensitivity.tex   booktabs table, ONE BLOCK PER AXIS: rows = arms, per
                        size the proposed agent's mean on that axis's signal,
                        the rank correlation with the primary arm, and the
                        proposed agent's rank.
  sens_macros.tex       \\newcommand macros for every headline number, so the
                        "the ranking survives an instrument swap" claim in the
                        prose is generated rather than typed.
  stats_sensitivity.md  human-readable dump (per axis x arm x size: rho, tau-b,
                        exact p, top-1, sign/Holm agreement, coverage).

The sweep is a 3-axis STAR of 5 arms, not an encoder-only sweep: four arms vary
the ENCODER (nemotron primary, bgem3, granite, tfidf) and one varies the BLOOM
CLASSIFIER (lexicon, on the primary encoder). A Bloom swap cannot move a cosine
and an encoder swap cannot move a Bloom level, so each axis is reported on the
signal it can actually move -- read per arm from pooled ``axis`` /
``agreement_metric``, never hard-coded here. Mixing them in one column would
print two different measurements under one header.

Naming: RQ2 is a flat 7-signal panel with NO primary endpoint, and "alignment"
names the protocol, never a metric -- so the similarity signals are called
similarity (mean-max rectified cosine) and the Bloom signal is called cognitive
congruence.

Separate from 09_gen_paper_tables.py on purpose: the sweep has its own macro
namespace and its own audience, exactly like the ablation's
ablation/11_gen_ablation_tables.py. Formatters, size words and signal display
names are imported from 09 rather than copied.

Usage:
  python scripts/alignmentgraph-isd-bench/14_gen_sensitivity_table.py
  python scripts/alignmentgraph-isd-bench/14_gen_sensitivity_table.py \
      --pooled results/pooled_ladder.json --outdir results/generated
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[2]  # isd-agent-benchmark/


def _load(name: str, mod_name: str):
    """Import a digit-prefixed sibling script."""
    path = Path(__file__).resolve().with_name(name)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # @dataclass resolves via sys.modules
    spec.loader.exec_module(module)
    return module


TAB = _load("09_gen_paper_tables.py", "_tables")
fmt_num, fmt_p = TAB.fmt_num, TAB.fmt_p
size_word, size_display, size_of_label = TAB.size_word, TAB.size_display, TAB.size_of_label
PANEL_DISPLAY = TAB.PANEL_DISPLAY

#: Display names for the sensitivity design (guides section 8). Unknown keys
#: (ad-hoc arms) fall back to the key itself.
ENCODER_DISPLAY = {
    "nemotron": "E1 Nemotron-8B",
    "bgem3": "E2 bge-m3",
    "granite": "E3 granite-311m",
    "tfidf": "E4 TF-IDF",
}

#: Display names for the Bloom-classifier axis. bertbloom is the primary arm.
BLOOM_DISPLAY = {
    "bertbloom": "B1 BERT-Bloom",
    "lexicon": "B2 Lexicon",
}

#: LaTeX macro names cannot contain digits, so every arm key needs a word.
ARM_WORDS = {
    "nemotron": "Nemotron",
    "bgem3": "BgeMThree",
    "granite": "Granite",
    "tfidf": "Tfidf",
    "bertbloom": "BertBloom",
    "lexicon": "Lexicon",
}

#: Axes in reporting order. Each entry: macro token, prose name, and the key in
#: alignment_sensitivity holding that axis's signal (per-arm agreement_metric
#: wins over this; it is only the fallback for the primary reference row).
AXES = (
    ("encoder", "EncoderAxis", "Encoder axis", "metric"),
    ("bloom", "BloomAxis", "Bloom-classifier axis", "metric_bloom_axis"),
)

#: Quantity token in the per-cell macro names, so the level macro says which
#: signal it holds and the two axes cannot collide on the primary arm.
AXIS_QUANTITY = {"encoder": "Sim", "bloom": "Cong"}

#: Macro name for "how many instruments this axis compared". Named per axis
#: because "5 encoders" would mislabel the design: 4 encoders + 2 Bloom
#: classifiers, sharing the primary configuration.
AXIS_COUNT_MACRO = {
    "encoder": ("rqTwoSensNumEncoders",
                "encoder arms compared, including the primary encoder"),
    "bloom": ("rqTwoSensNumBloomClassifiers",
              "Bloom classifiers compared, including the primary (its encoder "
              "is the primary encoder, unchanged)"),
}

_DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def arm_word(key: str) -> str:
    """Digit-free CamelCase token for a macro name."""
    if key in ARM_WORDS:
        return ARM_WORDS[key]
    out = []
    for part in "".join(c if c.isalnum() else " " for c in key).split():
        out.append("".join(_DIGIT_WORDS.get(c, c) for c in part).capitalize())
    return "".join(out) or "Arm"


def arm_display(key: str, axis: str) -> str:
    table = BLOOM_DISPLAY if axis == "bloom" else ENCODER_DISPLAY
    return table.get(key, key)


def signal_note(metric: str) -> str:
    """What the operation actually is, for macro comments and notes."""
    if metric.endswith("_similarity"):
        return " (mean-max rectified cosine max(0,cos) on [0,1])"
    return ""


def header_comment(pooled: dict, what: str) -> str:
    return "\n".join([
        "% AUTO-GENERATED by isd-agent-benchmark/scripts/alignmentgraph-isd-bench/"
        "14_gen_sensitivity_table.py — DO NOT EDIT BY HAND",
        f"% {what}",
        f"% generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"% source pooled JSON generated_at: {pooled.get('generated_at', '?')}",
        "",
    ])


# ── data access ──────────────────────────────────────────────────────────────

def axis_blocks(pooled: dict) -> list[dict]:
    """One block per swept axis, each with its own signal and its own rows.

    A block's rows are the primary configuration (the reference the arms are
    compared against -- leaving it out would make the table unreadable; its rho
    column is '--' because an arm cannot disagree with itself) followed by that
    axis's sweep arms. Level and delta cells are read on the block's signal, so
    every number in a row comes from the measurement that row's swap can move.
    """
    sens = pooled["alignment_sensitivity"]
    labels = [m["label"] for m in pooled["models"]]
    primary_align = pooled.get("alignment")

    blocks = []
    for axis, macro_tok, axis_name, metric_key in AXES:
        arms = [e for e in sens["encoders"] if (e.get("axis") or "encoder") == axis]
        metric = sens.get(metric_key) or next(
            (e.get("agreement_metric") for e in arms if e.get("agreement_metric")), None)
        if metric is None or (axis != "encoder" and not arms):
            continue
        rows = []
        if primary_align:
            rows.append(_primary_row(pooled, sens, axis, metric, labels))
        for entry in arms:
            rows.append(_arm_row(pooled, entry, axis, metric, labels))
        blocks.append({
            "axis": axis,
            "macro_token": macro_tok,
            "name": axis_name,
            "metric": metric,
            "metric_display": PANEL_DISPLAY.get(metric, metric.replace("_", " ")),
            "rows": rows,
        })
    return blocks


def _primary_row(pooled: dict, sens: dict, axis: str, metric: str,
                 labels: list[str]) -> dict:
    """The primary configuration, read on this axis's signal."""
    align = pooled["alignment"]
    proposed = pooled["config"]["proposed"]
    baselines = pooled["config"]["baselines"]
    key = (sens["primary_encoder"] if axis == "encoder"
           else (align.get("bloom_classifier") or "primary"))
    cells = _cells(align.get("models"), None, metric, labels, proposed, baselines)
    ranks = _primary_ranks(align, labels, metric, [proposed] + baselines, proposed)
    for label in labels:
        cells[label]["rank"] = ranks.get(label)
        cells[label]["p_holm_max"] = _holm_max(align, label, metric)
    return {
        "key": key, "axis": axis, "is_primary": True,
        "label_text": arm_display(key, axis),
        "encoder_label": align.get("encoder_label"),
        "bloom_classifier": align.get("bloom_classifier"),
        "coverage": align.get("n_scenario_files"),
        "metric": metric, "cells": cells,
    }


def _arm_row(pooled: dict, entry: dict, axis: str, block_metric: str,
             labels: list[str]) -> dict:
    """One sweep arm. Its own agreement_metric wins over the block default."""
    proposed = pooled["config"]["proposed"]
    baselines = pooled["config"]["baselines"]
    metric = entry.get("agreement_metric") or block_metric
    base = {"key": entry["key"], "axis": axis, "is_primary": False,
            "label_text": arm_display(entry["key"], axis), "metric": metric}
    if entry.get("skipped"):
        return {**base, "skipped": entry["skipped"], "cells": {}}
    agree = {r["label"]: r
             for r in entry["agreement_with_primary"].get("per_model") or []}
    cells = _cells(entry.get("models"), agree, metric, labels, proposed, baselines)
    for label in labels:
        cells[label]["p_holm_max"] = _holm_max(entry, label, metric)
    return {**base,
            "encoder_label": entry.get("encoder_label"),
            "bloom_classifier": entry.get("bloom_classifier"),
            "coverage": (entry.get("coverage") or {}).get("n_scenario_files"),
            "cells": cells}


def _cells(models_map: dict | None, agree_rows: dict | None, metric: str,
           labels: list[str], proposed: str, baselines: list[str]) -> dict:
    cells = {}
    for label in labels:
        agents = (models_map or {}).get(label) or {}
        means = {a: agents.get(a, {}).get(metric, {}).get("mean")
                 for a in [proposed] + baselines}
        means = {a: v for a, v in means.items() if v is not None}
        best_b = max((b for b in baselines if b in means),
                     key=lambda b: means[b], default=None)
        row = (agree_rows or {}).get(label) or {}
        cells[label] = {
            "metric": metric,
            "mean": means.get(proposed),
            "delta_best": (means[proposed] - means[best_b]
                           if best_b and proposed in means else None),
            "best_baseline": best_b,
            "rho": row.get("spearman_rho_agent_means"),
            "tau": row.get("kendall_tau_b_agent_means"),
            "p_exact": row.get("spearman_p_exact"),
            "rank": row.get("proposed_rank_here"),
            "top1": row.get("top1_agent_here"),
            "top1_same": (row.get("top1_agent_here") == row.get("top1_agent_primary")
                          if row.get("top1_agent_here") else None),
            "scen_rho": row.get("spearman_rho_scenario_level_proposed"),
            "sign": row.get("sign_agreement_vs_baselines"),
            "p_holm_max": None,     # filled by the caller on the row's metric
        }
    return cells


def _primary_ranks(align: dict, labels: list[str], metric: str,
                   agents: list[str], proposed: str) -> dict[str, int | None]:
    out = {}
    for label in labels:
        per_agent = (align.get("models") or {}).get(label) or {}
        means = {a: per_agent.get(a, {}).get(metric, {}).get("mean")
                 for a in agents}
        means = {a: v for a, v in means.items() if v is not None}
        out[label] = (sorted(means, key=lambda a: -means[a]).index(proposed) + 1
                      if proposed in means else None)
    return out


def _holm_max(stats_holder: dict, label: str, metric: str) -> float | None:
    """Worst within-signal Holm p among that size's paired comparisons.

    Read on the row's OWN signal: the panel is keyed by signal, and a row must
    not quote the p-values of a signal its swap cannot move.
    """
    for entry in (stats_holder.get("stats") or {}).get("per_model") or []:
        if entry.get("label") == label:
            panel = (entry.get("panel") or {}).get(metric) or {}
            ps = [r["p_holm"] for r in panel.get("comparisons") or [] if "p_holm" in r]
            return max(ps) if ps else None
    return None


# ── per-axis aggregates (recomputed here, per axis, from pooled fields) ───────

def axis_summary(block: dict) -> dict:
    """Worst-case agreement and worst Holm p WITHIN one axis, over its arms.

    Sweep arms only, like the pooled summary: the question is whether a verdict
    survives the swap, so the primary's own p-values do not belong in it (they
    are available per row as rqTwoSensPHolmMax<PrimaryArm>). pooled summary's
    max_p_holm_* reads the encoder-axis signal for every arm including the Bloom
    arm, so the per-axis worst case is recomputed here on each row's own signal.
    """
    arm_cells = [c for r in block["rows"] if not r.get("is_primary")
                 for c in r["cells"].values()]
    rhos = [c["rho"] for c in arm_cells if _is_num(c.get("rho"))]
    taus = [c["tau"] for c in arm_cells if _is_num(c.get("tau"))]
    holms = [c["p_holm_max"] for c in arm_cells if c.get("p_holm_max") is not None]
    return {
        "n_arms": sum(1 for r in block["rows"]
                      if not r.get("is_primary") and "skipped" not in r),
        "n_skipped": sum(1 for r in block["rows"] if "skipped" in r),
        "rho_min": min(rhos) if rhos else None,
        "tau_min": min(taus) if taus else None,
        "p_holm_max": max(holms) if holms else None,
        "n_cells": len(arm_cells),
        "n_cells_first": sum(1 for c in arm_cells if c.get("rank") == 1),
        "top1_stable": (all(c.get("top1_same") for c in arm_cells if c.get("top1"))
                        if arm_cells else None),
    }


def _is_num(x) -> bool:
    return isinstance(x, float) and not math.isnan(x)


# ── tab_sensitivity.tex ──────────────────────────────────────────────────────

def gen_table(pooled: dict, blocks: list[dict]) -> str:
    labels = [m["label"] for m in pooled["models"]]
    proposed_disp = TAB.AGENT_DISPLAY.get(pooled["config"]["proposed"],
                                          pooled["config"]["proposed"])
    ncols = 1 + 3 * len(labels)
    colspec = "@{}l" + "r@{\\hskip 3pt}r@{\\hskip 3pt}c" * len(labels) + "@{}"
    head = [
        header_comment(pooled, "Alignment instrument sensitivity, one block per swept "
                               "axis: level on that axis's score, rank agreement "
                               "with the primary arm, and the proposed agent's "
                               "rank, per model size."),
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{" + colspec + "}",
        "\\toprule",
        "Arm & " + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{Qwen3.5-{size_display(lb)}}}" for lb in labels
        ) + " \\\\",
        " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                 for i in range(len(labels))),
        " & " + " & ".join("Mean & $\\rho$ & h" for _ in labels) + " \\\\",
    ]
    body = []
    for block in blocks:
        body.append("\\midrule")
        body.append(f"\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\emph{{{block['name']}}} "
                    f"--- signal: {block['metric_display']}}} \\\\")
        for row in block["rows"]:
            name = row["label_text"]
            if row.get("is_primary"):
                name = f"\\textbf{{{name}}} (primary)"
            if "skipped" in row:
                body.append(f"{name} & " + " & ".join(
                    "\\multicolumn{3}{c}{--}" for _ in labels) + " \\\\")
                continue
            cells = []
            for label in labels:
                c = row["cells"].get(label, {})
                mean = fmt_num(c.get("mean"), 3) if c.get("mean") is not None else "--"
                rho = ("--" if row.get("is_primary")
                       else fmt_num(c["rho"], 3) if _is_num(c.get("rho")) else "--")
                rank = str(c["rank"]) if c.get("rank") else "--"
                cells += [mean, rho, rank]
            body.append(f"{name} & " + " & ".join(cells) + " \\\\")
    tail = [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Instrument sensitivity: the same runs re-scored by each arm, "
        "one block per swept axis. Per size, \\emph{Mean} = " + proposed_disp +
        "'s mean on that block's signal, $\\rho$ = Spearman of the per-agent "
        "mean vector vs.\\ the primary arm, \\emph{h} = its rank.}",
        "\\label{tab:rq2-sensitivity}",
        "\\end{table*}",
    ]
    return "\n".join(head + body + tail) + "\n"


# ── sens_macros.tex ──────────────────────────────────────────────────────────

def gen_macros(pooled: dict, blocks: list[dict]) -> str:
    sens = pooled["alignment_sensitivity"]
    labels = [m["label"] for m in pooled["models"]]
    tex = [header_comment(
        pooled,
        "Alignment instrument-sensitivity macros (encoder axis + Bloom-classifier "
        "axis); each axis is reported on the score it can move, and scores are "
        "named after the operation they compute — no primary endpoint.")]

    def newcmd(name: str, value: str, desc: str) -> None:
        tex.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {desc}")

    # ── design counts, per axis (the sweep is a star of arms, not 5 encoders)
    summary = sens["summary"]
    by_axis = {b["axis"]: axis_summary(b) for b in blocks}
    n_arms = summary["n_encoders"] + 1
    newcmd("rqTwoSensNumArms", str(n_arms),
           "sensitivity arms: the primary configuration + its single-axis deviations")
    for axis, tok, axis_name, _ in AXES:
        block = next((b for b in blocks if b["axis"] == axis), None)
        if block is None:
            continue
        n_rows = sum(1 for r in block["rows"] if "skipped" not in r)
        macro, desc = AXIS_COUNT_MACRO.get(
            axis, (f"rqTwoSensNum{tok}Arms",
                   f"arms compared on the {axis_name.lower()}, incl. the primary"))
        newcmd(macro, str(n_rows), desc)
        newcmd(f"rqTwoSensSignal{tok}", block["metric_display"],
               f"{axis_name} signal ({block['metric']})"
               f"{signal_note(block['metric'])}")
    newcmd("rqTwoSensPrimaryName",
           ENCODER_DISPLAY.get(sens["primary_encoder"], sens["primary_encoder"]),
           "primary encoder display name")
    primary_bloom = (pooled.get("alignment") or {}).get("bloom_classifier")
    if primary_bloom:
        newcmd("rqTwoSensPrimaryBloomName", BLOOM_DISPLAY.get(primary_bloom,
                                                             primary_bloom),
               "primary Bloom-classifier display name")

    # ── cross-axis worst cases (each arm on its own signal)
    for key, macro, desc in (
        ("min_spearman_rho_any_encoder_any_size", "rqTwoSensRhoMinAll",
         "min Spearman rho over every arm x size cell (each arm on its own axis signal)"),
        ("min_kendall_tau_b_any_encoder_any_size", "rqTwoSensTauMinAll",
         "min Kendall tau-b over every arm x size cell (each arm on its own axis signal)"),
    ):
        val = summary.get(key)
        newcmd(macro, fmt_num(val, 3) if _is_num(val) else "--", desc)
    holms = [s["p_holm_max"] for s in by_axis.values() if s["p_holm_max"] is not None]
    if holms:
        newcmd("rqTwoSensPHolmMaxAll", fmt_p(max(holms)),
               "worst within-signal Holm p over the sweep arms of both axes, each "
               "read on its own signal (recomputed per axis here, not taken from "
               "summary.max_p_holm_any_encoder_any_size, which reads the "
               "encoder-axis signal for every arm)")

    # ── per-axis worst cases
    for block in blocks:
        tok, s = block["macro_token"], by_axis[block["axis"]]
        low = block["name"].lower()
        if s["rho_min"] is not None:
            newcmd(f"rqTwoSensRhoMin{tok}", fmt_num(s["rho_min"], 3),
                   f"min Spearman rho vs primary over the {low} arms x sizes")
        if s["tau_min"] is not None:
            newcmd(f"rqTwoSensTauMin{tok}", fmt_num(s["tau_min"], 3),
                   f"min Kendall tau-b vs primary over the {low} arms x sizes")
        if s["p_holm_max"] is not None:
            newcmd(f"rqTwoSensPHolmMax{tok}", fmt_p(s["p_holm_max"]),
                   f"worst within-signal Holm p on the {low} "
                   f"({block['metric']}), sweep arms only")
        newcmd(f"rqTwoSensTopOneCells{tok}", f"{s['n_cells_first']}/{s['n_cells']}",
               f"{low} arm x size cells where the proposed agent ranks first "
               "(sweep arms only; the primary is the reference)")

    # ── per-arm, per-size cells
    for block in blocks:
        metric, q = block["metric"], AXIS_QUANTITY.get(block["axis"], "Level")
        note = signal_note(metric)
        for row in block["rows"]:
            if "skipped" in row:
                continue
            w = arm_word(row["key"])
            where = f"arm {row['key']}, {block['name'].lower()}"
            rhos = [c["rho"] for c in row["cells"].values() if _is_num(c.get("rho"))]
            if rhos:
                newcmd(f"rqTwoSensRhoMin{w}", fmt_num(min(rhos), 3),
                       f"min Spearman rho vs primary across sizes ({where})")
            arm_holms = [c["p_holm_max"] for c in row["cells"].values()
                         if c.get("p_holm_max") is not None]
            if arm_holms:
                newcmd(f"rqTwoSensPHolmMax{w}", fmt_p(max(arm_holms)),
                       f"worst within-signal Holm p on {metric} across sizes ({where})")
            for i, label in enumerate(labels):
                c = row["cells"].get(label) or {}
                sw = size_word(label, i)
                if c.get("mean") is not None:
                    newcmd(f"rqTwoSens{q}{w}{sw}", fmt_num(c["mean"], 3),
                           f"proposed mean {metric}{note} @ {label} ({where})")
                if c.get("delta_best") is not None:
                    newcmd(f"rqTwoSens{q}DeltaBest{w}{sw}", fmt_num(c["delta_best"], 3),
                           f"proposed - best baseline on {metric} @ {label} ({where})")
                if _is_num(c.get("rho")):
                    newcmd(f"rqTwoSensRho{w}{sw}", fmt_num(c["rho"], 3),
                           f"Spearman rho vs primary on {metric} @ {label} ({where})")
                if _is_num(c.get("tau")):
                    newcmd(f"rqTwoSensTau{w}{sw}", fmt_num(c["tau"], 3),
                           f"Kendall tau-b vs primary on {metric} @ {label} ({where})")
    return "\n".join(tex) + "\n"


# ── stats_sensitivity.md ─────────────────────────────────────────────────────

def gen_stats_md(pooled: dict, blocks: list[dict]) -> str:
    sens = pooled["alignment_sensitivity"]
    summary = sens["summary"]
    labels = [m["label"] for m in pooled["models"]]
    primary_bloom = (pooled.get("alignment") or {}).get("bloom_classifier")
    axis_line = "; ".join(f"{b['name']} on `{b['metric']}`" for b in blocks)
    out = [
        "<!-- AUTO-GENERATED by isd-agent-benchmark/scripts/alignmentgraph-isd-bench/"
        "14_gen_sensitivity_table.py — DO NOT EDIT BY HAND -->",
        "# Alignment instrument sensitivity (star of single-axis arms)",
        "",
        f"- source pooled JSON generated_at: `{pooled.get('generated_at', '?')}`",
        f"- arms: {summary['n_encoders'] + 1} = the primary configuration "
        f"(encoder `{sens['primary_encoder']}` + Bloom `{primary_bloom}`) plus "
        f"{summary['n_encoders']} single-axis deviations "
        f"(skipped: {summary['n_skipped']}).",
        f"- each axis is reported on a score that axis can move: {axis_line}. "
        "Read from the pooled per-arm `axis` / `agreement_metric`, not assumed: "
        "a Bloom swap cannot move a cosine and an encoder swap cannot move a "
        "Bloom level.",
        "- The alignment panel is flat with NO primary endpoint; the two scores "
        "below are each axis's sensitivity lead, not an endpoint.",
        "- similarity signals are mean-max rectified cosine, `max(0, cos)` on "
        "[0,1]; the measurement has no similarity threshold anywhere.",
        f"- pooled `definition` string, verbatim{'' if all(b['metric'] in sens['definition'] for b in blocks) else ' (it names only the encoder axis, so read the per-axis lines above over it)'}: {sens['definition']}",
        "",
    ]
    for block in blocks:
        s = axis_summary(block)
        out += [
            f"## {block['name']} — signal `{block['metric']}`"
            f"{signal_note(block['metric'])}",
            "",
            f"- sweep arms: {s['n_arms']} (skipped: {s['n_skipped']}); the primary "
            "row is the reference the arms are compared against.",
            "",
        ]
        for row in block["rows"]:
            tag = " (primary)" if row.get("is_primary") else ""
            if "skipped" in row:
                out += [f"### {row['label_text']} — SKIPPED", "",
                        f"{row['skipped']}", ""]
                continue
            out += [
                f"### {row['label_text']}{tag}",
                "",
                f"- axis: `{row['axis']}` · signal: `{row['metric']}`",
                f"- encoder in the artifacts: `{row.get('encoder_label')}` · "
                f"Bloom classifier: `{row.get('bloom_classifier')}`",
                f"- coverage: {row.get('coverage')} scenario score file(s)",
                "",
                f"| Size | proposed mean ({row['metric']}) | Δ vs best baseline "
                "| rank | ρ vs primary | τ-b | exact p | top-1 | scenario ρ "
                "| sign/Holm agree | worst Holm p |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for label in labels:
                c = row["cells"].get(label) or {}
                sign = c.get("sign") or {}
                sign_txt = (f"{sign.get('n_same_sign')}/{sign.get('n')} · "
                            f"{sign.get('n_same_holm_verdict_at_05')}/{sign.get('n')}"
                            if sign.get("n") else "--")
                top1 = c.get("top1") or "--"
                if c.get("top1_same") is False:
                    top1 += " (DIFFERS)"
                out.append(
                    f"| {label} | {_num(c.get('mean'), 3)} | {_num(c.get('delta_best'), 3)} "
                    f"| {c.get('rank') or '--'} | {_num(c.get('rho'), 3)} "
                    f"| {_num(c.get('tau'), 3)} | {_sci(c.get('p_exact'))} | {top1} "
                    f"| {_num(c.get('scen_rho'), 3)} | {sign_txt} "
                    f"| {_sci(c.get('p_holm_max'))} |"
                )
            out.append("")
        out += [
            f"**{block['name']} headline**",
            "",
            f"- min Spearman ρ over every arm × size: {_num(s['rho_min'], 3)}",
            f"- min Kendall τ-b: {_num(s['tau_min'], 3)}",
            f"- top-1 agent identical to the primary in every cell: {s['top1_stable']}",
            f"- proposed agent ranks first in {s['n_cells_first']}/{s['n_cells']} "
            "arm cells (sweep arms only)",
            f"- worst within-signal Holm p on `{block['metric']}` over the sweep "
            f"arms: {_sci(s['p_holm_max'])} (the primary row's own worst Holm p "
            "is in its table above, last column)",
            "",
        ]
    holms = [axis_summary(b)["p_holm_max"] for b in blocks]
    holms = [h for h in holms if h is not None]
    p_max_here = max(holms) if holms else None
    p_max_pooled = summary.get("max_p_holm_any_encoder_any_size")
    out += [
        "## Across axes",
        "",
        f"- min Spearman ρ over every arm × size, each arm on its own signal: "
        f"{_num(summary.get('min_spearman_rho_any_encoder_any_size'), 3)}",
        f"- min Kendall τ-b, same pooling: "
        f"{_num(summary.get('min_kendall_tau_b_any_encoder_any_size'), 3)}",
        f"- proposed agent ranks first in "
        f"{summary['n_cells_proposed_ranks_first']}/{summary['n_cells']} arm cells "
        "over both axes",
        f"- worst within-signal Holm p over the sweep arms of both axes, each "
        f"read on its own signal: {_sci(p_max_here)}",
        (f"- pooled `summary.max_p_holm_any_encoder_any_size` = "
         f"{_sci(p_max_pooled)} DIVERGES from the line above: that field reads "
         f"`{sens['metric']}` for every arm, so on the Bloom axis it quotes a "
         "signal the arm cannot move. Cite the per-axis numbers."
         if p_max_pooled is not None and p_max_here is not None
         and abs(p_max_pooled - p_max_here) > 1e-12 else
         f"- pooled `summary.max_p_holm_any_encoder_any_size` = "
         f"{_sci(p_max_pooled)}, which agrees with the per-axis recomputation."),
        "",
        "Sign/Holm agree = of the paired proposed-vs-baseline comparisons at that "
        "size, how many keep the same sign, and how many keep the same "
        "significant/not verdict at α=0.05, when the instrument is swapped. "
        "Worst Holm p = the largest within-signal Holm-corrected p among that "
        "size's paired comparisons on the row's own signal.",
        "",
    ]
    return "\n".join(out)


def _num(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def _sci(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    return f"{p:.2e}"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the RQ2 instrument-sensitivity table, macros and "
                    "stats dump.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pooled", type=Path,
                        default=BENCH_ROOT / "results" / "pooled_ladder.json")
    parser.add_argument("--outdir", type=Path,
                        default=BENCH_ROOT / "results" / "generated")
    args = parser.parse_args()

    if not args.pooled.exists():
        print(f"ERROR: {args.pooled} not found — run 07_pool_ladder_runs.py first.",
              file=sys.stderr)
        return 1
    pooled = json.loads(args.pooled.read_text(encoding="utf-8"))
    if not pooled.get("alignment_sensitivity"):
        print(f"ERROR: {args.pooled} has no 'alignment_sensitivity'. Score the sweep "
              "and re-pool:\n"
              "  python scripts/alignmentgraph-isd-bench/06_score_alignment.py "
              "--all-arms --overwrite <run dirs>\n"
              "  python scripts/alignmentgraph-isd-bench/07_pool_ladder_runs.py "
              "--auto-glob '...' --auto-encoders --auto-bloom", file=sys.stderr)
        return 1

    blocks = axis_blocks(pooled)
    if not blocks:
        print("ERROR: no sensitivity axis could be resolved from "
              "alignment_sensitivity (no metric and no arms).", file=sys.stderr)
        return 1
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("tab_sensitivity.tex", gen_table(pooled, blocks)),
        ("sens_macros.tex", gen_macros(pooled, blocks)),
        ("stats_sensitivity.md", gen_stats_md(pooled, blocks)),
    ):
        (args.outdir / name).write_text(text, encoding="utf-8")
        print(f"wrote {args.outdir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
