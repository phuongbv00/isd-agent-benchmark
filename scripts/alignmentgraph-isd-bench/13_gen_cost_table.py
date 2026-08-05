#!/usr/bin/env python3
"""Generate the cost / deployability profile from the ladder run artifacts.

Reads the per-scenario ``*_trajectory.json`` files of the ladder runs directly
(no pooled JSON, no LLM call, no benchmark re-run) and writes into
results/generated/ (benchmark-local; sync into the thesis paper tree with
docs/scripts/sync_generated.sh from the thesis repo root):

  tab_cost.tex     booktabs table* — agents x 4 model sizes, cell = median
                   per-scenario total tokens (k), model calls, wall-clock
                   seconds, plus the (scenario, run) coverage n
  cost_macros.tex  \\newcommand macros for every cost number prose will cite
  stats_cost.md    human-readable full grid (prompt/completion/total tokens,
                   calls, wall-clock; median + IQR + mean), the coverage
                   roster with failure causes, and the caveats

WHY THIS EXISTS. The paper motivates on-premise deployment (inference cost, API
independence, data control) and otherwise reports no resource number at all.
This script closes that loop from data already on disk.

Relation to what already exists: 07_pool_ladder_runs.py::pool_tokens puts a
mean/SD token layer into pooled_ladder.json, which 10_gen_paper_figures.py plots
as fig_cost_quality. Nothing tabulates it, and that layer carries no model-call
count, no wall-clock caveat, no coverage roster and no serving profile — which is
what a deployability claim actually needs. This script reads the run dirs itself
so the medians it prints are reproducible from the artifacts without the pooled
file; the means it reports agree with pool_tokens to the last digit.

THREE THINGS THIS SCRIPT REFUSES TO DO SILENTLY.

1. ``metadata.total_tokens`` at the top level is unreliable — most agents leave
   it 0 while their real accounting sits in ``metadata.token_usage``. Only
   ``token_usage`` is read; a trajectory without it is counted as
   ``n_no_token_usage`` and excluded from token statistics rather than folded
   in as a zero.

2. ``metadata.cost_usd`` is fiction on a self-hosted ladder (one agent even
   reports a per-scenario dollar figure computed from a hosted-API price list).
   It is never read. Any monetary number in the output is derived here from an
   explicitly labelled assumption block (SERVING_PROFILE) and carries "assumed"
   in its macro name.

3. Cost is conditional on the agent having produced an artifact. An agent that
   crashes leaves no trajectory, so its surviving runs are a biased-cheap
   sample. Every cell prints its n, cells below COVERAGE_FLAG are daggered, and
   the missing cells are classified by failure cause (the token-cap truncation
   crash vs everything else) in stats_cost.md.

Artifact text is ENGLISH and numbers use a decimal POINT (87.77).

Usage:
    python scripts/alignmentgraph-isd-bench/13_gen_cost_table.py
    python scripts/alignmentgraph-isd-bench/13_gen_cost_table.py \\
        --runs-glob 'results/test_90_benchmark_*_r*' --outdir results/generated
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[2]  # isd-agent-benchmark/
DEFAULT_OUTDIR = BENCH_ROOT / "results" / "generated"
DEFAULT_RUNS_GLOB = "results/test_90_benchmark_*_r*"

#: Display names, mirroring 09_gen_paper_tables.py.
AGENT_DISPLAY = {
    "baseline": "Baseline",
    "eduplanner": "EduPlanner",
    "addie-agent": "ADDIE-Agent",
    "rpisd-agent": "RPISD-Agent",
    "dick-carey-agent": "Dick-Carey-Agent",
    "react-isd": "ReAct-ISD",
    "alignmentgraph-isd": "AlignmentGraph-ISD",
    "alignmentgraph-isd-no-verifier": "no verifier",
    "alignmentgraph-isd-no-graph-ctx": "no graph context",
    "alignmentgraph-isd-skeleton": "neither (skeleton)",
}
AGENT_ORDER = [
    "baseline", "eduplanner", "addie-agent", "rpisd-agent",
    "dick-carey-agent", "react-isd", "alignmentgraph-isd",
]
#: Rendered as a separate block under a midrule: they are not baselines, and
#: their cost is what quantifies the marginal price of each harness component.
ABLATION_ORDER = [
    "alignmentgraph-isd-no-verifier",
    "alignmentgraph-isd-no-graph-ctx",
    "alignmentgraph-isd-skeleton",
]
PROPOSED = "alignmentgraph-isd"

#: A cell whose coverage falls below this is daggered in the table: its cost
#: numbers are conditional on the subset of scenarios the agent survived.
COVERAGE_FLAG = 0.95

# parsed size (in B params) -> digit-free macro word (mirrors 09's SIZE_WORDS)
SIZE_WORDS = {0.8: "ZeroEightB", 2.0: "TwoB", 4.0: "FourB", 9.0: "NineB"}
FALLBACK_WORDS = ["SizeA", "SizeB", "SizeC", "SizeD", "SizeE", "SizeF"]

#: SERVING ASSUMPTION BLOCK — NOT measured by this script.
#:
#: GPU class and VRAM: docs/benchmark_guides.md section 3.1 (slot -> GPU table)
#: and the RunPod pod templates that served the ladder (bf16, --max-model-len
#: 16384, --gpu-memory-utilization 0.95). The hourly figures are RunPod on-demand
#: SECURE-CLOUD LIST PRICES recorded at deploy time (2026-07); they are a price
#: quote, not a measurement, and they move. Every number derived from them
#: carries "Assumed" in its macro name and "assumption" in its prose label.
SERVING_PROFILE = {
    0.8: {"gpu": "RTX 4090", "vram_gb": 24, "usd_per_hour": 0.69},
    2.0: {"gpu": "RTX 4090", "vram_gb": 24, "usd_per_hour": 0.69},
    4.0: {"gpu": "RTX 4090", "vram_gb": 24, "usd_per_hour": 0.69},
    9.0: {"gpu": "RTX PRO 4500 Blackwell", "vram_gb": 32, "usd_per_hour": 0.74},
}
SERVING_SOURCE = ("docs/benchmark_guides.md section 3--4 (GPU per slot, bf16, "
                  "16k context) + RunPod on-demand list price recorded at "
                  "deploy time, 2026-07")

#: Signature of the crash that the shared per-call token cap produces: the
#: OpenAI client raises when a structured-output response is cut at the cap.
#: Counting it separately matters because an agent that dies this way is not
#: "cheap", it is truncated — see review_q2.md R4.
TRUNCATION_MARKER = "LengthFinishReasonError"


# -- formatting ---------------------------------------------------------------

def fmt_num(x, nd: int = 2) -> str:
    """87.77 -> '87.77'. Artifacts are English, so the decimal mark is a point."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


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
    return f"{s:g}B"


def quantiles(xs: list[float]) -> tuple[float, float, float, float]:
    """(median, p25, p75, mean) — plain order statistics, no interpolation fuss."""
    if not xs:
        nan = float("nan")
        return nan, nan, nan, nan
    s = sorted(xs)
    if len(s) == 1:
        return s[0], s[0], s[0], s[0]
    p25 = statistics.quantiles(s, n=4, method="inclusive")[0]
    p75 = statistics.quantiles(s, n=4, method="inclusive")[2]
    return statistics.median(s), p25, p75, statistics.fmean(s)


# -- collection ---------------------------------------------------------------

def parse_elapsed_seconds(log_path: Path) -> float | None:
    """'Total elapsed: 1h 11m' -> 4260.0. None if the run log has no such line."""
    if not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Total elapsed:\s*(?:(\d+)h\s*)?(?:(\d+)m)?(?:\s*(\d+)s)?", txt)
    if not m or not any(m.groups()):
        return None
    h, mi, se = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + se


def collect(run_dirs: list[Path]) -> dict:
    """Walk every trajectory once and build the per (agent, size) observation lists.

    Returns a dict with:
      obs[size_label][agent] -> list of per-(scenario, run) records
      cells[size_label]      -> total (scenario, run) cells at that size
      missing[size][agent]   -> {"total": int, "truncation": int, "other": int}
      runs                   -> per-run provenance (elapsed, endpoints, workers)
    """
    obs: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    cells: dict[str, int] = defaultdict(int)
    missing: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "truncation": 0, "other": 0}))
    no_tu: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    runs: list[dict] = []
    size_labels: dict[str, str] = {}

    for rd in run_dirs:
        summary_path = rd / "benchmark_summary.json"
        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}
        model_name = (summary.get("model") or {}).get("name") or rd.name
        # Label in the same shape 07/09 use ("qwen3.5-2b") so size_display works.
        label = model_name.split("/")[-1].lower()
        size_labels[label] = label

        cfg = summary.get("config") or {}
        endpoints = (summary.get("model") or {}).get("base_urls") or []
        elapsed = parse_elapsed_seconds(rd / "benchmark_progress.log")

        dataset_dirs = [p for p in rd.iterdir() if p.is_dir() and p.name != "generated"]
        n_traj_in_run = 0
        for ds in dataset_dirs:
            scen_dirs = sorted(p for p in ds.iterdir() if p.is_dir())
            cells[label] += len(scen_dirs)
            for sc in scen_dirs:
                for traj in sorted(sc.glob("*_trajectory.json")):
                    agent = traj.name[: -len("_trajectory.json")]
                    n_traj_in_run += 1
                    try:
                        j = json.loads(traj.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        no_tu[label][agent] += 1
                        continue
                    md = j.get("metadata") or {}
                    tu = md.get("token_usage")
                    rec = {
                        "scenario": sc.name,
                        "run": rd.name,
                        # NOTE: metadata.total_tokens is deliberately NOT read.
                        "wall": md.get("execution_time_seconds"),
                    }
                    if isinstance(tu, dict) and tu.get("total_tokens"):
                        rec.update(
                            prompt=tu.get("prompt_tokens"),
                            completion=tu.get("completion_tokens"),
                            total=tu.get("total_tokens"),
                            calls=tu.get("llm_calls"),
                        )
                    else:
                        no_tu[label][agent] += 1
                    obs[label][agent].append(rec)
                # Missing cells: a *_log.txt with no sibling trajectory is a
                # crash. Classify it, because "no artifact" and "cheap" look
                # identical in a token table otherwise.
                for lg in sorted(sc.glob("*_log.txt")):
                    agent = lg.name[: -len("_log.txt")]
                    if (sc / f"{agent}_trajectory.json").exists():
                        continue
                    bucket = missing[label][agent]
                    bucket["total"] += 1
                    txt = lg.read_text(encoding="utf-8", errors="ignore")
                    if TRUNCATION_MARKER in txt:
                        bucket["truncation"] += 1
                    else:
                        bucket["other"] += 1

        runs.append({
            "dir": rd.name,
            "label": label,
            "elapsed_seconds": elapsed,
            "n_endpoints": len(endpoints),
            "n_trajectories": n_traj_in_run,
            "max_workers": cfg.get("max_workers"),
            "scenario_max_workers": cfg.get("scenario_max_workers"),
            "agents": cfg.get("agents") or [],
        })

    return {
        "obs": obs, "cells": cells, "missing": missing, "no_token_usage": no_tu,
        "runs": runs,
        "size_labels": sorted(size_labels, key=lambda lb: size_of_label(lb)),
    }


def summarise(collected: dict) -> dict:
    """obs -> per (size, agent) statistics dict."""
    out: dict[str, dict[str, dict]] = {}
    for label in collected["size_labels"]:
        out[label] = {}
        total_cells = collected["cells"][label]
        for agent, recs in collected["obs"][label].items():
            tok = [r["total"] for r in recs if r.get("total")]
            pro = [r["prompt"] for r in recs if r.get("prompt")]
            com = [r["completion"] for r in recs if r.get("completion")]
            cal = [r["calls"] for r in recs if r.get("calls")]
            wal = [r["wall"] for r in recs if r.get("wall") is not None]
            miss = collected["missing"][label].get(
                agent, {"total": 0, "truncation": 0, "other": 0})
            stat = {
                "n_cells_with_trajectory": len(recs),
                "n_cells_total": total_cells,
                "coverage": len(recs) / total_cells if total_cells else float("nan"),
                "n_missing": miss["total"],
                "n_missing_truncation": miss["truncation"],
                "n_missing_other": miss["other"],
                "n_no_token_usage": collected["no_token_usage"][label].get(agent, 0),
                "sum_total_tokens": sum(tok),
            }
            for key, xs in (("total_tokens", tok), ("prompt_tokens", pro),
                            ("completion_tokens", com), ("llm_calls", cal),
                            ("wall_seconds", wal)):
                med, p25, p75, mean = quantiles([float(x) for x in xs])
                stat[key] = {"n": len(xs), "median": med, "p25": p25,
                             "p75": p75, "mean": mean}
            out[label][agent] = stat
    return out


def ratios(stats: dict, size_labels: list[str], baselines: list[str]) -> dict:
    """Proposed-vs-baseline cost multipliers, per size.

    Reported on medians, the same statistic the table prints, so a reader can
    reproduce the multiplier from two visible cells.
    """
    out = {}
    for lb in sorted(size_labels, key=size_of_label):
        row = stats.get(lb, {})
        prop = row.get(PROPOSED)
        if not prop:
            continue
        entry = {"per_baseline": {}}
        for b in baselines:
            s = row.get(b)
            if not s:
                continue
            e = {}
            for key in ("total_tokens", "llm_calls", "wall_seconds"):
                den = s[key]["median"]
                num = prop[key]["median"]
                e[key] = num / den if den else float("nan")
            e["coverage"] = s["coverage"]
            entry["per_baseline"][b] = e
        for key in ("total_tokens", "llm_calls"):
            vals = {b: e[key] for b, e in entry["per_baseline"].items()
                    if not math.isnan(e[key])}
            if not vals:
                continue
            cheapest = min(vals, key=lambda b: row[b][key]["median"])
            entry[f"{key}_vs_cheapest"] = {"baseline": cheapest, "ratio": vals[cheapest],
                                           "coverage": row[cheapest]["coverage"]}
            # The cheapest baseline overall is sometimes an agent that crashed
            # on most scenarios, whose median is a survivors-only figure. The
            # honest headline multiplier is against the cheapest baseline that
            # actually completed the benchmark, so both are computed.
            full = {b: v for b, v in vals.items()
                    if row[b]["coverage"] >= COVERAGE_FLAG}
            if full:
                cf = min(full, key=lambda b: row[b][key]["median"])
                entry[f"{key}_vs_cheapest_full_coverage"] = {
                    "baseline": cf, "ratio": full[cf],
                    "coverage": row[cf]["coverage"]}
            entry[f"{key}_vs_median_baseline"] = {
                "ratio": statistics.median(sorted(vals.values()))}
            entry[f"{key}_max"] = {"baseline": max(vals, key=vals.get),
                                   "ratio": max(vals.values())}
            entry[f"{key}_min"] = {"baseline": min(vals, key=vals.get),
                                   "ratio": min(vals.values())}
        # Marginal cost of each harness component, from the ablation arms that
        # ran inside the same ladder: what an operator pays for the verifier and
        # for graph-context prompt injection.
        comp = {}
        for arm in ABLATION_ORDER:
            s = row.get(arm)
            if not s:
                continue
            den = s["total_tokens"]["median"]
            comp[arm] = {
                "token_overhead_pct": (100.0 * (prop["total_tokens"]["median"] / den - 1.0)
                                       if den else float("nan")),
                "prompt_overhead_pct": (
                    100.0 * (prop["prompt_tokens"]["median"]
                             / s["prompt_tokens"]["median"] - 1.0)
                    if s["prompt_tokens"]["median"] else float("nan")),
            }
        entry["components"] = comp
        out[lb] = entry
    return out


def serving_cost(collected: dict, stats: dict) -> dict:
    """ASSUMPTION-DERIVED GPU cost per produced design, per size.

    Measured inputs: wall-clock of each whole run (benchmark_progress.log), the
    number of vLLM endpoints the run round-robined over (benchmark_summary.json),
    and how many trajectories the run produced.
    Assumed input: the hourly price of the GPU class in SERVING_PROFILE.

    The pod-hours figure is what an operator is BILLED: it includes the judging
    phase, during which the pods idle. Attribution to a single agent is pro-rata
    by that agent's share of the size's total tokens — a stated assumption, not a
    measurement, since 10 agents shared one set of pods.
    """
    per_size: dict[str, dict] = {}
    for r in collected["runs"]:
        lb = r["label"]
        d = per_size.setdefault(lb, {"pod_hours": 0.0, "designs": 0, "runs": 0,
                                     "elapsed_hours": 0.0, "n_endpoints": set()})
        if r["elapsed_seconds"] is None or not r["n_endpoints"]:
            continue
        d["pod_hours"] += r["elapsed_seconds"] / 3600.0 * r["n_endpoints"]
        d["elapsed_hours"] += r["elapsed_seconds"] / 3600.0
        d["designs"] += r["n_trajectories"]
        d["runs"] += 1
        d["n_endpoints"].add(r["n_endpoints"])

    out = {}
    for lb, d in per_size.items():
        size = size_of_label(lb)
        prof = SERVING_PROFILE.get(size)
        if not prof or not d["designs"]:
            continue
        usd = d["pod_hours"] * prof["usd_per_hour"]
        row = stats.get(lb, {})
        tok_total = sum(s["sum_total_tokens"] for s in row.values())
        per_agent = {}
        for a, s in row.items():
            share = s["sum_total_tokens"] / tok_total if tok_total else float("nan")
            n = s["n_cells_with_trajectory"]
            per_agent[a] = {
                "token_share": share,
                "usd_per_design_assumed": (usd * share / n) if n else float("nan"),
            }
        out[lb] = {
            "gpu": prof["gpu"], "vram_gb": prof["vram_gb"],
            "usd_per_hour_assumed": prof["usd_per_hour"],
            "n_endpoints": sorted(d["n_endpoints"]),
            "n_runs": d["runs"],
            "pod_hours_measured": d["pod_hours"],
            "wall_hours_measured": d["elapsed_hours"],
            "designs_produced": d["designs"],
            "usd_total_assumed": usd,
            "usd_per_design_all_agents_assumed": usd / d["designs"],
            "per_agent": per_agent,
        }
    return out


# -- generators ---------------------------------------------------------------

def header_comment(what: str, run_dirs: list[Path]) -> str:
    lines = [
        "% AUTO-GENERATED by isd-agent-benchmark/scripts/alignmentgraph-isd-bench/"
        "13_gen_cost_table.py — DO NOT EDIT BY HAND",
        f"% {what}",
        f"% generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"% source: {len(run_dirs)} ladder run dirs, read from *_trajectory.json "
        "metadata.token_usage",
    ]
    return "\n".join(lines) + "\n"


def concurrency_note(collected: dict, tex: bool = False) -> tuple[str, dict]:
    """The wall-clock caveat, built from what the runs actually recorded."""
    mw = sorted({r["max_workers"] for r in collected["runs"] if r["max_workers"]})
    sw = sorted({r["scenario_max_workers"] for r in collected["runs"]
                 if r["scenario_max_workers"]})
    ep = sorted({r["n_endpoints"] for r in collected["runs"] if r["n_endpoints"]})
    facts = {"max_workers": mw, "scenario_max_workers": sw, "n_endpoints": ep,
             "max_inflight": (max(mw) * max(sw)) if mw and sw else None}
    mw_s = "/".join(str(x) for x in mw) or "?"
    sw_s = "/".join(str(x) for x in sw) or "?"
    ep_s = "/".join(str(x) for x in ep) or "?"
    times = "$\\times$" if tex else "x"
    inflight = facts["max_inflight"]
    txt = (f"Wall-clock is throughput under load, not latency: the ladder ran "
           f"{sw_s} scenarios {times} {mw_s} agents concurrently (up to "
           f"{inflight} in-flight designs) against {ep_s} vLLM endpoints per "
           f"size, so per-scenario seconds include queueing behind the other "
           f"agents and are not a single-user response time.")
    return txt, facts


def gen_tab_cost(stats: dict, collected: dict, args_ablation: bool) -> str:
    labels = collected["size_labels"]
    agents = [a for a in AGENT_ORDER if any(a in stats[lb] for lb in labels)]
    abl = [a for a in ABLATION_ORDER if any(a in stats[lb] for lb in labels)] \
        if args_ablation else []

    colspec = "@{}l" + "r@{\\hskip 3pt}r@{\\hskip 3pt}r@{\\hskip 3pt}r" * len(labels) + "@{}"
    lines = [header_comment("tab_cost.tex — cost / deployability profile",
                            [Path(r["dir"]) for r in collected["runs"]])]
    lines.append("\\begin{table*}[t]")
    lines.append(
        "  \\caption{Cost profile on the Qwen size ladder, pooled over 3 runs."
        " Each cell: median per-scenario total tokens (thousands), model calls"
        " and wall-clock seconds, with $n$ = (scenario, run) cells that produced"
        f" a trajectory out of {max(collected['cells'].values())}.}}")
    lines.append("  \\label{tab:cost}")
    lines.append("  \\footnotesize")
    lines.append("  \\setlength{\\tabcolsep}{2pt}")
    lines.append(f"  \\begin{{tabular}}{{{colspec}}}")
    lines.append("    \\toprule")
    heads = " & ".join(
        f"\\multicolumn{{4}}{{c}}{{Qwen3.5-{size_display(lb)}}}" for lb in labels)
    lines.append(f"    & {heads} \\\\")
    lines.append("    " + "".join(
        f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(labels))))
    subs = " & ".join(["Tok.\\,k", "Calls", "Sec.", "$n$"] * len(labels))
    lines.append(f"    Agent & {subs} \\\\")
    lines.append("    \\midrule")

    def render(agent_ids: list[str], indent: bool = False) -> list[str]:
        rows = []
        for a in agent_ids:
            name = AGENT_DISPLAY.get(a, a)
            if a == PROPOSED:
                name = f"\\textbf{{{name}}}"
            if indent:
                name = f"\\quad {name}"
            cells = []
            for lb in labels:
                s = stats[lb].get(a)
                if not s:
                    cells += ["--"] * 4
                    continue
                n = s["n_cells_with_trajectory"]
                n_txt = str(n)
                if s["coverage"] < COVERAGE_FLAG:
                    n_txt = f"{n}\\textsuperscript{{*}}"
                cells += [
                    fmt_num(s["total_tokens"]["median"] / 1000.0, 1),
                    fmt_num(s["llm_calls"]["median"], 0),
                    fmt_num(s["wall_seconds"]["median"], 0),
                    n_txt,
                ]
            rows.append(f"    {name} & " + " & ".join(cells) + " \\\\")
        return rows

    lines += render(agents)
    if abl:
        lines.append("    \\midrule")
        lines.append("    \\multicolumn{" + str(1 + 4 * len(labels)) +
                     "}{@{}l}{\\itshape Ablation arms of the proposed harness}"
                     " \\\\")
        lines += render(abl, indent=True)
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    conc, _ = concurrency_note(collected, tex=True)
    lines.append(
        "  \\par\\smallskip\\footnotesize\\textsuperscript{*}$n$ below "
        f"{fmt_num(COVERAGE_FLAG * 100, 0)}\\% of cells: the agent crashed on the"
        " rest and left no artifact, so its cost here is conditional on the"
        " scenarios it survived and reads cheaper than it is."
        " \\emph{" + conc + "}"
        " Tokens are read from \\texttt{metadata.token\\_usage}; the top-level"
        " \\texttt{total\\_tokens} and \\texttt{cost\\_usd} fields are not"
        " populated consistently and are not used.")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def gen_macros(stats: dict, collected: dict, rat: dict, cost: dict) -> str:
    labels = collected["size_labels"]
    tex = [header_comment("cost_macros.tex — cost numbers for prose",
                          [Path(r["dir"]) for r in collected["runs"]])]

    def newcmd(name: str, value: str, comment: str) -> None:
        tex.append(f"% {comment}")
        tex.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for i, lb in enumerate(labels):
        w = size_word(lb, i)
        s = stats[lb].get(PROPOSED)
        if s:
            newcmd(f"costHarnessTokens{w}",
                   fmt_num(s["total_tokens"]["median"] / 1000.0, 1),
                   f"median per-scenario total tokens (thousands) of the proposed "
                   f"harness at {size_display(lb)}, pooled over 3 runs")
            newcmd(f"costHarnessCalls{w}", fmt_num(s["llm_calls"]["median"], 0),
                   f"median per-scenario model calls of the proposed harness at "
                   f"{size_display(lb)}")
            newcmd(f"costHarnessSeconds{w}", fmt_num(s["wall_seconds"]["median"], 0),
                   f"median per-scenario wall-clock seconds of the proposed harness "
                   f"at {size_display(lb)} — THROUGHPUT UNDER CONCURRENCY, not latency")
        r = rat.get(lb, {})
        if "total_tokens_vs_cheapest" in r:
            e = r["total_tokens_vs_cheapest"]
            newcmd(f"costTokenRatioVsCheapest{w}", fmt_num(e["ratio"], 1),
                   f"proposed / cheapest-baseline ({e['baseline']}) median total "
                   f"tokens at {size_display(lb)}; >1 means the harness costs more")
        if "total_tokens_vs_cheapest_full_coverage" in r:
            e = r["total_tokens_vs_cheapest_full_coverage"]
            newcmd(f"costTokenRatioVsCheapestComplete{w}", fmt_num(e["ratio"], 1),
                   f"proposed / cheapest baseline THAT COMPLETED the benchmark "
                   f"({e['baseline']}, coverage "
                   f"{fmt_num(e['coverage'] * 100, 0)}%) median total tokens at "
                   f"{size_display(lb)} — the honest headline multiplier, since a "
                   f"crashed baseline's median is a survivors-only figure")
        if "llm_calls_vs_cheapest_full_coverage" in r:
            e = r["llm_calls_vs_cheapest_full_coverage"]
            newcmd(f"costCallRatioVsCheapestComplete{w}", fmt_num(e["ratio"], 1),
                   f"proposed / cheapest completing baseline ({e['baseline']}) "
                   f"median model calls at {size_display(lb)}")
        if "total_tokens_vs_median_baseline" in r:
            newcmd(f"costTokenRatioVsMedianBaseline{w}",
                   fmt_num(r["total_tokens_vs_median_baseline"]["ratio"], 1),
                   f"proposed / median-across-baselines token multiplier at "
                   f"{size_display(lb)}")
        if "llm_calls_vs_cheapest" in r:
            e = r["llm_calls_vs_cheapest"]
            newcmd(f"costCallRatioVsCheapest{w}", fmt_num(e["ratio"], 1),
                   f"proposed / cheapest-baseline ({e['baseline']}) median model "
                   f"calls at {size_display(lb)}")
        if "llm_calls_vs_median_baseline" in r:
            newcmd(f"costCallRatioVsMedianBaseline{w}",
                   fmt_num(r["llm_calls_vs_median_baseline"]["ratio"], 1),
                   f"proposed / median-across-baselines model-call multiplier at "
                   f"{size_display(lb)}")
        comp = r.get("components", {})
        if "alignmentgraph-isd-no-verifier" in comp:
            newcmd(f"costVerifierTokenOverhead{w}",
                   fmt_num(comp["alignmentgraph-isd-no-verifier"]["token_overhead_pct"], 1),
                   f"percent extra median total tokens the verifier costs at "
                   f"{size_display(lb)} (full harness vs the no-verifier arm)")
        if "alignmentgraph-isd-no-graph-ctx" in comp:
            newcmd(f"costGraphContextTokenOverhead{w}",
                   fmt_num(comp["alignmentgraph-isd-no-graph-ctx"]["token_overhead_pct"], 1),
                   f"percent extra median total tokens graph-context injection "
                   f"costs at {size_display(lb)} (full harness vs the "
                   f"no-graph-context arm)")
        if "alignmentgraph-isd-skeleton" in comp:
            newcmd(f"costBothComponentsTokenOverhead{w}",
                   fmt_num(comp["alignmentgraph-isd-skeleton"]["token_overhead_pct"], 1),
                   f"percent extra median total tokens both components together "
                   f"cost at {size_display(lb)} (full harness vs the skeleton arm)")
        c = cost.get(lb)
        if c:
            newcmd(f"costGpuClass{w}", c["gpu"],
                   f"GPU class that served {size_display(lb)} (assumption block, "
                   f"{SERVING_SOURCE})")
            newcmd(f"costGpuVram{w}", str(c["vram_gb"]),
                   f"VRAM in GB of the {size_display(lb)} serving GPU")
            newcmd(f"costUsdPerHour{w}Assumed",
                   fmt_num(c["usd_per_hour_assumed"], 2),
                   "ASSUMED hourly list price of that GPU — a price quote, not a "
                   "measurement")
            pa = c["per_agent"].get(PROPOSED)
            if pa:
                newcmd(f"costUsdPerDesign{w}Assumed",
                       fmt_num(pa["usd_per_design_assumed"], 4),
                       "ASSUMPTION-DERIVED GPU dollars per design produced by the "
                       "proposed harness: measured billed pod-hours x assumed "
                       "hourly price, attributed pro-rata by token share")

    all_tok = [r[k]["ratio"] for r in rat.values()
               for k in ("total_tokens_max", "total_tokens_min") if k in r]
    if all_tok:
        newcmd("costTokenRatioMinAll", fmt_num(min(all_tok), 1),
               "smallest proposed/baseline median-token multiplier across all "
               "sizes and baselines")
        newcmd("costTokenRatioMaxAll", fmt_num(max(all_tok), 1),
               "largest proposed/baseline median-token multiplier across all "
               "sizes and baselines")

    # Coverage, because a cost table read without it is misleading.
    worst = None
    for lb in labels:
        for a, s in stats[lb].items():
            if worst is None or s["coverage"] < worst[2]:
                worst = (a, lb, s["coverage"], s)
    if worst:
        a, lb, covv, s = worst
        newcmd("costWorstCoverageAgent", AGENT_DISPLAY.get(a, a),
               "agent x size cell with the lowest artifact coverage in the ladder")
        newcmd("costWorstCoverageSize", size_display(lb),
               "model size of that lowest-coverage cell")
        newcmd("costWorstCoveragePct", fmt_num(covv * 100, 0),
               "percent of (scenario, run) cells in which that agent produced an "
               "artifact at all")
        if s["n_missing"]:
            newcmd("costWorstCoverageTruncPct",
                   fmt_num(100.0 * s["n_missing_truncation"] / s["n_missing"], 0),
                   "percent of that cell's missing artifacts whose log ends in the "
                   "per-call token-cap truncation error")

    _, facts = concurrency_note(collected)
    if facts["max_inflight"]:
        newcmd("costMaxInflightDesigns", str(facts["max_inflight"]),
               "peak concurrent agent runs during the ladder (scenario workers x "
               "agent workers) — why wall-clock is throughput, not latency")
    return "\n".join(tex) + "\n"


def gen_stats_md(stats: dict, collected: dict, rat: dict, cost: dict) -> str:
    labels = collected["size_labels"]
    conc, facts = concurrency_note(collected)
    md = [
        "# Cost / deployability profile (generated)",
        "",
        "AUTO-GENERATED by `scripts/alignmentgraph-isd-bench/13_gen_cost_table.py` "
        "— do not edit by hand.",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}  ",
        f"Source: {len(collected['runs'])} ladder run directories, "
        "`*_trajectory.json` -> `metadata.token_usage` and "
        "`metadata.execution_time_seconds`.",
        "",
        "## What is and is not measured",
        "",
        "- **Tokens come only from `metadata.token_usage`.** The top-level "
        "`metadata.total_tokens` is 0 for most agents and is never read.",
        "- **`metadata.cost_usd` is ignored.** On a self-hosted ladder there is no "
        "per-token price; one agent nonetheless writes a dollar figure from a "
        "hosted-API price list. Any dollar number below is derived here, from the "
        "labelled assumption block.",
        f"- **Wall-clock caveat.** {conc}",
        "- **Cost is conditional on survival.** An agent that crashes writes no "
        "trajectory, so its statistics describe only the scenarios it completed. "
        "Read every row together with its `n`.",
        "",
        "## Concurrency provenance (measured)",
        "",
        f"- agent workers per scenario: {facts['max_workers']}",
        f"- concurrent scenarios: {facts['scenario_max_workers']}",
        f"- vLLM endpoints per model size: {facts['n_endpoints']}",
        f"- peak in-flight designs: {facts['max_inflight']}",
        "",
    ]

    md.append("## Per agent x size (median [IQR], mean)")
    md.append("")
    for lb in labels:
        md.append(f"### Qwen3.5-{size_display(lb)}")
        md.append("")
        md.append("| Agent | Total tok. med [IQR] | mean | Prompt med | "
                  "Compl. med | Calls med [IQR] | Wall s med [IQR] | n | cov. |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        order = [a for a in AGENT_ORDER + ABLATION_ORDER if a in stats[lb]]
        order += [a for a in sorted(stats[lb]) if a not in order]
        for a in order:
            s = stats[lb][a]
            t, c_, w = s["total_tokens"], s["llm_calls"], s["wall_seconds"]
            flag = "" if s["coverage"] >= COVERAGE_FLAG else " *"
            md.append(
                f"| {AGENT_DISPLAY.get(a, a)} "
                f"| {fmt_num(t['median'], 0)} "
                f"[{fmt_num(t['p25'], 0)}--{fmt_num(t['p75'], 0)}] "
                f"| {fmt_num(t['mean'], 0)} "
                f"| {fmt_num(s['prompt_tokens']['median'], 0)} "
                f"| {fmt_num(s['completion_tokens']['median'], 0)} "
                f"| {fmt_num(c_['median'], 0)} "
                f"[{fmt_num(c_['p25'], 0)}--{fmt_num(c_['p75'], 0)}] "
                f"| {fmt_num(w['median'], 0)} "
                f"[{fmt_num(w['p25'], 0)}--{fmt_num(w['p75'], 0)}] "
                f"| {s['n_cells_with_trajectory']}/{s['n_cells_total']}{flag} "
                f"| {fmt_num(s['coverage'] * 100, 0)}% |")
        md.append("")

    md.append("## Coverage roster and failure causes")
    md.append("")
    md.append("Cells with no trajectory are crashes. `truncation` = the log ends in "
              "the per-call token-cap error (`LengthFinishReasonError`), i.e. the "
              "agent was cut off by the shared cap rather than being cheap; "
              "`other` = every other cause (schema/parse failures, connection "
              "errors).")
    md.append("")
    md.append("| Size | Agent | present | missing | truncation | other | "
              "no token_usage |")
    md.append("|---|---|---|---|---|---|---|")
    any_missing = False
    for lb in labels:
        order = [a for a in AGENT_ORDER + ABLATION_ORDER if a in stats[lb]]
        for a in order:
            s = stats[lb][a]
            if not s["n_missing"] and not s["n_no_token_usage"]:
                continue
            any_missing = True
            md.append(
                f"| {size_display(lb)} | {AGENT_DISPLAY.get(a, a)} "
                f"| {s['n_cells_with_trajectory']}/{s['n_cells_total']} "
                f"| {s['n_missing']} | {s['n_missing_truncation']} "
                f"| {s['n_missing_other']} | {s['n_no_token_usage']} |")
    if not any_missing:
        md.append("| -- | -- | full coverage everywhere | 0 | 0 | 0 | 0 |")
    md.append("")

    md.append("## Cost multiplier of the proposed harness over each baseline")
    md.append("")
    md.append("Ratio of medians (proposed / baseline). A value above 1 means the "
              "proposed harness spends more. Baselines whose coverage is low are "
              "marked: their denominator is a survivors-only median and therefore "
              "flatters them.")
    md.append("")
    for lb in labels:
        r = rat.get(lb)
        if not r:
            continue
        md.append(f"### Qwen3.5-{size_display(lb)}")
        md.append("")
        md.append("| Baseline | tokens x | calls x | wall-clock x | baseline cov. |")
        md.append("|---|---|---|---|---|")
        for b in AGENT_ORDER:
            e = r["per_baseline"].get(b)
            if not e:
                continue
            flag = "" if e["coverage"] >= COVERAGE_FLAG else " *"
            md.append(f"| {AGENT_DISPLAY.get(b, b)} | {fmt_num(e['total_tokens'], 2)} "
                      f"| {fmt_num(e['llm_calls'], 2)} "
                      f"| {fmt_num(e['wall_seconds'], 2)} "
                      f"| {fmt_num(e['coverage'] * 100, 0)}%{flag} |")
        md.append("")
        for key, nice in (("total_tokens", "tokens"), ("llm_calls", "model calls")):
            ch = r.get(f"{key}_vs_cheapest")
            cf = r.get(f"{key}_vs_cheapest_full_coverage")
            me = r.get(f"{key}_vs_median_baseline")
            if ch and me:
                md.append(f"- {nice}: {fmt_num(ch['ratio'], 2)}x the cheapest "
                          f"baseline ({AGENT_DISPLAY.get(ch['baseline'], ch['baseline'])}, "
                          f"coverage {fmt_num(ch['coverage'] * 100, 0)}%), "
                          f"{fmt_num(me['ratio'], 2)}x the median baseline.")
            if cf:
                md.append(f"  - against the cheapest baseline that actually "
                          f"completed the benchmark "
                          f"({AGENT_DISPLAY.get(cf['baseline'], cf['baseline'])}, "
                          f"coverage {fmt_num(cf['coverage'] * 100, 0)}%): "
                          f"{fmt_num(cf['ratio'], 2)}x {nice}.")
        comp = r.get("components") or {}
        if comp:
            md.append("")
            md.append("Marginal cost of the harness components at this size "
                      "(full harness vs the ablation arm, median total tokens):")
            md.append("")
            for arm in ABLATION_ORDER:
                e = comp.get(arm)
                if not e:
                    continue
                md.append(f"- vs `{arm}` ({AGENT_DISPLAY.get(arm, arm)}): "
                          f"{fmt_num(e['token_overhead_pct'], 1)}% more total "
                          f"tokens, {fmt_num(e['prompt_overhead_pct'], 1)}% more "
                          f"prompt tokens.")
        md.append("")

    md.append("## Serving profile and an ASSUMPTION-DERIVED dollar figure")
    md.append("")
    md.append(f"GPU class, VRAM and hourly price are **not measured here**. Source: "
              f"{SERVING_SOURCE}. Prices move; treat them as a quote.")
    md.append("")
    md.append("Measured per size: total billed pod-hours over the 3 runs "
              "(run wall-clock x number of endpoints, so it includes the judging "
              "phase during which the pods idle) and the number of designs those "
              "runs produced across all agents.")
    md.append("")
    md.append("| Size | GPU | VRAM | USD/h (assumed) | endpoints | pod-h (meas.) | "
              "designs | USD total (assumed) | USD per design, all agents |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for lb in labels:
        c = cost.get(lb)
        if not c:
            continue
        md.append(
            f"| {size_display(lb)} | {c['gpu']} | {c['vram_gb']} GB "
            f"| {fmt_num(c['usd_per_hour_assumed'], 2)} "
            f"| {c['n_endpoints']} | {fmt_num(c['pod_hours_measured'], 2)} "
            f"| {c['designs_produced']} "
            f"| {fmt_num(c['usd_total_assumed'], 2)} "
            f"| {fmt_num(c['usd_per_design_all_agents_assumed'], 4)} |")
    md.append("")
    md.append("Per-agent attribution below splits each size's dollar total "
              "pro-rata by the agent's share of that size's total tokens. That is "
              "an **assumption** (10 agents shared the pods; decode time is not "
              "exactly proportional to tokens), stated so a reader can reject it "
              "and still keep the token and call columns above.")
    md.append("")
    md.append("| Size | Agent | token share | USD per design (assumed) |")
    md.append("|---|---|---|---|")
    for lb in labels:
        c = cost.get(lb)
        if not c:
            continue
        for a in AGENT_ORDER + ABLATION_ORDER:
            pa = c["per_agent"].get(a)
            if not pa:
                continue
            md.append(f"| {size_display(lb)} "
                      f"| {AGENT_DISPLAY.get(a, a)} "
                      f"| {fmt_num(pa['token_share'] * 100, 1)}% "
                      f"| {fmt_num(pa['usd_per_design_assumed'], 4)} |")
    md.append("")
    md.append("## Run provenance")
    md.append("")
    md.append("| Run dir | size | elapsed | endpoints | trajectories |")
    md.append("|---|---|---|---|---|")
    for r in sorted(collected["runs"], key=lambda x: (size_of_label(x["label"]), x["dir"])):
        el = ("--" if r["elapsed_seconds"] is None
              else f"{r['elapsed_seconds'] / 3600.0:.2f} h")
        md.append(f"| {r['dir']} | {size_display(r['label'])} | {el} "
                  f"| {r['n_endpoints']} | {r['n_trajectories']} |")
    md.append("")
    return "\n".join(md) + "\n"


# -- main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the cost / deployability profile from ladder run artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs-glob", default=DEFAULT_RUNS_GLOB, metavar="PATTERN",
                        help="glob of ladder run directories, relative to the "
                             "benchmark root")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="output directory for generated artifacts")
    parser.add_argument("--no-ablation-rows", action="store_true",
                        help="omit the 3 ablation arms from tab_cost.tex "
                             "(they always stay in stats_cost.md)")
    args = parser.parse_args()

    run_dirs = sorted(p for p in BENCH_ROOT.glob(args.runs_glob) if p.is_dir())
    if not run_dirs:
        print(f"Error: no run directories matched {args.runs_glob!r} under "
              f"{BENCH_ROOT}", file=sys.stderr)
        sys.exit(1)
    print(f"reading {len(run_dirs)} run directories...")

    collected = collect(run_dirs)
    stats = summarise(collected)
    baselines = [a for a in AGENT_ORDER if a != PROPOSED]
    rat = ratios(stats, collected["size_labels"], baselines)
    cost = serving_cost(collected, stats)

    args.outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tab_cost.tex": gen_tab_cost(stats, collected, not args.no_ablation_rows),
        "cost_macros.tex": gen_macros(stats, collected, rat, cost),
        "stats_cost.md": gen_stats_md(stats, collected, rat, cost),
    }
    for name, content in outputs.items():
        path = args.outdir / name
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
