#!/usr/bin/env python3
"""Post-run audit of finished run dirs (guide sections 7.3-7.5).

Per run dir (default: every results/test_90_benchmark_*_r*), reports:

  Completeness (7.3)  scenario-dir count, per-agent missing outputs,
                      comparison_report.json presence, benchmark_summary.json
  Integrity (7.4)     stub/default outputs, tiny output (<1KB) / trajectory
                      (<200B) files, token_usage == 0, model name vs dir name,
                      suspicious durations (0s or > 1h)
  Metrics (7.5)       ranked entries have total/addie/trajectory/phase scores,
                      uniform judge count, judge errors
  Scored fields (7.6) the fields RQ2 actually consumes are non-empty, measured
                      by calling the evaluator's own extractors
  RQ2 readiness (8)   alignment_scores.json coverage, per encoder

Missing agents are INFO, not failure — capability failures (e.g. baseline
blowing the token cap on small models) are RQ1 data and are handled by the
pooling failure policies. Red = infrastructure-level junk (stubs, empty
outputs, zero token accounting, model/dir mismatch).

Usage:
  python scripts/alignmentgraph-isd-bench/04_audit_postrun.py                          # all ladder runs
  python scripts/alignmentgraph-isd-bench/04_audit_postrun.py results/test_90_benchmark_Qwen3.5-2B_r1_*
  python scripts/alignmentgraph-isd-bench/04_audit_postrun.py --fields-only results/single_<ts>

Exit code 0 = no red findings, 1 = red findings present.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluator" / "src"))

from isd_evaluator.metrics.alignment import (  # noqa: E402
    extract_activities,
    extract_assessment_items,
    extract_evaluation_texts,
    extract_objectives,
)

# The 10 agents a ladder run produces: 6 benchmark defaults + the proposed
# system + the 3 ablation arms.
AGENTS = ["eduplanner", "baseline", "react-isd", "addie-agent",
          "dick-carey-agent", "rpisd-agent", "alignmentgraph-isd",
          "alignmentgraph-isd-no-verifier", "alignmentgraph-isd-no-graph-ctx",
          "alignmentgraph-isd-skeleton"]
STUB_MARKS = ['"미지정"', '"problem_definition": null']
EXPECTED_SCENARIOS = 90

#: Every text the alignment metric consumes — evaluation included, since
#: objective_evaluation_alignment is a scored endpoint too.
SCORED_FIELDS = [
    ("objectives", extract_objectives),
    ("assessments", extract_assessment_items),
    ("activities", extract_activities),
    ("evaluation", extract_evaluation_texts),
]


def scored_texts(output_path: Path) -> tuple[list[str], list[str]]:
    """(texts the metric would score, names of fields that came back empty)."""
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], [name for name, _ in SCORED_FIELDS]
    addie = payload.get("addie_output") or payload
    texts, empty = [], []
    for name, extract in SCORED_FIELDS:
        try:
            items = extract(addie) or []
        except Exception:  # noqa: BLE001 - a broken output is data, not a crash
            items = []
        if not items:
            empty.append(name)
        for item in items:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(str(item.get("text") or ""))
            else:
                texts.append(str(item))
    return [t for t in texts if t], empty


def audit_run(run_dir: Path, fields_only: bool = False) -> dict:
    red: list[str] = []
    info: list[str] = []
    # A single-scenario dir (results/single_<ts>) keeps its outputs at the root;
    # treat the dir itself as the one scenario rather than expecting 90.
    flat_single = any(run_dir.glob("*_output.json")) if run_dir.exists() else False
    if flat_single:
        return audit_scored_fields(run_dir, [run_dir])
    variant_dirs = [d for d in run_dir.iterdir() if d.is_dir()] if run_dir.exists() else []
    scen_dirs: list[Path] = []
    for d in variant_dirs:
        subs = [s for s in d.iterdir() if s.is_dir()]
        if subs and any((s / "comparison_report.json").exists() for s in subs):
            scen_dirs = sorted(subs)
            break
    if not scen_dirs:  # flat single-run layout
        scen_dirs = sorted(d for d in variant_dirs
                           if (d / "comparison_report.json").exists())

    if fields_only:
        return audit_scored_fields(run_dir, scen_dirs)

    # ---- completeness (7.3)
    n = len(scen_dirs)
    if n != EXPECTED_SCENARIOS:
        red.append(f"scenario dirs = {n}, expected {EXPECTED_SCENARIOS}")
    missing: dict[str, int] = {a: 0 for a in AGENTS}
    no_report = 0
    for sd in scen_dirs:
        for a in AGENTS:
            if not (sd / f"{a}_output.json").exists():
                missing[a] += 1
        if not (sd / "comparison_report.json").exists():
            no_report += 1
    if no_report:
        red.append(f"{no_report} scenario(s) without comparison_report.json")
    for a, m in missing.items():
        if m:
            info.append(f"missing outputs: {a} x{m} (agent failure — OK if capability)")
    if not (run_dir / "benchmark_summary.json").exists():
        red.append("benchmark_summary.json missing at run root")

    # ---- integrity (7.4)
    stubs, tiny_out, tiny_traj, zero_tok, bad_dur = [], [], [], [], []
    for sd in scen_dirs:
        for out in sd.glob("*_output.json"):
            text = out.read_text(encoding="utf-8", errors="replace")
            if all(m in text for m in STUB_MARKS):
                stubs.append(f"{sd.name}/{out.name}")
            if out.stat().st_size < 1024:
                tiny_out.append(f"{sd.name}/{out.name}")
        for traj in sd.glob("*_trajectory.json"):
            if traj.stat().st_size < 200:
                tiny_traj.append(f"{sd.name}/{traj.name}")
                continue
            try:
                md = json.loads(traj.read_text(encoding="utf-8")).get("metadata") or {}
            except (json.JSONDecodeError, OSError):
                continue
            tu = (md.get("token_usage") or {}).get("total_tokens")
            if tu == 0:
                zero_tok.append(f"{sd.name}/{traj.name}")
            secs = md.get("execution_time_seconds")
            if isinstance(secs, (int, float)) and (secs == 0 or secs > 3600):
                bad_dur.append(f"{sd.name}/{traj.name} ({secs:.0f}s)")
    for label, items in [("stub outputs", stubs), ("outputs < 1KB", tiny_out),
                         ("trajectories < 200B", tiny_traj),
                         ("token_usage == 0", zero_tok)]:
        if items:
            red.append(f"{label}: {len(items)} (e.g. {items[0]})")
    if bad_dur:
        info.append(f"suspicious durations: {len(bad_dur)} (e.g. {bad_dur[0]})")

    # model name vs dir name
    summary_path = run_dir / "benchmark_summary.json"
    if summary_path.exists():
        try:
            model = json.loads(summary_path.read_text(encoding="utf-8"))["model"]["name"]
            if model.split("/")[-1] not in run_dir.name:
                red.append(f"model in summary ({model}) does not match dir name")
        except (json.JSONDecodeError, KeyError):
            info.append("could not read model name from benchmark_summary.json")

    # ---- metrics (7.5)
    bad_fields, judge_counts, judge_errors = 0, set(), 0
    for sd in scen_dirs:
        rep = sd / "comparison_report.json"
        if not rep.exists():
            continue
        try:
            rankings = json.loads(rep.read_text(encoding="utf-8"))["comparison"]["rankings"]
        except (json.JSONDecodeError, KeyError):
            red.append(f"unreadable comparison_report: {sd.name}")
            continue
        for r in rankings:
            for k in ("total_score", "addie_median", "trajectory_score", "phase_scores"):
                if r.get(k) in (None, {}, ""):
                    bad_fields += 1
            judges = r.get("judges", [])
            judge_counts.add(len(judges))
            judge_errors += sum(1 for j in judges if j.get("error"))
    if bad_fields:
        red.append(f"{bad_fields} ranked entries with missing metric fields")
    if len(judge_counts) > 1:
        red.append(f"non-uniform judge count across entries: {sorted(judge_counts)}")
    if judge_errors:
        red.append(f"{judge_errors} judge error(s)")

    # ---- scored fields (7.6)
    fields = audit_scored_fields(run_dir, scen_dirs)
    red += fields["red"]
    info += fields["info"]

    # ---- RQ2 readiness. Per encoder, not just the primary: a sweep filled in
    # one pod at a time is easy to leave half-covered, and partial coverage
    # pools as if it were the whole thing.
    coverage: dict[str, int] = {}
    for scen_dir in scen_dirs:
        for path in scen_dir.glob("alignment_scores*.json"):
            arm = path.name[len("alignment_scores"):-len(".json")].lstrip(".") or "primary"
            coverage[arm] = coverage.get(arm, 0) + 1
    primary_coverage = coverage.get("primary", 0)
    if 0 < primary_coverage < n:
        red.append(
            f"partial alignment_scores.json coverage: {primary_coverage}/{n} "
            "(resume 06_score_alignment.py before pooling RQ2)"
        )
    elif primary_coverage == 0:
        info.append(f"alignment_scores.json: {coverage.get('primary', 0)}/{n} "
                    "(run scripts/alignmentgraph-isd-bench/06_score_alignment.py before pooling RQ2)")
    for arm, count in sorted(coverage.items()):
        if arm != "primary" and count < n:
            red.append(f"partial alignment_scores.{arm}.json coverage: {count}/{n} "
                       "(resume that encoder arm before pooling its sensitivity results)")

    return {"n_scenarios": n, "missing": missing, "red": red, "info": info,
            "judge_counts": sorted(judge_counts), "n_agents": fields["n_agents"]}


def audit_scored_fields(run_dir: Path, scen_dirs: list[Path]) -> dict:
    """Are the fields RQ2 consumes actually populated?

    Measured through the evaluator's own extractors, so this sees exactly what
    the alignment metric sees: an output that parses but yields no objectives
    scores 0.0 and is indistinguishable from a genuinely misaligned one unless
    the emptiness is surfaced here.
    """
    info: list[str] = []
    agents: set[str] = set()
    empties: dict[str, int] = {}
    empty_fields: Counter[str] = Counter()
    for sd in scen_dirs:
        for out in sorted(sd.glob("*_output.json")):
            agent = out.name[: -len("_output.json")]
            agents.add(agent)
            _texts, empty = scored_texts(out)
            if empty:
                empties[agent] = empties.get(agent, 0) + 1
                empty_fields.update(empty)
    if empties:
        worst = max(empties.items(), key=lambda kv: kv[1])
        breakdown = ", ".join(
            f"{field} x{count}" for field, count in empty_fields.most_common()
        )
        info.append(f"empty scored field in {sum(empties.values())} output(s), "
                    f"worst {worst[0]} x{worst[1]} ({breakdown}) — "
                    "capability/parse failures; affected alignment endpoints "
                    "are scored at their metric floor")
    return {"n_scenarios": len(scen_dirs), "missing": {}, "judge_counts": [],
            "red": [], "info": info, "n_agents": len(agents)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="*",
                        help="Run dirs (default: results/test_90_benchmark_*_r*)")
    parser.add_argument("--fields-only", action="store_true",
                        help="Run only the scored-field check (7.6) — for smoke "
                             "runs with no judge scores or 90 scenarios yet.")
    args = parser.parse_args()

    dirs = [Path(p) for pat in args.run_dirs for p in sorted(glob.glob(pat))] \
        if args.run_dirs else sorted(REPO_ROOT.glob("results/test_90_benchmark_*_r*"))
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        print("No run dirs found.", file=sys.stderr)
        return 1

    total_red = 0
    for run_dir in dirs:
        r = audit_run(run_dir, args.fields_only)
        verdict = "OK" if not r["red"] else "RED"
        head = f"({r['n_scenarios']} scenarios"
        if not args.fields_only:
            head += f", judges/entry: {r['judge_counts']}"
        print(f"\n== {run_dir.name}  [{verdict}]  {head})")
        for line in r["red"]:
            print(f"  RED : {line}")
        for line in r["info"]:
            print(f"  info: {line}")
        total_red += len(r["red"])

    print(f"\n== {len(dirs)} run(s) audited — "
          f"{'ALL OK' if total_red == 0 else f'{total_red} RED finding(s)'} ==")
    return 0 if total_red == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
