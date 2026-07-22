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
  RQ2 readiness (8)   alignment_scores.json coverage

Missing agents are INFO, not failure — capability failures (e.g. baseline
blowing the token cap on small models) are RQ1 data and are handled by the
pooling failure policies. Red = infrastructure-level junk (stubs, empty
outputs, zero token accounting, model/dir mismatch).

Usage:
  python scripts/6_audit_postrun.py                          # all ladder runs
  python scripts/6_audit_postrun.py results/test_90_benchmark_Qwen3.5-2B_r1_*

Exit code 0 = no red findings, 1 = red findings present.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS = ["eduplanner", "baseline", "react-isd", "addie-agent",
          "dick-carey-agent", "rpisd-agent", "alignmentgraph-isd"]
STUB_MARKS = ['"미지정"', '"problem_definition": null']
EXPECTED_SCENARIOS = 90


def audit_run(run_dir: Path) -> dict:
    red: list[str] = []
    info: list[str] = []
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

    # ---- RQ2 readiness
    n_align = sum(1 for sd in scen_dirs if (sd / "alignment_scores.json").exists())
    if n_align < n:
        info.append(f"alignment_scores.json: {n_align}/{n} "
                    "(run scripts/7_score_alignment.py before pooling RQ2)")

    return {"n_scenarios": n, "missing": missing, "red": red, "info": info,
            "judge_counts": sorted(judge_counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="*",
                        help="Run dirs (default: results/test_90_benchmark_*_r*)")
    args = parser.parse_args()

    dirs = [Path(p) for pat in args.run_dirs for p in sorted(glob.glob(pat))] \
        if args.run_dirs else sorted(REPO_ROOT.glob("results/test_90_benchmark_*_r*"))
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        print("No run dirs found.", file=sys.stderr)
        return 1

    total_red = 0
    for run_dir in dirs:
        r = audit_run(run_dir)
        verdict = "OK" if not r["red"] else "RED"
        print(f"\n== {run_dir.name}  [{verdict}]  "
              f"({r['n_scenarios']} scenarios, judges/entry: {r['judge_counts']})")
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
