"""Leave-one-out ablation statistics for the alignmentgraph-isd variant agents.

Each arm removes exactly ONE mechanism from the full harness. The verify-repair
loop is a generation-mode-independent phase, so removing decomposition or
dynamic planning does not incidentally disable it — every arm is a clean
single-factor ablation from ``full``:

    full           : complete harness (reference)
    wo-ma  : full - multi-agent decomposition (single agent still builds
                     the graph and still runs verify-repair)
    wo-dynamic     : full - dynamic graph-state planning (fixed ADDIE order)
    wo-verifyrepair: full - verify-repair-cascade loop

All arms run side by side inside one benchmark run, so every comparison_report
contains all arms for the same scenario. Per run directory this script reports:

  * per-arm mean Total / ADDIE / Trajectory and per-phase ADDIE means;
  * each ablated arm's contribution = full - arm, paired by scenario, with
    Wilcoxon signed-rank + Holm correction + bootstrap 95% CI;
  * (default) an offline residual-violation audit: the deterministic verifier
    is replayed over every arm's final ADDIE output, so the verify-repair loop's
    effect is visible as violations remaining per arm;
  * LaTeX rows for the paper's RQ2 table.

Usage:
    python scripts/stats_ablation.py results/<run>/test_90 [more run dirs ...]
        [--no-audit] [--no-latex]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from stats_rq1 import bootstrap_ci, holm_correct, wilcoxon_signed_rank

FULL = "alignmentgraph-isd"
# Display order: reference first, then the leave-one-out arms.
LADDER = [
    ("alignmentgraph-isd", "full (reference)"),
    ("alignmentgraph-isd-wo-ma", "wo multi-agent"),
    ("alignmentgraph-isd-wo-qc", "wo QC tools"),
]
# Contribution of each removed mechanism = full - ablated arm.
CONTRASTS = [
    ("alignmentgraph-isd", "alignmentgraph-isd-wo-ma", "multi-agent decomposition"),
    ("alignmentgraph-isd", "alignmentgraph-isd-wo-qc", "per-agent verify/repair tools"),
]
# Legacy agent IDs from earlier ablation runs (pivot/ladder/leave-one-out) -> current.
_ALIASES = {
    "alignmentgraph-isd-wo-verifyrepair": "alignmentgraph-isd-wo-qc",
    "alignmentgraph-isd-wo-verifier": "alignmentgraph-isd-wo-qc",
    "alignmentgraph-isd-nocheck": "alignmentgraph-isd-wo-qc",
    "alignmentgraph-isd-single": "alignmentgraph-isd-wo-ma",
}
PHASES = ["analysis", "design", "development", "implementation", "evaluation"]

# Verifier findings that can be recomputed from the final ADDIE output alone
# (structural coverage checks need graph edges, which outputs do not carry).
_AUDIT_TYPES = (
    "bloom", "measurability", "rubric_consistency", "bloom_mismatch",
    "activity_engagement", "evaluation_quality", "actionability",
)


def load_records(run_dir: Path) -> dict[str, dict[str, dict]]:
    """scenario_id -> agent_id -> {total, addie, trajectory, phases}"""
    out: dict[str, dict[str, dict]] = {}
    for d in sorted(run_dir.iterdir()):
        rep = d / "comparison_report.json"
        if not rep.exists():
            continue
        data = json.loads(rep.read_text())
        per_agent = {}
        for r in data["comparison"]["rankings"]:
            agent_id = _ALIASES.get(r["agent_id"], r["agent_id"])
            per_agent[agent_id] = {
                "total": r["total_score"],
                "addie": r.get("addie_mean"),
                "trajectory": r.get("trajectory_score"),
                "phases": r.get("phase_scores") or {},
            }
        out[d.name] = per_agent
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _paired_contrast(records: dict, upper: str, lower: str) -> dict | None:
    diffs = [
        agents[upper]["total"] - agents[lower]["total"]
        for agents in records.values()
        if upper in agents and lower in agents
    ]
    if not diffs:
        return None
    _, p, _ = wilcoxon_signed_rank(diffs)
    lo, hi = bootstrap_ci(diffs)
    addie_diffs = [
        agents[upper]["addie"] - agents[lower]["addie"]
        for agents in records.values()
        if upper in agents and lower in agents
        and agents[upper]["addie"] is not None and agents[lower]["addie"] is not None
    ]
    return {
        "n": len(diffs),
        "mean": mean(diffs),
        "mean_addie": mean(addie_diffs) if addie_diffs else float("nan"),
        "lo": lo,
        "hi": hi,
        "p": p,
    }


def report_scores(run_dir: Path, records: dict, latex: bool) -> None:
    present = [
        (agent_id, label) for agent_id, label in LADDER
        if any(agent_id in agents for agents in records.values())
    ]
    if not present:
        print("  no ladder arms found in this run")
        return

    summary = {}
    for agent_id, label in present:
        rows = [agents[agent_id] for agents in records.values() if agent_id in agents]
        summary[agent_id] = {
            "label": label,
            "n": len(rows),
            "total": mean([r["total"] for r in rows]),
            "addie": mean([r["addie"] for r in rows if r["addie"] is not None]),
            "trajectory": mean([r["trajectory"] for r in rows if r["trajectory"] is not None]),
            "phases": {
                p: mean([r["phases"][p] for r in rows if p in r["phases"]])
                for p in PHASES
                if any(p in r["phases"] for r in rows)
            },
        }

    header = (
        f"{'arm':26s} {'n':>3s} {'Total':>7s} {'ADDIE':>7s} {'Traj.':>7s} "
        + " ".join(f"{p[:4].capitalize():>6s}" for p in PHASES)
    )
    print(header)
    for agent_id, _ in present:
        s = summary[agent_id]
        phase_cols = " ".join(f"{s['phases'].get(p, float('nan')):6.1f}" for p in PHASES)
        print(f"{s['label']:26s} {s['n']:3d} {s['total']:7.2f} {s['addie']:7.2f} {s['trajectory']:7.2f} {phase_cols}")

    # Mechanism contribution = full - ablated arm (leave-one-out), paired.
    contrasts = []
    for upper_id, lower_id, mechanism in CONTRASTS:
        stat = _paired_contrast(records, upper_id, lower_id)
        if stat:
            contrasts.append((mechanism, upper_id, lower_id, stat))
    if contrasts:
        print("\nMechanism contribution = full - (arm without it), paired on total score:")
        holm = holm_correct([s["p"] for _, _, _, s in contrasts])
        for (mechanism, upper_id, lower_id, s), ph in zip(contrasts, holm):
            print(
                f"  {mechanism:26s} n={s['n']:2d}  "
                f"dTotal={s['mean']:+6.2f} [{s['lo']:+6.2f}, {s['hi']:+6.2f}]  "
                f"dADDIE={s['mean_addie']:+6.2f}  p_holm={ph:.2e}"
            )

    if latex:
        print("\n% LaTeX rows (arm & Total & ADDIE & Traj.)")
        for agent_id, _ in present:
            s = summary[agent_id]
            print(f"{s['label']} & {s['total']:.2f} & {s['addie']:.2f} & {s['trajectory']:.2f} \\\\")
        print("% contrast rows (mechanism & dTotal & dADDIE & p_holm)")
        holm = holm_correct([s["p"] for _, _, _, s in contrasts]) if contrasts else []
        for (mechanism, _u, _l, s), ph in zip(contrasts, holm):
            print(f"{mechanism} & ${s['mean']:+.2f}$ & ${s['mean_addie']:+.2f}$ & {ph:.1e} \\\\")


def report_audit(run_dir: Path) -> None:
    """Replay the deterministic verifier over each arm's final ADDIE output.

    The graph is reconstructed approximately from the output sections (no
    edges), so only content-level checks are counted; the same reconstruction
    is applied to every arm, which keeps the comparison fair.
    """
    try:
        from alignmentgraph_isd import HarnessRunConfig
        from alignmentgraph_isd.core.agents.pedagogical_verifier import PedagogicalVerifierAgent
        from alignmentgraph_isd.core.graph import AlignmentGraph
        from alignmentgraph_isd.core.trajectory import TrajectoryLogger
    except ImportError as exc:
        print(f"\n(residual-violation audit skipped: alignmentgraph_isd not importable: {exc})")
        return

    config = HarnessRunConfig.from_ablation("full")
    arm_ids = {agent_id for agent_id, _ in LADDER} | set(_ALIASES)
    print("\nResidual verifier violations on final outputs (offline audit):")
    print(f"{'arm':26s} {'n':>3s} {'violations':>10s} {'mean/scen':>9s}  by type")
    labels = dict(LADDER)
    rows = []
    for d in sorted(run_dir.iterdir()):
        for out_file in d.glob("*_output.json"):
            agent_id = out_file.name[: -len("_output.json")]
            agent_id = _ALIASES.get(agent_id, agent_id)
            if agent_id not in {a for a, _ in LADDER}:
                continue
            try:
                output = json.loads(out_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            graph = AlignmentGraph()
            graph.add_node("objective", "objective_0", output.get("design") or {})
            development = output.get("development") or {}
            graph.add_node("assessment", "assessment_0", {"assessment_tools": development.get("assessment_tools") or []})
            graph.add_node("activity", "activity_0", development)
            graph.add_node("implementation_plan", "implementation_plan_0", output.get("implementation") or {})
            graph.add_node("evaluation", "evaluation_0", output.get("evaluation") or {})
            violations = [
                v for v in PedagogicalVerifierAgent(config, graph, TrajectoryLogger()).verify()
                if v.violation_type in _AUDIT_TYPES
            ]
            rows.append((agent_id, violations))

    for agent_id, label in LADDER:
        arm_rows = [violations for a, violations in rows if a == agent_id]
        if not arm_rows:
            continue
        total = sum(len(v) for v in arm_rows)
        by_type = Counter(v.violation_type for violations in arm_rows for v in violations)
        type_str = ", ".join(f"{t}={c}" for t, c in by_type.most_common(4))
        print(f"{label:26s} {len(arm_rows):3d} {total:10d} {total / len(arm_rows):9.2f}  {type_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Run directories containing per-scenario result folders",
    )
    parser.add_argument("--no-latex", action="store_true", help="Skip the LaTeX row output")
    parser.add_argument("--no-audit", action="store_true", help="Skip the residual-violation audit")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        records = load_records(run_dir)
        print(f"\n=== {run_dir} ({len(records)} scenarios) ===")
        report_scores(run_dir, records, latex=not args.no_latex)
        if not args.no_audit:
            report_audit(run_dir)


if __name__ == "__main__":
    main()
