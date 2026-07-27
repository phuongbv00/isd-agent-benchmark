#!/usr/bin/env python3
"""Repair transient multi-judge failures without rerunning agent generation.

The evaluator's ``compare --agents AGENT`` command rewrites the complete
``comparison_report`` with only the selected agent.  This script therefore
evaluates each affected agent in a temporary directory, validates the result,
and merges only that repaired ranking row back into the original report.

It also keeps the run-level ``benchmark_summary.json`` / ``SUMMARY.md`` in
sync, creates one-time ``.pre-judge-repair`` backups, and runs the post-run
audit when finished.

Usage:
  # Inspect the repair scope; no judge calls and no writes.
  python scripts/alignmentgraph-isd-bench/04_repair_ladder_results.py --dry-run

  # Repair every ladder run found by the default glob, then audit.
  python scripts/alignmentgraph-isd-bench/04_repair_ladder_results.py

  # Non-interactive use with an explicit set of run dirs.
  python scripts/alignmentgraph-isd-bench/04_repair_ladder_results.py --yes \
      'results/test_90_benchmark_Qwen3.5-*_r[123]_*'

The evaluator is always invoked with ``--no-run --multi-judge``.  Existing
agent output and trajectory files are read but never regenerated.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_SRC = REPO_ROOT / "evaluator" / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVALUATOR_SRC))

from isd_evaluator.reporters.comparison import ComparisonReporter  # noqa: E402
from run_benchmark import generate_summary_report  # noqa: E402

DEFAULT_RUN_GLOB = "results/test_90_benchmark_*_r*"
DEFAULT_BACKUP_SUFFIX = ".pre-judge-repair"
REQUIRED_METRICS = (
    "total_score",
    "addie_median",
    "trajectory_score",
    "phase_scores",
)


@dataclass(frozen=True)
class RepairTarget:
    """One agent ranking whose stored judge evaluation is incomplete."""

    run_dir: Path
    variant: str
    scenario_id: str
    scenario_dir: Path
    report_path: Path
    agent_id: str
    judge_error_count: int
    missing_metrics: tuple[str, ...]


def read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON file only after its complete replacement is on disk."""
    write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
    )


def write_text_atomic(path: Path, content: str) -> None:
    """Replace a text file only after its complete replacement is on disk."""
    temp_path = path.with_name(f".{path.name}.repair-tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def backup_once(path: Path, suffix: str) -> Path | None:
    """Keep the first pre-repair version; never overwrite that evidence."""
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def resolve_run_dirs(patterns: list[str]) -> list[Path]:
    """Expand quoted globs and explicit run directories, preserving order."""
    raw_patterns = patterns or [DEFAULT_RUN_GLOB]
    resolved: list[Path] = []
    seen: set[Path] = set()
    for pattern in raw_patterns:
        absolute_pattern = (
            pattern if Path(pattern).is_absolute() else str(REPO_ROOT / pattern)
        )
        for match in sorted(glob.glob(absolute_pattern)):
            path = Path(match).resolve()
            if path.is_dir() and path not in seen:
                resolved.append(path)
                seen.add(path)
    return resolved


def missing_metrics(row: dict[str, Any]) -> tuple[str, ...]:
    """Return metric fields that the post-run audit treats as missing."""
    return tuple(
        key for key in REQUIRED_METRICS if row.get(key) in (None, {}, "")
    )


def discover_targets(run_dirs: list[Path]) -> list[RepairTarget]:
    """Find ranking rows containing at least one judge error."""
    targets: list[RepairTarget] = []
    for run_dir in run_dirs:
        for report_path in sorted(run_dir.glob("*/*/comparison_report.json")):
            scenario_dir = report_path.parent
            variant = scenario_dir.parent.name
            report = read_json(report_path)
            rankings = report.get("comparison", {}).get("rankings", [])
            for row in rankings:
                error_count = sum(
                    1 for judge in row.get("judges", []) if judge.get("error")
                )
                if not error_count:
                    continue
                targets.append(
                    RepairTarget(
                        run_dir=run_dir,
                        variant=variant,
                        scenario_id=scenario_dir.name,
                        scenario_dir=scenario_dir,
                        report_path=report_path,
                        agent_id=str(row.get("agent_id", "unknown")),
                        judge_error_count=error_count,
                        missing_metrics=missing_metrics(row),
                    )
                )
    return targets


def judge_signature(row: dict[str, Any]) -> list[tuple[Any, Any]]:
    """Return judge identities in their recorded completion order."""
    return [
        (judge.get("provider"), judge.get("model"))
        for judge in row.get("judges", [])
    ]


def validate_repaired_entry(
    old_entry: dict[str, Any],
    new_entry: dict[str, Any],
) -> list[str]:
    """Return validation errors for a newly evaluated ranking row."""
    errors: list[str] = []
    if new_entry.get("agent_id") != old_entry.get("agent_id"):
        errors.append(
            f"agent mismatch: {old_entry.get('agent_id')} -> "
            f"{new_entry.get('agent_id')}"
        )
    old_roster = Counter(judge_signature(old_entry))
    new_roster = Counter(judge_signature(new_entry))
    if new_roster != old_roster:
        errors.append(
            f"judge roster changed: {judge_signature(old_entry)} -> "
            f"{judge_signature(new_entry)}"
        )
    failed_judges = [
        judge for judge in new_entry.get("judges", [])
        if judge.get("error") or judge.get("addie_score") is None
    ]
    if failed_judges:
        errors.append(f"{len(failed_judges)} judge(s) still failed")
    missing = missing_metrics(new_entry)
    if missing:
        errors.append(f"missing metrics: {', '.join(missing)}")
    return errors


def merge_repaired_entry(
    report: dict[str, Any],
    repaired_entry: dict[str, Any],
) -> dict[str, Any]:
    """Replace one agent row, then restore ranking and best-agent metadata."""
    merged = copy.deepcopy(report)
    rankings = merged["comparison"]["rankings"]
    agent_id = repaired_entry["agent_id"]
    matches = [
        index for index, row in enumerate(rankings)
        if row.get("agent_id") == agent_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one ranking row for {agent_id}, found {len(matches)}"
        )
    rankings[matches[0]] = repaired_entry
    rankings.sort(
        key=lambda row: (
            row.get("total_score")
            if row.get("total_score") is not None
            else float("-inf")
        ),
        reverse=True,
    )
    for rank, row in enumerate(rankings, 1):
        row["rank"] = rank
    merged["comparison"]["rankings"] = rankings
    merged["comparison"]["best_agent"] = rankings[0]["agent_id"]
    merged["generated_at"] = datetime.now().isoformat()
    return merged


def evaluate_target(
    target: RepairTarget,
    original_report: dict[str, Any],
    evaluator_bin: Path,
    max_attempts: int,
    verbose: bool,
) -> dict[str, Any]:
    """Evaluate one affected agent in isolation and return its clean row."""
    old_rows = [
        row for row in original_report["comparison"]["rankings"]
        if row.get("agent_id") == target.agent_id
    ]
    if len(old_rows) != 1:
        raise ValueError(
            f"{target.report_path}: expected one {target.agent_id} row"
        )
    old_entry = old_rows[0]
    output_path = target.scenario_dir / f"{target.agent_id}_output.json"
    trajectory_path = (
        target.scenario_dir / f"{target.agent_id}_trajectory.json"
    )
    if not output_path.exists():
        raise FileNotFoundError(f"missing agent output: {output_path}")
    scenario = original_report.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError(f"missing embedded scenario: {target.report_path}")

    last_errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="isd-judge-repair-") as temp:
        temp_dir = Path(temp)
        shutil.copy2(output_path, temp_dir / output_path.name)
        if trajectory_path.exists():
            shutil.copy2(trajectory_path, temp_dir / trajectory_path.name)
        scenario_path = temp_dir / f"{target.scenario_id}.json"
        write_json_atomic(scenario_path, scenario)

        command = [
            str(evaluator_bin),
            "compare",
            "--scenario",
            str(scenario_path),
            "--output-dir",
            str(temp_dir),
            "--agents",
            target.agent_id,
            "--no-run",
            "--use-llm",
            "--multi-judge",
        ]
        if verbose:
            command.append("--verbose")

        for attempt in range(1, max_attempts + 1):
            print(f"    judge attempt {attempt}/{max_attempts}", flush=True)
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            temp_report_path = temp_dir / "comparison_report.json"
            if completed.returncode != 0:
                last_errors = [
                    f"evaluator exited with code {completed.returncode}"
                ]
                continue
            if not temp_report_path.exists():
                last_errors = ["evaluator did not write comparison_report.json"]
                continue
            temp_report = read_json(temp_report_path)
            rows = temp_report.get("comparison", {}).get("rankings", [])
            if len(rows) != 1:
                last_errors = [
                    f"temporary report contains {len(rows)} ranking rows"
                ]
                continue
            last_errors = validate_repaired_entry(old_entry, rows[0])
            if not last_errors:
                return rows[0]
            print(f"    retry: {'; '.join(last_errors)}", flush=True)

    raise RuntimeError(
        f"no clean evaluation after {max_attempts} attempt(s): "
        f"{'; '.join(last_errors) or 'unknown evaluator failure'}"
    )


def sync_reports(
    target: RepairTarget,
    merged_report: dict[str, Any],
    backup_suffix: str,
    reporter: ComparisonReporter,
) -> None:
    """Write scenario and run-level reports from one merged evaluation."""
    scenario_md = target.scenario_dir / "comparison_report.md"
    scenario_markdown = reporter.generate_markdown(
        merged_report["comparison"],
        merged_report["scenario"],
    )

    summary_path = target.run_dir / "benchmark_summary.json"
    summary_md = target.run_dir / "SUMMARY.md"
    summary: dict[str, Any] | None = None
    summary_markdown: str | None = None
    if summary_path.exists():
        summary = read_json(summary_path)
        scenario_slot = (
            summary.get("scenarios", {})
            .get(target.variant, {})
            .get(target.scenario_id)
        )
        if not isinstance(scenario_slot, dict):
            raise KeyError(
                f"{summary_path}: no "
                f"scenarios/{target.variant}/{target.scenario_id}"
            )
        scenario_slot["evaluation"] = merged_report
        with tempfile.TemporaryDirectory(prefix="isd-summary-repair-") as temp:
            rendered_summary = Path(temp) / "SUMMARY.md"
            generate_summary_report(summary, rendered_summary)
            summary_markdown = rendered_summary.read_text(encoding="utf-8")
    else:
        print(
            f"    warning: {summary_path} is missing; scenario report repaired "
            "but run summary could not be synchronized",
            flush=True,
        )

    # Everything above is validation/rendering only. Start mutating the run
    # after every replacement artifact has been prepared successfully.
    backup_once(target.report_path, backup_suffix)
    backup_once(scenario_md, backup_suffix)
    write_json_atomic(target.report_path, merged_report)
    write_text_atomic(scenario_md, scenario_markdown)
    if summary is None or summary_markdown is None:
        return
    backup_once(summary_path, backup_suffix)
    backup_once(summary_md, backup_suffix)
    write_json_atomic(summary_path, summary)
    write_text_atomic(summary_md, summary_markdown)


def run_postrun_audit(run_dirs: list[Path]) -> int:
    """Run the canonical audit over exactly the repaired input run dirs."""
    audit_script = Path(__file__).with_name("04_audit_postrun.py")
    command = [sys.executable, str(audit_script), *map(str, run_dirs)]
    print("\nRunning post-run audit...", flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def print_scope(targets: list[RepairTarget]) -> None:
    """Print a concise, deterministic repair manifest."""
    judge_errors = sum(target.judge_error_count for target in targets)
    missing_count = sum(len(target.missing_metrics) for target in targets)
    print(
        f"Repair scope: {len(targets)} agent×scenario entries, "
        f"{judge_errors} errored judge records, "
        f"{missing_count} missing metric fields."
    )
    print(
        "Note: post-run audit counts RED findings per run/category; "
        "that total is not the number of repair targets."
    )
    for target in targets:
        missing = (
            f"; missing={','.join(target.missing_metrics)}"
            if target.missing_metrics
            else ""
        )
        print(
            f"  {target.run_dir.name}/{target.variant}/"
            f"{target.scenario_id}: {target.agent_id} "
            f"({target.judge_error_count} judge error(s){missing})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help=(
            "Run dirs or quoted globs. Default: "
            f"{DEFAULT_RUN_GLOB!r}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List affected entries without judge calls or file writes.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt (for non-interactive use).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum clean-evaluation attempts per affected entry (default: 3).",
    )
    parser.add_argument(
        "--evaluator-bin",
        type=Path,
        default=Path(sys.executable).with_name("isd-evaluator"),
        help="isd-evaluator executable (default: next to the current Python).",
    )
    parser.add_argument(
        "--backup-suffix",
        default=DEFAULT_BACKUP_SUFFIX,
        help=(
            "One-time backup suffix inserted before the extension "
            f"(default: {DEFAULT_BACKUP_SUFFIX!r})."
        ),
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Do not run 04_audit_postrun.py after repair.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Pass --verbose to isd-evaluator compare.",
    )
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be >= 1")
    if not args.backup_suffix:
        parser.error("--backup-suffix must not be empty")
    return args


def main() -> int:
    args = parse_args()
    run_dirs = resolve_run_dirs(args.run_dirs)
    if not run_dirs:
        print("No run dirs found.", file=sys.stderr)
        return 1
    if not args.evaluator_bin.exists():
        print(
            f"Evaluator executable not found: {args.evaluator_bin}",
            file=sys.stderr,
        )
        return 1

    print(f"Run dirs: {len(run_dirs)}")
    targets = discover_targets(run_dirs)
    print_scope(targets)

    if args.dry_run:
        return 0
    if not targets:
        print("No judge errors found; nothing to repair.")
        return 0 if args.skip_audit else run_postrun_audit(run_dirs)

    if not args.yes:
        answer = input(
            f"\nThis will call the configured judges for {len(targets)} "
            "affected entries and update their reports. Type 'yes' to continue: "
        )
        if answer.strip().lower() != "yes":
            print("Aborted. No files changed.")
            return 1

    reporter = ComparisonReporter()
    repaired = 0
    for index, target in enumerate(targets, 1):
        print(
            f"\n[{index}/{len(targets)}] {target.run_dir.name} / "
            f"{target.scenario_id} / {target.agent_id}",
            flush=True,
        )
        original_report = read_json(target.report_path)
        try:
            repaired_entry = evaluate_target(
                target=target,
                original_report=original_report,
                evaluator_bin=args.evaluator_bin,
                max_attempts=args.max_attempts,
                verbose=args.verbose,
            )
            merged_report = merge_repaired_entry(
                original_report,
                repaired_entry,
            )
            sync_reports(
                target=target,
                merged_report=merged_report,
                backup_suffix=args.backup_suffix,
                reporter=reporter,
            )
        except Exception as exc:  # noqa: BLE001 - preserve completed repairs
            print(
                f"\nRepair stopped after {repaired}/{len(targets)} completed "
                f"entry(s): {exc}",
                file=sys.stderr,
            )
            print(
                "Re-run the same command to resume; completed entries no "
                "longer contain judge errors.",
                file=sys.stderr,
            )
            return 1
        repaired += 1
        print(
            f"    repaired: total_score={repaired_entry['total_score']}, "
            f"judges={len(repaired_entry['judges'])}",
            flush=True,
        )

    remaining = discover_targets(run_dirs)
    if remaining:
        print_scope(remaining)
        print("Judge errors remain after repair.", file=sys.stderr)
        return 1

    print(f"\nRepair complete: {repaired} entry(s), 0 judge errors remain.")
    return 0 if args.skip_audit else run_postrun_audit(run_dirs)


if __name__ == "__main__":
    raise SystemExit(main())
