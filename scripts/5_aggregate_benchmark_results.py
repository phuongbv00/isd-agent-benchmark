#!/usr/bin/env python3
"""Aggregate ISD benchmark artifacts into paper-style comparison tables.

The benchmark runner writes per-scenario comparison reports plus an optional
benchmark_summary.json. This script reads those artifacts and computes a compact
overview table similar to the main result table in the paper:

  Agent | evaluator model scores | Avg | difficulty scores | phase scores

Examples:
  python scripts/5_aggregate_benchmark_results.py results/test_30_benchmark_...
  python scripts/5_aggregate_benchmark_results.py results/test_30_benchmark_... \
      --formats csv,md,json
  python scripts/5_aggregate_benchmark_results.py results/test_30_benchmark_... \
      --formats md --include-counts
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PHASES = ["analysis", "design", "development", "implementation", "evaluation"]
PHASE_LABELS = {
    "analysis": "Ana.",
    "design": "Des.",
    "development": "Dev.",
    "implementation": "Impl.",
    "evaluation": "Eval.",
}

DIFFICULTIES = ["Easy", "Medium", "Hard"]
DIFFICULTY_LABELS = {"Easy": "Easy", "Medium": "Med.", "Hard": "Hard"}

AGENT_LABELS = {
    "baseline": "Baseline",
    "eduplanner": "EduPlanner",
    "addie-agent": "ADDIE-Agent",
    "rpisd-agent": "RPISD-Agent",
    "dick-carey-agent": "Dick-Carey-Agent",
    "react-isd": "React-ADDIE",
    "alignmentgraph-isd": "AlignmentGraph-ISD (proposed)",
}

DEFAULT_AGENT_ORDER = [
    "baseline",
    "eduplanner",
    "addie-agent",
    "rpisd-agent",
    "dick-carey-agent",
    "react-isd",
    "alignmentgraph-isd",
]

@dataclass
class ScenarioRecord:
    variant: str
    scenario_id: str
    scenario_file: str | None
    rankings: list[dict[str, Any]]
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class AgentAggregate:
    agent_id: str
    evaluated: int = 0
    success: int = 0
    failed: int = 0
    total_scores: list[float] = field(default_factory=list)
    addie_scores: list[float] = field(default_factory=list)
    trajectory_scores: list[float] = field(default_factory=list)
    judge_scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    difficulty_scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    phase_scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def mean_total(self) -> float | None:
        return mean_or_none(self.total_scores)

    def mean_addie(self) -> float | None:
        return mean_or_none(self.addie_scores)

    def mean_trajectory(self) -> float | None:
        return mean_or_none(self.trajectory_scores)

    def mean_judge(self, model: str) -> float | None:
        return mean_or_none(self.judge_scores.get(model, []))

    def mean_difficulty(self, difficulty: str) -> float | None:
        return mean_or_none(self.difficulty_scores.get(difficulty, []))

    def mean_phase(self, phase: str) -> float | None:
        return mean_or_none(self.phase_scores.get(phase, []))


def mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return statistics.fmean(values)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_scenario_id(name: str) -> str:
    return Path(name).stem


def normalize_difficulty(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    if "easy" in text:
        return "Easy"
    if "medium" in text or "moderate" in text or text.startswith("med"):
        return "Medium"
    if "hard" in text:
        return "Hard"
    return None


def short_model_name(model: str) -> str:
    raw = model.split("/")[-1]
    lowered = raw.lower()
    if "gemini" in lowered:
        return "Gemini"
    if "gpt" in lowered or "openai" in lowered:
        return "GPT"
    if "solar" in lowered:
        return "Solar"
    if "gemma" in lowered:
        return "Gemma"
    cleaned = re.sub(r"[-_]?20\d{2}[-_]?\d{2}[-_]?\d{2}", "", raw)
    cleaned = cleaned.replace("-mini", "").replace("-flash-lite", "")
    return cleaned[:18] or model


def make_unique_labels(models: list[str]) -> dict[str, str]:
    base = {model: short_model_name(model) for model in models}
    counts: dict[str, int] = defaultdict(int)
    for label in base.values():
        counts[label] += 1
    labels: dict[str, str] = {}
    for model in models:
        label = base[model]
        if counts[label] > 1:
            label = model.split("/")[-1]
        labels[model] = label
    return labels


def infer_repo_root(run_dir: Path) -> Path:
    current = run_dir.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "scenarios").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return Path(__file__).resolve().parents[1]


def load_records_from_summary(summary_path: Path) -> list[ScenarioRecord]:
    data = load_json(summary_path)
    records: list[ScenarioRecord] = []
    for variant, scenarios in data.get("scenarios", {}).items():
        for scenario_id, scenario_result in scenarios.items():
            comparison = scenario_result.get("evaluation", {}).get("comparison", {})
            records.append(
                ScenarioRecord(
                    variant=variant,
                    scenario_id=normalize_scenario_id(scenario_id),
                    scenario_file=scenario_result.get("scenario"),
                    rankings=comparison.get("rankings", []) or [],
                    agents=scenario_result.get("agents", {}) or {},
                )
            )
    return records


def load_records_from_reports(run_dir: Path) -> list[ScenarioRecord]:
    records: list[ScenarioRecord] = []
    for report_path in sorted(run_dir.glob("*/*/comparison_report.json")):
        report = load_json(report_path)
        comparison = report.get("comparison", report)
        scenario = report.get("scenario", {}) or {}
        variant = report_path.parent.parent.name
        scenario_id = normalize_scenario_id(
            scenario.get("scenario_id") or report_path.parent.name
        )
        records.append(
            ScenarioRecord(
                variant=variant,
                scenario_id=scenario_id,
                scenario_file=f"{scenario_id}.json",
                rankings=comparison.get("rankings", []) or [],
                agents={},
            )
        )
    return records


def load_records(run_dir: Path) -> list[ScenarioRecord]:
    summary_path = run_dir / "benchmark_summary.json"
    if summary_path.exists():
        return load_records_from_summary(summary_path)
    return load_records_from_reports(run_dir)


def find_scenario_file(
    scenario_root: Path, variant: str, scenario_id: str, scenario_file: str | None
) -> Path | None:
    candidates: list[Path] = []
    if scenario_file:
        scenario_path = Path(scenario_file)
        if scenario_path.is_absolute():
            candidates.append(scenario_path)
        else:
            candidates.extend(
                [
                    scenario_root / variant / scenario_path,
                    scenario_root / scenario_path,
                ]
            )
    candidates.extend(
        [
            scenario_root / variant / f"{scenario_id}.json",
            scenario_root / "test" / f"{scenario_id}.json",
            scenario_root / "test_30" / f"{scenario_id}.json",
            scenario_root / "test_90" / f"{scenario_id}.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def difficulty_for_record(
    record: ScenarioRecord, scenario_root: Path, cache: dict[tuple[str, str], str | None]
) -> str | None:
    key = (record.variant, record.scenario_id)
    if key in cache:
        return cache[key]
    scenario_path = find_scenario_file(
        scenario_root, record.variant, record.scenario_id, record.scenario_file
    )
    difficulty = None
    if scenario_path:
        try:
            scenario = load_json(scenario_path)
            difficulty = normalize_difficulty(scenario.get("difficulty"))
        except json.JSONDecodeError:
            difficulty = None
    cache[key] = difficulty
    return difficulty


def judge_total(judge: dict[str, Any]) -> float | None:
    addie = judge.get("addie_score")
    trajectory = judge.get("trajectory_score")
    if addie is None:
        return None
    if trajectory is None:
        return float(addie)
    return 0.7 * float(addie) + 0.3 * float(trajectory)


def collect_judge_models(records: list[ScenarioRecord]) -> list[str]:
    seen: set[str] = set()
    models: list[str] = []
    for record in records:
        for ranking in record.rankings:
            for judge in ranking.get("judges", []) or []:
                model = judge.get("model")
                if model and model not in seen:
                    seen.add(model)
                    models.append(model)
    return sorted(models, key=lambda m: (short_model_name(m), m))


def aggregate(records: list[ScenarioRecord], scenario_root: Path) -> dict[str, AgentAggregate]:
    aggregates: dict[str, AgentAggregate] = {}
    difficulty_cache: dict[tuple[str, str], str | None] = {}

    def get_agent(agent_id: str) -> AgentAggregate:
        if agent_id not in aggregates:
            aggregates[agent_id] = AgentAggregate(agent_id=agent_id)
        return aggregates[agent_id]

    for record in records:
        difficulty = difficulty_for_record(record, scenario_root, difficulty_cache)

        for agent_id, result in record.agents.items():
            agent = get_agent(agent_id)
            if result.get("success"):
                agent.success += 1
            else:
                agent.failed += 1

        for ranking in record.rankings:
            agent_id = ranking.get("agent_id")
            if not agent_id:
                continue
            agent = get_agent(agent_id)
            agent.evaluated += 1

            total_score = ranking.get("total_score")
            if total_score is not None:
                score = float(total_score)
                agent.total_scores.append(score)
                if difficulty:
                    agent.difficulty_scores[difficulty].append(score)

            addie_score = ranking.get("addie_median", ranking.get("addie_score"))
            if addie_score is not None:
                agent.addie_scores.append(float(addie_score))

            trajectory_score = ranking.get("trajectory_score")
            if trajectory_score is not None:
                agent.trajectory_scores.append(float(trajectory_score))

            for judge in ranking.get("judges", []) or []:
                model = judge.get("model")
                score = judge_total(judge)
                if model and score is not None:
                    agent.judge_scores[model].append(score)

            for phase, value in (ranking.get("phase_scores") or {}).items():
                if phase in PHASES and value is not None:
                    agent.phase_scores[phase].append(float(value))

    return aggregates


def ordered_agents(aggregates: dict[str, AgentAggregate]) -> list[str]:
    known = [agent for agent in DEFAULT_AGENT_ORDER if agent in aggregates]
    unknown = sorted(agent for agent in aggregates if agent not in DEFAULT_AGENT_ORDER)
    return known + unknown


def build_rows(
    aggregates: dict[str, AgentAggregate], judge_models: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for agent_id in ordered_agents(aggregates):
        agg = aggregates[agent_id]
        judge_means = {model: agg.mean_judge(model) for model in judge_models}
        available_judge_means = [value for value in judge_means.values() if value is not None]
        evaluator_avg = mean_or_none(available_judge_means)
        if evaluator_avg is None:
            evaluator_avg = agg.mean_total()

        rows.append(
            {
                "agent_id": agent_id,
                "agent": AGENT_LABELS.get(agent_id, agent_id),
                "evaluated": agg.evaluated,
                "success": agg.success,
                "failed": agg.failed,
                "judges": judge_means,
                "avg": evaluator_avg,
                "addie": agg.mean_addie(),
                "trajectory": agg.mean_trajectory(),
                "difficulty": {
                    difficulty: agg.mean_difficulty(difficulty)
                    for difficulty in DIFFICULTIES
                },
                "phases": {phase: agg.mean_phase(phase) for phase in PHASES},
            }
        )
    return rows


def numeric_columns(judge_models: list[str]) -> list[tuple[str, tuple[str, str | None]]]:
    columns: list[tuple[str, tuple[str, str | None]]] = []
    columns.extend((f"judge:{model}", ("judges", model)) for model in judge_models)
    columns.append(("avg", ("avg", None)))
    columns.append(("addie", ("addie", None)))
    columns.append(("trajectory", ("trajectory", None)))
    columns.extend((f"difficulty:{difficulty}", ("difficulty", difficulty)) for difficulty in DIFFICULTIES)
    columns.extend((f"phase:{phase}", ("phases", phase)) for phase in PHASES)
    return columns


def row_value(row: dict[str, Any], source: tuple[str, str | None]) -> float | None:
    section, key = source
    if section in {"avg", "addie", "trajectory"}:
        return row.get(section)
    return row.get(section, {}).get(key)


def best_values(rows: list[dict[str, Any]], judge_models: list[str]) -> dict[str, float]:
    best: dict[str, float] = {}
    for column_id, source in numeric_columns(judge_models):
        values = [row_value(row, source) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            best[column_id] = max(values)
    return best


def format_number(
    value: float | None,
    decimals: int,
    bold: bool = False,
    latex: bool = False,
) -> str:
    if value is None:
        return "-"
    text = f"{value:.{decimals}f}"
    if bold:
        return f"\\textbf{{{text}}}" if latex else f"**{text}**"
    return text


def is_best(value: float | None, best: float | None) -> bool:
    if value is None or best is None:
        return False
    return abs(value - best) < 1e-9


def render_markdown(
    rows: list[dict[str, Any]],
    judge_models: list[str],
    labels: dict[str, str],
    include_counts: bool,
    bold_best: bool,
) -> str:
    best = best_values(rows, judge_models) if bold_best else {}

    def table(headers: list[str], body_rows: list[list[str]]) -> list[str]:
        aligns = ["---"] + ["---:"] * (len(headers) - 1)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(aligns) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body_rows)
        return lines

    lines: list[str] = []

    headers = ["Agent"]
    if include_counts:
        headers.extend(["Eval n", "Success", "Failed"])
    headers.extend(labels[model] for model in judge_models)
    headers.append("Avg")
    body_rows = []
    for row in rows:
        cells = [row["agent"]]
        if include_counts:
            cells.extend([str(row["evaluated"]), str(row["success"]), str(row["failed"])])
        for model in judge_models:
            column_id = f"judge:{model}"
            value = row["judges"].get(model)
            cells.append(format_number(value, 2, is_best(value, best.get(column_id))))
        cells.append(format_number(row["avg"], 2, is_best(row["avg"], best.get("avg"))))
        body_rows.append(cells)

    lines.append("## Evaluator Model")
    lines.extend(table(headers, body_rows))
    lines.append("")

    headers = ["Agent", "Total", "ADDIE", "Trajectory"]
    body_rows = []
    for row in rows:
        body_rows.append(
            [
                row["agent"],
                format_number(row["avg"], 2, is_best(row["avg"], best.get("avg"))),
                format_number(row["addie"], 2, is_best(row["addie"], best.get("addie"))),
                format_number(
                    row["trajectory"],
                    2,
                    is_best(row["trajectory"], best.get("trajectory")),
                ),
            ]
        )

    lines.append("## Score Components")
    lines.extend(table(headers, body_rows))
    lines.append("")

    headers = ["Agent", "Easy", "Med.", "Hard"]
    body_rows = []
    for row in rows:
        cells = [row["agent"]]
        for difficulty in DIFFICULTIES:
            column_id = f"difficulty:{difficulty}"
            value = row["difficulty"].get(difficulty)
            cells.append(format_number(value, 2, is_best(value, best.get(column_id))))
        body_rows.append(cells)

    lines.append("## Difficulty")
    lines.extend(table(headers, body_rows))
    lines.append("")

    model_count = len(judge_models)
    title = f"## Phase-wise ({model_count}-model avg)" if model_count else "## Phase-wise"
    headers = ["Agent"] + [PHASE_LABELS[phase] for phase in PHASES]
    body_rows = []
    for row in rows:
        cells = [row["agent"]]
        for phase in PHASES:
            column_id = f"phase:{phase}"
            value = row["phases"].get(phase)
            cells.append(format_number(value, 1, is_best(value, best.get(column_id))))
        body_rows.append(cells)

    lines.append(title)
    lines.extend(table(headers, body_rows))

    return "\n".join(lines) + "\n"


def render_csv(rows: list[dict[str, Any]], judge_models: list[str], labels: dict[str, str]) -> str:
    from io import StringIO

    def fmt(value: Any, decimals: int = 2) -> Any:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.{decimals}f}"
        return value

    buffer = StringIO()
    writer = csv.writer(buffer)
    header = ["agent_id", "agent", "evaluated", "success", "failed"]
    header.extend(labels[model] for model in judge_models)
    header.extend(["Avg", "ADDIE", "Trajectory", "Easy", "Medium", "Hard"])
    header.extend(PHASE_LABELS[phase] for phase in PHASES)
    writer.writerow(header)

    for row in rows:
        values = [
            row["agent_id"],
            row["agent"],
            row["evaluated"],
            row["success"],
            row["failed"],
        ]
        values.extend(fmt(row["judges"].get(model), 2) for model in judge_models)
        values.append(fmt(row["avg"], 2))
        values.append(fmt(row["addie"], 2))
        values.append(fmt(row["trajectory"], 2))
        values.extend(fmt(row["difficulty"].get(difficulty), 2) for difficulty in DIFFICULTIES)
        values.extend(fmt(row["phases"].get(phase), 1) for phase in PHASES)
        writer.writerow(values)

    return buffer.getvalue()


def rounded(value: float | None, decimals: int) -> float | None:
    if value is None:
        return None
    return round(value, decimals)


def render_json(rows: list[dict[str, Any]], judge_models: list[str], labels: dict[str, str]) -> str:
    model_count = len(judge_models)
    evaluator_rows = []
    component_rows = []
    difficulty_rows = []
    phase_rows = []

    for row in rows:
        base = {
            "agent_id": row["agent_id"],
            "agent": row["agent"],
        }
        evaluator_rows.append(
            {
                **base,
                "evaluated": row["evaluated"],
                "success": row["success"],
                "failed": row["failed"],
                "scores": {
                    labels[model]: rounded(row["judges"].get(model), 2)
                    for model in judge_models
                },
                "avg": rounded(row["avg"], 2),
            }
        )
        component_rows.append(
            {
                **base,
                "scores": {
                    "Total": rounded(row["avg"], 2),
                    "ADDIE": rounded(row["addie"], 2),
                    "Trajectory": rounded(row["trajectory"], 2),
                },
            }
        )
        difficulty_rows.append(
            {
                **base,
                "scores": {
                    difficulty: rounded(row["difficulty"].get(difficulty), 2)
                    for difficulty in DIFFICULTIES
                },
            }
        )
        phase_rows.append(
            {
                **base,
                "scores": {
                    PHASE_LABELS[phase]: rounded(row["phases"].get(phase), 1)
                    for phase in PHASES
                },
            }
        )

    payload = {
        "judge_models": [{"model": model, "label": labels[model]} for model in judge_models],
        "tables": {
            "evaluator_model": {
                "title": "Evaluator Model",
                "columns": [labels[model] for model in judge_models] + ["Avg"],
                "rows": evaluator_rows,
            },
            "score_components": {
                "title": "Score Components",
                "columns": ["Total", "ADDIE", "Trajectory"],
                "rows": component_rows,
            },
            "difficulty": {
                "title": "Difficulty",
                "columns": DIFFICULTIES,
                "rows": difficulty_rows,
            },
            "phase_wise": {
                "title": (
                    f"Phase-wise ({model_count}-model avg)"
                    if model_count
                    else "Phase-wise"
                ),
                "columns": [PHASE_LABELS[phase] for phase in PHASES],
                "rows": phase_rows,
            },
        },
        "notes": [
            "Evaluator model columns are the mean of per-judge final scores when judge details are available.",
            "Avg is the mean across evaluator model columns, falling back to stored total_score when judge details are absent.",
            "Score Components uses stored total_score, addie_median/addie_score, and trajectory_score averaged across evaluated scenarios.",
            "Difficulty columns use stored total_score grouped by scenario difficulty.",
            "Phase-wise columns use phase_scores available in comparison_report artifacts.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_formats(value: str) -> list[str]:
    aliases = {"markdown": "md"}
    formats: list[str] = []
    for item in value.split(","):
        fmt = aliases.get(item.strip().lower(), item.strip().lower())
        if not fmt:
            continue
        if fmt not in {"csv", "md", "json"}:
            raise argparse.ArgumentTypeError(
                f"Unsupported format '{item}'. Use a comma-separated subset of csv,md,json."
            )
        if fmt not in formats:
            formats.append(fmt)
    if not formats:
        raise argparse.ArgumentTypeError("At least one format is required.")
    return formats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark result artifacts into paper-style tables."
    )
    parser.add_argument("run_dir", type=Path, help="Benchmark run directory.")
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=None,
        help="Scenario directory. Defaults to <repo>/scenarios.",
    )
    parser.add_argument(
        "--formats",
        type=parse_formats,
        default=["csv", "md", "json"],
        help="Comma-separated output formats written into run_dir. Default: csv,md,json.",
    )
    parser.add_argument(
        "--include-counts",
        action="store_true",
        help="Include evaluated/success/failed counts in Markdown output.",
    )
    parser.add_argument(
        "--no-bold-best",
        action="store_true",
        help="Do not bold the best value in each numeric column.",
    )
    return parser.parse_args()


def render(
    fmt: str,
    rows: list[dict[str, Any]],
    judge_models: list[str],
    labels: dict[str, str],
    include_counts: bool,
    bold_best: bool,
) -> str:
    if fmt in {"markdown", "md"}:
        return render_markdown(rows, judge_models, labels, include_counts, bold_best)
    if fmt == "csv":
        return render_csv(rows, judge_models, labels)
    if fmt == "json":
        return render_json(rows, judge_models, labels)
    raise ValueError(f"Unsupported format: {fmt}")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    repo_root = infer_repo_root(run_dir)
    scenario_root = (args.scenario_root or repo_root / "scenarios").resolve()

    records = load_records(run_dir)
    if not records:
        raise SystemExit(f"No comparison artifacts found under {run_dir}")

    judge_models = collect_judge_models(records)
    labels = make_unique_labels(judge_models)
    aggregates = aggregate(records, scenario_root)
    rows = build_rows(aggregates, judge_models)

    bold_best = not args.no_bold_best

    written: list[Path] = []
    for fmt in args.formats:
        output_path = run_dir / f"aggregate_table.{fmt}"
        content = render(
            fmt,
            rows,
            judge_models,
            labels,
            args.include_counts,
            bold_best,
        )
        write_output(output_path, content)
        written.append(output_path)

    print("Wrote aggregate tables:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
