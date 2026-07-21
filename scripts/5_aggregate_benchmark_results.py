#!/usr/bin/env python3
"""Aggregate ISD benchmark artifacts into paper-style tables + analyses.

One entry point for post-run analysis of a SINGLE results dir (this merges the
former stats_rq1.py and 6_aggregate_token_usage.py). For the multi-run model
ladder (RQ1/RQ2 across sizes and repeated runs), use
``7_pool_ladder_runs.py`` instead — this script does not pool across runs.
Selected via ``--sections`` (comma-separated, or ``all``); default is ``table``:

  table    : the paper's main comparison table (Agent | judge scores | Avg |
             difficulty | phase), written to run_dir as csv/md/json.
  rq1      : proposed vs each baseline, paired Wilcoxon + Holm + bootstrap 95% CI.
  tokens   : per-agent token usage (prompt/completion/total/calls, per scenario).
  errors   : per-agent hard-failure rate (from *_log.txt) + harness-internal
             error reasons (from trajectory metadata.errors), bucketed.

Works on both a flat single-run dir and a nested dataset dir.

Examples:
  python scripts/5_aggregate_benchmark_results.py results/test_30_benchmark_...
  python scripts/5_aggregate_benchmark_results.py results/test_30_... --sections all
  python scripts/5_aggregate_benchmark_results.py results/single_... --sections tokens,errors
  python scripts/5_aggregate_benchmark_results.py results/test_90_... --sections rq1 --baselines baseline,react-isd
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
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
    alignment_scores: list[float] = field(default_factory=list)
    judge_scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    difficulty_scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    phase_scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def mean_total(self) -> float | None:
        return mean_or_none(self.total_scores)

    def mean_addie(self) -> float | None:
        return mean_or_none(self.addie_scores)

    def mean_trajectory(self) -> float | None:
        return mean_or_none(self.trajectory_scores)

    def mean_alignment(self) -> float | None:
        return mean_or_none(self.alignment_scores)

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


def apply_alignment_scores(run_dir: Path, aggregates: dict[str, AgentAggregate]) -> int:
    """Fold per-scenario ``alignment_scores.json`` files (written by
    scripts/6_score_alignment.py) into the aggregates. The alignment composite
    is a separate, deterministic [0,1] metric — it is reported alongside and
    NEVER mixed into the 0.7/0.3 Total. Returns the number of files found."""
    found = 0
    for scores_path in sorted(run_dir.rglob("alignment_scores.json")):
        try:
            payload = load_json(scores_path)
        except (json.JSONDecodeError, OSError):
            continue
        agents = payload.get("agents")
        if not isinstance(agents, dict):
            continue
        found += 1
        for agent_id, score in agents.items():
            composite = (score or {}).get("composite")
            if composite is None:
                continue
            if agent_id not in aggregates:
                aggregates[agent_id] = AgentAggregate(agent_id=agent_id)
            aggregates[agent_id].alignment_scores.append(float(composite))
    return found


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
                "alignment": agg.mean_alignment(),
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
    columns.append(("alignment", ("alignment", None)))
    columns.extend((f"difficulty:{difficulty}", ("difficulty", difficulty)) for difficulty in DIFFICULTIES)
    columns.extend((f"phase:{phase}", ("phases", phase)) for phase in PHASES)
    return columns


def row_value(row: dict[str, Any], source: tuple[str, str | None]) -> float | None:
    section, key = source
    if section in {"avg", "addie", "trajectory", "alignment"}:
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

    headers = ["Agent", "Total", "ADDIE", "Trajectory", "Alignment"]
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
                format_number(
                    row["alignment"],
                    3,
                    is_best(row["alignment"], best.get("alignment")),
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
    header.extend(["Avg", "ADDIE", "Trajectory", "Alignment", "Easy", "Medium", "Hard"])
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
        values.append(fmt(row["alignment"], 3))
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
                    "Alignment": rounded(row["alignment"], 3),
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
                "columns": ["Total", "ADDIE", "Trajectory", "Alignment"],
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
            "Alignment is the deterministic, LLM-free constructive-alignment composite in [0,1] "
            "(scripts/6_score_alignment.py); it is reported separately and never mixed into Total.",
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


# ═══════════════════════════════════════════════════════════════════════════
# Analysis sections (merged from the former stats_rq1.py /
# 6_aggregate_token_usage.py). Each prints a stdout report; only the default
# "table" section writes files. Selected via --sections.
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PROPOSED = "alignmentgraph-isd"
DEFAULT_BASELINES = ["baseline", "react-isd", "dick-carey-agent", "addie-agent", "rpisd-agent", "eduplanner"]


# ── shared statistics (paired Wilcoxon + Holm + bootstrap CI) ────────────────

def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank (zeros discarded, tie-corrected normal
    approximation). Returns (W_plus, p_value, n_effective)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0, 0
    absd = sorted((abs(x), i) for i, x in enumerate(d))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1][0] == absd[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[absd[k][1]] = avg
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    cnt = Counter(abs(x) for x in d)
    var -= sum(t**3 - t for t in cnt.values()) / 48
    if var <= 0:
        return w_plus, 1.0, n
    z = (w_plus - mu) / math.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return w_plus, p, n


def bootstrap_ci(diffs: list[float], n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    if not diffs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def holm_correct(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted: list[float] = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# ── loaders that handle BOTH a flat single-run dir and a nested dataset dir ──

def iter_comparison_reports(run_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield (scenario_id, comparison_report.json path) for a flat single-run
    dir (one report at the top) OR a dataset dir (one per scenario subfolder)."""
    flat = run_dir / "comparison_report.json"
    if flat.exists():
        yield run_dir.name, flat
    for child in sorted(run_dir.iterdir()):
        if child.is_dir():
            rep = child / "comparison_report.json"
            if rep.exists():
                yield child.name, rep


def load_rankings(run_dir: Path) -> dict[str, dict[str, dict]]:
    """scenario_id -> agent_id -> ranking dict (total_score, addie_mean, ...)."""
    out: dict[str, dict[str, dict]] = {}
    for scenario_id, rep in iter_comparison_reports(run_dir):
        try:
            data = json.loads(rep.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out[scenario_id] = {r["agent_id"]: r for r in data.get("comparison", {}).get("rankings", [])}
    return out


# ── RQ1: proposed vs each baseline, paired on total_score ────────────────────

def section_rq1(run_dir: Path, proposed: str, baselines: list[str]) -> None:
    rankings = load_rankings(run_dir)
    print(f"\n## RQ1 — {proposed} vs baselines ({len(rankings)} scenarios), paired total_score")
    results = []
    for b in baselines:
        diffs = [
            per[proposed]["total_score"] - per[b]["total_score"]
            for per in rankings.values() if proposed in per and b in per
        ]
        if not diffs:
            continue
        _, p, _ = wilcoxon_signed_rank(diffs)
        lo, hi = bootstrap_ci(diffs)
        results.append((b, len(diffs), _mean(diffs), lo, hi, p))
    if not results:
        print("  (no paired scenarios found)")
        return
    holm = holm_correct([r[5] for r in results])
    for (b, n, md, lo, hi, p), ph in zip(results, holm):
        print(f"  {b:18s} n={n:2d}  mean diff={md:+6.2f}  95% CI [{lo:+6.2f}, {hi:+6.2f}]  p={p:.2e}  p_holm={ph:.2e}")


# ── Token usage (merged from 6_aggregate_token_usage.py) ─────────────────────

def _extract_token_usage(traj_doc: dict) -> tuple[int, int, int] | None:
    metadata = traj_doc.get("metadata") or {}
    trajectory = traj_doc.get("trajectory") or {}
    usage = metadata.get("token_usage")
    if isinstance(usage, dict) and (usage.get("total_tokens") or usage.get("prompt_tokens") or usage.get("completion_tokens")):
        return int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0), int(usage.get("llm_calls", 0) or 0)
    inner = trajectory.get("token_usage")
    if isinstance(inner, dict) and (inner.get("prompt_tokens") or inner.get("completion_tokens")):
        return int(inner.get("prompt_tokens", 0) or 0), int(inner.get("completion_tokens", 0) or 0), int(trajectory.get("llm_calls", 0) or 0)
    total = metadata.get("total_tokens")
    if isinstance(total, (int, float)) and total:
        return 0, 0, 0
    return None


def section_tokens(run_dir: Path) -> None:
    per_agent: dict[str, dict] = defaultdict(lambda: {"prompt": 0, "completion": 0, "total_only": 0, "calls": 0, "n": 0, "missing": 0})
    for traj in sorted(run_dir.rglob("*_trajectory.json")):
        try:
            doc = json.loads(traj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        agent_id = doc.get("agent_id") or traj.stem.replace("_trajectory", "")
        row = per_agent[agent_id]
        row["n"] += 1
        extracted = _extract_token_usage(doc)
        if extracted is None:
            total = (doc.get("metadata") or {}).get("total_tokens")
            if isinstance(total, (int, float)) and total:
                row["total_only"] += int(total)
            else:
                row["missing"] += 1
            continue
        prompt, completion, calls = extracted
        row["prompt"] += prompt
        row["completion"] += completion
        row["calls"] += calls
        if not (prompt or completion):
            total = (doc.get("metadata") or {}).get("total_tokens")
            if isinstance(total, (int, float)):
                row["total_only"] += int(total)
    print(f"\n## Token usage ({sum(r['n'] for r in per_agent.values())} runs)")
    if not per_agent:
        print("  (no *_trajectory.json found)")
        return
    rows = []
    for agent_id, r in per_agent.items():
        total = r["prompt"] + r["completion"] + r["total_only"]
        n = max(r["n"], 1)
        rows.append((agent_id, r["n"], r["prompt"], r["completion"], total, r["calls"], total / n, r["calls"] / n, r["missing"]))
    rows.sort(key=lambda x: x[4], reverse=True)
    print(f"  {'agent':28s} {'n':>3s} {'prompt':>10s} {'compl.':>10s} {'total':>11s} {'calls':>6s} {'tot/scen':>10s} {'calls/scen':>10s} {'missing':>8s}")
    for agent_id, n, p, c, tot, calls, tps, cps, miss in rows:
        print(f"  {agent_id:28s} {n:3d} {p:10d} {c:10d} {tot:11d} {calls:6d} {tps:10.1f} {cps:10.2f} {miss:8d}")


# ── Error rate + reasons (NEW) ───────────────────────────────────────────────

def _normalize_error(msg: str) -> str:
    """Collapse an error message to a comparable reason bucket (strip ids,
    numbers, and paths so the same failure mode groups together)."""
    msg = re.sub(r"0x[0-9a-fA-F]+", "0x…", str(msg))
    msg = re.sub(r"\b\d+\b", "N", msg)
    msg = re.sub(r"(/[^\s'\"]+)+", "<path>", msg)
    return msg.strip()[:120]


def section_errors(run_dir: Path) -> None:
    """Per-agent hard-failure rate (from *_log.txt Status) + harness-internal
    error reasons (from trajectory metadata.errors, present even on a SUCCESS
    run — e.g. a truncated LLM response that degraded one section)."""
    per_agent: dict[str, dict] = defaultdict(lambda: {"runs": 0, "failed": 0, "internal_errs": 0, "reasons": Counter()})

    for log in sorted(run_dir.rglob("*_log.txt")):
        agent_id = log.stem.replace("_log", "")
        text = log.read_text(encoding="utf-8", errors="replace")
        row = per_agent[agent_id]
        row["runs"] += 1
        if "Status: FAILED" in text:
            row["failed"] += 1
            m = re.search(r"^Error:\s*(.+)$", text, re.MULTILINE)
            if m:
                row["reasons"][_normalize_error(m.group(1))] += 1

    for traj in sorted(run_dir.rglob("*_trajectory.json")):
        try:
            doc = json.loads(traj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        agent_id = doc.get("agent_id") or traj.stem.replace("_trajectory", "")
        errors = (doc.get("metadata") or {}).get("errors") or []
        if isinstance(errors, list) and errors:
            row = per_agent[agent_id]
            row["internal_errs"] += len(errors)
            for e in errors:
                row["reasons"][_normalize_error(e)] += 1

    print(f"\n## Error rate & reasons ({sum(r['runs'] for r in per_agent.values())} runs)")
    if not per_agent:
        print("  (no *_log.txt found)")
        return
    print(f"  {'agent':28s} {'runs':>5s} {'failed':>7s} {'fail%':>6s} {'internalErr':>12s}  top reasons")
    for agent_id in sorted(per_agent, key=lambda a: (per_agent[a]['failed'], per_agent[a]['internal_errs']), reverse=True):
        r = per_agent[agent_id]
        rate = 100.0 * r["failed"] / r["runs"] if r["runs"] else 0.0
        reasons = "; ".join(f"{reason} (×{cnt})" for reason, cnt in r["reasons"].most_common(2)) or "-"
        print(f"  {agent_id:28s} {r['runs']:5d} {r['failed']:7d} {rate:5.1f}% {r['internal_errs']:12d}  {reasons}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark result artifacts into paper-style tables + analysis sections."
    )
    parser.add_argument("run_dir", type=Path, help="Benchmark run directory (flat single-run or nested dataset).")
    parser.add_argument(
        "--sections",
        default="table",
        help="Comma-separated analyses to run: table,rq1,tokens,errors (or 'all'). Default: table.",
    )
    parser.add_argument("--proposed", default=DEFAULT_PROPOSED, help="Proposed agent id for the RQ1 section.")
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES), help="Comma-separated baseline ids for RQ1.")
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


_ALL_SECTIONS = ["table", "rq1", "tokens", "errors"]


def _run_table_section(run_dir: Path, args: argparse.Namespace) -> None:
    repo_root = infer_repo_root(run_dir)
    scenario_root = (args.scenario_root or repo_root / "scenarios").resolve()
    records = load_records(run_dir)
    if not records:
        print(f"## Comparison table\n  (no comparison artifacts found under {run_dir})")
        return
    judge_models = collect_judge_models(records)
    labels = make_unique_labels(judge_models)
    aggregates = aggregate(records, scenario_root)
    n_alignment_files = apply_alignment_scores(run_dir, aggregates)
    if n_alignment_files:
        print(f"Folded alignment_scores.json from {n_alignment_files} scenario dir(s) "
              "into the Alignment column.")
    rows = build_rows(aggregates, judge_models)
    bold_best = not args.no_bold_best

    written: list[Path] = []
    for fmt in args.formats:
        output_path = run_dir / f"aggregate_table.{fmt}"
        content = render(fmt, rows, judge_models, labels, args.include_counts, bold_best)
        write_output(output_path, content)
        written.append(output_path)
    print("Wrote aggregate tables:")
    for path in written:
        print(f"  {path}")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()

    requested = args.sections.strip().lower()
    sections = _ALL_SECTIONS if requested in ("all", "*") else [s.strip() for s in requested.split(",") if s.strip()]
    unknown = [s for s in sections if s not in _ALL_SECTIONS]
    if unknown:
        raise SystemExit(f"Unknown section(s): {unknown}. Valid: {_ALL_SECTIONS} (or 'all').")

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    print(f"=== {run_dir.name} ===")
    if "table" in sections:
        _run_table_section(run_dir, args)
    if "rq1" in sections:
        section_rq1(run_dir, args.proposed, baselines)
    if "tokens" in sections:
        section_tokens(run_dir)
    if "errors" in sections:
        section_errors(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
