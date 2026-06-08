"""
ISD Evaluator CLI

교수설계 Agent 평가 CLI 인터페이스
"""

import json
import os
from pathlib import Path
from typing import Optional

# .env 파일에서 환경변수 로드
try:
    from dotenv import load_dotenv
    current_dir = Path(__file__).parent
    for _ in range(5):
        env_file = current_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
        current_dir = current_dir.parent
    else:
        load_dotenv()
except ImportError:
    pass

import typer

from isd_evaluator.metrics import CompositeEvaluator, MultiJudgeEvaluator
from isd_evaluator.metrics.multi_judge import load_judges_from_env, DEFAULT_JUDGES
from isd_evaluator.runners import AgentRunner
from isd_evaluator.reporters import ComparisonReporter

app = typer.Typer(
    name="isd-evaluator",
    help="교수설계 Agent 평가 CLI",
    add_completion=False,
)


def load_json(path: Path) -> dict:
    """JSON 파일 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """JSON 파일 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


@app.command()
def evaluate(
    output_file: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="ADDIE output JSON file to evaluate",
        exists=True,
    ),
    scenario_file: Optional[Path] = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Original scenario JSON file (optional)",
    ),
    trajectory_file: Optional[Path] = typer.Option(
        None,
        "--trajectory",
        "-t",
        help="Trajectory JSON file (optional)",
    ),
    result_file: Optional[Path] = typer.Option(
        None,
        "--result",
        "-r",
        help="Evaluation result save path",
    ),
    use_llm: bool = typer.Option(
        True,
        "--use-llm/--no-llm",
        help="Use LLM for ADDIE evaluation",
    ),
    multi_judge: bool = typer.Option(
        False,
        "--multi-judge/--single-judge",
        help="Use multi-judge evaluation",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output mode",
    ),
) -> None:
    """
    단일 ADDIE 산출물을 평가합니다.

    Example:
        isd-evaluator evaluate --output result.json --scenario scenario.json
    """
    if verbose:
        typer.echo("ADDIE 산출물 평가")
        typer.echo(f"  산출물: {output_file}")
        if scenario_file:
            typer.echo(f"  시나리오: {scenario_file}")
        if trajectory_file:
            typer.echo(f"  궤적: {trajectory_file}")
        typer.echo()

    # 데이터 로드
    addie_output = load_json(output_file)

    scenario = None
    if scenario_file:
        scenario = load_json(scenario_file)

    trajectory = None
    metadata = None
    if trajectory_file:
        traj_data = load_json(trajectory_file)
        trajectory = traj_data.get("trajectory", traj_data)
        metadata = traj_data.get("metadata")

    # Run evaluation
    if multi_judge:
        judges = load_judges_from_env() or DEFAULT_JUDGES
        typer.echo(f"Using Multi-Judge evaluation ({len(judges)} LLMs)...")
        evaluator = MultiJudgeEvaluator(judges=judges, parallel=True, max_workers=max(1, len(judges)))
        multi_result = evaluator.evaluate(
            addie_output=addie_output,
            scenario=scenario,
            trajectory=trajectory,
            metadata=metadata,
        )

        # Multi-judge result output
        typer.echo("\n=== Evaluation Results (Multi-Judge) ===")
        typer.echo(f"Total Score (median): {multi_result.total_score:.1f}/100")
        typer.echo(f"ADDIE Score (median): {multi_result.median_addie_score:.1f}/100")
        if multi_result.median_trajectory_score:
            typer.echo(f"Trajectory Score (median): {multi_result.median_trajectory_score:.1f}/100")

        typer.echo("\nIndividual Judge Scores:")
        for jr in multi_result.judge_results:
            if jr.normalized_score is not None:
                typer.echo(f"  [{jr.provider}] {jr.model}: {jr.normalized_score:.1f}")
            else:
                typer.echo(f"  [{jr.provider}] {jr.model}: Error - {jr.error}")

        stats = multi_result.agreement_stats
        if "addie_stdev" in stats:
            typer.echo(f"\nAgreement: stdev={stats['addie_stdev']:.2f}, range={stats['addie_range']:.1f}")
            if stats.get("addie_high_disagreement"):
                typer.echo("  ⚠️  High disagreement detected!")

        # Save result
        if result_file:
            save_json(result_file, multi_result.to_dict())
            typer.echo(f"\nResult saved: {result_file}")
        return

    evaluator = CompositeEvaluator(use_llm=use_llm)
    score = evaluator.evaluate(
        addie_output=addie_output,
        scenario=scenario,
        trajectory=trajectory,
        metadata=metadata,
    )

    # Result output
    typer.echo("\n=== Evaluation Results ===")
    typer.echo(f"Total Score: {score.total:.1f}/100")
    typer.echo(f"ADDIE Score: {score.addie_score:.1f}/100")

    if score.trajectory:
        typer.echo(f"Trajectory 점수: {score.trajectory_score:.1f}/100")

    typer.echo("\nADDIE 단계별 점수:")
    for phase, ps in score.addie.phases.items():
        typer.echo(f"  {phase.value.capitalize()}: {ps.percentage:.1f}% ({ps.raw_score:.1f}/{ps.max_score:.0f})")

    typer.echo("\n항목별 점수 (0-10):")
    for phase, ps in score.addie.phases.items():
        items_str = ", ".join([f"{item.item_id}:{item.score:.1f}" for item in ps.items])
        typer.echo(f"  {phase.value.capitalize()}: {items_str}")

    # 강점/개선점 출력
    if score.addie.strengths:
        typer.echo("\n강점:")
        for s in score.addie.strengths[:3]:
            typer.echo(f"  - {s}")

    if score.addie.improvements:
        typer.echo("\n개선점:")
        for i in score.addie.improvements[:3]:
            typer.echo(f"  - {i}")

    # 결과 저장
    if result_file:
        result = score.addie.to_dict()
        result["total"] = score.total
        if score.trajectory:
            result["trajectory_score"] = score.trajectory_score
        save_json(result_file, result)
        typer.echo(f"\n결과 저장됨: {result_file}")


@app.command()
def compare(
    scenario_file: Path = typer.Option(
        ...,
        "--scenario",
        "-s",
        help="Scenario JSON file",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-d",
        help="Output directory",
    ),
    agents: Optional[str] = typer.Option(
        None,
        "--agents",
        "-a",
        help="Agent list (comma-separated)",
    ),
    run_agents: bool = typer.Option(
        False,
        "--run/--no-run",
        help="Run agents (CLI call)",
    ),
    use_llm: bool = typer.Option(
        True,
        "--use-llm/--no-llm",
        help="Use LLM for ADDIE evaluation",
    ),
    multi_judge: bool = typer.Option(
        False,
        "--multi-judge/--single-judge",
        help="Use multi-judge evaluation",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output mode",
    ),
) -> None:
    """
    여러 Agent를 비교 평가합니다.

    Example:
        isd-evaluator compare --scenario scenario.json --output-dir results/ --run
    """
    if verbose:
        typer.echo("Agent 비교 평가")
        typer.echo(f"  시나리오: {scenario_file}")
        typer.echo(f"  출력: {output_dir}")
        typer.echo()

    # 시나리오 로드
    scenario = load_json(scenario_file)

    # Agent 목록
    agent_list = None
    if agents:
        agent_list = [a.strip() for a in agents.split(",")]

    results = []

    if run_agents:
        # Agent 실행
        if verbose:
            typer.echo("Agent 실행 중...")

        runner = AgentRunner()
        run_results = runner.run_all_agents(
            scenario_path=scenario_file,
            output_dir=output_dir,
            agents=agent_list,
        )

        for r in run_results:
            if r.get("success"):
                results.append(r)
            else:
                typer.echo(f"경고: {r.get('error', 'Unknown error')}", err=True)
    else:
        # 기존 결과 파일 로드
        if verbose:
            typer.echo("기존 결과 로드 중...")

        output_dir = Path(output_dir)
        target_agents = agent_list or ["eduplanner", "baseline", "react-isd"]

        for agent_id in target_agents:
            output_path = output_dir / f"{agent_id}_output.json"
            trajectory_path = output_dir / f"{agent_id}_trajectory.json"

            if output_path.exists():
                addie_output = load_json(output_path)
                trajectory = None
                metadata = None

                if trajectory_path.exists():
                    traj_data = load_json(trajectory_path)
                    trajectory = traj_data.get("trajectory", {})
                    metadata = traj_data.get("metadata", {})

                results.append({
                    "agent_id": agent_id,
                    "addie_output": addie_output,
                    "trajectory": trajectory,
                    "metadata": metadata,
                })

    if not results:
        typer.echo("평가할 결과가 없습니다.", err=True)
        raise typer.Exit(code=1)

    # Comparison evaluation
    if verbose:
        typer.echo("Evaluating...")

    if multi_judge:
        judges = load_judges_from_env() or DEFAULT_JUDGES
        typer.echo(f"Using Multi-Judge evaluation ({len(judges)} LLMs)...")
        for judge in judges:
            typer.echo(f"  - {judge.model} ({judge.provider})")
        typer.echo()

        evaluator = MultiJudgeEvaluator(judges=judges, parallel=True, max_workers=max(1, len(judges)))
        comparison = evaluator.compare_agents(results, scenario)

        # Print detailed results for multi-judge
        typer.echo("\n" + "=" * 70)
        typer.echo("MULTI-JUDGE COMPARISON RESULTS")
        typer.echo("=" * 70)

        for rank_info in comparison["rankings"]:
            typer.echo(f"\n#{rank_info['rank']} {rank_info['agent_id']}")
            typer.echo(f"   Total Score:   {rank_info['total_score']:.1f}/100")
            typer.echo(f"   ADDIE median:  {rank_info['addie_median']:.1f}  mean: {rank_info['addie_mean']:.1f}")
            typer.echo(f"   Reliability:   {rank_info['reliability_score']:.2f} ({rank_info['reliability']})")
            typer.echo(f"   Score Range:   {rank_info['score_range']}  stdev: {rank_info['stdev']:.2f}")
            if rank_info['high_disagreement']:
                typer.echo(f"   ⚠️  HIGH DISAGREEMENT")
            typer.echo(f"   Judges:")
            for j in rank_info['judges']:
                if j['addie_score'] is not None:
                    dev = j['deviation_from_median']
                    dev_str = f"+{dev:.1f}" if dev >= 0 else f"{dev:.1f}"
                    typer.echo(f"      {j['provider']:12} {j['addie_score']:5.1f} ({dev_str})")
                else:
                    typer.echo(f"      {j['provider']:12} ERROR: {j['error']}")

        typer.echo("\n" + "=" * 70)
    else:
        evaluator = CompositeEvaluator(use_llm=use_llm)
        comparison = evaluator.compare_agents(results, scenario)

    # 리포트 생성
    reporter = ComparisonReporter()

    # 콘솔 출력
    reporter.print_summary(comparison)

    # 마크다운 리포트
    md_path = Path(output_dir) / "comparison_report.md"
    reporter.generate_markdown(comparison, scenario, md_path)
    typer.echo(f"마크다운 리포트: {md_path}")

    # JSON 리포트
    json_path = Path(output_dir) / "comparison_report.json"
    reporter.generate_json(comparison, scenario, json_path)
    typer.echo(f"JSON 리포트: {json_path}")


@app.command()
def info() -> None:
    """
    ISD Evaluator 정보를 출력합니다.
    """
    typer.echo("ISD Evaluator")
    typer.echo("=============")
    typer.echo()
    typer.echo("교수설계 Agent 평가 시스템")
    typer.echo()
    typer.echo("평가 메트릭:")
    typer.echo("  1. ADDIE 루브릭 평가 (산출물 품질)")
    typer.echo("     - Analysis: A1(요구분석), A2(학습자/환경), A3(과제/목표)")
    typer.echo("     - Design: D1(목표/평가), D2(교수전략), D3(구조설계)")
    typer.echo("     - Development: Dev1(프로토타입), Dev2(검토)")
    typer.echo("     - Implementation: I1(실행준비), I2(실행)")
    typer.echo("     - Evaluation: E1(형성), E2(총괄), E3(개선)")
    typer.echo()
    typer.echo("  2. Trajectory (생성 과정) - BFCL 기반")
    typer.echo("     - Tool Correctness (도구 정확성)")
    typer.echo("     - Argument Accuracy (인자 정확성)")
    typer.echo("     - Redundancy Avoidance (중복 회피)")
    typer.echo("     - Result Utilization (결과 활용도)")
    typer.echo()
    typer.echo("점수 체계:")
    typer.echo("  - ADDIE: 0-10점 (항목별) x 13개 항목 -> 정규화 100점")
    typer.echo("  - Trajectory: 0-25점 x 4개 항목 = 100점")
    typer.echo("  - 종합: ADDIE x 0.7 + Trajectory x 0.3")
    typer.echo()
    typer.echo("지원 Agent:")
    typer.echo("  - eduplanner (3-Agent 협업)")
    typer.echo("  - baseline (단일 프롬프트)")
    typer.echo("  - react-isd (ReAct 패턴)")
    typer.echo()
    typer.echo("버전: 0.2.0")


def main() -> None:
    """CLI 메인 엔트리포인트"""
    app()


if __name__ == "__main__":
    main()
