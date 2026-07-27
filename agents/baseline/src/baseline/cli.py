"""
Baseline ISD Agent CLI

Single prompt ADDIE generator CLI interface.
Uses the shared provider-neutral LLM configuration.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Load environment variables from .env
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

from baseline.generator import BaselineGenerator
from shared.llm import llm_config_from_env

app = typer.Typer(
    name="baseline",
    help="Baseline ISD Agent CLI (provider-neutral LLM support)",
    add_completion=False,
)


def load_scenario(input_path: Path) -> dict:
    """시나리오 JSON 파일 로드"""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        typer.echo(f"Error: input file not found: {input_path}", err=True)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        typer.echo(f"Error: JSON parsing failed: {e}", err=True)
        raise typer.Exit(code=1)


def save_output(output_path: Path, data: dict) -> None:
    """결과를 JSON 파일로 저장"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        typer.echo(f"Error: failed to save output file: {e}", err=True)
        raise typer.Exit(code=1)


def export_slides_to_marp(
    addie_output: dict,
    output_path: Path,
    title: str = "Training Slides",
    verbose: bool = False,
) -> Optional[str]:
    """ADDIE 출력에서 슬라이드를 추출하여 Marp Markdown 파일로 저장"""
    try:
        # shared/utils/marp_exporter 임포트
        # cli.py 위치: agents/baseline/src/baseline/cli.py
        # project_root: 프로젝트 루트 (5단계 상위)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        from shared.utils.marp_exporter import export_to_file

        # development.materials에서 slide_contents 추출
        development = addie_output.get("development", {})
        materials = development.get("materials", [])

        all_slides: List[Dict[str, Any]] = []
        for material in materials:
            if material and material.get("slide_contents"):
                slide_contents = material.get("slide_contents", [])
                for slide in slide_contents:
                    if hasattr(slide, "model_dump"):
                        all_slides.append(slide.model_dump())
                    elif isinstance(slide, dict):
                        all_slides.append(slide)

        if not all_slides:
            if verbose:
                typer.echo("  No slide content, so no Marp file is generated.")
            return None

        # 출력 파일 경로 생성
        slide_output_path = output_path.with_name(
            output_path.stem.replace("_output", "") + "_slides.md"
        )

        # Marp Markdown 파일 생성
        result_path = export_to_file(
            slide_contents=all_slides,
            output_path=str(slide_output_path),
            title=title,
            theme="default",
        )

        return result_path

    except Exception as e:
        if verbose:
            typer.echo(f"  Slide file generation failed: {e}", err=True)
        return None


@app.command()
def run(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to the input scenario JSON file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_file: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to the output result JSON file",
    ),
    trajectory_file: Optional[Path] = typer.Option(
        None,
        "--trajectory",
        "-t",
        help="Path to the trajectory JSON file (optional)",
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="LLM model override (otherwise resolved from shared LLM config)",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        help="Generation temperature (0.0-2.0)",
        min=0.0,
        max=2.0,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output mode",
    ),
) -> None:
    """
    Generate ADDIE outputs from an instructional design scenario.

    Example:
        baseline run --input scenario.json --output result.json
        baseline run -i scenario.json -o result.json -t trajectory.json -v
    """
    if verbose:
        typer.echo("Baseline ISD Agent running")
        typer.echo(f"  Input: {input_file}")
        typer.echo(f"  Output: {output_file}")
        typer.echo(f"  Model: {model}")
        typer.echo(f"  Temperature: {temperature}")
        typer.echo()

    # 시나리오 로드
    if verbose:
        typer.echo("Loading scenario...")
    scenario = load_scenario(input_file)

    if verbose:
        typer.echo(f"  Scenario ID: {scenario.get('scenario_id', 'unknown')}")
        typer.echo(f"  Title: {scenario.get('title', 'No title')}")
        typer.echo()

    # 생성기 초기화
    if verbose:
        typer.echo("Initializing generator...")

    llm_config = llm_config_from_env(model=model, temperature=temperature, max_tokens=32768)
    generator = BaselineGenerator(llm_config=llm_config)

    # 실행
    if verbose:
        typer.echo("Generating instructional design...")
        typer.echo()

    try:
        result = generator.generate(scenario)
    except Exception as e:
        typer.echo(f"Error: generation failed: {e}", err=True)
        raise typer.Exit(code=1)

    # ADDIE 출력 저장
    addie_output = result["addie_output"]
    save_output(output_file, addie_output)

    if verbose:
        typer.echo(f"ADDIE output saved: {output_file}")

    # 슬라이드 Marp Markdown 파일 생성
    slide_path = export_slides_to_marp(
        addie_output=addie_output,
        output_path=output_file,
        title=scenario.get("title", "Training Slides"),
        verbose=verbose,
    )
    if slide_path and verbose:
        typer.echo(f"Slide file saved: {slide_path}")

    # Trajectory 저장 (선택)
    if trajectory_file:
        trajectory_data = {
            "scenario_id": result["scenario_id"],
            "agent_id": result["agent_id"],
            "timestamp": result["timestamp"],
            "trajectory": result["trajectory"],
            "metadata": result["metadata"],
        }
        save_output(trajectory_file, trajectory_data)

        if verbose:
            typer.echo(f"Trajectory saved: {trajectory_file}")

    # 결과 요약 출력
    if verbose:
        metadata = result["metadata"]
        typer.echo()
        typer.echo("=== Run result summary ===")
        typer.echo(f"  Execution time: {metadata['execution_time_seconds']:.2f}s")
        typer.echo(f"  Total tokens: {metadata['total_tokens']}")
        typer.echo(f"  Cost (USD): ${metadata['cost_usd']:.6f}")

        addie = result["addie_output"]
        typer.echo(f"  Number of learning objectives: {len(addie.get('design', {}).get('learning_objectives', []))}")
        typer.echo(f"  Number of modules: {len(addie.get('development', {}).get('lesson_plan', {}).get('modules', []))}")
        typer.echo(f"  Number of quiz items: {len(addie.get('evaluation', {}).get('quiz_items', []))}")

    typer.echo("Done")


@app.command()
def validate(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to the scenario JSON file to validate",
        exists=True,
    ),
) -> None:
    """
    Validate a scenario JSON file.

    Example:
        baseline validate --input scenario.json
    """
    try:
        scenario = load_scenario(input_file)

        # 필수 필드 확인
        required_fields = ["scenario_id", "title", "context", "learning_goals"]
        missing = [f for f in required_fields if f not in scenario]

        if missing:
            typer.echo(f"Invalid: missing required fields - {missing}", err=True)
            raise typer.Exit(code=1)

        typer.echo("Valid scenario.")
        typer.echo(f"  ID: {scenario['scenario_id']}")
        typer.echo(f"  Title: {scenario['title']}")
        typer.echo(f"  Target audience: {scenario.get('context', {}).get('target_audience', 'Not specified')}")
        typer.echo(f"  Number of goals: {len(scenario.get('learning_goals', []))}")

    except Exception as e:
        typer.echo(f"Invalid scenario: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """
    Print information about the Baseline ISD Agent.
    """
    typer.echo("Baseline ISD Agent")
    typer.echo("==================")
    typer.echo()
    typer.echo("Generates ADDIE instructional design outputs with a single LLM call.")
    typer.echo("Supports hosted and local backends through shared LLM config.")
    typer.echo()
    typer.echo("Features:")
    typer.echo("  - Single prompt based (no iteration)")
    typer.echo("  - Provider-neutral LLM support")
    typer.echo("  - Minimal complexity")
    typer.echo("  - Comparison baseline")
    typer.echo()
    typer.echo("Reference frameworks:")
    typer.echo("  - Bloom's Taxonomy")
    typer.echo("  - Gagné's 9 Events of Instruction")
    typer.echo()
    typer.echo("Version: 0.1.0")


def main() -> None:
    """CLI 메인 엔트리포인트"""
    app()


if __name__ == "__main__":
    main()
