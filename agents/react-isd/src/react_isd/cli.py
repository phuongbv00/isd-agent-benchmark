"""
ReAct-ISD CLI

LangGraph ReAct 패턴 기반 교수설계 에이전트 CLI 인터페이스
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

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

from react_isd.agent import ReActISDAgent
from shared.llm import llm_config_from_env

app = typer.Typer(
    name="react-isd",
    help="ReAct 패턴 기반 교수설계 에이전트 CLI",
    add_completion=False,
)


def load_scenario(input_path: Path) -> dict:
    """시나리오 JSON 파일 로드"""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        typer.echo(f"오류: 입력 파일을 찾을 수 없습니다: {input_path}", err=True)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        typer.echo(f"오류: JSON 파싱 실패: {e}", err=True)
        raise typer.Exit(code=1)


def save_output(output_path: Path, data: dict) -> None:
    """결과를 JSON 파일로 저장"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        typer.echo(f"오류: 출력 파일 저장 실패: {e}", err=True)
        raise typer.Exit(code=1)


def export_slides_to_marp(
    addie_output: dict,
    output_path: Path,
    title: str = "교육 슬라이드",
    verbose: bool = False,
) -> Optional[str]:
    """ADDIE 출력에서 슬라이드를 추출하여 Marp Markdown 파일로 저장"""
    try:
        # shared/utils/marp_exporter 임포트
        # cli.py 위치: agents/react-isd/src/react_isd/cli.py
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
                typer.echo("  슬라이드 콘텐츠가 없어 Marp 파일을 생성하지 않습니다.")
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
            typer.echo(f"  슬라이드 파일 생성 실패: {e}", err=True)
        return None


@app.command()
def run(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="입력 시나리오 JSON 파일 경로",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_file: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="출력 결과 JSON 파일 경로",
    ),
    trajectory_file: Optional[Path] = typer.Option(
        None,
        "--trajectory",
        "-t",
        help="궤적(trajectory) JSON 파일 경로 (선택)",
    ),
    model: str = typer.Option(
        "solar-mini",
        "--model",
        help="사용할 LLM 모델 (기본: solar-mini)",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        help="생성 온도 (0.0-2.0)",
        min=0.0,
        max=2.0,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="상세 출력 모드",
    ),
) -> None:
    """
    교수설계 시나리오를 입력받아 ADDIE 산출물을 생성합니다.

    Example:
        react-isd run --input scenario.json --output result.json
        react-isd run -i scenario.json -o result.json -t trajectory.json -v
    """
    if verbose:
        typer.echo("ReAct-ISD Agent 실행")
        typer.echo(f"  입력: {input_file}")
        typer.echo(f"  출력: {output_file}")
        typer.echo(f"  모델: {model}")
        typer.echo(f"  온도: {temperature}")
        typer.echo()

    # 시나리오 로드
    if verbose:
        typer.echo("시나리오 로드 중...")
    scenario = load_scenario(input_file)

    if verbose:
        typer.echo(f"  시나리오 ID: {scenario.get('scenario_id', 'unknown')}")
        typer.echo(f"  제목: {scenario.get('title', '제목 없음')}")
        typer.echo()

    # 에이전트 초기화
    if verbose:
        typer.echo("에이전트 초기화 중...")

    llm_config = llm_config_from_env(model=model)

    if verbose:
        typer.echo(f"  Model Provider: {llm_config.provider}")
        typer.echo(f"  Model API Spec: {llm_config.api_spec}")
        typer.echo(f"  Model: {llm_config.model}")

    agent = ReActISDAgent(
        temperature=temperature,
        llm_config=llm_config,
    )

    # 실행
    if verbose:
        typer.echo("교수설계 생성 중 (ReAct 패턴)...")
        typer.echo()

    try:
        result = agent.run(scenario)
    except Exception as e:
        typer.echo(f"오류: 에이전트 실행 실패: {e}", err=True)
        raise typer.Exit(code=1)

    # ADDIE 출력 저장
    addie_output = result["addie_output"]
    save_output(output_file, addie_output)

    if verbose:
        typer.echo(f"ADDIE 산출물 저장됨: {output_file}")

    # 슬라이드 Marp Markdown 파일 생성
    slide_path = export_slides_to_marp(
        addie_output=addie_output,
        output_path=output_file,
        title=scenario.get("title", "교육 슬라이드"),
        verbose=verbose,
    )
    if slide_path and verbose:
        typer.echo(f"슬라이드 파일 저장됨: {slide_path}")

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
            typer.echo(f"궤적 저장됨: {trajectory_file}")

    # 결과 요약 출력
    if verbose:
        metadata = result["metadata"]
        trajectory = result["trajectory"]
        typer.echo()
        typer.echo("=== 실행 결과 요약 ===")
        typer.echo(f"  실행 시간: {metadata['execution_time_seconds']:.2f}초")
        typer.echo(f"  도구 호출 횟수: {metadata.get('tool_calls_count', 0)}")

        addie = result["addie_output"]
        typer.echo(f"  학습 목표 수: {len(addie.get('design', {}).get('learning_objectives', []))}")
        typer.echo(f"  모듈 수: {len(addie.get('development', {}).get('lesson_plan', {}).get('modules', []))}")
        typer.echo(f"  퀴즈 문항 수: {len(addie.get('evaluation', {}).get('quiz_items', []))}")

        typer.echo()
        typer.echo("도구 호출 순서:")
        for tc in trajectory.get("tool_calls", [])[:10]:
            typer.echo(f"  {tc['step']}. {tc['tool']}")
        if len(trajectory.get("tool_calls", [])) > 10:
            typer.echo(f"  ... 외 {len(trajectory['tool_calls']) - 10}개")

    typer.echo("완료")


@app.command()
def validate(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="검증할 시나리오 JSON 파일 경로",
        exists=True,
    ),
) -> None:
    """
    시나리오 JSON 파일의 유효성을 검증합니다.

    Example:
        react-isd validate --input scenario.json
    """
    try:
        scenario = load_scenario(input_file)

        # 필수 필드 확인
        required_fields = ["scenario_id", "title", "context", "learning_goals"]
        missing = [f for f in required_fields if f not in scenario]

        if missing:
            typer.echo(f"유효하지 않음: 필수 필드 누락 - {missing}", err=True)
            raise typer.Exit(code=1)

        typer.echo("유효한 시나리오입니다.")
        typer.echo(f"  ID: {scenario['scenario_id']}")
        typer.echo(f"  제목: {scenario['title']}")
        typer.echo(f"  대상: {scenario.get('context', {}).get('target_audience', '미지정')}")
        typer.echo(f"  목표 수: {len(scenario.get('learning_goals', []))}")

    except Exception as e:
        typer.echo(f"유효하지 않은 시나리오: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """
    ReAct-ISD 에이전트 정보를 출력합니다.
    """
    typer.echo("ReAct-ISD Agent")
    typer.echo("===============")
    typer.echo()
    typer.echo("LangGraph ReAct 패턴 기반 ADDIE 교수설계 에이전트")
    typer.echo()
    typer.echo("특징:")
    typer.echo("  - ReAct (Reasoning + Acting) 패턴")
    typer.echo("  - ADDIE 단계별 도구 분리")
    typer.echo("  - 에이전트 자율적 도구 선택")
    typer.echo("  - 상세한 궤적(trajectory) 기록")
    typer.echo()
    typer.echo("ADDIE 도구 (11개):")
    typer.echo("  Analysis: analyze_learners, analyze_context, analyze_task")
    typer.echo("  Design: design_objectives, design_assessment, design_strategy")
    typer.echo("  Development: create_lesson_plan, create_materials")
    typer.echo("  Implementation: create_implementation_plan")
    typer.echo("  Evaluation: create_quiz_items, create_rubric")
    typer.echo()
    typer.echo("버전: 0.1.0")


def main() -> None:
    """CLI 메인 엔트리포인트"""
    app()


if __name__ == "__main__":
    main()
