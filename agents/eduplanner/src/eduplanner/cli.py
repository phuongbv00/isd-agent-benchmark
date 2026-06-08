"""
EduPlanner CLI

교수설계 에이전트 CLI 인터페이스
공통 인터페이스 규격을 준수합니다.

Usage:
    eduplanner run --input scenario.json --output result.json
    eduplanner run --input scenario.json --output result.json --trajectory trace.json
"""

# Python 3.14 + LangChain Pydantic V1 호환성 경고 필터링
# LangChain Core가 내부적으로 pydantic.v1을 사용하여 Python 3.14에서 경고 발생
import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# .env 파일에서 환경변수 로드
try:
    from dotenv import load_dotenv
    # 프로젝트 루트의 .env 파일 찾기
    current_dir = Path(__file__).parent
    for _ in range(5):  # 최대 5단계 상위 디렉토리까지 탐색
        env_file = current_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
        current_dir = current_dir.parent
    else:
        load_dotenv()  # 기본 위치에서 로드 시도
except ImportError:
    pass  # python-dotenv가 없으면 무시

import typer

from eduplanner.agents import EduPlannerAgent
from eduplanner.models.schemas import ScenarioInput
from shared.llm import llm_config_from_env

app = typer.Typer(
    name="eduplanner",
    help="EduPlanner 교수설계 에이전트 CLI",
    add_completion=False,
)


def load_scenario(input_path: Path) -> ScenarioInput:
    """시나리오 JSON 파일 로드"""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ScenarioInput(**data)
    except FileNotFoundError:
        typer.echo(f"오류: 입력 파일을 찾을 수 없습니다: {input_path}", err=True)
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        typer.echo(f"오류: JSON 파싱 실패: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"오류: 시나리오 로드 실패: {e}", err=True)
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
        # cli.py 위치: agents/eduplanner/src/eduplanner/cli.py
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
                # Pydantic 모델이면 dict로 변환
                for slide in slide_contents:
                    if hasattr(slide, "model_dump"):
                        all_slides.append(slide.model_dump())
                    elif isinstance(slide, dict):
                        all_slides.append(slide)

        if not all_slides:
            if verbose:
                typer.echo("  슬라이드 콘텐츠가 없어 Marp 파일을 생성하지 않습니다.")
            return None

        # 출력 파일 경로 생성 (같은 디렉토리에 _slides.md 확장자)
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
    max_iterations: int = typer.Option(
        3,
        "--max-iterations",
        help="최대 반복 횟수 (기본값: 3, 타임아웃 방지)",
        min=1,
        max=20,
    ),
    target_score: float = typer.Option(
        90.0,
        "--target-score",
        help="목표 점수 (0-100)",
        min=0.0,
        max=100.0,
    ),
    model: str = typer.Option(
        "solar-mini",
        "--model",
        help="사용할 LLM 모델 (기본: solar-mini)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="상세 출력 모드",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="디버그 모드 (selective_merge 상세 로그)",
    ),
) -> None:
    """
    교수설계 시나리오를 입력받아 ADDIE 산출물을 생성합니다.

    Example:
        eduplanner run --input scenario.json --output result.json
        eduplanner run -i scenario.json -o result.json -t trajectory.json -v
    """
    if verbose:
        typer.echo(f"EduPlanner Agent 실행")
        typer.echo(f"  입력: {input_file}")
        typer.echo(f"  출력: {output_file}")
        typer.echo(f"  모델: {model}")
        typer.echo(f"  최대 반복: {max_iterations}")
        typer.echo(f"  목표 점수: {target_score}")
        typer.echo()

    # 시나리오 로드
    if verbose:
        typer.echo("시나리오 로드 중...")
    scenario = load_scenario(input_file)

    if verbose:
        typer.echo(f"  시나리오 ID: {scenario.scenario_id}")
        typer.echo(f"  제목: {scenario.title}")
        typer.echo()

    # 에이전트 초기화
    if verbose:
        typer.echo("에이전트 초기화 중...")

    from eduplanner.agents.base import AgentConfig

    llm_config = llm_config_from_env(model=model)

    if verbose:
        typer.echo(f"  Model Provider: {llm_config.provider}")
        typer.echo(f"  Model API Spec: {llm_config.api_spec}")
        typer.echo(f"  Model: {llm_config.model}")

    config = AgentConfig(model=llm_config.model, provider=llm_config.provider, llm_config=llm_config)

    agent = EduPlannerAgent(
        config=config,
        max_iterations=max_iterations,
        target_score=target_score,
        debug=debug,
    )

    # 실행
    if verbose:
        typer.echo("교수설계 생성 중...")
        typer.echo()

    try:
        result = agent.run(scenario)
    except Exception as e:
        typer.echo(f"오류: 에이전트 실행 실패: {e}", err=True)
        raise typer.Exit(code=1)

    # ADDIE 출력 저장 (표준 스키마로 변환)
    addie_output = result.addie_output.to_standard_dict()
    save_output(output_file, addie_output)

    if verbose:
        typer.echo(f"ADDIE 산출물 저장됨: {output_file}")

    # 슬라이드 Marp Markdown 파일 생성
    slide_path = export_slides_to_marp(
        addie_output=addie_output,
        output_path=output_file,
        title=scenario.title,
        verbose=verbose,
    )
    if slide_path and verbose:
        typer.echo(f"슬라이드 파일 저장됨: {slide_path}")

    # Trajectory 저장 (선택)
    if trajectory_file:
        trajectory_data = {
            "scenario_id": result.scenario_id,
            "agent_id": result.agent_id,
            "timestamp": str(result.timestamp),
            "trajectory": result.trajectory.model_dump(),
            "metadata": result.metadata.model_dump(),
        }
        save_output(trajectory_file, trajectory_data)

        if verbose:
            typer.echo(f"궤적 저장됨: {trajectory_file}")

    # 결과 요약 출력
    if verbose:
        typer.echo()
        typer.echo("=== 실행 결과 요약 ===")
        typer.echo(f"  실행 시간: {result.metadata.execution_time_seconds:.2f}초")
        typer.echo(f"  반복 횟수: {result.metadata.iterations}")
        typer.echo(f"  학습 목표 수: {len(result.addie_output.design.learning_objectives)}")
        typer.echo(f"  모듈 수: {len(result.addie_output.development.lesson_plan.modules)}")
        typer.echo(f"  퀴즈 문항 수: {len(result.addie_output.evaluation.quiz_items)}")

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
        eduplanner validate --input scenario.json
    """
    try:
        scenario = load_scenario(input_file)
        typer.echo(f"유효한 시나리오입니다.")
        typer.echo(f"  ID: {scenario.scenario_id}")
        typer.echo(f"  제목: {scenario.title}")
        typer.echo(f"  대상: {scenario.context.target_audience}")
        typer.echo(f"  목표 수: {len(scenario.learning_goals)}")
    except Exception as e:
        typer.echo(f"유효하지 않은 시나리오: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """
    EduPlanner 에이전트 정보를 출력합니다.
    """
    typer.echo("EduPlanner Agent")
    typer.echo("================")
    typer.echo()
    typer.echo("3-Agent 협업 기반 교수설계 생성 시스템")
    typer.echo()
    typer.echo("구성 에이전트:")
    typer.echo("  1. Generator - 초기 ADDIE 산출물 생성")
    typer.echo("  2. Evaluator - ADDIE Rubric 13항목 평가")
    typer.echo("  3. Optimizer - 피드백 기반 최적화")
    typer.echo("  4. Analyst - 오류 분석")
    typer.echo()
    typer.echo("평가 기준 (ADDIE Rubric 13항목):")
    typer.echo("  Analysis (분석) - 가중치 25%")
    typer.echo("    - A1: 학습자 분석의 적절성")
    typer.echo("    - A2: 수행 맥락 및 환경 분석의 타당성")
    typer.echo("    - A3: 요구분석 및 수행 격차 정의의 명확성")
    typer.echo("  Design (설계) - 가중치 25%")
    typer.echo("    - D1: 학습목표와 요구분석 간 정렬도")
    typer.echo("    - D2: 평가 설계의 타당성 및 정합성")
    typer.echo("    - D3: 교수전략 및 학습경험 설계의 이론적 적절성")
    typer.echo("  Development (개발) - 가중치 20%")
    typer.echo("    - Dev1: 프로토타입 개발")
    typer.echo("    - Dev2: 개발 결과 검토 및 수정")
    typer.echo("  Implementation (실행) - 가중치 15%")
    typer.echo("    - I1: 운영 계획의 현실성과 실행 가능성")
    typer.echo("    - I2: 교수자 지원 및 수업 가이드의 구체성")
    typer.echo("  Evaluation (평가) - 가중치 15%")
    typer.echo("    - E1: 형성평가")
    typer.echo("    - E2: 총괄평가 및 채택 결정")
    typer.echo("    - E3: 프로그램 개선 및 환류")
    typer.echo()
    typer.echo("버전: 0.2.0")


def main() -> None:
    """CLI 메인 엔트리포인트"""
    app()


if __name__ == "__main__":
    main()
