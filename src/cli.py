"""
CLI entrypoint for Fred-LLM.

Usage:
    python -m src.cli run --config config.yaml
    python -m src.cli evaluate --input data/processed/results.json
    python -m src.cli convert --format rpn --input data/raw/equations.json
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="fred-llm",
    help="Solve Fredholm integral equations using LLMs",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    config: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without making API calls",
    ),
) -> None:
    """Run the LLM-based Fredholm equation solver pipeline."""
    from src.config import load_config
    from src.main import FredLLMPipeline

    console.print(f"[bold blue]Fred-LLM[/bold blue] - Loading config from {config}")

    cfg = load_config(config)
    pipeline = FredLLMPipeline(cfg)

    if dry_run:
        console.print("[yellow]Dry run mode - no API calls will be made[/yellow]")

    # TODO: Implement full pipeline execution
    pipeline.run(dry_run=dry_run)

    console.print("[bold green]Pipeline completed successfully![/bold green]")


@app.command()
def evaluate(
    input_file: Path = typer.Argument(
        ...,
        help="Path to results file for evaluation",
    ),
    mode: str = typer.Option(
        "both",
        "--mode",
        "-m",
        help="Evaluation mode: symbolic, numeric, or both",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for evaluation results",
    ),
) -> None:
    """Evaluate LLM-generated solutions against ground truth."""
    from src.llm.evaluate import evaluate_solutions

    console.print(f"[bold blue]Evaluating solutions from {input_file}[/bold blue]")

    # TODO: Implement evaluation logic
    results = evaluate_solutions(input_file, mode=mode)

    if output:
        console.print(f"[green]Results saved to {output}[/green]")
    else:
        console.print(results)


@app.command()
def convert(
    input_file: Path = typer.Argument(
        ...,
        help="Input file to convert",
    ),
    format: str = typer.Option(
        "rpn",
        "--format",
        "-f",
        help="Output format: rpn, latex, tokenized, sympy",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path",
    ),
) -> None:
    """Convert equations between different formats."""
    from src.data.format_converter import convert_format

    console.print(f"[bold blue]Converting {input_file} to {format} format[/bold blue]")

    # TODO: Implement format conversion
    result = convert_format(input_file, target_format=format)

    if output:
        console.print(f"[green]Converted file saved to {output}[/green]")
    else:
        console.print(result)


@app.command()
def prompt(
    equation: str = typer.Argument(
        ...,
        help="Equation string or file path",
    ),
    style: str = typer.Option(
        "chain-of-thought",
        "--style",
        "-s",
        help="Prompt style: basic, chain-of-thought, few-shot, tool-assisted",
    ),
) -> None:
    """Generate a prompt for a given equation."""
    from src.llm.prompt_templates import generate_prompt

    # TODO: Implement prompt generation
    prompt_text = generate_prompt(equation, style=style)
    console.print(prompt_text)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
