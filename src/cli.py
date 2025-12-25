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
        help="Output format: rpn, latex, infix, tokenized, python",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (.json or .csv)",
    ),
    output_type: str = typer.Option(
        "json",
        "--output-type",
        "-t",
        help="Output file type: json or csv",
    ),
) -> None:
    """Convert equations between different formats and save as JSON or CSV."""
    import json

    from src.data.format_converter import FormatConverter, convert_format

    console.print(f"[bold blue]Converting {input_file} to {format} format[/bold blue]")

    # Load and convert equations
    result = convert_format(input_file, target_format=format)

    if output:
        output = Path(output)
        converter = FormatConverter()

        # Determine output type from extension if not explicitly set
        if output.suffix == ".csv":
            output_type = "csv"
        elif output.suffix == ".json":
            output_type = "json"

        if output_type == "csv":
            # Export to CSV
            converter.convert_to_csv(result, output, format=format)
            console.print(
                f"[green]✓ Converted {len(result)} equations saved to CSV: {output}[/green]"
            )
        else:
            # Export to JSON
            with open(output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            console.print(
                f"[green]✓ Converted {len(result)} equations saved to JSON: {output}[/green]"
            )
    else:
        # Print first few results
        console.print(f"\n[bold]Converted {len(result)} equations[/bold]")
        console.print("\n[bold cyan]Sample (first 3):[/bold cyan]")
        for i, eq in enumerate(result[:3], 1):
            console.print(
                f"\n{i}. u_{format}: {eq.get(f'u_{format}', eq.get('u', 'N/A'))}"
            )
            console.print(f"   f_{format}: {eq.get(f'f_{format}', eq.get('f', 'N/A'))}")
        console.print(f"\n[yellow]Use --output to save all results[/yellow]")


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


@app.command()
def dataset(
    action: str = typer.Argument(
        "info",
        help="Action: download, info, stats, sample",
    ),
    variant: str = typer.Option(
        "sample",
        "--variant",
        "-v",
        help="Dataset variant: 'sample' (5K) or 'full' (500K)",
    ),
    output_dir: Path = typer.Option(
        "data/raw",
        "--output",
        "-o",
        help="Output directory for downloaded data",
    ),
    max_samples: int | None = typer.Option(
        None,
        "--max-samples",
        "-n",
        help="Maximum samples to display/load",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download",
    ),
) -> None:
    """Download and explore the Fredholm-LLM dataset from Zenodo."""
    from rich.table import Table

    from src.data.fredholm_loader import FredholmDatasetLoader

    # Convert string to Path if needed
    output_path = Path(output_dir)

    if action == "download":
        from src.data.dataset_fetcher import FredholmDatasetFetcher

        console.print(
            f"[bold blue]Downloading Fredholm-LLM dataset ({variant})...[/bold blue]"
        )
        fetcher = FredholmDatasetFetcher(data_dir=output_path)

        try:
            path = fetcher.download_dataset(variant=variant, force=force)
            console.print(f"[bold green]✓ Dataset downloaded to: {path}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]✗ Download failed: {e}[/bold red]")
            raise typer.Exit(1) from e

    elif action == "info":
        from src.data.dataset_fetcher import FredholmDatasetFetcher

        console.print("[bold blue]Fredholm-LLM Dataset Information[/bold blue]\n")
        console.print("Source: https://github.com/alirezaafzalaghaei/Fredholm-LLM")
        console.print("DOI: 10.5281/zenodo.16784707\n")

        console.print("[bold]Available variants:[/bold]")
        console.print("  • sample: ~5,000 equations (recommended for testing)")
        console.print("  • full: ~500,000 equations (complete dataset)\n")

        console.print("[bold]Dataset schema:[/bold]")
        schema_table = Table(show_header=True, header_style="bold cyan")
        schema_table.add_column("Field", style="green")
        schema_table.add_column("Description")
        schema_table.add_row("u", "Solution function u(x)")
        schema_table.add_row("f", "Right-hand side function f(x)")
        schema_table.add_row("kernel", "Kernel function K(x, t)")
        schema_table.add_row("lambda", "Parameter λ")
        schema_table.add_row("a, b", "Integration bounds")
        schema_table.add_row("*_is_polynomial", "Boolean: is polynomial?")
        schema_table.add_row("*_is_trigonometric", "Boolean: has sin/cos/tan?")
        schema_table.add_row("*_is_hyperbolic", "Boolean: has sinh/cosh/tanh?")
        schema_table.add_row("*_is_exponential", "Boolean: has exp?")
        schema_table.add_row("*_max_degree", "Maximum polynomial degree")
        console.print(schema_table)

        # Check for available files
        console.print("\n[bold]Checking available files on Zenodo...[/bold]")
        try:
            fetcher = FredholmDatasetFetcher()
            files = fetcher.list_available_files()
            for f in files:
                size_mb = f.get("size", 0) / (1024 * 1024)
                console.print(f"  • {f['key']}: {size_mb:.2f} MB")
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Could not fetch file list: {e}[/yellow]")

    elif action == "stats":
        console.print("[bold blue]Loading dataset statistics...[/bold blue]")

        loader = FredholmDatasetLoader(
            data_path=output_path
            / f"Fredholm_Dataset{'_Sample' if variant == 'sample' else ''}.csv",
            auto_download=True,
            variant=variant,
            max_samples=max_samples,
        )

        try:
            stats = loader.get_statistics()

            console.print(f"\n[bold]Total equations:[/bold] {stats['total_equations']}")

            console.print("\n[bold]Solution (u) expression types:[/bold]")
            for expr_type, count in stats["u_types"].items():
                pct = (
                    (count / stats["total_equations"]) * 100
                    if stats["total_equations"] > 0
                    else 0
                )
                console.print(f"  • {expr_type}: {count} ({pct:.1f}%)")

            console.print("\n[bold]Kernel expression types:[/bold]")
            for expr_type, count in stats["kernel_types"].items():
                pct = (
                    (count / stats["total_equations"]) * 100
                    if stats["total_equations"] > 0
                    else 0
                )
                console.print(f"  • {expr_type}: {count} ({pct:.1f}%)")

        except FileNotFoundError:
            console.print(
                "[yellow]Dataset not found. Run 'dataset download' first.[/yellow]"
            )

    elif action == "sample":
        console.print("[bold blue]Sample equations from dataset:[/bold blue]\n")

        loader = FredholmDatasetLoader(
            data_path=output_dir
            / f"Fredholm_Dataset{'_Sample' if variant == 'sample' else ''}.csv",
            auto_download=True,
            variant=variant,
            max_samples=max_samples or 5,
        )

        try:
            equations = loader.load()

            for i, eq in enumerate(equations[:5]):
                console.print(f"[bold cyan]Equation {i + 1}:[/bold cyan]")
                console.print(f"  u(x) = {eq.u}")
                console.print(f"  f(x) = {eq.f}")
                console.print(f"  K(x,t) = {eq.kernel}")
                console.print(f"  lambda = {eq.lambda_val}")
                console.print(f"  bounds = \\[{eq.a}, {eq.b}]")
                if eq.metadata.get("u_type"):
                    console.print(f"  Type: {eq.metadata['u_type'].value}")
                console.print()

        except FileNotFoundError:
            console.print(
                "[yellow]Dataset not found. Run 'dataset download' first.[/yellow]"
            )

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: download, info, stats, sample")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
