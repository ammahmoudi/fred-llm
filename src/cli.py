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
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables from .env file
load_dotenv()

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
        help="Show execution plan without running",
    ),
) -> None:
    """Run the adaptive LLM pipeline for Fredholm equations."""
    from src.adaptive_config import AdaptivePipelineConfig
    from src.adaptive_pipeline import AdaptivePipeline

    console.print(f"[bold blue]Fred-LLM[/bold blue] - Loading config from {config}")

    # Load adaptive config
    adaptive_config = AdaptivePipelineConfig.from_yaml(config)

    # Create and run adaptive pipeline
    pipeline = AdaptivePipeline(adaptive_config)
    results = pipeline.run(dry_run=dry_run)

    if not dry_run:
        console.print("[bold green]✓ Pipeline completed successfully![/bold green]")
        if results.get("metrics"):
            console.print(f"[cyan]Metrics: {results['metrics']}[/cyan]")


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


prompt_app = typer.Typer(help="Prompt generation commands")
app.add_typer(prompt_app, name="prompt")


@prompt_app.command("single")
def prompt_single(
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
    """Generate a prompt for a single equation."""
    from src.llm.prompt_templates import generate_prompt

    prompt_text = generate_prompt(equation, style=style)
    console.print(prompt_text)


@prompt_app.command("generate")
def prompt_generate(
    input_file: Path = typer.Argument(
        ...,
        help="Input CSV file with equations (train/test data)",
    ),
    output_dir: Path = typer.Option(
        Path("data/prompts"),
        "--output",
        "-o",
        help="Output directory for generated prompts",
    ),
    style: str = typer.Option(
        "chain-of-thought",
        "--style",
        "-s",
        help="Prompt style: basic, chain-of-thought, few-shot, tool-assisted",
    ),
    format_type: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Format type: infix, latex, rpn (auto-detected if not specified)",
    ),
    include_ground_truth: bool = typer.Option(
        True,
        "--ground-truth/--no-ground-truth",
        help="Include solutions in output",
    ),
    include_examples: bool = typer.Option(
        True,
        "--examples/--no-examples",
        help="Include few-shot examples (for few-shot style)",
    ),
    num_examples: int = typer.Option(
        2,
        "--num-examples",
        "-n",
        help="Number of few-shot examples to include",
    ),
    edge_case_mode: str = typer.Option(
        "none",
        "--edge-case-mode",
        "-e",
        help="Edge case mode: none, guardrails, hints",
    ),
) -> None:
    """Generate prompts from a dataset (CSV file)."""
    from src.prompts import create_processor

    console.print("\n" + "=" * 60)
    console.print(f"[bold blue]🎯 Generating {style} prompts[/bold blue]")
    console.print(f"[cyan]Input:[/cyan] {input_file}")
    console.print(f"[cyan]Edge case mode:[/cyan] {edge_case_mode}")
    console.print("=" * 60 + "\n")

    # Create processor
    processor = create_processor(
        style=style,
        output_dir=output_dir,
        include_ground_truth=include_ground_truth,
        include_examples=include_examples,
        num_examples=num_examples,
        edge_case_mode=edge_case_mode,
    )

    # Auto-detect format if not specified
    if format_type is None:
        file_str = str(input_file).lower()
        if "latex" in file_str:
            format_type = "latex"
        elif "rpn" in file_str:
            format_type = "rpn"
        else:
            format_type = "infix"
        console.print(f"[cyan]Auto-detected format: {format_type}[/cyan]")

    try:
        # Process dataset
        output_file = processor.process_dataset(
            input_csv=input_file,
            format_type=format_type,
            show_progress=True,
        )

        console.print("\n" + "=" * 60)
        console.print(f"[bold green]✓ Prompts generated successfully![/bold green]")
        console.print(f"[cyan]Output:[/cyan] {output_file}")
        console.print(f"[cyan]Format:[/cyan] {format_type}")
        console.print(f"[cyan]Style:[/cyan] {style}")
        console.print("=" * 60 + "\n")

    except Exception as e:
        console.print(f"[bold red]✗ Error generating prompts: {e}[/bold red]")
        raise typer.Exit(1) from e


@prompt_app.command("batch")
def prompt_batch(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing CSV files (e.g., data/processed/training_data_v2/)",
    ),
    output_dir: Path = typer.Option(
        Path("data/prompts"),
        "--output",
        "-o",
        help="Output directory for generated prompts",
    ),
    styles: Optional[str] = typer.Option(
        None,
        "--styles",
        "-s",
        help="Comma-separated styles (e.g., 'basic,chain-of-thought') or 'all'",
    ),
    pattern: str = typer.Option(
        "*.csv",
        "--pattern",
        "-p",
        help="File pattern to match (e.g., 'train_*.csv')",
    ),
    include_ground_truth: bool = typer.Option(
        True,
        "--ground-truth/--no-ground-truth",
        help="Include solutions in output",
    ),
) -> None:
    """Generate prompts for multiple datasets in batch."""
    from src.prompts import create_processor

    # Parse styles
    if styles is None or styles == "all":
        style_list = ["basic", "chain-of-thought", "few-shot", "tool-assisted"]
    else:
        style_list = [s.strip() for s in styles.split(",")]

    # Find CSV files
    input_files = list(Path(input_dir).glob(pattern))
    if not input_files:
        console.print(f"[red]No CSV files found matching pattern: {pattern}[/red]")
        raise typer.Exit(1)

    console.print("\n" + "=" * 60)
    console.print(f"[bold blue]📦 Batch Prompt Generation[/bold blue]")
    console.print(f"[cyan]Files:[/cyan] {len(input_files)}")
    console.print(f"[cyan]Styles:[/cyan] {', '.join(style_list)}")
    console.print(f"[cyan]Output:[/cyan] {output_dir}")
    console.print("=" * 60 + "\n")

    total_generated = 0

    # Process each style
    for style in style_list:
        console.print(f"\n[bold yellow]Processing style: {style}[/bold yellow]")

        processor = create_processor(
            style=style,
            output_dir=output_dir / style,
            include_ground_truth=include_ground_truth,
        )

        # Process all files
        try:
            output_files = processor.process_multiple_datasets(
                input_files=input_files,
                show_progress=True,
            )
            total_generated += len(output_files)
            console.print(
                f"[green]✓ Generated {len(output_files)} prompt files for {style}[/green]"
            )

        except Exception as e:
            console.print(f"[red]✗ Error processing {style}: {e}[/red]")
            continue

    console.print("\n" + "=" * 60)
    console.print(f"[bold green]✓ Batch generation complete![/bold green]")
    console.print(f"[cyan]Generated:[/cyan] {total_generated} prompt files")
    console.print(f"[cyan]Location:[/cyan] {output_dir}")
    console.print("=" * 60 + "\n")


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
