#!/usr/bin/env python3
"""
Prompt generation runner script for Fred-LLM.

Orchestrates batch prompt generation using the src.prompts module.
Provides a high-level interface for common prompt generation workflows.

Usage:
    # Generate prompts for all training data with multiple styles
    python scripts/run_prompt_generation.py --input data/processed/training_data_v2/ --styles all
    
    # Generate specific style for test data
    python scripts/run_prompt_generation.py --input data/processed/training_data_v2/test_infix.csv --style chain-of-thought
    
    # Generate prompts without ground truth (for inference)
    python scripts/run_prompt_generation.py --input data/processed/test.csv --no-ground-truth --style few-shot
    
    # Custom output directory
    python scripts/run_prompt_generation.py --input data/processed/ --output data/my_prompts/ --styles basic,few-shot
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate prompts for Fredholm equation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file or directory containing CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompts"),
        help="Output directory for generated prompts (default: data/prompts)",
    )
    parser.add_argument(
        "--styles",
        type=str,
        default="chain-of-thought",
        help="Comma-separated prompt styles: basic, chain-of-thought, few-shot, tool-assisted, or 'all'",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="File pattern to match (default: *.csv)",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Exclude ground truth solutions (for inference mode)",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Exclude few-shot examples",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=2,
        help="Number of few-shot examples (default: 2)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["infix", "latex", "rpn"],
        default=None,
        help="Force specific format (default: auto-detect from filename)",
    )

    return parser.parse_args()


def get_csv_files(input_path: Path, pattern: str) -> list[Path]:
    """Get list of CSV files from input path."""
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        return sorted(input_path.glob(pattern))
    else:
        console.print(f"[red]✗ Input path not found: {input_path}[/red]")
        return []


def expand_styles(styles_str: str) -> list[str]:
    """Expand style string to list of styles."""
    if styles_str.lower() == "all":
        return ["basic", "chain-of-thought", "few-shot", "tool-assisted"]
    else:
        return [s.strip() for s in styles_str.split(",")]


def display_config(args: argparse.Namespace, csv_files: list[Path], styles: list[str]) -> None:
    """Display configuration summary."""
    console.print()
    console.print(Panel.fit("🎯 Prompt Generation Configuration", style="bold blue"))
    
    table = Table(show_header=False, box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Input", str(args.input))
    table.add_row("Output", str(args.output))
    table.add_row("Files", f"{len(csv_files)} CSV files")
    table.add_row("Styles", ", ".join(styles))
    table.add_row("Ground Truth", "❌ No" if args.no_ground_truth else "✅ Yes")
    table.add_row("Examples", "❌ No" if args.no_examples else f"✅ Yes ({args.num_examples})")
    
    console.print(table)
    console.print()


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # Import here to show args parsing is fast
    from src.prompts import create_processor

    # Get CSV files
    csv_files = get_csv_files(args.input, args.pattern)
    if not csv_files:
        console.print("[yellow]⚠ No CSV files found[/yellow]")
        return

    # Expand styles
    styles = expand_styles(args.styles)

    # Display configuration
    display_config(args, csv_files, styles)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Generate prompts for each style
    total_files = len(csv_files) * len(styles)
    generated_files = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        for style in styles:
            task = progress.add_task(f"[cyan]Generating {style} prompts...", total=len(csv_files))
            
            # Create processor for this style
            processor = create_processor(
                style=style,
                output_dir=args.output / style,
                include_ground_truth=not args.no_ground_truth,
                include_examples=not args.no_examples,
                num_examples=args.num_examples,
            )

            for csv_file in csv_files:
                progress.update(task, description=f"[cyan]{style}: {csv_file.name}")
                
                # Determine format
                format_type = args.format
                if format_type is None:
                    # Auto-detect from filename
                    filename_lower = csv_file.name.lower()
                    if "latex" in filename_lower:
                        format_type = "latex"
                    elif "rpn" in filename_lower:
                        format_type = "rpn"
                    else:
                        format_type = "infix"

                try:
                    output_file = processor.process_dataset(
                        input_csv=csv_file,
                        format_type=format_type,
                        show_progress=False,
                    )
                    generated_files.append(output_file)
                except Exception as e:
                    console.print(f"[red]✗ Error processing {csv_file.name}: {e}[/red]")
                
                progress.update(task, advance=1)

    # Display results
    console.print()
    console.print(Panel.fit("✅ Prompt Generation Complete", style="bold green"))
    
    results_table = Table(title="Generated Files", show_lines=True)
    results_table.add_column("Style", style="cyan")
    results_table.add_column("File", style="white")
    results_table.add_column("Size", justify="right", style="yellow")
    
    for output_file in generated_files:
        style_name = output_file.parent.name
        file_size = output_file.stat().st_size / 1024  # KB
        results_table.add_row(
            style_name,
            output_file.name,
            f"{file_size:.1f} KB"
        )
    
    console.print(results_table)
    console.print()
    console.print(f"[green]✓ Generated {len(generated_files)} prompt files[/green]")
    console.print(f"[cyan]📁 Output directory: {args.output}[/cyan]")


if __name__ == "__main__":
    main()
