#!/usr/bin/env python3
"""
Dataset preparation script for Fred-LLM.

Simple runner script that orchestrates the data processing pipeline.
All logic is in the src.data modules.

Usage:
    python scripts/prepare_dataset.py --input data/raw/Fredholm_Dataset_Sample.csv --output data/processed/
    python scripts/prepare_dataset.py --max-samples 100 --augment --convert --validate
    python scripts/prepare_dataset.py --max-samples 200 --augment --split
"""

import argparse
import json
from pathlib import Path

from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare Fredholm dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/Fredholm_Dataset.csv"),
        help="Input dataset path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment the dataset",
    )
    parser.add_argument(
        "--augment-multiplier",
        type=int,
        default=2,
        help="Augmentation multiplier",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        default=True,
        help="Convert to all formats (infix, latex, rpn, tokenized, python). Use --no-convert to disable.",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Disable format conversion",
    )
    parser.add_argument(
        "--convert-formats",
        nargs="+",
        default=["infix", "latex", "rpn", "tokenized", "python"],
        choices=["infix", "latex", "rpn", "tokenized", "python"],
        help="Specific formats to convert (default: all)",
    )
    parser.add_argument(
        "--convert-limit",
        type=int,
        default=None,
        help="Limit number of equations to convert (None = all, default: all)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate equations",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val/test sets",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv", "both"],
        default="both",
        help="Output file format (json, csv, or both; default: both)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point - orchestrates the pipeline using src.data modules."""
    args = parse_args()

    # Import here to keep runner thin
    from src.data.augmentation import DataAugmenter
    from src.data.format_converter import FormatConverter
    from src.data.fredholm_loader import FredholmDatasetLoader
    from src.data.validator import validate_dataset

    console.print("[bold blue]Fred-LLM Dataset Preparation[/bold blue]\n")
    console.print(f"Input: {args.input}")
    console.print(f"Output: {args.output}\n")

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data using FredholmDatasetLoader
    console.print("[bold]Step 1: Loading data[/bold]")
    loader = FredholmDatasetLoader(args.input)
    if args.max_samples:
        loader.max_samples = args.max_samples
    equations = loader.load()
    
    # Convert to dictionaries with progress indicator
    from rich.progress import track
    console.print("  Converting to dictionaries...")
    data = [eq.to_dict() for eq in track(equations, description="  Processing")]
    console.print(f"  ✓ Loaded {len(data)} equations\n")

    # Save base data
    console.print("  Saving base data...")
    _save_data(data, args.output / "base_equations", "base", args.output_format)
    console.print(f"  ✓ Saved base data\n")

    # Step 2: Augment data if requested
    augmented_data = None
    if args.augment:
        console.print(f"[bold]Step 2: Augmenting data (multiplier={args.augment_multiplier})[/bold]")
        augmenter = DataAugmenter(strategies=["scale", "shift", "substitute"])
        augmented_data = augmenter.augment(data, multiplier=args.augment_multiplier)
        console.print(f"  ✓ Generated {len(augmented_data)} augmented equations\n")
        _save_data(augmented_data, args.output / "augmented_equations", "augmented", args.output_format)

    # Step 3: Convert formats (enabled by default)
    should_convert = args.convert and not args.no_convert
    
    if should_convert:
        console.print("[bold]Step 3: Converting formats[/bold]")
        converter = FormatConverter()
        convert_data = augmented_data if augmented_data else data
        
        # Use specified formats or all by default
        formats_to_convert = args.convert_formats
        console.print(f"  Formats: {', '.join([f.upper() for f in formats_to_convert])}")
        
        # Apply conversion limit if specified
        if args.convert_limit:
            sample_data = convert_data[:args.convert_limit]
            console.print(f"  Converting {len(sample_data)} equations (limited)")
        else:
            sample_data = convert_data
            console.print(f"  Converting all {len(sample_data)} equations")
        
        from rich.progress import track
        
        # Get base filename from input (without extension)
        base_filename = args.input.stem
        
        # Convert each format
        for fmt in formats_to_convert:
            console.print(f"\n  Converting to {fmt.upper()}...")
            formatted_equations = []
            
            for eq_dict in track(sample_data, description=f"  {fmt.upper()}"):
                try:
                    if fmt == "infix":
                        # Infix is already the source format
                        formatted_equations.append(_extract_equation_fields(eq_dict))
                    else:
                        # Convert to target format
                        formatted_eq = {
                            "u": converter.convert(eq_dict["u"], "infix", fmt),
                            "f": converter.convert(eq_dict["f"], "infix", fmt),
                            "kernel": converter.convert(eq_dict["kernel"], "infix", fmt),
                            "lambda": eq_dict.get("lambda", eq_dict.get("lambda_val", "1.0")),
                            "a": eq_dict["a"],
                            "b": eq_dict["b"],
                        }
                        formatted_equations.append(formatted_eq)
                except Exception as e:
                    console.print(f"    [yellow]Warning: {e}[/yellow]")
            
            # Save with input filename + format suffix: Fredholm_Dataset_Sample_latex.csv
            output_path = args.output / f"{base_filename}_{fmt}"
            
            _save_data(formatted_equations, output_path, fmt, args.output_format)
            console.print(f"  ✓ {fmt.upper()}: {len(formatted_equations)} equations")
        
        console.print()

    # Step 4: Validate if requested
    if args.validate:
        console.print("[bold]Step 4: Validating equations[/bold]")
        validate_data = augmented_data if augmented_data else data
        results = validate_dataset(validate_data[:100], strict=False)
        console.print(f"  ✓ Valid: {results['valid']}/{results['total']}")
        console.print(f"  ✗ Invalid: {results['invalid']}/{results['total']}\n")
        
        with open(args.output / "validation_report.json", "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"  ✓ Validation report saved\n")

    # Step 5: Split if requested
    if args.split:
        console.print("[bold]Step 5: Splitting dataset[/bold]")
        split_data = augmented_data if augmented_data else data
        train, val, test = _split_dataset(split_data, args.train_ratio, args.val_ratio)
        console.print(f"  ✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}\n")
        
        _save_data(train, args.output / "train", "train", args.output_format)
        _save_data(val, args.output / "val", "val", args.output_format)
        _save_data(test, args.output / "test", "test", args.output_format)

    # Summary
    console.print("[bold green]✓ Dataset preparation complete![/bold green]\n")
    console.print(f"Output directory: {args.output.absolute()}")
    # Show both JSON and CSV files
    for file in sorted(args.output.glob("*.*")):
        if file.suffix in [".json", ".csv"]:
            size_kb = file.stat().st_size / 1024
            console.print(f"  • {file.name} ({size_kb:.1f} KB)")


def _extract_equation_fields(eq_dict: dict) -> dict:
    """Extract core equation fields."""
    return {
        "u": eq_dict["u"],
        "f": eq_dict["f"],
        "kernel": eq_dict["kernel"],
        "lambda": eq_dict.get("lambda", eq_dict.get("lambda_val", "1.0")),
        "a": eq_dict["a"],
        "b": eq_dict["b"],
    }


def _save_data(data: list[dict], path: Path, label: str, format: str = "json") -> None:
    """Save data as JSON, CSV, or both, handling enum serialization."""
    import pandas as pd
    from rich.progress import track
    
    # Prepare serializable data
    serializable = []
    for item in track(data, description=f"    Serializing {len(data)} items"):
        item_copy = item.copy()
        # Convert enums to strings
        for key in ["u_type", "f_type", "kernel_type"]:
            if key in item_copy and hasattr(item_copy[key], "value"):
                item_copy[key] = item_copy[key].value
        serializable.append(item_copy)
    
    saved_files = []
    
    # Save as JSON
    if format in ["json", "both"]:
        console.print(f"    Writing JSON to {path.with_suffix('.json').name}...")
        json_path = path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        saved_files.append(json_path.name)
        console.print(f"    ✓ JSON saved")
    
    # Save as CSV
    if format in ["csv", "both"]:
        console.print(f"    Writing CSV to {path.with_suffix('.csv').name}...")
        csv_path = path.with_suffix(".csv")
        df = pd.DataFrame(serializable)
        df.to_csv(csv_path, index=False)
        saved_files.append(csv_path.name)
        console.print(f"    ✓ CSV saved")
    
    files_str = " & ".join(saved_files)
    console.print(f"  ✓ Saved {label} data: {len(data)} samples → {files_str}")


def _split_dataset(
    data: list[dict], train_ratio: float, val_ratio: float
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split data into train/val/test sets."""
    import random
    
    random.seed(42)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


if __name__ == "__main__":
    main()
