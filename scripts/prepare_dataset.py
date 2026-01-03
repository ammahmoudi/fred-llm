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
        type=float,
        default=2,
        help="Augmentation multiplier (can be fractional, e.g., 1.5 for 50%% more data)",
    )
    parser.add_argument(
        "--augment-strategies",
        nargs="+",
        default=["substitute", "scale", "shift"],
        choices=[
            # Basic transformations (exact_symbolic)
            "substitute",
            "scale",
            "shift",
            "compose",
            # Solution-type folders (each runs all strategies in folder)
            "none_solution",  # 4 strategies × 3 variants = 12 edge cases
            "approx_coef",  # 5 strategies × 3 variants = 15 edge cases
            "discrete_points",  # 2 strategies × 3 variants = 6 edge cases
            "series",  # 1 strategy × 3 variants = 3 edge cases
            "family",  # 1 strategy × 3 variants = 3 edge cases
            "regularized",  # 1 strategy × 3 variants = 3 edge cases
        ],
        help=(
            "Solution-type-based augmentation strategies:\n"
            "none_solution: eigenvalue_cases + range_violation + divergent_kernel + disconnected_support (12 variants)\n"
            "approx_coef: boundary_layer + oscillatory + weakly_singular + mixed_type + compact_support (15 variants)\n"
            "discrete_points: complex_kernels + near_resonance (6 variants)\n"
            "series: neumann_series (3 variants)\n"
            "family: resonance (3 variants)\n"
            "regularized: ill_posed (3 variants)"
        ),
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
        "--include-edge-metadata",
        action="store_true",
        help="Include detailed edge case metadata fields in output (singularity_type, layer_location, etc.)",
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
        default=0.0,
        help="Validation set ratio (default: 0.0, test will be 0.2)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv", "both"],
        default="both",
        help="Output file format (json, csv, or both; default: both)",
    )
    return parser.parse_args()


def _filter_edge_metadata(data: list[dict]) -> list[dict]:
    """
    Filter out detailed edge case metadata fields from augmented equations.

    Keeps essential fields but removes verbose metadata like singularity details,
    oscillation parameters, numerical method specifics, etc.

    Core fields kept:
    - Basic equation: u, f, kernel, lambda_val, a, b
    - Augmentation: augmented, augmentation_type, augmentation_variant
    - Solution: has_solution, solution_type, edge_case, reason
    - Methods: recommended_methods, numerical_challenge

    Detailed metadata removed (60+ fields):
    - Singularity: singularity_type, singularity_order, singularity_strength, etc.
    - Boundary layers: layer_location, layer_width_estimate, gradient_scale, etc.
    - Oscillations: oscillation_type, oscillation_frequency, angular_frequency, etc.
    - And many more technical details...
    """
    # Define detailed metadata fields to exclude by default
    DETAILED_METADATA_FIELDS = {
        # Singularity details
        "singularity_type",
        "singularity_order",
        "singularity_strength",
        # Boundary layer details
        "layer_location",
        "layer_width_estimate",
        "gradient_scale",
        "minimum_points_in_layer",
        # Oscillation details
        "oscillation_type",
        "oscillation_frequency",
        "angular_frequency",
        "num_cycles_in_domain",
        "nyquist_samples_required",
        "sampling_rate_needed",
        "modulation",
        "frequencies",
        "beat_frequency",
        "spectrum_complexity",
        # Kernel details
        "kernel_description",
        "equation_type",
        "split_point",
        "causal_structure",
        "volterra_region",
        "fredholm_region",
        "mathematical_form",
        "transition_width",
        "mathematical_explanation",
        "kernel_split",
        "integral_form",
        "solution_strategy",
        "kernel_structure",
        # Support details
        "support_type",
        "support_width",
        "zero_fraction",
        "matrix_structure",
        "bandwidth_parameter",
        "rank_deficient_risk",
        "memory_efficiency",
        "support_region",
        "active_fraction",
        "physical_interpretation",
        "num_support_regions",
        # Numerical method details
        "numerical_method",
        "sample_points",
        "sample_values",
        # Resonance details
        "is_critical",
        "near_critical_value",
        "distance_to_resonance",
        "condition_number_estimate",
        # Series details
        "series_type",
        "series_terms",
        "convergence_estimate",
        "truncation_error",
        # Family details
        "eigenvalue_approximate",
        "eigenfunction",
        "solution_multiplicity",
        "general_solution",
        # Ill-posed details
        "equation_form",
        "is_ill_posed",
        "requires_regularization",
        "regularization_param",
        # Additional metadata
        "notes",
        "operator_property",
        "orthogonality_violated",
        "kernel_rank",
        "range_basis",
        "divergence_type",
        "contrast_with",
        "physically_invalid",
        "problem_structure",
        "mathematical_issue",
        "warning",
    }

    filtered_data = []
    for eq in data:
        # Create copy with only essential fields
        filtered_eq = {k: v for k, v in eq.items() if k not in DETAILED_METADATA_FIELDS}
        filtered_data.append(filtered_eq)

    return filtered_data


def main() -> None:
    """Main entry point - orchestrates the pipeline using src.data modules."""
    args = parse_args()

    # Import here to keep runner thin
    from src.data.augmentation import DataAugmenter
    from src.data.format_converter import FormatConverter
    from src.data.fredholm_loader import FredholmDatasetLoader
    from src.data.splitter import get_split_statistics, split_dataset
    from src.data.validator import validate_dataset

    console.print("[bold blue]Fred-LLM Dataset Preparation[/bold blue]\n")
    console.print(f"Input: {args.input}")
    console.print(f"Output: {args.output}\n")

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    # Track created files for summary
    created_files = []

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

    # Add source file metadata to all items
    source_filename = args.input.name
    for item in data:
        item["source_file"] = source_filename
        item["source_path"] = str(args.input)

    console.print(f"  ✓ Loaded {len(data)} equations\n")

    # Save base data (in root of output directory)
    console.print("  Saving base data...")
    base_filename = args.input.stem  # e.g., "Fredholm_Dataset_Sample"
    created_files.extend(
        _save_data(
            data, args.output / f"{base_filename}_base", "base", args.output_format
        )
    )
    console.print(f"  ✓ Saved base data\n")

    # Step 2: Augment data if requested
    augmented_data = None
    if args.augment:
        console.print(
            f"[bold]Step 2: Augmenting data (multiplier={args.augment_multiplier})[/bold]"
        )
        console.print(f"  Strategies: {', '.join(args.augment_strategies)}")
        augmenter = DataAugmenter(strategies=args.augment_strategies)
        augmented_data = augmenter.augment(data, multiplier=args.augment_multiplier)

        # Filter out detailed edge case metadata unless explicitly requested
        if not args.include_edge_metadata:
            augmented_data = _filter_edge_metadata(augmented_data)

        console.print(f"  ✓ Generated {len(augmented_data)} augmented equations\n")
        # Create augmented subdirectory
        augmented_dir = args.output / "augmented"
        augmented_dir.mkdir(parents=True, exist_ok=True)
        created_files.extend(
            _save_data(
                augmented_data,
                augmented_dir / f"{base_filename}_augmented",
                "augmented",
                args.output_format,
            )
        )

    # Step 3: Validate if requested (before splitting)
    if args.validate:
        console.print("[bold]Step 3: Validating equations[/bold]")
        validate_data = augmented_data if augmented_data else data
        results = validate_dataset(validate_data[:100], strict=False)
        console.print(f"  ✓ Valid: {results['valid']}/{results['total']}")
        console.print(f"  ✗ Invalid: {results['invalid']}/{results['total']}\n")

        with open(args.output / "validation_report.json", "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"  ✓ Validation report saved\n")

    # Step 4: Split dataset first (before formatting)
    train_data = None
    val_data = None
    test_data = None

    if args.split:
        console.print("[bold]Step 4: Splitting dataset[/bold]")
        split_data = augmented_data if augmented_data else data
        train_data, val_data, test_data = split_dataset(
            split_data, args.train_ratio, args.val_ratio, seed=42
        )
        console.print(
            f"  ✓ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}\n"
        )

        # Get and display split statistics
        stats = get_split_statistics(train_data, val_data, test_data)
        console.print("  📊 Split balance:")
        for split_name in ["train", "val", "test"]:
            split_stats = stats[split_name]
            console.print(
                f"    {split_name.capitalize()}: "
                f"{split_stats['original_count']} original "
                f"({split_stats['original_ratio']:.1%}), "
                f"{split_stats['augmented_count']} augmented"
            )
        console.print()

    # Step 5: Convert formats (after splitting to prevent data leakage)
    should_convert = args.convert and not args.no_convert

    if should_convert:
        console.print("[bold]Step 5: Converting formats[/bold]")
        converter = FormatConverter()

        # Use specified formats or all by default
        formats_to_convert = args.convert_formats
        console.print(
            f"  Formats: {', '.join([f.upper() for f in formats_to_convert])}"
        )
        console.print()

        from rich.progress import track

        # Determine what data to convert
        if args.split:
            # Convert each split separately
            splits_to_convert = [
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
            ]
        else:
            # Convert full dataset (backward compatibility)
            convert_data = augmented_data if augmented_data else data

            # Apply conversion limit if specified
            if args.convert_limit:
                convert_data = convert_data[: args.convert_limit]
                console.print(f"  Converting {len(convert_data)} equations (limited)")

            splits_to_convert = [("full", convert_data)]

        # Convert each split to each format
        for split_name, split_data in splits_to_convert:
            if not split_data:  # Skip empty splits (e.g., val_ratio=0)
                continue

            console.print(
                f"  Converting {split_name} split ({len(split_data)} equations):"
            )

            for fmt in formats_to_convert:
                formatted_equations = []

                for eq_dict in track(split_data, description=f"    {fmt.upper()}"):
                    try:
                        if fmt == "infix":
                            # Infix is already the source format - keep all columns
                            formatted_equations.append(eq_dict.copy())
                        else:
                            # Convert to target format - preserve all original columns
                            formatted_eq = eq_dict.copy()
                            # Only update the expression fields
                            formatted_eq["u"] = converter.convert(
                                eq_dict["u"], "infix", fmt
                            )
                            formatted_eq["f"] = converter.convert(
                                eq_dict["f"], "infix", fmt
                            )
                            formatted_eq["kernel"] = converter.convert(
                                eq_dict["kernel"], "infix", fmt
                            )
                            formatted_equations.append(formatted_eq)
                    except Exception as e:
                        console.print(f"      [yellow]Warning: {e}[/yellow]")

                # Save formatted split
                if args.split:
                    output_path = args.output / f"{split_name}_{fmt}"
                else:
                    # Backward compatibility: save in formatted/ subdirectory
                    formatted_dir = args.output / "formatted"
                    formatted_dir.mkdir(parents=True, exist_ok=True)
                    base_filename = args.input.stem
                    output_path = formatted_dir / f"{base_filename}_{fmt}"

                created_files.extend(
                    _save_data(
                        formatted_equations,
                        output_path,
                        f"{split_name}_{fmt}",
                        args.output_format,
                    )
                )
                console.print(
                    f"    ✓ {fmt.upper()}: {len(formatted_equations)} equations"
                )

            console.print()
    elif args.split:
        # Save unformatted splits
        console.print("[bold]Saving unformatted splits[/bold]")
        _save_data(train_data, args.output / "train", "train", args.output_format)
        _save_data(val_data, args.output / "val", "val", args.output_format)
        _save_data(test_data, args.output / "test", "test", args.output_format)
        console.print()

    # Summary
    console.print("[bold green]✓ Dataset preparation complete![/bold green]\n")
    console.print(f"Output directory: {args.output.absolute()}")
    console.print("\n[bold]Files created in this run:[/bold]")
    for file in created_files:
        size_kb = file.stat().st_size / 1024
        # Show relative path from output directory
        rel_path = file.relative_to(args.output)
        console.print(f"  • {rel_path} ({size_kb:.1f} KB)")


def _extract_equation_fields(eq_dict: dict) -> dict:
    """Extract core equation fields."""
    return {
        "u": eq_dict["u"],
        "f": eq_dict["f"],
        "kernel": eq_dict["kernel"],
        "lambda_val": eq_dict.get("lambda_val", "1.0"),
        "a": eq_dict["a"],
        "b": eq_dict["b"],
    }


def _save_data(
    data: list[dict], path: Path, label: str, format: str = "json"
) -> list[Path]:
    """Save data as JSON, CSV, or both, handling enum and SymPy serialization. Returns list of created file paths."""
    import pandas as pd
    import sympy as sp
    from rich.progress import track

    # Prepare serializable data
    serializable = []
    for item in track(data, description=f"    Serializing {len(data)} items"):
        item_copy = item.copy()

        # Convert all values to JSON-serializable types
        for key, value in list(item_copy.items()):
            # Convert enums to strings
            if hasattr(value, "value"):
                item_copy[key] = value.value
            # Convert SymPy expressions to strings
            elif isinstance(value, (sp.Basic, sp.Expr)):
                item_copy[key] = str(value)
            # Convert lists containing SymPy objects
            elif isinstance(value, list):
                item_copy[key] = [
                    str(v) if isinstance(v, (sp.Basic, sp.Expr)) else v for v in value
                ]
            # Convert NaN to None for proper JSON null representation
            elif pd.isna(value):
                item_copy[key] = None

        serializable.append(item_copy)

    saved_files = []

    # Save as JSON
    if format in ["json", "both"]:
        console.print(f"    Writing JSON to {path.with_suffix('.json').name}...")
        json_path = path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        saved_files.append(json_path)
        console.print(f"    ✓ JSON saved")

    # Save as CSV
    if format in ["csv", "both"]:
        console.print(f"    Writing CSV to {path.with_suffix('.csv').name}...")
        csv_path = path.with_suffix(".csv")
        df = pd.DataFrame(serializable)
        # Preserve empty strings instead of converting to NaN
        df.to_csv(csv_path, index=False, na_rep="")
        saved_files.append(csv_path)
        console.print(f"    ✓ CSV saved")

    files_str = " & ".join([f.name for f in saved_files])
    console.print(f"  ✓ Saved {label} data: {len(data)} samples → {files_str}")
    return saved_files

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


if __name__ == "__main__":
    main()
