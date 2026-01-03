#!/usr/bin/env python3
"""
Data validation script for augmented Fredholm dataset.

Checks that all augmentation strategies produce expected data patterns:
- Strategy names match expected values
- has_solution and solution_type are consistent
- u field patterns match expectations
- No folder names leak into augmentation_type
"""

import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def validate_dataset(csv_path: Path) -> dict:
    """
    Validate augmented dataset structure and consistency.

    Returns:
        dict with validation results and errors
    """
    console.print(f"\n[bold]Validating dataset:[/bold] {csv_path}")

    # Load data
    df = pd.read_csv(csv_path, keep_default_na=False)
    console.print(f"✓ Loaded {len(df):,} equations\n")

    errors = []
    warnings = []

    # Expected strategy configurations
    # Format: strategy_name: (expected_solution_types, expected_has_solution, allow_empty_u)
    strategy_configs = {
        # no_solution folder
        "eigenvalue_cases": (["none"], [False], True),
        "range_violation": (["none"], [False], True),
        "divergent_kernel": (["none"], [False], True),
        "disconnected_support": (["none"], [False], True),
        # numerical_only folder
        "complex_kernels": (["numerical"], [True], True),  # No closed-form
        "boundary_layer": (["numerical"], [True], False),  # Has exact formula
        "oscillatory_solution": (["numerical"], [True], False),
        "weakly_singular": (["numerical"], [True], False),
        "mixed_type": (["numerical"], [True], False),
        "compact_support": (
            ["numerical"],
            [True],
            False,
        ),  # Now only numerical variants
        "near_resonance": (
            ["numerical"],
            [True],
            True,
        ),  # Near-eigenvalue ill-conditioned
        # regularization_required folder
        "ill_posed": (
            ["regularized"],
            [True],
            True,
        ),  # Has solution, needs regularization
        # non_unique_solution folder
        "resonance": (
            ["family"],
            [True],
            True,
        ),  # Only exact resonance with family solutions
    }

    # 1. Check for folder names in augmentation_type
    console.print(
        "[bold blue]1. Checking for folder names in augmentation_type...[/bold blue]"
    )
    folder_names = {
        "no_solution",
        "numerical_only",
        "regularization_required",
        "non_unique_solution",
        "approximate_only",
    }
    aug_types = set(df[df["augmented"] == True]["augmentation_type"].unique())
    bad_values = aug_types & folder_names

    if bad_values:
        errors.append(f"Found folder names in augmentation_type: {bad_values}")
        console.print(f"  [red]✗ ERROR:[/red] Folder names detected: {bad_values}")
    else:
        console.print(f"  [green]✓ PASS:[/green] No folder names in augmentation_type")

    # 2. Validate each strategy
    console.print("\n[bold blue]2. Validating individual strategies...[/bold blue]")

    aug_df = df[df["augmented"] == True]
    strategy_results = []

    for strategy in sorted(aug_df["augmentation_type"].unique()):
        if strategy == "original":
            continue

        subset = aug_df[aug_df["augmentation_type"] == strategy]

        # Check if strategy is expected
        if strategy not in strategy_configs:
            warnings.append(f"Unknown strategy: {strategy}")
            console.print(
                f"  [yellow]⚠ WARNING:[/yellow] Unknown strategy '{strategy}'"
            )
            continue

        expected_sol_types, expected_has_sol, allow_empty = strategy_configs[strategy]

        # Check solution_type
        actual_sol_types = set(subset["solution_type"].unique())
        unexpected_sol = actual_sol_types - set(expected_sol_types)
        if unexpected_sol:
            errors.append(
                f"{strategy}: Unexpected solution_type {unexpected_sol}, expected {expected_sol_types}"
            )
            console.print(
                f"  [red]✗ {strategy}:[/red] Wrong solution_type {unexpected_sol}"
            )

        # Check has_solution
        actual_has_sol = set(subset["has_solution"].unique())
        # Convert to bool for comparison
        actual_has_sol_bool = {
            val for val in actual_has_sol if val in [True, False, "True", "False"]
        }
        actual_has_sol_bool = {
            bool(val) if isinstance(val, str) else val for val in actual_has_sol_bool
        }
        unexpected_has = actual_has_sol_bool - set(expected_has_sol)
        if unexpected_has:
            errors.append(
                f"{strategy}: Unexpected has_solution {unexpected_has}, expected {expected_has_sol}"
            )
            console.print(
                f"  [red]✗ {strategy}:[/red] Wrong has_solution {unexpected_has}"
            )

        # Check empty u consistency
        empty_count = (subset["u"] == "").sum()
        has_sol_count = (
            (subset["has_solution"] == True).sum() if True in expected_has_sol else 0
        )

        if not allow_empty and empty_count > 0:
            errors.append(
                f"{strategy}: Has {empty_count} empty u but should have exact formulas"
            )
            console.print(
                f"  [red]✗ {strategy}:[/red] {empty_count} empty u (should have formulas)"
            )

        # Analyze u field patterns
        def classify_u(u_str):
            """Classify u field content."""
            if not u_str or u_str == "":
                return "empty"
            # Check if contains floats
            import re

            if re.search(r"\d+\.\d+", str(u_str)):
                return "num_coef"
            return "symbolic"

        u_patterns = subset["u"].apply(classify_u).value_counts()
        u_pattern_str = ", ".join([f"{pat}({cnt})" for pat, cnt in u_patterns.items()])

        # Success message
        if strategy in strategy_configs and not any(
            e.startswith(strategy) for e in errors
        ):
            console.print(
                f"  [green]✓ {strategy}:[/green] {len(subset)} equations, all checks passed"
            )

        # Collect stats
        strategy_results.append(
            {
                "Strategy": strategy,
                "Total": len(subset),
                "Has Solution": (subset["has_solution"] == True).sum(),
                "No Solution": (subset["has_solution"] == False).sum(),
                "Solution Types": ", ".join(
                    [
                        f"{st}({cnt})"
                        for st, cnt in subset["solution_type"].value_counts().items()
                    ]
                ),
                "u Patterns": u_pattern_str,
                "Status": "✓"
                if not any(e.startswith(strategy) for e in errors)
                else "✗",
            }
        )

    # 3. Check original dataset
    console.print("\n[bold blue]3. Checking original dataset...[/bold blue]")
    orig_df = df[df["augmented"] == False]

    if len(orig_df) > 0:
        wrong_sol_type = orig_df[orig_df["solution_type"] != "exact"]
        wrong_has_sol = orig_df[orig_df["has_solution"] != True]

        if len(wrong_sol_type) > 0:
            errors.append(
                f"Original dataset: {len(wrong_sol_type)} rows with solution_type != 'exact'"
            )
            console.print(
                f"  [red]✗ ERROR:[/red] {len(wrong_sol_type)} original with wrong solution_type"
            )
        else:
            console.print(
                f"  [green]✓ PASS:[/green] All {len(orig_df)} original have solution_type='exact'"
            )

        if len(wrong_has_sol) > 0:
            errors.append(
                f"Original dataset: {len(wrong_has_sol)} rows with has_solution != True"
            )
            console.print(
                f"  [red]✗ ERROR:[/red] {len(wrong_has_sol)} original with has_solution != True"
            )
        else:
            console.print(
                f"  [green]✓ PASS:[/green] All original have has_solution=True"
            )

    # 4. Summary table
    console.print("\n[bold blue]4. Strategy Summary:[/bold blue]")

    # Check which configured strategies are missing
    configured_strategies = set(strategy_configs.keys())
    actual_strategies = set(aug_df["augmentation_type"].unique()) - {"original"}
    missing_strategies = configured_strategies - actual_strategies

    if missing_strategies:
        console.print(
            f"\n[yellow]⚠ Note:[/yellow] {len(missing_strategies)} configured strategies generated no data:"
        )
        for strat in sorted(missing_strategies):
            console.print(f"  - {strat}")
        console.print()

    table = Table(
        show_header=True,
        header_style="bold magenta",
        title="Strategies that Generated Data",
    )
    table.add_column("Strategy")
    table.add_column("Total", justify="right")
    table.add_column("Has Sol", justify="right")
    table.add_column("No Sol", justify="right")
    table.add_column("Solution Types")
    table.add_column("u Patterns")
    table.add_column("Why u Pattern?")
    table.add_column("Status")

    # Add explanations for u patterns
    u_explanations = {
        "eigenvalue_cases": "Violates Fredholm Alternative",
        "range_violation": "f not in range of operator",
        "divergent_kernel": "Kernel integral diverges",
        "disconnected_support": "Rank-deficient operator",
        "complex_kernels": "No closed-form antiderivative",
        "boundary_layer": "Numerical methods needed",
        "oscillatory_solution": "High-frequency components",
        "weakly_singular": "Integrable singularities",
        "mixed_type": "Combined Volterra+Fredholm",
        "compact_support": "Sparse structure",
        "near_resonance": "Ill-conditioned near eigenvalue",
        "ill_posed": "Fredholm 1st kind",
        "resonance": "Infinite solution family u=C*φ",
    }

    for row in sorted(strategy_results, key=lambda x: x["Total"], reverse=True):
        status_color = "green" if row["Status"] == "✓" else "red"
        explanation = u_explanations.get(row["Strategy"], "N/A")
        table.add_row(
            row["Strategy"],
            str(row["Total"]),
            str(row["Has Solution"]),
            str(row["No Solution"]),
            row["Solution Types"],
            row["u Patterns"],
            explanation,
            f"[{status_color}]{row['Status']}[/{status_color}]",
        )

    console.print(table)

    # Final summary
    console.print("\n[bold]Validation Summary:[/bold]")
    console.print(f"  Total equations: {len(df):,}")
    console.print(f"  Original: {len(orig_df):,}")
    console.print(f"  Augmented: {len(aug_df):,}")
    console.print(f"  Unique strategies: {len(aug_df['augmentation_type'].unique())}")

    if errors:
        console.print(f"\n[bold red]✗ FAILED:[/bold red] {len(errors)} errors found")
        for err in errors:
            console.print(f"  - {err}")
    else:
        console.print(f"\n[bold green]✓ ALL CHECKS PASSED[/bold green]")

    if warnings:
        console.print(
            f"\n[bold yellow]⚠ WARNINGS:[/bold yellow] {len(warnings)} warnings"
        )
        for warn in warnings:
            console.print(f"  - {warn}")

    return {
        "total": len(df),
        "errors": errors,
        "warnings": warnings,
        "strategy_results": strategy_results,
    }


if __name__ == "__main__":
    # Find the dataset
    project_root = Path(__file__).parent.parent
    csv_path = (
        project_root
        / "data"
        / "processed"
        / "training_data_v3"
        / "augmented"
        / "Fredholm_Dataset_Sample_augmented.csv"
    )

    if not csv_path.exists():
        console.print(f"[red]✗ ERROR:[/red] Dataset not found at {csv_path}")
        sys.exit(1)

    results = validate_dataset(csv_path)

    # Exit with error code if validation failed
    sys.exit(0 if not results["errors"] else 1)
