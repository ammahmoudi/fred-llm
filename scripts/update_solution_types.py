"""
Script to update solution_type values across all augmentation strategies.
Run this once to migrate from 5-type to 8-type solution taxonomy.
"""

import re
from pathlib import Path

# Define the mappings
UPDATES = {
    # Approx coef strategies - have functional form in u
    "boundary_layer.py": [
        ('solution_type": "numerical"', 'solution_type": "approx_coef"')
    ],
    "oscillatory_solution.py": [
        ('solution_type": "numerical"', 'solution_type": "approx_coef"')
    ],
    "weakly_singular.py": [
        ('solution_type": "numerical"', 'solution_type": "approx_coef"')
    ],
    "mixed_type.py": [('solution_type": "numerical"', 'solution_type": "approx_coef"')],
    "compact_support.py": [
        ('solution_type": "numerical"', 'solution_type": "approx_coef"')
    ],
}


def update_file(file_path: Path, replacements: list[tuple[str, str]]):
    """Apply replacements to a file."""
    content = file_path.read_text(encoding="utf-8")
    original = content

    for old, new in replacements:
        content = content.replace(old, new)

    if content != original:
        file_path.write_text(content, encoding="utf-8")
        print(f"✓ Updated {file_path.name}")
        return True
    else:
        print(f"✗ No changes in {file_path.name}")
        return False


def main():
    base_dir = Path(__file__).parent.parent / "src" / "data" / "augmentations"

    print("=" * 80)
    print("SOLUTION TYPE MIGRATION: 5-type → 8-type taxonomy")
    print("=" * 80)

    updated_count = 0

    # Process numerical_only folder
    numerical_dir = base_dir / "numerical_only"
    for filename, replacements in UPDATES.items():
        file_path = numerical_dir / filename
        if file_path.exists():
            if update_file(file_path, replacements):
                updated_count += 1
        else:
            print(f"⚠ File not found: {file_path}")

    print(f"\n{updated_count} files updated successfully!")

    print("\nMigration Summary:")
    print("- exact → exact_symbolic (substitute, scale, shift, compose) ✓")
    print("- numerical → discrete_points (complex_kernels, near_resonance) ✓")
    print(
        "- numerical → approx_coef (boundary_layer, oscillatory, weakly_singular, mixed_type, compact_support)"
    )
    print("- none, family, regularized → unchanged ✓")


if __name__ == "__main__":
    main()
