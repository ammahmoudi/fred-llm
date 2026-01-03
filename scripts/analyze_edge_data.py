#!/usr/bin/env python3
"""Quick analysis of edge case augmented data."""

import json
from collections import defaultdict
from pathlib import Path

# Load augmented data
data_path = Path(
    "data/processed/training_data/augmented/Fredholm_Dataset_Sample_augmented.json"
)
data = json.load(open(data_path))

# Filter edge cases
edge_cases = [d for d in data if d.get("edge_case")]
print(f"Total edge cases: {len(edge_cases)}")
print(f"Total original: {len(data) - len(edge_cases)}")
print(f"Edge case ratio: {len(edge_cases) / len(data) * 100:.1f}%")
print()

# Group by solution type
by_type = defaultdict(list)
for ec in edge_cases:
    sol_type = ec.get("solution_type", "unknown")
    by_type[sol_type].append(ec)

print("Distribution by solution_type:")
for sol_type in sorted(by_type.keys()):
    count = len(by_type[sol_type])
    print(f"  {sol_type}: {count} ({count / len(edge_cases) * 100:.1f}%)")
print()

# Check u field patterns
print("Sample u values by solution_type:")
for sol_type in sorted(by_type.keys()):
    samples = by_type[sol_type]
    u_values = [s.get("u", "") for s in samples]
    non_empty = [u for u in u_values if u]

    # Check for approximate coefficients
    with_floats = [u for u in non_empty if "." in str(u)]

    print(f"\n{sol_type}:")
    print(f"  Total: {len(samples)}")
    print(f"  Non-empty u: {len(non_empty)}")
    print(f"  With floats: {len(with_floats)}")

    # Show unique samples
    unique_u = list(set(non_empty))[:5]
    if unique_u:
        print(f"  Sample u values:")
        for u in unique_u:
            print(f"    {u[:100]}")
    else:
        print(f"  All u values are empty")
