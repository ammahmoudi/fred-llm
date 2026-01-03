#!/usr/bin/env python3
"""Detailed analysis of u field patterns in augmented data."""

import json
from collections import defaultdict
from pathlib import Path

# Load augmented data
data_path = Path(
    "data/processed/training_data_v2/augmented/Fredholm_Dataset_Sample_augmented.json"
)
data = json.load(open(data_path))

print("=" * 80)
print("AUGMENTED DATA QUALITY CHECK")
print("=" * 80)

# 1. Check has_solution field
print("\n1. has_solution field completeness:")
total = len(data)
missing_has = [d for d in data if d.get("has_solution") is None]
print(f"   Total entries: {total}")
print(
    f"   Missing has_solution: {len(missing_has)} ({len(missing_has) / total * 100:.1f}%)"
)

if missing_has:
    by_type = defaultdict(int)
    for d in missing_has:
        by_type[d.get("solution_type", "unknown")] += 1
    print(f"   Breakdown by solution_type:")
    for t, count in sorted(by_type.items()):
        print(f"     {t}: {count}")

# 2. Check solution_type='none' with non-empty u
print("\n2. solution_type='none' consistency:")
none_type = [d for d in data if d.get("solution_type") == "none"]
none_with_u = [d for d in none_type if d.get("u")]
print(f"   Total 'none' type: {len(none_type)}")
print(
    f"   With non-empty u: {len(none_with_u)} ({len(none_with_u) / len(none_type) * 100:.1f}%)"
)

if none_with_u:
    print(f"   Sample inconsistent entries:")
    for i, d in enumerate(none_with_u[:3]):
        print(
            f"     {i + 1}. u='{d.get('u')[:50]}', variant={d.get('augmentation_variant')}"
        )

# 3. Analyze u field patterns
print("\n3. u field patterns by solution_type:")
edge_cases = [d for d in data if d.get("edge_case")]
by_sol_type = defaultdict(list)
for d in edge_cases:
    by_sol_type[d.get("solution_type", "unknown")].append(d)

for sol_type in sorted(by_sol_type.keys()):
    entries = by_sol_type[sol_type]
    u_values = [d.get("u", "") for d in entries]

    empty_u = sum(1 for u in u_values if not u)
    non_empty_u = len(u_values) - empty_u
    with_floats = sum(1 for u in u_values if u and "." in str(u))

    print(f"\n   {sol_type}:")
    print(f"     Total: {len(entries)}")
    print(f"     Empty u: {empty_u} ({empty_u / len(entries) * 100:.1f}%)")
    print(f"     Non-empty u: {non_empty_u} ({non_empty_u / len(entries) * 100:.1f}%)")
    print(
        f"     With float coefficients: {with_floats} ({with_floats / len(entries) * 100:.1f}%)"
    )

    # Show samples
    unique_u = list(set(u for u in u_values if u))[:3]
    if unique_u:
        print(f"     Sample u values:")
        for u in unique_u:
            print(f"       {u[:80]}")

# 4. Check original data pattern
print("\n4. Original data baseline:")
original = [d for d in data if not d.get("edge_case")]
orig_empty_u = sum(1 for d in original if not d.get("u"))
orig_with_floats = sum(1 for d in original if d.get("u") and "." in str(d.get("u")))

print(f"   Total original: {len(original)}")
print(f"   Empty u: {orig_empty_u}")
print(f"   With float coefficients: {orig_with_floats}")
print(f"   All should have: has_solution=True, solution_type='exact'")

# 5. Summary
print("\n" + "=" * 80)
print("ISSUES FOUND:")
print("=" * 80)

issues = []
if missing_has:
    issues.append(f"✗ {len(missing_has)} entries missing 'has_solution' field")
if none_with_u:
    issues.append(f"✗ {len(none_with_u)} 'solution_type=none' entries have non-empty u")
if orig_empty_u:
    issues.append(f"✗ {orig_empty_u} original entries have empty u")

if issues:
    for issue in issues:
        print(f"  {issue}")
else:
    print("  ✓ No issues found!")
