# Data Augmentation Guide

Comprehensive guide to edge case augmentation strategies for Fredholm integral equations.

## Overview

The augmentation system generates realistic edge cases organized by **solution type**, teaching LLMs to recognize when equations require different solution approaches.

## Folder-Based Strategy System

Strategies are organized by **solution type**. Specifying a folder name runs ALL strategies in that folder:

| Folder | Strategies | Variants | What it teaches |
|--------|------------|----------|-----------------|
| `approx_coef` | 5 | 15 | Functional forms with numerical params (boundary layers, oscillations, weak singularities) |
| `discrete_points` | 2 | 6 | Pure point samples, no formula (complex kernels, near-resonance) |
| `series` | 1 | 3 | Truncated series expansions (Neumann series, N=4 terms) |
| `family` | 1 | 3 | Non-unique solution families (exact resonance) |
| `regularized` | 1 | 3 | Ill-posed problems requiring regularization (Fredholm 1st kind) |
| `none_solution` | 4 | 12 | No solution exists (eigenvalue issues, range violations, divergent kernels) |
| **Total** | **14** | **42** | Comprehensive edge case recognition |

## Solution Type Taxonomy (7 types)

The augmentation system produces equations with 7 distinct solution types:

| Solution Type | Description | Example | How to Solve |
|--------------|-------------|---------|--------------|
| `exact_symbolic` | Closed-form symbolic solution | u(x) = sin(x) + x² | Analytical methods |
| `approx_coef` | Approximate with NUMERIC coefficients | u(x) = 0.5*sin(x) + 1.2*x | Collocation, least squares |
| `discrete_points` | Solution only at discrete points | [(0, 1.2), (0.5, 3.4), ...] | Numerical integration |
| `series` | Infinite series solution | u(x) = Σ aₙxⁿ | Neumann series, perturbation |
| `family` | Non-unique with ARBITRARY parameters | u(x) = c₁*sin(x) + c₂*cos(x) | Identify eigenspace |
| `regularized` | Requires regularization | Tikhonov, Landweber | Ill-posed, stabilize |
| `none` | No solution exists | N/A | Recognize impossibility |

### Why 7 Types?

**Pedagogical Clarity**: Each type represents a distinct mathematical concept and solution methodology.

**Evaluation Strategy**: Different types require different evaluation approaches:
- `exact_symbolic`: Symbolic equivalence (SymPy)
- `approx_coef`: Coefficient comparison, numerical error
- `discrete_points`: Point-wise accuracy (MSE, MAE)
- `series`: Convergence analysis
- `family`: Identify solution space dimension
- `regularized`: Solution stability, regularization parameter
- `none`: Binary classification (has_solution=False)

## Unified Output Schema

All augmentation strategies produce the same 18 core fields:

```python
{
    # Original equation components
    "u": str,           # Solution (empty "" if none exists)
    "f": str,           # Right-hand side
    "kernel": str,      # Kernel function K(x,t)
    "lambda_val": float,
    "a": float,
    "b": float,
    
    # Edge case metadata
    "augmentation_type": str,      # Strategy name
    "has_solution": bool,          # Solution exists?
    "solution_type": str,          # One of 7 types
    "edge_case": str,              # Brief description
    "reason": str,                 # Why this edge case occurs
    "recommended_methods": list,   # Suggested solution approaches
    
    # Dataset management
    "equation_id": str,            # Unique identifier
    "is_augmented": bool,          # True for edge cases
    "augmented_from": str,         # Original equation ID
    "augmentation_count": int,     # Number of variants
    "augmentation_date": str,      # ISO timestamp
    "augmentation_version": str    # Schema version
}
```

**Optional metadata** (60+ fields): Add `--include-edge-metadata` flag to include detailed technical fields like `singularity_type`, `layer_location`, `oscillation_frequency`, etc.

## Edge Case Strategies

### 1. Approximate Coefficient Solutions (`approx_coef`)

**Strategies**: `weakly_singular`, `boundary_layer`, `oscillatory`, `mixed_type`, `compact_support`

**Characteristics**:
- Solution exists but no closed-form expression
- Can be approximated as functional form with numerical coefficients
- Example: u(x) ≈ 0.5 + 0.3·sin(x) - 0.1·x²

**Use cases**:
- Weakly singular kernels (K ~ 1/√|x-t|)
- Rapid boundary layers at endpoints
- High-frequency oscillations
- Mixed hyperbolic-parabolic behavior
- Compact support constraints

### 2. Discrete Point Solutions (`discrete_points`)

**Strategies**: `complex_kernels`, `near_resonance`

**Characteristics**:
- No analytical form, even approximate
- Solution available only as sample points
- Requires numerical integration methods

**Use cases**:
- Highly complex, non-separable kernels
- Near-resonance conditions (ill-conditioned but solvable)

### 3. Series Solutions (`series`)

**Strategies**: `neumann_series`

**Characteristics**:
- Solution as truncated series expansion
- 4-term Neumann series: u = f + λKf + λ²K²f + λ³K³f + ...

**Use cases**:
- Small λ (|λ| < 1/||K||)
- Well-behaved kernels
- Perturbation methods

### 4. Solution Families (`family`)

**Strategies**: `resonance`

**Characteristics**:
- Non-unique solutions (exact resonance)
- u(x) = u_particular(x) + C·φ(x) where Kφ = λφ
- Infinite solutions parameterized by constant C

**Use cases**:
- λ is eigenvalue of kernel operator
- Fredholm alternative applies
- Solution space has dimension > 1

### 5. Regularized Solutions (`regularized`)

**Strategies**: `ill_posed`

**Characteristics**:
- Fredholm 1st kind: ∫K(x,t)u(t)dt = g(x)
- Ill-posed, requires regularization
- Extremely sensitive to noise

**Use cases**:
- Inverse problems
- Tikhonov regularization
- Truncated SVD

### 6. No Solution (`none_solution`)

**Strategies**: `eigenvalue_issue`, `range_violation`, `divergent_kernel`, `disconnected_support`

**Characteristics**:
- No solution exists (has_solution=False)
- u = "" (empty string)

**Use cases**:
- λ at eigenvalue, Fredholm alternative violated
- f(x) not in range of (I - λK)
- Kernel diverges/undefined
- Domain connectivity issues

## Dataset Balance Recommendations

**Target distribution** for realistic training:

| Category | Percentage | Count (5K base) | Strategies |
|----------|-----------|-----------------|------------|
| Original (exact_symbolic) | 85-90% | 4,250-4,500 | - |
| Edge cases | 10-15% | 500-750 | 14 strategies |

**Within edge cases** (13% of total):

| Solution Type | % of Edge | % of Total | Count | Folders |
|--------------|-----------|------------|-------|---------|
| approx_coef | 35% | 4.6% | 230 | 5 strategies, 15 variants |
| discrete_points | 15% | 2.0% | 100 | 2 strategies, 6 variants |
| series | 8% | 1.0% | 50 | 1 strategy, 3 variants |
| family | 8% | 1.0% | 50 | 1 strategy, 3 variants |
| regularized | 8% | 1.0% | 50 | 1 strategy, 3 variants |
| none | 26% | 3.4% | 170 | 4 strategies, 12 variants |

**Multiplier calculation**:
- For 15% edge cases: `--augment-multiplier 1.176` (1 / 0.85)
- For 13% edge cases: `--augment-multiplier 1.149` (1 / 0.87)
- For 10% edge cases: `--augment-multiplier 1.111` (1 / 0.90)

## Usage Examples

### All Edge Cases (Recommended)

```bash
# Use main CLI: uv run python -m src.cli run --config <yaml> \
  --input data/raw/Fredholm_Dataset_Sample.csv \
  --output data/processed/training_data \
  --augment \
  --augment-multiplier 1.15 \
  --augment-strategies approx_coef discrete_points series family regularized none_solution
```

### Specific Categories Only

```bash
# Only approximate coefficient cases
# Use main CLI: uv run python -m src.cli run --config <yaml> \
  --augment --augment-multiplier 1.2 \
  --augment-strategies approx_coef

# Only no-solution cases
# Use main CLI: uv run python -m src.cli run --config <yaml> \
  --augment --augment-multiplier 1.3 \
  --augment-strategies none_solution

# Combination: numerical-only + no-solution
# Use main CLI: uv run python -m src.cli run --config <yaml> \
  --augment --augment-multiplier 1.25 \
  --augment-strategies approx_coef discrete_points none_solution
```

### Individual Strategies

```bash
# Only weakly singular kernels
# Use main CLI: uv run python -m src.cli run --config <yaml> \
  --augment --augment-multiplier 1.1 \
  --augment-strategies weakly_singular

# Only boundary layer problems
# Use main CLI: uv run python -m src.cli run --config <yaml> \
  --augment --augment-multiplier 1.15 \
  --augment-strategies boundary_layer
```

## Validation

Validate augmented data quality:

```bash
# Validation integrated in pipeline \
  data/processed/training_data/augmented/Fredholm_Dataset_augmented.csv \
  --strategies all \
  --check-schema \
  --check-balance \
  --verbose
```

**Validation checks**:
- Schema compliance (18 required fields present)
- Solution type consistency
- Empty string for has_solution=False
- Non-empty u for has_solution=True
- Edge case distribution balance
- Variant count per strategy

## Implementation Notes

### SymPy Expression Compatibility (February 11, 2026)

All augmentation strategies now generate **parseable SymPy expressions** for kernel definitions:

**✅ Piecewise Expressions**: Replaced placeholder strings and ternary operators with valid Piecewise syntax:
- ❌ Old: `"Piecewise: nonzero in two disconnected regions"`
- ✅ New: `"Piecewise((1, (x>=a1) & (x<=b1) & (t>=a2) & (t<=b2)), (0, True))"`

**✅ Logical Operators**: Use SymPy operators `&`, `|`, `~` instead of Python keywords:
- ❌ Old: `"sin(x)*cos(t) if (c <= x <= d and c <= t <= d) else 0"`
- ✅ New: `"Piecewise((sin(x)*cos(t), (x>=c) & (x<=d) & (t>=c) & (t<=d)), (0, True))"`

**✅ Series Expansions**: Use `Integral` notation instead of placeholders:
- ❌ Old: `"f(x) + λ*∫... + λ³*[triple_integral]"`
- ✅ New: `"f(x) + λ*Integral(exp(-Abs(x-t))*f(t), (t, a, b)) + ..."`

**Fixed strategies**: `disconnected_support` (2 locations), `mixed_type`, `compact_support` (2 locations), `neumann_series`

### Overflow-Safe Evaluation Points (February 11, 2026)

The `_generate_evaluation_points()` method in `BaseAugmentation` now handles numerical overflow gracefully:

**Error Suppression**: Wraps lambdify evaluation in `np.errstate()` context:
```python
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    u_values = np.array([u_lambda(float(xi)) for xi in x_values], dtype=float)
```

**Non-Finite Filtering**: Automatically drops inf/nan values from overflow cases:
```python
finite_mask = np.isfinite(u_values)
x_values = x_values[finite_mask]
u_values = u_values[finite_mask]
```

**Benefits**:
- No RuntimeWarnings from `exp(100*x)`, `cosh(large_value)` overflows
- Robust evaluation for high-frequency oscillations, boundary layers
- All `has_solution=True` augmentations inherit this safety

## See Also

- [Edge Case Documentation](EDGE_CASES.md) - Detailed mathematical definitions
- [Quick Start](QUICKSTART.md) - Getting started guide
- [Features](FEATURES.md) - Implementation status
- Augmentation code: `src/data/augmentations/`

