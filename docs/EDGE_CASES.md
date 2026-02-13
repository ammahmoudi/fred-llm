# Edge Case Augmentation Strategies

This document explains how edge cases are created for the Fredholm-LLM dataset to improve model robustness and real-world applicability.

## Overview

Edge cases are augmented variants of Fredholm integral equations that represent challenging or special scenarios that a model might encounter in practice. Unlike the original dataset which contains equations with exact symbolic solutions, edge cases test a model's ability to:

- Recognize when no solution exists
- Handle equations requiring numerical approximation
- Identify ill-posed problems needing regularization
- Detect non-unique solutions and solution families
- Distinguish between different solution representations

**Folder-Based Organization** - Strategies are organized by solution type (folder names match solution_type values). When you use a folder name, ALL strategies in that folder are executed:

- `none_solution/` → 4 strategies → 12 variants per equation
- `approx_coef/` → 5 strategies → 15 variants per equation
- `discrete_points/` → 2 strategies → 6 variants per equation
- `series/` → 1 strategy → 3 variants per equation
- `family/` → 1 strategy → 3 variants per equation
- `regularized/` → 1 strategy → 3 variants per equation

**Total: 14 edge case strategies × 3 variants each = 42 edge case types**

## Solution Type Taxonomy (7 Types)

**Updated:** January 3, 2026 - Removed `exact_coef` (redundant with `family`, impossible to evaluate).

| Solution Type | Description | u Field Content | Has Solution | Example Strategies |
|---------------|-------------|-----------------|--------------|-------------------|
| `exact_symbolic` | Closed-form analytical expressions | Formula: `2*x + sin(x)` | ✅ True | substitute, scale, shift |
| `approx_coef` | Functional form with numerical parameters | Formula: `exp(-x/0.01)` | ✅ True | boundary_layer, oscillatory |
| `discrete_points` | Pure point samples, no closed form | Empty `""` + sample arrays | ✅ True | complex_kernels, near_resonance |
| `series` | Truncated Neumann/Taylor series | Series formula: `f + λ∫K + λ²∫K²` | ✅ True | neumann_series |
| `family` | Non-unique solutions (infinite family) | General form: `C*φ(x) + u_p` | ✅ True | resonance |
| `regularized` | Ill-posed, requires Tikhonov/TSVD | Empty `""` | ✅ True | ill_posed |
| `none` | No solution exists | Empty `""` | ❌ False | eigenvalue_cases, range_violation |

### Evaluation Guidelines by Solution Type

- **With u formula** (`exact_symbolic`, `approx_coef`, `series`, `family`): Compare symbolic/numerical match
- **Empty u** (`discrete_points`, `regularized`, `none`): Check correct `has_solution` and `solution_type` classification

See [FEATURES.md](FEATURES.md) for complete implementation history and the [Augmentation Strategies README](../src/data/augmentations/README.md) for usage details.

## Why Edge Cases Matter

Real-world applications of Fredholm equations often involve:
- **Incomplete information** → No exact solution available
- **Complex kernels** → Only numerical solutions feasible  
- **Ill-conditioned problems** → Small data changes cause large solution variations

Training exclusively on well-posed equations with exact solutions can lead to models that:
- Hallucinate solutions where none exist
- Fail to recognize when approximation is needed
- Miss opportunities to recommend appropriate numerical methods

## Output Format

### Consistent Data Structure

All augmented equations maintain **consistent key-value pairs** for reliable training:

**Standard Fields** (present in all entries):
- `u`, `f`, `kernel`, `lambda_val`, `a`, `b`
- `has_solution`, `solution_type`, `augmented`, `augmentation_type`, `augmentation_variant`
- `source_file`, `source_path`

**Edge Case Fields** (strategy-specific):
- `edge_case` - Type identifier (e.g., "boundary_layer", "weakly_singular")
- Strategy metadata (e.g., `layer_location`, `singularity_type`, `oscillation_frequency`)
- `recommended_methods` - Suggested solution approaches
- `numerical_challenge` - Description of the computational difficulty

## Edge Case Types

**Directory Structure**: Organized by **solution type** (folder names = solution_type values)

```
src/data/augmentations/
├── exact_symbolic/      # → Exact closed-form solutions
├── approx_coef/        # → Functional form with numerical params
├── discrete_points/    # → Pure point samples (empty u)
├── series/             # → Truncated series expansions
├── family/             # → Solution family: u = u₀ + C*φ
├── regularized/        # → "Needs regularization" (empty u)
└── none_solution/      # → "No solution exists" (empty u)
```

| Solution Type | Strategies | What LLM Outputs |
|---------------|-----------|------------------|
| **None** | eigenvalue_cases, range_violation, divergent_kernel, disconnected_support | "No solution exists because..." |
| **Approx Coef** | boundary_layer, oscillatory_solution, weakly_singular, mixed_type, compact_support | Functional form: exp(-x/ε), sin(ωx) |
| **Discrete Points** | complex_kernels, near_resonance | Point values: u(x₁)≈v₁, u(x₂)≈v₂... |
| **Series** | neumann_series | Truncated series: f + λ∫K + λ²∫K² + ... |
| **Regularization Required** | ill_posed | "Ill-posed, needs Tikhonov/TSVD..." |
| **Non-Unique** | resonance | "Solution family: u = C*φ(x) + u_p" |

**Total: 14 strategies × 3 variants each = 42 edge case types**

---

## Group 1: Original Edge Cases (Production-Tested)

### 1. Eigenvalue Cases (`eigenvalue_cases`)

**Purpose**: Teach models to recognize unsolvable equations where λ is an eigenvalue.

**Creation Strategy**:
For each base equation `u(x) - λ∫K(x,t)u(t)dt = f(x)`, three variants are created:

#### Variant 1: Contradictory Kernel
- **Modification**: Set kernel `K(x,t) = 0` while keeping `λ ≠ 0`
- **Mathematical Effect**: Reduces equation to `u(x) = f(x)`, but f(x) chosen inconsistently
- **Example**:
  ```
  Original: u(x) - 2∫[0,1] sin(x*t) u(t) dt = cos(x)
  Modified: u(x) - 2∫[0,1] 0 u(t) dt = cos(x) + sin(x)
  Result: u(x) = cos(x) + sin(x), but this contradicts the requirement
  ```
- **Metadata Added**:
  - `has_solution: false`
  - `reason: "contradictory_kernel"`
  - `edge_case: "eigenvalue_cases"`

#### Variant 2: Invalid Domain
- **Modification**: Set integration bounds where `a > b`
- **Mathematical Effect**: Creates meaningless integral
- **Example**:
  ```
  Original: Domain [0, 1]
  Modified: Domain [5, 1]  (invalid)
  ```
- **Metadata Added**:
  - `has_solution: false`
  - `reason: "invalid_domain"`
  - `edge_case: "eigenvalue_cases"`

#### Variant 3: Incompatible Lambda
- **Modification**: Set `λ = 1/eigenvalue` where eigenvalue is of integral operator
- **Mathematical Effect**: Creates singular/non-invertible operator
- **Example**:
  ```
  For operator with eigenvalue λ₀ = 2
  Set λ = 0.5 = 1/λ₀
  Equation becomes non-solvable (Fredholm alternative)
  ```
- **Metadata Added**:
  - `has_solution: false`
  - `reason: "incompatible_lambda"`
  - `edge_case: "eigenvalue_cases"`

---

### 2. Complex Kernels (`complex_kernels`)

**Purpose**: Represent equations requiring numerical methods.

**Creation Strategy**:
Transform equations to have complex kernels without known closed-form solutions:

#### Variant 1: Non-Separable Kernel
- **Modification**: Replace `K(x,t)` with `exp(-(x-t)²) * sin(x*t)`
- **Mathematical Effect**: Creates non-separable kernel requiring discretization
- **Example**:
  ```
  Original: K(x,t) = x*t  (separable)
  Modified: K(x,t) = exp(-(x-t)²) * sin(x*t)  (non-separable)
  ```
- **Numerical Methods**:
  - Nyström method
  - Collocation method
  - Quadrature-based discretization

#### Variant 2: Highly Oscillatory Kernel
- **Modification**: Add high-frequency component `K(x,t) → K(x,t) * cos(10πxt)`
- **Mathematical Effect**: Requires fine discretization
- **Example**:
  ```
  Original: K(x,t) = x + t
  Modified: K(x,t) = (x + t) * cos(10*π*x*t)
  ```

#### Variant 3: Discontinuous Kernel
- **Modification**: Introduce piecewise definition or absolute values
- **Mathematical Effect**: Breaks smoothness assumptions
- **Example**:
  ```
  Original: K(x,t) = x² - t²
  Modified: K(x,t) = |x - t| + sign(x - t) * (x² + t²)
  ```

**Metadata Added**:
- `has_solution: true`
- `solution_type: "discrete_points"`
- `recommended_methods: ["nystrom", "collocation", "quadrature"]`
- `edge_case: "complex_kernels"`

---

### 3. Ill-Posed Cases (`ill_posed`)

**Purpose**: Identify problems requiring regularization techniques.

**Creation Strategy**:
Modify equations to exhibit sensitivity to perturbations:

#### Variant 1: Nearly Singular Kernel
- **Modification**: Make kernel almost rank-deficient
- **Mathematical Effect**: Small data changes → large solution changes
- **Example**:
  ```
  K(x,t) = ε * original_kernel(x,t)  where ε ≈ 10⁻⁶
  Solution becomes unstable
  ```

#### Variant 2: Compact Support Issues
- **Modification**: Use Green's function with near-singularities
- **Mathematical Effect**: Creates sensitivity to discretization
- **Example**:
  ```
  K(x,t) = 1/|x - t + ε|  where ε → 0
  ```

#### Variant 3: Ill-Conditioned Operator
- **Modification**: Eigenvalues cluster near zero
- **Mathematical Effect**: Condition number → ∞
- **Example**:
  ```
  Modify λ to approach critical value
  Operator (I - λK) becomes nearly non-invertible
  ```

**Metadata Added**:
- `has_solution: true` (regularized solution exists)
- `solution_type: "regularized"`
- `is_ill_posed: true`
- `requires_regularization: true`
- `recommended_methods: ["tikhonov", "truncated_svd", "iterative"]`
- `condition_estimate: large_number`
- `edge_case: "ill_posed"`

**Note**: Updated January 2, 2026 - Ill-posed problems DO have solutions (they're just unstable without regularization). The `has_solution` field is now correctly set to `true` for all ill_posed variants.

---

## Group 2: Advanced Edge Cases

### 4. Weakly Singular Kernels (`weakly_singular`)

**Purpose**: Kernels with integrable but problematic singularities.

- **Variant 1**: Logarithmic singularity `K(x,t) = log|x-t| + x + t`
- **Variant 2**: Power law singularity `K(x,t) = |x-t|^(-0.5) * (x + t)`
- **Variant 3**: Mixed smooth + singular `K(x,t) = log|x-t + 0.1| * sin(πx)`

**Key Features**: Graded mesh generation, singularity order tracking

### 5. Boundary Layer Solutions (`boundary_layer`)

**Purpose**: Solutions with sharp gradients near boundaries (ε=0.01).

- **Variant 1**: Left boundary layer `u(x) = exp(-(x-a)/ε)`
- **Variant 2**: Right boundary layer `u(x) = exp(-(b-x)/ε)`
- **Variant 3**: Double layer `u(x) = tanh((x-c)/ε)`

**Key Features**: Exponentially graded mesh, gradient scale estimation

### 6. Resonance (`resonance`) - exact eigenvalue → infinite solution families

- **Variant 1**: Separable at eigenvalue (λ=2, K=sin(πx)sin(πt))
- **Variant 2**: Constant kernel at critical value (λ=1, K=1)

**Key Features**: Solution multiplicity, general solution representation u = C*φ(x)

**Metadata**: 
- `solution_type: "family"`
- `has_solution: true`
- Symbolic u: "C*sin(pi*x)" where C is arbitrary constant

---

### 8. Near-Resonance Cases (`near_resonance`)

**Purpose**: λ near eigenvalue → ill-conditioned but unique solutions.

- **Variant 1**: Close to separable eigenvalue (λ=2.1, λ_critical=2.0)
- **Variant 2**: Near constant kernel critical value (λ=1.05, λ_critical=1.0)
- **Variant 3**: Very close approach (λ=2.01, λ_critical=2.0)

**Key Features**: Large amplitude solutions, high condition numbers, sensitivity to perturbations

**Metadata**:
- `solution_type: "discrete_points"`
- `has_solution: true`
- `u: ""` (empty - no simple closed form)
- `near_critical_value`, `distance_to_resonance`, `condition_number_estimate`

**Physical Analogy**: Like forcing a spring near its natural frequency - solution exists but has very large amplitude.

**Mathematical Distinction from Resonance**:
- **resonance**: λ = λ_critical → infinite solutions (u = C*φ)
- **near_resonance**: λ ≈ λ_critical → unique solution with large amplitude

**Key Features**: Solution multiplicity, general solution representation

---

## Group 3: Extended Edge Cases

### 7. Range Violation (`range_violation`)

**Purpose**: RHS not in range of operator → no solution exists.

- **Variant 1**: Even/odd parity mismatch
- **Variant 2**: Separable orthogonal structure
- **Variant 3**: Finite rank operator with incompatible RHS

**Key Features**: Operator property analysis, orthogonality checking

### 9. Divergent Kernel (`divergent_kernel`)

**Purpose**: Non-integrable singularities (integral diverges).

- **Variant 1**: Cauchy singularity `K(x,t) = 1/|x-t|`
- **Variant 2**: Second-order pole `K(x,t) = 1/|x-t|²`
- **Variant 3**: Mixed smooth + divergent

**Key Features**: Contrasts with weakly_singular, singularity order ≥ 1

### 10. Mixed Type (`mixed_type`)

**Purpose**: Part Volterra (causal), part Fredholm (acausal).

- **Variant 1**: Piecewise split at x=t
- **Variant 2**: Smooth transition with tanh
- **Variant 3**: Explicit two-integral formulation

**Key Features**: Causal structure analysis, split point identification

### 11. Oscillatory Solution (`oscillatory_solution`)

**Purpose**: Rapidly oscillating u(x) → Nyquist sampling required.

- **Variant 1**: High-frequency sine (ω=100π)
- **Variant 2**: Amplitude modulated
- **Variant 3**: Multi-frequency beating

**Key Features**: Fine mesh generation, Nyquist criterion validation

### 12. Compact Support (`compact_support`)

**Purpose**: Kernel zero in large regions → sparse structure, requires specialized numerical methods.

- **Variant 1**: Band-limited `K(x,t) = K₀(x,t) if |x-t|<δ else 0`
- **Variant 2**: Localized box function

**Key Features**: Sparse matrix exploitation, efficient storage/computation

**Note**: Previously included a "disconnected regions" variant which created rank-deficient operators. This has been separated into its own strategy.

---

### 13. Disconnected Support (`disconnected_support`)

**Purpose**: Kernels with disconnected support regions → rank-deficient operators → no solution.

- **Variant 1**: Two disjoint support intervals for x and t
- **Variant 2**: Asymmetric disconnected regions (different gaps for x vs t)
- **Variant 3**: Narrow window supports separated by zero regions

**Mathematical Background**: When kernel support is split into disconnected regions, the integral operator loses rank and may not be invertible, leading to equations with no solution.

**Why Separated from compact_support**: 
- **compact_support**: Sparse but full-rank → numerical solution exists
- **disconnected_support**: Rank-deficient → no solution exists

This distinction teaches models to recognize structural causes of non-existence versus computational complexity.

---

## Usage

### Generate Edge Cases

Use the preparation script for flag-based runs:

```bash
python scripts/prepare_dataset.py \
  --input data/raw/Fredholm_Dataset_Sample.csv \
  --output data/processed/training_data \
  --augment \
  --augment-multiplier 1.15 \
  --augment-strategies approx_coef discrete_points series family regularized none_solution \
  --no-convert
```

Folder names map to strategy groups:

- `none_solution`: eigenvalue_cases, range_violation, divergent_kernel, disconnected_support
- `approx_coef`: boundary_layer, oscillatory_solution, weakly_singular, mixed_type, compact_support
- `discrete_points`: complex_kernels, near_resonance
- `series`: neumann_series
- `family`: resonance
- `regularized`: ill_posed

### Dataset Balance Recommendations

| Multiplier | Original (Exact) | Edge Cases | Use Case |
|-----------|------------------|------------|----------|
| **1.1-1.2** | **83-91%** | **9-17%** | **Recommended for 14 strategies** |
| 1.3 | 77% | 23% | More aggressive (use selectively) |
| 1.5 | 67% | 33% | Maximum diversity |

**Example (5,000 base, multiplier 1.15):**
- Original equations: 5,000 (87%)
- Edge cases: 750 (13%, ~54 per strategy)
- **Total**: 5,750

## Testing Status

- Tests live in `tests/test_augmentation.py`.
- Run all augmentation tests:

```bash
pytest tests/test_augmentation.py -v
```

## Implementation Details

### Code Location

- Strategy implementations: `src/data/augmentations/<solution_type>/`
- Integration: `src/data/augmentation.py`
- Validation: `src/data/validator.py`
- Tests: `tests/test_augmentation.py`
- Docs: `docs/EDGE_CASES.md`, `src/data/augmentations/README.md`

### Metadata Schema

Edge case equations include additional fields:

```json
{
  "u": "solution_expression",
  "f": "rhs_expression",
  "kernel": "kernel_expression",
  "lambda_val": 1.0,
  "a": 0,
  "b": 1,

  "augmented": true,
  "augmentation_type": "boundary_layer",
  "augmentation_variant": "v1",
  "edge_case": "boundary_layer",
  "has_solution": true,
  "solution_type": "approx_coef",
  "reason": "boundary layer at x=a",
  "recommended_methods": ["collocation", "least_squares"],
  "numerical_challenge": "stiff boundary layer",

  "evaluation_points": {
    "x_values": [0.0, 0.1],
    "u_values": [1.0, 0.9],
    "n_points": 50
  }
}
```

---

## References

1. Fredholm, I. (1903). "Sur une classe d'équations fonctionnelles"
2. Kress, R. (2014). "Linear Integral Equations" (3rd ed.)
3. Atkinson, K. E. (1997). "The Numerical Solution of Integral Equations of the Second Kind"
4. Hansen, P. C. (2010). "Discrete Inverse Problems" (regularization techniques)

---

## See Also

- [Main README](../README.md) - Project overview and setup
- [Augmentation Strategies README](../src/data/augmentations/README.md) - Detailed augmentation documentation
- [Pipeline Documentation](pipeline-diagram.md) - Full data processing pipeline
- [Features Document](FEATURES.md) - Complete feature list

