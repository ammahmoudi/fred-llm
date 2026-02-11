# Augmentation Strategies

This directory contains augmentation strategies for Fredholm integral equations. Each strategy is implemented as a separate class inheriting from `BaseAugmentation`.

## ⚠️ UNIFIED OUTPUT SCHEMA - ALL ENTRIES HAVE SAME KEYS

**CRITICAL**: ALL dataset entries (both original and augmented) output the EXACT SAME 18 keys. No exceptions. This ensures consistent training data for machine learning.

### Complete Required Schema

Every single dataset entry (original from Fredholm-LLM and all augmented variants) includes these **core fields**:

| Field | Type | Always Present | Description |
|-------|------|----------------|-------------|
| `u` | str | ✅ | Solution function |
| `f` | str | ✅ | Right-hand side function |
| `kernel` | str | ✅ | Kernel function K(x,t) |
| `lambda_val` | str | ✅ | Lambda parameter value (numeric string) |
| `a` | str | ✅ | Lower integration bound |
| `b` | str | ✅ | Upper integration bound |
| `u_type` | ExpressionType | ✅ | Expression type for u (from CSV metadata) |
| `f_type` | ExpressionType | ✅ | Expression type for f (from CSV metadata) |
| `kernel_type` | ExpressionType | ✅ | Expression type for kernel (from CSV metadata) |
| `augmented` | bool | ✅ | `False` for original, `True` for augmented |
| `augmentation_type` | str | ✅ | Strategy name (e.g., "scale", "eigenvalue_cases", "boundary_layer") |
| `augmentation_variant` | str | ✅ | Specific variant (e.g., "scale_2.0x") |
| `has_solution` | bool | ✅ | Solution exists? |
| `solution_type` | str | ✅ | `"exact_symbolic"` \| `"approx_coef"` \| `"discrete_points"` \| `"series"` \| `"family"` \| `"regularized"` \| `"none"` |
| `edge_case` | str \| None | ✅ | Edge case type or `None` for basic |
| `reason` | str | ✅ | Explanation of augmentation/case |
| `recommended_methods` | list[str] | ✅ | Suggested methods (empty list `[]` if none) |
| `numerical_challenge` | str \| None | ✅ | Challenge description or `None` |

### Optional Detailed Metadata (60+ fields)

**By default excluded** for cleaner output. Include with `--include-edge-metadata` flag:

**Singularity details**: `singularity_type`, `singularity_order`, `singularity_strength`  
**Boundary layers**: `layer_location`, `layer_width_estimate`, `gradient_scale`, `minimum_points_in_layer`  
**Oscillations**: `oscillation_type`, `oscillation_frequency`, `angular_frequency`, `num_cycles_in_domain`, `nyquist_samples_required`, `sampling_rate_needed`, `modulation`, `frequencies`, `beat_frequency`, `spectrum_complexity`  
**Kernel details**: `kernel_description`, `equation_type`, `split_point`, `causal_structure`, `volterra_region`, `fredholm_region`, `mathematical_form`, `transition_width`, `mathematical_explanation`, `kernel_split`, `integral_form`, `solution_strategy`, `kernel_structure`  
**Support details**: `support_type`, `support_width`, `zero_fraction`, `matrix_structure`, `bandwidth_parameter`, `rank_deficient_risk`, `memory_efficiency`, `support_region`, `active_fraction`, `physical_interpretation`, `num_support_regions`  
**Numerical methods**: `numerical_method`, `sample_points`, `sample_values`  
**Resonance**: `is_critical`, `near_critical_value`, `distance_to_resonance`, `condition_number_estimate`  
**Series**: `series_type`, `series_terms`, `convergence_estimate`, `truncation_error`  
**Family**: `eigenvalue_approximate`, `eigenfunction`, `solution_multiplicity`, `general_solution`  
**Ill-posed**: `equation_form`, `is_ill_posed`, `requires_regularization`, `regularization_param`  
**Other**: `notes`, `operator_property`, `orthogonality_violated`, `kernel_rank`, `range_basis`, `divergence_type`, `contrast_with`, `physically_invalid`, `problem_structure`, `mathematical_issue`, `warning`

**Rationale**: Most LLM training only needs essential fields. Researchers analyzing edge cases can opt-in for full technical metadata.

### Value Patterns by Strategy Type

**Basic Transformations** (substitute, scale, shift, compose):
- `solution_type`: `"exact"`
- `edge_case`: `null`
- `has_solution`: `true`
- `recommended_methods`: `[]`
- `numerical_challenge`: `null`

**None Solution Cases** (none_solution folder):
- `solution_type`: `"none"`
- `u`: `""` (empty string)
- `has_solution`: `false`
- `edge_case`: `"eigenvalue_cases"` | `"range_violation"` | `"divergent_kernel"` | `"disconnected_support"`

**Approx Coef** (approx_coef folder):
- `solution_type`: `"approx_coef"`
- `u`: Functional form with numerical params (e.g., `"exp(-x/0.01)"`, `"sin(100*pi*x)"`)
- `has_solution`: `true`
- `edge_case`: `"boundary_layer"` | `"oscillatory_solution"` | `"weakly_singular"` | etc.

**Discrete Points** (discrete_points folder):
- `solution_type`: `"discrete_points"`
- `u`: `""` (empty string, uses sample_points arrays)
- `has_solution`: `true`
- `edge_case`: `"complex_kernels"` | `"near_resonance"`

**Series** (series folder):
- `solution_type`: `"series"`
- `u`: Series expansion (e.g., `"f(x) + lambda*int(K) + lambda^2*int(K^2) + ..."`)
- `has_solution`: `true`
- `edge_case`: `"neumann_series"`

**Regularized** (regularized folder):
- `solution_type`: `"regularized"`
- `u`: `""` (empty string, requires Tikhonov/TSVD)
- `has_solution`: `true`
- `edge_case`: `"ill_posed"`

**Family** (family folder):
- `solution_type`: `"family"`
- `u`: Shows solution family (e.g., `"C * sin(pi*x) + u_p"`, `"C"` for simple cases)
- `has_solution`: `true`
- `edge_case`: `"resonance"`

### Example Outputs (All Have 18 Identical Keys)

**Original Dataset Entry:**
```json
{
  "u": "x",
  "f": "x",
  "kernel": "x*t",
  "lambda_val": "1",
  "a": "0",
  "b": "1",
  "u_type": "polynomial",
  "f_type": "polynomial",
  "kernel_type": "polynomial",
  "augmented": false,
  "augmentation_type": "original",
  "augmentation_variant": "fredholm_dataset",
  "has_solution": true,
  "solution_type": "exact",
  "edge_case": null,
  "reason": "Original Fredholm-LLM dataset equation (DOI: 10.5281/zenodo.16784707) - well-posed second kind with exact symbolic solution",
  "recommended_methods": ["symbolic_solution", "Neumann_series", "separable_kernel_method"],
  "numerical_challenge": null
}
```

**Basic Transformation:**
```json
{
  "u": "x",
  "f": "x",
  "kernel": "x*t",
  "lambda_val": "2.0",
  "a": "0",
  "b": "1",
  "u_type": "polynomial",
  "f_type": "polynomial",
  "kernel_type": "polynomial",
  "augmented": true,
  "augmentation_type": "scale",
  "augmentation_variant": "scale_2.0x",
  "has_solution": true,
  "solution_type": "exact",
  "edge_case": null,
  "reason": "Lambda coefficient scaled by factor 2.0",
  "recommended_methods": [],
  "numerical_challenge": null
}
```

**No Solution:**
```json
{
  "u": "",
  "f": "x",
  "kernel": "1",
  "lambda_val": "1.0",
  "lambda_val": "1.0",
  "a": "0",
  "b": "1",
  "augmented": true,
  "augmentation_type": "eigenvalue_cases",
  "augmentation_variant": "constant_kernel_eigenvalue",
  "has_solution": false,
  "solution_type": "none",
  "edge_case": "eigenvalue_cases",
  "reason": "Violates Fredholm Alternative - λ is eigenvalue of constant kernel",
  "recommended_methods": ["Check Fredholm Alternative conditions", "Verify eigenvalue"],
  "numerical_challenge": null
}
```

**Numerical Only:**
```json
{
  "u": "Numerical",
  "f": "1",
  "kernel": "exp(-(x**2 + t**2))",
  "lambda_val": "0.5",
  "lambda_val": "0.5",
  "a": "0",
  "b": "1",
  "augmented": true,
  "augmentation_type": "approximate_only",
  "augmentation_variant": "gaussian_kernel",
  "has_solution": true,
  "solution_type": "numerical",
  "edge_case": "approximate_only",
  "reason": "Gaussian kernel has no symbolic antiderivative",
  "recommended_methods": ["fixed_point_iteration", "quadrature", "Neumann_series"],
  "numerical_challenge": "No symbolic antiderivative - requires numerical integration"
}
```

**Ill-Posed:**
```json
{
  "u": "Requires regularization",
  "f": "x**2",
  "kernel": "x*t",
  "lambda_val": "N/A",
  "lambda_val": "0",
  "a": "0",
  "b": "1",
  "augmented": true,
  "augmentation_type": "ill_posed",
  "augmentation_variant": "simple_first_kind",
  "has_solution": true,
  "solution_type": "regularized",
  "edge_case": "ill_posed",
  "reason": "First kind equation - extremely sensitive to noise in f(x)",
  "recommended_methods": ["Tikhonov", "Truncated SVD", "Landweber iteration"],
  "numerical_challenge": "Solution unstable without regularization"
}
```

## Directory Structure

Organized by **solution type** (folder names match solution_type values):

```
src/data/augmentations/
├── exact_symbolic/          # solution_type: "exact_symbolic"
│   ├── substitute.py         # Variable transformations
│   ├── scale.py              # Lambda scaling
│   ├── shift.py              # Domain shifting
│   └── compose.py            # Kernel composition
├── approx_coef/             # solution_type: "approx_coef"
│   ├── boundary_layer.py     # Sharp gradient solutions (exp(-x/ε))
│   ├── oscillatory_solution.py  # High-frequency oscillations (sin(ωx))
│   ├── weakly_singular.py    # Integrable singularities
│   ├── mixed_type.py         # Volterra + Fredholm mix
│   └── compact_support.py    # Sparse kernel structure
├── discrete_points/         # solution_type: "discrete_points"
│   ├── complex_kernels.py    # No symbolic antiderivative
│   └── near_resonance.py     # Ill-conditioned systems
├── series/                  # solution_type: "series"
│   └── neumann_series.py     # Neumann expansion (N=4 terms)
├── family/                  # solution_type: "family"
│   └── resonance.py          # Eigenvalue resonance (non-unique)
├── regularized/             # solution_type: "regularized"
│   └── ill_posed.py          # Ill-conditioned problems
├── none_solution/           # solution_type: "none"
│   ├── eigenvalue_cases.py   # Eigenvalue violations
│   ├── range_violation.py    # RHS not in operator range
│   ├── divergent_kernel.py   # Non-integrable singularities
│   └── disconnected_support.py # Rank-deficient operators
├── base.py                  # Base class for all strategies
└── README.md
```

**Organization Philosophy**: Grouped by **what type of answer the LLM should give**, not by mathematical taxonomy. This makes training more effective because:
1. ✅ LLM learns to recognize **solution strategy** needed
2. ✅ Consistent with how mathematicians approach problems
3. ✅ Clear mapping from problem characteristics → solution approach

**Total**: 11 edge case strategies × 3 variants each = **33 edge case types**

## Architecture

All augmentation strategies follow a consistent interface:

```python
class MyAugmentation(BaseAugmentation):
    @property
    def strategy_name(self) -> str:
        return "my_strategy"
    
    @property
    def description(self) -> str:
        return "Brief description"
    
    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        # Generate augmented versions
        return results
```

## Dataset Balance Guidelines

When augmenting datasets with edge cases, maintain proper balance for optimal LLM training:

### Recommended Multipliers

**When Using All 11 Edge Case Strategies (33 variants):**
- Multiplier: **1.1-1.2** (recommended: **1.15**)
- Result: 87% exact solutions, 13% edge cases
- Example: 5,000 original → ~5,750 total (5,000 exact + ~750 edge cases)
- Use case: Comprehensive mathematical reasoning training

**When Using Subset (e.g., 3-5 strategies):**
- Multiplier: **1.2-1.25**
- Result: 80-83% exact solutions, 17-20% edge cases
- Example: 5,000 original → ~6,000-6,250 total
- Use case: Balanced production training

**When Using 1-2 Specific Strategies:**
- Multiplier: **1.3-1.5**
- Result: 67-77% exact solutions, 23-33% edge cases
- Example: 5,000 original → ~6,500-7,500 total
- Use case: Targeted training on specific edge cases

### Why NOT 1:1 Balance?

1. **Real-world distribution**: Most Fredholm equations have exact solutions; edge cases are rare exceptions
2. **Task priority**: Primary goal is teaching solution methods, edge case recognition is secondary
3. **Bias prevention**: Equal representation causes models to incorrectly flag solvable equations as edge cases
4. **Learning efficiency**: Solution patterns are complex and need more training examples

### Practical Examples

```bash
# All 6 edge case folders (runs all 14 strategies): Conservative multiplier (1.15x = 87% exact)
uv run python scripts/prepare_dataset.py `
  --input data/raw/Fredholm_Dataset_Sample.csv `
  --augment --augment-multiplier 1.15 `
  --augment-strategies none approx_coef discrete_points series family regularized `
  --no-convert
# Output: ~5,750 total (5,000 exact + ~750 edge cases = 87% exact)

# Core edge cases only (none + discrete_points + regularized): Higher multiplier (1.33x = 75% exact)
uv run python scripts/prepare_dataset.py `
  --input data/raw/Fredholm_Dataset_Sample.csv `
  --augment --augment-multiplier 1.33 `
  --augment-strategies none discrete_points regularized `
  --no-convert

# Specific folder only: Target one solution category
uv run python scripts/prepare_dataset.py `
  --augment --augment-multiplier 1.25 `
  --augment-strategies numerical_only  # Runs all 6 strategies in numerical_only folder
# Output: 80% exact, 20% numerical edge cases
```

## Folder-Based Strategy System

**Strategies are organized into folders by solution type.** When you specify a folder name, ALL strategies in that folder are executed:

```bash
# Using "none" runs 4 strategies:
--augment-strategies none
# → eigenvalue_cases.py (3 variants)
# → range_violation.py (3 variants)  
# → divergent_kernel.py (3 variants)
# → disconnected_support.py (3 variants)
# Total: 12 edge cases per equation

# Using "approx_coef" runs 5 strategies:
--augment-strategies approx_coef
# → boundary_layer.py, oscillatory_solution.py, weakly_singular.py,
#   mixed_type.py, compact_support.py
# Total: 15 edge cases per equation
```

**Available Folder Strategies (18 strategies, 42 variants total):**
- `none` → 4 strategies → 12 variants per equation
- `approx_coef` → 5 strategies → 15 variants per equation
- `discrete_points` → 2 strategies → 6 variants per equation
- `series` → 1 strategy → 3 variants per equation
- `family` → 1 strategy → 3 variants per equation
- `regularized` → 1 strategy → 3 variants per equation
```

## Available Strategies

### Basic Transformations

> ⚠️ **Note**: Basic transformation strategies (substitute, scale, shift, compose) are implemented but **not currently tested or validated**. They are maintained for future use but not recommended for production datasets until comprehensive testing is completed. Use edge case strategies instead for production training.

#### 1. Variable Substitution (`substitute.py`)
**Purpose**: Transform variables with expressions to test function composition understanding.

**Status**: ⚠️ Not tested/validated

**Transformations**:
- `x → 2*x` (double_x): Tests scaling behavior
- `x → x²` (square_x): Tests quadratic transformations
- `x → x + 1` (shift_x): Tests translation invariance

**Example**:
```python
# Original: u(x) = x, f(x) = x, K(x,t) = x*t
# After double_x: u(2x) = 2x, f(2x) = 2x, K(2x,t) = 2x*t
```

**Use Case**: Teaches LLMs about variable transformations and how they propagate through equations.

---

#### 2. Coefficient Scaling (`scale.py`)
**Purpose**: Scale the λ parameter to test sensitivity to magnitude.

**Status**: ⚠️ Not tested/validated

**Scale Factors**: 0.5, 2.0, 0.1, 10.0

**Example**:
```python
# Original: λ = 1.0
# After scale: λ ∈ {0.5, 2.0, 0.1, 10.0}
```

**Use Case**: Helps LLMs understand how λ affects solution stability and convergence.

---

#### 3. Domain Shifting (`shift.py`)
**Purpose**: Shift integration bounds to test domain understanding.

**Status**: ⚠️ Not tested/validated

**Transformations**:
- `[a, b] → [a-1, b-1]` (shift_left): Move domain left
- `[a, b] → [a+1, b+1]` (shift_right): Move domain right
- `[a, b] → [a, b+1]` (extend_right): Extend domain

**Example**:
```python
# Original: [0, 1]
# After shift_left: [-1, 0]
# After shift_right: [1, 2]
# After extend_right: [0, 2]
```

**Use Case**: Tests understanding of how integration bounds affect solutions.

---

#### 4. Kernel Composition (`compose.py`)
**Purpose**: Create more complex kernels through composition.

**Status**: ⚠️ Not tested/validated

**Compositions**:
- `K(x,t) → K(x,t) + x` (add_x): Add x-dependence
- `K(x,t) → K(x,t) + t` (add_t): Add t-dependence
- `K(x,t) → K(x,t) * x` (mul_x): Multiply by x

**Example**:
```python
# Original: K(x,t) = x*t
# After add_x: K(x,t) = x*t + x
# After mul_x: K(x,t) = x²*t
```

**Use Case**: Teaches LLMs about kernel structure and separability.

---

### Edge Cases (FIE-Edge-Cases)

> ✅ **Production Ready**: All edge case strategies are thoroughly tested and validated for production use.

These augmentations create realistic problem scenarios where standard symbolic methods fail.

#### 5. Eigenvalue Cases (`eigenvalue_cases.py`)
**Category**: The "No Solution" (Singular) Case

**Purpose**: Generate equations where λ is an eigenvalue of the kernel, violating the Fredholm Alternative.

**Logic**:
- For kernel K(x,t) with eigenvalue λ₀, setting λ = λ₀ creates a singular problem
- If f(x) is not orthogonal to the corresponding eigenfunction, no solution exists
- Common eigenvalue cases:
  - K(x,t) = 1: λ = 1/(b-a)
  - K(x,t) = x*t: λ ≈ 3/(b³-a³)

---

### Advanced Edge Cases

> 🆕 **Newly Added**: Advanced mathematical edge cases for specialized training.

#### 8. Weakly Singular Kernels (`weakly_singular.py`)
**Category**: Integrable Singularities

**Purpose**: Generate kernels with integrable but problematic singularities.

**Mathematical Forms**:
- Logarithmic: `K(x,t) = log|x-t|`
- Power law: `K(x,t) = |x-t|^(-1/2)`
- Mixed: `K(x,t) = (x+t)/sqrt|x-t|`

**The Challenge**:
- Kernel explodes at x=t but integral converges
- Standard quadrature fails near singularity
- Requires product integration or singularity subtraction

**Metadata**:
```python
{
  "edge_case": "weakly_singular",
  "singularity_type": "logarithmic|power_law|algebraic_mixed",
  "singularity_order": 0.0 or 0.5,
  "recommended_methods": ["product_integration", "singularity_subtraction"]
}
```

**Usage**:
```bash
--augment-strategies weakly_singular
```

---

#### 9. Boundary Layer Solutions (`boundary_layer.py`)
**Category**: Sharp Gradient Solutions

**Purpose**: Create solutions with rapid variation near boundaries.

**Mathematical Forms**:
- Left layer: `u(x) = exp(-(x-a)/ε)` where ε = 0.01
- Right layer: `u(x) = exp((x-b)/ε)`
- Double layer: `u(x) = tanh((x-a)/ε) + tanh((b-x)/ε)`

**The Challenge**:
- Solution changes rapidly in thin boundary layer of width ε
- Uniform mesh misses rapid variation
- Requires adaptive mesh refinement or graded mesh

**Metadata**:
```python
{
  "edge_case": "boundary_layer",
  "layer_location": "left|right|both",
  "layer_width_estimate": 0.01,
  "gradient_scale": 100.0,  # 1/ε
  "recommended_methods": ["adaptive_mesh_refinement", "exponential_grading"]
}
```

**Usage**:
```bash
--augment-strategies boundary_layer
```

---

#### 10. Resonance/Critical Points (`resonance.py`)
**Category**: Non-Unique Solutions

**Purpose**: Set λ at eigenvalue of operator (bifurcation/resonance point).

**Mathematical Logic**:
- When λ = 1/μₙ (eigenvalue), operator (I - λK) is singular
- Homogeneous equation has non-trivial solutions
- Solutions form a family: u = u_particular + C*φ(x)

**Common Examples**:
- Separable: `K(x,t) = sin(πx)sin(πt)` → λ_critical = 2
- Constant: `K(x,t) = 1` → λ_critical = 1

**The Challenge**:
- Solution is not unique (infinite family parameterized by constant)
- LLM must recognize resonance condition
- Must provide general solution form

**Metadata**:
```python
{
  "edge_case": "resonance",
  "is_critical": True,
  "eigenvalue_approximate": 0.5,
  "eigenfunction": "sin(pi*x)",
  "solution_multiplicity": "infinite",
  "general_solution": "C * sin(pi*x) for any constant C"
}
```

**Usage**:
```bash
--augment-strategies resonance
```
  - K(x,t) = cos(x-t): λ = 1

**Generated Cases**:
1. **Constant kernel**: K=1, λ=1/(b-a), f=x
2. **Separable kernel**: K=x*t, λ=3/(b³-a³), f=x²
3. **Symmetric kernel**: K=cos(x-t), λ=1, f=sin(x)

**Labels**:
```json
{
  "has_solution": false,
  "solution_type": "none",
  "reason": "Violates Fredholm Alternative - λ is eigenvalue",
  "edge_case": "eigenvalue_cases"
}
```

**LLM Task**: Recognize singular cases and explain why no solution exists.

---

#### 6. Complex Kernels (`complex_kernels.py`)
**Category**: The "Discrete Points Only" Case

**Purpose**: Generate equations with no closed-form symbolic solution, requiring numerical methods.

**Logic**:
- Use kernels without symbolic antiderivatives
- Force numerical integration (quadrature, Neumann series)
- Provide sample points and numerical values for training

**Generated Cases**:
1. **Gaussian kernel**: K = exp(-(x²+t²)), f = 1
2. **Exponential decay**: K = exp(-|x-t|), f = x
3. **Sinc-like kernel**: K = sin(x*t)/(x*t), f = cos(x)

**Labels**:
```json
{
  "solution_type": "numerical",
  "numerical_method": "quadrature",
  "sample_points": [0.0, 0.1, ..., 1.0],
  "sample_values": [u₀, u₁, ..., uₙ],
  "edge_case": "approximate_only",
  "reason": "Gaussian kernel has no symbolic antiderivative"
}
```

**LLM Task**: 
1. Recognize no symbolic solution exists
2. Apply numerical methods
3. Return approximation with sample points

---

#### 7. Ill-Posed Cases (`ill_posed.py`)
**Category**: The "Ill-Posed" Case (Fredholm 1st Kind)

**Purpose**: Generate Fredholm equations of the first kind, which are extremely ill-posed.

**Logic**:
- Transform from 2nd kind: `u(x) - λ∫K(x,t)u(t)dt = f(x)`
- To 1st kind: `∫K(x,t)u(t)dt = f(x)` (λ → ∞, no u(x) term)
- These are ill-conditioned: tiny changes in f(x) cause huge changes in u(x)
- Require regularization (Tikhonov, TSVD, Landweber)

**Generated Cases**:
1. **Simple first kind**: K = x*t, f = x²
2. **Exponential kernel**: K = exp(x*t), f = exp(x)
3. **Oscillatory kernel**: K = sin(x-t), f = sin(2x)

**Labels**:
```json
{
  "equation_type": "fredholm_first_kind",
  "equation_form": "∫K(x,t)u(t)dt = f(x)",
  "is_ill_posed": true,
  "requires_regularization": true,
  "recommended_methods": ["Tikhonov", "TSVD", "Landweber"],
  "solution_type": "regularized",
  "edge_case": "ill_posed",
  "regularization_param": 0.01
}
```

**LLM Task**:
1. Identify equation as Fredholm 1st kind
2. Recognize it's ill-posed
3. Recommend regularization techniques
4. Note solution instability without regularization

---

## Usage

### Basic Usage

```python
from src.data.augmentation import DataAugmenter

# Use default strategies (substitute, scale, shift)
augmenter = DataAugmenter()
augmented_data = augmenter.augment(data, multiplier=2)

# Use specific strategies
augmenter = DataAugmenter(strategies=["substitute", "scale"])
augmented_data = augmenter.augment(data, multiplier=3)

# Include edge cases (use folder names)
augmenter = DataAugmenter(
    strategies=["substitute", "none", "discrete_points", "regularized"]
)
augmented_data = augmenter.augment(data, multiplier=2)
```

### Command Line

```bash
# Edge cases only (recommended for production)
uv run python scripts/prepare_dataset.py \
  --augment \
  --augment-multiplier 1.33 \
  --augment-strategies none discrete_points regularized

# Include all edge case types
uv run python scripts/prepare_dataset.py \
  --augment \
  --augment-multiplier 1.5 \
  --augment-strategies \
    none \
    approx_coef \
    discrete_points \
    series \
    family \
    regularized
    approximate_only \
    ill_posed \
    weakly_singular \
    boundary_layer \
    resonance

# Test individual strategy
uv run python scripts/prepare_dataset.py \
  --augment \
  --augment-multiplier 1.1 \
  --augment-strategies weakly_singular

# All strategies (basic transformations + all edge case folders)
uv run python scripts/prepare_dataset.py \
  --augment \
  --augment-strategies substitute scale shift compose \
    none approx_coef discrete_points series family regularized
```

### Direct Usage

```python
from src.data.augmentations import NoSolutionAugmentation

# Create augmenter
augmenter = NoSolutionAugmentation()

# Apply to single equation
original = {
    "u": "x",
    "f": "x",
    "kernel": "x*t",
    "lambda_val": "1",
    "a": "0",
    "b": "1"
}

edge_cases = augmenter.augment(original)
# Returns 3 no-solution variants
```

---

## Adding New Strategies

To add a new augmentation strategy:

1. **Create a new file** in `src/data/augmentations/`:

```python
# src/data/augmentations/my_strategy.py
from typing import Any
from src.data.augmentations.base import BaseAugmentation

class MyStrategyAugmentation(BaseAugmentation):
    @property
    def strategy_name(self) -> str:
        return "my_strategy"
    
    @property
    def description(self) -> str:
        return "Description of what this does"
    
    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        results = []
        # Generate augmented versions
        new_item = item.copy()
        new_item["augmented"] = True
        new_item["augmentation_type"] = "my_strategy"
        results.append(new_item)
        return results
```

2. **Add to `__init__.py`**:

```python
from src.data.augmentations.my_strategy import MyStrategyAugmentation

__all__ = [..., "MyStrategyAugmentation"]
```

3. **Register in `augmentation.py`**:

```python
elif strategy == "my_strategy":
    augmenter = MyStrategyAugmentation()
    return augmenter.augment(item)
```

4. **Update `prepare_dataset.py`** choices:

```python
choices=["substitute", ..., "my_strategy"]
```

---

## Output Format

### Field Consistency

All augmented entries produce **clean, consistent output** with no debug or internal fields:

✅ **Included in Output**:
- Core equation fields: `u`, `f`, `kernel`, `lambda`, `lambda_val`, `a`, `b`
- Solution metadata: `has_solution`, `solution_type`, `edge_case`
- Strategy metadata: `augmented`, `augmentation_type`, `augmentation_variant`
- Strategy-specific fields: `singularity_type`, `layer_location`, `nyquist_samples_required`, etc.
- Source tracking: `source_file`, `source_path`

❌ **Excluded from Output** (internal/debug only):
- `sample_points` - Not needed in training data
- Internal mesh generation results
- Intermediate computation values

**Example Output**:
```json
{
  "u": "exp(-(x - 0.0)/0.01)",
  "f": "exp(-(x - 0.0)/0.01)",
  "kernel": "x * t",
  "lambda_val": "0.2",
  "a": "0.0",
  "b": "1.0",
  "has_solution": true,
  "solution_type": "numerical",
  "edge_case": "boundary_layer",
  "layer_location": "left",
  "layer_width_estimate": 0.01,
  "gradient_scale": 100.0,
  "recommended_methods": ["adaptive_mesh_refinement"],
  "numerical_challenge": "Rapid variation in layer of width 0.01",
  "minimum_points_in_layer": 10,
  "augmented": true,
  "augmentation_type": "boundary_layer",
  "augmentation_variant": "left_exponential_layer",
  "source_file": "Fredholm_Dataset_Sample.csv"
}
```

---

## Design Principles

1. **Modularity**: Each strategy is independent and self-contained
2. **Consistency**: All strategies follow the same interface
3. **Metadata**: All augmented items include:
   - `augmented: true`
   - `augmentation_type: str`
   - Strategy-specific metadata
4. **Clean Output**: Only training-relevant fields in final data
5. **Error Handling**: Graceful failures with debug logging
6. **Testability**: Each strategy can be tested independently

---

## Testing Edge Cases

The edge case augmentations are designed to teach LLMs about real-world complexities:

### Training Objective
- **No-solution cases**: Teach to recognize singular problems
- **Approximate-only cases**: Teach when to use numerical methods
- **Ill-posed cases**: Teach to identify instability and recommend regularization

### Expected LLM Behavior

For **no-solution** cases:
```
Input: u(x) - ∫₀¹ u(t)dt = x, λ=1
Output: "No solution exists. This violates the Fredholm Alternative 
         because λ=1 is an eigenvalue of the constant kernel K=1."
```

For **approximate-only** cases:
```
Input: u(x) - 0.5∫₀¹ exp(-(x²+t²))u(t)dt = 1
Output: "Symbolic solution not available. Using numerical methods:
         u(0.0) ≈ 1.234, u(0.5) ≈ 1.456, u(1.0) ≈ 1.678"
```

For **ill-posed** cases:
```
Input: ∫₀¹ (x*t)u(t)dt = x²
Output: "This is a Fredholm equation of the first kind - ill-posed problem.
         Requires regularization (Tikhonov, TSVD). Solution unstable without it."
```

---

## References

- Fredholm Alternative: [Wikipedia](https://en.wikipedia.org/wiki/Fredholm_alternative)
- Ill-posed Problems: Hadamard's definition of well-posed problems
- Regularization Methods: Tikhonov, Landweber, Truncated SVD
- FIE-500k Dataset: [Zenodo DOI 10.5281/zenodo.16784707](https://doi.org/10.5281/zenodo.16784707)
