# Augmentation Strategies

This directory contains augmentation strategies for Fredholm integral equations. Each strategy is implemented as a separate class inheriting from `BaseAugmentation`.

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

**For 5,000 Sample Dataset (Development/Testing):**
- Multiplier: **1.25-1.43** (recommended: **1.33**)
- Total size: 6,250-7,150 equations
- Composition: 70-80% exact solutions, 20-30% edge cases
- Use case: Development, testing, proof-of-concept
- Example with 1.33x: 6,650 total (5,000 exact + 1,650 edge cases = **75% exact**)

**For 500,000 Full Dataset (Production Training):**
- Multiplier: **1.25-1.43** (recommended: **1.33**)
- Total size: 625,000-715,000 equations
- Composition: 70-80% exact solutions, 20-30% edge cases
- Use case: Production LLM training, research experiments
- Example with 1.33x: 665,000 total (500k exact + 165k edge cases = **75% exact**)

**Multiplier Guide:**
- **1.25** → 80% exact, 20% edge cases (conservative)
- **1.33** → 75% exact, 25% edge cases (recommended)
- **1.43** → 70% exact, 30% edge cases (more edge case exposure)
- **1.50** → 67% exact, 33% edge cases (balanced)
- **2.00** → 50% exact, 50% edge cases (only for specialized research)

### Why NOT 1:1 Balance?

1. **Real-world distribution**: Most Fredholm equations have exact solutions; edge cases are rare exceptions
2. **Task priority**: Primary goal is teaching solution methods, edge case recognition is secondary
3. **Bias prevention**: Equal representation causes models to incorrectly flag solvable equations as edge cases
4. **Learning efficiency**: Solution patterns are complex and need more training examples

### Practical Examples

```bash
# Sample dataset: Balanced for testing (1.33x multiplier = 75% exact)
# Windows PowerShell - use backtick (`) for line continuation
uv run python scripts/prepare_dataset.py `
  --input data/raw/Fredholm_Dataset_Sample.csv `
  --augment --augment-multiplier 1.33 `
  --augment-strategies no_solution approximate_only ill_posed `
  --no-convert
# Output: ~6,650 total (5,000 exact + ~1,650 edge cases = 75% exact)

# Full dataset: Production training (1.33x multiplier = 75% exact)
uv run python scripts/prepare_dataset.py `
  --input data/raw/Fredholm_Dataset.csv `
  --augment --augment-multiplier 1.33 `
  --augment-strategies no_solution approximate_only ill_posed `
  --no-convert
# Output: ~665,000 total (500k exact + ~165k edge cases = 75% exact)

# Conservative balance (1.25x multiplier = 80% exact)
uv run python scripts/prepare_dataset.py `
  --augment --augment-multiplier 1.25 `
  --augment-strategies no_solution approximate_only ill_posed
# Output: 80% exact, 20% edge cases

# More edge case exposure (1.5x multiplier = 67% exact)
uv run python scripts/prepare_dataset.py `
  --augment --augment-multiplier 1.5 `
  --augment-strategies no_solution approximate_only ill_posed
# Output: 67% exact, 33% edge cases

# Linux/macOS - use backslash (\) for line continuation
uv run python scripts/prepare_dataset.py \
  --input data/raw/Fredholm_Dataset_Sample.csv \
  --augment --augment-multiplier 1.33 \
  --augment-strategies no_solution approximate_only ill_posed \
  --no-convert
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

#### 5. No-Solution Cases (`no_solution.py`)
**Category**: The "No Solution" (Singular) Case

**Purpose**: Generate equations where λ is an eigenvalue of the kernel, violating the Fredholm Alternative.

**Logic**:
- For kernel K(x,t) with eigenvalue λ₀, setting λ = λ₀ creates a singular problem
- If f(x) is not orthogonal to the corresponding eigenfunction, no solution exists
- Common eigenvalue cases:
  - K(x,t) = 1: λ = 1/(b-a)
  - K(x,t) = x*t: λ ≈ 3/(b³-a³)
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
  "edge_case": "no_solution"
}
```

**LLM Task**: Recognize singular cases and explain why no solution exists.

---

#### 6. Approximate-Only Cases (`approximate_only.py`)
**Category**: The "Approximate Only" Case

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

# Include edge cases
augmenter = DataAugmenter(
    strategies=["substitute", "no_solution", "approximate_only", "ill_posed"]
)
augmented_data = augmenter.augment(data, multiplier=2)
```

### Command Line

```bash
# Default strategies
uv run python scripts/prepare_dataset.py --augment --augment-multiplier 3

# Specific strategies
uv run python scripts/prepare_dataset.py \
  --augment \
  --augment-strategies substitute scale no_solution

# All strategies including edge cases
uv run python scripts/prepare_dataset.py \
  --augment \
  --augment-strategies substitute scale shift compose no_solution approximate_only ill_posed
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
    "lambda": "1",
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

## Design Principles

1. **Modularity**: Each strategy is independent and self-contained
2. **Consistency**: All strategies follow the same interface
3. **Metadata**: All augmented items include:
   - `augmented: true`
   - `augmentation_type: str`
   - Strategy-specific metadata
4. **Error Handling**: Graceful failures with debug logging
5. **Testability**: Each strategy can be tested independently

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
