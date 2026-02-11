# Evaluation Strategy Comprehensive Review

**Date:** February 11, 2026  
**Status:** Analysis & Recommendations  
**Last Updated:** February 11, 2026

---

## 🎯 Implementation Status Update (February 11, 2026)

### ✅ Phase 1: Task 1.1 - COMPLETED

**Evaluation Points Generation**: Successfully implemented in `src/data/augmentations/base.py`

- ✅ Added `_generate_evaluation_points()` method to BaseAugmentation class
- ✅ **Overflow-safe numeric evaluation** with `np.errstate()` context manager
- ✅ **Non-finite value filtering**: Automatically drops inf/nan from exp/cosh overflows
- ✅ Critical point inclusion: boundaries, midpoint, near-boundary points (50 total)
- ✅ Used by all `has_solution=True` augmentation strategies (inherited from base class)
- ✅ Tested and working: 22 passing tests for overflow filtering and evaluation points

**Key Features Implemented:**
```python
# Suppresses RuntimeWarning: overflow in exp/cosh
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    u_values = np.array([u_lambda(float(xi)) for xi in x_values], dtype=float)

# Filters non-finite values (inf/nan)
finite_mask = np.isfinite(u_values)
x_values = x_values[finite_mask]
u_values = u_values[finite_mask]
```

### ✅ Augmentation Expression Compatibility - COMPLETED

**SymPy-Parseable Kernel Definitions**: All augmentation strategies now use valid Piecewise expressions

- ✅ Replaced placeholder strings (`"Piecewise: nonzero in..."`) with parseable syntax
- ✅ Converted ternary operators (`"t if t <= x else x"`) to Piecewise notation
- ✅ Fixed 5 augmentation files:
  - `disconnected_support.py` (2 cases)
  - `mixed_type.py` (1 case)
  - `compact_support.py` (2 cases: banded + localized)
  - `neumann_series.py` (1 case)
- ✅ All kernels now use logical operators (`&`, `|`, `~`) instead of Python keywords
- ✅ LaTeX conversion works without relational warnings from augmentation layer

**Example Fix:**
```python
# Before: "t if t <= x else x"
# After: "Piecewise((t, t <= x), (x, True))"

# Before: "sin(x)*cos(t) if (c <= x <= d and c <= t <= d) else 0"
# After: "Piecewise((sin(x)*cos(t), (x>=c) & (x<=d) & (t>=c) & (t<=d)), (0, True))"
```

### 📋 Remaining Tasks

- ⏳ **Task 1.2**: Standardize discrete_points and series formats (not started)
- ⏳ **Task 1.3**: Add discrete_points parser (not started)
- ⏳ **Phase 2**: Implement specialized evaluators (not started)
- ⏳ **Phase 3**: Enhanced reporting and metrics (not started)

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Critical Gaps](#critical-gaps)
3. [Recommended Evaluation Strategy](#recommended-evaluation-strategy)
4. [High-Priority Improvements](#high-priority-improvements)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Prompt Modifications](#prompt-modifications)
7. [Expected Outcomes](#expected-outcomes)

---

## Current State Analysis

### ✅ What Works Well

#### Symbolic Evaluation
- **Math-Verify integration** for robust LaTeX/infix parsing
- **Multiple simplification strategies**: `simplify()`, `expand()`, `trigsimp()`
- Well-tested for exact_symbolic solutions
- Fast-path boolean comparison before heavy simplification

#### Numeric Evaluation
- Point-based sampling (default: 100 points)
- Computes **RMSE**, **MAE**, **max_error**
- Works reliably for continuous symbolic functions
- Domain-aware evaluation

#### Edge Case Metrics
- **`has_solution` accuracy**: Binary classification (yes/no)
- **`solution_type` accuracy**: 7-class classification
- Per-type breakdown in metrics output
- Tracks parse errors and API failures

#### Postprocessing Pipeline
- Multi-strategy extraction (Math-Verify primary, regex fallback)
- Confidence scoring (0.8 for MV, 0.7 for SymPy, 0.3 for errors)
- Handles LaTeX, infix, and RPN formats
- Structured output format validation

---

### ❌ Critical Gaps

#### 1. Solution Type Coverage

| Type | Current Evaluation | Issue |
|------|-------------------|-------|
| `exact_symbolic` | ✅ Full support | None |
| `approx_coef` | ⚠️ Full expression only | **No per-coefficient comparison** |
| `discrete_points` | ❌ No specialized method | **No point-list parsing/comparison** |
| `series` | ⚠️ Treated as symbolic | **No term-by-term comparison** |
| `family` | ⚠️ Structural match only | **No coefficient accuracy check** |
| `regularized` | ✅ Type classification only | None (no solution to evaluate) |
| `none` | ✅ Binary check | None |

**Key Problems:**

- **`series`**: No standardized output format for LLMs
  - Ground truth: Series expansion with 4-6 terms
  - LLM output: Varies wildly (symbolic, text description, truncated)
  - Evaluation: Currently uses symbolic equivalence (fails for truncated series)

- **`discrete_points`**: No structured output format
  - Ground truth: Function values at specific points
  - LLM output: No standard format specified in prompts
  - Evaluation: No parser exists

- **`approx_coef`**: No coefficient extraction
  - Ground truth: `-1447.128*x**2 + 0.567*cosh(x)`
  - Evaluation: Only checks full expression symbolic match (fails often)
  - Missing: Individual coefficient comparison with tolerance

- **`family`**: Insufficient validation
  - Current: Only checks structural form (ratio test)
  - Missing: Verify correct number of arbitrary parameters
  - Missing: Check parameter naming (c_1, c_2 vs random symbols)

#### 2. Format Limitations

**RPN Format:**
- Math-Verify doesn't support RPN (LaTeX/infix only)
- Requires manual conversion via `rpn_to_sympy()`
- Adds complexity and potential errors

**Series Notation:**
- No standardized LLM output format
- Prompts don't specify how to express series
- Ground truth varies (symbolic sum vs coefficient list)

**Discrete Points:**
- No structured format in prompts
- No example in documentation
- Parsing not implemented

#### 3. Ground Truth Issues

**No Pre-computed Evaluation Points:**
```python
# Current approach (in evaluate.py):
def numeric_compare(solution, ground_truth, domain=(0,1), n_points=100):
    test_points = np.linspace(a, b, n_points)  # ⚠️ Generated each time
    # Evaluate at random points...
```

**Problems:**
- Inconsistent metrics across evaluation runs
- Random point generation → different RMSE each time
- No control over challenging points (boundaries, discontinuities)
- Domain information sometimes missing from predictions

**Should be:**
```python
# Augmented dataset should include:
{
    "u": "x**2 + sin(x)",
    "evaluation_points": {
        "x_values": [0.0, 0.1, 0.2, ..., 1.0],  # Fixed 20-50 points
        "u_values": [0.0, 0.109, 0.241, ...]    # Pre-computed ground truth
    }
}
```

#### 4. Evaluation Reliability Issues

**Family Comparison Weaknesses:**
```python
def family_compare(solution, ground_truth):
    # Only checks structural ratio, doesn't validate:
    # - Correct number of parameters
    # - Parameter naming conventions (c_1, c_2)
    # - Non-trivial cases (e.g., c_1=0 edge case)
```

**Symbolic Comparison Failures:**
- Complex expressions with special functions fail
- Timeout on heavy simplification
- No handling for near-equivalent expressions (e.g., `1e-10` vs `0`)

**Numeric Comparison Issues:**
- No per-type tolerance adjustment
- Fixed tolerance doesn't account for solution magnitude
- Fails on expressions with free symbols (family, series)

---

## Recommended Evaluation Strategy

### Per-Type Evaluation Matrix

| Solution Type | Primary Method | Fallback Method | LLM Output Format | Pre-compute in Dataset |
|---------------|----------------|-----------------|-------------------|------------------------|
| **`exact_symbolic`** | ✅ Symbolic equivalence | ✅ Numeric RMSE | Natural math expression | ❌ None needed |
| **`approx_coef`** | 🆕 **Coefficient extraction + tolerance** | ✅ Numeric RMSE | **📝 Standard: numeric values only** | ✅ **Store coefficient dict** |
| **`discrete_points`** | 🆕 **Point-wise comparison** | ❌ N/A | **📝 Structured: `[(x1,y1), ...]`** | ✅ **Store evaluation points** |
| **`series`** | 🆕 **Term-by-term comparison** | ✅ Numeric RMSE (truncated) | **📝 Standard: coefficient list** | ✅ **Store series coefficients** |
| **`family`** | ✅ Structural match + 🆕 parameter check | ✅ Numeric (substitute c=1) | **📝 Use `c_1, c_2, ...`** | ❌ Current OK |
| **`regularized`** | ✅ Type classification only | ❌ N/A | Text description | ❌ None needed |
| **`none`** | ✅ `has_solution==False` check | ❌ N/A | Text: "No solution" | ❌ None needed |

**Legend:**
- ✅ = Currently implemented
- 🆕 = New implementation needed
- ⚠️ = Needs improvement
- ❌ = Not applicable
- 📝 = Requires prompt/format update

---

## High-Priority Improvements

### 1. Add Evaluation Points to Dataset 🔥 **CRITICAL**

**Problem:** Inconsistent numeric evaluation across runs due to random point generation

**Solution:** Pre-compute and store evaluation points in augmented data

#### Implementation

**✅ IMPLEMENTED (February 11, 2026)** - `src/data/augmentations/base.py`

**Modify augmentation base classes:**

```python
class BaseAugmentation:
    def _generate_evaluation_points(self, u_expr, a, b, n_points=50):
        """Generate fixed evaluation points for consistent metrics."""
        x = sp.Symbol('x')
        u_func = sp.sympify(u_expr)
        u_lambda = sp.lambdify(x, u_func, modules=['numpy'])
        
        # Generate uniform grid
        x_uniform = np.linspace(a, b, n_points)
        
        # Add critical points: boundaries, midpoint, near-boundary points
        critical_points = [
            a,  # Left boundary
            b,  # Right boundary
            (a + b) / 2,  # Midpoint
            a + 0.1 * (b - a),  # Near left boundary
            b - 0.1 * (b - a),  # Near right boundary
        ]
        
        # Combine and remove duplicates
        x_values = np.concatenate([x_uniform, critical_points])
        x_values = np.sort(np.unique(x_values))
        
        # ✅ NEW: Overflow-safe evaluation with non-finite filtering
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            u_values = np.array(
                [u_lambda(float(xi)) for xi in x_values],
                dtype=float,
            )
        
        # Filter non-finite values (inf/nan from overflow)
        finite_mask = np.isfinite(u_values)
        if not np.any(finite_mask):
            raise ValueError("All evaluation points produced non-finite values")
        
        x_values = x_values[finite_mask]
        u_values = u_values[finite_mask]
        
        return {
            "x_values": x_values.tolist(),
            "u_values": u_values.tolist(),
            "n_points": len(x_values)
        }
```

**Status**: ✅ **Implemented and tested** - Available in all augmentation strategies via inheritance

**Update evaluate.py** to use stored points:

```python
def numeric_compare_fixed_points(solution, ground_truth_points, tolerance=1e-6):
    """Use pre-computed evaluation points for consistent metrics."""
    x_values = ground_truth_points["x_values"]
    u_true = np.array(ground_truth_points["u_values"])
    
    # Evaluate solution at same points
    x = sp.Symbol('x')
    sol_lambda = sp.lambdify(x, solution, modules=['numpy'])
    u_pred = np.array([sol_lambda(xi) for xi in x_values])
    
    # Compute errors
    errors = np.abs(u_pred - u_true)
    return {
        "max_error": float(np.max(errors)),
        "mean_error": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "match": np.max(errors) < tolerance
    }
```

**Benefits:**
- ✅ Consistent RMSE/MAE across all evaluation runs
- ✅ Faster evaluation (no re-computation)
- ✅ Can include challenging points (boundaries, discontinuities, inflection points)
- ✅ Reproducible research results

**Scope:**
- ✅ Update all 14 augmentation strategies in `src/data/augmentations/` (inherited via BaseAugmentation)
- ⏳ Modify CSV/JSONL writers to include evaluation_points field (not yet implemented)
- ⏳ Update evaluate.py to prioritize stored points over random generation (not yet implemented)

**Additional Improvements (February 11, 2026):**
- ✅ **Kernel Expression Compatibility**: All augmentation kernels now use parseable SymPy Piecewise notation
  - Fixed placeholder strings in `disconnected_support.py` (2 locations)
  - Converted ternary operators to Piecewise in `mixed_type.py`, `compact_support.py` (2 locations)
  - Replaced series placeholders in `neumann_series.py` with Integral notation
  - All kernels use logical operators (`&`, `|`) instead of Python keywords (`and`, `or`)
- ✅ **Windows YAML Compatibility**: Replaced Unicode characters (✅, ⚠️, →) with ASCII in all config files

---

### 2. Standardize `discrete_points` Format 🔥 **CRITICAL**

**Current Problem:** No structured output format specified in prompts or implemented in parser

#### LLM Prompt Addition

**Add to all 4 prompt styles** (`src/prompts/styles/*.py`):

```python
# Modify SOLUTION_TYPE descriptions:
- discrete_points: Solution only at discrete points
+ discrete_points: Point values only. Format: [(x1, y1), (x2, y2), ...]

# Example in prompt:
"""
For discrete_points:
  SOLUTION: [(0.0, 1.234), (0.25, 2.456), (0.5, 3.789), (0.75, 1.123), (1.0, 0.567)]
"""
```

#### Parser Implementation

**Add to `postprocess.py`**:

```python
def extract_discrete_points(response: str) -> list[tuple[float, float]]:
    """
    Extract discrete point list from LLM response.
    
    Format: [(0.0, 1.234), (0.25, 2.456), ...]
    """
    pattern = r'\[\s*\(\s*[\d.+-]+\s*,\s*[\d.+-]+\s*\)(?:\s*,\s*\(\s*[\d.+-]+\s*,\s*[\d.+-]+\s*\))*\s*\]'
    match = re.search(pattern, response)
    
    if match:
        points_str = match.group(0)
        # Parse individual tuples
        tuple_pattern = r'\(\s*([\d.+-]+)\s*,\s*([\d.+-]+)\s*\)'
        points = []
        for match in re.finditer(tuple_pattern, points_str):
            x, y = float(match.group(1)), float(match.group(2))
            points.append((x, y))
        return points
    
    return []
```

#### Evaluation Implementation

**Add to `evaluate.py`**:

```python
def evaluate_discrete_points(
    pred_points: list[tuple[float, float]], 
    gt_points: list[tuple[float, float]], 
    x_tolerance: float = 1e-3,
    y_tolerance: float = 1e-3
) -> dict[str, Any]:
    """
    Compare discrete point predictions.
    
    Matches x-coordinates within tolerance, compares y-values.
    """
    matched = 0
    total_error = 0.0
    errors = []
    
    for x_pred, y_pred in pred_points:
        # Find closest x in ground truth
        closest_gt = min(gt_points, key=lambda p: abs(p[0] - x_pred))
        x_gt, y_gt = closest_gt
        
        # Check if x-coordinates match
        if abs(x_pred - x_gt) < x_tolerance:
            error = abs(y_pred - y_gt)
            errors.append(error)
            if error < y_tolerance:
                matched += 1
    
    return {
        "matched_points": matched,
        "total_points": len(pred_points),
        "accuracy": matched / len(pred_points) if pred_points else 0.0,
        "mean_error": np.mean(errors) if errors else float('inf'),
        "max_error": np.max(errors) if errors else float('inf')
    }
```

---

### 3. Standardize `series` Format 🔥 **CRITICAL**

**Current Problem:** No consistent series representation, varies from symbolic to text

#### LLM Prompt Addition (Minimal)

**Add to prompt styles:**

```python
- series: Infinite series solution (e.g., u(x) = Σ aₙxⁿ)
+ series: Series expansion. Format: f(x) + λK·f + λ²K²·f + ... (4-6 terms)
```

**Alternative (if LLM struggles):**

```python
"""
For series solutions, either:
  (1) Express as sum: f(x) + λ∫K·f + λ²∫K²·f + ...
  (2) Provide coefficients: SERIES_COEFFICIENTS: [a_0, a_1, a_2, a_3]
"""
```

#### Parser Implementation

**Add to `postprocess.py`**:

```python
def extract_series_coefficients(response: str) -> Optional[list[float]]:
    """Extract series coefficients from marked section."""
    pattern = r'SERIES_COEFFICIENTS:\s*\[([\d.,\s+-eE]+)\]'
    match = re.search(pattern, response)
    
    if match:
        coef_str = match.group(1)
        coefficients = [float(c.strip()) for c in coef_str.split(',')]
        return coefficients
    
    return None
```

#### Evaluation Implementation

**Add to `evaluate.py`**:

```python
def evaluate_series(
    pred_coeffs: list[float],
    gt_coeffs: list[float],
    tolerance: float = 1e-3
) -> dict[str, Any]:
    """
    Compare series solutions term-by-term.
    
    Truncates to shortest length for comparison.
    """
    n_terms = min(len(pred_coeffs), len(gt_coeffs))
    
    # Per-term errors
    errors = [abs(pred_coeffs[i] - gt_coeffs[i]) for i in range(n_terms)]
    
    # Check if all terms within tolerance
    all_match = all(e < tolerance for e in errors)
    
    return {
        "terms_compared": n_terms,
        "pred_length": len(pred_coeffs),
        "gt_length": len(gt_coeffs),
        "mean_term_error": np.mean(errors),
        "max_term_error": np.max(errors),
        "all_terms_match": all_match,
        "per_term_errors": errors
    }
```

**Dataset Enhancement:**

Store series coefficients in ground truth:

```python
# For series-type equations in augmentation:
{
    "solution_type": "series",
    "u": "f(x) + λ*∫K·f + λ²*∫K²·f + λ³*∫K³·f",
    "series_coefficients": [1.0, 0.5, 0.25, 0.125],  # Neumann series coefficients
    "truncation_order": 4
}
```

---

### 4. Improve `approx_coef` Evaluation 📊

**Current Problem:** Only checks full expression equivalence, ignores individual coefficients

#### Dataset Enhancement

**Store coefficient dictionaries:**

```python
# In augmentation strategies (boundary_layer, oscillatory, etc.):
{
    "solution_type": "approx_coef",
    "u": "-1447.128*x**2 + 0.567*cosh(x)",
    "coefficients": {
        "x**2": -1447.128,
        "cosh(x)": 0.567,
        "constant": 0.0
    },
    "basis_functions": ["x**2", "cosh(x)"]
}
```

#### Parser Implementation

**Add to `postprocess.py`**:

```python
def extract_coefficients(expr: sp.Expr, basis_functions: list[str]) -> dict[str, float]:
    """
    Extract coefficients from expression for given basis functions.
    
    Example:
        expr = -1447.128*x**2 + 0.567*cosh(x)
        basis = ["x**2", "cosh(x)"]
        → {"x**2": -1447.128, "cosh(x)": 0.567}
    """
    coefficients = {}
    x = sp.Symbol('x')
    
    for basis_str in basis_functions:
        basis = sp.sympify(basis_str)
        # Extract coefficient
        coef = expr.coeff(basis)
        if coef is not None:
            coefficients[basis_str] = float(coef)
        else:
            coefficients[basis_str] = 0.0
    
    return coefficients
```

#### Evaluation Implementation

**Add to `evaluate.py`**:

```python
def evaluate_approx_coef(
    pred_expr: sp.Expr,
    gt_coeffs: dict[str, float],
    basis_functions: list[str],
    relative_tolerance: float = 0.1  # 10% relative error
) -> dict[str, Any]:
    """
    Compare approximate coefficient solutions.
    
    Extracts coefficients and compares with relative tolerance.
    """
    # Extract predicted coefficients
    pred_coeffs = extract_coefficients(pred_expr, basis_functions)
    
    # Compare each coefficient
    matches = []
    errors = []
    relative_errors = []
    
    for basis in basis_functions:
        pred_val = pred_coeffs.get(basis, 0.0)
        gt_val = gt_coeffs.get(basis, 0.0)
        
        abs_error = abs(pred_val - gt_val)
        rel_error = abs_error / abs(gt_val) if gt_val != 0 else float('inf')
        
        matches.append(rel_error < relative_tolerance)
        errors.append(abs_error)
        relative_errors.append(rel_error)
    
    return {
        "coefficient_match_rate": sum(matches) / len(matches),
        "all_coefficients_match": all(matches),
        "mean_absolute_error": np.mean(errors),
        "mean_relative_error": np.mean(relative_errors),
        "per_coefficient_errors": dict(zip(basis_functions, errors)),
        "per_coefficient_relative_errors": dict(zip(basis_functions, relative_errors))
    }
```

---

### 5. Enhance `family` Evaluation 📊

**Current Problem:** Only checks structural match, doesn't validate parameters

#### Improved Implementation

**Update `evaluate.py`:**

```python
def evaluate_family_improved(
    pred_expr: sp.Expr, 
    gt_expr: sp.Expr, 
    domain: tuple[float, float]
) -> dict[str, Any]:
    """
    Enhanced family solution evaluation.
    
    Checks:
    1. Structural equivalence (ratio test)
    2. Correct number of free parameters
    3. Parameter naming convention (c_1, c_2, ...)
    """
    x = sp.Symbol('x')
    standard_vars = {"x", "t"}
    
    # Extract arbitrary parameters
    gt_params = [s for s in gt_expr.free_symbols if s.name not in standard_vars]
    pred_params = [s for s in pred_expr.free_symbols if s.name not in standard_vars]
    
    # Check 1: Structural match (existing method)
    structural_match = family_compare(pred_expr, gt_expr)
    
    # Check 2: Correct number of parameters
    param_count_match = len(pred_params) == len(gt_params)
    
    # Check 3: Parameter naming convention (c_1, c_2, c_3, ...)
    naming_convention = all(
        re.match(r'^c_\d+$', p.name) or p.name == 'C' 
        for p in pred_params
    )
    
    # Check 4: Non-trivial solution (not just zero or constant)
    is_nontrivial = x in pred_expr.free_symbols
    
    # Overall correctness
    correct = (
        structural_match and 
        param_count_match and 
        naming_convention and 
        is_nontrivial
    )
    
    return {
        "structural_match": structural_match,
        "param_count_match": param_count_match,
        "naming_convention": naming_convention,
        "is_nontrivial": is_nontrivial,
        "correct": correct,
        "pred_params": [p.name for p in pred_params],
        "gt_params": [p.name for p in gt_params],
        "param_count": {"predicted": len(pred_params), "ground_truth": len(gt_params)}
    }
```

---

## Implementation Roadmap

### Phase 1: Critical Fixes (1-2 days) 🚨

**Priority: Must-have for reliable evaluation**

#### Task 1.1: Add evaluation_points to augmentation system ✅ **COMPLETED**

**Implementation Date:** February 11, 2026

**Files modified:**
- ✅ `src/data/augmentations/base.py` - Added `_generate_evaluation_points()` method
- ✅ All 14 augmentation strategies - Inherit from BaseAugmentation (automatic)
- ⏳ `src/data/loaders/fredholm_loader.py` - Handle evaluation_points field (not yet needed)

**Completed Steps:**
1. ✅ Added helper function to BaseAugmentation class with overflow handling
2. ✅ All augmentation strategies inherit the method automatically
3. ✅ Tested on full sample dataset (5000 → 5750 equations)
4. ✅ Fixed Piecewise expression compatibility in 5 augmentation files
5. ✅ Added 22 passing tests for overflow filtering and evaluation points

**Testing:**
```bash
uv run python -m src.cli run --config configs/prepare_data.yaml
# ✅ Successfully generates 5750 augmented equations
# ✅ Pipeline runs end-to-end without errors
# ✅ Warnings reduced from 100+ to ~20 (remaining are from base dataset)
```

**Achieved outcome:** 
- ✅ All augmented equations can generate evaluation points (via inherited method)
- ✅ Overflow-safe evaluation with non-finite filtering
- ✅ All kernel expressions are SymPy-parseable
- ⏳ Evaluation points not yet stored in output files (future enhancement)

---

#### Task 1.2: Standardize discrete_points and series formats ⏳ **IN PROGRESS**

**Implementation Date:** February 11, 2026 (discrete_points completed)

**Files to modify:**
- ✅ `src/prompts/styles/basic.py` - Added discrete_points format specification
- ✅ `src/prompts/styles/chain_of_thought.py` - Added discrete_points format specification
- ✅ `src/prompts/styles/few_shot.py` - Added discrete_points format specification
- ✅ `src/prompts/styles/tool_assisted.py` - Added discrete_points format specification

**Changes implemented:**
```python
# Before:
- discrete_points: Solution only at discrete points

# After:
- discrete_points: Point values only. Format: [(x1, y1), (x2, y2), ...]
```

**Status:**
- ✅ discrete_points format specification complete (4/4 files updated)
- ⏳ series format specification pending
- ✅ All tests passing (37/37 prompt tests)

**Testing:**
```bash
# Verify prompt generation
uv run pytest tests/test_prompting.py tests/test_prompt_generation.py -v
# ✅ 37/37 tests passing
```

**Next:** Update series format specification, then proceed to Task 1.3 (parser implementation)

---

#### Task 1.3: Add discrete_points parser ✅ **COMPLETED**

**Implementation Date:** February 11, 2026

**Files modified:**
- ✅ `src/llm/postprocess.py` - Added `extract_discrete_points()` function
- ✅ `src/llm/postprocess.py` - Integrated into `parse_llm_output()` function
- ✅ `tests/test_discrete_points_parser.py` - Created comprehensive test suite (11 tests)

**Implementation:**
```python
def extract_discrete_points(response: str) -> Optional[list[tuple[float, float]]]:
    """
    Extract discrete point list from LLM response.
    
    Looks for format: [(x1, y1), (x2, y2), ...]
    Handles: scientific notation, negative values, extra whitespace
    Validates: >= 2 points, finite values, reasonable magnitudes
    """
    # Pattern matching for SOLUTION: line or standalone list
    # Parses individual (x, y) tuples
    # Returns list of float tuples or None
    pass
```

**Integration in parse_llm_output():**
```python
def parse_llm_output(response, extract_solution=True, validate=True):
    result = {
        "solution_str": None,
        "solution_sympy": None,
        "discrete_points": None,  # NEW: for discrete_points type
        # ... other fields
    }
    
    # Extract structured fields
    result["has_solution"] = _extract_has_solution(response)
    result["solution_type"] = _extract_solution_type(response)
    
    # NEW: Special handling for discrete_points
    if result["solution_type"] == "discrete_points":
        points = extract_discrete_points(response)
        if points:
            result["discrete_points"] = points
            result["solution_str"] = str(points)
            result["confidence"] = 0.8
        else:
            result["confidence"] = 0.3
        return result  # Don't try to parse as SymPy expression
    
    # ... continue with regular solution extraction
```

**Testing:**
```bash
# Run discrete_points parser tests
uv run pytest tests/test_discrete_points_parser.py -v
# ✅ 11/11 tests passing
```

**Test Coverage:**
- ✅ Standard format: [(x, y), ...]
- ✅ Whitespace handling: [ ( x , y ) , ... ]
- ✅ Scientific notation: [(0.0, 1.23e-2), ...]
- ✅ Negative values: [(-1.0, -2.5), ...]
- ✅ In context extraction from full responses
- ✅ Validation: rejects < 2 points, invalid formats, unreasonable values
- ✅ Integration with parse_llm_output

**Status:**
- ✅ Parser implementation complete
- ✅ Integration with postprocessing complete
- ✅ 11 unit tests passing
- ✅ Confidence scoring: 0.8 for successful extraction, 0.3 for failures

---

### Phase 2: Enhanced Evaluation (2-3 days) 📈

**Priority: Quality improvements**

#### Task 2.1: Implement specialized evaluators

**Files to create/modify:**
- `src/llm/evaluate.py` - Add new evaluation functions

**New functions:**
1. `evaluate_discrete_points()` - Point-wise comparison
2. `evaluate_series()` - Term-by-term comparison
3. `evaluate_approx_coef()` - Coefficient extraction + comparison
4. `evaluate_family_improved()` - Enhanced family validation

**Integration:**
```python
class SolutionEvaluator:
    def evaluate(self, solution, ground_truth, domain, solution_type=None):
        # Dispatch to specialized evaluator
        if solution_type == "discrete_points":
            return self.evaluate_discrete_points(...)
        elif solution_type == "series":
            return self.evaluate_series(...)
        elif solution_type == "approx_coef":
            return self.evaluate_approx_coef(...)
        elif solution_type == "family":
            return self.evaluate_family_improved(...)
        else:
            # Standard symbolic + numeric
            return self._evaluate_standard(...)
```

---

#### Task 2.2: Store coefficients for approx_coef

**Files to modify:**
- `src/data/augmentations/approx_coef/boundary_layer.py`
- `src/data/augmentations/approx_coef/oscillatory.py`
- (Other approx_coef strategies)

**Pattern:**
```python
def augment(self, entry):
    # ... existing augmentation ...
    
    # NEW: Extract and store coefficients
    if result["solution_type"] == "approx_coef":
        result["coefficients"] = self._extract_coefficients(
            result["u"], 
            basis_functions=["x**2", "cosh(x)", "sin(x)", ...]
        )
        result["basis_functions"] = basis_functions
    
    return result
```

---

#### Task 2.3: Update evaluate.py to use evaluation_points

**Files to modify:**
- `src/llm/evaluate.py` - Prioritize stored points over random generation

**Change:**
```python
def numeric_compare(solution, ground_truth, domain, n_points=100):
    # OLD: Always generate random points
    test_points = np.linspace(a, b, n_points)
    
    # NEW: Use stored points if available
    if hasattr(ground_truth, 'evaluation_points'):
        test_points = ground_truth.evaluation_points['x_values']
        gt_values = ground_truth.evaluation_points['u_values']
    else:
        # Fallback to random generation
        test_points = np.linspace(a, b, n_points)
        gt_values = [ground_truth(x) for x in test_points]
```

---

### Phase 3: Reporting & Analysis (1 day) 📊

**Priority: Insights and debugging**

#### Task 3.1: Enhanced metrics output

**Files to modify:**
- `src/llm/evaluate.py` - Expand metrics dictionary

**New metrics structure:**
```python
{
    "total": 100,
    "correct": 42,
    "accuracy": 0.42,
    "symbolic_accuracy": 0.15,
    "numeric_accuracy": 0.38,
    
    "per_type": {
        "exact_symbolic": {
            "total": 25,
            "correct": 20,
            "accuracy": 0.80
        },
        "approx_coef": {
            "total": 20,
            "correct": 15,
            "accuracy": 0.75,
            "avg_coef_error": 0.023,              # NEW
            "coef_match_rate": 0.80,              # NEW
            "per_basis_accuracy": {               # NEW
                "x**2": 0.85,
                "cosh(x)": 0.70
            }
        },
        "discrete_points": {
            "total": 15,
            "correct": 10,
            "accuracy": 0.67,
            "avg_point_error": 0.12,              # NEW
            "matched_point_rate": 0.75            # NEW
        },
        "series": {
            "total": 6,
            "correct": 4,
            "accuracy": 0.67,
            "avg_term_error": 0.15,               # NEW
            "avg_truncation_length": 4.2          # NEW
        },
        "family": {
            "total": 10,
            "correct": 8,
            "accuracy": 0.80,
            "structural_match_rate": 0.90,        # NEW
            "naming_convention_rate": 0.85        # NEW
        }
    },
    
    "has_solution_accuracy": 0.88,
    "solution_type_accuracy": 0.72,
    
    "confusion_matrix": {                         # NEW
        "approx_coef_vs_family": 3,  # Predicted family, was approx_coef
        "series_vs_exact_symbolic": 2,
        ...
    }
}
```

---

#### Task 3.2: Per-equation detailed output

**Files to modify:**
- `src/llm/evaluate.py` - Add detailed evaluation breakdown to predictions JSONL

**Enhanced predictions output:**
```jsonl
{
    "equation_id": "test100_approx_coef_14",
    "solution_type": "approx_coef",
    
    "solution_str": "-1447.128*x**2 + 0.567*cosh(x)",
    "ground_truth": "-1447.128*x**2 + 0.567*cosh(x)",
    
    "evaluation": {
        "correct": true,
        "symbolic_match": false,
        "numeric_match": true,
        
        "approx_coef_details": {
            "coefficients": {
                "x**2": {"pred": -1447.13, "gt": -1447.128, "error": 0.002, "match": true},
                "cosh(x)": {"pred": 0.568, "gt": 0.567, "error": 0.001, "match": true}
            },
            "all_coef_match": true,
            "mean_relative_error": 0.0015
        },
        
        "numeric_details": {
            "rmse": 0.0023,
            "max_error": 0.0051,
            "mean_error": 0.0019,
            "evaluation_points_used": 50
        }
    }
}
```

---

#### Task 3.3: Confusion matrix for solution_type

**Files to modify:**
- `src/llm/evaluate.py` - Track misclassifications

**Implementation:**
```python
def build_confusion_matrix(results):
    """Track solution_type prediction errors."""
    confusion = defaultdict(int)
    
    for r in results:
        pred_type = r.get("solution_type")
        gt_type = r.get("ground_truth_solution_type")
        
        if pred_type != gt_type:
            key = f"{gt_type}_predicted_as_{pred_type}"
            confusion[key] += 1
    
    return dict(confusion)
```

**Use case:**
- Identify if LLMs frequently confuse `approx_coef` with `family`
- Track `series` vs `exact_symbolic` misclassifications
- Measure impact of removing `exact_coef`

---

## Prompt Modifications

### Keep Minimal - Don't Inflate Prompts

**Goal:** Add only essential format hints without confusing LLMs

### ✅ Recommended Changes (1 line each)

#### For `discrete_points`:
```diff
SOLUTION_TYPE: [exact_symbolic/approx_coef/discrete_points/series/family/regularized/none]

For SOLUTION_TYPE:
  - exact_symbolic: Closed-form symbolic solution (e.g., u(x) = sin(x))
  - approx_coef: Approximate with NUMERIC coefficients (e.g., u(x) = 0.5*sin(x) + 1.2*x)
- - discrete_points: Solution only at discrete points
+ - discrete_points: Point values only. Format: [(x1, y1), (x2, y2), ...]
  - series: Infinite series solution (e.g., u(x) = Σ aₙxⁿ)
```

#### For `series`:
```diff
- - series: Infinite series solution (e.g., u(x) = Σ aₙxⁿ)
+ - series: Series expansion. Express as sum: f + λK·f + λ²K²·f + ... (4-6 terms)
```

#### For `family` (already updated):
```python
- family: Non-unique solutions (arbitrary c_1, c_2, ...)  # ✅ Already good
```

#### For `approx_coef` (already clear):
```python
- approx_coef: Approximate with NUMERIC coefficients (e.g., u(x) = 0.5*sin(x) + 1.2*x)
# ✅ No change needed - already specifies numeric values
```

**Total prompt inflation:** ~20-30 words across all solution types

---

### ❌ Avoid Over-Specification

**Don't do this:**
```python
# ❌ TOO VERBOSE - Will confuse LLMs
"""
For series solutions:
  (1) If you can express the series symbolically, provide the first 4-6 terms explicitly
  (2) Use the Neumann series expansion: f(x) + λ∫K·f(t)dt + λ²∫∫K²·f(t)dt + ...
  (3) If truncating, clearly state the truncation order
  (4) Alternatively, if coefficients are easier, provide them as a list: [a_0, a_1, a_2, ...]
  (5) Ensure all terms are simplified before presenting
"""
# This is 5 lines of instructions → too much cognitive load
```

**Instead:**
```python
# ✅ CONCISE - Single actionable instruction
- series: Series expansion. Express as sum: f + λK·f + λ²K²·f + ... (4-6 terms)
```

---

## Expected Outcomes

### After Phase 1 (Critical Fixes)

✅ **Consistent Metrics**
- RMSE/MAE identical across evaluation runs
- Fixed evaluation points enable reproducible research
- Baseline for comparison across different model runs

✅ **Format Compliance**
- LLMs know how to output discrete_points: `[(x, y), ...]`
- LLMs know how to output series: first 4-6 terms explicitly
- Parser successfully extracts structured formats

### After Phase 2 (Enhanced Evaluation)

✅ **Per-Type Accuracy Reporting**
```
Overall Accuracy: 42%

By Solution Type:
  exact_symbolic:   80% (20/25) ✅ Strong
  approx_coef:      75% (15/20) ✅ Good, avg coefficient error: 2.3%
  discrete_points:  67% (10/15) ⚠️  Moderate, avg point error: 12%
  series:           67% (4/6)   ⚠️  Moderate, avg 4.2 terms predicted
  family:           80% (8/10)  ✅ Good, naming convention: 85%
  regularized:     100% (5/5)   ✅ Perfect (type classification only)
  none:             90% (9/10)  ✅ Good
```

✅ **Coefficient-Level Analysis**
- Track which basis functions LLMs predict accurately
- Identify if certain coefficients (e.g., exponential decay rates) are harder
- Compare relative vs absolute coefficient errors

✅ **Series Convergence Analysis**
- Measure truncation quality (do 4 terms suffice?)
- Track if LLMs provide correct leading terms
- Identify if Neumann expansion is understood

### After Phase 3 (Reporting & Analysis)

✅ **Debugging Capability**
- Identify which solution types LLMs struggle with
- Track common misclassifications (confusion matrix)
- Pinpoint exact equations that fail (per-equation details)

✅ **Publication-Ready Metrics**
- Reproducible results for academic papers
- Per-type breakdown for figures and tables
- Statistical significance testing (fixed evaluation points)

✅ **Model Comparison**
- Compare GPT-4 vs GPT-3.5 vs local models reliably
- Track improvement over time (before/after prompt changes)
- Identify which solution types benefit from larger models

### Research Insights Enabled

**Can now answer:**

1. **Solution Type Recognition Performance:**
   - "GPT-4 achieves 72% solution_type accuracy"
   - "Most confused: approx_coef vs family (15 cases)"
   - "Removed exact_coef reduced confusion by 8%"

2. **Solution Prediction Performance:**
   - "exact_symbolic: 80% accuracy (symbolic equivalence)"
   - "approx_coef: 75% accuracy, 2.3% avg coefficient error"
   - "series: 67% accuracy, avg truncation at 4.2 terms"

3. **Format Impact:**
   - "LaTeX format: 45% accuracy"
   - "Infix format: 42% accuracy"
   - "RPN format: 38% accuracy (no Math-Verify support)"

4. **Prompt Strategy Comparison:**
   - "Chain-of-thought: 48% accuracy"
   - "Few-shot: 45% accuracy"
   - "Basic: 38% accuracy"

5. **Model Scaling:**
   - "GPT-4: 42% overall, 80% exact_symbolic, 67% series"
   - "GPT-3.5: 28% overall, 60% exact_symbolic, 40% series"
   - "Llama-3-70B: 35% overall, 70% exact_symbolic, 55% series"

---

## Files to Modify Summary

### Phase 1 Files (Critical)

1. ✅ `src/data/augmentations/base.py` - Add evaluation_points generation **[COMPLETED Feb 11, 2026]**
2. ✅ All 14 augmentation strategies (exact_symbolic/, approx_coef/, etc.) **[COMPLETED - inherited from base]**
   - ✅ `disconnected_support.py` - Fixed Piecewise expressions (2 locations)
   - ✅ `mixed_type.py` - Converted ternary to Piecewise
   - ✅ `compact_support.py` - Fixed 2 kernel definitions (banded + localized)
   - ✅ `neumann_series.py` - Replaced placeholder with Integral notation
3. ✅ `src/prompts/styles/basic.py` - Added discrete_points format specification
4. ✅ `src/prompts/styles/chain_of_thought.py` - Added discrete_points format specification
5. ✅ `src/prompts/styles/few_shot.py` - Added discrete_points format specification
6. ✅ `src/prompts/styles/tool_assisted.py` - Added discrete_points format specification
7. ✅ `src/llm/postprocess.py` - Added discrete_points parser
8. ✅ `tests/test_discrete_points_parser.py` - Added 11 tests for parser

**Phase 1 Progress: 4/7 primary tasks complete (57%) + 5 additional fixes**

### Phase 2 Files (Enhanced Evaluation)

8. ⏳ `src/llm/evaluate.py` - Add specialized evaluators (4 functions)
9. ⏳ Approx_coef augmentation strategies - Store coefficients
10. ✅ `src/data/augmentations/base.py` - Evaluation points generation available **[COMPLETED - can be integrated]**

### Phase 3 Files (Reporting)

11. ⏳ `src/llm/evaluate.py` - Expand metrics output
12. ⏳ `src/llm/evaluate.py` - Add confusion matrix tracking
13. ⏳ Predictions JSONL output - Enhanced per-equation details

**Total:** ~13 files to modify, ~15 functions to add/update

**Overall Progress (February 11, 2026):**
- ✅ Phase 1: 4/7 tasks complete (57%) - discrete_points format + parser done, series format pending
- ⏳ Phase 2: 0/5 tasks complete (0%)
- ⏳ Phase 3: 0/3 tasks complete (0%)
- **Infrastructure Foundation: SOLID** - Evaluation points, expression parsing, and discrete_points extraction all working

---

## Next Steps

### ✅ Completed (February 11, 2026)

1. ✅ **Evaluation points infrastructure** - Overflow-safe generation in BaseAugmentation
2. ✅ **SymPy expression compatibility** - All augmentation kernels use parseable Piecewise syntax
3. ✅ **Non-finite value filtering** - Automatic handling of exp/cosh overflows
4. ✅ **Windows YAML compatibility** - ASCII-only config files
5. ✅ **Test coverage** - 22 passing tests for new features

### Immediate Actions

1. **Review this document** - Confirm strategy aligns with project goals
2. **Prioritize phases** - Decide which phases to implement first
3. **Allocate resources** - Estimate implementation timeline
4. **Test incrementally** - Implement Phase 1 → test → Phase 2 → test

### Questions to Resolve

1. **Dataset regeneration:** ✅ **RESOLVED** - Evaluation points can be generated on-demand via `BaseAugmentation._generate_evaluation_points()`. Future enhancement: Store in output files for faster evaluation.
2. **Backward compatibility:** Should old predictions still be evaluable?
3. **Series format preference:** Symbolic sum vs coefficient list?
4. **Tolerance tuning:** What relative tolerance for approx_coef (currently 10%)?

### Validation Plan

**✅ Completed for Phase 1 - Task 1.1 (February 11, 2026):**

1. ✅ **Unit tests** for evaluation point generation and overflow filtering (22/22 passing)
   - `tests/test_evaluation_points.py` - Overflow handling
   - `tests/test_disconnected_support_piecewise.py` - Piecewise parsing
   - `tests/test_augmentation.py` - Existing augmentation tests still pass
2. ✅ **Integration tests** on full sample dataset (5000 → 5750 equations)
   - All 14 augmentation strategies working
   - Pipeline runs end-to-end successfully
   - Warnings reduced from 100+ to ~20 (remaining from base dataset only)
3. ⏳ **Full evaluation** on test_100 dataset with known results (pending LLM inference)
4. ⏳ **Comparison** before/after metrics to verify improvements (pending Phase 2 evaluator implementations)

**Pending for Remaining Tasks:**

1. ⏳ **Unit tests** for discrete_points parser (pending Task 1.3)
2. ⏳ **Unit tests** for each specialized evaluator function (pending Phase 2)
3. ⏳ **Integration tests** on sample dataset (10-20 equations per type)
4. ⏳ **Full evaluation** on test_100 dataset with known results
5. ⏳ **Comparison** before/after metrics to verify improvements

---

## 📋 Change Log

### February 11, 2026 - Phase 1 (Task 1.1) Implementation

**Completed Features:**
1. ✅ Evaluation points generation with overflow handling in `BaseAugmentation._generate_evaluation_points()`
2. ✅ Non-finite value filtering (automatic inf/nan removal)
3. ✅ SymPy-parseable Piecewise expressions in all 14 augmentation strategies
4. ✅ Fixed 5 augmentation files with invalid kernel definitions
5. ✅ Windows YAML ASCII compatibility
6. ✅ 22 passing tests for new functionality
7. ✅ Full pipeline validation (5000 → 5750 equations, 80% reduction in warnings)

**Impact:**
- Robust numeric evaluation infrastructure ready for Phase 2 integration
- All augmentation strategies produce valid SymPy expressions
- Dataset preparation pipeline runs cleanly on Windows and Unix systems

**Next Priority:** Task 1.2 - Standardize discrete_points and series output formats in prompt templates

### February 11, 2026 - Phase 1 (Task 1.2) Partial Implementation

**Completed Features:**
1. ✅ discrete_points format specification added to all 4 prompt styles
   - [src/prompts/styles/basic.py](src/prompts/styles/basic.py)
   - [src/prompts/styles/chain_of_thought.py](src/prompts/styles/chain_of_thought.py)
   - [src/prompts/styles/few_shot.py](src/prompts/styles/few_shot.py)
   - [src/prompts/styles/tool_assisted.py](src/prompts/styles/tool_assisted.py)
2. ✅ New format: "Point values only. Format: [(x1, y1), (x2, y2), ...]"
3. ✅ All prompt tests passing (37/37)

**Impact:**
- LLMs now have explicit instructions for discrete_points output
- Enables Task 1.3: discrete_points parser implementation
- Structured format ensures consistent evaluation

**Next Priority:** Complete series format specification, then implement discrete_points parser (Task 1.3)

### February 11, 2026 - Phase 1 (Task 1.3) Implementation

**Completed Features:**
1. ✅ `extract_discrete_points()` function in postprocess.py
2. ✅ Integration with `parse_llm_output()` for automatic detection
3. ✅ Comprehensive test suite: 11 tests covering all formats
4. ✅ Validation: minimum 2 points, finite values, reasonable magnitudes
5. ✅ Error handling: graceful fallback with confidence scoring

**Supported Formats:**
- Standard: `[(0.0, 1.234), (0.25, 2.456), ...]`
- Scientific notation: `[(0.0, 1.23e-2), (0.5, 3.45e1), ...]`
- Negative values: `[(-1.0, -2.5), (0.0, 0.0), ...]`
- Extra whitespace: `[ ( 0.0 , 1.2 ) , ( 0.5 , 3.4 ) ]`

**Impact:**
- Reliable extraction of discrete_points solutions from LLM responses
- Enables specialized evaluation for discrete_points equations (Phase 2)
- Foundation for point-wise comparison metrics

**Next Priority:** Implement series format specification, then specialized evaluators (Phase 2)

---

**END OF DOCUMENT**
