# Evaluation Strategy Comprehensive Review

**Date:** February 12, 2026  
**Status:** Phase 2 Task 2.3 Complete - family evaluator metadata + termwise + multi-sample numeric  
**Last Updated:** February 12, 2026

---

## 🎯 Implementation Status Update (February 12, 2026)

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

- ✅ **Task 1.2**: Standardize discrete_points format in prompts - COMPLETED
- ✅ **Task 1.3**: Add discrete_points parser - COMPLETED
- ✅ **Phase 2, Task 2.1**: discrete_points evaluation - COMPLETED (February 12, 2026)
- ✅ **Phase 2, Task 2.2**: series evaluation - COMPLETED
- ✅ **Phase 2, Task 2.3**: family evaluator enhancements - COMPLETED
- ✅ **Phase 2, Task 2.4**: use stored evaluation_points in evaluate.py - COMPLETED
- ✅ **Phase 2, Task 2.5**: write evaluation_points into dataset outputs - COMPLETED
- ⏳ **Phase 3**: Enhanced reporting and metrics (not started)

---

### ✅ Phase 2, Task 2.1 - discrete_points Evaluation - COMPLETED (February 12, 2026)

**Point-wise Solution Comparison Metrics**: Successfully implemented in `src/llm/evaluate.py`

- ✅ Added `evaluate_discrete_points()` function (75 lines, standalone)
- ✅ Integrated `SolutionEvaluator.evaluate_discrete_points_type()` method
- ✅ **Tolerance-based point matching**: x_tolerance (default 1e-3), y_tolerance (configurable)
- ✅ **Computed metrics**:
  - `matched_points`: Count of ground truth points found in predictions (within tolerance)
  - `accuracy`: Percentage of ground truth points matched (%)
  - `max_error`: Maximum y-value difference across all matches
  - `mean_error`: Average y-value difference across all matches
  - `rmse`: Root mean squared error for matched points
- ✅ **Match classification**: 80% threshold for "match" status (matched_points / total_points >= 0.80)
- ✅ **Integration**: Works with SolutionEvaluator tracking and summary statistics
- ✅ **Test coverage**: 13 passing unit tests covering all scenarios

**Key Function Signature:**
```python
def evaluate_discrete_points(pred_points, gt_points, x_tolerance=1e-3, y_tolerance=1e-3):
    """
    Compare predicted and ground truth points with tolerance-based matching.
    
    Returns: {
        'match': bool,           # 80% threshold for match classification
        'matched_points': int,   # Count of matched points
        'total_points_pred': int,
        'total_points_gt': int,
        'accuracy': float,       # % of ground truth points matched
        'max_error': float,      # Maximum y-value error
        'mean_error': float,     # Average y-value error
        'rmse': float            # Root mean squared error
    }
    """
```

**Test Coverage (13 Tests):**
- Exact point matches (2 tests)
- Partial point matches (3 tests)
- Tolerance-based x/y coordinate matching (2 tests)
- Tolerance threshold validation (1 test)
- Edge cases: negative values, scientific notation, empty lists (4 tests)
- SolutionEvaluator integration (2 tests)

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
| `approx_coef` | ✅ Symbolic + numeric + per-term coeffs | None |
| `discrete_points` | ✅ Point-wise comparison | None |
| `series` | ✅ Symbolic + numeric + term-by-term | None |
| `family` | ✅ Structural + numeric + metadata | None (metadata is informational) |
| `regularized` | ✅ Type classification only | None (no solution to evaluate) |
| `none` | ✅ Binary check | None |

**Key Problems:**

- **`series`**: Standardized 4-term output format now defined
    - Ground truth: Series expansion with 4 terms
    - LLM output: Fixed 4-term sum in SOLUTION
    - Evaluation: Symbolic + numeric + term-by-term RMSE

- **`discrete_points`**: Standardized output and parser implemented
    - Ground truth: Function values at specific points
    - LLM output: Structured list format
    - Evaluation: Point-wise metrics with tolerance

- **`approx_coef`**: Per-term coefficient comparison implemented
    - Ground truth: `-1447.128*x**2 + 0.567*cosh(x)`
    - Evaluation: Symbolic + numeric + per-term coefficient errors

- **`family`**: Validation now includes numeric checks + metadata
    - Structural match (ratio test)
    - Multi-sample numeric comparison
    - Parameter metadata (count + naming) recorded for analysis

#### 2. Format Limitations

**RPN Format:**
- Math-Verify doesn't support RPN (LaTeX/infix only)
- Requires manual conversion via `rpn_to_sympy()`
- Adds complexity and potential errors

**Series Notation:**
- Standardized 4-term series format in prompts
- Ground truth uses truncated 4-term expansions

**Discrete Points:**
- Structured format in prompts
- Parser implemented and tested

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
- Fails on expressions with free symbols (family) without constant substitution

---

## Recommended Evaluation Strategy

### Per-Type Evaluation Matrix

| Solution Type | Primary Method | Fallback Method | LLM Output Format | Pre-compute in Dataset |
|---------------|----------------|-----------------|-------------------|------------------------|
| **`exact_symbolic`** | ✅ Symbolic equivalence | ✅ Numeric RMSE | Natural math expression | ❌ None needed |
| **`approx_coef`** | ✅ **Per-term coefficient comparison** | ✅ Numeric RMSE | **📝 Standard: numeric values only** | ❌ None |
| **`discrete_points`** | ✅ **Point-wise comparison** | ❌ N/A | **📝 Structured: `[(x1,y1), ...]`** | ✅ **Store evaluation points** |
| **`series`** | ✅ **Term-by-term comparison** | ✅ Numeric RMSE (truncated) | **📝 Standard: 4 explicit terms in SOLUTION** | ❌ None |
| **`family`** | ✅ Structural + param metadata | ✅ Numeric (multi-sample constants) | **📝 Use `c_1, c_2, ...`** | ✅ **Store points with samples** |
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
                "n_points": len(x_values),
                "constant_samples": [-1.0, 1.0, 2.0],  # for family solutions
                "u_values_samples": ["..."],  # per-sample u(x) values when free constants exist
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

- **Benefits:**
- ✅ Consistent RMSE/MAE across all evaluation runs
- ✅ Faster evaluation (no re-computation)
- ✅ Can include challenging points (boundaries, discontinuities, inflection points)
- ✅ Reproducible research results
- ✅ Family solutions supported by multi-sample constant substitution during point generation

**Scope:**
- ✅ Update all 14 augmentation strategies in `src/data/augmentations/` (inherited via BaseAugmentation)
- ✅ Modify CSV/JSONL writers to include evaluation_points field (completed)
- ✅ Update evaluate.py to prioritize stored points over random generation (completed)

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

### 3. Standardize `series` Format ✅ **COMPLETED (February 12, 2026)**

**Current Problem (resolved):** Inconsistent series representation across LLM outputs

#### LLM Prompt Addition (Minimal)

**Add to prompt styles:**

```python
- series: Truncated series with exactly 4 explicit terms in SOLUTION
```

#### Parser Implementation

**No extra parser required** - series stays in SOLUTION and is parsed as a single expression.

#### Evaluation Implementation

- Symbolic + numeric evaluation uses standard pipeline
- Added term-by-term numeric evaluation for per-term RMSE (extra metric)

**Add to `evaluate.py`**:

```python
def evaluate_series_terms(
    solution: sp.Expr,
    ground_truth: sp.Expr,
    domain: tuple[float, float] = (0, 1),
    n_points: int = 100,
    tolerance: float = 1e-6
) -> dict[str, Any]:
    """
    Compare series solutions term-by-term using numeric RMSE.

    Splits each expression into top-level terms and compares
    term i against term i over shared evaluation points.
    """
    # Returns per-term RMSE, mean/max RMSE, and match rate
    pass
```

**Dataset Enhancement:**

Not required. Series remains a single expression in `u` with 4 explicit terms.

---

### 4. Improve `approx_coef` Evaluation ✅ **COMPLETED (February 12, 2026)**

**Approach:** Compare coefficients per top-level term in the expression.

#### Evaluation Implementation

**Added to `evaluate.py`:**

```python
def evaluate_approx_coeffs(
    solution: sp.Expr,
    ground_truth: sp.Expr,
    tolerance: float = 1e-6,
    relative_tolerance: float = 0.1
) -> dict[str, Any]:
    """
    Compare approx_coef solutions by extracting per-term coefficients.
    """
    # Returns match_rate, mean/max errors, and per-term errors
    pass
```

**Notes:**
- No dataset coefficient storage required
- Metrics are recorded in `approx_coef_eval` and summarized in `approx_coef_stats`

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

#### Task 1.2: Standardize discrete_points and series formats ✅ **COMPLETED (February 12, 2026)**

**Implementation Date:** February 12, 2026

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
- ✅ series format specification complete (4 explicit terms in SOLUTION)
- ✅ All tests passing (37/37 prompt tests)

**Testing:**
```bash
# Verify prompt generation
uv run pytest tests/test_prompting.py tests/test_prompt_generation.py -v
# ✅ 37/37 tests passing
```

**Next:** Proceed to Phase 3 (reporting + metrics)

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

#### Task 2.1: Implement specialized evaluators - ✅ **COMPLETED (February 12, 2026)**

**discrete_points Evaluation Implementation:**

**Files modified:**
- ✅ `src/llm/evaluate.py` - Added `evaluate_discrete_points()` function and integration method

**Functions Implemented:**
1. ✅ `evaluate_discrete_points()` - Point-wise comparison with tolerance (75 lines)
2. ✅ `evaluate_series_terms()` - Term-by-term numeric comparison (completed)
3. ✅ `evaluate_approx_coeffs()` - Per-term coefficient comparison (completed)
4. ✅ `evaluate_family_improved()` - Enhanced family validation (completed)

**Integration:**
```python
class SolutionEvaluator:
    def evaluate_discrete_points_type(self, pred_points, gt_points):
        """Evaluate discrete_points solution type."""
        result = evaluate_discrete_points(
            pred_points, 
            gt_points, 
            x_tolerance=self.numeric_tolerance,
            y_tolerance=self.numeric_tolerance
        )
        # Track result in SolutionEvaluator for summary statistics
        self.results[solution_type].append(result)
        return result
```

**Discrete_points Evaluation Metrics:**
```python
{
    'match': bool,           # 80% threshold for match classification
    'matched_points': int,   # Count of matched points
    'total_points_pred': int,
    'total_points_gt': int,
    'accuracy': float,       # % of ground truth points matched
    'max_error': float,      # Maximum y-value error
    'mean_error': float,     # Average y-value error
    'rmse': float            # Root mean squared error
}
```

**Algorithm:**
1. For each ground truth point, find closest predicted point by x-coordinate
2. If x-distance <= x_tolerance, mark as potential match
3. If y-distance <= y_tolerance, mark as confirmed match
4. Calculate accuracy as matched_points / total_gt_points
5. Classify as "match" if accuracy >= 0.80 (80% threshold)
6. Compute error metrics for matched pairs only

**Test Coverage:**
- ✅ 13 comprehensive unit tests (all passing)
- Exact point matches (2 tests)
- Partial point matches (3 tests)
- Tolerance-based x/y coordinate matching (2 tests)
- Tolerance threshold validation (1 test)
- Edge cases: negative values, scientific notation, empty lists (4 tests)
- SolutionEvaluator integration (2 tests)

**Integration with SolutionEvaluator:**
- ✅ Added to SolutionEvaluator.evaluate_discrete_points_type() method
- ✅ Tracked in self.results dictionary per solution type
- ✅ Aggregated in summary() for per-type accuracy statistics
- ✅ No breaking changes to existing evaluation pipeline

**Status:**
- ✅ Complete implementation with full metrics
- ✅ All 13 unit tests passing
- ✅ No regressions in existing evaluate.py tests (18/18 passing)
- ✅ Production-ready for discrete_points evaluation

**Next Tasks:**
- ⏳ Phase 3: Enhanced reporting and metrics

---

#### Task 2.2: Implement series evaluation ✅ **COMPLETED (February 12, 2026)**

**Goal:** Keep series evaluation consistent with other solution types (symbolic + numeric), plus term-by-term numeric evaluation and term-count tracking.

**Implementation Summary:**
- Series solutions are evaluated with the existing symbolic and numeric comparison pipeline (same as exact_symbolic).
- Added term-by-term numeric evaluation with per-term RMSE.
- Added a lightweight term-count helper to record the number of top-level terms in the predicted series expression.
- Term count metadata is stored on each evaluation result and summarized in aggregate metrics.

**Files modified:**
- ✅ `src/llm/evaluate.py`
    - Added `count_series_terms()` helper
    - Added `evaluate_series_terms()` for term-by-term numeric RMSE
    - Added `series_term_count`, `series_term_target`, and `series_term_match` to per-result output when `solution_type == "series"`
    - Added `series_term_eval` to per-result output
    - Added `series_term_stats` to `SolutionEvaluator.summary()` for reporting
- ✅ `src/adaptive_pipeline.py`
    - Writes `predictions_evaluated_<timestamp>.jsonl` with per-prediction `evaluation` (includes `series_term_eval`)
- ✅ `tests/test_evaluate.py`
    - Added tests for series term metadata and summary stats

**Behavior:**
- **Correctness:** unchanged (still `symbolic` OR `numeric` match)
- **Extra evaluation:** term-by-term numeric RMSE is recorded in `series_term_eval`
- **Metadata only:** term-count match does not affect correctness yet

**Tests:**
```bash
uv run pytest tests/test_evaluate.py -v
# ✅ 18/18 passing
```

---

#### Task 2.3: Implement family evaluator ✅ **COMPLETED (February 12, 2026)**

**Files modified:**
- ✅ `src/llm/evaluate.py` - Added family numeric comparison (multi-sample) + termwise metrics + parameter metadata
- ✅ `src/data/augmentations/base.py` - Multi-sample constants and per-sample u(x) values in evaluation_points

**Implementation status:**
- ✅ Added term-by-term numeric evaluation for family (`family_term_eval`)
- ✅ Added multi-sample numeric comparison for free constants
- ✅ Added parameter metadata: count + naming info (`family_param_eval`)
- ✅ Tests added for family termwise + parameter metadata

---

#### Task 2.4: Update evaluate.py to use evaluation_points ✅ **COMPLETED (February 12, 2026)**

**Files modified:**
- ✅ `src/llm/evaluate.py` - Numeric comparison uses stored evaluation_points when available
- ✅ `src/adaptive_pipeline.py` - Persist evaluation_points in predictions and use during evaluation
- ✅ `src/prompts/base.py` + `src/prompts/batch_processor.py` - Carry evaluation_points into prompt metadata

**Implementation summary:**
- Numeric comparison prefers stored points for consistency
- Family numeric comparison uses pre-computed multi-sample points when available

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
    - series: Truncated series with exactly 4 explicit terms
```

#### For `series`:
```diff
- - series: Truncated series with exactly 4 explicit terms
+ - series: Truncated series with exactly 4 explicit terms
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
    (1) If you can express the series symbolically, provide exactly 4 terms
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
- series: Truncated series with exactly 4 explicit terms
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
- LLMs know how to output series: exactly 4 explicit terms
- Parser successfully extracts structured formats

### After Phase 2 (Enhanced Evaluation)

✅ **Per-Type Accuracy Reporting**
```
Overall Accuracy: 42%

By Solution Type:
  exact_symbolic:   80% (20/25) ✅ Strong
  approx_coef:      75% (15/20) ✅ Good, avg coefficient error: 2.3%
  discrete_points:  67% (10/15) ⚠️  Moderate, avg point error: 12%
    series:           67% (4/6)   ⚠️  Moderate, avg 4.0 terms predicted
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

8. ✅ `src/llm/evaluate.py` - Add specialized evaluators (series, approx_coef, discrete_points, family)
9. ✅ Family evaluator implementation
10. ✅ `src/data/augmentations/base.py` - Evaluation points generation available **[COMPLETED - can be integrated]**
11. ✅ `src/llm/evaluate.py` - Use stored evaluation_points
12. ✅ Dataset writers - Persist evaluation_points into CSV/JSONL outputs

### Phase 3 Files (Reporting)

11. ⏳ `src/llm/evaluate.py` - Expand metrics output
12. ⏳ `src/llm/evaluate.py` - Add confusion matrix tracking
13. ⏳ Predictions JSONL output - Enhanced per-equation details

**Total:** ~15 files to modify, ~17 functions to add/update

**Overall Progress (February 12, 2026):**
- ✅ Phase 1: 5/7 tasks complete (71%) - discrete_points + series format complete
- ✅ Phase 2: 5/5 tasks complete (100%) - discrete_points, series, approx_coef, family evaluation complete + evaluation_points integration
- ⏳ Phase 3: 0/3 tasks complete (0%)
- **Infrastructure Foundation: SOLID** - Evaluation points, expression parsing, and specialized evaluators are working

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

1. **Dataset regeneration:** ✅ **RESOLVED** - Evaluation points are generated on-demand and persisted in dataset outputs for faster evaluation.
2. **Backward compatibility:** Should old predictions still be evaluable?
3. **Series format preference:** Fixed 4-term sum (resolved)
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

1. ⏳ **Integration tests** on sample dataset (10-20 equations per type)
2. ⏳ **Full evaluation** on test_100 dataset with known results
3. ⏳ **Comparison** before/after metrics to verify improvements

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

**Next Priority:** Phase 3 reporting and metrics

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

**Next Priority:** Phase 3 reporting and metrics

---

**END OF DOCUMENT**
