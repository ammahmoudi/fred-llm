# Data Quality Summary

## has_solution and u Field Patterns

### Summary (after fixes applied in training_data_v2/)

✅ **All issues resolved!**

- **has_solution field**: Now present for all 5767 entries (100% complete)
- **solution_type='none' consistency**: All 221 entries correctly have empty u
- **No data quality issues found**

### u Field Patterns by solution_type

| solution_type | Total | Empty u | Non-empty u | With Float Coefficients |
|---------------|-------|---------|-------------|-------------------------|
| **exact** (original) | 5000 | 0 (0%) | 5000 (100%) | 2935 (59%) |
| **family** | 46 | 0 (0%) | 46 (100%) | 0 (0%) |
| **none** | 221 | 221 (100%) | 0 (0%) | 0 (0%) |
| **numerical** | 431 | 95 (22%) | 336 (78%) | 232 (54%) |
| **regularized** | 69 | 69 (100%) | 0 (0%) | 0 (0%) |

### What This Means

1. **exact** (original dataset):
   - All have solutions
   - ~59% contain float values (may be approximations from numerical integration, e.g., `3.14159*x`)
   - ~41% are purely symbolic with no floats (e.g., `x**2 + 2*x`)
   - **Note**: Floats can arise from SymPy's numerical evaluation of definite integrals with numerical bounds

2. **family** (non-unique solutions):
   - All have symbolic solutions like `C` or `C * sin(pi*x)`
   - No numerical coefficients (pure symbolic)

3. **none** (no solution exists):
   - **All correctly have empty u=""**
   - Examples: eigenvalue violations, range violations, divergent kernels

4. **numerical** (numerical methods only):
   - 22% empty (no known solution)
   - 78% have reference solutions (often with float values from numerical evaluation)
   - Examples: `exp((x - 4.28)/0.01)`, `tanh((x - 2.05)/0.01)`
   - **Note**: Floats here may be approximations from symbolic integration with numerical bounds

5. **regularized** (ill-posed problems):
   - **All correctly have empty u=""**
   - These are Fredholm 1st kind equations requiring regularization

### Design Intent

The u field patterns are **intentional**:

- Empty `u=""` signals: "No analytical solution exists/is known"
- Non-empty with floats: May contain approximate values from numerical integration or randomly generated coefficients
- Non-empty without floats: Purely symbolic expressions

**Important**: Float values in expressions can be:
1. Random coefficients inserted during generation (via `np.random.uniform()`)
2. Approximations from SymPy's numerical evaluation of definite integrals (when bounds are numerical)

This teaches LLMs to:
1. Recognize when no solution exists (`has_solution=False`, `u=""`)
2. Distinguish symbolic vs numerical solutions
3. Handle edge cases requiring special methods

### Fixed Issues

**Before** (training_data/):
- ✗ 69 entries missing `has_solution` field (all regularized type)
- ✗ 24 `solution_type='none'` entries had non-empty u values

**After** (training_data_v2/):
- ✓ All entries have `has_solution` field
- ✓ All `solution_type='none'` entries have empty u
- ✓ Data is consistent and ready for training

### Files

- Analysis script: `scripts/check_data_quality.py`
- Fixed data: `data/processed/training_data_v2/augmented/`
- Original (with issues): `data/processed/training_data/augmented/`
