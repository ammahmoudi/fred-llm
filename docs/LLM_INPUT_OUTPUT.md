# LLM Input/Output Specification

This document clarifies what LLMs receive as input and what they are expected to provide as output for each solution type in the Fred-LLM project.

## Key Principle

**Fred-LLM's purpose: Finding SYMBOLIC/FUNCTIONAL solutions to Fredholm equations.**

LLMs should provide symbolic expressions of the solution function, not numeric values. The evaluation differs based on solution category:

### Evaluation Criteria by Category

**Symbolic Solution Categories** (need BOTH correct type + correct expression):
- ✅ `exact_symbolic` - Must predict correct function/constant
- ✅ `approx_coef` - Must predict correct function with numeric coefficients
- ✅ `series` - Must predict correct series form
- ✅ `family` - Must predict correct parameterized solution

**Edge Case Categories** (correct type classification is sufficient):
- ✅ `none` - Just correctly identify "no solution exists"
- ✅ `discrete_points` - Just correctly identify "solution only at discrete points"
- ✅ `regularized` - Just correctly identify "requires regularization method"

---

## ✅ Successful LLM Predictions

This section shows REAL EXAMPLES from actual test runs where LLMs correctly solved Fredholm equations.

### Perfect Match: Simple Sine Solution (eq_1 from e2e_test)

**Equation:** `u(x) - 0.5 * ∫K(x,t)*u(t) dt = sin(x)`

**Ground Truth:** `sin(x)`

**LLM Prediction:** `sin(x)` ✅

**Solution Type:** `exact_symbolic` ✅

**Success Rate:** 100/100

### Perfect Match: Constant Family Solution (test100_family_8)

**Equation:** `u(x) - 1.001 * ∫u(t) dt = 0`

**Ground Truth:** `C` (arbitrary constant)

**LLM Prediction:** `c` ✅

**Solution Type:** `family` ✅

**Success Rate:** 100/100

### Excellent Match: Separable Kernel Family (test100_family_2)

**Equation:** `u(x) - 2.001 * ∫sin(πt)sin(πx)*u(t) dt = 0`

**Ground Truth:** `C*sin(πx)`

**LLM Prediction:** `C*sin(πx)` ✅

**Solution Type:** `family` ✅

**Success Rate:** 95/100

### Close Approximation: Constant Solution (test100_exact_symbolic_18)

**Equation:** `u(x) - 0.742 * ∫cos(t)*u(t) dt = -11.498`

**Ground Truth:** `-17.2108896628`

**LLM Prediction:** `-17.857` ✅ (96% accurate)

**Solution Type:** `exact_symbolic` ✅

**Success Rate:** 84/100

---

## General Input/Output Format

### What LLMs Receive (Input)

```
Solve the following Fredholm integral equation:

u(x) - λ * ∫_a^b K(x,t) * u(t) dt = f(x)

Domain: [a, b]
Format: [infix/latex/rpn]
```

### What LLMs Should Provide (Output)

```
SOLUTION: u(x) = [symbolic expression]
HAS_SOLUTION: [yes/no]
SOLUTION_TYPE: [exact_symbolic/approx_coef/discrete_points/series/family/regularized/none]
[Optional: REASONING: explanatory text]
```

---

## Solution Types

### 1. `exact_symbolic` - Closed-Form Symbolic Solution

**Definition:** Complete analytical solution, fully specified with no unknowns. Can be a function of x OR a constant.

**Format:**
- ✅ `sin(x) + x**2` (depends on x)
- ✅ `exp(-x)*cos(x)` (depends on x)
- ✅ `-17.21` or `C` (constant - no x dependence)
- ❌ NOT point values like `u(0.5) = 1.234` (only at one point, not a complete solution)

**Note on Constants:** Some Fredholm equations have pure constant solutions with no x dependence. These are still `exact_symbolic` when they fully solve the equation.

**Example (function):**
```
SOLUTION: u(x) = sin(x)
HAS_SOLUTION: yes
SOLUTION_TYPE: exact_symbolic
```

**Example (constant):**
```
SOLUTION: u(x) = -17.21
HAS_SOLUTION: yes
SOLUTION_TYPE: exact_symbolic
```

---

### 2. `approx_coef` - Approximate with Numeric Coefficients

**Definition:** Functional form with numeric (constant) parameters.

**Format:**
- ✅ `-1447.128*x**2 - 169.763*cosh(x**4)`
- ✅ `0.5*sin(x) + 0.3*cos(x) + 1.2*exp(-x)`
- ❌ NOT symbolic: `C*sin(x)`

**Evaluation:**
- Symbolic + numeric evaluation
- Extra per-term coefficient comparison is recorded for analysis

**Example:**
```
SOLUTION: u(x) ≈ -1447.128*x**2 - 169.763*cosh(x**4)
HAS_SOLUTION: yes
SOLUTION_TYPE: approx_coef
```

---

### 3. `discrete_points` - Solution at Discrete Points Only

**Definition:** Function values only at specific/discrete points (no continuous formula).

**Format:**
- ✅ Point list: `u(0) ≈ 1.23, u(0.5) ≈ 2.45, u(1) ≈ 0.89`
- ✅ Values from numerical table/grid
- ❌ NOT continuous function like `x**2`

**Example:**
```
SOLUTION: u(x) defined at points:
  x=0.0: u ≈ 1.234
  x=0.25: u ≈ 2.456
  x=0.5: u ≈ 1.789
HAS_SOLUTION: yes
SOLUTION_TYPE: discrete_points
```

---

### 4. `series` - Truncated Series Solution

**Definition:** Truncated series expansion (Neumann series, power series, etc.).

**Format:**
- ✅ `f(x) + lambda*K*f(x) + lambda**2*K**2*f(x) + lambda**3*K**3*f(x)` (exactly 4 terms)
- ✅ Explicit expanded terms (no sigma)
- ❌ NOT just `Σ` without explicit terms

**Evaluation:**
- Symbolic + numeric evaluation
- Extra term-by-term numeric RMSE is recorded for analysis

**Example:**
```
SOLUTION: u(x) = f(x) + 0.3*Integral(K(x,t)*f(t), (t, a, b)) + 0.09*Integral(Integral(K(x,s)*K(s,t)*f(t), (t, a, b)), (s, a, b)) + 0.027*Integral(Integral(Integral(K(x,r)*K(r,s)*K(s,t)*f(t), (t, a, b)), (s, a, b)), (r, a, b))
HAS_SOLUTION: yes
SOLUTION_TYPE: series
REASONING: Neumann series converges for |lambda| < 1/||K||
```

---

### 5. `family` - Family of Solutions with Arbitrary Parameters

**Definition:** Solution expressed with ARBITRARY (undetermined) parameters like c₁, c₂, C, A, B.

**Format:**
- ✅ `c_1*sin(x) + c_2*cos(x)` (basis coefficients)
- ✅ `x - 6130.173 - 8.366*C` (parameterized by C)
- ✅ `u_p(x) + C*sin(πx)` (particular + homogeneous with arbitrary C)
- ❌ NOT all numeric like `0.5*sin(x) + 0.3*cos(x)`

**Example:**
```
SOLUTION: u(x) = c_1*sin(x) + c_2*cos(x)
HAS_SOLUTION: yes
SOLUTION_TYPE: family
```

---

### 6. `regularized` - Ill-Posed Problem

**Definition:** First kind equation or singular kernel requiring regularization.

**Format:**
- ✅ Methodology: "Tikhonov regularization with α=0.01"
- ✅ Explain ill-posedness reason
- ❌ NOT an explicit u(x) formula

**Example:**
```
SOLUTION: Ill-posed problem. Apply Tikhonov regularization:
  min ||K*u - g||² + α||u||²
HAS_SOLUTION: yes
SOLUTION_TYPE: regularized
REASONING: Fredholm first kind; small perturbations cause large solution changes
```

---

### 7. `none` - No Solution Exists

**Definition:** Problem has no solution (mathematically impossible).

**Format:**
- ✅ Explain why: "Kernel rank deficient", "Incompatible RHS", etc.
- ❌ NOT guessing or claiming solution doesn't exist without reason

**Example:**
```
SOLUTION: No solution exists.
HAS_SOLUTION: no
SOLUTION_TYPE: none
REASONING: Kernel is rank deficient and RHS not in range of integral operator
```

---

## Summary Table

| Type | Format | Coefficients | Example |
|------|--------|--------------|---------|
| exact_symbolic | Function | Numeric constants | `x**2 + sin(x)` |
| approx_coef | Numeric params | Values | `-1447.128*x**2 + 0.567*cosh(x)` |
| discrete_points | Point values | x,y pairs | `(0,1.2), (0.5,2.4)` |
| series | Series terms | Coefficients | `term1 + term2 + term3 + term4` |
| family | Parameterized | Arbitrary C | `x - 100*C` |
| regularized | Method description | N/A | "Tikhonov with α=0.01" |
| none | Explanation | N/A | "No solution: ill-posed" |

---

## Common Mistakes

### ❌ Wrong: Numeric values instead of functions
```
BAD:  u(0) = 1.5, u(0.5) = 2.3  (for exact_symbolic)
GOOD: u(x) = x**2 + sin(x)
```

### ❌ Wrong: Mixing coefficient types
```
BAD:  u(x) = 0.5*sin(x) + c_2*cos(x)
GOOD (family): c_1*sin(x) + c_2*cos(x)
GOOD (approx_coef): 0.5*sin(x) + 0.3*cos(x)
```

### ❌ Wrong: Hallucinating solutions
```
BAD:  u(x) = sin(x)  (when has_solution should be "no")
GOOD: No solution exists because [mathematical reason]
```

---

## LLM Performance Statistics

Based on analysis of 28 prediction files (~2000+ equations):

| Solution Type | Success Rate | Best Cases |
|---|---|---|
| `family` | **90%+** ✅ | Homogeneous equations, separable kernels |
| `exact_symbolic` | **40-70%** ⚠️ | Constants, simple polynomials |
| `none` | **80%+** ✅ | Correctly identifies unsolvable cases |
| `series` | **<5%** ❌ | Rarely recognizes Neumann series |
| `approx_coef` | **10-30%** ❌ | Often misclassifies |

**Key Insights:**
- ✅ **Best:** Family solutions with separable kernels
- ⚠️ **Moderate:** Simple constants and polynomials
- ❌ **Poor:** Complex nested functions, series expansions

---

## Real LLM Predictions: Input/Output Examples

This section contains actual examples from test runs showing complete prompts and LLM responses for **correctly predicted equations**.

### Example 1: `exact_symbolic` - Perfect Match

**Equation ID:** eq_1 (from e2e_test)

**LLM Input (Full Prompt):**
```
You are an expert mathematician specializing in integral equations.
Given a Fredholm integral equation, find the solution u(x).

The equation may be of the second kind:
  u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)

Or of the first kind (ill-posed, requires regularization):
  ∫_a^b K(x, t) u(t) dt = g(x)

**IMPORTANT**: Express your solution in infix notation (e.g., x**2 + sin(x), exp(-x)*cos(x)).

Provide your answer in the following format:
SOLUTION: u(x) = [your solution here]
HAS_SOLUTION: [yes/no]
SOLUTION_TYPE: [exact_symbolic/approx_coef/discrete_points/series/family/regularized/none]

Solve the following Fredholm integral equation:

u(x) - 0.5 * ∫_0^1 sin(x) * u(t) dt = sin(x)

Domain: [0, 1]

Provide the solution u(x).
```

**Ground Truth:**
- Solution: `sin(x)`
- Solution Type: `exact_symbolic`
- Has Solution: `yes`

**LLM Output (Raw Response):**
```
To solve the given Fredholm integral equation of the second kind:

u(x) - 0.5 * ∫_0^1 sin(x) * u(t) dt = sin(x)

Let me rearrange this:
u(x) = 0.5 * sin(x) * ∫_0^1 u(t) dt + sin(x)

The integral ∫_0^1 u(t) dt is a constant. If u(x) = sin(x), then:
∫_0^1 sin(t) dt = [−cos(t)]_0^1 = −cos(1) + 1 ≈ 0.46

Let me verify: u(x) = 0.5 * sin(x) * 0.46 + sin(x)

This suggests u(x) = sin(x) is the solution.

Let me verify directly: If u(x) = sin(x), then:
u(x) - 0.5 * sin(x) * ∫_0^1 sin(t) dt = sin(x) - 0.5 * sin(x) * (1 - cos(1)) = sin(x)

Therefore:

SOLUTION: u(x) = sin(x)
HAS_SOLUTION: yes
SOLUTION_TYPE: exact_symbolic
```

**LLM Prediction:**
- Solution: `sin(x)` ✅
- Solution Type: `exact_symbolic` ✅
- Has Solution: `yes` ✅

**Match Score:** 100/100 (PERFECT)

---

### Example 2: `family` - Perfect Match

**Equation ID:** test100_family_8 (from test_100)

**LLM Input (Full Prompt):**
```
You are an expert mathematician specializing in integral equations.
Given a Fredholm integral equation, find the solution u(x).

Solve the following Fredholm integral equation:

u(x) - 1.001 * ∫_6.330207264818775^7.346979435881526 1 * u(t) dt = 0

Domain: [6.330207264818775, 7.346979435881526]

**IMPORTANT**: Express your solution in LaTeX notation.

Provide your answer in the following format:
SOLUTION: u(x) = [your solution here]
HAS_SOLUTION: [yes/no]
SOLUTION_TYPE: [family/exact_symbolic/none/...]
```

**Ground Truth:**
- Solution: `C` (arbitrary constant)
- Solution Type: `family`
- Has Solution: `yes`

**LLM Output (Raw Response, Summarized):**
```
To solve the given Fredholm integral equation of the second kind:

u(x) - 1.001 ∫_{6.330207264818775}^{7.346979435881526} u(t) dt = 0

This indicates that u(x) is constant for all x in the interval.

Let's denote the integral as C:
C = ∫_{6.330207264818775}^{7.346979435881526} u(t) dt

Since u(x) is constant, u(t) = k for some constant k.

Thus: C = k · (7.346979435881526 - 6.330207264818775) ≈ k · 1.016772

Substituting back:
u(x) = 1.001 C = 1.001 · k · 1.016772

This shows that u(x) can take any constant value, leading to a family of solutions.

Therefore:

SOLUTION: u(x) = c
HAS_SOLUTION: yes
SOLUTION_TYPE: family
```

**LLM Prediction:**
- Solution: `c` ✅ (matches symbolic form `C`)
- Solution Type: `family` ✅
- Has Solution: `yes` ✅

**Match Score:** 100/100 (PERFECT)

---

### Example 3: `approx_coef` - Excellent Match

**Equation ID:** test100_approx_coef_14 (from test_100_gpt4o)

**LLM Input (Equation Part):**
```
u(x) - (-0.787...) * ∫_{-0.207...}^{4.282...} sin(x) * cos(t) * u(t) dt = 
  - tanh(100.0*x - 428.288...) / 2 + tanh(100.0*x + 20.788...) / 2
```

**Ground Truth:**
- Solution: `- tanh(100.0*x - 428.288...) / 2 + tanh(100.0*x + 20.788...) / 2`
- Solution Type: `approx_coef`
- Has Solution: `yes`

**LLM Output (Summarized):**
```
This is a linear integral equation with a separable kernel K(x,t) = sin(x)*cos(t).

Given the complexity, an exact symbolic solution is unlikely. Instead, 
we can express the solution in terms of an approximate method with 
coefficients determined numerically...

SOLUTION: u(x) ≈ - tanh(100.0*x - 428.288...) / 2 + tanh(100.0*x + 20.788...) / 2
HAS_SOLUTION: yes
SOLUTION_TYPE: approx_coef
```

**LLM Prediction:**
- Solution: `-tanh(100.0*x - 428.288...) / 2 + tanh(100.0*x + 20.788...) / 2` ✅
- Solution Type: `approx_coef` ✅
- Has Solution: `yes` ✅

**Match Score:** 110/100 (EXCELLENT - exact formula match)

---

### Example 4: `family` - High Similarity Match

**Equation ID:** test100_family_2 (from test_100)

**Equation:**
```
u(x) - 2.001 * ∫ sin(πt)*sin(πx)*u(t) dt = 0
```

**Ground Truth:**
- Solution: `C*sin(πx)`
- Solution Type: `family`

**LLM Output (Key Part):**
```
...The kernel K(x,t) = sin(πt)*sin(πx) is separable...
...Using eigenfunctions method:

u(x) = C*sin(πx)

where C is a constant determined by boundary conditions.

SOLUTION: u(x) = C*sin(πx)
HAS_SOLUTION: yes
SOLUTION_TYPE: family
```

**LLM Prediction:**
- Solution: `C*sin(πx)` ✅
- Solution Type: `family` ✅
- Has Solution: `yes` ✅

**Match Score:** 95/100 (EXCELLENT)

### ✅ SUCCESS Criteria

For **symbolic solution types** (exact_symbolic, approx_coef, series, family):
```
✅ PASS: Correct function/expression + correct solution_type
❌ FAIL: Wrong expression OR wrong solution_type
```

For **edge case types** (none, discrete_points, regularized):
```
✅ PASS: Correct solution_type identified (no symbolic solution to predict)
❌ FAIL: Wrong solution_type
```

### Example Outcomes

**Equation 1:** u(x) - λ∫sin(t)sin(x)u(t)dt = cos(x)
- Ground truth: `u(x) = sin(x)` [exact_symbolic]
- LLM predicts: `u(x) = sin(x)` [exact_symbolic] → ✅ SUCCESS
- LLM predicts: `u(x) = 2*sin(x)` [exact_symbolic] → ❌ FAIL (wrong expression)
- LLM predicts: `u(x) = c*sin(x)` [family] → ✅ CORRECT (matches exact_symbolic ground truth with arbitrary parameter)
- LLM predicts: `u(x) = 0.9*sin(x) + 0.1*cos(x)` [approx_coef] → ❌ FAIL (wrong type)

**Equation 2:** Equation with no solution
- Ground truth: "No solution" [none]
- LLM predicts: "No solution" [none] → ✅ SUCCESS (type correct, no formula needed)
- LLM predicts: `u(x) = sin(x)` [exact_symbolic] → ❌ FAIL (wrong type)

**Equation 3:** Integral equation requiring regularization  
- Ground truth: "Ill-posed, apply Tikhonov" [regularized]
- LLM predicts: "Requires Tikhonov regularization" [regularized] → ✅ SUCCESS (type correct)
- LLM predicts: `u(x) = some_formula` [exact_symbolic] → ❌ FAIL (ignored ill-posedness)
