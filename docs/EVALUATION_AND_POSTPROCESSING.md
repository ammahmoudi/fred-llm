# Evaluation and Post-Processing Pipeline

Comprehensive guide to how LLM predictions are processed, evaluated, and scored in fred-llm.

## Overview

The pipeline consists of two main stages:

1. **Post-Processing** – Extract and parse raw LLM output into structured fields
2. **Evaluation** – Compare extracted solutions against ground truth using symbolic and numeric methods

```mermaid
flowchart TD
    Start[Raw LLM Response] --> PostProc[Post-Processing]
    PostProc --> Extract[Extract solution expression]
    PostProc --> Parse[Parse to SymPy]
    PostProc --> Meta[Extract metadata]
    PostProc --> Score[Store confidence scores]
    
    Extract --> PredJSON[Predictions JSONL]
    Parse --> PredJSON
    Meta --> PredJSON
    Score --> PredJSON
    
    PredJSON --> Eval[Evaluation]
    Eval --> Symbolic[Symbolic comparison]
    Eval --> Numeric[Numeric comparison]
    Eval --> Edge[Edge case metrics]
    Eval --> Aggregate[Aggregate metrics]
    
    Symbolic --> MetricsJSON[Metrics JSON]
    Numeric --> MetricsJSON
    Edge --> MetricsJSON
    Aggregate --> MetricsJSON
```

---

## Post-Processing Flow Diagram

### Complete Pipeline (All Formats)

```mermaid
flowchart TD
    LLM[Raw LLM Response<br/>with reasoning text] --> MetaExtract[Metadata Extraction]
    
    MetaExtract --> HasSol[Extract HAS_SOLUTION]
    MetaExtract --> SolType[Extract SOLUTION_TYPE]
    MetaExtract --> Reason[Extract REASONING]
    
    HasSol --> FormatDetect[Format Detection]
    SolType --> FormatDetect
    Reason --> FormatDetect
    
    FormatDetect --> |LaTeX| LaTeXPath[LaTeX Format]
    FormatDetect --> |Infix| InfixPath[Infix Format]
    FormatDetect --> |RPN| RPNPath[RPN Format]
    
    LaTeXPath --> Strategy1[Strategy 1: Math-Verify]
    InfixPath --> Strategy1
    RPNPath --> Strategy2[Strategy 2: Regex Fallback]
    
    Strategy1 --> |Available & Not RPN| MV1[① Targeted u&#40;x&#41;= pattern]
    Strategy1 --> |Fallback| MV2[② SOLUTION: marker]
    Strategy1 --> |Fallback| MV3[③ Full response parse]
    
    MV1 --> MVSuccess{Success?}
    MV2 --> MVSuccess
    MV3 --> MVSuccess
    
    MVSuccess --> |Yes| MVResult[Extract expr, str<br/>confidence = 0.8]
    MVSuccess --> |No| Strategy2
    
    Strategy2 --> RegexPat[Apply Regex Patterns]
    RegexPat --> |Priority 1| Pat1["SOLUTION: u&#40;x&#41; = ..."]
    RegexPat --> |Priority 2| Pat2["Solution: u&#40;x&#41; = ..."]
    RegexPat --> |Priority 3| Pat3["Generic u&#40;x&#41; = ..."]
    
    Pat1 --> Clean[_clean_expression]
    Pat2 --> Clean
    Pat3 --> Clean
    
    Clean --> SolStr[solution_str<br/>extracted string]
    
    MVResult --> ParseSymPy[Parse to SymPy]
    SolStr --> ParseSymPy
    
    ParseSymPy --> |LaTeX| MVParse[Math-Verify parse]
    ParseSymPy --> |LaTeX fallback| LatexInfix[_latex_to_infix + parse_expr]
    ParseSymPy --> |Infix| DirectParse[parse_expr directly]
    ParseSymPy --> |RPN| RPNConvert[rpn_to_sympy]
    
    MVParse --> ParseSuccess{Success?}
    LatexInfix --> ParseSuccess
    DirectParse --> ParseSuccess
    RPNConvert --> ParseSuccess
    
    ParseSuccess --> |Yes| SymPyExpr[solution_sympy = SymPy Expr<br/>confidence = 0.7-0.8]
    ParseSuccess --> |No| ParseFail[solution_sympy = None<br/>confidence = 0.0-0.3]
    
    SymPyExpr --> Output[Output Dictionary]
    ParseFail --> Output
    
    Output --> |Contains| RawResp[raw_response: str]
    Output --> |Contains| SolStrOut[solution_str: str]
    Output --> |Contains| SolSymOut[solution_sympy: Expr]
    Output --> |Contains| HasSolOut[has_solution: bool]
    Output --> |Contains| SolTypeOut[solution_type: str]
    Output --> |Contains| ReasonOut[reasoning: str]
    Output --> |Contains| ConfOut[confidence: float]
    
    RawResp --> WriteJSON[Write to predictions.jsonl]
    SolStrOut --> WriteJSON
    SolSymOut --> WriteJSON
    HasSolOut --> WriteJSON
    SolTypeOut --> WriteJSON
    ReasonOut --> WriteJSON
    ConfOut --> WriteJSON
    
    style MVResult fill:#90EE90
    style Strategy1 fill:#FFD700
    style Strategy2 fill:#FFA500
    style SymPyExpr fill:#90EE90
    style ParseFail fill:#FF6B6B
```

### Format-Specific Processing

#### LaTeX Format

```mermaid
flowchart LR
    LaTeX["LaTeX String<br/>x^2 + \sin&#40;x&#41;"] --> Try1[Try Math-Verify parse]
    
    Try1 --> |Available| MVParser[ANTLR grammar parser]
    MVParser --> MVSuccess[✓ SymPy expression<br/>confidence = 0.8]
    
    Try1 --> |Unavailable/Failed| Fallback[Fallback: _latex_to_infix]
    Fallback --> Replace1["\sin → sin"]
    Replace1 --> Replace2["^{2} → **2"]
    Replace2 --> Replace3["\frac{a}{b} → &#40;a&#41;/&#40;b&#41;"]
    Replace3 --> ParseExpr[parse_expr&#40;infix_str&#41;]
    ParseExpr --> InfixSuccess[✓ SymPy expression<br/>confidence = 0.7]
    
    style MVSuccess fill:#90EE90
    style InfixSuccess fill:#FFD700
```

#### Infix Format

```mermaid
flowchart LR
    Infix["Infix String<br/>x**2 + sin&#40;x&#41;"] --> Direct[parse_expr directly]
    Direct --> SymPy[SymPy standard parser]
    SymPy --> Success[✓ SymPy expression<br/>confidence = 0.7]
    
    style Success fill:#FFD700
```

#### RPN Format

```mermaid
flowchart TD
    RPN["RPN String<br/>x 2 ^ x sin +"] --> Convert[rpn_to_sympy]
    Convert --> Stack[Stack-based evaluation]
    
    Stack --> Step1["① Push: x, 2"]
    Step1 --> Step2["② Pop 2, Apply ^<br/>Push: x**2"]
    Step2 --> Step3["③ Push: x"]
    Step3 --> Step4["④ Pop 1, Apply sin<br/>Push: sin&#40;x&#41;"]
    Step4 --> Step5["⑤ Pop 2, Apply +<br/>Result: x**2 + sin&#40;x&#41;"]
    Step5 --> Result[✓ SymPy expression<br/>confidence = 0.7]
    
    Convert -.->|Note| NoMV[⚠️ Math-Verify<br/>cannot parse RPN]
    
    style Result fill:#FFD700
    style NoMV fill:#FF6B6B
```

---

## Part 1: Post-Processing Pipeline

### Location
- **Main module**: [src/llm/postprocess.py](src/llm/postprocess.py)
- **Math-Verify adapter**: [src/llm/math_verify_adapter.py](src/llm/math_verify_adapter.py)

### Function: `parse_llm_output()`

Extracts mathematical content from raw LLM response text.

```python
result = parse_llm_output(
    response="To solve...\nSOLUTION: u(x) = x**2 + sin(x)\n...",
    extract_solution=True,
    validate=True,
)
```

### Output Fields

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `raw_response` | str | Input | Untouched full LLM response with reasoning |
| `solution_str` | str | Extraction | Extracted solution expression (e.g., `-x**2 + sin(x)`) |
| `solution_sympy` | SymPy | Parsing | Canonicalized SymPy expression object |
| `has_solution` | bool \| None | Regex | Whether solution exists (parsed from YES/NO) |
| `solution_type` | str \| None | Regex | Solution classification (exact_symbolic, series, family, etc.) |
| `confidence` | float | Strategy | Confidence score (0.8 for Math-Verify, 0.7 for SymPy, 0.3 for errors) |
| `reasoning` | str \| None | Extraction | Extracted reasoning steps (if present) |

---

## Part 2: Solution Extraction Strategies

### Strategy 1: Math-Verify (Primary) ✓ LaTeX & Infix Only

**Available for**: LaTeX, Infix formats

Uses HuggingFace Math-Verify library for robust mathematical expression parsing.

```python
from src.llm.math_verify_adapter import extract_solution_from_response

mv_result = extract_solution_from_response(response)
if mv_result is not None:
    sympy_expr, raw_str = mv_result
    # raw_str = "-797089.48628811292*x + ..."
    # sympy_expr = SymPy expression (ready for comparison)
```

**Multi-strategy approach (most-specific first)**:

1. **Targeted `u(x) =` line** – Find last line matching pattern, parse with Math-Verify
   ```
   Last line: "u(x) = x**2 + sin(x)" → Parse RHS only
   ```

2. **Structured `SOLUTION:` marker** – Extract after `SOLUTION:` line
   ```
   SOLUTION: u(x) = x**2 + sin(x) → Parse content
   ```

3. **Full response parse** – Hand entire response to Math-Verify, find `Eq(u(x), rhs)`
   - Detects scrambled text (e.g., "No solution" parsed as `n*o*s*o*l*u*t*i*o*n`)
   - Filters out nonsense expressions

**Confidence**: 0.8

---

### Strategy 2: Regex Fallback (Legacy)

**Available for**: All formats

When Math-Verify unavailable or returns None, use regex patterns:

```python
patterns = [
    r"^SOLUTION\s*:\s*u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$)",  # SOLUTION: u(x) = ...
    r"[Ss]olution[:\s]+u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$)",    # Solution: u(x) = ...
    r"u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$|\.(?:\s|$))",          # Generic u(x) = ...
]
```

**Cleaning steps**:
- Trim whitespace and trailing periods
- Remove LaTeX delimiters (`$...$`, `\(...\)`)
- Clean up reasoning fragments

**Confidence**: 0.7

---

### Strategy 3: RPN Support

**Available for**: RPN (Reverse Polish Notation)

RPN format: `x 2 ^ x sin +` (postfix notation)

**Process**:
1. Regex extracts the RPN string
2. [rpn_formatter.py](src/data/formatters/rpn_formatter.py) converts RPN → SymPy

**Limitation**: Math-Verify cannot parse RPN (requires LaTeX/infix first)

**Workaround**: Convert RPN to infix before evaluation:
```python
from src.data.formatters.rpn_formatter import rpn_to_sympy

sympy_expr = rpn_to_sympy("x 2 ^ x sin +")
infix_str = str(sympy_expr)  # "x**2 + sin(x)"
```

---

## Part 3: Expression Parsing

### LaTeX → SymPy Conversion

**Process** ([math_verify_adapter.py](src/llm/math_verify_adapter.py)):

```
LaTeX input: "x^{2} + \sin(x)"
    ↓ (Math-Verify)
    ↓ (or custom _latex_to_infix)
Infix form: "x**2 + sin(x)"
    ↓ (parse_expr + TRANSFORMATIONS)
SymPy: x**2 + sin(x)
```

**Symbol Dictionary** (`FREDHOLM_LOCAL_DICT`):

```python
{
    "x": Symbol('x'),
    "t": Symbol('t'),
    "C": Symbol('C'),  # Constants
    "c_1": Symbol('c_1'),  # Subscripted constants
    "pi": π,
    "e": ℯ,
    "Integral": sp.Integral,
    "oo": sp.oo,
}
```

---

## Part 4: Metadata Extraction

### `has_solution` Field

Extracted from YES/NO patterns in response:

```python
def _extract_has_solution(response: str) -> bool | None:
    """Extract HAS_SOLUTION: [yes/no] from response."""
    # Looks for patterns like:
    # - HAS_SOLUTION: yes
    # - SOLUTION_EXISTS: no
    # - "No solution exists"
```

**Fallback logic**:
- If `HAS_SOLUTION` field is missing AND solution was extracted → infer `True`
- If `HAS_SOLUTION` field is missing AND no solution found → infer `False`

---

### `solution_type` Field

Classification of solution structure. Extracted from `SOLUTION_TYPE:` marker:

| Type | Meaning | Example |
|------|---------|---------|
| `exact_symbolic` | Closed-form symbolic | `u(x) = sin(x)` |
| `approx_coef` | Approximate with coefficients | `u(x) ≈ a₀ + a₁x + a₂x²` |
| `series` | Truncated series | `u(x) = term1 + term2 + term3 + term4` |
| `family` | Family of solutions (non-unique) | `u(x) = C·sin(πx)` |
| `discrete_points` | Solution at discrete points only | `[(0.0, 1.2), (0.5, 2.4)]` |
| `regularized` | Ill-posed, requires regularization | |
| `none` | No solution exists | |

---

## Part 5: Evaluation Pipeline

### Location
- **Main module**: [src/llm/evaluate.py](src/llm/evaluate.py)

### Function: `evaluate_solutions()`

Compares extracted solution against ground truth across multiple metrics.

```python
metrics = evaluate_solutions(
    predictions=[
        {
            "solution_sympy": sp.sympify("x**2"),
            "ground_truth": sp.sympify("x**2"),
            "has_solution": True,
            "solution_type": "exact_symbolic",
            ...
        },
        ...
    ],
    domains=[(-2, 2), ...],
)
```

---

### Comparison Methods

#### 1. Symbolic Comparison (`symbolic_compare()`)

Tests whether two expressions are **mathematically equivalent**.

**Process**:

```python
def symbolic_compare(solution, ground_truth):
    # Step 1: Math-Verify fast-path (if available)
    if HAS_MATH_VERIFY:
        mv_result = math_verify_compare(solution, ground_truth)
        if mv_result is True:
            return {"equivalent": True, "simplified_match": True}
    
    # Step 2: SymPy simplification strategies
    # Try direct equality
    if simplify(solution - ground_truth) == 0:
        return {"equivalent": True}
    
    # Try trig simplification
    if trigsimp(solution - ground_truth) == 0:
        return {"equivalent": True}
    
    # Try expansion
    if expand(solution - ground_truth) == 0:
        return {"equivalent": True}
    
    return {"equivalent": False}
```

**Metric**: `symbolic_accuracy` – % of solutions matching ground truth

---

#### 2. Numeric Comparison (`numeric_compare()`)

Tests equivalence by **sampling domain points**.

**Process**:

```python
def numeric_compare(solution, ground_truth, domain=(-1, 1), n_points=100):
    # Sample n_points uniformly across domain
    # Evaluate both expressions at each point
    # Check if |solution(x) - ground_truth(x)| < tolerance
    
    # Examples:
    # x**2 vs x*x → All points match ✓
    # x**3 vs x**2 → Most points differ ✗
```

**Metric**: `numeric_accuracy` – % of sampled points within tolerance

**Note:** If `evaluation_points` are present in the prediction metadata, numeric comparison uses those stored points instead of generating new samples.

---

### Type-Specific Evaluators

Additional evaluators run when `solution_type` matches and record metadata in each prediction's `evaluation` field.

#### `discrete_points`
- Point-wise comparison with x/y tolerances
- Metrics: `matched_points`, `accuracy`, `max_error`, `mean_error`, `rmse`

#### `series`
- Term-by-term numeric RMSE over top-level terms
- Metadata: `series_term_eval` + term count stats

#### `approx_coef`
- Per-term coefficient comparison
- Metadata: `approx_coef_eval` + aggregated stats

#### `family`
- Multi-sample numeric comparison for free constants
- Term-by-term numeric RMSE after substitution
- Metadata: `family_param_eval` (parameter count + naming)

---

### Edge Case Metrics

#### `has_solution_accuracy`

Compares `has_solution` (bool) against `ground_truth_has_solution`:

```python
if predicted["has_solution"] == ground_truth["has_solution"]:
    correct += 1
```

**Metric**: % of correct YES/NO predictions

---

#### `solution_type_accuracy`

Compares `solution_type` (string classification) against `ground_truth_solution_type`:

```python
if predicted["solution_type"] == ground_truth["solution_type"]:
    correct += 1
```

**Metric**: % of correct type classifications

---

### Metrics Output

#### `metrics_*.json`

Aggregated evaluation results across all predictions:

```json
{
  "total": 100,
  "correct": 12,
  "accuracy": 0.12,                      // Overall symbolic accuracy
  "symbolic_accuracy": 0.03,             // % matching via simplification
  "numeric_accuracy": 0.03,              // % matching via sampling
  
  "per_type": {                          // Breakdown by solution_type
    "exact_symbolic": {
      "total": 25,
      "correct": 0,
      "accuracy": 0.0
    },
    "series": { "total": 6, ... },
    "family": { "total": 10, ... },
    ...
  },
  
  "has_solution_accuracy": 0.66,        // Edge case: YES/NO correctness
  "solution_type_accuracy": 0.07,       // Edge case: type classification
  
  "api_errors": 0,                      // Failed API calls
  "parse_errors": 10,                   // Failed extractions
}
```

---

### Predictions Output

#### `predictions_*.jsonl`

Each line is one test case (JSON objects separated by newlines):

```jsonl
{
  "equation_id": "test100_series_2",
  "prompt": "You are an expert...",
  "ground_truth": "- 797089.48628811292 x + ...",
  
  "raw_response": "To solve...\nSOLUTION: u(x) = ...",
  "api_error": false,
  
  "solution_str": "-797089.48628811292*x + ...",
  "solution_sympy": "-797089.48628811292*x + cosh(x) - ...",
  "has_solution": false,
  "solution_type": "none",
  "reasoning": "This equation...",
  "confidence": 0.8
}
```

#### `predictions_evaluated_*.jsonl`

When running the adaptive pipeline, an evaluated file is also emitted with per-case metrics:

```jsonl
{
    "equation_id": "test100_series_2",
    "solution_type": "series",
    "solution_str": "term1 + term2 + term3 + term4",
    "ground_truth": "term1 + term2 + term3 + term4",
    "evaluation": {
        "correct": true,
        "symbolic_match": false,
        "numeric_match": true,
        "series_term_eval": {
            "term_rmse": [0.0, 0.01, 0.02, 0.03]
        },
        "numeric": {
            "x_values": [0.0, 0.1],
            "y_pred": [0.0, 0.1],
            "y_true": [0.0, 0.1],
            "points_source": "evaluation_points"
        }
    }
}
```

**Reading predictions**:

```python
import json

with open("outputs/test_100/predictions_20260206_120237.jsonl") as f:
    for line in f:
        data = json.loads(line)
        print(f"{data['equation_id']}: {data['solution_str']}")
        print(f"  Ground truth: {data['ground_truth']}")
        print(f"  Type: {data['solution_type']}")
```

---

## Part 6: Format Support

### Infix Format

**Example**: `x**2 + sin(x)`

**Prompt instruction**:
```
Express your solution in infix notation (e.g., x**2 + sin(x), exp(-x)*cos(x)).
```

**Math-Verify**: ✓ Supported

**SymPy**: ✓ Native format

---

### LaTeX Format

**Example**: `x^2 + \sin(x)` or `x^{2} + \sin(x)`

**Prompt instruction**:
```
Express your solution in LaTeX notation (e.g., x^2 + \sin(x), e^{-x}\cos(x)).
```

**Math-Verify**: ✓ Fully supported (primary use case)

**SymPy**: Requires conversion via `_latex_to_infix()`

---

### RPN Format

**Example**: `x 2 ^ x sin +` (postfix notation)

**Prompt instruction**:
```
Express your solution in Reverse Polish Notation (e.g., x 2 ^ x sin +, x neg exp x cos *).
```

**Math-Verify**: ✗ Not supported (requires LaTeX/infix)

**SymPy**: Via [rpn_formatter.py](src/data/formatters/rpn_formatter.py)

**Workaround** (if Math-Verify needed):
```python
from src.data.formatters.rpn_formatter import rpn_to_sympy

rpn_solution = "x 2 ^ x sin +"
sympy_expr = rpn_to_sympy(rpn_solution)
infix_str = str(sympy_expr)  # "x**2 + sin(x)"
# Now can use with Math-Verify
```

---

## Part 7: Confidence Scores

| Source | Score | Meaning |
|--------|-------|---------|
| Math-Verify extraction | 0.8 | Robust LaTeX parsing, high reliability |
| Regex + SymPy parsing | 0.7 | Successful extraction and validation |
| Fallback (errors ignored) | 0.3 | Partial parsing, validation failed |
| No extraction | 0.0 | Failed to extract any solution |

```python
if result["confidence"] >= 0.7:
    # Safe to use for evaluation
    evaluate_solution(result)
else:
    logger.warning(f"Low confidence: {result['confidence']}")
```

---

## Part 8: Text Similarity Metrics (BLEU/ROUGE)

Currently **NOT computed** during evaluation. To extract solution pairs for text metrics:

```python
import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

with open("predictions.jsonl") as f:
    for line in f:
        data = json.loads(line)
        
        # Extract clean solutions from raw response
        predicted = data["raw_response"]
        reference = data["ground_truth"]
        
        # Compute BLEU
        bleu = sentence_bleu(
            [reference.split()],
            predicted.split()
        )
        
        # Compute ROUGE-L
        scorer = rouge_scorer.RougeScorer(['rougeL'])
        rouge = scorer.score(reference, predicted)
```

---

## Part 9: Common Issues & Troubleshooting

### Issue: `has_solution` is None

**Cause**: Response didn't contain explicit YES/NO marker

**Fix**: 
- Prompt more clearly asks for `HAS_SOLUTION: [yes/no]`
- Post-processing infers from presence of solution expression

### Issue: `solution_sympy` is None but `solution_str` exists

**Cause**: Expression couldn't be parsed to SymPy (syntax error, unknown symbols)

**Fix**:
- Check `raw_response` for malformed LaTeX
- Add symbol to `FREDHOLM_LOCAL_DICT` if domain-specific

### Issue: Math-Verify returns None

**Cause**: 
- Expression not in LaTeX/infix format (e.g., RPN)
- Malformed LaTeX
- Module not installed (`HAS_MATH_VERIFY = False`)

**Fix**:
- Install: `pip install math-verify[antlr4_13_2]`
- Convert RPN to infix first
- Check LaTeX syntax

### Issue: `accuracy` very low (< 5%)

**Cause**: 
- Solutions are symbolically correct but not matching simplification
- Numeric evaluation with wrong domain
- Ground truth format mismatch

**Debug**:
```python
# Check a specific case
data = json.loads(predictions_line)
print(f"Extracted: {data['solution_str']}")
print(f"Ground truth: {data['ground_truth']}")
print(f"SymPy extracted: {data['solution_sympy']}")

# Try manual symbolic comparison
from src.llm.evaluate import symbolic_compare
result = symbolic_compare(
    sp.sympify(data['solution_str']),
    sp.sympify(data['ground_truth'])
)
print(f"Match: {result['equivalent']}")
```

---

## Part 10: Running Evaluation

### Full Pipeline

```bash
# 1. Generate predictions
python -m src.cli run --config configs/test_100.yaml

# 2. Outputs created:
# - outputs/test_100/predictions_TIMESTAMP.jsonl
# - outputs/test_100/metrics_TIMESTAMP.json
# - outputs/test_100/cost_summary_TIMESTAMP.json
```

### Evaluate Existing Predictions

```python
from src.llm.evaluate import evaluate_solutions
import json

predictions = []
with open("outputs/test_100/predictions_20260206_120237.jsonl") as f:
    for line in f:
        predictions.append(json.loads(line))

metrics = evaluate_solutions(predictions)
print(f"Overall accuracy: {metrics['accuracy']:.1%}")
print(f"Symbolic accuracy: {metrics['symbolic_accuracy']:.1%}")
```

---

## References

- [Math-Verify Library](https://github.com/openai/math-verify)
- [SymPy Documentation](https://docs.sympy.org/)
- [LaTeX Math Reference](https://en.wikibooks.org/wiki/LaTeX/Mathematics)
- Project docs: [FEATURES.md](FEATURES.md), [QUICKSTART.md](QUICKSTART.md)
