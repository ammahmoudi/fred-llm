# Post-Processing Flow

This document covers how raw LLM responses are parsed into structured fields.

Related docs:
- [docs/EVALUATION_PIPELINE.md](docs/EVALUATION_PIPELINE.md)
- [docs/METRICS_REFERENCE.md](docs/METRICS_REFERENCE.md)

## Overview

The post-processing stage extracts solution text, parses SymPy expressions, and collects metadata.

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
```

---

## Complete Pipeline (All Formats)

```mermaid
flowchart TD
    LLM[Raw LLM Response
with reasoning text] --> MetaExtract[Metadata Extraction]

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

    Strategy1 --> |Available and not RPN| MV1[Targeted u x pattern]
    Strategy1 --> |Fallback| MV2[SOLUTION marker]
    Strategy1 --> |Fallback| MV3[Full response parse]

    MV1 --> MVSuccess{Success?}
    MV2 --> MVSuccess
    MV3 --> MVSuccess

    MVSuccess --> |Yes| MVResult[Extract expr, str
confidence = 0.8]
    MVSuccess --> |No| Strategy2

    Strategy2 --> RegexPat[Apply Regex Patterns]
    RegexPat --> |Priority 1| Pat1[SOLUTION u x]
    RegexPat --> |Priority 2| Pat2[Solution u x]
    RegexPat --> |Priority 3| Pat3[Generic u x]

    Pat1 --> Clean[_clean_expression]
    Pat2 --> Clean
    Pat3 --> Clean

    Clean --> SolStr[solution_str
extracted string]

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

    ParseSuccess --> |Yes| SymPyExpr[solution_sympy to SymPy Expr
confidence = 0.7-0.8]
    ParseSuccess --> |No| ParseFail[solution_sympy none
confidence = 0.0-0.3]

    SymPyExpr --> Output[Output Dictionary]
    ParseFail --> Output

    Output --> |Contains| RawResp[raw_response str]
    Output --> |Contains| SolStrOut[solution_str str]
    Output --> |Contains| SolSymOut[solution_sympy Expr]
    Output --> |Contains| HasSolOut[has_solution bool]
    Output --> |Contains| SolTypeOut[solution_type str]
    Output --> |Contains| ReasonOut[reasoning str]
    Output --> |Contains| ConfOut[confidence float]

    RawResp --> WriteJSON[Write to predictions.jsonl]
    SolStrOut --> WriteJSON
    SolSymOut --> WriteJSON
    HasSolOut --> WriteJSON
    SolTypeOut --> WriteJSON
    ReasonOut --> WriteJSON
    ConfOut --> WriteJSON
```

---

## Format-Specific Processing

### LaTeX Format

```mermaid
flowchart LR
    LaTeX[LaTeX String
x^2 + sin x] --> Try1[Try Math-Verify parse]

    Try1 --> |Available| MVParser[ANTLR grammar parser]
    MVParser --> MVSuccess[OK: SymPy expression
confidence = 0.8]

    Try1 --> |Unavailable/Failed| Fallback[Fallback _latex_to_infix]
    Fallback --> Replace1[sin to sin]
    Replace1 --> Replace2[caret two to **2]
    Replace2 --> Replace3[frac a b to a over b]
    Replace3 --> ParseExpr[parse_expr infix_str]
    ParseExpr --> InfixSuccess[OK: SymPy expression
confidence = 0.7]
```

### Infix Format

```mermaid
flowchart LR
    Infix[Infix String
x**2 + sin x] --> Direct[parse_expr directly]
    Direct --> SymPy[SymPy standard parser]
    SymPy --> Success[OK: SymPy expression
confidence = 0.7]
```

### RPN Format

```mermaid
flowchart LR
    RPN[RPN String
x 2 ^ x sin +] --> Convert[rpn_to_sympy]
    Convert --> Stack[Stack-based evaluation]

    Stack --> Step1[Push: x, 2]
    Step1 --> Step2[Pop 2, apply ^
Push: x**2]
    Step2 --> Step3[Push: x]
    Step3 --> Step4[Pop 1, apply sin
Push: sin x]
    Step4 --> Step5[Pop 2, apply +
Result x**2 + sin x]
    Step5 --> Result[OK: SymPy expression
confidence = 0.7]

    Convert -.->|Note| NoMV[Math-Verify cannot parse RPN]
```

---

## Location

- Main module: [src/llm/postprocess.py](src/llm/postprocess.py)
- Math-Verify adapter: [src/llm/math_verify_adapter.py](src/llm/math_verify_adapter.py)

## Function: parse_llm_output()

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
| raw_response | str | Input | Untouched full LLM response with reasoning |
| solution_str | str | Extraction | Extracted solution expression (e.g., -x**2 + sin(x)) |
| solution_sympy | SymPy | Parsing | Canonicalized SymPy expression object |
| has_solution | bool or None | Regex | Whether solution exists (parsed from YES/NO) |
| solution_type | str or None | Regex | Solution classification (exact_symbolic, series, family, etc.) |
| confidence | float | Strategy | Confidence score (0.8 for Math-Verify, 0.7 for SymPy, 0.3 for errors) |
| reasoning | str or None | Extraction | Extracted reasoning steps (if present) |

---

## Solution Extraction Strategies

### Strategy 1: Math-Verify (Primary, LaTeX and Infix)

```python
from src.llm.math_verify_adapter import extract_solution_from_response

mv_result = extract_solution_from_response(response)
if mv_result is not None:
    sympy_expr, raw_str = mv_result
```

Multi-strategy approach (most-specific first):

1. Targeted u(x) = line
2. Structured SOLUTION marker
3. Full response parse

Confidence: 0.8

---

### Strategy 2: Regex Fallback (All Formats)

```python
patterns = [
    r"^SOLUTION\s*:\s*u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$)",
    r"[Ss]olution[:\s]+u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$)",
    r"u\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\n|$|\.(?:\s|$))",
]
```

Cleaning steps:
- Trim whitespace and trailing periods
- Remove LaTeX delimiters ($...$, \(...\))
- Clean up reasoning fragments

Confidence: 0.7

---

### Strategy 3: RPN Support

RPN format: x 2 ^ x sin +

Process:
1. Regex extracts the RPN string
2. [src/data/formatters/rpn_formatter.py](src/data/formatters/rpn_formatter.py) converts RPN -> SymPy

Limitation: Math-Verify cannot parse RPN

```python
from src.data.formatters.rpn_formatter import rpn_to_sympy

sympy_expr = rpn_to_sympy("x 2 ^ x sin +")
```

---

## Expression Parsing

### LaTeX -> SymPy Conversion

Process ([src/llm/math_verify_adapter.py](src/llm/math_verify_adapter.py)):

```
LaTeX input: "x^{2} + \sin(x)"
    -> Math-Verify
    -> or custom _latex_to_infix
Infix form: "x**2 + sin(x)"
    -> parse_expr + transformations
SymPy: x**2 + sin(x)
```

Symbol dictionary (FREDHOLM_LOCAL_DICT):

```python
{
    "x": Symbol("x"),
    "t": Symbol("t"),
    "C": Symbol("C"),
    "c_1": Symbol("c_1"),
    "pi": sp.pi,
    "E": sp.E,
    "Integral": sp.Integral,
    "oo": sp.oo,
}
```

---

## Metadata Extraction

### has_solution Field

Extracted from YES/NO patterns in response.

Fallback logic:
- If HAS_SOLUTION missing and a solution was extracted -> infer True
- If HAS_SOLUTION missing and no solution found -> infer False

### solution_type Field

Classification of solution structure. Extracted from SOLUTION_TYPE marker:

| Type | Meaning | Example |
|------|---------|---------|
| exact_symbolic | Closed-form symbolic | u(x) = sin(x) |
| approx_coef | Approximate with coefficients | u(x) ~= a0 + a1*x + a2*x**2 |
| series | Truncated series | u(x) = term1 + term2 + term3 |
| family | Family of solutions (non-unique) | u(x) = C*sin(pi*x) |
| discrete_points | Solution at discrete points only | [(0.0, 1.2), (0.5, 2.4)] |
| regularized | Ill-posed, requires regularization | |
| none | No solution exists | |

---

## Confidence Scores

| Source | Score | Meaning |
|--------|-------|---------|
| Math-Verify extraction | 0.8 | Robust LaTeX parsing, high reliability |
| Regex + SymPy parsing | 0.7 | Successful extraction and validation |
| Fallback (errors ignored) | 0.3 | Partial parsing, validation failed |
| No extraction | 0.0 | Failed to extract any solution |

```python
if result["confidence"] >= 0.7:
    evaluate_solution(result)
else:
    logger.warning(f"Low confidence: {result['confidence']}")
```
