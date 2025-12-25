# Formatters Module

This module contains formatters for converting mathematical expressions between different formats.

## Structure

Each formatter is in its own file and inherits from `BaseFormatter`:

- **`base.py`** - Abstract base class defining the formatter interface
- **`latex_formatter.py`** - LaTeX mathematical notation (e.g., `x^2 + 2x + 1`)
- **`rpn_formatter.py`** - Reverse Polish Notation (e.g., `x 2 ^ 2 x * + 1 +`)
- **`infix_formatter.py`** - Standard infix notation (e.g., `x**2 + 2*x + 1`)
- **`python_formatter.py`** - Python code representation
- **`tokenized_formatter.py`** - Space-separated tokens
- **`fredholm_formatter.py`** - Complete Fredholm equation formatter with special tokens
- **`series_formatter.py`** - Series expansions (Taylor, Neumann)

## Usage

### Via FormatConverter

```python
from src.data.format_converter import FormatConverter

converter = FormatConverter()

# Convert infix to LaTeX
latex = converter.convert("x**2 + 2*x + 1", source_format="infix", target_format="latex")
# Result: "x^{2} + 2 x + 1"

# Convert to RPN
rpn = converter.convert("x**2 + 2*x + 1", source_format="infix", target_format="rpn")
# Result: "x 2 ^ 2 x * + 1 +"

# Convert to Python code
python_code = converter.convert("x**2 + sin(x)", source_format="infix", target_format="python")
# Result: "x**2 + math.sin(x)"

# Convert to tokenized format
tokenized = converter.convert("x**2 + 2*x", source_format="infix", target_format="tokenized")
# Result: "x ** 2 + 2 * x"
```

### Direct Formatter Usage

```python
from src.data.formatters import (
    LaTeXFormatter, 
    RPNFormatter, 
    PythonFormatter,
    TokenizedFormatter
)
import sympy as sp

# Create formatter instances
latex_fmt = LaTeXFormatter()
rpn_fmt = RPNFormatter()
python_fmt = PythonFormatter()
tokenized_fmt = TokenizedFormatter()

# Convert infix to SymPy expression
expr = sp.sympify("x**2 + 2*x + 1")

# Convert to different formats
latex = latex_fmt.from_sympy(expr)        # "x^{2} + 2 x + 1"
rpn = rpn_fmt.from_sympy(expr)            # "x 2 ^ 2 x * + 1 +"
python = python_fmt.from_sympy(expr)      # "x**2 + 2*x + 1"
tokenized = tokenized_fmt.from_sympy(expr) # "x ** 2 + 2 * x + 1"
```

### Format Examples

| Format | Example | Description |
|--------|---------|-------------|
| **Infix** | `x**2 + 2*x + 1` | Standard mathematical notation |
| **LaTeX** | `x^{2} + 2 x + 1` | LaTeX typesetting format |
| **RPN** | `x 2 ^ 2 x * + 1 +` | Reverse Polish Notation (postfix) |
| **Python** | `x**2 + 2*x + 1` | Python executable code |
| **Tokenized** | `x ** 2 + 2 * x + 1` | Space-separated tokens |

### Python Formatter Details

The Python formatter generates executable Python code:

```python
from src.data.formatters import PythonFormatter
import sympy as sp

python_fmt = PythonFormatter()

# Trigonometric functions
expr = sp.sympify("sin(x) + cos(x)")
python = python_fmt.from_sympy(expr)
# Result: "math.sin(x) + math.cos(x)"

# Complex expressions
expr = sp.sympify("exp(x**2) + log(x)")
python = python_fmt.from_sympy(expr)
# Result: "math.exp(x**2) + math.log(x)"
```

### Tokenized Formatter Details

The tokenized formatter adds spaces around all operators for easy parsing:

```python
from src.data.formatters import TokenizedFormatter
import sympy as sp

tokenized_fmt = TokenizedFormatter()

# Simple expression
expr = sp.sympify("x**2+2*x")
tokenized = tokenized_fmt.from_sympy(expr)
# Result: "x ** 2 + 2 * x"

# With functions
expr = sp.sympify("sin(x**2)")
tokenized = tokenized_fmt.from_sympy(expr)
# Result: "sin ( x ** 2 )"

# Parse back to SymPy
parsed = tokenized_fmt.to_sympy("x ** 2 + 2 * x")
# Result: SymPy expression x**2 + 2*x
```

## Fredholm Equation Formatter

The `FredholmEquationFormatter` formats complete Fredholm integral equations with all components (u, f, kernel, lambda, bounds) into formatted strings suitable for LLM input/output.

### Basic Usage

```python
from src.data.formatters.fredholm_formatter import FredholmEquationFormatter

# Create formatter with infix expressions
formatter = FredholmEquationFormatter(expression_format="infix")

# Format complete equation
equation = formatter.format_equation(
    u="u(x)",
    f="x**2 + 2*x",
    kernel="x*t",
    lambda_val="0.5",
    a="0",
    b="1"
)
# Result: "u(x) - 0.5 * ∫[0,1] (x*t) u(t) dt = x**2 + 2*x"

# Format with LaTeX expressions
latex_formatter = FredholmEquationFormatter(expression_format="latex")
latex_eq = latex_formatter.format_equation(
    u="u(x)",
    f="x**2",
    kernel="x*t",
    lambda_val="1",
    a="0",
    b="1"
)
# Result: "u(x) - 1 \int_{0}^{1} x t \, u(t) \, dt = x^{2}"

# Format with RPN expressions
rpn_formatter = FredholmEquationFormatter(expression_format="rpn")
rpn_eq = rpn_formatter.format_equation(
    u="u(x)",
    f="x + 1",
    kernel="x*t",
    lambda_val="1",
    a="0",
    b="1"
)
# Result: RPN format with equation structure
```

### Tokenized Equation Formatter

For LLM training, the `TokenizedEquationFormatter` adds special tokens to mark equation components:

```python
from src.data.formatters.fredholm_formatter import TokenizedEquationFormatter

tokenized = TokenizedEquationFormatter()
eq = tokenized.format_equation(
    u="u(x)",
    f="x**2",
    kernel="x*t",
    lambda_val="0.5",
    a="0",
    b="1"
)
# Result: "u ( x ) - <LAMBDA> 0.5 <INT> <LOWER> 0 <UPPER> 1 ( x * t ) u ( t ) dt <SEP> x**2"
```

Special tokens:
- `<LAMBDA>` - Marks the lambda coefficient
- `<INT>` - Marks the beginning of integral
- `<LOWER>` - Marks the lower bound
- `<UPPER>` - Marks the upper bound
- `<SEP>` - Separates left and right sides of equation

These special tokens help LLMs better understand equation structure by explicitly marking mathematical components.

## Series Expansion Formatters

For approximation-based solutions and iterative methods, we provide two specialized formatters.

### SeriesFormatter

Formats expressions as Taylor/Maclaurin series expansions:

```python
from src.data.formatters.series_formatter import SeriesFormatter
import sympy as sp

# Create formatter with order 5, expand around x=0
formatter = SeriesFormatter(order=5, x_var="x", x0=0)

# Taylor series for sin(x)
expr = sp.sin(sp.Symbol("x"))
series = formatter.from_sympy(expr)
# Result: "x - x**3/6 + x**5/120 + O(x**6)"

# Polynomial approximation (removes O() term)
poly = formatter.format_polynomial_approximation("exp(x)", degree=3)
# Result: "1 + x + x**2/2 + x**3/6"

# Taylor series for custom function
series = formatter.format_taylor_series("cos(x) + x**2", n_terms=4)
# Result: "1 - x**2/2 + x**2 + x**4/24 + O(x**5)"
```

### NeumannSeriesFormatter

Specialized for Fredholm equation Neumann series solutions (u = f + λKf + λ²K²f + ...):

```python
from src.data.formatters.series_formatter import NeumannSeriesFormatter

# Create formatter with 3 terms
formatter = NeumannSeriesFormatter(n_terms=3, include_symbolic=True)

# Format complete Neumann series
series = formatter.format_neumann_series(
    f="x**2",
    kernel="x*t",
    lambda_val=0.5,
    bounds=(0, 1)
)
# Result: "u(x) = x**2 + (0.5)^1 * ∫[0,1] K^1(x,t) f(t) dt + (0.5)^2 * ∫[0,1] K^2(x,t) f(t) dt + O(λ^3)"

# Format truncated solution (without O() notation)
truncated = formatter.format_truncated_solution(
    f="x",
    kernel="x*t",
    lambda_val=1.0,
    n_terms=3
)
# Result: "u(x) ≈ x + λ^1 * K^1(f) + λ^2 * K^2(f)"
```

**Use cases:**
- Iterative solution methods
- Convergence analysis
- Approximation-based solutions
- Educational demonstrations

## Canonicalization and Simplification

All formatters support a `simplify` parameter to canonicalize expressions for consistent formatting:

```python
from src.data.formatters import InfixFormatter
import sympy as sp

formatter = InfixFormatter()
expr = sp.sympify("x + x + 1")

# Without simplification
result1 = formatter.from_sympy(expr, simplify=False)
# Result: "x + x + 1"

# With simplification
result2 = formatter.from_sympy(expr, simplify=True)
# Result: "2*x + 1"
```

This is useful for:
- Creating consistent training data
- Normalizing equivalent expressions
- Reducing expression complexity

## Adding a New Formatter

1. Create a new file in this directory (e.g., `json_formatter.py`)
2. Inherit from `BaseFormatter`
3. Implement `format_name`, `to_sympy()`, and `from_sympy()` methods
4. Add to `__init__.py` and `format_converter.py`

Example:

```python
from src.data.formatters.base import BaseFormatter
import sympy as sp

class MyFormatter(BaseFormatter):
    @property
    def format_name(self) -> str:
        return "my_format"
    
    def to_sympy(self, expression: str) -> sp.Expr:
        # Convert your format to SymPy
        pass
    
    def from_sympy(self, expr: sp.Expr) -> str:
        # Convert SymPy to your format
        pass
```

## Architecture

All formatters use **SymPy as the intermediate canonical form**:

```
Source Format → SymPy Expression → Target Format
```

This design ensures:
- Consistent conversion between any two formats
- Single source of truth (SymPy)
- Easy addition of new formats (just implement two methods)
- Testability (can test each formatter independently)
