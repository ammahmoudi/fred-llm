"""Build code-representation prompts for the code-prompt experiment.

Reads the basic test_100_v2 prompts jsonl (same equations, metadata, ground
truth) and re-renders each equation as executable SymPy definitions. Emits two
prompt sets:

- ``code``: the model sees the Python representation and answers directly in
  the standard SOLUTION:/HAS_SOLUTION:/SOLUTION_TYPE: format.
- ``code_exec``: the model must reply with ONE python code block; the
  CodeExecModelRunner executes it and the script's printed output feeds the
  unchanged postprocess/evaluate pipeline.

Also emits a diverse 8-item pilot subset (one per solution type + 1 extra).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

SRC = REPO / "data/prompts/basic/test_100_v2/test_100_samples.jsonl"

# Verbatim from the basic prompts so parse_llm_output applies identically.
FORMAT_BLOCK = """Provide your answer in the following format:
SOLUTION: u(x) = [your solution here]
HAS_SOLUTION: [yes/no]
SOLUTION_TYPE: [exact_symbolic/exact_coef/approx_coef/discrete_points/series/family/regularized/none]

For SOLUTION_TYPE:
- exact_symbolic: Closed-form symbolic solution (e.g., u(x) = sin(x))
- exact_coef: Exact with unknown coefficients (e.g., u(x) = c₁sin(x) + c₂cos(x))
- approx_coef: Approximate with coefficients (e.g., u(x) ≈ a₀ + a₁x + a₂x²)
- discrete_points: Solution only at discrete points
- series: Infinite series solution (e.g., u(x) = Σ aₙxⁿ)
- family: Family of solutions (non-unique)
- regularized: Ill-posed, requires regularization
- none: No solution exists

If no solution exists, write "No solution" for SOLUTION.
"""


def code_block(md: dict) -> str:
    """Render the equation as executable SymPy definitions."""
    from src.llm.math_verify_adapter import parse_latex_to_sympy

    kernel = parse_latex_to_sympy(md["kernel"])
    f = parse_latex_to_sympy(md["f"])
    a, b = md["domain"]
    lam = md["lambda_val"]
    if lam == 0:  # first kind (ill-posed): integral term only, no u(x) outside
        equation = (
            "# Fredholm integral equation of the FIRST kind (ill-posed):\n"
            "#   Integral(K * u(t), (t, a, b)) = f"
        )
    else:
        equation = (
            "# Fredholm integral equation of the second kind:\n"
            "#   u(x) - lam * Integral(K * u(t), (t, a, b)) = f"
        )
    return f"""```python
from sympy import *

x, t = symbols('x t')

lam = {lam!r}
a, b = {a!r}, {b!r}  # integration domain
K = {kernel}  # kernel K(x, t)
f = {f}  # right-hand side f(x)

{equation}
```"""


def build_code_prompt(md: dict) -> str:
    return f"""You are an expert computational mathematician and Python/SymPy programmer.
The following Python code defines a Fredholm integral equation to solve for u(x):

{code_block(md)}

Solve for u(x). Approach this as a computational problem: mentally execute the
SymPy steps you would code (e.g., for a separable kernel K = sum_i g_i(x)*h_i(t),
reduce to a linear system for the coefficients c_i = Integral(h_i(t)*u(t), (t, a, b))
and solve it exactly; check singular systems for family/none cases).

**IMPORTANT**: Express your solution in LaTeX notation (e.g., x^2 + \\sin(x), e^{{-x}}\\cos(x)).

{FORMAT_BLOCK}"""


def build_code_exec_prompt(md: dict) -> str:
    return f"""You are an expert computational mathematician and Python/SymPy programmer.
The following Python code defines a Fredholm integral equation to solve for u(x):

{code_block(md)}

Write ONE complete, self-contained Python 3 script that solves this equation
for u(x) using SymPy and prints the answer. The script will be executed as-is.

Requirements:
- Standalone script; sympy, numpy and scipy are available.
- It must finish within 60 seconds. The coefficients are large floats: prefer
  the separable-kernel linear system (solve for the integral coefficients
  exactly) over brute-force solve/simplify on huge expressions.
- Decide the outcome programmatically: a nonsingular system gives a unique
  solution; singular-but-consistent means a one-parameter solution family;
  singular-and-inconsistent means no solution; a first-kind equation needs
  regularization (or report the least-squares/regularized solution).
- The LAST lines the script prints must be exactly this format, with the
  solution expressed in LaTeX (use sympy.latex()):
SOLUTION: u(x) = [solution in LaTeX, or "No solution"]
HAS_SOLUTION: [yes/no]
SOLUTION_TYPE: [exact_symbolic/exact_coef/approx_coef/discrete_points/series/family/regularized/none]

For SOLUTION_TYPE:
- exact_symbolic: Closed-form symbolic solution (e.g., u(x) = sin(x))
- exact_coef: Exact with unknown coefficients (e.g., u(x) = c₁sin(x) + c₂cos(x))
- approx_coef: Approximate with coefficients (e.g., u(x) ≈ a₀ + a₁x + a₂x²)
- discrete_points: Solution only at discrete points
- series: Infinite series solution (e.g., u(x) = Σ aₙxⁿ)
- family: Family of solutions (non-unique)
- regularized: Ill-posed, requires regularization
- none: No solution exists

Reply with ONLY a single ```python code block. No text outside it.
"""


def main() -> None:
    rows = [json.loads(line) for line in SRC.read_text().splitlines()]
    failures = []
    out = {"code": [], "code_exec": []}
    for row in rows:
        try:
            block = code_block(row["metadata"])  # noqa: F841 — parse check
        except Exception as e:
            failures.append((row["equation_id"], str(e)[:100]))
            continue
        for style, builder in (
            ("code", build_code_prompt),
            ("code_exec", build_code_exec_prompt),
        ):
            out[style].append(
                {
                    "equation_id": row["equation_id"],
                    "prompt": builder(row["metadata"]),
                    "style": style,
                    "format_type": "latex",
                    "ground_truth": row["ground_truth"],
                    "metadata": row["metadata"],
                }
            )

    # Diverse pilot: first item of each solution type + 1 extra exact_symbolic
    def pilot_subset(items: list[dict]) -> list[dict]:
        picked, seen = [], set()
        for it in items:
            st = it["metadata"]["solution_type"]
            if st not in seen:
                seen.add(st)
                picked.append(it)
        extra = [
            it
            for it in items
            if it["metadata"]["solution_type"] == "exact_symbolic"
            and it not in picked
        ]
        return picked + extra[:1]

    for style, items in out.items():
        for name, subset in (
            ("test_100_v2", items),
            ("pilot_8", pilot_subset(items)),
        ):
            d = REPO / f"data/prompts/{style}/{name}"
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{name}_samples.jsonl"
            path.write_text("".join(json.dumps(it) + "\n" for it in subset))
            print(f"wrote {len(subset):>3} prompts -> {path.relative_to(REPO)}")

    if failures:
        print(f"\nPARSE FAILURES ({len(failures)}):")
        for eq_id, err in failures:
            print(f"  {eq_id}: {err}")
    else:
        print("\nall equations parsed to sympy code")


if __name__ == "__main__":
    main()
