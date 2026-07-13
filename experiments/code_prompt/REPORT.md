# Code-Prompt: Program-of-Thought Solving for Fredholm Integral Equations

**Hypothesis.** Frontier models are stronger programmers than mathematicians.
Render each Fredholm equation as executable Python/SymPy definitions instead of
math notation, and (a) let the model reason over the code representation, or
(b) have it *write a SymPy script we actually execute* — program-of-thought
(PAL/PoT) — and score the script's printed answer with the unchanged
evaluation pipeline.

**TL;DR — findings.**

1. **Executing the model's code beats every prompting method tried, on both
   models.** code_exec scores **42/100** on `test_100_v2` with gpt-5.4 — the
   best GPT-family result on this benchmark: +7 pp over the basic-prompt
   baseline (35) and +4 pp over the 5-agent agentic workflow (38) at 44% of
   agentic's cost ($4.71 vs $10.62). On gpt-5.4-mini it scores **35/100** —
   +14 pp over baseline (21), above mini agentic (32), and **equal to the
   plain gpt-5.4 baseline at ~26% of its cost** ($0.78 vs $3.00). The
   scaffolding-helps-weaker-models-more pattern from the agentic pilot
   repeats (+14 mini vs +7 gpt-5.4), but with roughly triple the lift at
   both scales. The fine-tuned 360M specialist still leads (46 plain,
   53 with solver routing).
2. **The lift is from execution, not representation.** The ablation — same
   code-shaped prompt, model answers directly with no execution — scores
   24/100, only +3 over baseline. Seeing `K = sin(x)*cos(t)` instead of
   `\sin(x)\cos(t)` barely matters; *running* the linear-system script is
   what pays.
3. **Where it wins and where it loses is mechanistic, not statistical.**
   exact_symbolic jumps 6/30 → **20/28** (the separable-kernel linear system
   is deterministic algebra once executed). But none-detection collapses to
   **0/15** (baseline 2/15, agentic 5/15): with float coefficients the
   linear system is never exactly singular, so scripts always emit a
   "solution" for unsolvable equations. A hybrid (script + decision guard)
   is the obvious next step: the 360M SFT router already shows decisions are
   the cheap part.

## 1. Method

### Code representation

`experiments/code_prompt/build_prompts.py` re-renders each `test_100_v2` item
(same equations, ground truth, metadata as the basic prompts) as:

```python
from sympy import *

x, t = symbols('x t')

lam = 0.6800510922329444
a, b = -1.1508025669664157, 9.746297944109074  # integration domain
K = sin(x)*cos(t)  # kernel K(x, t)
f = ...  # right-hand side f(x)

# Fredholm integral equation of the second kind:
#   u(x) - lam * Integral(K * u(t), (t, a, b)) = f
```

LaTeX → SymPy conversion via the repo's `parse_latex_to_sympy`; all 100 items
parse, and every generated block exec-validates. First-kind items
(`lambda_val == 0`, the same rule the basic renderer uses) are rendered as
`Integral(K * u(t), (t, a, b)) = f` — no ground-truth leakage.

### Two variants

- **`code` (representation only).** The model sees the code block and answers
  directly in the standard `SOLUTION:/HAS_SOLUTION:/SOLUTION_TYPE:` format.
  Isolates the representation effect.
- **`code_exec` (program-of-thought).** The model must reply with one
  ```python block that solves the equation and *prints* the standard format
  (solution in LaTeX via `sympy.latex()`). `CodeExecModelRunner`
  (`src/llm/code_exec_runner.py`, config `model.code_exec`) executes the block
  in an isolated subprocess (`python -I`, 60 s timeout), feeds one failure
  back for a repair round, and falls back to the raw text. The script's
  stdout replaces the raw response, so postprocess/evaluate run unchanged —
  same pattern as the agentic runner, same metrics as every prior run.

Prompts ask for the separable-kernel reduction explicitly (solve the linear
system for the integral coefficients; singular+consistent → family,
singular+inconsistent → none) — the same method hints the agentic
degenerate-kernel directive gives.

### Runs

Same eval settings as the July 7 pilot (temperature 0.1, max_tokens 4096,
type tolerances series 0.01 / approx_coef 0.001 / regularized 0.001).
Models via the rastar.dev router. 8-item diverse pilot first, then full 100.

## 2. Results (test_100_v2, n=100)

| Model | Method | Correct/100 | Acc (evaluated) | Type acc | Has-sol acc | Cost |
|---|---|---|---|---|---|---|
| gpt-5.4-mini | baseline (basic prompt) | 21/100 | 0.214 | 0.253 | 0.740 | — |
| gpt-5.4-mini | agentic multi-method | 32/100 | 0.320 | 0.310 | 0.750 | — |
| gpt-5.4-mini | **code** (repr only) | 24/100 | 0.245 | 0.256 | 0.790 | $0.76 |
| gpt-5.4-mini | **code_exec** (PoT) | **35/100** | **0.376** | 0.198 | 0.727 | $0.78 |
| gpt-5.4 | baseline (basic prompt) | 35/100 | 0.365 | 0.385 | 0.740 | $3.01 |
| gpt-5.4 | agentic multi-method | 38/100 | 0.384 | 0.420 | 0.750 | $10.62 |
| gpt-5.4 | **code_exec** (PoT) | **42/100** | **0.442** | 0.244 | 0.778 | $4.71 |
| SmolLM2-360M | plain SFT | 46/100 | 0.505 | 0.680 | 0.920 | local |
| SmolLM2-360M | plain SFT + solver routing | 53/100 | 0.582 | 0.680 | 0.920 | local |
| (no LM) | separable solver always | 28/100 | 0.359 | 0.300 | 0.850 | free |

### Per-type, gpt-5.4-mini

| Method | exact_symbolic | approx_coef | series | discrete_pts | family | regularized | none |
|---|---|---|---|---|---|---|---|
| baseline | 6/30 | 3/14 | 0/10 | 0/9 | 10/10 | 0/10 | 2/15 |
| agentic | 13/30 | 2/15 | 0/10 | 0/10 | 10/10 | 2/10 | 5/15 |
| code | 9/30 | 0/15 | 0/9 | 0/9 | 10/10 | 0/10 | 5/15 |
| code_exec | **20/28** | 3/13 | 0/10 | 0/7 | 10/10 | 2/10 | **0/15** |

gpt-5.4 code_exec per-type: exact_symbolic **23/29**, regularized **6/10**
(best regularized result of any GPT method — the scripts actually compute a
least-squares/regularized solution instead of refusing), family 10/10,
approx_coef 2/14, none **1/15** (same collapse as mini), series and
discrete_points 0.

The gpt-5.4 `code` (representation-only) run was skipped: the mini ablation
already showed the representation effect is small (+3), and the marginal $3
buys no new conclusion.

### Execution reliability (code_exec)

- mini: 97/100 exec_ok — 92 first try, repair recovered 5 of 7 (one residual
  traceback, one 60 s timeout); 2 fallback_raw, 1 no_code (prose answer).
- gpt-5.4: 96/100 exec_ok — 87 first try, repair recovered 9 of 13;
  4 fallback_raw.

## 3. Reading

- **code_exec turns a prompting problem into a compilation problem.** The
  model doesn't need to *do* the algebra — only to *set up* the linear system
  correctly. That's why mini+code_exec matches the gpt-5.4 baseline: setup
  ability saturates earlier than symbolic-manipulation ability.
- **The failure mode is principled.** Every `none` item is missed because
  numeric linear solves don't detect near-singularity; the resonance items
  (`family`, f=0) survive because the models special-case f=0 in code. A
  `cond(A) > 1/eps` guard in the prompt, or routing the decision to the SFT
  specialist and the algebra to the script, is the natural hybrid.
- **Agentic vs code_exec.** Agentic buys its lift with 2.2× requests and
  verification plumbing; code_exec gets more lift on both models from one
  call + one subprocess (113 requests total on gpt-5.4, of which 13 were
  repairs). The methods are orthogonal: the agentic verifier could score
  script outputs.

Total experiment spend: ≈$6.4 (two pilots $0.11, mini full runs $1.54,
gpt-5.4 code_exec $4.71), plus an unrecoverable partial run that was killed
mid-batch on the first gpt-5.4 attempt.

## 4. Reproduce

```bash
.venv/bin/python experiments/code_prompt/build_prompts.py
.venv/bin/python -m src.cli run --config configs/test_100v2_gpt54mini_code_exec.yaml
.venv/bin/python experiments/code_prompt/compare.py
```

Artifacts: `results/code_prompt_2026-07-13/` (metrics, evaluated predictions,
code_exec traces with every generated script, cost summaries).
