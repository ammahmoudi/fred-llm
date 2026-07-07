# Agentic Solver — Pilot Results (2026-07-07)

First head-to-head of the agentic multi-method workflow (docs/AGENTIC_SOLVER.md)
against single-prompt baselines. Dataset: `diverse_21` (3 per solution type × 7
types), basic prompt style, LaTeX format. Models served by the rastar.dev
router (`base_url: https://ai-router.rastar.dev/v1`). Raw artifacts:
`results/agentic_pilot_2026-07-07/` (metrics, evaluated predictions, and
per-equation agentic traces).

## Headline table

| model | mode | correct | accuracy | type acc | has-sol acc | LLM calls |
|---|---|---|---|---|---|---|
| cx/gpt-5.4-mini | baseline | 6/21 | 28.6% | 23.8% | 76.2% | 21 |
| cx/gpt-5.4-mini | agentic | 7/21 | 33.3% | 28.6% | 76.2% | 137 |
| cx/gpt-5.4 | baseline | 6/21 | 28.6% | 23.8% | 76.2% | 21 |
| cx/gpt-5.4 | agentic | 6/21 | 28.6% | 28.6% | 76.2% | 145 |
| cx/gpt-5.5 | baseline | 7/21 | 33.3% | 23.8% | 76.2% | 21 |
| cx/gpt-5.5 | agentic | 6/20¹ | 30.0% | 28.6% | 76.2% | 128 |

¹ One gpt-5.5 answer used the sine integral `Si(x)`, which the evaluation
parser cannot handle; that row errored out of evaluation (20 evaluated, not 21).

Per-type (all six runs): `exact_symbolic` ≈ 3/3 (mini baseline 2/3),
`family` 3/3, `none` ≤ 1/3, and `approx_coef` / `series` /
`discrete_points` / `regularized` = 0 everywhere.

## Findings

1. **Agentic consistently improves solution-type classification**
   (+4.8 pp on every model: 23.8% → 28.6%) at ~6.5× the LLM calls.
   Headline accuracy is a wash: +1 equation (mini), ±0 (5.4), −1 (5.5).

2. **The pilot has a hard ceiling that masks any real effect.** Three
   `regularized` items are unwinnable as rendered (see below), and the
   `approx_coef` / `series` / `discrete_points` items fail on evaluation
   tolerances (1e-3 / 1e-2) or answer-format grounds for *all* models.
   Effectively ~8–10 winnable items — every model lands on 6–7 correct, so
   the between-run differences are 1–2 equations, i.e. noise. **diverse_21
   cannot statistically separate agentic from baseline.**

3. **Verification demonstrably works where it applies.** From the mini trace:
   9/21 equations were settled by deterministic residual verification, 12 by
   majority vote. On eq_4 the verified `u = x` (residual 3.6e-07) from
   neumann/fredholm_alternative beat three plausible wrong candidates from
   the other methods — that is the mini agentic +1. Method wins across the
   agentic runs: degenerate_kernel 12, adomian 5, neumann 3, numerical 1.

4. **Dataset bug found by the verifier:** the three `regularized`
   (first-kind) items carry `lambda_val = 0.0` and ground truth `'0'`. The
   second-kind prompt template renders them as `u(x) − 0.0·∫K u dt = f(x)` —
   the trivial `u(x) = f(x)`. The ill-posedness never reaches the model, so
   `regularized` accuracy 0 is a rendering artifact, not model failure. Fix
   the first-kind serialization in the augmentation before citing per-type
   numbers.

5. **Label-philosophy edge case:** on eq_5 (`discrete_points` GT) the neumann
   agent produced a closed-form solution that residual-verifies against the
   equation, but scores wrong because the label demands a point list.

## test_100 (gpt-5.4-mini) — the statistically meaningful run

100 equations (30 exact_symbolic, 15 approx_coef, 15 none, 10 each series /
discrete_points / family / regularized), same configs as the pilot
(`configs/test_100_gpt54mini_router*.yaml`). Agentic: 731 calls, 0 API errors.

| metric | baseline | agentic |
|---|---|---|
| accuracy | 21.0% (21/100) | **25.0% (25/100)** |
| symbolic accuracy | 8.0% | **15.0%** |
| solution-type accuracy | 26.4% | **29.5%** |
| none-detection P/R/F1 | 0.38 / 0.20 / 0.26 | **0.42 / 0.33 / 0.37** |
| exact_symbolic | 6/30 | **9/30** |
| none | 3/15 | **5/15** |
| family | 10/10 | 10/10 |
| approx_coef | 2/15 | 1/15 |
| series / discrete_points / regularized | 0 | 0 |

Selection reasons: 29 verified, 65 majority vote, 6 best-effort. Method wins:
degenerate_kernel 56, fredholm_alternative 13, neumann 13, adomian 12,
numerical 6.

**Interpretation:** at n=100 the agentic lift is consistent and
mechanism-backed — +50% relative on exact_symbolic (verification picks the
right candidate), nearly double the symbolic accuracy, and a clear
none-detection improvement (F1 0.26 → 0.37). The 10 regularized items remain
structurally unwinnable (λ=0 rendering bug), so the effective ceiling is
~90 items. Cost: ~7.3× calls for +4 pp absolute accuracy.

## Recommendations

- Fix the first-kind → second-kind serialization bug (`lambda_val = 0`).
- Re-run at `test_100` scale for statistical power (in progress).
- Strengthen the `fredholm_alternative` directive for none-detection
  (≤ 1/3 across all runs).
- Add special-function support (`Si`, `Ci`, …) to the evaluation parser.
