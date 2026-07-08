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

## test_100 v2 (first-kind fix) — gpt-5.4-mini, 2026-07-08

After fixing the first-kind rendering (prompts now show `∫K u dt = f` for
regularized items; 90 other prompts byte-identical to v1) and switching
regularized scoring to type classification:

| metric | baseline | agentic |
|---|---|---|
| accuracy | 21.4% (21/98) | **32.0% (32/100)** |
| exact_symbolic | 6/30 | **13/30** |
| regularized | 0/10 | **2/10** |
| none | 2/15 | 5/15 |
| family | 10/10 | 10/10 |

- **Regularized is now a genuine hard task:** the baseline still never says
  "regularized" (0/10). The agentic run got 2/10; the confusion matrix shows
  8/10 first-kind items classified as `none` — models sense the equation is
  pathological but pick the wrong label.
- The v2 agentic-baseline gap (+10.6 pp) is larger than v1 (+4 pp); part of
  that spread is run-to-run variance (temperature 0.1), so cite with care.

**Incomplete runs (stopped 2026-07-08 ~00:53):** gpt-5.4 test_100 agentic was
lost mid-batch (results were memory-only; per-equation checkpointing has since
been added so this cannot recur). The gpt-5.5 test_100 pair was never started.

**Salvaged gpt-5.4 test_100 baseline (partial, 59/100, v1 prompts).** The
baseline inference was itself cut off at 59 of 100 equations
(`results/.../test_100_gpt54_baseline_unevaluated/predictions_20260708_000045.jsonl`).
The full symbolic/numeric eval-only pass hangs on an untimed `symbolic_compare`
for one pathological v1 prediction (`evaluate_solutions` does not thread its
`mode` down to gate the sympy step). A hang-proof, sympy-free type-only score
(`metrics_partial59_typeonly.json`) is stored instead: **solution-type accuracy
37.0% (20/54)**, has-solution 81.4%, none-detection F1 0.154 (1 TP / 3 FP / 8
FN). Per-type type-match: exact_symbolic 16/19, approx_coef 2/10, family 1/5,
none 1/9, series 0/5, discrete_points 0/4, regularized 0/7 — the familiar
"classifies closed-form well, misses every pathological type" signature. This
partial is exact_symbolic-heavy (19/54), has no matching agentic run, and is
**not a head-to-head**; treat the number as indicative only.

## Recommendations

- Fix the first-kind → second-kind serialization bug (`lambda_val = 0`).
- Re-run at `test_100` scale for statistical power (in progress).
- Strengthen the `fredholm_alternative` directive for none-detection
  (≤ 1/3 across all runs).
- Add special-function support (`Si`, `Ci`, …) to the evaluation parser.
