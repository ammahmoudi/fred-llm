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

## test_100 v2 — gpt-5.4 (the bigger model), 2026-07-08

Same v2 prompts and scoring as the mini run above, on `cx/gpt-5.4` — the
head-to-head at a stronger base model. gpt-5.4 is a reasoning model (~45 s/call
and it emits gnarly expressions), which forced two pieces of hardening now in
the pipeline: parallel baseline inference (8 workers) and bounded evaluation
(symbolic + metric-leaf SIGALRM timeouts, plus a verifier complexity guard) so
untimed sympy can no longer hang the run. A few pathological answers time out
of the sympy scoring and drop from each side's denominator.

| metric | baseline | agentic |
|---|---|---|
| correct equations | 35/100 | **38/100** |
| accuracy (of evaluated) | 36.5% (35/96) | 38.4% (38/99) |
| symbolic accuracy | 26.0% | 28.3% |
| solution-type accuracy | 38.5% | **42.0%** |
| exact_symbolic | 21/30 | **24/30** |
| regularized | 1/10 | 2/10 |
| family | 10/10 | 10/10 |
| none | 2/15 | 2/15 |
| approx_coef / series / discrete_points | ~0 | ~0 |

Agentic: 735 LLM calls, 0 API errors. Selection: 33 verified, 66 majority vote,
1 best-effort. Method wins: degenerate_kernel 57, adomian 20, neumann 13,
fredholm_alternative 9, numerical 1.

**Headline finding — the agentic lift shrinks as the base model gets stronger.**
gpt-5.4's single-prompt baseline (36.5%) already sits far above mini's (21.4%),
so there is little left for the scaffolding to fix: agentic adds +3 equations
(+1.9 pp) here versus +10.6 pp on mini. The verification mechanism still fires —
33/100 equations settled by deterministic residual checks (up from mini's 29) —
and it nudges exact_symbolic 21 → 24/30 and type accuracy +3.5 pp. But on a
capable base model the marginal value of parallel-method + verify is modest. The
two models together sketch the trend: **agentic multi-method solving pays off
most exactly where the base model is weakest.**

(Denominators differ — baseline evaluated 96, agentic 99 — because a few
pathological answers time out of the sympy evaluation on each side; the
correct-count-over-100 framing, 35 vs 38, is the cleanest comparison.)

## test_100 v2 — gpt-5.5 (top tier), 2026-07-08

Same v2 prompts and scoring, on `cx/gpt-5.5`. Both sides re-scored through the
identical `evaluate_solutions` path so the comparison is method-for-method with
the runs above. gpt-5.5 answers fast (~3 s on trivial prompts) but still emits
gnarly closed forms, so the same bounded-eval scaffolding applies. One point of
note: a `_MetricTimeout` from a runaway `doit()` integration escaped the
pipeline's own evaluation loop (which caught only `except Exception`, and the
timeout is a `BaseException` by design) and crashed the baseline eval. Fixed by
adding the `except _MetricTimeout` guard the module-level loop already had;
predictions were saved before the crash, so no inference was re-paid.

| metric | baseline | agentic |
|---|---|---|
| correct equations | 36/100 | **38/100** |
| accuracy (of evaluated) | 38.3% (36/94) | 38.0% (38/100) |
| symbolic accuracy | 26.6% | 28.0% |
| solution-type accuracy | 28.4% | **36.7%** |
| exact_symbolic | 23/30 | 22/30 |
| regularized | 0/10 | 1/10 |
| family | 10/10 | 10/10 |
| none | 1/15 | 2/15 |
| approx_coef | 2/13 | 3/15 |
| series / discrete_points | 0 | 0 |

Agentic: 695 LLM calls, 0 API errors. Selection: 38 verified, 60 majority vote,
2 best-effort. Method wins: degenerate_kernel 59, adomian 17, neumann 12,
fredholm_alternative 7, numerical 5.

**The lift has flattened to noise.** gpt-5.5's single-prompt baseline (36/100)
already sits at the top of the table, and the agentic run lands on 38/100 — the
same absolute count the workflow reached on gpt-5.4. In accuracy-of-evaluated
terms the two are indistinguishable (38.3% vs 38.0%). The mechanism has not
stopped working: verification fires *more* here than on any weaker model
(38 verified selections, up from 33 on gpt-5.4 and 29 on mini), and
type-classification still gains a solid +8.3 pp (28.4% → 36.7%). The agentic
side also evaluated all 100 items where the baseline dropped 6 to pathological
sympy — a small robustness edge from selecting verified, tamer candidates. But
on headline accuracy the base model has closed the gap the scaffolding used to
fill.

(Baseline evaluated 94, agentic 100 — six baseline answers timed out or failed
in sympy; correct-count-over-100, 36 vs 38, is the cleanest comparison.)

## Model-scaling summary (v2, test_100)

| model | baseline | agentic | agentic lift |
|---|---|---|---|
| gpt-5.4-mini | 21.4% (21/98) | 32.0% (32/100) | **+10.6 pp / +11 eq** |
| gpt-5.4 | 36.5% (35/96) | 38.4% (38/99) | +1.9 pp / +3 eq |
| gpt-5.5 | 38.3% (36/94) | 38.0% (38/100) | ~flat / +2 eq |

The agentic workflow is a bigger win on the weaker model, and the lift shrinks
monotonically as the base model gets stronger: **+11, +3, +2 equations** across
mini → gpt-5.4 → gpt-5.5. The absolute agentic output plateaus at 38/100 for
both gpt-5.4 and gpt-5.5 while the baseline climbs (21 → 35 → 36), so the gap
closes from the top down. Verification still fires most on the strongest model
(38 vs 33 vs 29 verified selections) and type classification keeps improving —
the mechanism works throughout; what vanishes is the headroom a strong base
model leaves for it to recover.

**Superseded:** an earlier partial gpt-5.4 v1 baseline (59/100, salvaged as a
type-only score of 37.0% type accuracy) is retained under
`test_100_gpt54_baseline_unevaluated/` for provenance but is replaced by the
full v2 baseline above.

## Recommendations

- Fix the first-kind → second-kind serialization bug (`lambda_val = 0`).
- Re-run at `test_100` scale for statistical power (in progress).
- Strengthen the `fredholm_alternative` directive for none-detection
  (≤ 1/3 across all runs).
- Add special-function support (`Si`, `Ci`, …) to the evaluation parser.
