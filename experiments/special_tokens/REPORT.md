# FRED-Token: Decision-Token Fine-Tuning for Fredholm Integral Equations

**TL;DR — three findings.**

1. **A 360M local specialist beats frontier generalists on this benchmark.**
   SmolLM2-360M-Instruct fine-tuned on-device (MLX, M1 Pro, ~2 h) on the
   repo's augmented equation data reaches **0.505** evaluator accuracy
   (0.460 strict) on `test_100_v2` — above every GPT run we have, including
   gpt-5.5 (0.383) and gpt-5.4 agentic (0.384). The Gorilla/APIGen lesson
   (narrow specialist + verified data > frontier generalist) transfers to
   symbolic integral equations.
2. **The novel special-token scheme *underperforms its own plain ablation*.**
   The FRED-token model (15 new tokens: structure markers + 7 first-token
   solution-type decision tokens, Breeze-2 style) scores 0.370 (0.340
   strict) — worse than the same model fine-tuned on the same equations
   rendered as plain text. The decision-token hypothesis fails here, and the
   ablation is what proves it. Details in §3.
3. **v2 — routing to a sympy solver lifts the specialist to 0.582 (0.530
   strict).** Treating the model's type decision as a *tool trigger* — when
   it says "solvable", a deterministic degenerate-kernel sympy solver solves
   the equation from the prompt and its answer is spliced in — raises
   exact_symbolic from 0.23 to **0.600** with zero retraining. The router is
   load-bearing: the same solver with no LM router scores 0.359 (it "solves"
   unsolvable equations, 0.0 on none/regularized). And the v1 conclusion
   survives v2: the plain-label router routes exactly as well as the
   special-token router (both capture exact 0.600) while keeping its better
   decision accuracy. Details in §4.

## 1. Method

### Token scheme (15 new tokens)

Input structure (masked from the loss):

```
<|fred|> <|kernel|> {K(x,t) LaTeX} <|lambda|> {λ} <|rhs|> {f(x) LaTeX} <|domain|> {a} {b} <|solve|>
```

Completion (trained):

```
<|T_{type}|> <|sol|> {solution LaTeX} <|end|>
```

The 7 decision tokens (`<|T_exact_symbolic|>`, `<|T_approx_coef|>`,
`<|T_discrete_points|>`, `<|T_series|>`, `<|T_family|>`, `<|T_regularized|>`,
`<|T_none|>`) commit the model to a solution type with its *first generated
token* (Breeze-2, arXiv:2501.13921). Expression bodies stay in the pretrained
tokenizer's native surface form (arXiv:2110.03501 — new tokens only for
structure/control). Each new embedding row initialized to the mean of the
original tokenizer's subword embeddings of a short description (GTI,
arXiv:2604.02324); SmolLM2 ties embeddings so one matrix is resized
(49152 → 49216, padded to a multiple of 64 for Metal).

Why decision tokens should have worked on this benchmark: in the repo
evaluator, correctness for `none` (has_solution False), `regularized` (type
match) and `family` (family match) is **purely a decision** — 35 of the 100
test items — and the GPT baselines systematically fail them (they hallucinate
exact solutions instead of saying "no solution" / "ill-posed").

### Data

11,100 examples from the repo's own augmentation pipeline
(`src/data/augmentation.py`) applied to raw base rows, excluding any base row
whose λ (6 dp) appears in `test_100_v2`: exact_symbolic 3000, none 1800,
approx_coef 1500, series/regularized/family/discrete_points 1200 each.
Everything rendered `sympy.latex`, floats rounded to 4 significant figures
(train and eval identically), examples >1000 tokens dropped (~1.8%). The same
equations emitted twice: special-token format (`data_special/`) and plain NL
with the repo's `SOLUTION:/HAS_SOLUTION:/SOLUTION_TYPE:` format
(`data_plain/`). 90/10 split, prompts masked from loss in both variants.

### Training (`train_full.py`, MLX)

Full fine-tune, batch 4, max-seq 1024, Adam, grad-norm clip 1.0.
Special: 2500 steps @1e-5 (diverged at ~1250, resumed from clean checkpoint)
+ 1500 steps @5e-6 with special-token loss upweighting (see §3). Plain: 500
steps @1e-5 + 2000 @5e-6 (one NaN restart). Peak 11.6–13.6 GB on 16 GB M1 Pro.

Three MLX training findings not in any of the papers:

1. **The mlx-lm LoRA CLI never trains embeddings** — `--fine-tune-type full`
   unfreezes only `model.layers`; `embed_tokens` stays frozen even with
   `--num-layers -1`. With frozen rows the model *never learns to emit* the
   new tokens, while teacher-forced CE still looks healthy (~1.1) because
   body tokens dominate. A custom loop with `model.unfreeze()` is required.
2. **bf16 full FT diverges without gradient clipping** (loss 0.5 → 10.5
   pinned in one run; a NaN in another at LR 1e-5 even with clipping —
   5e-6 held).
3. **One decision token per ~40 body tokens dilutes its gradient 40:1.**
   After 2500 stable steps first-token accuracy was ~30%. Upweighting
   special-token targets ×10 in the CE (ToolkenGPT Eq. 2 in spirit) lifted it
   to 51% on valid — but see §3 for what that trade bought.

## 2. Results (test_100_v2, repo evaluator, mode=both, per-type tolerances)

"Evaluator acc" uses the evaluator's convention (unparseable/timed-out items
leave the denominator; n varies by run). "Strict" counts them as wrong
(correct/100). Generated by `compare.py`:

| model | evaluator acc (n) | strict acc (/100) | exact_symbolic | approx_coef | series | discrete_points | none | family | regularized |
|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-360M special-token | 0.370 (92) | 0.340 | 0.185 | 0.308 | 0.000 | 0.000 | 0.400 | 1.000 | 0.900 |
| SmolLM2-360M plain (ablation) | 0.505 (91) | 0.460 | 0.233 | 0.733 | 0.000 | 0.000 | 0.533 | 1.000 | 1.000 |
| SmolLM2-360M special + solver (v2) | 0.495 (97) | 0.480 | 0.600 | 0.357 | 0.000 | 0.000 | 0.400 | 1.000 | 0.900 |
| **SmolLM2-360M plain + solver (v2)** | **0.582 (91)** | **0.530** | 0.600 | 0.467 | 0.000 | 0.000 | 0.533 | 1.000 | 1.000 |
| sympy solver, no LM router (v2) | 0.359 (78) | 0.280 | 0.833 | 0.375 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| gpt-5.4-mini baseline | 0.214 (98) | 0.210 | 0.200 | 0.214 | 0.000 | 0.000 | 0.133 | 1.000 | 0.000 |
| gpt-5.4-mini agentic | 0.320 (100) | 0.320 | 0.433 | 0.133 | 0.000 | 0.000 | 0.333 | 1.000 | 0.200 |
| gpt-5.4 baseline | 0.365 (96) | 0.350 | 0.700 | 0.071 | 0.000 | 0.000 | 0.133 | 1.000 | 0.100 |
| gpt-5.4 agentic | 0.384 (99) | 0.380 | 0.800 | 0.000 | 0.000 | 0.000 | 0.133 | 1.000 | 0.200 |
| gpt-5.5 baseline | 0.383 (94) | 0.360 | 0.767 | 0.154 | 0.000 | 0.000 | 0.067 | 1.000 | 0.000 |
| gpt-5.5 agentic | 0.380 (100) | 0.380 | 0.733 | 0.200 | 0.000 | 0.000 | 0.133 | 1.000 | 0.100 |

Denominator note: the plain run had 9/10 discrete_points items dropped as
unparseable (special: 3/10). Every model scores 0 on every discrete item it
evaluates, so the strict column is the fair overall comparison — the ordering
is unchanged either way.

## 3. Why the special tokens lost to plain text

The ablation isolates the token scheme: same base model, same equations, same
optimizer recipe, same eval. Plain wins on every type where they differ.
What the evidence supports:

- **A decision doesn't need a dedicated token when it's just an output
  label.** The plain model expresses the same triage as ordinary text
  (`HAS_SOLUTION: no`, `SOLUTION_TYPE: regularized`) using tokens whose
  embeddings carry full pretraining knowledge — and hits none 0.533 /
  regularized 1.0, above the special model's 0.400 / 0.900. Decision tokens
  à la ToolkenGPT/Breeze-2 earn their keep when a token must *trigger
  machinery* (mode switch, tool call, argument subroutine) — not when it
  merely names a class the evaluator reads back off the transcript.
- **Cold-start embeddings at 10k examples are a real tax.** 15 rows learned
  from scratch (description-mean init or not) vs zero new parameters for
  plain. The frozen-embedding false start and the ×10 loss reweighting were
  both remediations for this — plain needed neither.
- **The reweighting traded body accuracy for decision accuracy, and lost
  net.** approx_coef is scored on the solution body: plain 0.733 vs special
  0.308. Weighting structure tokens ×10 starves body tokens of relative
  gradient; the decisions it bought (first-token 30%→51%) were worth fewer
  benchmark points than the bodies it cost. (Hypothesis, but the ablation
  controls everything else in the recipe except the divergence-recovery
  history.)
- Both fine-tunes crush the GPTs on the decision types (regularized 0.9–1.0
  vs ≤0.2), so the *triage* win comes from supervised exposure to
  in-distribution edge cases — not from how the answer is tokenized.

Honesty caveats: our models train on the same augmentation pipeline that
generated the test set (in-distribution by construction; the GPTs are
zero-shot). Single seed, single run, 100 items — ±0.05 is noise. The 3
regularized test items render with λ=0 (known repo artifact) and regularized
accuracy is decision-only. The special model's optimization history was
messier than plain's (extra false-start epochs); a cleaner rerun could narrow
the gap, though it would have to overcome the 0.31-vs-0.73 body deficit.

## 4. v2 — decisions as tool triggers (`solver.py`, `eval_v2.py`)

The ToolkenGPT loop simplifies drastically here: the full equation is already
in the prompt, so a "tool call" needs no generated arguments — the type
decision alone routes. Harness (no retraining, no regeneration; v1's stored
raw outputs are re-parsed): when the model's decision is a solvable type
(exact_symbolic / approx_coef / series), parse K, f, λ, [a,b] from the item,
run a deterministic degenerate-kernel solver, splice its answer in as the
solution body; otherwise the model's own answer stands.

**Solver** (`solver.py`): decompose K(x,t) = Σ gᵢ(x)hᵢ(t) (refuse mixed
factors like |t−x| or cos(xt)); Aᵢⱼ = ∫hᵢgⱼ dt, bᵢ = ∫hᵢf dt with
alarm-bounded symbolic integration and scipy-quad fallback; entries reduced
to floats immediately (symbolic Bareiss determinants on unevaluated
`e^{122}·erf(...)` constants spin forever — found by faulthandler stack
dump); (I − λA)c = b solved numerically; u = f + λΣcⱼgⱼ. Singular system →
refuse (resonance is family/none territory). 59/100 test equations are
solvable this way; the 10 unparseable kernels are all `|t−x|` forms
(non-separable anyway).

**Input-parse trust**: Math-Verify silently drops terms on some LaTeX
(discovered in v1), so the harness parses with sympy-antlr and cross-checks
Math-Verify numerically at random points — disagreement means "skip the
item" rather than "solve the wrong equation" (12/200 fields vetoed).

What the three v2 rows show:

- **The tool closes the solving gap.** exact_symbolic 0.185/0.233 → **0.600**
  for both routed configs — from a quarter of GPT-5.5's level to parity-minus
  (GPTs: 0.7–0.8). Solver ceiling when called on everything: 0.833.
- **The router is where the LM earns its place.** solver-always scores 0.359
  (0.280 strict): it answers "here is a solution" on every no-solution and
  ill-posed item (none/regularized 0.0). The plain-routed config keeps those
  at 0.533/1.0 — worth +25 strict points over the raw tool.
- **v2 re-confirms v1's negative result.** Even as tool triggers — the job
  special tokens were theoretically built for — they add nothing over plain
  labels: both routers capture exactly 0.600 on exact_symbolic, and the plain
  router's better decisions carry it to 0.582 vs 0.495. Routing needs a
  readable decision, not a dedicated token.
- Residual gaps: routers capture 18/30 exact items vs the solver's 25 (7 lost
  to type misdecisions); approx_coef kernels are mostly non-separable
  (weakly-singular/oscillatory), where quad fallback recovers only some;
  series stays 0.0 — those kernels are mostly e^{−|t−x|} (non-separable), and
  where solvable, the exact answer differs from the truncated-Neumann ground
  truth by more than the 1% tolerance.

## 5. Novelty

All published neural approaches to Fredholm equations are numeric (operator
learning / PINN: 2408.09484, 2401.07003, 2408.12389, 2409.01899). We found no
published work that tokenizes Fredholm equations with dedicated structure
tokens for a pretrained LM producing *symbolic* solutions, or that does
solution-type triage via first-token decision tokens. This experiment fills
that cell — with a negative result for the token scheme and a positive one
for plain specialist SFT.

## 6. Limitations & future work

- **v2 tested the routing form of the ToolkenGPT loop and it delivered
  (§4)** — but the argument-generating form (model emits a kernel
  decomposition or solver parameters mid-generation, outputs masked per their
  Eq. 2) remains untested; it would matter for non-separable kernels where
  the deterministic decomposition refuses.
- approx_coef beyond 0.467 needs non-separable machinery (weakly-singular
  quadrature, Nyström); series/discrete_points are ~unwinnable as scored
  (0.0 for every run, all eleven configs).
- `none` detection tops out at 0.533 — some none-augmentations (eigenvalue
  cases) likely need spectral reasoning, not surface cues.
- Found+fixed while evaluating: the evaluator's residual-verification path
  (`verify_solution` → sympy heurisch) was unbounded and hung indefinitely on
  one item; now wrapped in the existing `_metric_alarm` (+ regression test in
  `tests/test_evaluate.py`).

## 7. Artifacts

| path | what |
|---|---|
| `experiments/special_tokens/build_data.py` | data builder (both variants) |
| `experiments/special_tokens/resize_model.py` | tokenizer + embedding surgery |
| `experiments/special_tokens/train_full.py` | full FT incl. embeddings, clipping, weighted loss |
| `experiments/special_tokens/eval_finetuned.py` | test_100_v2 eval harness |
| `experiments/special_tokens/solver.py` | v2 separable-kernel sympy solver + validated LaTeX parse |
| `experiments/special_tokens/eval_v2.py` | v2 solver-routed eval (reuses v1 raw outputs) |
| `experiments/special_tokens/compare.py` | comparison table generator |
| `experiments/special_tokens/model_{special,plain}/` | trained models (not committed) |
| `results/special_tokens/test_100v2_smollm360_*/` | predictions + metrics |
| `experiments/special_tokens/HANDOFF.md` | original research brief + design |
