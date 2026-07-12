# Special-Token Fine-Tuning Experiment — HANDOFF

> **STATUS: COMPLETED 2026-07-12.** See `REPORT.md` for results and analysis
> (headline: plain SFT 0.505 beat both the special-token scheme 0.370 and all
> GPT baselines). This document is kept as the original research brief/design.

**Goal (user's words):** try a new way of solving Fredholm equations — read the
ToolkenGPT-line papers, design a new way of *adding special tokens for
differential/integral equations*, fine-tune a tiny MLX-community model on those
new special tokens, then evaluate and compare against the other methods already
in this repo (GPT-5.4-mini / 5.4 / 5.5 baseline + agentic).

This doc is a complete pickup point. A fresh session should be able to execute
from here with no re-discovery. Status at handoff: **research done, env ready,
design finalized, no training code written yet.**

---

## 0. Status checklist

- [x] Research the papers (ToolkenGPT etc.) — brief in §1
- [x] Map the repo eval harness + recover the test set — §3
- [x] MLX env installed + tiny model verified — §4
- [ ] Build training data with special tokens — §5 (script spec ready, NOT written)
- [ ] Resize model embeddings for new tokens — §6 (approach nailed down, NOT written)
- [ ] Fine-tune on MLX — §7
- [ ] Evaluate on test_100_v2 + compare — §8
- [ ] Write report + update FEATURES.md/memory — §9

TodoList IDs in the session: #4 build data, #5 fine-tune, #6 eval, #7 report.

---

## 1. Research brief (the papers the user pointed at)

Primary line = **tool/decision tokens added to a (possibly frozen) LM**, not
symbolic-math digit tokenization. Full extraction below; every claim has an
arXiv id.

### ToolkenGPT (Hao et al., NeurIPS 2023, arXiv:2305.11554) — the core mechanism
- Each tool = **one new token ("toolken")** with its own embedding row. LM head
  is `[W_vocab ; W_toolken]`; next-token is a single softmax over the union.
- Generation alternates **reasoning mode** (normal text) → when a toolken is
  sampled, **tool mode**: re-prompt with in-context demos for that tool to emit
  arguments, execute, splice the result back, resume.
- **Training: LM fully FROZEN; only the new embedding rows `W_toolken` trained.**
  Loss masks tool *outputs* (`[N/A]` indicator, their Eq. 2) — the model learns
  to *call* tools, never to imitate their outputs. **This masking is the single
  most important detail** if we do solver-toolkens.
- Data efficiency (Table 6, KAMEL): 10 demos/tool → 0.56, 20 → 0.90, 40 → 0.95.
- Ablation (Table 4, LLaMA-7B, FuncQA): LoRA FT **0.62** vs embeddings-only
  **0.55** — full/LoRA FT wins on accuracy; embeddings-only wins ~20× on compute.
  Their frozen-embeddings trick was validated on **13B–33B** backbones.

### Toolken+ (arXiv:2410.12004) — reject option
- Adds a `Rej` "tool" (T' = T ∪ {Rej}); top-k toolken candidates reranked with
  each tool's doc prepended; if `Rej` wins, fall back to ordinary generation.
- Cuts false tool-triggering. Cheap to add.

### Grounded Token Initialization (GTI, arXiv:2604.02324, 2026) — cold-start answer
- **Validated on Qwen3-0.6B — exactly our scale.**
- Mean-init of a whole *batch* of new tokens **collapses their distinctions**;
  random init breaks the softmax. Instead: (1) **grounding** — freeze backbone,
  train only new rows on `description → token` pairs; (2) use those as the init
  for normal SFT. Gain: **+21.63% rel P@5** over vanilla SFT; **+15.25 pp** from
  grounding alone.
- Sibling arXiv:2506.14248 (Re-Init Token Learning): init each toolken from the
  subword embeddings of its *name/description* and regularize toward it.

### Llama 3 (arXiv:2407.21783) — `<|python_tag|>`
- One token flags "this is a tool call"; the **stop token** choice encodes intent
  (`<|eom_id|>` = expect a tool result next; `<|eot_id|>` = done). Minimal 2–3
  token protocol trained by ordinary SFT.

### Breeze 2 (MediaTek, arXiv:2501.13921) — **decision tokens** (our main template)
- Two tokens `<|use_tool|>` / `<|answer|>`; the **first generated token** commits
  the model to an action.
- Plain SFT on 139k FC instances incl. **FC-NF-10k = 10k negative examples**
  (functions available but the right move is `<|answer|>`). lr 1e-6, 4 epochs.
- Payoff: relevance detection (correctly NOT calling) 80% BFCL. **Direct analog:
  our `none`/`regularized` types = "don't hallucinate an exact solution."**

### Small-model function-calling (one line each)
- **Gorilla (2305.15334):** full-FT LLaMA-7B beats GPT-4 on API calls, no special
  tokens. A narrow specialist beats a frontier generalist.
- **APIGen/xLAM (2406.18518):** 60k *verified, executable* FC examples; 1B beats
  GPT-3.5/Claude-3-Haiku. **Verified data >> model size** — our sympy-verified
  solutions are the analog.
- **Hammer (2410.04587):** function masking (randomize identifiers) so the model
  keys on *meaning* not surface names. Adapt: vary variable/kernel symbols across
  training examples.
- **ToolACE (2409.00920):** dual-layer verification; 8B rivals GPT-4.

### Novelty check (transformers × Fredholm)
All existing neural Fredholm work is **numeric** (operator learning / PINNs):
FNN 2408.09484, oscillatory 2401.07003, FIE-NO 2408.12389, PINNIES 2409.01899.
**No published work tokenizes Fredholm integral equations for a pretrained LM to
produce symbolic solutions, and none does solution-type classification via
decision tokens.** The proposed experiment is a genuine empty cell — defensible
novelty claim.

### Design recommendations that drove our choices
- **R1:** ~15–80 new tokens, three tiers: structure markers, 7 solution-type
  **decision tokens** (emitted first, Breeze-2), optional solver-toolkens. **Do
  NOT** re-tokenize expression bodies — keep them in the pretrained tokenizer's
  native surface form (arXiv:2110.03501). New tokens only for STRUCTURE/CONTROL.
- **R3:** at 360M, **full fine-tune** (not embeddings-only). No latent Fredholm
  competence to "unlock." Optionally GTI grounding first.
- **R4:** init each new row at the **mean of subword embeddings of a short text
  description** of it (never random, never batch-mean). SmolLM2 ties
  input/output embeddings → one init suffices.
- **R6:** numbers — SmolLM2 already does single-digit tokenization (good); render
  rationals structurally; round floats to ~4 sig figs to bound sequence length.
- **R8 pitfalls:** catastrophic forgetting (use conservative LR / small epochs /
  optionally mix generic text); over-triggering (add negatives + reject);
  surface-name shortcut (vary symbols); **always mask solver outputs** if we add
  solver-toolkens; pad resized vocab to a multiple of 64 for Metal.

Full brief with all numbers is reproduced verbatim at the bottom (§A).

---

## 2. The chosen design ("FRED-token" scheme)

**Lazy-but-real scope for v1:** structure tokens + 7 decision tokens. Skip the
full ToolkenGPT sympy-splice solver loop for v1 (big engineering lift; note as
future work). The decision-token + structure scheme is already the novel
contribution and is directly testable.

**New special tokens (15 total):**

Input/structure (appear in the prompt, masked in loss):
```
<|fred|>  <|kernel|>  <|lambda|>  <|rhs|>  <|domain|>  <|solve|>
```
Output decision tokens (first token of the completion — Breeze-2 commitment):
```
<|T_exact_symbolic|> <|T_approx_coef|> <|T_discrete_points|> <|T_series|>
<|T_family|> <|T_regularized|> <|T_none|>
```
Output structure:
```
<|sol|>   <|end|>   (<|end|> is the stop token)
```

**Prompt (masked):**
```
<|fred|> <|kernel|> {kernel_latex} <|lambda|> {lam} <|rhs|> {f_latex} <|domain|> {a} {b} <|solve|>
```
**Completion (trained):**
```
<|T_{type}|> <|sol|> {solution_latex} <|end|>
```
- `none`: `<|T_none|> <|sol|> No solution <|end|>`
- `regularized`: `<|T_regularized|> <|sol|> requires regularization <|end|>`
  (scoring only needs the type token — see §3)
- `family`: `<|T_family|> <|sol|> C \sin{\left(\pi x\right)} <|end|>`
- `discrete_points`: give a placeholder body; correctness realistically driven by
  type (everyone scored ~0 on discrete anyway).

**Render both input and output expressions as LaTeX via `sympy.latex(expr)`** so
the SAME evaluator parse path (`parse_latex_to_sympy`) works unchanged — this is
the whole reason to use LaTeX rather than python surface form.

**Init (R4):** each new row = mean of the ORIGINAL tokenizer's subword embeddings
of a short description, e.g. `<|T_series|>` ← mean(embed("series solution")),
`<|kernel|>` ← mean(embed("the integral kernel K(x,t)")). Fold "grounding" into
full FT; the description-mean init handles cold-start.

**Ablation (the scientific control):** fine-tune the SAME model on the SAME
equations rendered as **plain text (no special tokens)** — isolates the
special-token contribution. Then both go head-to-head vs the GPT methods.

---

## 3. Eval harness facts (verified by reading the code)

- Entry point: `src/evaluation/core.py::evaluate_solutions(results_path, mode,
  symbolic_tolerance, numeric_tolerance, n_test_points, use_math_verify,
  type_tolerances, include_points)`. Reads a JSON/JSONL of prediction dicts.
- Run it exactly like the baselines via the eval-only pipeline (config
  `dataset.evaluation_only.predictions_path`, `automation_level: eval-only`) OR
  just call `evaluate_solutions(path, mode="both", type_tolerances={...})`
  directly. Type tolerances used by test_100_v2: `series 0.01, approx_coef 0.001,
  regularized 0.001`, symbolic 1e-10, numeric 1e-6, num_test_points 100.
- **Prediction dict fields the evaluator reads** (`result.get(...)`):
  `equation_id`, `ground_truth`, `ground_truth_has_solution`,
  `ground_truth_solution_type`, `ground_truth_domain`, `solution_str`,
  `has_solution`, `solution_type`, and (optional, enables residual check)
  `ground_truth_kernel`, `ground_truth_f`, `ground_truth_lambda`,
  `evaluation_points`. `solution_sympy` is stored by the pipeline but the
  evaluator re-parses `solution_str` via `parse_latex_to_sympy`.
- **Correctness definition** (`src/evaluation/core.py`, the `Evaluator` methods):
  - standard (exact_symbolic / approx_coef / series):
    `correct = symbolic.equivalent OR numeric.match` (per-type tolerance).
  - **none**: `correct = (pred has_solution is False)`.
  - **regularized**: `correct = (pred solution_type == "regularized")`.
  - **family**: `family_match OR symbolic OR numeric`.
  - **discrete_points**: numeric point match (needs points; test items lack them
    → realistically ~0 for all methods, confirmed gpt-5.4-mini 0/9).
  → **So `none`/`regularized`/`family`/(type) correctness is a pure DECISION —
  exactly what the decision tokens target. 35 of 100 test items are
  correct-by-decision (30? see counts below).**

**Test set:** `data/prompts/basic/test_100_v2/test_100_samples.jsonl` (100 items).
Per-item schema: `equation_id`, `prompt`, `ground_truth`, `metadata{kernel, f,
lambda_val, domain[a,b], has_solution, solution_type}`. Type counts:
`exact_symbolic 30, approx_coef 15, none 15, series 10, discrete_points 10,
family 10, regularized 10`. The evaluator reports 98 evaluated (2 drop).

**Baselines to beat/compare** (all on test_100_v2, same evaluator):
`results/agentic_pilot_2026-07-07/` →
- `test_100v2_gpt54mini` (baseline): **accuracy 0.214** (21/98). Per-type:
  series 0.0, exact_symbolic 0.20, approx_coef 0.214, discrete 0.0, plus
  none/family/regularized from decision. metrics file:
  `test_100v2_gpt54mini/metrics_20260708_002101.json`.
- `test_100v2_gpt54mini_agentic`, `test_100v2_gpt54_baseline`,
  `test_100v2_gpt54_agentic` — read their metrics json for head-to-head.
- Memory note: "agentic-lift scales inverse to model" (+10.6 pp on mini, +1.9 on
  5.4). Also diverse_21 numbers exist under `results/.../diverse_21_*`.

Build our prediction JSONL by taking each test item's fixed `ground_truth*`
fields from the test file, rendering our special-token INPUT, generating, parsing
`<|T_*|>`→(solution_type, has_solution) and the `<|sol|>...<|end|>` body →
`solution_str`. `ground_truth_domain = metadata.domain`,
`ground_truth_has_solution = metadata.has_solution`,
`ground_truth_solution_type = metadata.solution_type`, `ground_truth =` item
`ground_truth`. Optionally add `ground_truth_kernel/f/lambda` from metadata.

---

## 4. Environment (ready)

- venv: `/Users/shahriar/PV/fun/fred-llm/.venv` (Python 3.11). Activate with
  `source .venv/bin/activate`.
- Installed: **mlx, mlx-lm 0.31.3, transformers**, plus sympy 1.14, numpy 2.4,
  pandas 2.3, scipy 1.16. Hardware: **Apple M1 Pro, 16 GB** (MLX GPU device OK).
- Model verified: **`mlx-community/SmolLM2-360M-Instruct`** loads & generates.
  `vocab_size=49152`, `hidden_size=960`, `num_hidden_layers=32`,
  **`tie_word_embeddings=True`**, model_type llama. `2+2=` → "4. ...".
  (360M is the pick; alt fallback `mlx-community/Qwen2.5-0.5B-Instruct-bf16`.)
- mlx-lm model object: `m.model` is the LlamaModel; embeddings at
  `m.model.embed_tokens` (nn.Embedding, weight shape (49152, 960)); tied head via
  `as_linear`. **`m.args` is a `ModelArgs` dataclass with `vocab_size`.**

---

## 5. Training-data builder (SPEC — not yet written)

Write `experiments/special_tokens/build_data.py`. Verified facts it relies on:

- Raw base rows: `data/raw/Fredholm_Dataset_Sample.csv` (~5000 rows, all
  `exact_symbolic`; python surface form: `u`,`f`,`kernel` like `x**2`,
  `cosh(x**2)`, columns `u,f,kernel,lambda,a,b`). Full set is
  `data/raw/Fredholm_Dataset.csv` (500k) if more volume needed.
- Edge-case generator: `from src.data.augmentation import _apply_augmentation`.
  Call per strategy on a base dict `{u,f,kernel,lambda_val,a,b}`. **Verified
  yields per base row:** none ~11, approx_coef ~14, discrete_points ~6, series
  ~3, regularized ~3, family ~2. Outputs are dicts with python-form
  `u/f/kernel`, `has_solution`, `solution_type`. `u` is EMPTY for
  none/regularized/discrete_points; `family` u=`C * sin(pi*x)`; `series` u is a
  long Neumann partial sum; `approx_coef` u is a closed form.
- Parse python → sympy with `sympy.sympify`; render with `sympy.latex`. Wrap in
  try/except, skip failures. **Round Floats to ~4 sig figs** before latex to
  bound sequence length (helper: walk expr, replace each `sp.Float` with a
  rounded Float; or `expr.xreplace({f: sp.Float(f, 4) for f in expr.atoms(sp.Float)})`).
- **Exclude test overlap:** drop any base row whose `(round(lambda,6),
  round(a,6), round(b,6))` matches a test_100_v2 item (29 raw rows matched a test
  lambda when checked — exclude to be safe). Also never draw edge variants from
  excluded bases.
- Target composition (~11k, aim to give each rare type enough signal per R7 —
  at least high-hundreds each): exact_symbolic ~3000 (from raw), approx_coef
  ~1500, series ~1200, none ~1800, regularized ~1200, family ~1200,
  discrete_points ~1200. Augment ~400 base rows then cap/sample per type.
  Vary kernel/variable rendering across examples if cheap (Hammer, R8-iv).
- Emit two dataset variants into two dirs:
  - `experiments/special_tokens/data_special/{train,valid}.jsonl` — special-token
    format (§2), lines `{"prompt": "...", "completion": "..."}`.
  - `experiments/special_tokens/data_plain/{train,valid}.jsonl` — ablation, plain
    NL prompt ("Solve the Fredholm equation ... K=..., f=..., λ=..., domain ...")
    with completion `SOLUTION: ...\nHAS_SOLUTION: ...\nSOLUTION_TYPE: ...` (mirror
    the repo's basic-style so parsing reuses `src/postprocessing/parse.py`).
  - 90/10 train/valid split. **`{"prompt","completion"}` is masked by mlx_lm** —
    only the completion contributes to loss (this is what we want: train on the
    decision token + solution, not the equation).

Leave one runnable check: a `__main__` that asserts every emitted line
round-trips (`json.loads`), the completion starts with a `<|T_*|>` token
(special variant), and all 7 types are present in train.

---

## 6. Embedding surgery (approach — not yet written)

Write `experiments/special_tokens/resize_model.py`:
1. `model, tok = mlx_lm.load("mlx-community/SmolLM2-360M-Instruct")`.
2. Add tokens to the underlying HF tokenizer:
   `tok._tokenizer.add_special_tokens({'additional_special_tokens': NEW_TOKENS})`
   (mlx-lm wraps HF tokenizer; underlying object is `tok._tokenizer`). Confirm
   new ids are contiguous at the end (49152..49166).
3. Grow `model.model.embed_tokens.weight` from (49152, 960) to (49167→pad to
   **multiple of 64**, 960). New rows = description-mean init (§2/R4): for each
   new token, tokenize its description with the ORIGINAL vocab, index the old
   embedding rows, mean → new row. Pad extra rows (to reach mult-of-64) with the
   overall embedding mean.
4. **Tied embeddings** → only the input embedding matrix exists; the head reuses
   it via `as_linear`, so one resize suffices. Update `model.args.vocab_size`.
5. Save a fresh HF-format dir mlx-lm can reload: use mlx-lm's own save util
   (`mlx_lm.utils.save_model` / `save_config`) or `model.save_weights(...)` +
   copy tokenizer files (`tok._tokenizer.save_pretrained(dir)`) + write
   `config.json` with the new `vocab_size` (pad value) and
   `tie_word_embeddings=true`. Output dir:
   `experiments/special_tokens/base_resized/`.
6. Sanity: reload from that dir, tokenize a §2 prompt, assert each `<|T_*|>` and
   structure token maps to exactly ONE id.

Gotcha to watch: mlx-lm may expect `vocab_size` in config to match the weight
matrix's first dim exactly — keep them equal (both = padded size). If mlx-lm's
`load` complains about tokenizer/embedding mismatch, the padded vocab_size in
config.json must equal the safetensors embed row count.

---

## 7. Fine-tuning (plan)

Use the mlx-lm LoRA CLI with **full fine-tune** on the resized base:
```
python -m mlx_lm lora \
  --model experiments/special_tokens/base_resized \
  --train --data experiments/special_tokens/data_special \
  --fine-tune-type full \
  --batch-size 4 --iters <~2-3 epochs worth> --max-seq-length 1024 \
  --learning-rate 1e-5 --steps-per-report 50 --steps-per-eval 200 \
  --adapter-path experiments/special_tokens/adapter_special
```
- Conservative LR (1e-5..2e-5) + few epochs to limit forgetting (R8-i). Breeze-2
  used 1e-6/4ep at larger scale; 1e-5 reasonable for 360M on ~10k.
- If full FT OOMs on 16 GB, fall back to LoRA (`--fine-tune-type lora
  --num-layers 16`) BUT then the new-token embeddings won't train under LoRA —
  in that case do the GTI grounding stage first (freeze backbone, train only the
  new rows on description→token, a few thousand steps) so the embeddings are
  usable before LoRA. Full FT is preferred precisely because it trains the new
  rows for free.
- `--fine-tune-type full` writes full weights; may need
  `mlx_lm.fuse`/no-fuse depending on version. Verify the trained model reloads
  and that `<|T_*|>` tokens are actually being emitted by generation.
- Repeat for the plain ablation into `adapter_plain` (base can be the ORIGINAL
  unresized model since it has no special tokens).

Bound the run: pick `--iters` for ~2–3 epochs; watch val loss; stop early if it
plateaus. Each full-FT run est. 30–90 min on M1 Pro.

---

## 8. Evaluation & comparison (plan)

Write `experiments/special_tokens/eval_finetuned.py`:
1. Load fine-tuned model. For each of the 100 test_100_v2 items, render the §2
   special-token INPUT (round floats identically to training), `generate` with
   stop on `<|end|>`, temperature ~0 (greedy) / low.
2. Parse output: first `<|T_*|>` → `solution_type` + `has_solution`
   (`<|T_none|>`→False else True); body between `<|sol|>` and `<|end|>` →
   `solution_str` (strip). Handle malformed output (no type token) → default
   has_solution True, solution_type exact_symbolic, empty solution.
3. Emit predictions JSONL with the §3 fields (pull ground_truth* from the test
   file verbatim). Save to
   `results/special_tokens/test_100v2_smollm360_special/predictions.jsonl`.
4. Score: `from src.evaluation import evaluate_solutions` with
   `type_tolerances={"series":0.01,"approx_coef":0.001,"regularized":0.001}`,
   `mode="both"`. Save metrics json next to predictions.
5. Repeat for the plain-ablation model (parse repo basic-style output with
   `src.postprocessing.parse.parse_llm_output`).
6. Build a comparison table: our-special vs our-plain vs gpt-5.4-mini
   baseline/agentic vs gpt-5.4 vs gpt-5.5, overall accuracy + per-type. Read the
   baseline metrics jsons under `results/agentic_pilot_2026-07-07/test_100v2_*`.

**Expected honest result to frame:** a 360M model will likely win the DECISION
types (none/regularized/family/type-id — cheap wins from decision tokens) and
struggle on exact solving (exact_symbolic/series/approx_coef numeric match — the
hard inverse problem). The special-token model should beat its own plain
ablation on decision-driven accuracy; whether it beats GPT baselines overall is
the open question the experiment answers. Report it straight either way.

---

## 9. Report & bookkeeping

- Write `experiments/special_tokens/REPORT.md`: method, token scheme, data,
  training, results table, analysis, novelty statement (§1), limitations, future
  work (the ToolkenGPT solver-toolken loop with masked outputs).
- Update `docs/FEATURES.md` (repo rule: always update it when completing a
  feature).
- Save durable findings to memory
  (`/Users/shahriar/.claude/projects/-Users-shahriar-PV-fun-fred-llm/memory/`),
  one fact per file + MEMORY.md pointer. Candidate memories: the special-token
  method + its result; "correctness for none/regularized/family is a pure
  decision" (already partly implied); MLX env + SmolLM2-360M facts.

---

## 10. Pitfalls already discovered (don't rediscover)

- Regularized test items render with **λ=0** (first-kind) — known repo artifact;
  they're trivially `u=f` as prompted and regularized "accuracy" is decision-only.
- Math-Verify parse needs `$...$` wrapping and creates `real=True` symbols
  (normalize with `.subs()`); the repo's adapter
  `src/llm/math_verify_adapter.py` is the single source of truth
  (`FREDHOLM_LOCAL_DICT`). Our LaTeX output must be parseable by
  `parse_latex_to_sympy`.
- A ruff hook reformats files on write (imports etc.) — re-check file state after
  writes.
- Long floats blow up sequence length → round to 4 sig figs consistently in BOTH
  train and test rendering.
- SmolLM2 ties embeddings — resize once; pad vocab to multiple of 64 for Metal.
- `Date.now`/random unavailable in Workflow scripts (irrelevant here unless a
  workflow is used).

---

## §A. Full research brief (verbatim, for reference)

(Reproduced from the research subagent; numbers/ids above are the distilled
version. Kept here so nothing is lost.)

See §1 for the distilled version. The subagent's full brief covered, in addition
to the above: Lample & Charton ICLR 2020 (arXiv:1912.01412, prefix notation +
digit-level ints, beam-50 integration 96–99%); Charton Linear Algebra
(arXiv:2112.01898, P10/P1000/B1999/FP15 number encodings, and the OOD lesson that
training-distribution *breadth* beats realism — Laplace-trained eigenvalue models
generalize 95–100%, Wigner-trained collapse to 0–26%); xVal (arXiv:2310.02989,
single [NUM] token scaled by value + number head — AVOID, needs architecture
surgery and fails on multimodal targets); Charton advanced computations
(arXiv:2006.06462, transformers predict qualitative properties — precedent for
type classification); Kamienny end-to-end SR (arXiv:2204.10532, floats as 3
tokens sign/mantissa/exponent + BFGS constant refinement 3× boost — precedent for
"generate skeleton, refine constants numerically", our sympy verify can play that
role); number tokenization arXiv:2402.14903 (single-digit / right-to-left
grouping helps most at <1B scale); Hewitt vocab-expansion init theory (mean-init
bounds KL blow-up).
