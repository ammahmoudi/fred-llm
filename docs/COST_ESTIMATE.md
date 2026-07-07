# Full Benchmark Cost Estimate (OpenRouter)

**Prepared:** 2026-05-31 · **Prices:** OpenRouter `/api/v1/models`, fetched 2026-05-31 (USD)
**Reproduce:** `uv run python scripts/cost_estimate.py`

This estimates the OpenRouter dollar cost of running the Fredholm benchmark at full
scale across several SOTA models (closed + open). It is grounded in **measured token
usage** from 2,200+ real calls already logged in `outputs/*/cost_details_*.jsonl`, not
guesses.

---

## 1. Headline — cost by sample size

Every sample size below is run through **all 4 prompt styles** (`basic` +
`chain-of-thought` + `few-shot` + `tool-assisted`), one response per prompt, across a
**10-model SOTA panel (5 closed + 5 open)**. So requests per model = sample × 4.

| Sample size | Requests / model | Total requests (×10) | **Expected** | **Conservative** |
|---|---:|---:|---:|---:|
| 21 (`diverse_21`) | 84 | 840 | **$11.99** | $24.15 |
| 100 (`test_100`) | 400 | 4,000 | **$57.07** | $114.99 |
| 250 | 1,000 | 10,000 | **$142.69** | $287.47 |
| 500 | 2,000 | 20,000 | **$285.37** | $574.95 |
| 1,000 | 4,000 | 40,000 | **$570.74** | $1,149.90 |
| **1,692 (full test split)** | 6,768 | 67,680 | **$965.69** | $1,945.63 |
| 5,000 (full sample) | 20,000 | 200,000 | **$2,853.71** | $5,749.49 |

**Bottom line for the proposal:** the full test split (1,692 eq), all 4 prompt styles,
across 10 SOTA models, costs **~$966 expected (~$1,950 worst-case)**. A 500-equation
balanced subset gives strong coverage for **~$285**. Budget **$1,200** to run the full
test split with headroom, or **$400** for the 500-eq version.

Per-model breakdown of the all-4-styles run (shows what drives the total):

| Model | tier | n=100 | n=500 | n=1,692 | n=5,000 |
|---|---|---:|---:|---:|---:|
| openai/gpt-5.5 | closed reasoning | $27.30 | $136.49 | **$461.87** | $1,364.85 |
| anthropic/claude-opus-4.8 | closed reasoning | $22.97 | $114.86 | **$388.67** | $1,148.55 |
| google/gemini-3.5-flash | closed | $2.11 | $10.54 | $35.67 | $105.40 |
| mistralai/mistral-medium-3.5 | open | $1.82 | $9.12 | $30.86 | $91.19 |
| openai/gpt-5.4-mini | closed | $1.05 | $5.27 | $17.83 | $52.70 |
| qwen/qwen3.7-max | open | $1.05 | $5.23 | $17.70 | $52.30 |
| google/gemini-3.1-flash-lite | closed | $0.35 | $1.76 | $5.94 | $17.57 |
| deepseek/deepseek-v4-pro | open | $0.28 | $1.41 | $4.76 | $14.08 |
| google/gemma-4-31b-it | open small | $0.10 | $0.51 | $1.73 | $5.12 |
| qwen/qwen3.5-9b | open small | $0.04 | $0.20 | $0.66 | $1.96 |

The two flagship reasoning models (GPT-5.5 + Claude Opus 4.8) are **~88% of every total**;
the other 8 models combined run the full test split (all 4 styles) for **~$115**.

> The full 500k-equation dataset is **cost-prohibitive** (~$70k–$142k across the panel)
> and is not the intended "benchmark" — it is the training/source pool.

---

## 2. What drives the cost: reasoning models dominate

~90% of the panel cost comes from 2–3 **flagship reasoning** models, because they emit
4–5× more output tokens **and** are priced 5–20× higher per output token.

| Model | $/1k requests (expected) | Test split, 1 config (1,692) |
|---|---:|---:|
| openai/gpt-5.5 (reasoning) | $67.28 | $113.85 |
| anthropic/claude-opus-4.8 (reasoning) | $56.47 | $95.55 |
| openai/gpt-5.4 (reasoning) | $33.64 | $56.91 |
| x-ai/grok-4.3 (reasoning) | $6.01 | $10.17 |
| google/gemini-3.5-flash | $4.98 | $8.43 |
| mistralai/mistral-medium-3.5 | $4.27 | $7.23 |
| openai/gpt-5.4-mini | $2.49 | $4.22 |
| qwen/qwen3.7-max | $2.38 | $4.02 |
| google/gemini-3.1-flash-lite | $0.83 | $1.41 |
| deepseek/deepseek-v4-pro | $0.62 | $1.05 |
| google/gemma-4-31b-it | $0.23 | $0.39 |
| qwen/qwen3.5-9b | $0.09 | $0.15 |

> The **entire open-weight + efficient-closed set (7 models)** runs the test split for
> **under ~$30 total**. The cost question is really "how many frontier reasoning models
> do we include, and at what reasoning effort / output cap."

---

## 3. Measured inputs (the basis for every number)

From `outputs/*/cost_details_*.jsonl` (Feb 2026 runs, `basic` prompt style):

| Quantity | Expected (mean) | Conservative (p90) | Source |
|---|---:|---:|---|
| Input tokens / req (test split) | 465 | 515 | `test_100` (n=700): mean 463, p90 514 |
| Input tokens / req (balanced) | 615 | 760 | `diverse_21` (n=63): mean 613 |
| Output tokens — **standard** model | 460 | 730 | gpt-4o/4o-mini: mean 420–470, p90 ~600–730 |
| Output tokens — **reasoning** model | 2,100 | 4,500 | gpt-5.2 @ effort=low: mean 1,935–2,318, p90 4–5k (max hit the 16,384 cap) |

**Validation against actually-billed runs** (confidence check):

| Run | Model | Predicted (this model) | Actually recorded |
|---|---|---:|---:|
| `test_100_gpt4o` | gpt-4o | $0.50 / 100 req | $0.494–$0.506 |
| `test_100_gpt4omini` | gpt-4o-mini | $0.035 / 100 req | $0.034–$0.036 |

The model reproduces real OpenRouter charges to within ~2%.

---

## 4. Assumptions & cost levers

**Fixed facts from the codebase:**
- 1 response per prompt (no self-consistency / `n>1`) → no hidden multiplier.
- Retries up to 3× on failure (tenacity); modeled as a flat **+3%** operational buffer.
- `max_tokens`: 2,048 standard, **16,384 for reasoning models** — the reasoning cap is
  why GPT-5.x/Opus output (and cost) can spike; some calls already hit it.

**Multipliers you control (each is linear in cost):**
| Lever | Effect |
|---|---|
| Dataset size | 1,692 (test) → 5,000 (sample) = **3×**; → 500,000 (raw) = **296×** |
| Prompt styles | basic/CoT/tool ≈ 1×; **few-shot ≈ 2.6× input tokens** (embedded examples) |
| Edge-case modes (none/guardrails/hints) | up to **3×** if all run |
| Full prompt matrix (4 styles × 3 modes) | up to **12×** |
| Repetitions per prompt | 3 runs = **3×**, 5 runs = **5×** |
| Reasoning effort / output cap | lowering effort or `max_tokens` cuts reasoning output (the dominant term) |

**Caveats:** OpenRouter prices change. The prices in `scripts/cost_estimate.py` are
constants captured from the catalog on 2026-05-31 — refresh the `MODELS` table from
`https://openrouter.ai/api/v1/models` before committing a budget. Reasoning-model output tokens are
high-variance; the "conservative" column uses p90 token counts, not worst case. Few-shot
input inflation is applied as a blended 2.6× only in the 4-style scenario.

---

## 5. Recommended plan to give the professor

All options use **all 4 prompt styles**, mode `none`, 1 rep, the 10-model panel.

1. **Full benchmark:** test split (1,692) × 4 styles × 10 models → **~$966 expected
   (budget $1,200 with headroom).**
2. **Lean alternative:** 500-equation balanced subset × 4 styles × 10 models → **~$285**
   — strong coverage at a third of the cost.
3. **Cost control:** ~88% of the bill is GPT-5.5 + Claude Opus 4.8. Dropping to one
   frontier reasoning model, or capping reasoning effort / `max_tokens`, roughly halves
   the total. The other 8 models run the full test split for **~$115**.

Skip the full 500k pass entirely — report it as the source pool, benchmark on the
stratified test split.
