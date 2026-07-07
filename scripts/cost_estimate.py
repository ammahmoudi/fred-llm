"""OpenRouter cost estimator for the full Fredholm benchmark.

Token assumptions are derived from MEASURED data in outputs/*/cost_details_*.jsonl
(2,200+ real OpenRouter calls, Feb 2026). Prices are USD per 1M tokens, pulled
live from the OpenRouter /api/v1/models catalog on 2026-05-31.

Run:  uv run python scripts/cost_estimate.py
Adjust the CONSTANTS block to match the exact benchmark you intend to run.
"""

# ---------------------------------------------------------------------------
# 1. MEASURED TOKEN PROFILES  (from outputs/*/cost_details_*.jsonl)
# ---------------------------------------------------------------------------
# Prompt (input) tokens are stable per dataset because every model sees the same
# prompts. Completion (output) tokens depend almost entirely on whether the model
# "thinks" (reasoning) or answers directly (standard).
TOKENS = {
    # input tokens per request, "basic" prompt style
    "input_test_split": {"expected": 465, "conservative": 515},   # test_100 runs: mean 463, p90 514
    "input_diverse":    {"expected": 615, "conservative": 760},   # diverse_21 runs: mean 613 (long-eq outliers)
    # output tokens per request
    "out_standard":  {"expected": 460,  "conservative": 730},     # gpt-4o / 4o-mini: mean ~420-470, p90 ~600-730
    "out_reasoning": {"expected": 2100, "conservative": 4500},    # gpt-5.2 @ effort=low: mean ~1935-2318, p90 ~4-5k (max hit 16384 cap)
}

RETRY_BUFFER = 1.03   # tenacity retries up to 3x on failure; ~3% operational overhead
FEWSHOT_INPUT_MULT = 2.6  # few-shot style inflates INPUT tokens (examples embedded)

# ---------------------------------------------------------------------------
# 2. LIVE OPENROUTER PRICES  (USD per 1,000,000 tokens, verified 2026-05-31)
#    profile: which output-token assumption applies to this model
# ---------------------------------------------------------------------------
MODELS = [
    # name                       in     out    profile      tier
    ("openai/gpt-5.5",           5.00,  30.00, "reasoning", "closed-frontier"),
    ("openai/gpt-5.4",           2.50,  15.00, "reasoning", "closed-frontier"),
    ("openai/gpt-5.4-mini",      0.75,   4.50, "standard",  "closed-efficient"),
    ("openai/gpt-5.4-nano",      0.20,   1.25, "standard",  "closed-efficient"),
    ("anthropic/claude-opus-4.8",5.00,  25.00, "reasoning", "closed-frontier"),
    ("google/gemini-3.5-flash",  1.50,   9.00, "standard",  "closed-efficient"),
    ("google/gemini-3.1-flash-lite",0.25,1.50, "standard",  "closed-efficient"),
    ("deepseek/deepseek-v4-pro", 0.435,  0.87, "standard",  "open-frontier"),
    ("deepseek/deepseek-v4-flash",0.0983,0.1966,"standard", "open-efficient"),
    ("qwen/qwen3.7-max",         1.25,   3.75, "standard",  "open-frontier"),
    ("qwen/qwen3.6-plus",        0.325,  1.95, "standard",  "open-efficient"),
    ("x-ai/grok-4.3",            1.25,   2.50, "reasoning", "open-frontier"),
    ("mistralai/mistral-medium-3.5",1.50,7.50, "standard",  "open-frontier"),
    ("google/gemma-4-31b-it",    0.12,   0.37, "standard",  "open-small"),
    ("qwen/qwen3.5-9b",          0.04,   0.15, "standard",  "open-small"),
]

# A representative "several SOTA, closed + open" panel for the headline number.
PANEL = [
    "openai/gpt-5.5", "openai/gpt-5.4-mini", "anthropic/claude-opus-4.8",
    "google/gemini-3.5-flash", "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-pro", "qwen/qwen3.7-max", "mistralai/mistral-medium-3.5",
    "google/gemma-4-31b-it", "qwen/qwen3.5-9b",
]

# ---------------------------------------------------------------------------
# 3. BENCHMARK SCOPE
# ---------------------------------------------------------------------------
DATASETS = {
    "diverse_21 (balanced)":     21,
    "test_100":                  100,
    "test split (canonical)":    1692,
    "full sample":               5000,
    "full raw (everything)":     500000,
}


def cost_per_request(in_price, out_price, profile, level, dataset="test"):
    in_key = "input_test_split" if dataset == "test" else "input_diverse"
    in_tok = TOKENS[in_key][level]
    out_tok = TOKENS["out_reasoning" if profile == "reasoning" else "out_standard"][level]
    return (in_tok / 1e6) * in_price + (out_tok / 1e6) * out_price


def fmt(x):
    return f"${x:,.2f}" if x >= 1 else f"${x:.4f}"


print("=" * 92)
print("UNIT COST PER 1,000 REQUESTS  (test-split prompts, basic style, 1 response each)")
print("=" * 92)
print(f"{'model':<32}{'profile':<11}{'expected':>14}{'conservative':>16}")
print("-" * 92)
by_name = {m[0]: m for m in MODELS}
for name, ip, op, prof, tier in MODELS:
    e = cost_per_request(ip, op, prof, "expected") * 1000 * RETRY_BUFFER
    c = cost_per_request(ip, op, prof, "conservative") * 1000 * RETRY_BUFFER
    print(f"{name:<32}{prof:<11}{fmt(e):>14}{fmt(c):>16}")

print()
print("=" * 92)
print("HEADLINE: full benchmark across the 10-model SOTA panel (closed + open)")
print("=" * 92)
scenarios = [
    ("Test split, 1 config (1,692 req/model)",     1692, 1),
    ("Test split, 4 prompt styles (6,768 req/model)", 1692, 4),
    ("Full sample, 1 config (5,000 req/model)",    5000, 1),
]
for label, n_eq, configs in scenarios:
    print(f"\n--- {label} ---")
    total_e = total_c = 0.0
    for name in PANEL:
        _, ip, op, prof, tier = by_name[name]
        # few-shot present when 4 styles -> blend input multiplier across 4 styles
        style_mult = ((1 + 1 + FEWSHOT_INPUT_MULT + 1) / 4) if configs == 4 else 1.0
        reqs = n_eq * configs
        e = cost_per_request(ip * style_mult, op, prof, "expected") * reqs * RETRY_BUFFER
        c = cost_per_request(ip * style_mult, op, prof, "conservative") * reqs * RETRY_BUFFER
        total_e += e
        total_c += c
        print(f"  {name:<32}{fmt(e):>12}  ..  {fmt(c):>12}")
    print(f"  {'PANEL TOTAL':<32}{fmt(total_e):>12}  ..  {fmt(total_c):>12}")

print()
print("=" * 104)
print("ALL-4-STYLES BENCHMARK: cost by sample size, 10-model panel")
print("(each equation run through basic + chain-of-thought + few-shot + tool-assisted)")
print("=" * 104)
SAMPLE_SIZES = [21, 100, 250, 500, 1000, 1692, 5000]
STYLE_COUNT = 4
# avg INPUT inflation across the 4 styles (few-shot embeds examples, others ~1x)
STYLE_INPUT_BLEND = (1 + 1 + FEWSHOT_INPUT_MULT + 1) / 4

hdr = f"{'model':<26}" + "".join(f"{f'n={s:,}':>11}" for s in SAMPLE_SIZES)
print(hdr)
print("-" * 104)
tot_e = {s: 0.0 for s in SAMPLE_SIZES}
tot_c = {s: 0.0 for s in SAMPLE_SIZES}
for name in PANEL:
    _, ip, op, prof, tier = by_name[name]
    row = f"{name:<26}"
    for s in SAMPLE_SIZES:
        reqs = s * STYLE_COUNT
        e = cost_per_request(ip * STYLE_INPUT_BLEND, op, prof, "expected") * reqs * RETRY_BUFFER
        c = cost_per_request(ip * STYLE_INPUT_BLEND, op, prof, "conservative") * reqs * RETRY_BUFFER
        tot_e[s] += e
        tot_c[s] += c
        row += f"{fmt(e):>11}"
    print(row)
print("-" * 104)
print(f"{'PANEL TOTAL (expected)':<26}" + "".join(f"{fmt(tot_e[s]):>11}" for s in SAMPLE_SIZES))
print(f"{'PANEL TOTAL (conservative)':<26}" + "".join(f"{fmt(tot_c[s]):>11}" for s in SAMPLE_SIZES))
print(f"{'requests / model':<26}" + "".join(f"{s*STYLE_COUNT:>11,}" for s in SAMPLE_SIZES))
print(f"{'requests total (x10)':<26}" + "".join(f"{s*STYLE_COUNT*len(PANEL):>11,}" for s in SAMPLE_SIZES))

print()
print("Add edge-case modes (none/guardrails/hints): multiply totals by up to 3x.")
print("Add repetitions: multiply by the number of runs per prompt.")

print()
print("=" * 92)
print("STAGED BUDGET: affordable-8 (no flagship reasoning) vs full 10-model panel")
print("all 4 prompt styles, 1 response each")
print("=" * 92)
FLAGSHIPS = ["openai/gpt-5.5", "anthropic/claude-opus-4.8"]
AFFORDABLE = [m for m in PANEL if m not in FLAGSHIPS]


def panel_cost(models, n, level):
    tot = 0.0
    for name in models:
        _, ip, op, prof, _ = by_name[name]
        reqs = n * STYLE_COUNT
        tot += cost_per_request(ip * STYLE_INPUT_BLEND, op, prof, level) * reqs * RETRY_BUFFER
    return tot


print(f"{'sample':>8}  {'affordable-8 (exp)':>20}  {'full-10 (exp)':>16}  {'full-10 (cons)':>16}")
print("-" * 92)
for n in [21, 100, 250, 500, 1000, 1692]:
    a = panel_cost(AFFORDABLE, n, "expected")
    f = panel_cost(PANEL, n, "expected")
    fc = panel_cost(PANEL, n, "conservative")
    print(f"{n:>8,}  {fmt(a):>20}  {fmt(f):>16}  {fmt(fc):>16}")
