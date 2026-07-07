---
title: "Cost Estimate: Evaluating Language Models on the Fred-LLM Benchmark"
author: "Amirhossein Mahmoudi · Shahriar Shariati Motlagh"
date: "31 May 2026"
geometry: "a4paper, margin=0.85in"
fontsize: 10pt
linkcolor: black
header-includes:
  - \usepackage{booktabs}
  - \renewcommand{\arraystretch}{1.05}
  - \setlength{\parskip}{4pt}
---

This note estimates the cost of evaluating a panel of current language models on
the Fred-LLM benchmark through the OpenRouter API. The figures rest on token counts
measured in our own pilot runs and on OpenRouter prices from 31 May 2026, and they
reproduce our already-billed pilot costs to within about two percent. The main
point is that the experiment scales down gracefully: a complete benchmark of eight
efficient and open-weight models over the full test set costs about **US\$115**, and
only two premium reasoning models carry the ideal configuration up toward
**US\$966**.

## What it costs

Two choices set the cost: how many of the 1,692 held-out test equations we
evaluate, and whether we include the two premium reasoning models, GPT-5.5 and
Claude Opus 4.8. Every equation is run through all four prompting styles, with one
response each. Table 1 puts the affordable eight-model panel next to the full
ten-model panel as the evaluation set grows. The conservative column uses
90th-percentile token counts rather than averages.

Table: Cost by evaluation size, all four prompting styles (USD).

| Equations | Affordable 8 | Full 10 (expected) | Full 10 (conservative) |
|---|---:|---:|---:|
| 21 | \$1.43 | \$11.99 | \$24.15 |
| 100 | \$6.81 | \$57.07 | \$114.99 |
| 250 | \$17.02 | \$142.69 | \$287.47 |
| 500 | \$34.03 | \$285.37 | \$574.95 |
| 1,000 | \$68.06 | \$570.74 | \$1,149.90 |
| **1,692 (full test set)** | **\$115.16** | **\$965.69** | **\$1,945.63** |

The two premium models account for roughly seven eighths of the full-panel total.
The other eight evaluate the entire test set for about US\$115, which makes them the
natural core of the study.

## A staged plan

The work can be funded in steps, each one yielding a complete result and stoppable
at any point.

- **Pilot, about US\$1.50.** The eight affordable models on a 21-equation balanced
  sample. A working leaderboard that also checks the pipeline before any larger
  outlay.
- **Preliminary, about US\$7.** The same models on 100 equations, enough for first
  accuracy figures broken down by solution type.
- **Full affordable benchmark, about US\$115.** The eight models on all 1,692 test
  equations. On its own a complete, publishable comparison; this is the firm target.
- **Premium extension, about US\$57 to trial.** Add GPT-5.5 and Claude Opus 4.8 on
  100 equations first, then to the full test set for the ideal configuration at
  about US\$966 (ceiling US\$1,950).

We propose requesting about **US\$150** to secure the eight-model benchmark, and
treating the two premium models as an extension to pursue if their early results
justify the cost. The expensive part is optional, and most of the science is in
place without it.

## Models and method

The panel spans proprietary and open-weight families across price tiers (Table 2).
Prompt length is stable across models, near 465 tokens; what varies is output
length. A direct-answering model produces about 460 tokens per response, a
reasoning model about 2,100, and that difference, together with higher per-token
prices, is why the two reasoning models dominate the total. We evaluate the
held-out test set, not the full 560,000-equation corpus, which would cost far more
and is not the object of study. Prices are an OpenRouter snapshot and should be
refreshed before any commitment.

Table: Model panel and OpenRouter prices (USD per million tokens, 31 May 2026).

| Model | Access | Prompt | Completion |
|---|---|---:|---:|
| openai/gpt-5.5 (reasoning) | proprietary | 5.00 | 30.00 |
| anthropic/claude-opus-4.8 (reasoning) | proprietary | 5.00 | 25.00 |
| google/gemini-3.5-flash | proprietary | 1.50 | 9.00 |
| openai/gpt-5.4-mini | proprietary | 0.75 | 4.50 |
| google/gemini-3.1-flash-lite | proprietary | 0.25 | 1.50 |
| qwen/qwen3.7-max | open-weight | 1.25 | 3.75 |
| mistralai/mistral-medium-3.5 | open-weight | 1.50 | 7.50 |
| deepseek/deepseek-v4-pro | open-weight | 0.435 | 0.87 |
| google/gemma-4-31b-it | open-weight | 0.12 | 0.37 |
| qwen/qwen3.5-9b | open-weight | 0.04 | 0.15 |
