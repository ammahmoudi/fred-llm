# Agentic Solver

Multi-agent workflow for solving Fredholm integral equations: parallel "method
specialist" agents each attempt a different classical solution technique, a
deterministic SymPy verifier checks every candidate against the original
equation, failed candidates get one feedback-driven repair round, and a
selector picks the winner. The final answer flows through the **unchanged**
postprocessing and evaluation pipeline, so agentic runs are directly comparable
with single-prompt baseline runs.

## Motivation

A human mathematician doesn't commit to one technique. They look at the
equation, try direct computation if the kernel is separable, fall back to a
Neumann series, check the Fredholm alternative when λ looks like a
characteristic value — and they *verify by substituting back*. The agentic
workflow reproduces exactly this behavior with LLM agents:

```
                       ┌─────────────────────────┐
                       │   original prompt +      │
                       │   equation (K, f, λ, [a,b])│
                       └────────────┬────────────┘
              ┌─────────┬──────────┼──────────┬───────────┐
              ▼         ▼          ▼          ▼           ▼
        ┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐
        │degenerate│ │adomian │ │neumann │ │fredholm_ │ │numerical │
        │_kernel   │ │        │ │        │ │alternative│ │          │
        └────┬─────┘ └───┬────┘ └───┬────┘ └────┬─────┘ └────┬─────┘
             └───────────┴─────┬────┴───────────┴────────────┘
                               ▼   (parallel LLM calls, same base model)
                    ┌────────────────────┐
                    │ SymPy verifier      │  residual of
                    │ (deterministic,     │  u(x) − λ∫K(x,t)u(t)dt − f(x)
                    │  no LLM, no GT)     │
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ repair round (≤1)   │  failed candidates retry with
                    │                     │  residual feedback
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ selector            │  verified & smallest residual,
                    │ (deterministic)     │  else majority vote
                    └─────────┬──────────┘
                              ▼
                 winner's raw response → parse_llm_output → evaluate
                 (identical to baseline path)
```

## Framework Decision

Candidates evaluated (state of the ecosystem, mid-2026 — see sources below):

| Framework | Verdict | Reason |
|---|---|---|
| **LangGraph** | runner-up | Model-agnostic graph runtime, would work; but our workflow is a static fan-out → verify → repair → select DAG. No checkpointing/streaming/human-in-loop needed. Adds 2+ dependencies and a learning curve for zero extra capability here. |
| **CrewAI** | rejected | Role-play scaffolding injects up to ~3× token overhead — an experimental **confound**: we could no longer attribute accuracy gains to the workflow vs. the hidden prompt scaffolding. |
| **OpenAI Agents SDK** | rejected | Handoff-centric (sequential control transfer), not fan-out/fan-in; OpenAI-centric. |
| **Claude Agent SDK** | rejected | Claude-only. Our experiments compare many models through OpenRouter; the agentic layer must run on the *same* model as the baseline for a fair comparison. |
| **AutoGen/AG2** | rejected | Conversation-centric multi-agent chat; heavier abstraction than needed. |
| **Thin custom orchestrator** (chosen) | ✅ | ~1 file on top of the existing `ModelRunner`. Zero new dependencies (`openai` + `ThreadPoolExecutor` + `tenacity` retries already in the stack). Full prompt control (no hidden scaffolding), exact per-call cost accounting via the existing `CostTracker`, and byte-identical output contract with the baseline. |

**Why this matters scientifically:** the research question is "does agentic
orchestration beat single-prompt inference *for the same model*?" Any framework
that injects its own prompts, retries, or role narration adds uncontrolled
variables. A thin orchestrator keeps the only difference between baseline and
agentic runs the workflow itself. If the workflow later needs cycles,
checkpointing, or dynamic planning, LangGraph is the documented upgrade path.

## Method Agents

Each agent is the **same base LLM** with a method-specific directive prepended
to the original prompt (so the output-format instructions from the prompt style
are preserved). The roster covers the classical solution methods for Fredholm
equations (Wazwaz, *Linear and Nonlinear Integral Equations*; see sources) and
maps onto the project's 7 solution types:

| Agent | Classical method | Primarily targets |
|---|---|---|
| `degenerate_kernel` | Direct computation: separable kernel K(x,t)=Σgᵢ(x)hᵢ(t) reduces the equation to a linear algebraic system for the coefficients cᵢ = ∫hᵢ(t)u(t)dt | `exact_symbolic`; singular system → `family` / `none` |
| `adomian` | Adomian decomposition: u = Σuₙ with u₀ = f, uₙ₊₁ = λ∫K uₙ dt; often telescopes to closed form | `exact_symbolic`, `series` |
| `neumann` | Successive approximations / Neumann series, with the convergence check \|λ\|·M·(b−a) < 1 | `series`, `exact_symbolic` |
| `fredholm_alternative` | Solvability analysis: characteristic values, orthogonality of f to the adjoint eigenfunctions; detects first-kind (ill-posed) equations | `family`, `none`, `regularized` |
| `numerical` | Nyström / collocation with quadrature, then fit a symbolic form | `approx_coef`, `discrete_points` |

## Workflow Stages (per equation)

1. **Dispatch** — the configured method agents are called in parallel
   (`ThreadPoolExecutor`; the sync OpenAI client is thread-safe, tenacity
   retries stay per-call).
2. **Verify** — each candidate is parsed with the existing `parse_llm_output`
   and checked with the existing `verify_solution`
   (`src/evaluation/types/verify.py`) in numeric-only mode (12 quadrature
   sample points; `sympy.integrate` has no timeout and dominated wall time
   when tried first). The equation components are parsed to SymPy once per
   equation, not per candidate. Free symbols other than `x` (e.g. family
   constant `C`) are substituted with test values. **The verifier sees only
   the equation (K, f, λ, domain) — never the ground truth.**
3. **Repair** (≤ `max_repair_rounds`, default 1) — if no candidate verifies,
   failing agents are re-prompted with their previous answer and its measured
   residual as feedback.
4. **Select** — deterministic:
   - any verified candidate → smallest max-residual wins;
   - none verified → majority vote on `(has_solution, solution_type)`,
     ties broken by residual, then by method priority
     (`degenerate_kernel` > `fredholm_alternative` > `adomian` > `neumann` > `numerical`);
   - all calls failed → empty response (same as baseline API-error behavior).
5. **Emit** — the winner's *raw response text* becomes `raw_response`, so
   `parse_llm_output` → predictions JSONL → `SolutionEvaluator` run unchanged.

## Evaluation Integration & Fairness

- **Same model, same prompts** — agents wrap the original generated prompt;
  the base model and provider come from the same `model:` config the baseline
  uses. Compare runs by toggling one config key.
- **Same output contract** — predictions keep the standard 15-key schema
  (`equation_id`, …, `solution_str`, `has_solution`, `solution_type`, …);
  `_evaluate` and `evaluate_solutions` need zero changes.
- **Cost accounting** — all sub-calls go through the base runner, so the
  existing `CostTracker` captures every call. Expect roughly
  `len(methods) × (1 + repair_rate)` × baseline cost; the trace file records
  per-equation call counts so cost-vs-accuracy plots are straightforward.
- **No ground-truth leakage** — the orchestrator receives only
  `kernel`, `f`, `lambda_val`, `domain` from prompt metadata;
  `has_solution` / `solution_type` are never passed in.
- **Trace** — `agentic_trace_<ts>.jsonl` logs every candidate, verification
  verdict, repair round and the selection reason per equation, for qualitative
  analysis (e.g. which methods win on which solution types).

## Configuration

```yaml
model:
  provider: openrouter          # same providers as baseline
  name: openai/gpt-4o-mini
  agentic:                      # presence of this section enables agentic mode
    methods:                    # optional, default: all 5
      - degenerate_kernel
      - adomian
      - neumann
      - fredholm_alternative
      - numerical
    max_repair_rounds: 1        # 0 disables repair
    parallel_workers: 5         # concurrent LLM calls per equation
    verify_tolerance: 1.0e-6    # max-residual threshold for "verified"
```

See `configs/diverse_21_gpt4omini_agentic.yaml` for a runnable example — it is
the agentic twin of `configs/diverse_21_gpt4omini.yaml`; removing the
`agentic:` section reproduces the baseline exactly.

## Limitations / Next Steps

- Repair is capped at one round; deeper solver–verifier–corrector loops (as in
  the IMO-2025 self-verification pipelines) are a natural extension.
- The verifier substitutes numeric values for free constants; genuinely
  parametric verification of `family` solutions is approximate.
- Truncated `series` candidates never verify to tolerance by construction —
  they compete on residual magnitude instead.
- Equations run concurrently (`equation_workers`, default 2) on top of the
  per-equation method parallelism; total in-flight calls ≈
  `equation_workers × parallel_workers`.
- Dataset caveat (found in the first pilot run, 2026-07-07): the three
  `regularized` items in `diverse_21` carry `lambda_val = 0.0` and ground
  truth `0` — the first-kind equation is serialized into the second-kind
  template, so the rendered problem is the trivial `u(x) = f(x)`. As
  presented, no solver (baseline or agentic) can produce the expected
  `regularized` answer; those 3 items measure a rendering artifact, not
  model ability.

## Sources

Framework landscape:
- [2026 AI Agent Framework Showdown: LangGraph vs CrewAI vs AG2 vs Claude SDK vs OpenAI (QubitTool)](https://qubittool.com/blog/ai-agent-framework-comparison-2026)
- [Agentic AI Frameworks 2026: LangGraph vs CrewAI vs OpenAI SDK (Uvik)](https://uvik.net/blog/agentic-ai-frameworks/)
- [Best Multi-Agent Frameworks in 2026 (GuruSup)](https://gurusup.com/blog/best-multi-agent-frameworks-2026)
- [Best open source frameworks for building AI agents in 2026 (Firecrawl)](https://www.firecrawl.dev/blog/best-open-source-agent-frameworks)

Multi-agent math solving (solver/verifier/corrector, parallel aggregation):
- [MarsRL: Advancing Multi-Agent Reasoning System via RL with Agentic Pipeline Parallelism (arXiv 2511.11373)](https://arxiv.org/pdf/2511.11373)
- [Verification-Aware Planning for Multi-Agent Systems (arXiv 2510.17109)](https://arxiv.org/pdf/2510.17109)
- [MathChat: Converse to Tackle Challenging Math Problems with LLM Agents (arXiv 2306.01337)](https://arxiv.org/pdf/2306.01337)

Fredholm solution methods:
- Wazwaz, *Linear and Nonlinear Integral Equations* — [Fredholm chapter (Springer)](https://link.springer.com/chapter/10.1007/978-3-642-21449-3_4)
- [Numerical Methods for Solving Fredholm Integral Equations of Second Kind (Ray, 2013)](https://onlinelibrary.wiley.com/doi/10.1155/2013/426916)
- [Adomian Decomposition Method for Fredholm equations (comparison study)](https://ijmttjournal.org/archive/ijmtt-v66i6p525)
