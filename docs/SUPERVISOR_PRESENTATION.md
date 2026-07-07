# Fred-LLM — Supervisor Presentation Notes (30 min)

**Talk:** *Fred-LLM: Large Language Models for Solving Fredholm Integral Equations of the Second Kind*
**Authors:** Amirhossein Mahmoudi · Shahriar Shariati Motlagh — target: IEEE Transactions on AI (TAI)
**Audience framing:** supervisor whose interests are *spectral methods* and *solving differential/integral equations with LLMs*. Throughout, connect to (a) the classical spectral/Galerkin–Nyström solver landscape and (b) what an LLM can and cannot do relative to a numerical spectral solve.

**Timing budget:** ~24 min talk + ~6 min Q&A. Slide count ~14. Pace ≈ 2 min/slide.

---

## One-line pitch (say this first)
> "We built a benchmark and pipeline to ask a precise question: *where does an LLM sit on the spectrum between a symbolic oracle and a numerical spectral solver for Fredholm integral equations?* The short answer: LLMs handle the **analytical/eigenstructure** part well and the **numerical-quadrature** part not at all — and the stronger model hallucinates confidently on ill-posed cases."

---

## Slide 1 — Title & framing (1 min)
- Fredholm equation of the 2nd kind: **u(x) − λ∫ₐᵇ K(x,t)u(t)dt = f(x)**.
- Why a spectral-methods audience should care: this is *the* operator equation that Galerkin, collocation, and Nyström methods were built to solve. We are not replacing those solvers — we are measuring whether an LLM can produce the **interpretable closed form** and recognize the **solution structure** that numerical methods only approximate.
- **Bridge to DEs:** linear BVPs ↔ Green's-function reformulation ↔ Fredholm IE. Studying LLMs on Fredholm IE is a controlled proxy for "LLMs solving DEs in integral form."

## Slide 2 — Motivation & the gap (2 min)
- Two mature worlds: (1) **classical/spectral numerics** — Nyström with Gauss–Legendre / Clenshaw–Curtis quadrature, Galerkin with global orthogonal bases, Sinc-collocation, wavelets (spectral accuracy on smooth kernels); (2) **learning-based operator solvers** — Neural Integral Equations, FIE-NO (Fourier features + neural operator), Fredholm Neural Networks from fixed-point iteration.
- **Both output discretized solutions or learned operators.** In scientific workflows the *symbolic functional form* still matters — interpretability, downstream analysis, reuse.
- **Underexplored question:** can an LLM emit a *closed-form / structured* candidate solution, and can we *verify it rigorously* — by functional equivalence, not just pointwise fit?
- Enabler: **FIE-500k**, a large symbolic SKFIE dataset with constructed ground truth.

## Slide 3 — Research questions & contributions (2 min)
- **RQ1:** Can LLMs produce correct symbolic solutions to 2nd-kind Fredholm equations?
- **RQ2:** Does the symbolic-vs-numerical character of a problem predict success?
- **RQ3:** Do LLMs recognize ill-posedness / no-solution (avoid hallucination)?
- **Contributions:** (1) a 560k-equation benchmark over **7 solution types** with edge cases; (2) a modular pipeline — data prep → prompting → inference → evaluation; (3) a **structured output schema** enabling automated scoring; (4) a verification-first evaluation (symbolic equivalence + numeric + **residual satisfaction**); (5) a reproducible experimental protocol across models, prompts, formats.

## Slide 4 — The 7 solution types (2 min) — *the spine of the whole talk*
Order them by the spectral-methods lens: from "analytical" to "needs a numerical solve".

| Type | What it demands | Spectral/numerical analogue |
|---|---|---|
| **Exact symbolic** | closed-form via substitution+integration | basis-aligned exact projection |
| **Family (resonance)** | u = uₚ + C·φ, φ spans nullspace | **Fredholm alternative / eigenstructure of the operator** |
| **Approx. coefficient** | parametric fit (weak singularity, boundary layer, oscillatory) | fitted spectral coefficients |
| **Series (Neumann)** | u ≈ f + λKf + λ²K²f + … | truncated operator-Neumann expansion |
| **Discrete points** | sampled (x,y), non-separable kernel, near-resonance | quadrature on a mesh |
| **Regularized (1st kind)** | Tikhonov: argmin ‖Ku−g‖²+α‖u‖² | ill-posed inverse, SVD/spectral filtering |
| **No solution** | detect contradiction, flag `has_solution=no` | spectral obstruction (λ = eigenvalue, range violation) |
- **Punchline to plant now:** the top two are *analytical/eigenstructure* recognition; the bottom five all require an actual *numerical computation*. Remember this split — the results fall exactly along it.

## Slide 5 — The benchmark / dataset (2 min)
- **Base:** FIE-500k — 500k exact-symbolic equations from context-free grammars; 5 functional families × 100k (polynomial, trig, exponential, log, mixed). Each record: K(x,t), f(x), λ, [a,b], and ground-truth u(x).
- **Augmentation:** +60k **edge cases** via 14 strategies / 6 categories / 42 variants — weakly singular & oscillatory kernels, boundary layers, near-resonance, Neumann-series, resonance families, Tikhonov-regularized 1st-kind, and four no-solution constructions (eigenvalue violation, range violation, divergent kernel, disconnected support). **Total: 560,000.**
- **Pre-computed evaluation domain:** hybrid mesh — 50 uniform points **+ injected critical features** (boundaries a,b, midpoint, near-boundary ±10% for boundary layers); non-finite points filtered; family-type evaluated at fixed C ∈ {−1,1,2}.
  - *Spectral aside to mention:* the residual integral is currently sampled on this hybrid mesh; an obvious upgrade is to evaluate the residual on **Gauss–Legendre nodes** for spectrally accurate ∫K u — flag this as a planned improvement.
- Stratified splits over functional family × solution type × edge-case label.

## Slide 6 — Pipeline & methodology (2 min)
Four stages (one figure):
1. **Data prep** — augmentation + format conversion (infix / LaTeX / RPN / tokenized).
2. **Prompt engineering** — 4 styles × 3 edge-case modes + structured schema.
3. **LLM methods** — in-context (zero/few-shot, CoT), tool-assisted code-gen, and LoRA/QLoRA fine-tuning of open-weight models.
4. **Evaluation** — symbolic equivalence + numeric + residual + structural metrics.
- **Prompt styles:** basic (zero-shot, the baseline) · chain-of-thought · few-shot (k stratified demos) · **tool-assisted (emit SymPy/quadrature code)** — note this last one is literally "LLM drives a numerical/spectral solver."
- **Edge-case modes** (information-disclosure axis): `none` (pure inference) · `guardrails` (warn it ill-posed cases are valid) · `hints` (disclose ground-truth type).

## Slide 7 — Evaluation framework (2 min) — *our rigor; the part a numerics person will scrutinize*
- **Category-specific routing:** symbolic types → CAS pipeline; discrete → pointwise numeric only; regularized/none → strict classification (existence flag + failure mode).
- **Symbolic equivalence:** parse prediction & ground truth into a **CAS tree** (math-verify/ANTLR for LaTeX, custom stack parser for RPN, direct infix), inject domain symbols & bounds, test whether P(x)−G(x) reduces to 0 via a simplification cascade — *functional* equivalence, not string match.
- **Numeric:** MAE, RMSE, and **scale-invariant Relative L²** = ‖P−G‖₂/‖G‖₂.
- **Residual verification (physics check):** R(x) = u(x) − λ∫K(x,t)u(t)dt − f(x), sampled for max/mean/RMSE — validates the *integral equation itself*, even without a symbolic ground truth.
- **Structural:** Operator-F1 (multiset of operators: Add, Mul, Integral, sin…), BLEU, None-detection F1.

## Slide 8 — Experimental setup (1–2 min)
- **Models reported:** GPT-4o-mini, GPT-4o, **GPT-5.2 (reasoning)** — all OpenAI, via API.
- **Baseline configuration:** basic zero-shot · **LaTeX** input · `none` edge-case mode · near-greedy decoding (T≈0).
- **Be upfront about scale:** the reported headline numbers are a **balanced pilot of 21 equations (3 per solution type × 7 types)** — enough to expose the per-type structure, not yet the full leaderboard. Scaling to the full ~1,692-equation held-out test set is the funded next step (see Slide 12).
- Full experiment matrix = models × 4 prompt styles × 3 input formats × 3 edge-case modes; ablations vary one factor at a time.

## Slide 9 — Headline results (2 min)

| Model | Overall Acc ↑ | Symbolic ↑ | Median rL² ↓ | Op-F1 ↑ | None-F1 ↑ |
|---|---|---|---|---|---|
| GPT-4o-mini | 15.0% | 0.0% | 4.47 | 33.7 | 0.0 |
| GPT-4o | 19.0% | 4.8% | 1.50 | 26.4 | **50.0** |
| **GPT-5.2** | **30.0%** | **15.0%** | **0.08** | **39.8** | 0.0 |
- GPT-5.2 leads everywhere except none-detection. Use **medians** for rL² — a few catastrophic wrong-form predictions (>10¹⁹) blow up the mean.
- Read rL² for a numerics audience: GPT-5.2's *typical wrong answer* is still within ~8% relative L² — i.e., right functional form, off coefficients; the weaker models' typical error *exceeds the solution's own magnitude* (rL² > 1).

## Slide 10 — Per-type results: the clean boundary (2 min) — *the money slide*

| Solution type | 4o-mini | 4o | 5.2 |
|---|---|---|---|
| **Family (resonance/eigenstructure)** | 100 | 100 | **100** |
| **Exact symbolic** | 0 | 0 | **100** |
| None | 0 | **33** | 0 |
| Approx. coef / Discrete / Series / Regularized | 0 | 0 | 0 |
- **Finding 1 — the analytical/numerical wall.** *Every* model solves **family-type perfectly** — recognizing the nullspace structure u = uₚ + C·φ is **eigenstructure pattern recognition**, exactly the Fredholm-alternative reasoning, and it needs no computation. **Exact symbolic** requires multi-step algebra and only GPT-5.2's extended reasoning gets it. **All four numerical types score 0 for everyone** — no quadrature, no Neumann truncation, no Tikhonov solve, no eigenvalue computation happens inside a forward pass.
- **Say it plainly:** LLMs do the part of a spectral method that is *symbolic/analytical*; they cannot do the part that is *a numerical linear-algebra solve on a discretized operator.*

## Slide 11 — Finding 2: the hallucination asymmetry (2 min)
- **Counterintuitive:** the *strongest* solver (GPT-5.2) **never abstains** — it emits a plausible closed form for *every* input, including provably unsolvable ones (None-F1 = 0). The *mid-tier* GPT-4o catches **1/3 of no-solution cases with perfect precision** (None-F1 = 50).
- These hallucinations **parse cleanly and follow the kernel's functional form** — undetectable without the **residual check**. This is the argument for our residual-verification metric and for the `guardrails`/`hints` modes.
- Spectral-methods analogue: a numerical solver *fails loudly* near a resonance (singular system, blow-up); the LLM *fails silently* with a confident wrong form. That contrast is a genuinely interesting reliability story.

## Slide 12 — Current state: what's done vs. pending (2 min) — *be honest here*
**Done:**
- Full pipeline (data → prompts → inference → eval) and 560k benchmark with augmentation.
- Multi-modal evaluation incl. CAS equivalence, Relative L², residual verification, Operator-F1.
- Baseline: 3 OpenAI models, zero-shot, pilot sample — the per-type structure above.

**Scaffolded but not yet filled (the article has these sections stubbed):**
- Ablations: **prompt style** (CoT / few-shot / tool-assisted), **edge-case mode**, **input format** (infix/LaTeX/RPN), **scale & difficulty** (kernel depth, |λ|, domain width), **decoding sensitivity** (T, reasoning effort, self-consistency).
- **Fine-tuning** (LoRA/QLoRA on LLaMA-3.x, Qwen-Math, Phi-4).
- **Classical baselines — directly your wheelhouse:** Nyström with **Gauss–Legendre & Clenshaw–Curtis** quadrature, and a SymPy CAS upper bound. These establish the *deterministic ceiling* the LLM is measured against.
- Failure-mode taxonomy + qualitative case studies.

**Cost plan (ready):** full 8-model panel over the entire ~1,692-equation test set ≈ **$115**; adding 2 premium reasoning models pushes the "ideal" config to ≈ **$966** (premium models are ~7/8 of the cost). Proposed ask: ~**$150** to secure the core 8-model benchmark, premium as an optional extension.

## Slide 13 — Connections to spectral methods & LLMs-for-DEs (2 min) — *land the supervisor's interests*
- **LLM as symbolic oracle, spectral solver as numerical engine → hybrid.** Operator predictions can serve as **priors that constrain a symbolic-regression / finite-expression-method (FEX) search**, cutting cost and improving interpretability (cf. PDE symbolic-discovery work).
- **Tool-assisted prompting = LLM dispatching a spectral/quadrature solve** — the natural fix for the four numerical types that currently score 0.
- **Resonance/family success** suggests LLMs internalize **Fredholm-alternative / eigenvalue structure** at the pattern level — worth probing whether this generalizes (and whether it survives near-resonance perturbations).
- **Evaluation upgrade:** move residual integration to **Gauss–Legendre nodes** for spectral accuracy; report residuals as the physics-faithful metric independent of symbolic ground truth.

## Slide 14 — Summary & asks (1 min)
- We have a **benchmark + verification-first pipeline** that cleanly separates *analytical* from *numerical* competence in equation solving.
- **Headline:** LLMs solve eigenstructure (family) and, for the strongest, exact symbolic; they solve **nothing** that needs a numerical quadrature/iteration; the strongest model hallucinates on ill-posed inputs.
- **Asks of the supervisor:** (1) sign-off on scaling to the full test set + model panel (~$150); (2) guidance on the **classical spectral baselines** (Nyström quadrature choice, n, Galerkin basis); (3) is the LLM-prior → FEX/spectral-refinement hybrid worth elevating from "future work" to a second contribution?

---

## Anticipated questions (rehearse these)
- **"Why not just run a Nyström/Galerkin solver?"** We are not competing on numerical accuracy — that's the deterministic ceiling we'll benchmark against. We measure whether an LLM can yield an *interpretable closed form* and *recognize solution structure*; the hybrid (LLM prior → spectral refine) is where the two meet.
- **"Why do all models fail the numerical types?"** A single forward pass has no iterative numerical loop — no quadrature, no linear solve. That's exactly what tool-assisted prompting and fine-tuning are meant to fix; it's a capability boundary, not a tuning artifact.
- **"21 equations — is that real?"** It's an honest pilot that already exposes the per-type structure; arithmetic checks out (e.g., GPT-5.2 = 6/21 ≈ 30%). The full ~1,692-test-set run is the immediate, costed next step.
- **"Family-type 100% — leakage or too easy?"** It's structural recognition (nullspace + free constant), not computation; the open question is whether it holds under near-resonance perturbation — a good probe to add.
- **"How spectrally accurate is your evaluation?"** Numeric eval uses a hybrid mesh (uniform + boundary/critical points) and residual sampling; we plan to move residual integration to Gauss–Legendre nodes. Symbol handling uses math-verify with normalized assumptions.
- **"Is the structured-output schema penalizing models?"** Possible — it may disadvantage free-form generation; a stated limitation, and a candidate ablation.

## Backup facts to have on hand
- Equation: u(x) − λ∫ₐᵇK(x,t)u(t)dt = f(x).
- Dataset: 500k base (5 families × 100k) + 60k edge cases = 560k; eval split ~5k, held-out test ~1,692; reported pilot = 21.
- Metrics: Overall/Symbolic Acc, Median Relative L², Operator-F1, BLEU, None-F1; residual R(x).
- Models pending: Gemini Flash/Pro, Gemma-3, Claude Haiku/Sonnet, LLaMA-3.x, Qwen-Math, Phi-4.
