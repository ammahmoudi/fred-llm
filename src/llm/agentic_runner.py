"""
Agentic multi-method solver for Fredholm integral equations.

Wraps a base model runner and, per equation, dispatches parallel "method
specialist" agents (same base model, method-specific directive), verifies
each candidate deterministically with SymPy residual checks, optionally runs
a feedback-driven repair round, and selects a winner. The winner's raw
response is returned so the standard postprocess/evaluate pipeline applies
unchanged. See docs/AGENTIC_SOLVER.md.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn

from src.evaluation.types.verify import verify_solution
from src.llm.math_verify_adapter import parse_latex_to_sympy
from src.llm.model_runner import BaseModelRunner
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Method directives prepended to the original prompt. The original prompt's
# output-format instructions (SOLUTION:/HAS_SOLUTION:/SOLUTION_TYPE:) are
# preserved so the standard parser applies to every candidate.
METHOD_DIRECTIVES: dict[str, str] = {
    "degenerate_kernel": (
        "You are a mathematician who solves Fredholm integral equations by the "
        "DIRECT COMPUTATION METHOD for degenerate (separable) kernels.\n"
        "If K(x,t) = sum_i g_i(x)*h_i(t), write u(x) = f(x) + lambda * sum_i "
        "c_i * g_i(x) with c_i = integral of h_i(t)*u(t) dt, substitute back, "
        "and solve the resulting linear algebraic system for the c_i exactly.\n"
        "If the system is singular and consistent, the solution is a "
        "one-parameter family; if singular and inconsistent, no solution "
        "exists. If the kernel is not separable, say so and do your best with "
        "a close separable approximation.\n\n"
    ),
    "adomian": (
        "You are a mathematician who solves Fredholm integral equations by the "
        "ADOMIAN DECOMPOSITION METHOD.\n"
        "Decompose u(x) = sum_n u_n(x) with u_0(x) = f(x) and "
        "u_{n+1}(x) = lambda * integral of K(x,t)*u_n(t) dt. Compute at least "
        "the first four components. If the series telescopes to a closed form "
        "(e.g. geometric), give the exact closed-form solution; otherwise give "
        "the truncated series.\n\n"
    ),
    "neumann": (
        "You are a mathematician who solves Fredholm integral equations by "
        "SUCCESSIVE APPROXIMATIONS (Neumann series).\n"
        "Iterate u_{n+1}(x) = f(x) + lambda * integral of K(x,t)*u_n(t) dt "
        "starting from u_0(x) = f(x). First check the convergence condition "
        "|lambda| * max|K| * (b - a) < 1 and report it. Sum the series to a "
        "closed form when the pattern is recognizable; otherwise give the "
        "truncated series.\n\n"
    ),
    "fredholm_alternative": (
        "You are a mathematician who analyzes SOLVABILITY of Fredholm integral "
        "equations via the FREDHOLM ALTERNATIVE before solving.\n"
        "Find the characteristic values of the kernel. If lambda is NOT a "
        "characteristic value, a unique solution exists - compute it. If "
        "lambda IS a characteristic value, check whether f is orthogonal to "
        "the eigenfunctions of the adjoint homogeneous equation: if yes, give "
        "the solution family with a free constant C; if no, state that no "
        "solution exists. If the equation is of the FIRST kind (u appears only "
        "under the integral), state that it is ill-posed and requires "
        "regularization.\n\n"
    ),
    "numerical": (
        "You are a numerical analyst who solves Fredholm integral equations "
        "NUMERICALLY.\n"
        "Discretize with the Nystrom method: choose quadrature nodes on the "
        "domain (e.g. Simpson's rule with at least 5 nodes), form the linear "
        "system (I - lambda*K*W)u = f, and solve it. Then fit a simple "
        "symbolic form to the computed values if one is apparent (report it "
        "with numeric coefficients), otherwise report the solution as discrete "
        "points.\n\n"
    ),
}

_REPAIR_TEMPLATE = (
    "\n\n--- PREVIOUS ATTEMPT (INCORRECT) ---\n"
    "You previously answered: u(x) = {solution}\n"
    "Substituting it back into the equation gives a maximum residual of "
    "{residual:.3e}, so it does not satisfy the equation. Find your error, "
    "rework the problem carefully, and answer again in the same required "
    "format."
)


class AgenticModelRunner(BaseModelRunner):
    """Multi-method agentic wrapper around a base model runner."""

    def __init__(
        self,
        base_runner: BaseModelRunner,
        methods: list[str] | None = None,
        max_repair_rounds: int = 1,
        parallel_workers: int = 5,
        equation_workers: int = 2,
        verify_tolerance: float = 1e-6,
        use_math_verify: bool = True,
    ) -> None:
        """
        Initialize agentic runner.

        Args:
            base_runner: Runner that performs the actual LLM calls.
            methods: Method agents to dispatch (keys of METHOD_DIRECTIVES).
                Order doubles as tie-break priority. Defaults to all.
            max_repair_rounds: Feedback retry rounds when nothing verifies.
            parallel_workers: Concurrent LLM calls per equation.
            equation_workers: Equations processed concurrently. Total
                concurrent LLM calls ≈ equation_workers * parallel_workers.
            verify_tolerance: Max-residual threshold for "verified".
            use_math_verify: Passed through to expression parsing.
        """
        super().__init__()
        self.base = base_runner
        self.methods = methods or list(METHOD_DIRECTIVES)
        unknown = set(self.methods) - set(METHOD_DIRECTIVES)
        if unknown:
            raise ValueError(f"Unknown agentic methods: {sorted(unknown)}")
        self.max_repair_rounds = max_repair_rounds
        self.parallel_workers = parallel_workers
        self.equation_workers = equation_workers
        self.verify_tolerance = verify_tolerance
        self.use_math_verify = use_math_verify
        self.traces: list[dict[str, Any]] = []

    def set_cost_tracker(self, tracker: Any) -> None:
        """Delegate cost tracking to the base runner (which makes the calls)."""
        self.cost_tracker = tracker
        self.base.set_cost_tracker(tracker)

    # ------------------------------------------------------------------ #
    # Core workflow                                                      #
    # ------------------------------------------------------------------ #

    def generate(
        self, prompt: str, equation: dict[str, Any] | None = None, **kwargs: Any
    ) -> str:
        """
        Run the multi-method workflow for one equation.

        Args:
            prompt: The original generated prompt (equation + format rules).
            equation: Problem-statement fields for verification:
                ``kernel``, ``f``, ``lambda_val``, ``domain``, optional ``id``.
                Must NOT contain ground-truth fields.
            **kwargs: Passed to the base runner's generate.

        Returns:
            The winning candidate's raw response text ("" if all calls fail).
        """
        t0 = time.time()
        eq_parsed = self._parse_equation(equation)
        candidates = self._dispatch(
            {m: METHOD_DIRECTIVES[m] + prompt for m in self.methods}, **kwargs
        )
        for cand in candidates:
            cand.update(self._verify(cand, eq_parsed))
            cand["round"] = 0

        rounds = [list(candidates)]
        for round_no in range(1, self.max_repair_rounds + 1):
            if any(c["status"] == "verified" for c in candidates):
                break
            repair_prompts = {}
            for cand in candidates:
                if cand["status"] != "failed":
                    continue  # only repair parseable-but-wrong answers
                repair_prompts[cand["method"]] = (
                    METHOD_DIRECTIVES[cand["method"]]
                    + prompt
                    + _REPAIR_TEMPLATE.format(
                        solution=cand["solution_str"],
                        residual=cand["residual_max"],
                    )
                )
            if not repair_prompts:
                break
            repaired = self._dispatch(repair_prompts, **kwargs)
            for cand in repaired:
                cand.update(self._verify(cand, eq_parsed))
                cand["round"] = round_no
            rounds.append(repaired)
            # Repaired answers replace their round-0 counterparts for selection
            by_method = {c["method"]: c for c in candidates}
            for cand in repaired:
                prev = by_method[cand["method"]]
                if self._rank_key(cand) < self._rank_key(prev):
                    by_method[cand["method"]] = cand
            candidates = [by_method[m] for m in self.methods if m in by_method]

        winner, reason = self._select(candidates)
        duration = time.time() - t0
        eq_id = (equation or {}).get("id")
        logger.info(
            f"{eq_id or 'eq'}: winner={winner['method'] if winner else None} "
            f"reason={reason} calls={sum(len(r) for r in rounds)} "
            f"({duration:.0f}s)"
        )
        trace = {
            "equation_id": eq_id,
            "duration_s": round(duration, 1),
            "winner_method": winner["method"] if winner else None,
            "selection_reason": reason,
            "llm_calls": sum(len(r) for r in rounds),
            "candidates": [
                {
                    k: c.get(k)
                    for k in (
                        "method",
                        "round",
                        "status",
                        "residual_max",
                        "solution_str",
                        "solution_type",
                        "has_solution",
                    )
                }
                for r in rounds
                for c in r
            ],
        }
        self.traces.append(trace)
        return winner["response"] if winner else ""

    def batch_generate(
        self,
        prompts: list[str],
        equations: list[dict[str, Any]] | None = None,
        rate_limit_delay: float = 1.0,
        show_progress: bool = True,
        checkpoint_path: str | Path | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Run the agentic workflow for each prompt. Equations run concurrently
        (``equation_workers``) and method agents within an equation run in
        parallel (``parallel_workers``).

        Args:
            prompts: Original generated prompts.
            equations: Per-prompt equation dicts (parallel to ``prompts``).
            rate_limit_delay: Unused (kept for interface compatibility);
                concurrency is bounded by the worker pools instead.
            show_progress: Whether to show a progress spinner.
            checkpoint_path: When set, each completed equation is appended to
                this JSONL immediately (a hung/killed batch loses nothing),
                and existing entries are resumed instead of re-run.
            **kwargs: Passed to the base runner's generate.

        Returns:
            List of winning raw responses, aligned with ``prompts``.
        """
        logger.info(
            f"Agentic batch: {len(prompts)} equations x {len(self.methods)} methods "
            f"({self.equation_workers} equations concurrently)"
        )
        equations = equations or [None] * len(prompts)
        if len(equations) != len(prompts):
            raise ValueError("equations must be aligned with prompts")
        pending = object()  # sentinel: not yet run (distinct from "" = failed)
        results: list[Any] = [pending] * len(prompts)

        checkpoint_lock = threading.Lock()
        checkpoint_file = None
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                for line in checkpoint_path.read_text().splitlines():
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # partial line from a crash
                    i = rec.get("i")
                    if (
                        isinstance(i, int)
                        and 0 <= i < len(prompts)
                        and (equations[i] or {}).get("id") == rec.get("id")
                    ):
                        results[i] = rec.get("response", "")
                        if rec.get("trace"):
                            self.traces.append(rec["trace"])
                resumed = sum(r is not pending for r in results)
                if resumed:
                    logger.info(f"Resumed {resumed} equations from checkpoint")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_file = open(checkpoint_path, "a")

        def work(i: int) -> None:
            if results[i] is not pending:
                return  # resumed from checkpoint
            try:
                results[i] = self.generate(
                    prompts[i], equation=equations[i], **kwargs
                )
            except Exception as e:
                logger.warning(f"Agentic generation failed for prompt {i}: {e}")
                results[i] = ""
            if checkpoint_file is not None:
                eq_id = (equations[i] or {}).get("id")
                with checkpoint_lock:
                    trace = next(
                        (
                            t
                            for t in reversed(self.traces)
                            if t.get("equation_id") == eq_id
                        ),
                        None,
                    )
                    checkpoint_file.write(
                        json.dumps(
                            {
                                "i": i,
                                "id": eq_id,
                                "response": results[i],
                                "trace": trace,
                            }
                        )
                        + "\n"
                    )
                    checkpoint_file.flush()

        def run_all(progress: Progress | None = None, task_id: Any = None) -> None:
            with ThreadPoolExecutor(
                max_workers=max(1, self.equation_workers)
            ) as pool:
                futures = [pool.submit(work, i) for i in range(len(prompts))]
                for future in as_completed(futures):
                    future.result()
                    if progress is not None:
                        progress.advance(task_id)

        try:
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    task = progress.add_task(
                        f"Agentic solving ({len(self.methods)} methods/eq)...",
                        total=len(prompts),
                    )
                    run_all(progress, task)
            else:
                run_all()
        finally:
            if checkpoint_file is not None:
                checkpoint_file.close()

        return [r if isinstance(r, str) else "" for r in results]

    # ------------------------------------------------------------------ #
    # Stages                                                             #
    # ------------------------------------------------------------------ #

    def _dispatch(self, prompts_by_method: dict[str, str], **kwargs: Any) -> list[dict]:
        """Call the base model once per method, in parallel."""

        def call(item: tuple[str, str]) -> dict[str, Any]:
            method, prompt = item
            try:
                response = self.base.generate(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Method agent '{method}' failed: {e}")
                response = ""
            return {"method": method, "response": response}

        with ThreadPoolExecutor(
            max_workers=min(self.parallel_workers, len(prompts_by_method))
        ) as pool:
            return list(pool.map(call, prompts_by_method.items()))

    def _parse_equation(
        self, equation: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """
        Parse equation components to SymPy once per equation (kernels can be
        multi-KB LaTeX; re-parsing per candidate was the old bottleneck).
        Returns None when data is missing or unparseable → candidates become
        "unverifiable" and selection falls back to majority vote.
        """
        if (
            not equation
            or not equation.get("kernel")
            or not equation.get("f")
            or equation.get("lambda_val") is None
        ):
            return None
        try:
            # lambda_val == 0 is the first-kind sentinel (ill_posed
            # augmentation); the second-kind residual formula does not apply,
            # so skip verification and let majority vote decide.
            if float(equation["lambda_val"]) == 0:
                return None
        except (TypeError, ValueError):
            return None
        try:
            return {
                "kernel": parse_latex_to_sympy(
                    str(equation["kernel"]), use_math_verify=self.use_math_verify
                ),
                "f": parse_latex_to_sympy(
                    str(equation["f"]), use_math_verify=self.use_math_verify
                ),
                "lambda_val": float(equation["lambda_val"]),
                "domain": tuple(equation.get("domain") or (0, 1)),
            }
        except Exception as e:
            logger.debug(f"Equation parse failed ({equation.get('id')}): {e}")
            return None

    def _verify(
        self, candidate: dict[str, Any], eq_parsed: dict[str, Any] | None
    ) -> dict[str, Any]:
        """
        Parse a candidate response and residual-check it against the equation.

        Uses numeric-only verification (symbolic=False): sympy.integrate has
        no timeout and dominated wall time; quadrature at a few sample points
        detects wrong solutions just as well.

        Status: "verified" (residual within tolerance), "failed" (parseable
        solution with measured residual above tolerance), "unverifiable"
        (no-solution claims, discrete points, missing equation data, or
        parse/verification errors).
        """
        from src.postprocessing.parse import parse_llm_output

        result: dict[str, Any] = {
            "solution_str": None,
            "solution_type": None,
            "has_solution": None,
            "status": "unverifiable",
            "residual_max": None,
        }
        if not candidate["response"]:
            return result

        try:
            parsed = parse_llm_output(
                candidate["response"], use_math_verify=self.use_math_verify
            )
        except Exception as e:
            logger.debug(f"Candidate parse failed ({candidate['method']}): {e}")
            return result

        result["solution_str"] = parsed.get("solution_str")
        result["solution_type"] = parsed.get("solution_type")
        result["has_solution"] = parsed.get("has_solution")

        if (
            eq_parsed is None
            or result["has_solution"] is False
            or not result["solution_str"]
        ):
            return result

        try:
            expr = parse_latex_to_sympy(
                result["solution_str"], use_math_verify=self.use_math_verify
            )
            # ponytail: free constants (family solutions) are checked at C=1
            # only; true parametric verification if family accuracy matters.
            for sym in expr.free_symbols:
                if sym.name != "x":
                    expr = expr.subs(sym, 1)
            verification = verify_solution(
                expr,
                eq_parsed["kernel"],
                eq_parsed["f"],
                eq_parsed["lambda_val"],
                domain=eq_parsed["domain"],
                tolerance=self.verify_tolerance,
                symbolic=False,
                n_points=12,
            )
            result["residual_max"] = verification["residual_max"]
            result["status"] = (
                "verified" if verification["verified"] else "failed"
            )
        except Exception as e:
            logger.debug(f"Verification errored ({candidate['method']}): {e}")

        return result

    def _rank_key(self, cand: dict[str, Any]) -> tuple:
        """Sort key: verified first, then smallest residual, then method priority."""
        status_rank = {"verified": 0, "failed": 1, "unverifiable": 2}
        residual = cand.get("residual_max")
        return (
            status_rank.get(cand.get("status"), 3),
            residual if residual is not None else float("inf"),
            self.methods.index(cand["method"]),
        )

    def _select(
        self, candidates: list[dict[str, Any]]
    ) -> tuple[dict[str, Any] | None, str]:
        """Pick the winner: verified > majority vote > best effort."""
        live = [c for c in candidates if c["response"]]
        if not live:
            return None, "all_calls_failed"

        verified = [c for c in live if c["status"] == "verified"]
        if verified:
            return min(verified, key=self._rank_key), "verified"

        # Majority vote on (has_solution, solution_type)
        votes: dict[tuple, list[dict]] = {}
        for c in live:
            votes.setdefault((c["has_solution"], c["solution_type"]), []).append(c)
        majority = max(votes.values(), key=len)
        if len(majority) > 1:
            return min(majority, key=self._rank_key), "majority_vote"

        return min(live, key=self._rank_key), "best_effort"
