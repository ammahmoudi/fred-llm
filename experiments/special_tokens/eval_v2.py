"""v2 eval: decision tokens as tool triggers routing to the sympy solver.

Reuses the v1 models' stored raw outputs (no regeneration): when the model's
decision is a solvable type, the harness parses the equation from the prompt
metadata, runs the separable-kernel solver, and splices the result in as the
solution body. Also scores a no-LM "solver always" row to isolate the router's
contribution.
"""

import json
import warnings
from pathlib import Path

import sympy as sp

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
REPO = HERE.parent.parent
TEST_JSONL = REPO / "data/prompts/basic/test_100_v2/test_100_samples.jsonl"
RESULTS = REPO / "results/special_tokens"
SOLVABLE = {"exact_symbolic", "approx_coef", "series"}
TYPE_TOLERANCES = {"series": 0.01, "approx_coef": 0.001, "regularized": 0.001}


def main() -> None:
    from eval_finetuned import parse_plain, parse_special
    from solver import parse_eq_latex, solve_separable

    from src.evaluation.core import evaluate_solutions

    test_items = {
        json.loads(x)["equation_id"]: json.loads(x)
        for x in TEST_JSONL.read_text().splitlines()
    }

    solve_cache: dict[str, sp.Expr | None] = {}

    def solve_item(eq_id: str) -> sp.Expr | None:
        if eq_id not in solve_cache:
            md = test_items[eq_id]["metadata"]
            kernel = parse_eq_latex(md["kernel"])
            f = parse_eq_latex(md["f"])
            u = None
            if kernel is not None and f is not None:
                u = solve_separable(
                    kernel, f, md["lambda_val"], md["domain"][0], md["domain"][1]
                )
            solve_cache[eq_id] = u
            print(f"  solve {eq_id}: {'ok' if u is not None else 'skip'}")
        return solve_cache[eq_id]

    configs = []
    for variant, parse in (("special", parse_special), ("plain", parse_plain)):
        preds = [
            json.loads(x)
            for x in (RESULTS / f"test_100v2_smollm360_{variant}/predictions.jsonl")
            .read_text()
            .splitlines()
        ]
        configs.append((f"{variant}_solver", preds, parse))
    configs.append(("solver_always", None, None))

    for name, preds, parse in configs:
        outdir = RESULTS / f"test_100v2_smollm360_{name}"
        outdir.mkdir(parents=True, exist_ok=True)
        rows = []
        if name == "solver_always":
            for eq_id, item in test_items.items():
                md = item["metadata"]
                u = solve_item(eq_id)
                rows.append(
                    {
                        "equation_id": eq_id,
                        "ground_truth": item["ground_truth"],
                        "ground_truth_has_solution": md["has_solution"],
                        "ground_truth_solution_type": md["solution_type"],
                        "ground_truth_domain": md["domain"],
                        "solution_str": sp.latex(u) if u is not None else "",
                        "has_solution": True,
                        "solution_type": "exact_symbolic",
                        "solver_used": u is not None,
                    }
                )
        else:
            for pred in preds:
                body, has_sol, stype = parse(pred["raw_output"])
                solver_used = False
                if has_sol and stype in SOLVABLE:
                    u = solve_item(pred["equation_id"])
                    if u is not None:
                        body = sp.latex(u)
                        solver_used = True
                rows.append(
                    {
                        **{k: v for k, v in pred.items() if k != "raw_output"},
                        "solution_str": body,
                        "has_solution": has_sol,
                        "solution_type": stype,
                        "solver_used": solver_used,
                    }
                )
        pred_path = outdir / "predictions.jsonl"
        with open(pred_path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        n_solved = sum(r["solver_used"] for r in rows)
        print(f"== {name}: solver used on {n_solved}/{len(rows)} items")
        metrics = evaluate_solutions(
            pred_path, mode="both", type_tolerances=TYPE_TOLERANCES
        )
        (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        pt = {k: round(v["accuracy"], 3) for k, v in metrics["per_type"].items()}
        print(
            f"   accuracy {metrics['accuracy']:.3f} "
            f"({metrics['evaluated_count']} evaluated) per-type {pt}"
        )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE))
    main()
