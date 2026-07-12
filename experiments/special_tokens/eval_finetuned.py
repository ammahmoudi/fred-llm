"""Evaluate fine-tuned FRED-token models on test_100_v2 with the repo evaluator.

Usage:
    python experiments/special_tokens/eval_finetuned.py --variant special
    python experiments/special_tokens/eval_finetuned.py --variant plain
"""

import argparse
import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
REPO = HERE.parent.parent
TEST_JSONL = REPO / "data/prompts/basic/test_100_v2/test_100_samples.jsonl"
SIG_FIGS = 4
TYPE_TOLERANCES = {"series": 0.01, "approx_coef": 0.001, "regularized": 0.001}


def round_latex(latex_str: str) -> str:
    """Round decimal literals in LaTeX to 4 sig figs, train-consistent style.

    Pure string rewrite — parsing test LaTeX with Math-Verify silently drops
    variables in some expressions, so no round-trip through sympy here.
    """

    def repl(m: re.Match) -> str:
        s = f"{float(m.group()):.{SIG_FIGS}g}"
        if "e" in s:  # sympy renders big/small floats as \cdot 10^{n}
            mant, exp = s.split("e")
            return rf"{mant} \cdot 10^{{{int(exp)}}}"
        return s

    return re.sub(r"\d+\.\d+", repl, latex_str)


def special_prompt(md: dict) -> str:
    k, f = round_latex(md["kernel"]), round_latex(md["f"])
    lam = f"{md['lambda_val']:.4g}"
    a, b = (f"{v:.4g}" for v in md["domain"])
    return (
        f"<|fred|> <|kernel|> {k} <|lambda|> {lam} <|rhs|> {f} "
        f"<|domain|> {a} {b} <|solve|>"
    )


def plain_prompt(md: dict) -> str:
    k, f = round_latex(md["kernel"]), round_latex(md["f"])
    lam = f"{md['lambda_val']:.4g}"
    a, b = (f"{v:.4g}" for v in md["domain"])
    return (
        "Solve the Fredholm integral equation of the second kind:\n"
        "u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)\n"
        f"with K(x, t) = {k}, f(x) = {f}, λ = {lam}, domain [{a}, {b}].\n"
        "Respond in the format:\n"
        "SOLUTION: u(x) = [your solution in LaTeX]\n"
        "HAS_SOLUTION: [yes/no]\n"
        "SOLUTION_TYPE: [exact_symbolic/approx_coef/discrete_points/series/"
        "family/regularized/none]"
    )


def parse_special(text: str) -> tuple[str, bool, str]:
    m = re.search(r"<\|T_(\w+)\|>", text)
    stype = m.group(1) if m else "exact_symbolic"
    body = ""
    sm = re.search(r"<\|sol\|>(.*?)(?:<\|end\|>|$)", text, re.S)
    if sm:
        body = sm.group(1).strip()
    return body, stype != "none", stype


def parse_plain(text: str) -> tuple[str, bool, str]:
    from src.postprocessing.parse import parse_llm_output

    p = parse_llm_output(text)
    has_sol = p["has_solution"] if p["has_solution"] is not None else True
    return p["solution_str"] or "", has_sol, p["solution_type"] or "exact_symbolic"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["special", "plain"], required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    from mlx_lm import load, stream_generate

    if args.variant == "special":
        make_prompt, parse = special_prompt, parse_special
    else:
        make_prompt, parse = plain_prompt, parse_plain
    model, tok = load(str(HERE / f"model_{args.variant}"))

    outdir = REPO / f"results/special_tokens/test_100v2_smollm360_{args.variant}"
    outdir.mkdir(parents=True, exist_ok=True)
    pred_path = outdir / "predictions.jsonl"

    items = [json.loads(x) for x in TEST_JSONL.read_text().splitlines()]
    with open(pred_path, "w") as fh:
        for i, item in enumerate(items):
            md = item["metadata"]
            ids = tok.apply_chat_template(
                [{"role": "user", "content": make_prompt(md)}],
                add_generation_prompt=True,
            )
            text = ""
            for resp in stream_generate(model, tok, ids, max_tokens=args.max_tokens):
                text += resp.text
                if "<|end|>" in text:
                    break
            solution_str, has_solution, solution_type = parse(text)
            fh.write(
                json.dumps(
                    {
                        "equation_id": item["equation_id"],
                        "ground_truth": item["ground_truth"],
                        "ground_truth_has_solution": md["has_solution"],
                        "ground_truth_solution_type": md["solution_type"],
                        "ground_truth_domain": md["domain"],
                        "ground_truth_kernel": md["kernel"],
                        "ground_truth_f": md["f"],
                        "ground_truth_lambda": md["lambda_val"],
                        "solution_str": solution_str,
                        "has_solution": has_solution,
                        "solution_type": solution_type,
                        "raw_output": text,
                    }
                )
                + "\n"
            )
            print(
                f"[{i + 1}/{len(items)}] {item['equation_id']}: "
                f"{solution_type} | {solution_str[:60]}"
            )

    from src.evaluation.core import evaluate_solutions

    metrics = evaluate_solutions(
        pred_path, mode="both", type_tolerances=TYPE_TOLERANCES
    )
    metrics_path = outdir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    agg = metrics.get("aggregate", metrics)
    print(json.dumps(agg, indent=2, default=str)[:2000])
    print(f"\npredictions: {pred_path}\nmetrics: {metrics_path}")


if __name__ == "__main__":
    main()
