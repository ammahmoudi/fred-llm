"""Build special-token and plain-ablation training data for the FRED-token experiment.

Reads the raw Fredholm CSV, generates edge-case variants via the repo's
augmentation strategies, renders everything as rounded LaTeX, and emits two
dataset variants (special-token format and plain-NL ablation) as
{"prompt","completion"} JSONL for mlx_lm lora --mask-prompt.
"""

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
REPO = HERE.parent.parent
RAW_CSV = REPO / "data/raw/Fredholm_Dataset_Sample.csv"
TEST_JSONL = REPO / "data/prompts/basic/test_100_v2/test_100_samples.jsonl"

TYPE_TARGETS = {
    "exact_symbolic": 3000,
    "none": 1800,
    "approx_coef": 1500,
    "series": 1200,
    "regularized": 1200,
    "family": 1200,
    "discrete_points": 1200,
}
MAX_TOKENS = 1000  # train max-seq-length is 1024
SIG_FIGS = 4
SEED = 42

FIXED_BODIES = {
    "none": "No solution",
    "regularized": "requires regularization",
    "discrete_points": "discrete point values",
}


def round_floats(expr: sp.Expr) -> sp.Expr:
    return expr.xreplace({f: sp.Float(f, SIG_FIGS) for f in expr.atoms(sp.Float)})


def to_latex(expr_str: str) -> str:
    expr = sp.sympify(expr_str)
    return sp.latex(round_floats(expr))


def render(item: dict) -> tuple[dict, dict] | None:
    """Render one augmented/raw item to (special, plain) prompt/completion pairs."""
    try:
        k_ltx = to_latex(item["kernel"])
        f_ltx = to_latex(item["f"])
        lam = f"{float(item['lambda_val']):.4g}"
        a = f"{float(item['a']):.4g}"
        b = f"{float(item['b']):.4g}"
        stype = item["solution_type"]
        body = FIXED_BODIES.get(stype) or to_latex(item["u"])
    except Exception:
        return None
    if not body.strip():
        return None

    special = {
        "prompt": (
            f"<|fred|> <|kernel|> {k_ltx} <|lambda|> {lam} <|rhs|> {f_ltx} "
            f"<|domain|> {a} {b} <|solve|>"
        ),
        "completion": f"<|T_{stype}|> <|sol|> {body} <|end|>",
    }
    has_sol = "no" if stype == "none" else "yes"
    plain = {
        "prompt": (
            "Solve the Fredholm integral equation of the second kind:\n"
            "u(x) - λ ∫_a^b K(x, t) u(t) dt = f(x)\n"
            f"with K(x, t) = {k_ltx}, f(x) = {f_ltx}, λ = {lam}, "
            f"domain [{a}, {b}].\n"
            "Respond in the format:\n"
            "SOLUTION: u(x) = [your solution in LaTeX]\n"
            "HAS_SOLUTION: [yes/no]\n"
            "SOLUTION_TYPE: [exact_symbolic/approx_coef/discrete_points/series/"
            "family/regularized/none]"
        ),
        "completion": (
            f"SOLUTION: u(x) = {body}\nHAS_SOLUTION: {has_sol}\nSOLUTION_TYPE: {stype}"
        ),
    }
    return special, plain


def main() -> dict:
    from transformers import AutoTokenizer

    from src.data.augmentation import _apply_augmentation

    tok = AutoTokenizer.from_pretrained("mlx-community/SmolLM2-360M-Instruct")

    def fits(pair: dict) -> bool:
        toks = tok.apply_chat_template(
            [
                {"role": "user", "content": pair["prompt"]},
                {"role": "assistant", "content": pair["completion"]},
            ],
            tokenize=True,
            return_dict=False,  # BatchEncoding is not a dict; len() would be 2
        )
        return len(toks) <= MAX_TOKENS

    # -- load raw rows, exclude test overlap ---------------------------------
    df = pd.read_csv(RAW_CSV, skipinitialspace=True)
    test_lambdas = set()
    for line in TEST_JSONL.read_text().splitlines():
        md = json.loads(line)["metadata"]
        test_lambdas.add(round(md["lambda_val"], 6))
    rows = [
        {
            "u": str(r["u"]).strip(),
            "f": str(r["f"]).strip(),
            "kernel": str(r["kernel"]).strip(),
            "lambda_val": float(r["lambda"]),
            "a": float(r["a"]),
            "b": float(r["b"]),
        }
        for _, r in df.iterrows()
        if round(float(r["lambda"]), 6) not in test_lambdas
    ]
    print(f"raw rows after test-overlap exclusion: {len(rows)}")

    rng = random.Random(SEED)
    np.random.seed(SEED)
    rng.shuffle(rows)

    # -- collect examples per type -------------------------------------------
    buckets: dict[str, list[tuple[dict, dict]]] = {t: [] for t in TYPE_TARGETS}

    for row in rows:
        if len(buckets["exact_symbolic"]) >= TYPE_TARGETS["exact_symbolic"]:
            break
        pair = render({**row, "solution_type": "exact_symbolic"})
        if pair and fits(pair[0]) and fits(pair[1]):
            buckets["exact_symbolic"].append(pair)

    strategy_of = {
        "none": "none_solution",
        "approx_coef": "approx_coef",
        "series": "series",
        "regularized": "regularized",
        "family": "family",
        "discrete_points": "discrete_points",
    }
    aug_pool = rows[TYPE_TARGETS["exact_symbolic"] :]
    for i, row in enumerate(aug_pool):
        needed = [t for t in strategy_of if len(buckets[t]) < TYPE_TARGETS[t]]
        if not needed:
            break
        for t in needed:
            try:
                outs = _apply_augmentation(dict(row), strategy_of[t])
            except Exception:
                continue
            for out in outs:
                if len(buckets[t]) >= TYPE_TARGETS[t]:
                    break
                if out.get("solution_type") != t:
                    continue
                pair = render(out)
                if pair and fits(pair[0]) and fits(pair[1]):
                    buckets[t].append(pair)
        if i % 50 == 0:
            print(
                f"aug base {i}: "
                + " ".join(f"{t}={len(v)}" for t, v in buckets.items())
            )

    counts = {t: len(v) for t, v in buckets.items()}
    print("final counts:", counts)

    # -- stratified 90/10 split, write both variants -------------------------
    for variant, idx in (("data_special", 0), ("data_plain", 1)):
        train, valid = [], []
        for pairs in buckets.values():
            examples = [p[idx] for p in pairs]
            rng.shuffle(examples)
            n_valid = max(1, len(examples) // 10)
            valid += examples[:n_valid]
            train += examples[n_valid:]
        rng.shuffle(train)
        rng.shuffle(valid)
        outdir = HERE / variant
        outdir.mkdir(exist_ok=True)
        for name, data in (("train", train), ("valid", valid)):
            with open(outdir / f"{name}.jsonl", "w") as fh:
                for ex in data:
                    fh.write(json.dumps(ex) + "\n")
        print(f"{variant}: train={len(train)} valid={len(valid)}")
    return counts


if __name__ == "__main__":
    main()
    # self-check: round-trip, decision-token-first completions, all types present
    types_seen = set()
    for variant in ("data_special", "data_plain"):
        for split in ("train", "valid"):
            for line in (HERE / variant / f"{split}.jsonl").read_text().splitlines():
                ex = json.loads(line)
                assert ex["prompt"] and ex["completion"]
                if variant == "data_special":
                    assert ex["completion"].startswith("<|T_"), ex["completion"][:60]
                    if split == "train":
                        types_seen.add(ex["completion"].split("|>")[0][4:])
    assert types_seen == set(TYPE_TARGETS), types_seen
    print("self-check OK:", sorted(types_seen))
