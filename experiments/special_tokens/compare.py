"""Print the head-to-head comparison table: FRED-token models vs GPT baselines."""

import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent

RUNS = [
    (
        "SmolLM2-360M special-token",
        "results/special_tokens/test_100v2_smollm360_special/metrics.json",
    ),
    (
        "SmolLM2-360M plain (ablation)",
        "results/special_tokens/test_100v2_smollm360_plain/metrics.json",
    ),
    (
        "SmolLM2-360M special + solver (v2)",
        "results/special_tokens/test_100v2_smollm360_special_solver/metrics.json",
    ),
    (
        "SmolLM2-360M plain + solver (v2)",
        "results/special_tokens/test_100v2_smollm360_plain_solver/metrics.json",
    ),
    (
        "sympy solver, no LM router (v2)",
        "results/special_tokens/test_100v2_smollm360_solver_always/metrics.json",
    ),
    (
        "gpt-5.4-mini baseline",
        "results/agentic_pilot_2026-07-07/test_100v2_gpt54mini/metrics_20260708_002101.json",
    ),
    (
        "gpt-5.4-mini agentic",
        "results/agentic_pilot_2026-07-07/test_100v2_gpt54mini_agentic/eval_metrics_20260708_005736.json",
    ),
    (
        "gpt-5.4 baseline",
        "results/agentic_pilot_2026-07-07/test_100v2_gpt54_baseline/eval_metrics_v2_baseline.json",
    ),
    (
        "gpt-5.4 agentic",
        "results/agentic_pilot_2026-07-07/test_100v2_gpt54_agentic/metrics_20260708_122253.json",
    ),
    ("gpt-5.5 baseline", "outputs/test_100v2_gpt55/eval_metrics_v2_baseline.json"),
    (
        "gpt-5.5 agentic",
        "outputs/test_100v2_gpt55_agentic/eval_metrics_v2_agentic.json",
    ),
]
TYPES = [
    "exact_symbolic",
    "approx_coef",
    "series",
    "discrete_points",
    "none",
    "family",
    "regularized",
]


def main() -> None:
    print(
        "| model | evaluator acc (n) | strict acc (/100) | " + " | ".join(TYPES) + " |"
    )
    print("|" + "---|" * (len(TYPES) + 3))
    for name, rel in RUNS:
        path = REPO / rel
        if not path.exists():
            # metric filenames carry timestamps; take any metrics json in the dir
            cands = (
                sorted(path.parent.glob("*metrics*.json"))
                if path.parent.exists()
                else []
            )
            if not cands:
                print(f"| {name} | (missing) |" + " |" * (len(TYPES) + 1))
                continue
            path = cands[0]
        d = json.loads(path.read_text())
        agg = d.get("aggregate", d)
        pt = agg.get("per_type", {})
        cells = []
        correct = 0
        for t in TYPES:
            v = pt.get(t)
            if isinstance(v, dict):
                correct += v.get("correct", 0)
                cells.append(f"{v['accuracy']:.3f}")
            elif isinstance(v, (int, float)):
                cells.append(f"{v:.3f}")  # older metrics: only accuracy known
            else:
                cells.append("-")
        n = agg.get("evaluated_count") or agg.get("total") or 0
        # strict = dropped items count as wrong; derivable only when correct
        # counts are present, else reconstruct from accuracy * n
        if not correct and agg.get("accuracy") is not None and n:
            correct = round(agg["accuracy"] * n)
        strict = f"{correct / 100:.3f}" if correct else "-"
        print(
            f"| {name} | {agg.get('accuracy'):.3f} ({n}) | {strict} | "
            + " | ".join(cells)
            + " |"
        )


if __name__ == "__main__":
    main()
