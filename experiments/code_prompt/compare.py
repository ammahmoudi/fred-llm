"""Comparison table for the code-prompt experiment vs prior methods.

Collects metrics from prior test_100_v2 runs (baseline, agentic, fine-tuned
SmolLM2, solver router) and the new code / code_exec runs, and prints a
markdown table. Missing files are skipped, so it can run before all runs land.
"""

import glob
import json
from pathlib import Path

REPO = Path(__file__).parent.parent.parent

ROWS = [
    ("gpt-5.4-mini", "baseline (basic prompt)",
     "results/agentic_pilot_2026-07-07/test_100v2_gpt54mini/metrics_20260708_002101.json"),
    ("gpt-5.4-mini", "agentic multi-method",
     "results/agentic_pilot_2026-07-07/test_100v2_gpt54mini_agentic/eval_metrics_20260708_005736.json"),
    ("gpt-5.4-mini", "code prompt (repr only)",
     "results/code_prompt_2026-07-13/test_100v2_gpt54mini_code/metrics_*.json"),
    ("gpt-5.4-mini", "code_exec (PoT, executed)",
     "results/code_prompt_2026-07-13/test_100v2_gpt54mini_code_exec/metrics_*.json"),
    ("gpt-5.4", "baseline (basic prompt)",
     "results/agentic_pilot_2026-07-07/test_100v2_gpt54_baseline/eval_metrics_v2_baseline.json"),
    ("gpt-5.4", "agentic multi-method",
     "results/agentic_pilot_2026-07-07/test_100v2_gpt54_agentic/metrics_20260708_135701.json"),
    ("gpt-5.4", "code prompt (repr only)",
     "results/code_prompt_2026-07-13/test_100v2_gpt54_code/metrics_*.json"),
    ("gpt-5.4", "code_exec (PoT, executed)",
     "results/code_prompt_2026-07-13/test_100v2_gpt54_code_exec/metrics_*.json"),
    ("SmolLM2-360M", "plain SFT",
     "results/special_tokens/test_100v2_smollm360_plain/metrics.json"),
    ("SmolLM2-360M", "plain SFT + solver routing",
     "results/special_tokens/test_100v2_smollm360_plain_solver/metrics.json"),
    ("(no LM)", "separable solver always",
     "results/special_tokens/test_100v2_smollm360_solver_always/metrics.json"),
]


def load(pattern: str) -> dict | None:
    hits = sorted(glob.glob(str(REPO / pattern)))
    if not hits:
        return None
    return json.loads(Path(hits[-1]).read_text())


def main() -> None:
    print("| Model | Method | Correct/100 | Acc (evaluated) | Type acc | Has-sol acc |")
    print("|---|---|---|---|---|---|")
    for model, method, pattern in ROWS:
        m = load(pattern)
        if m is None:
            print(f"| {model} | {method} | _not run_ | | | |")
            continue
        n_total = m.get("total_predictions") or m.get("total_results") or m["total"]
        print(
            f"| {model} | {method} | {m['correct']}/{n_total} "
            f"| {m['accuracy']:.3f} ({m['correct']}/{m['total']}) "
            f"| {m.get('solution_type_accuracy', float('nan')):.3f} "
            f"| {m.get('has_solution_accuracy', float('nan')):.3f} |"
        )


if __name__ == "__main__":
    main()
