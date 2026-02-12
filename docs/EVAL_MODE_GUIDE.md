# Evaluation-Only Mode Guide

This document explains how to use the Fred-LLM pipeline in **evaluation-only mode**, where you can evaluate pre-existing LLM predictions without running inference.

## Important: Predictions vs Evaluated Predictions

**Predictions file** (`predictions_*.jsonl`) from LLM inference:
- Contains: LLM outputs, ground truth, model confidence
- **Does NOT contain**: RMSE, MAE, symbolic/numeric comparison results

**Evaluated predictions file** (`evaluated_predictions_*.jsonl`) after evaluation:
- Contains: Same as above **PLUS** all evaluation metrics
- Includes: RMSE, MAE, max_error, symbolic match status, numeric match status

When you run only the **inference stage** (Stage 4), you get predictions without metrics. Use **evaluation-only mode** to add evaluation metrics to them.

## Quick Start: Evaluation-Only Mode

If you already have LLM predictions (from another pipeline run or external source), you can evaluate them directly:

```bash
python -m src.cli evaluate predictions.jsonl --output metrics.json
```

This reads `predictions.jsonl` (which has no evaluation metrics) and creates:
- `evaluated_predictions_*.jsonl` - Predictions **with** RMSE, MAE, and comparison results
- `metrics.json` - Summary statistics

Or using the adaptive pipeline configuration:

```bash
python -m src.cli run --config configs/eval_only.yaml
```

## Pipeline Modes Compared

The Fred-LLM pipeline supports 4 automation levels:

| Mode | Input | Steps | Use Case |
|------|-------|-------|----------|
| **full** | Raw dataset | Augment → Split → Convert → Prompts → Inference → Eval | Development, experimentation |
| **partial** | Pre-split data | Prompts → Inference → Eval | When data is ready |
| **manual** | Pre-generated prompts | Inference → Eval | When prompts are fixed |
| **eval-only** | LLM predictions | Eval only | Analyzing existing results |

## Eval-Only Configuration

### Using Pre-configured Template

A pre-configured eval-only template is provided:

```bash
python -m src.cli run --config configs/eval_only.yaml
```

This uses sample predictions from `data/samples/sample_predictions.jsonl`.

### Create Custom Config

Create your own config (e.g., `my_eval.yaml`):

```yaml
dataset:
  evaluation_only:
    predictions_path: outputs/my_run/predictions.jsonl

evaluation:
  mode: both
  symbolic_tolerance: 1e-10
  numeric_tolerance: 1e-6
  num_test_points: 100

output:
  dir: outputs/eval_results
  save_metrics: true
```

Then run:

```bash
python -m src.cli run --config my_eval.yaml
```

### Direct CLI Usage

```bash
# Basic usage
python -m src.cli evaluate predictions.jsonl

# With custom tolerances and output
python -m src.cli evaluate predictions.jsonl \
  --mode both \
  --symbolic-tolerance 1e-10 \
  --numeric-tolerance 1e-6 \
  --test-points 100 \
  --output results/metrics.json
```

## Input File Format: LLM Predictions

### JSONL Format (Line-delimited JSON)

Each line is a JSON object with prediction data:

```jsonl
{
  "equation_id": "eq_0",
  "prompt": "Solve: u(x) - ∫K(x,t)u(t)dt = f(x)",
  "ground_truth": "5.266805598437712",
  "ground_truth_has_solution": true,
  "ground_truth_solution_type": "scalar",
  "ground_truth_domain": [0, 1],
  "solution_str": "5.267",
  "solution_sympy": "5.267",
  "has_solution": true,
  "solution_type": "scalar",
  "raw_response": "The integral equation simplifies to u(x) = 5.267",
  "confidence": 0.95,
  "evaluation_points": {
    "u": "5.267",
    "points": [0.1, 0.2, 0.3, 0.4, 0.5]
  }
}
{
  "equation_id": "eq_1",
  "prompt": "Solve: u(x) - 2*∫(x-t)*u(t)dt = x",
  "ground_truth": "x + 2*sin(x)",
  "ground_truth_has_solution": true,
  "ground_truth_solution_type": "elementary",
  "ground_truth_domain": [0, 1],
  "solution_str": "x + 2*sin(x)",
  "solution_sympy": "x + 2*sin(x)",
  "has_solution": true,
  "solution_type": "elementary",
  "raw_response": "Using variation of parameters...",
  "confidence": 0.92
}
```

### JSON Format (List of Objects)

```json
[
  {
    "equation_id": "eq_0",
    "prompt": "...",
    "ground_truth": "...",
    "solution_str": "...",
    ...
  },
  {
    "equation_id": "eq_1",
    ...
  }
]
```

## Required vs Optional Fields

### Required Fields for Evaluation

- `equation_id` - Unique identifier for the equation
- `ground_truth` - True solution (LaTeX string)
- `solution_str` - Predicted solution (LaTeX string)

### Recommended Fields

- `ground_truth_solution_type` - Type: "scalar", "elementary", "series", etc.
- `ground_truth_domain` - Domain tuple: [0, 1]
- `solution_type` - Predicted solution type
- `evaluation_points` - Custom evaluation points:
  ```json
  {
    "u": "solution_expression",
    "points": [0.1, 0.2, 0.3, ...]
  }
  ```

### Optional Fields

- `prompt` - Original prompt (for reference)
- `raw_response` - Raw LLM output
- `confidence` - Model confidence score (for analysis)
- `has_solution` - Whether solution exists

## Output File Format: Metrics

After evaluation, metrics are saved as JSON:

```json
{
  "mode": "both",
  "total": 4,
  "correct": 2,
  "accuracy": 0.5,
  "symbolic_accuracy": 0.75,
  "numeric_accuracy": 0.25,
  "evaluated_count": 4,
  "total_results": 5,
  "parse_errors": 1,
  "per_type": {
    "scalar": {
      "total": 2,
      "correct": 2,
      "accuracy": 1.0
    },
    "elementary": {
      "total": 2,
      "correct": 0,
      "accuracy": 0.0
    }
  },
  "has_solution_accuracy": 1.0,
  "has_solution_total": 4,
  "solution_type_accuracy": 0.75,
  "solution_type_total": 4,
  "confusion_matrix": {
    "scalar_predicted_as_elementary": 1
  }
}
```

## Step-by-Step Workflow

### Step 1: Generate LLM Predictions

Run inference to get predictions (using full, partial, or manual modes):

```bash
python -m src.cli run --config inference_config.yaml
```

This saves predictions to a file like:
```
outputs/run_20260212_140313/predictions_20260212_140313.jsonl
```

### Step 2: Evaluate Predictions

Run evaluation with the predictions file:

```bash
python -m src.cli evaluate outputs/run_20260212_140313/predictions_20260212_140313.jsonl \
  --output outputs/eval_results/metrics.json
```

### Step 3: Analyze Results

Load and analyze the metrics:

```python
import json

with open("outputs/eval_results/metrics.json") as f:
    metrics = json.load(f)

print(f"Overall accuracy: {metrics['accuracy']:.2%}")
print(f"Symbolic: {metrics['symbolic_accuracy']:.2%}")
print(f"Numeric: {metrics['numeric_accuracy']:.2%}")

# Per-type breakdown
for stype, counts in metrics.get('per_type', {}).items():
    acc = counts['correct'] / counts['total']
    print(f"{stype}: {acc:.2%} ({counts['correct']}/{counts['total']})")
```

## Example: Re-evaluate Existing Results

### Scenario 1: Different Tolerances

You have predictions from run A, but want to re-evaluate with stricter tolerances:

```bash
# Original evaluation with loose tolerance
python -m src.cli evaluate predictions.jsonl --numeric-tolerance 1e-4

# Re-evaluate with stricter tolerance
python -m src.cli evaluate predictions.jsonl --numeric-tolerance 1e-8 \
  --output metrics_strict.json
```

### Scenario 2: Development Environment

You're testing improvements:

```bash
# Config for dev evaluation
cat > dev_eval_config.yaml << 'EOF'
dataset:
  evaluation_only:
    predictions_path: outputs/dev/predictions.jsonl

evaluation:
  mode: both
  symbolic_tolerance: 1e-10
  numeric_tolerance: 1e-6
  num_test_points: 50  # Fewer points for faster evaluation

output:
  dir: outputs/dev_evaluation
  save_metrics: true
EOF

python -m src.cli run --config dev_eval_config.yaml
```

### Scenario 3: Comparing Multiple Runs

```bash
#!/bin/bash

# Evaluate multiple prediction files
for pred_file in outputs/run_*/predictions_*.jsonl; do
  output_file="${pred_file/predictions/eval_metrics}"
  echo "Evaluating $pred_file..."
  python -m src.cli evaluate "$pred_file" --output "$output_file"
done

# Compare results
python << 'EOF'
import json
import glob

for metrics_file in sorted(glob.glob("outputs/run_*/eval_metrics_*.json")):
    with open(metrics_file) as f:
        metrics = json.load(f)
    print(f"{metrics_file}: {metrics['accuracy']:.2%} accuracy")
EOF
```

## Configuration Examples

### Minimal Config (Eval Only)

```yaml
dataset:
  evaluation_only:
    predictions_path: predictions.jsonl
```

### Full Config (Eval Only)

```yaml
dataset:
  evaluation_only:
    predictions_path: outputs/predictions.jsonl

evaluation:
  mode: both
  symbolic_tolerance: 1e-10
  numeric_tolerance: 1e-6
  num_test_points: 100
  use_math_verify: true
  type_tolerances:
    series: 1e-2
    approx_coef: 1e-3
    regularized: 1e-3

output:
  dir: outputs/evaluation
  save_metrics: true
  log_level: INFO
```

## Understanding Evaluation Metrics

### Accuracy Scores

- **Overall Accuracy**: $(correct / total) \times 100\%$
- **Symbolic Accuracy**: Correct via symbolic comparison (e.g., exact match, simplification)
- **Numeric Accuracy**: Correct via numeric comparison (evaluated at test points)

### Per-Type Metrics

For different solution types (scalar, elementary, series, etc.):
- `total`: How many predictions of this type
- `correct`: How many were correct
- `accuracy`: $correct / total$

### Edge Case Metrics

- **has_solution_accuracy**: Correctly predicted whether solution exists
- **solution_type_accuracy**: Correctly predicted the solution type

## Troubleshooting

### "File not found" Error

```bash
# Check file exists
ls -la predictions.jsonl

# Verify JSON format
python -c "import json; json.load(open('predictions.jsonl'))"
```

### "Parse errors" in Output

If you see `"parse_errors": 5` in metrics, some predictions couldn't be parsed:

```python
import json

# Check which equations failed
with open("predictions.jsonl") as f:
    for i, line in enumerate(f):
        pred = json.loads(line)
        try:
            solution = pred['solution_str']
            if not solution:
                print(f"Equation {pred['equation_id']}: Missing solution_str")
        except KeyError as e:
            print(f"Equation {i}: Missing field {e}")
```

### Timeout During Evaluation

For large prediction files, use fewer test points:

```bash
python -m src.cli evaluate large_predictions.jsonl \
  --test-points 10 \
  --output metrics.json
```

## Performance Notes

- **Evaluation time** depends on:
  - Number of predictions
  - Number of test points (default: 100)
  - Complexity of symbolic expressions
  - Whether symbolic simplification is used

- **For 1000 predictions with 100 test points**: ~10-30 seconds

- **To speed up evaluation**:
  ```bash
  # Fewer test points
  --test-points 10
  
  # Numeric mode only (skip symbolic)
  --mode numeric
  ```

## API Documentation

### evaluate() Command

```bash
python -m src.cli evaluate INPUT_FILE [OPTIONS]
```

**Arguments:**
- `INPUT_FILE`: Path to predictions file (.json or .jsonl)

**Options:**
- `--mode {symbolic, numeric, both}`: Evaluation mode (default: both)
- `--symbolic-tolerance FLOAT`: Tolerance for symbolic comparison (default: 1e-10)
- `--numeric-tolerance FLOAT`: Tolerance for numeric comparison (default: 1e-6)
- `--test-points INT`: Number of test points (default: 100)
- `--output PATH`: Save metrics to file (optional)

### AdaptivePipeline with eval-only mode

```python
from src.adaptive_config import AdaptivePipelineConfig
from src.adaptive_pipeline import AdaptivePipeline

config = AdaptivePipelineConfig.from_yaml("eval_config.yaml")
pipeline = AdaptivePipeline(config)
results = pipeline.run()  # Runs evaluation only

print(metrics := results['metrics'])
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

## Related Documentation

- [Pipeline Architecture](pipeline-diagram.md)
- [Evaluation Strategy](EVALUATION_STRATEGY_REVIEW.md)
- [LLM Input/Output Formats](LLM_INPUT_OUTPUT.md)

