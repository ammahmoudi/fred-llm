# Pipeline Data Formats Reference

This document shows example data structures at each stage of the Fred-LLM pipeline.

## Full Pipeline Data Flow

```
Raw Data → Prepared Data → Prompts → Predictions → Metrics
   ↓          ↓               ↓           ↓           ↓
  CSV/       train/         JSONL       JSONL        JSON
  JSON       val/test         +          +
             CSV             LaTeX      Ground
                                        Truth
```

## Stage 1: Raw Dataset

**File Format:** JSON or CSV

### JSON Format

```json
{
  "equations": [
    {
      "id": "eq_0",
      "u": "sin(x)",
      "f": "sin(x)",
      "kernel": "exp(x*t)",
      "lambda_val": 0.5,
      "a": 0,
      "b": 1,
      "has_solution": true,
      "solution_type": "exact_symbolic"
    },
    {
      "id": "eq_1",
      "u": "x + 2*sin(x)",
      "f": "x",
      "kernel": "2*(x-t)",
      "lambda_val": 1.0,
      "a": 0,
      "b": 1,
      "has_solution": true,
      "solution_type": "exact_symbolic"
    }
  ]
}
```

### CSV Format

```csv
id,u,f,kernel,lambda_val,a,b,has_solution,solution_type
eq_0,sin(x),sin(x),exp(x*t),0.5,0,1,true,exact_symbolic
eq_1,x + 2*sin(x),x,2*(x-t),1.0,0,1,true,exact_symbolic
```

## Stage 2: Prepared Data

**File Format:** CSV or JSON (train/val/test splits)

**Location:** `data/processed/run_<timestamp>/`

### Training Data Format

```csv
id,equation_id,u,f,kernel,lambda_val,a,b,has_solution,solution_type,augmentation_type,augmentation_variant,evaluation_points
eq_0_train,eq_0,sin(x),sin(x),exp(x*t),0.5,0,1,true,exact_symbolic,original,fredholm_dataset,{"x_values":[0.0,0.1],"u_values":[0.0,0.109]}
```

**Key Columns:**
- `id`: Unique identifier for this training sample
- `equation_id`: Original equation ID (may be augmented)
- `u`, `f`, `kernel`: Core equation fields
- `lambda_val`, `a`, `b`: Fredholm parameters
- `has_solution`, `solution_type`: Metadata
- `evaluation_points`: Optional JSON with x_values/u_values

### File Listing

```
data/processed/run_20260212_140310/
├── train.csv         # ~70% of samples
├── val.csv           # ~15% of samples
├── test.csv          # ~15% of samples
└── metadata.json     # Split statistics

# If format conversion is enabled
data/processed/run_20260212_140310/formatted/
├── train_infix.csv
├── train_latex.csv
├── train_rpn.csv
├── val_infix.csv
└── test_infix.csv
```

## Stage 3: Prompts

**File Format:** JSONL (line-delimited JSON)

**Location:** `data/prompts/<style>/`

### Basic Prompt Style

Each line is a prompt + ground truth pair:

```jsonl
{
  "id": "eq_0_basic",
  "equation_id": "eq_0",
  "prompt": "Solve the Fredholm integral equation:\nu(x) - 0.5*∫₀¹ e^(x*t) * u(t) dt = sin(x)\n\nProvide the solution as a LaTeX expression.",
  "format": "latex",
  "ground_truth": "\\sin(x) / (1 - 0.5*\\int_0^1 e^{xt} dt)",
  "domain": [0, 1]
}
```

### Chain-of-Thought Prompt Style

```jsonl
{
  "id": "eq_0_cot",
  "equation_id": "eq_0",
  "prompt": "Solve the Fredholm integral equation step by step:\n\nu(x) - 0.5*∫₀¹ e^(x*t) * u(t) dt = sin(x)\n\n1. Identify the kernel K(x,t) = e^(x*t)\n2. Identify the forcing function f(x) = sin(x)\n3. Look for patterns or special structure\n4. Apply appropriate solution method\n\nProvide the solution as a LaTeX expression.",
  "format": "latex",
  "reasoning_hints": [
    "Try assuming a separable form",
    "Consider the structure of the kernel"
  ],
  "ground_truth": "\\sin(x) / (1 - 0.5*\\int_0^1 e^{xt} dt)",
  "domain": [0, 1]
}
```

### File Listing

```
data/prompts/chain-of-thought/
├── train.jsonl       # Prompts for training
├── val.jsonl         # Prompts for validation
├── test.jsonl        # Prompts for testing
└── metadata.json     # Statistics
```

## Stage 4: LLM Predictions

**File Format:** JSONL (line-delimited JSON)

**Location:** `outputs/<run_name>/predictions_<timestamp>.jsonl`

### Minimal Prediction (Required Fields)

```jsonl
{
  "equation_id": "eq_0",
  "ground_truth": "\\sin(x) / (1 - 0.5*\\int_0^1 e^{xt} dt)",
  "solution_str": "\\sin(x) / (1 - 0.5*\\int_0^1 e^{xt} dt)"
}
```

### Standard Prediction (Recommended Fields)

```jsonl
{
  "equation_id": "eq_0",
  "prompt": "Solve the Fredholm integral equation:\nu(x) - 0.5*∫₀¹ e^(x*t) * u(t) dt = sin(x)",
  "ground_truth": "\\sin(x) / (1 - 0.5*\\int_0^1 e^{xt} dt)",
  "ground_truth_domain": [0, 1],
  "ground_truth_has_solution": true,
  "ground_truth_solution_type": "exact_symbolic",
  "solution_str": "\\sin(x) / (1 - 0.5*\\int_0^1 e^{xt} dt)",
  "solution_sympy": "sin(x) / (1 - 0.5*integrate(exp(x*t), (t, 0, 1)))",
  "has_solution": true,
  "solution_type": "exact_symbolic",
  "raw_response": "Looking at this Fredholm equation of the second kind...",
  "reasoning": "The kernel structure suggests...",
  "confidence": 0.92,
  "model": "gpt-4o",
  "timestamp": "2026-02-12T14:03:13Z"
}
```

### With Evaluation Points (For Numeric Evaluation)

```jsonl
{
  "equation_id": "eq_0",
  "ground_truth": "\\sin(x) / (1 - 0.5*\\int_0^1 e^{xt} dt)",
  "solution_str": "\\sin(x) / (1 - 0.5*\\int_0^1 e^{xt} dt)",
  "evaluation_points": {
    "x_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "u_values": [0.0, 0.1005, 0.2018, 0.3047, 0.4108, 0.5235, 0.6453, 0.7826, 0.9433, 1.1381, 1.4436],
    "n_points": 50
  },
  "has_solution": true,
  "solution_type": "exact_symbolic"
}
```

### File Listing

```
outputs/run_20260212_140313/
├── predictions_20260212_140313.jsonl
├── cost_details_20260212_140313.jsonl    # Token usage, costs
├── cost_summary_20260212_140313.json     # Aggregate costs
└── metrics_20260212_140314.json          # Evaluation results (see Stage 5)
```

**Important Note:** Prediction files do NOT contain evaluation metrics (RMSE, MAE, symbolic/numeric comparison results). Those are computed and saved separately in Stage 5 when evaluation runs.

## Stage 4b: Evaluated Predictions (After Evaluation)

**File Format:** JSONL (line-delimited JSON)

**Location:** `outputs/<run_name>/predictions_evaluated_<timestamp>.jsonl`

**When Created:** After running evaluation (Stage 5)

This file is created when evaluation runs. It takes the predictions from Stage 4 and adds symbolic/numeric evaluation results:

### Example Evaluated Prediction Entry

```jsonl
{
  "equation_id": "eq_0",
  "prompt": "Solve: u(x) - 0.5*∫₀¹ e^(x*t) * u(t) dt = sin(x)",
  "ground_truth": "\\sin(x) / (1 - 0.5*\\int_0^1 e^{xt} dt)",
  "solution_str": "\\sin(x) / (1 - 0.5*\\int_0^1 e^{xt} dt)",
  "solution_sympy": "sin(x) / (1 - 0.5*integrate(exp(x*t), (t, 0, 1)))",
  "has_solution": true,
  "solution_type": "exact_symbolic",
  "evaluation": {
    "symbolic": {
      "equivalent": true,
      "simplified_match": true
    },
    "numeric": {
      "match": true,
      "max_error": 1.23e-8,
      "mean_error": 4.56e-9,
      "mae": 4.56e-9,
      "rmse": 5.67e-9,
      "evaluation_points_used": 100,
      "points_source": "generated",
      "x_values": [0.0, 0.01, 0.02, ..., 1.0],
      "y_pred": [0.0, 0.00996, 0.01992, ..., 1.436],
      "y_true": [0.0, 0.00996, 0.01992, ..., 1.436]
    },
    "symbolic_match": true,
    "numeric_match": true,
    "correct": true,
    "solution_type": "exact_symbolic"
  }
}
```

**Numeric Evaluation Fields:**
- `max_error` - Maximum absolute error across all test points
- `mean_error` / `mae` - Mean Absolute Error: $\frac{1}{n}\sum|y_{pred} - y_{true}|$
- `rmse` - Root Mean Square Error: $\sqrt{\frac{1}{n}\sum(y_{pred} - y_{true})^2}$
- `x_values` - Test points where solution was evaluated
- `y_pred` - Predicted solution values at test points
- `y_true` - Ground truth solution values at test points
- `evaluation_points_used` - Number of test points
- `points_source` - Where points came from: "generated" or "evaluation_points" (pre-computed)

**Symbolic Evaluation Fields:**
- `equivalent` - Solutions are mathematically equivalent
- `simplified_match` - Simplification made them match

## Stage 5: Evaluation Metrics (Aggregate Summary)

**File Format:** JSON

**Location:** `outputs/<run_name>/metrics_<timestamp>.json`

**When Created:** After running evaluation (Stage 5)

**Quick Test:** To see example metrics, run:
```bash
python -m src.cli run --config configs/eval_only.yaml
```

This file contains aggregate statistics computed from all evaluated predictions:


```json
{
  "mode": "both",
  "total": 100,
  "correct": 75,
  "accuracy": 0.75,
  "symbolic_accuracy": 0.82,
  "numeric_accuracy": 0.68,
  "evaluated_count": 100,
  "total_predictions": 100,
  "parse_errors": 0,
  "api_errors": 0,
  
  "per_type": {
    "exact_symbolic": {
      "total": 40,
      "correct": 35,
      "accuracy": 0.875,
      "symbolic": 38,
      "numeric": 32
    },
    "series": {
      "total": 30,
      "correct": 15,
      "accuracy": 0.5,
      "symbolic": 20,
      "numeric": 10
    },
    "approx_coef": {
      "total": 10,
      "correct": 5,
      "accuracy": 0.5,
      "symbolic": 4,
      "numeric": 6
    }
  },
  
  "has_solution_accuracy": 0.95,
  "has_solution_total": 95,
  "solution_type_accuracy": 0.88,
  "solution_type_total": 98,
  
  "confusion_matrix": {
    "exact_symbolic_predicted_as_approx_coef": 1,
    "series_predicted_as_exact_symbolic": 5
  }
}
```

## Eval-Only Mode: Just Stages 4 & 5

When running in evaluation-only mode, you start with **Stage 4** predictions and directly compute **Stage 5** metrics:

```yaml
# eval_config.yaml
dataset:
  evaluation_only:
    predictions_path: outputs/existing_run/predictions.jsonl

evaluation:
  mode: both
  symbolic_tolerance: 1e-10
  numeric_tolerance: 1e-6
  num_test_points: 100
```

Running this is equivalent to re-evaluating predictions with potentially different tolerance settings.

## Converting Between Formats

### LaTeX to Infix

```
Input:  \sin(x) * e^{-x}
Output: sin(x) * e^(-x)
```

### LaTeX to RPN (Reverse Polish Notation)

```
Input:  \sin(x) + \cos(x)
Output: x sin cos +
```

### LaTeX to SymPy

```python
from src.llm.math_verify_adapter import parse_latex_to_sympy

expr = parse_latex_to_sympy("\\sin(x) * e^{-x}")
# expr = sin(x)*exp(-x)
```

## Data Size Examples

### Typical Dataset Sizes

| Dataset | Equations | After Augmentation | After Split | Total Train Rows |
|---------|-----------|-------------------|-------------|-----------------|
| Small   | 100       | ~115              | 70/15/15    | ~81             |
| Medium  | 1000      | ~1150             | 70/15/15    | ~805            |
| Large   | 10000     | ~11500            | 70/15/15    | ~8050           |

### File Size Estimates

| Stage | Format | 1000 eqs | 10K eqs |
|-------|--------|----------|---------|
| Raw   | JSON   | ~500 KB  | ~5 MB   |
| Prepared | CSV | ~2-5 MB | ~20-50 MB |
| Prompts | JSONL | ~5-10 MB | ~50-100 MB |
| Predictions | JSONL | ~50-100 MB | ~500-1000 MB |
| Evaluated Predictions | JSONL | ~100-200 MB | ~1-2 GB |
| Metrics | JSON | ~50 KB | ~100 KB |

## Summary Table

| # | Stage | Input | Output Files | Contents |
|---|-------|-------|--------------|----------|
| 1 | Raw Data | User data | `raw/` | Original equations |
| 2 | Prepare | Raw equations | train.csv, val.csv, test.csv | Split, formatted data |
| 3 | Prompts | Prepared data | `train.jsonl`, `val.jsonl`, `test.jsonl` | Prompts + ground truth |
| 4 | **LLM Inference** | Prompts | `predictions_*.jsonl`, `cost_summary_*.json`, `cost_details_*.jsonl` | **LLM outputs only (NO evaluation)** |
| 4b | **Evaluation** | Predictions | `predictions_evaluated_*.jsonl`, `metrics_*.json` | **Symbolic/Numeric metrics added** |

## Critical Clarification: When Are Evaluation Metrics Added?

**If you run the pipeline ONLY to the prediction stage (Stage 4):**

```bash
python -m src.cli run --config inference_only_config.yaml
```

❌ **NO evaluation happens**
❌ `predictions_*.jsonl` has **NO** RMSE, MAE, or symbolic/numeric comparison results
❌ Only these files are created:
  - `predictions_*.jsonl` - LLM outputs only
  - `cost_summary_*.json` - Token/cost summary
  - `cost_details_*.jsonl` - Per-call cost details

**To get RMSE, MAE, and symbolic/numeric evaluation**, you must run evaluation:

```bash
# Option 1: Include evaluation in config
python -m src.cli run --config full_pipeline_config.yaml  # Has 'model' + 'evaluation'

# Option 2: Run evaluation-only after inference
python -m src.cli evaluate outputs/run_xxx/predictions_*.jsonl \
  --output metrics.json
```

**After evaluation runs**, these files are created:
- `predictions_evaluated_*.jsonl` - Predictions **+ evaluation results**
- `metrics_*.json` - Summary statistics

## Data Flow with Evaluation

```
Predictions File (Stage 4)          Evaluation Process          Output Files (Stage 5)
┌─────────────────────────┐                                    ┌──────────────────┐
│ equation_id             │                                    │ equation_id      │
│ ground_truth            │    ┌─────────────────────────┐    │ ground_truth     │
│ solution_str            │───>│ Parse & Compare         │───>│ solution_str     │
│ solution_type           │    │ - Symbolic (SymPy/Math-Verify) │    │ evaluation:      │
│ confidence              │    │ - Numeric (100 points)  │    │   symbolic: {...}│
│ (NO metrics)            │    │ - Compute RMSE, MAE     │    │   numeric: {     │
│                         │    └─────────────────────────┘    │     rmse: 1e-6   │
│                         │                                    │     mae: 2e-6    │
└─────────────────────────┘                                    │   }              │
                                                               │ (metrics added)  │
                                                               └──────────────────┘
```

## Example: Prediction-Only Then Eval-Only

**Run 1: Inference only (no evaluation)**
```yaml
# inference_config.yaml
dataset:
  prompts:
    prompts_dir: data/prompts/chain-of-thought

model:
  provider: openai
  name: gpt-4o
  api_key: ${OPENAI_API_KEY}

evaluation:   # <-- SKIP THIS or set to null
  
output:
  dir: outputs/run_1
```

```bash
python -m src.cli run --config inference_config.yaml
# Creates: outputs/run_1/predictions_20260212_140313.jsonl (NO metrics)
```

**Run 2: Evaluate the predictions with custom tolerances**
```bash
python -m src.cli evaluate outputs/run_1/predictions_20260212_140313.jsonl \
  --numeric-tolerance 1e-8 \
  --output outputs/run_1/metrics_strict.json

# Creates: 
# - outputs/run_1/predictions_evaluated_20260212_150000.jsonl (with metrics)
# - outputs/run_1/metrics_strict.json (summary stats)
```

## Summary Table (Pipeline Stages)

| Stage | Input | Output | Contains Eval Metrics? | Contains RMSE/MAE? |
|-------|-------|--------|------------------------|--------------------|
| 1-3 | Raw data | Prompts | N/A | N/A |
| **4: LLM** | Prompts | `predictions_*.jsonl` | **NO** ❌ | **NO** ❌ |
| **4b: Evaluation** | Predictions | `predictions_evaluated_*.jsonl` | **YES** ✅ | **YES** ✅ |
| **5: Metrics** | Predictions | `metrics_*.json` | **Summary only** ✅ | **Per-type aggregates** ✅ |

