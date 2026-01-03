# Example Configurations

This directory contains example configurations for different workflows.

## Available Configs

| Config | Purpose | Use Case |
|--------|---------|----------|
| [template.yaml](template.yaml) | Complete reference | Shows ALL available options |
| [prepare_data.yaml](prepare_data.yaml) | Data preparation | Prepare dataset without inference |
| [run_inference.yaml](run_inference.yaml) | LLM inference | Run inference on prepared prompts |

## Usage

### 1. Data Preparation

Prepare your dataset with augmentation, validation, and format conversion:

```bash
uv run python -m src.cli run --config configs/examples/prepare_data.yaml
```

**What it does:**
- Loads raw CSV
- Augments with edge cases (15% increase)
- Validates equations
- Splits into train/val/test (70/15/15)
- Converts to 5 formats (infix, latex, rpn, tokenized, python)
- Generates few-shot prompts
- Stops before inference

**Output:**
- Prepared data: `data/processed/run_<timestamp>/`
- Prompts: `data/prompts/few-shot/`

**Advanced: Using Scripts Directly**

For custom workflows, you can call the internal scripts:

```bash
# Data preparation
python scripts/prepare_dataset.py \
  --input data/raw/Fredholm_Dataset_Sample.csv \
  --output data/processed/my_data \
  --augment --validate --split --convert

# Prompt generation
python scripts/run_prompt_generation.py \
  --input data/processed/my_data \
  --output data/prompts/my_prompts \
  --styles few-shot
```

---

### 2. Run Inference

Run LLM inference on prepared prompts:

```bash
# Set your API key
export OPENAI_API_KEY=your_key_here

# Run inference
uv run python -m src.cli run --config configs/examples/run_inference.yaml
```

**What it does:**
- Loads pre-generated prompts
- Runs LLM inference (gpt-4o-mini)
- Evaluates results (symbolic + numeric)
- Saves predictions and metrics

**Output:**
- Predictions: `outputs/inference/predictions.json`
- Metrics: `outputs/inference/metrics.json`

---

## Configuration Tips

### Customize Data Preparation

Edit `prepare_data.yaml`:

```yaml
# Change augmentation
dataset:
  raw:
    augment_multiplier: 2.0  # Double dataset size
    augment_strategies: [approx_coef, discrete_points]  # Specific strategies

# Change formats
    convert_formats: [infix, latex]  # Only 2 formats

# Change prompting
  prompting:
    style: chain-of-thought  # Different prompt style
    num_examples: 5  # More examples
```

### Customize Inference

Edit `run_inference.yaml`:

```yaml
# Change model
model:
  provider: openrouter
  name: anthropic/claude-3.5-sonnet
  api_key_env: OPENROUTER_API_KEY

# Change evaluation
evaluation:
  mode: symbolic  # Only symbolic evaluation
  num_test_points: 50  # Fewer test points
```

---

## Complete Reference

See [template.yaml](template.yaml) for all available options with documentation.

Common workflows:

**Quick test (small sample):**
```yaml
dataset:
  raw:
    max_samples: 100  # Only 100 equations
    convert_formats: [infix]  # Single format
```

**Full production run:**
```yaml
dataset:
  raw:
    augment_multiplier: 1.5  # More augmentation
    convert_formats: [infix, latex, rpn, tokenized, python]  # All formats
  prompting:
    style: few-shot
    num_examples: 10  # More examples
```

**Dry run (preview):**
```bash
uv run python -m src.cli run --config config.yaml --dry-run
```
```
Prepared Data → Generate Prompts → Save → STOP
```

**Output:** JSONL prompt files organized by style

**Use when:**
- Already have prepared data
- Want to generate prompts for multiple styles
- Testing different prompt strategies

**Run:**
```bash
uv run python -m src.cli run --config configs/examples/prompts_only.yaml
```

**Configured features:**
- ✅ Few-shot style with 3 examples
- ✅ Guardrails mode for edge cases
- ✅ Ground truth included
- ✅ Format auto-detection
- ✅ Prompts saved to data/prompts_generated/

---

### 3. inference_only.yaml
**Purpose:** LLM inference with pre-generated prompts

**What it does:**
```
Pre-generated Prompts → LLM Inference → Evaluate → Save Results
```

**Output:** Predictions, metrics, detailed evaluation

**Use when:**
- Benchmarking models fairly (same prompts)
- Running multiple experiments
- Production inference

**Run:**
```bash
uv run python -m src.cli run --config configs/examples/inference_only.yaml
```

**Configured features:**
- ✅ Complete model parameters (temperature, top_p, etc.)
- ✅ Batch processing with rate limiting
- ✅ Comprehensive evaluation (symbolic + numeric + edge cases)
- ✅ Detailed metrics export
- ✅ Saves raw LLM responses
- ✅ Confusion matrix generation

---

### 4. full_automation.yaml
**Purpose:** Complete end-to-end pipeline

**What it does:**
```
Raw CSV → Augment → Split → Convert → Generate Prompts → Inference → Evaluate
```

**Output:** Everything (prepared data + prompts + predictions + metrics)

**Use when:**
- First-time setup
- Testing entire pipeline
- Reproducing experiments from scratch

**Run:**
```bash
uv run python -m src.cli run --config configs/examples/full_automation.yaml
```

**Configured features:**
- ✅ All data preparation steps
- ✅ On-the-fly prompt generation (saved for reuse)
- ✅ LLM inference
- ✅ Complete evaluation

---

### 5. partial_automation.yaml
**Purpose:** Development workflow (prompts + inference)

**What it does:**
```
Prepared Data → Generate Prompts → Inference → Evaluate
```

**Output:** Prompts + predictions + metrics

**Use when:**
- Fast iteration during development
- Already have prepared data
- Testing different prompt styles

**Run:**
```bash
uv run python -m src.cli run --config configs/examples/partial_automation.yaml
```

**Configured features:**
- ✅ Few-shot prompts with guardrails
- ✅ Prompts saved for inspection
- ✅ Format auto-detection
- ✅ Quick evaluation

---

### 6. manual_control.yaml
**Purpose:** Reproducible benchmarking

**What it does:**
```
Pre-generated Prompts → Inference → Evaluate
```

**Output:** Predictions + metrics

**Use when:**
- Comparing models fairly
- Production evaluation
- Deterministic results needed (temp=0.0)

**Run:**
```bash
uv run python -m src.cli run --config configs/examples/manual_control.yaml
```

**Configured features:**
- ✅ GPT-4 for high quality
- ✅ Temperature 0.0 (deterministic)
- ✅ High-precision evaluation (1e-12 tolerance)
- ✅ 200 test points for numeric evaluation

---

## Configuration Options Reference

### Dataset Section

**Raw data (preparation):**
```yaml
dataset:
  raw:
    path: data/raw/dataset.csv
    output_dir: data/processed/prepared
    max_samples: null  # Process all
    augment: true
    augment_multiplier: 1.15
    augment_strategies: [approx_coef, discrete_points, ...]
    include_edge_metadata: false
    validate_data: true
    split: true
    split_ratios: [0.7, 0.15, 0.15]
    seed: 42
    convert_formats: [infix, latex, rpn, tokenized, python]
    convert_limit: null  # Convert all
```

**Prepared data:**
```yaml
dataset:
  prepared:
    train_path: data/train.csv
    val_path: data/val.csv
    test_path: data/test.csv
    format: null  # auto-detect or specify: infix/latex/rpn
    max_samples: null
```

**Prompt generation:**
```yaml
dataset:
  prompting:
    output_dir: data/prompts
    style: chain-of-thought  # basic, chain-of-thought, few-shot, tool-assisted
    edge_case_mode: none  # none, guardrails, hints
    num_examples: 2
    include_ground_truth: true
    include_examples: true
    format: null  # auto-detect
```

**Pre-generated prompts:**
```yaml
dataset:
  prompts:
    prompts_dir: data/prompts/style_name
    style: chain-of-thought  # Must match directory
```

### Model Section

```yaml
model:
  provider: openai  # openai, openrouter, local
  name: gpt-4o-mini
  
  # Generation parameters
  temperature: 0.1
  max_tokens: 2000
  top_p: 1.0
  frequency_penalty: 0.0
  presence_penalty: 0.0
  
  # API settings
  api_key_env: OPENAI_API_KEY
  timeout: 60
  max_retries: 3
  
  # Batch settings
  batch_size: 10
  rate_limit_delay: 0.1
```

### Output Section

```yaml
output:
  dir: outputs/experiment_name
  format: jsonl  # jsonl or json
  save_prompts: true
  save_predictions: true
  save_raw_responses: true
  timestamp: true
  experiment_name: null  # Optional identifier
```

### Evaluation Section

```yaml
evaluation:
  mode: both  # symbolic, numeric, or both
  
  # Symbolic evaluation
  symbolic:
    method: sympy
    simplify: true
    timeout: 5.0
  
  # Numeric evaluation
  numeric:
    tolerance: 1e-6
    num_test_points: 100
    test_domain: [0, 1]
    method: trapezoidal
  
  # Edge case evaluation
  edge_cases:
    check_recognition: true
    check_handling: true
  
  # Metrics
  metrics:
    - accuracy
    - symbolic_match
    - numeric_error
    - edge_case_precision
    - edge_case_recall
    - solution_type_accuracy
  
  # Export
  export:
    save_detailed: true
    save_summary: true
    save_failures: true
    save_confusion_matrix: true
```

## Dry Run Mode

Preview what any config will do:

```bash
uv run python -m src.cli run --config config.yaml --dry-run
```

**Example output:**
```
🤖 Adaptive Pipeline
Automation Level: full

📋 Execution Plan (Dry Run)

  1. Load raw dataset: data/raw/dataset.csv
     • Output: data/processed/prepared
     • Augment: Yes
     • Strategies: [6 strategies]
     • Split: (0.7, 0.15, 0.15)
     • Convert to: [infix, latex, rpn, tokenized, python]
  2. Generate prompts on-the-fly
     • Output: data/prompts/chain-of-thought
     • Style: chain-of-thought
  3. Run LLM inference
     • Provider: openai
     • Model: gpt-4o-mini
  4. Evaluate results
     • Output: outputs/experiment

Run without --dry-run to execute
```

## Combining Configs

You can mix and match stages by creating custom configs:

**Example: Prepare + Generate Prompts (no inference)**
```yaml
dataset:
  raw:
    path: data/raw/dataset.csv
    # ... preparation settings ...
  prompting:
    # ... prompt generation settings ...
  # No prompts section = won't run inference
```

**Example: Use Prepared Data + Pre-generated Prompts**
```yaml
dataset:
  prompts:
    prompts_dir: data/my_prompts/
    style: chain-of-thought
# Prepared data not needed when using pre-generated prompts
```

## Next Steps

- See [ADAPTIVE_PIPELINE.md](../../docs/ADAPTIVE_PIPELINE.md) for complete documentation
- See [configs/README.md](../README.md) for config file format
- See [QUICKSTART.md](../../docs/QUICKSTART.md) for first-time setup
