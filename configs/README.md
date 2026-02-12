# Configuration Files

This directory contains YAML configuration files for the Fred-LLM pipeline.

## Quick Start Configs

| Config | Purpose | Use Case |
|--------|---------|----------|
| [template.yaml](template.yaml) | Complete reference | View all available options and defaults |
| [prepare_data.yaml](prepare_data.yaml) | Data preparation | Prepare dataset without running inference |
| [run_inference.yaml](run_inference.yaml) | LLM inference | Run inference on prepared prompts |
| [eval_only.yaml](eval_only.yaml) | Evaluation only | Evaluate existing LLM predictions |
| [stratified_sample.yaml](stratified_sample.yaml) | Stratified sampling | Create balanced test sets with N samples per solution type |

## Usage

### 1. Evaluate Existing Predictions (Eval-Only Mode)

```bash
python -m src.cli run --config configs/eval_only.yaml
```

**Use case:** Analyze predictions from external sources or re-evaluate with different tolerances.

**Output:**
- Evaluated predictions: `outputs/eval_only/evaluated_predictions_<timestamp>.jsonl`
- Metrics: `outputs/eval_only/metrics_<timestamp>.json`

### 2. Prepare Dataset

```bash
uv run python -m src.cli run --config configs/prepare_data.yaml
```

**Output:**
- Prepared data in `data/processed/run_<timestamp>/`
- Prompts in `data/prompts/few-shot/`

### 3. Create Stratified Sample (Balanced Test Sets)

```bash
# Create diverse test set: 1 sample per solution type (includes edge cases!)
uv run python -m src.cli run --config configs/stratified_sample.yaml

# Or with augmentation for maximum diversity
uv run python -m src.cli run --config configs/stratified_sample.yaml \
  --set dataset.raw.augment=true

# Or balanced train+test: 5 samples per type
uv run python -m src.cli run --config configs/stratified_sample.yaml \
  --set dataset.raw.samples_per_type=5 \
  --set dataset.raw.split=true
```

**Use case:** Testing LLM performance across all solution types.

✅ **Pipeline order:** Load → Augment → Sample → Split
- Augmentation (optional) adds edge cases to original data
- Stratified sampling selects N equations per type **from augmented data**
- Result: Diverse samples that include both original and edge case equations!
- Perfect for testing LLM robustness across all solution categories

**Output:**
- Balanced dataset: `data/processed/stratified_sample/`
- Predictions: `outputs/stratified_sample/predictions.json`
- Metrics by solution type

### 4. Run Inference

```bash
export OPENAI_API_KEY=your_key_here
uv run python -m src.cli run --config configs/run_inference.yaml
```

**Output:**
- Predictions: `outputs/inference/predictions.json`
- Metrics: `outputs/inference/metrics.json`

---

## Model-Specific Configs

| Config | Description | Model | Use Case |
|--------|-------------|-------|----------|
| `default.yaml` | Balanced settings | gpt-4o-mini | General usage |
| `development.yaml` | Fast iteration | gpt-4o-mini | Quick testing (10 samples) |
| `production.yaml` | Full evaluation | gpt-4o | Production runs |
| `local.yaml` | Self-hosted | Phi-3-mini | Local inference |
| `openrouter.yaml` | Third-party API | claude-3.5-sonnet | Claude, Llama via OpenRouter |
| `fine_tuning.yaml` | Training | flan-t5-base | Fine-tuning experiments |

**Note:** These assume you've already prepared data and prompts using `prepare_data.yaml`.

---

## Configuration Structure

```yaml
# Data paths
dataset:
  train_path: data/processed/training_data/train_infix.csv
  format: null  # Auto-detected from filename

# Prompting (skip if using pre-generated prompts)
prompting:
  style: few-shot
  num_examples: 3
  edge_case_mode: none

# Model (omit for data-only workflows)
model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.1

# Evaluation (omit for data-only workflows)
evaluation:
  symbolic: true
  numeric: true
  robustness: false
```

See [template.yaml](template.yaml) for complete documentation of all options.

---

## Advanced: Using Scripts Directly

For custom workflows beyond the config-based pipeline:

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
  --styles few-shot chain-of-thought
```
