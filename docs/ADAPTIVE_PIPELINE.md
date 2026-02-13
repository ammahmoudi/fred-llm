# Fred-LLM Pipeline Configuration

The pipeline automatically determines what processing steps to run based on your configuration. It supports **smart defaults** with automatic path chaining between stages.

## Key Features

✅ **Smart Defaults** - Omit paths for auto-generated timestamped directories  
✅ **Path Chaining** - Output from one stage automatically becomes input to next  
✅ **Conflict Detection** - Validates configs before execution  
✅ **Flexible Workflows** - Run preparation only or full inference  
✅ **Proper Folder Structure** - Organized under `data/` and `outputs/`  

## Pipeline Workflows

### 1. Data Preparation 📊
**Use when:** Need to prepare, augment, and format data (no inference)

**What it does:**
```
Raw CSV → Augment → Validate → Split → Convert Formats → Generate Prompts → STOP
```

**Config:** [configs/prepare_data.yaml](../configs/prepare_data.yaml)

**Run:**
```bash
uv run python -m src.cli run --config configs/prepare_data.yaml
```

**Use config-based approach first**, then scripts for advanced customization:

```bash
# Advanced: Direct script usage
python scripts/prepare_dataset.py \
  --input data/raw/data.csv \
  --augment --validate --split --convert
```

---

### 2. Run Inference 🤖
**Use when:** Have pre-generated prompts ready for LLM

**What it does:**
```
Pre-generated Prompts → LLM Inference → Evaluate
```

**Config:** [configs/run_inference.yaml](../configs/run_inference.yaml)

**Run:**
```bash
export OPENAI_API_KEY=your_key_here
uv run python -m src.cli run --config configs/run_inference.yaml
```

---

### 3. Evaluation-Only ✅
**Use when:** You already have LLM predictions and want to score them

**What it does:**
```
Predictions JSONL/JSON -> Evaluate -> Metrics
```

**Config:** [configs/eval_only.yaml](../configs/eval_only.yaml)

**Run:**
```bash
uv run python -m src.cli run --config configs/eval_only.yaml
```

**Direct CLI alternative:**
```bash
python -m src.cli evaluate outputs/run_*/predictions_*.jsonl --output outputs/eval_results/metrics.json
```

---

## Smart Defaults & Path Chaining

### Default Output Paths

**Raw data preparation:**
- Omit `output_dir` → Auto-generates `data/processed/run_<timestamp>/`
- Specify `output_dir` → Uses your path

**Prompt generation:**
- Omit `output_dir` → Auto-generates `data/prompts/<style>/`
- Specify `output_dir` → Uses your path

**Example with defaults:**
```yaml
dataset:
  raw:
    path: data/raw/dataset.csv  # output_dir omitted
  prompting:
    style: few-shot  # output_dir omitted
```

**Result:**
```
data/processed/run_20260103_161033/  ← Auto-generated
data/prompts/few-shot/                ← Auto-generated
```

### Automatic Path Chaining

Outputs from one stage automatically become inputs to the next:

**Full automation:**
```yaml
dataset:
  raw:
    path: data/raw/dataset.csv
    # output_dir → data/processed/run_XXX/
  prompting:
    style: chain-of-thought
    # input_dir auto-chains from raw.output_dir
    # output_dir → data/prompts/chain-of-thought/
```

**Partial automation:**
```yaml
dataset:
  prepared:
    train_path: data/processed/exp1/train.csv
    # Parent dir = data/processed/exp1/
  prompting:
    style: few-shot
    # input_dir auto-chains from prepared location
```

### Manual Override

You can always specify explicit paths:

```yaml
dataset:
  raw:
    path: data/raw/dataset.csv
    output_dir: data/processed/my_experiment  # Explicit
  prompting:
    style: few-shot
    input_dir: data/processed/my_experiment   # Explicit
    output_dir: data/prompts/my_experiment    # Explicit
```

---

## Conflict Detection

The pipeline validates your config and prevents common mistakes:

### ❌ Cannot have both prompting and prompts
```yaml
dataset:
  prepared:
    train_path: data/train.csv
  prompting:
    style: few-shot  # Generate prompts
  prompts:
    prompts_dir: data/prompts/few-shot  # Use existing prompts
# ERROR: Choose one - generate OR use existing
```

### ❌ Cannot have same input and output
```yaml
dataset:
  prepared:
    train_path: data/train.csv
  prompting:
    input_dir: data/prompts
    output_dir: data/prompts  # Same as input!
# ERROR: Output would overwrite input
```

---

## Folder Structure

The pipeline uses organized folder structure:

```
project/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/
│   │   ├── run_20260103_161033/  # Auto-generated runs
│   │   └── my_experiment/        # Named experiments
│   └── prompts/
│       ├── basic/
│       ├── chain-of-thought/
│       ├── few-shot/
│       └── tool-assisted/
└── outputs/
    ├── experiment_1/
    └── experiment_2/
```

---

## Decision Tree

The pipeline automatically detects workflow based on config:

```
Does config have 'prompts.prompts_dir'?
├─ YES → Prompts → Inference
└─ NO
    │
    Does config have 'prepared.train_path'?
    ├─ YES → Prepared → Prompts → Inference
    └─ NO
        │
        Does config have 'raw.path'?
        ├─ YES → Raw → Full Pipeline
        └─ NO → ERROR: No dataset specified
```

## Key Features

### 🔍 Format Auto-Detection
Don't know what format your data is in? Set `format: null` and the pipeline will:
1. Check filename patterns (`train_infix.csv` → infix)
2. Analyze content if needed (regex patterns for infix/latex/rpn)
3. Validate consistency across train/val/test

**Detection rules:**
- **Infix**: `x**2 + sin(x)`, function calls like `exp(-x)`
- **LaTeX**: `\sin`, `\cos`, `^{...}`, `\frac{...}{...}`
- **RPN**: Space-separated postfix `x 2 ^ x sin +`

### 📝 Flexible Prompting
Two modes available:

**Pre-generated prompts** (faster, consistent):
```yaml
dataset:
  prompts:
    prompts_dir: data/prompts/chain-of-thought
```

**On-the-fly generation** (simpler, flexible):
```yaml
dataset:
  prompting:
    style: chain-of-thought
    num_examples: 3
    edge_case_mode: none
```

### ✅ Smart Validation
The pipeline validates:
- Configuration structure (Pydantic models)
- File existence
- Format consistency
- Split ratio validity

## Dry Run Mode

Preview execution plan before running:

```bash
uv run python -m src.cli run --config config.yaml --dry-run
```

Shows resolved paths and execution stages.

## Configuration Examples

See [configs/README.md](../configs/README.md) for complete examples.

**Minimal configs:**

```yaml
# Full automation
dataset:
  raw:
    path: data/raw/dataset.csv
  prompting:
    style: chain-of-thought

# Partial automation  
dataset:
  prepared:
    train_path: data/train.csv
    val_path: data/val.csv
    test_path: data/test.csv
  prompting:
    style: few-shot

# Manual control
dataset:
  prompts:
    prompts_dir: data/prompts/basic
    style: basic
```

## Testing

Run config tests:

```bash
uv run pytest tests/test_adaptive_config.py -v
```

**Note:** The pipeline uses `scripts/prepare_dataset.py` and `scripts/run_prompt_generation.py` internally. Always use the main CLI for running workflows.

Tests cover:
- Validation (requires at least one dataset source)
- Conflict detection (cannot have both prompting and prompts)
- Path resolution (defaults and chaining)
- Automation levels (full, partial, manual)

**Full pipeline (minimal):**
```yaml
dataset:
  raw:
    path: data/raw/dataset.csv
  prompting:
    style: basic
model:  # Optional - omit to skip inference
  provider: openai
  name: gpt-4o-mini
```

**Data preparation only:**
```yaml
dataset:
  raw:
    path: data/raw/dataset.csv
    augment: true
    convert_formats: [infix, latex, rpn]
# No model section = stops after data prep
```

**Prompt generation only:**
```yaml
dataset:
  prepared:
    train_path: data/train.csv
    val_path: data/val.csv
    test_path: data/test.csv
  prompting:
    style: few-shot
# No model section = stops after prompts
```

## Next Steps

- See [configs/README.md](../configs/README.md) for detailed examples
- See [QUICKSTART.md](QUICKSTART.md) for first-time setup
- See [FEATURES.md](FEATURES.md) for complete feature list
