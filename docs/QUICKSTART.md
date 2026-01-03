# Quick Start Guide

Get up and running with Fred-LLM in 5 minutes.

## 1. Setup Environment

```bash
# Clone and enter the repository
git clone https://github.com/ammahmoudi/fred-llm.git
cd fred-llm

# Create virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"

# Activate environment
# Windows PowerShell:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

## 2. Download Dataset

```bash
# Download sample dataset (5,000 equations) from Zenodo
python -m src.cli dataset download --variant sample

# Or download full dataset (500,000 equations)
python -m src.cli dataset download --variant full
```

## 3. Prepare Training Data with Edge Cases

```bash
# Recommended: Use edge case folders for realistic distribution
# This generates edge cases organized by solution type

# Windows PowerShell:
python scripts/prepare_dataset.py `
  --input data/raw/Fredholm_Dataset_Sample.csv `
  --output data/processed/training_data `
  --max-samples 5000 `
  --augment `
  --augment-multiplier 1.15 `
  --augment-strategies approx_coef discrete_points series family regularized none_solution `
  --validate `
  --split `
  --convert `
  --convert-formats infix latex rpn

# Linux/macOS:
python scripts/prepare_dataset.py \
  --input data/raw/Fredholm_Dataset_Sample.csv \
  --output data/processed/training_data \
  --max-samples 5000 \
  --augment \
  --augment-multiplier 1.15 \
  --augment-strategies approx_coef discrete_points series family regularized none_solution \
  --validate \
  --split \
  --convert \
  --convert-formats infix latex rpn
```

**What this does:**
- Loads 5,000 equations from the sample dataset
- Applies **6 edge case folders** containing 14 strategies total (42 variants)
  - `approx_coef`: Boundary layers, oscillations, weak singularities (15 variants)
  - `discrete_points`: Complex kernels, near-resonance (6 variants)
  - `series`: Neumann series expansions (3 variants)
  - `family`: Non-unique solutions, exact resonance (3 variants)
  - `regularized`: Fredholm 1st kind, ill-posed problems (3 variants)
  - `none_solution`: Eigenvalue issues, no solution cases (12 variants)
- Uses multiplier 1.15 → ~5,750 total equations (87% exact, 13% edge cases)
- **Note**: Without `--augment-strategies`, only 3 basic transformations are applied (substitute, scale, shift)
- Validates data quality (checks 100 random samples)
- Splits into train (80%) and test (20%) sets with stratification
- Converts to 3 formats (infix, LaTeX, RPN) for LLM training
- Output: `data/processed/training_data/` (with train/test splits)

**Note**: By default, detailed edge case metadata (~60 fields like `singularity_type`, `layer_location`, etc.) is excluded for cleaner output. Add `--include-edge-metadata` to include all technical details.

### Understanding Augmentation Strategies

**Default behavior (no `--augment-strategies`):**
```bash
python scripts/prepare_dataset.py --augment
# Applies ONLY 3 basic transformations: substitute, scale, shift
```

**For edge cases, specify folder names explicitly:**
```bash
# All 6 edge case folders (42 variants) - RECOMMENDED
--augment-strategies approx_coef discrete_points series family regularized none_solution

# Specific use cases:
--augment-strategies approx_coef none_solution           # Numerical-only + no-solution cases
--augment-strategies discrete_points regularized         # Complex kernels + ill-posed problems
--augment-strategies series family                       # Series expansions + non-unique solutions

# Include basic transformations (exact_symbolic) + edge cases
--augment-strategies substitute scale shift compose approx_coef discrete_points series family regularized none_solution
```

**Folder-based strategies** (each runs multiple strategies):
- `approx_coef` → 5 strategies × 3 variants = 15 edge cases
- `discrete_points` → 2 strategies × 3 variants = 6 edge cases  
- `series` → 1 strategy × 3 variants = 3 edge cases
- `family` → 1 strategy × 3 variants = 3 edge cases
- `regularized` → 1 strategy × 3 variants = 3 edge cases
- `none_solution` → 4 strategies × 3 variants = 12 edge cases

## 4. Generate Prompts for LLM Training

```bash
# Generate prompts with all styles for the formatted datasets
# Windows PowerShell:
python scripts/run_prompt_generation.py `
  --input data/processed/training_data/formatted/ `
  --output data/prompts `
  --styles all `
  --pattern "*_infix.csv"

# Linux/macOS:
python scripts/run_prompt_generation.py \
  --input data/processed/training_data/formatted/ \
  --output data/prompts \
  --styles all \
  --pattern "*_infix.csv"

# Or generate for specific format (LaTeX, RPN, etc.):
python scripts/run_prompt_generation.py \
  --input data/processed/training_data/formatted/ \
  --pattern "*_latex.csv"  # LLM will output in LaTeX format

python scripts/run_prompt_generation.py \
  --input data/processed/training_data/formatted/ \
  --pattern "*_rpn.csv"  # LLM will output in RPN format

# Or generate specific styles only:
python scripts/run_prompt_generation.py \
  --input data/processed/training_data/formatted/ \
  --styles basic,chain-of-thought,few-shot \
  --output data/prompts
```

**What this does:**
- Processes all `*_infix.csv` files in formatted directory
- Generates prompts in 4 styles: basic, chain-of-thought, few-shot, tool-assisted
- **Format-specific prompts**: System detects format from filename and generates targeted instructions
  - Processing: `*_infix.csv` → Prompt: "Express solution in infix notation (x**2 + sin(x))"
  - Processing: `*_latex.csv` → Prompt: "Express solution in LaTeX notation (x^2 + \sin(x))"
  - Processing: `*_rpn.csv` → Prompt: "Express solution in RPN notation (x 2 ^ x sin +)"
- Includes ground truth solutions for training
- Outputs JSONL files to `data/prompts/{style}/` organized by style
- Each prompt includes metadata (kernel, lambda, domain, edge case info)
- **All prompts specify structured output format for evaluation**: `SOLUTION:`, `HAS_SOLUTION:`, `SOLUTION_TYPE:`

**Edge case handling modes:**
- `--edge-case-mode none` - Pure inference (default for test sets)
- `--edge-case-mode guardrails` - Include edge case instructions
- `--edge-case-mode hints` - Include has_solution and solution_type in prompts (for training)

**Structured output format:**
All generated prompts instruct LLMs to respond with:
```
SOLUTION: u(x) = [your solution in the same format as input]
HAS_SOLUTION: yes/no
SOLUTION_TYPE: exact_symbolic/exact_coef/approx_coef/discrete_points/series/family/regularized/none
```

**Format examples:**
- **Infix input**: `u(x) - 2*sin(x*t)*u(t) = x**2` → **Infix output**: `u(x) = x**2 + 0.5*cos(x)`
- **LaTeX input**: `u(x) - 2\sin(xt)u(t) = x^2` → **LaTeX output**: `u(x) = x^2 + 0.5\cos(x)`
- **RPN input**: `u x - 2 x t * sin u t * * = x 2 ^` → **RPN output**: `u x = x 2 ^ 0.5 x cos * +`

This enables automated evaluation of solution correctness and edge case recognition accuracy.

## Next Steps

- **Configure your LLM**: Edit `config.yaml` to set your API keys and model preferences
- **Train/Fine-tune**: Use generated prompts for fine-tuning or in-context learning
- **Evaluate**: Run evaluation metrics on model outputs
- **Explore**: Check out the notebooks in `notebooks/` for data exploration

## Common Issues

**Problem**: Dataset download fails
- **Solution**: Check internet connection, Zenodo may be slow. Try again or download manually from [DOI: 10.5281/zenodo.16784707](https://doi.org/10.5281/zenodo.16784707)

**Problem**: Out of memory during augmentation
- **Solution**: Reduce `--max-samples` or `--augment-multiplier`

**Problem**: Validation fails
- **Solution**: Check input CSV format matches expected schema (u, f, kernel, lambda, a, b columns)

## See Also

- [Edge Case Strategies Guide](EDGE_CASES.md) - Deep dive into augmentation strategies
- [Pipeline Architecture](pipeline-diagram.md) - System design overview
- [Features Tracking](FEATURES.md) - Implementation status
- [Configuration Guide](../configs/README.md) - Detailed configuration options
