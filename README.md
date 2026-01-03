# Fred-LLM

**Solving Fredholm Integral Equations using Large Language Models**

This project provides tools for solving and approximating Fredholm integral equations of the second kind using LLMs. It supports both symbolic and approximate (series-based) solutions, with flexible prompting strategies and evaluation metrics.

## Equation Form

The general Fredholm integral equation of the second kind:

$$u(x) - \lambda \int_a^b K(x, t) u(t) \, dt = f(x)$$

where:
- $u(x)$ is the unknown function to solve for
- $K(x, t)$ is the kernel function
- $f(x)$ is the known right-hand side
- $\lambda$ is a parameter
- $[a, b]$ is the integration domain

## Quick Start

Get up and running in 5 minutes:

### 1. Setup Environment

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

### 2. Download Dataset

```bash
# Download sample dataset (5,000 equations) from Zenodo
python -m src.cli dataset download --variant sample

# Or download full dataset (500,000 equations)
python -m src.cli dataset download --variant full
```

### 3. Prepare Training Data with Edge Cases

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
- Applies all 14 edge case strategies (42 variants total)
- Uses multiplier 1.15 → ~5,750 total equations (87% exact, 13% edge cases)
- Validates data quality (checks 100 random samples)
- Splits into train (80%) and test (20%) sets with stratification
- Converts to 3 formats (infix, LaTeX, RPN) for LLM training
- Output: `data/processed/training_data/` (with train/test splits)

### 4. Generate Prompts for LLM Training

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

# Or generate specific styles only:
python scripts/run_prompt_generation.py \
  --input data/processed/training_data/formatted/ \
  --styles basic,chain-of-thought,few-shot \
  --output data/prompts
```

**What this does:**
- Processes all `*_infix.csv` files in formatted directory
- Generates prompts in 4 styles: basic, chain-of-thought, few-shot, tool-assisted
- Includes ground truth solutions for training
- Outputs JSONL files to `data/prompts/{style}/` organized by style
- Each prompt includes metadata (kernel, lambda, domain, edge case info)

**Edge case handling modes:**
- `--edge-case-mode none` - Pure inference (default for test sets)
- `--edge-case-mode guardrails` - Include edge case instructions
- `--edge-case-mode hints` - Include has_solution and solution_type in prompts (for training)

### Folder-Based Strategy System

Strategies are organized by **solution type**. Specifying a folder name runs ALL strategies in that folder:

| Folder | Strategies | Variants | What it teaches |
|--------|------------|----------|-----------------|
| `approx_coef` | 5 | 15 | Functional forms with numerical params (boundary layers, oscillations, weak singularities) |
| `discrete_points` | 2 | 6 | Pure point samples, no formula (complex kernels, near-resonance) |
| `series` | 1 | 3 | Truncated series expansions (Neumann series, N=4 terms) |
| `family` | 1 | 3 | Non-unique solution families (exact resonance) |
| `regularized` | 1 | 3 | Ill-posed problems requiring regularization (Fredholm 1st kind) |
| `none_solution` | 4 | 12 | No solution exists (eigenvalue issues, range violations, divergent kernels) |
| **Total** | **14** | **42** | Comprehensive edge case recognition |

### Alternative: Specific Edge Case Training

```bash
# Example 1: Train on approximate coefficient cases (boundary layers, oscillations)
python scripts/prepare_dataset.py \
  --augment --augment-multiplier 1.2 \
  --augment-strategies approx_coef \
  --convert --convert-formats infix latex

# Example 2: Focus on discrete points and series expansions
python scripts/prepare_dataset.py \
  --augment --augment-multiplier 1.25 \
  --augment-strategies discrete_points series

# Example 3: No-solution and regularized cases only
python scripts/prepare_dataset.py \
  --augment --augment-multiplier 1.3 \
  --augment-strategies none_solution regularized
```

---

## Features

- 🤖 **Multi-provider LLM support**: OpenAI API, local models (HuggingFace, vLLM)
- 📝 **Multiple prompting styles**: Basic, Chain-of-Thought, Few-Shot, Tool-Assisted
- 🔢 **Symbolic & numeric evaluation**: SymPy-based parsing and verification
- 📊 **8 specialized formatters**: LaTeX, RPN, Infix, Python, Tokenized, Fredholm equations, Series expansions
- 🎯 **LLM-optimized formatting**: Special tokens for improved model understanding
- 🔄 **18 augmentation strategies**: 4 basic transformations (substitute, scale, shift, compose) + 14 edge case strategies organized by solution type (approx_coef, discrete_points, series, family, regularized, none) - total 42 variants
- 📈 **Comprehensive metrics**: Symbolic equivalence, numeric accuracy, solution verification
- ⚠️ **Realistic edge cases**: Teach LLMs to recognize singular problems, numerical-only solutions, and ill-posed equations
- 🎲 **Stratified dataset splitting**: Maintains dataset balance using scikit-learn, preserves original/augmented ratios and solution types

## Pipeline Architecture

The project follows a modular 4-stage pipeline:

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'fontSize': '14px'}}}%%
flowchart LR
 subgraph Dataset_Preparation["Module 1: Dataset Preparation"]
    direction TB
    DP1["Data Augmentation"]
    DP1a(["Add no-solution"])
    DP1b(["Add special function"])
    DP1c(["Generate numeric ground truth"])
    DP2["Format Conversion"]
    DP2a(["To LaTeX"])
    DP2b(["To RPN"])
    DP2c(["Tokenize for LLM"])
    DP1 --> DP1a & DP1b & DP1c
    DP1 --> DP2
    DP2 --> DP2a & DP2b & DP2c
  end

 subgraph Prompt_Engineering["Module 2: Prompt Engineering"]
    direction TB
    PR1["Prompt Design"]
    PR1a(["Direct prompts"])
    PR1b(["Chain of thought"])
    PR1c(["Approximation prompts"])
    PR2["Output Format"]
    PR2a(["Symbolic"])
    PR2b(["Series"])
    PR2c(["Code format"])
    PR1 --> PR1a & PR1b & PR1c
    PR1 --> PR2
    PR2 --> PR2a & PR2b & PR2c
  end

 subgraph LLM_Methods["Module 3: LLM Methods"]
    direction TB
    LLM1["Fine Tuning"]
    LLM1a(["Supervised pairs"])
    LLM1b(["Use Phi or T5"])
    LLM2["In Context Learning"]
    LLM2a(["Few-shot examples"])
    LLM2b(["Chain of thought prompts"])
    LLM3["Tool Use"]
    LLM3a(["Generate Python"])
    LLM3b(["Use symbolic tools"])
    LLM1 --> LLM1a & LLM1b
    LLM1 --> LLM2
    LLM2 --> LLM2a & LLM2b
    LLM2 --> LLM3
    LLM3 --> LLM3a & LLM3b
  end

 subgraph Evaluation["Module 4: Evaluation"]
    direction TB
    EV1["Symbolic Eval"]
    EV1a(["Exact match"])
    EV1b(["BLEU / TeX BLEU"])
    EV2["Numeric Eval"]
    EV2a(["MAE / MSE"])
    EV2b(["Test points"])
    EV3["Robustness"]
    EV3a(["Prompt variation"])
    EV3b(["Unseen function types"])
    EV1 --> EV1a & EV1b
    EV1 --> EV2
    EV2 --> EV2a & EV2b
    EV2 --> EV3
    EV3 --> EV3a & EV3b
  end

    Dataset_Preparation --> Prompt_Engineering
    Prompt_Engineering --> LLM_Methods
    LLM_Methods --> Evaluation

     DP1:::submod
     DP1a:::step
     DP1b:::step
     DP1c:::step
     DP2:::submod
     DP2a:::step
     DP2b:::step
     DP2c:::step
     PR1:::submod
     PR1a:::step
     PR1b:::step
     PR1c:::step
     PR2:::submod
     PR2a:::step
     PR2b:::step
     PR2c:::step
     LLM1:::submod
     LLM1a:::step
     LLM1b:::step
     LLM2:::submod
     LLM2a:::step
     LLM2b:::step
     LLM3:::submod
     LLM3a:::step
     LLM3b:::step
     EV1:::submod
     EV1a:::step
     EV1b:::step
     EV2:::submod
     EV2a:::step
     EV2b:::step
     EV3:::submod
     EV3a:::step
     EV3b:::step
    classDef submod fill:#e3f2fd,stroke:#1976d2,stroke-width:1px,color:#0d47a1
    classDef step fill:#ffffff,stroke:#42a5f5,stroke-width:1px,color:#0d47a1
```

### Module Overview

| Module | Purpose | Key Components |
|--------|---------|----------------|
| **Dataset Preparation** | Prepare and augment training data | 18 augmentation strategies (4 basic + 14 edge cases in 6 solution-type folders), 8 formatters (LaTeX/RPN/Python/etc.), CSV/JSON export, **8 solution types** |
| **Prompt Engineering** | Design effective prompts for LLMs | Direct, CoT, approximation prompts; symbolic/series/code output |
| **LLM Methods** | Model training and inference | Fine-tuning (Phi/T5), few-shot learning, tool use |
| **Evaluation** | Assess solution quality | Symbolic matching, BLEU, MAE/MSE, robustness testing |

### Data Augmentation Strategies

The project includes **4 basic + 14 edge case = 18 total augmentation strategies** organized by solution type to expand and diversify training data:

**Basic Transformations:** (solution_type: `exact_symbolic`)
- **Substitute**: Variable transformations (x → x², 2x, x+1)
- **Scale**: Lambda coefficient scaling (×0.5, ×2.0, ×0.1, ×10.0)
- **Shift**: Integration domain shifting ([a,b] → [a±1, b±1])
- **Compose**: Kernel composition (K → K+x, K+t, K×x)

**Edge Cases - 14 Comprehensive Strategies:**

Organized in 6 solution-type folders matching the solution taxonomy:

**exact_symbolic/** - 4 strategies (basic transformations above)
- Produce exact closed-form analytical solutions
- Strategies: substitute, scale, shift, compose

**approx_coef/** (solution_type: `approx_coef`) - 5 strategies × 3 variants = 15 edge cases
- Functional forms with numerical parameters (e.g., exp(-x/0.01), sin(100πx))
- **boundary_layer**: Sharp gradients near boundaries (ε=0.01, ε=0.001, ε=0.0001)
- **oscillatory_solution**: Rapid oscillations requiring Nyquist sampling (ω=50, ω=100, ω=200)
- **weakly_singular**: Integrable singularities [log|x-t|, |x-t|^(-0.5), |x-t|^(-0.3)]
- **mixed_type**: Volterra + Fredholm hybrid (α=0.3, α=0.5, α=0.7)
- **compact_support**: Sparse kernel structure - band-limited, localized Gaussian, piecewise constant

**discrete_points/** (solution_type: `discrete_points`) - 2 strategies × 3 variants = 6 edge cases
- Pure point samples, no functional form (empty u field, uses sample_points arrays)
- **complex_kernels**: No symbolic solution (Gaussian, exponential decay, sinc kernels)
- **near_resonance**: Ill-conditioned near eigenvalue (δ=0.01, δ=0.001, δ=0.0001)

**series/** (solution_type: `series`) - 1 strategy × 3 variants = 3 edge cases
- Truncated Neumann series expansions (N=4 terms)
- **neumann_series**: f + λ∫Kf + λ²∫K²f + λ³∫K³f (λ=0.5, λ=0.3, λ=0.1)

**family/** (solution_type: `family`) - 1 strategy × 3 variants = 3 edge cases
- Non-unique solution families
- **resonance**: Exact eigenvalue resonance → u = C*φ + u_p (constant, linear, quadratic kernels)

**regularized/** (solution_type: `regularized`) - 1 strategy × 3 variants = 3 edge cases
- Ill-posed Fredholm 1st kind equations
- **ill_posed**: Requires Tikhonov/TSVD regularization (α=0.01, α=0.001, α=0.0001)

**none_solution/** (solution_type: `none`) - 4 strategies × 3 variants = 12 edge cases
- No solution exists
- **eigenvalue_cases**: λ is eigenvalue violating Fredholm Alternative (constant, linear, exponential kernels)
- **range_violation**: f(x) not in operator range (orthogonal to kernel range)
- **divergent_kernel**: Non-integrable singularities [|x-t|^(-1), |x-t|^(-1.5), log(|x-t|)^(-1)]
- **disconnected_support**: Rank-deficient - kernels with disconnected support regions

**Total: 18 strategies × various variants = 42 augmented types**

#### Solution Type Taxonomy (8 types)

| Solution Type | Description | u Field | Example Strategies |
|---------------|-------------|---------|-------------------|
| `exact_symbolic` | Closed-form expressions | Formula | substitute, scale, shift, compose |
| `exact_coef` | Finite basis, exact weights | Coefficients (future) | TBD |
| `approx_coef` | Functional form, numerical params | Formula | boundary_layer, oscillatory, weakly_singular |
| `discrete_points` | Pure point samples | Empty `""` | complex_kernels, near_resonance |
| `series` | Neumann/Taylor expansion | Series formula | neumann_series |
| `family` | Non-unique solutions | General form | resonance |
| `regularized` | Ill-posed, needs regularization | Empty `""` | ill_posed |
| `none` | No solution exists | Empty `""` | eigenvalue_cases, range_violation |

**Usage**: Specify folder names to run all contained strategies:
```bash
--augment-strategies approx_coef discrete_points  # Runs 7 strategies (21 variants)
--augment-strategies none_solution regularized     # Runs 5 strategies (15 variants)
--augment-strategies approx_coef                   # Runs 5 strategies (15 variants)
```

Total: **42 edge case variants** (14 strategies × 3 variants each) teach LLMs to recognize when standard symbolic methods fail and special treatment is needed. The 8 solution types provide clear pedagogical signals: from exact symbolic solutions to discrete point samples, series expansions, and cases with no solution. 

**Folder structure matches solution taxonomy** for clarity and consistency. See [Edge Cases Documentation](docs/EDGE_CASES.md) and [Augmentation Strategies README](src/data/augmentations/README.md) for complete details.

#### Unified Output Schema

**All dataset entries (original and augmented) output identical 18 fields** for consistent ML training:

| Field | Type | Description |
|-------|------|-------------|
| **Core Equation** (6 fields) |
| `u`, `f`, `kernel` | str | Equation components |
| `lambda_val` | str | Lambda parameter (numeric string) |
| `a`, `b` | str | Integration bounds |
| **Expression Types** (3 fields) |
| `u_type`, `f_type`, `kernel_type` | ExpressionType | Expression type metadata from CSV |
| **Augmentation Tracking** (3 fields) |
| `augmented` | bool | `False` for original, `True` for augmented |
| `augmentation_type` | str | Strategy name (e.g., `"scale"`, `"no_solution"`) |
| `augmentation_variant` | str | Specific variant (e.g., `"scale_2.0x"`) |
| **Solution Metadata** (6 fields) |
| `has_solution` | bool | Solution exists? |
| `solution_type` | str | `"exact"` \| `"none"` \| `"numerical"` \| `"regularized"` \| `"family"` |
| `edge_case` | str \| None | Edge case type or `None` for basic |
| `reason` | str | Explanation of augmentation/case |
| `recommended_methods` | list[str] | Suggested solution methods |
| `numerical_challenge` | str \| None | Computational challenge description |

See [Augmentation README](src/data/augmentations/README.md) for complete schema documentation with examples.

### Dataset Balance Recommendations

For optimal LLM training, use appropriate multipliers based on strategy count:

**Using All 6 Edge Case Folders (14 strategies, 42 variants):**
- Multiplier: **1.1-1.15** (recommended: **1.15**)
- Result: 87% exact solutions, 13% edge cases
- Example: 5,000 original → ~5,750 total
- Use case: Comprehensive mathematical reasoning training

**Using Subset (2-3 folders, ~7-10 strategies, 21-30 variants):**
- Multiplier: **1.2-1.25**
- Result: 80-83% exact solutions, 17-20% edge cases
- Example: 5,000 original → ~6,000-6,250 total
- Use case: Balanced production training

**Using 1 Folder (1-5 strategies, 3-15 variants):**
- Multiplier: **1.3-1.5**
- Result: 67-77% exact solutions, 23-33% edge cases
- Example: 5,000 original → ~6,500-7,500 total
- Use case: Targeted training on specific edge cases

**Rationale:**
- Most real-world Fredholm equations have exact solutions
- Primary goal is teaching solution methods, not just edge case detection
- Lower edge case ratio prevents models from being overly cautious
- Higher multipliers with fewer strategies maintain similar absolute edge case counts

```bash
# Recommended: All 6 edge case folders with conservative multiplier
# Windows PowerShell:
uv run python scripts/prepare_dataset.py `
  --input data/raw/Fredholm_Dataset_Sample.csv `
  --augment --augment-multiplier 1.15 `
  --augment-strategies approx_coef discrete_points series family regularized none_solution

# Linux/macOS:
uv run python scripts/prepare_dataset.py \
  --input data/raw/Fredholm_Dataset_Sample.csv \
  --augment --augment-multiplier 1.15 \
  --augment-strategies approx_coef discrete_points series family regularized none_solution

# Subset: Focus on approximate coefficients and discrete points
uv run python scripts/prepare_dataset.py \
  --augment --augment-multiplier 1.2 \
  --augment-strategies approx_coef discrete_points

# Single folder: Deep dive into approximate coefficient edge cases
uv run python scripts/prepare_dataset.py \
  --augment --augment-multiplier 1.3 \
  --augment-strategies approx_coef
```

---

## Data Validation

The project includes comprehensive validation to ensure data quality and consistency throughout the pipeline.

### Features

- **Schema validation**: Verifies all required fields (u, f, kernel, lambda, domain)
- **Type checking**: Ensures correct data types and value ranges
- **Edge case validation**: Checks consistency of has_solution and solution_type fields
- **Strategy validation**: Verifies all 14 edge case strategies generate data correctly
- **Distribution analysis**: Reports balance across solution types and strategies
- **Sample validation**: Tests random subset (default 100 samples) to catch issues quickly

### Usage

```bash
# Validate during data preparation (recommended)
python scripts/prepare_dataset.py \
  --input data/raw/Fredholm_Dataset_Sample.csv \
  --augment --augment-multiplier 1.15 \
  --validate

# Standalone validation of augmented data
python scripts/validate_augmented_data.py \
  --input data/processed/training_data/augmented/
```

**Output**: Validation report showing:
- Total equations by solution type
- Strategy coverage (all 14 strategies validated)
- Edge case distribution
- Any errors or inconsistencies found

---

## Dataset Splitting with Stratification

The project includes production-ready stratified splitting to maintain dataset balance across train/validation/test sets.

### Features

- **Industry-standard libraries**: Uses scikit-learn's `train_test_split` with pandas DataFrames
- **Stratified splitting**: Maintains balance across:
  - Original vs augmented equations (typically 87% / 13%)
  - Solution types: exact_symbolic, exact_coef, approx_coef, discrete_points, series, family, regularized, none
  - Source strategies (basic transformations and 14 edge case strategies)
- **Flexible split ratios**: Default 80/0/20 (train/val/test), customizable
- **Robust edge case handling**:
  - Auto-adjusts invalid ratios
  - Handles empty datasets and single items
  - Graceful fallback for small strata
  - Reproducible with seed control

### Usage

```bash
# Default split (80% train, 0% validation, 20% test)
python scripts/prepare_dataset.py \
  --input data/raw/Fredholm_Dataset_Sample.csv \
  --split

# Custom split ratios
python scripts/prepare_dataset.py \
  --input data/raw/Fredholm_Dataset_Sample.csv \
  --split --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

# Full pipeline: augment + validate + split
python scripts/prepare_dataset.py \
  --input data/raw/Fredholm_Dataset_Sample.csv \
  --augment --augment-multiplier 1.15 \
  --augment-strategies no_solution numerical_only regularization_required non_unique_solution \
  --validate \
  --split --train-ratio 0.8 --val-ratio 0.0 --test-ratio 0.2 \
  --convert --convert-formats infix latex
```

**Output**: Creates separate files for each split (e.g., `training_data_train.csv`, `training_data_test.csv`) with balanced distributions.

---

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fred-llm.git
cd fred-llm
```

2. Create virtual environment and install dependencies:
```bash
uv venv
uv pip install -e ".[dev]"
```

3. Activate the environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

> **Note**: You can skip activating the virtual environment by using `uv run` to execute commands (e.g., `uv run python -m src.cli ...`). This ensures dependencies are automatically managed.

4. Set up environment variables:
```bash
# Copy the sample env file
cp .env.sample .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your-key-here
```

## Usage

### CLI

> **Note**: All commands below use `uv run` to automatically manage dependencies. If you have activated the virtual environment, you can omit `uv run` and use `python` directly.

Run the main pipeline:
```bash
uv run python -m src.cli run --config config.yaml
```

Evaluate results:
```bash
uv run python -m src.cli evaluate data/processed/results.json --mode both
```

Convert equation formats:
```bash
# Convert to RPN format and save as JSON
uv run python -m src.cli convert data/raw/equations.csv --format rpn --output data/processed/equations_rpn.json

# Convert to LaTeX format and save as CSV
uv run python -m src.cli convert data/raw/equations.csv --format latex --output data/processed/equations_latex.csv

# Convert to tokenized format (default JSON output)
uv run python -m src.cli convert data/raw/equations.csv --format tokenized --output data/processed/equations_tokenized.json

# Auto-detect output type from file extension
uv run python -m src.cli convert data/raw/equations.csv --format rpn --output output.csv  # Creates CSV
uv run python -m src.cli convert data/raw/equations.csv --format rpn --output output.json  # Creates JSON
```

Generate prompts:
```bash
uv run python -m src.cli prompt "u(x) - ∫_0^1 x*t*u(t)dt = x" --style chain-of-thought
```

### Python API

```python
from src.config import load_config
from src.main import FredLLMPipeline

# Load configuration
config = load_config("config.yaml")

# Create and run pipeline
pipeline = FredLLMPipeline(config)
results = pipeline.run()

# Solve a single equation
solution = pipeline.run_single(
    equation="u(x) - ∫_0^1 x*t*u(t)dt = x",
    kernel="x*t",
    f="x",
    lambda_val=1.0
)
```

### Scripts

Prepare dataset:
```bash
# Default: converts full dataset to ALL formats (both CSV and JSON output)
uv run python scripts/prepare_dataset.py

# This creates (if input is Fredholm_Dataset.csv):
# data/processed/
#   ├── base_equations.json + base_equations.csv
#   ├── Fredholm_Dataset_infix.json + Fredholm_Dataset_infix.csv
#   ├── Fredholm_Dataset_latex.json + Fredholm_Dataset_latex.csv
#   ├── Fredholm_Dataset_rpn.json + Fredholm_Dataset_rpn.csv
#   ├── Fredholm_Dataset_tokenized.json + Fredholm_Dataset_tokenized.csv
#   └── Fredholm_Dataset_python.json + Fredholm_Dataset_python.csv

# JSON only
uv run python scripts/prepare_dataset.py --output-format json

# CSV only
uv run python scripts/prepare_dataset.py --output-format csv

# Disable conversion (just load and save base data)
uv run python scripts/prepare_dataset.py --no-convert

# Convert specific formats only
uv run python scripts/prepare_dataset.py --convert-formats latex rpn

# Full pipeline with edge case augmentations and stratified splitting
uv run python scripts/prepare_dataset.py \
  --augment --augment-multiplier 1.15 \
  --validate \
  --split --train-ratio 0.8 --val-ratio 0.0 --test-ratio 0.2 \
  --output-format csv

# Include edge cases: folder-based strategies run all contained strategies
# These teach LLMs to recognize equations without symbolic solutions
uv run python scripts/prepare_dataset.py \
  --augment \
  --augment-strategies substitute scale none_solution approx_coef regularized

# All augmentation strategies (4 basic + 6 edge case folders = 18 strategies, 42 variants)
uv run python scripts/prepare_dataset.py \
  --augment \
  --augment-strategies substitute scale shift compose approx_coef discrete_points series family regularized none_solution \
  --augment-multiplier 5

# Quick test with 100 samples
uv run python scripts/prepare_dataset.py \
  --max-samples 100 \
  --convert-limit 100
```

Convert to RPN:
```bash
uv run python scripts/convert_to_rpn.py --input data/processed/train.json --output data/processed/train_rpn.json
```

Generate prompts:
```bash
uv run python scripts/generate_prompts.py --input data/processed/test.json --style chain-of-thought
```

## Dataset

### Fredholm-LLM Dataset

This project uses the Fredholm-LLM dataset, a collection of ~500,000 Fredholm integral equations of the second kind.

**Source**: [Fredholm-LLM on GitHub](https://github.com/alirezaafzalaghaei/Fredholm-LLM)  
**DOI**: [10.5281/zenodo.16784707](https://doi.org/10.5281/zenodo.16784707)

### Dataset Schema

| Field | Description |
|-------|-------------|
| `u` | Solution function u(x) |
| `f` | Right-hand side function f(x) |
| `kernel` | Kernel function K(x, t) |
| `lambda` | Parameter λ |
| `a`, `b` | Integration bounds |

### Download Dataset

Use the CLI to download the dataset from Zenodo:

```bash
# Download sample dataset (~5K equations, recommended for testing)
python -m src.cli dataset download --variant sample

# Download full dataset (~500K equations)
python -m src.cli dataset download --variant full

# Show dataset info
python -m src.cli dataset info

# Show dataset statistics
python -m src.cli dataset stats

# Display sample equations
python -m src.cli dataset sample --max-samples 5
```

### Expression Types

The dataset includes equations with various expression types:

- **Polynomial**: `x**2`, `t**3 + t`
- **Trigonometric**: `sin(x)`, `cos(t)`
- **Hyperbolic**: `sinh(x)`, `cosh(t)`
- **Exponential**: `exp(x)`, `exp(-t**2)`

## Data Formatters

The project includes 8 specialized formatters for converting mathematical expressions:

### Basic Formatters
- **InfixFormatter** - Standard notation: `x**2 + 2*x + 1`
- **LaTeXFormatter** - LaTeX format: `x^{2} + 2 x + 1`
- **RPNFormatter** - Reverse Polish Notation: `x 2 ^ 2 x * + 1 +`
- **PythonFormatter** - Executable Python code
- **TokenizedFormatter** - Space-separated tokens for LLMs

### Equation Formatters
- **FredholmEquationFormatter** - Complete equations with all components (u, f, kernel, λ, bounds)
  ```python
  # Result: "u(x) - 0.5 * ∫[0,1] (x*t) u(t) dt = x**2 + 2*x"
  ```
- **TokenizedEquationFormatter** - With special tokens for structured representation
  ```python
  # Result: "u ( x ) - <LAMBDA> 0.5 <INT> <LOWER> 0 <UPPER> 1 ..."
  # Special tokens: <LAMBDA>, <INT>, <LOWER>, <UPPER>, <SEP>
  ```

### Series Formatters
- **SeriesFormatter** - Taylor/Maclaurin series expansions
  ```python
  # Result: "x - x**3/6 + x**5/120 + O(x**6)"
  ```
- **NeumannSeriesFormatter** - Iterative Fredholm solutions
  ```python
  # Result: "u(x) = f + λKf + λ²K²f + O(λ³)"
  ```

All formatters support:
- **Canonicalization**: Optional `simplify` parameter for consistent formatting
- **Roundtrip conversion**: Expression → SymPy → Target format
- **SymPy integration**: Uses SymPy as intermediate canonical form
- **CSV export**: Direct export to CSV format matching original dataset structure

See [src/data/formatters/README.md](src/data/formatters/README.md) for detailed examples.

## Project Structure

```text
fred-llm/
├── .env.sample              # Environment variables template
├── config.yaml              # Default runtime configuration
├── pyproject.toml           # Dependencies and project metadata
├── configs/                 # Configuration presets
│   ├── default.yaml         # Standard settings
│   ├── development.yaml     # Fast iteration settings
│   ├── production.yaml      # Full evaluation settings
│   ├── local.yaml           # Local model settings
│   └── fine_tuning.yaml     # Training settings
├── src/
│   ├── cli.py               # CLI entrypoint
│   ├── config.py            # Configuration loader
│   ├── main.py              # Pipeline orchestrator
│   ├── llm/                 # LLM-related modules
│   │   ├── model_runner.py  # Model inference
│   │   ├── prompt_templates.py
│   │   ├── postprocess.py   # Output parsing
│   │   └── evaluate.py      # Evaluation metrics
│   ├── data/                # Data handling
│   │   ├── loader.py
│   │   ├── augmentation.py
│   │   ├── format_converter.py
│   │   └── validator.py
│   └── utils/               # Utilities
│       ├── math_utils.py
│       ├── logging_utils.py
│       └── visualization.py
├── scripts/                 # Utility scripts
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit tests
├── docs/                    # Documentation
└── data/                    # Data directory
    ├── raw/
    ├── processed/
    └── prompts/
```

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  provider: openai
  name: gpt-4
  temperature: 0.1

prompting:
  style: chain-of-thought
  include_examples: true
  num_examples: 3

evaluation:
  mode: both  # symbolic, numeric, or both
  symbolic_tolerance: 1e-10
  numeric_tolerance: 1e-6
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
# Format with Ruff
ruff format .

# Lint
ruff check .
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [SymPy](https://www.sympy.org/) for symbolic mathematics
- [OpenAI](https://openai.com/) for GPT models
- Research on LLMs for mathematical reasoning
