# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fred-LLM trains and evaluates LLMs on solving Fredholm integral equations. The pipeline handles data preparation, augmentation with edge cases, prompt generation, LLM inference, and evaluation.

**Fredholm Equation (Second Kind)**: u(x) - λ∫K(x,t)u(t)dt = f(x)

## Build & Development Commands

```bash
# Install dependencies
uv venv && uv pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run single test file
pytest tests/test_augmentation.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Format code
ruff format .

# Lint code
ruff check .

# Fix linting issues
ruff check --fix .

# Type check
mypy src/
```

## Main CLI Commands

```bash
# Run full pipeline with config
uv run python -m src.cli run --config configs/prepare_data.yaml

# Dry run to preview execution
uv run python -m src.cli run --config config.yaml --dry-run

# Download dataset
uv run python -m src.cli dataset download --variant sample

# Generate prompts from dataset
uv run python -m src.cli prompt generate data/train.csv --style chain-of-thought

# Batch generate prompts
uv run python -m src.cli prompt batch data/processed/ --styles all

# Convert equation formats
uv run python -m src.cli convert data/equations.json --format rpn --output output.csv
```

## Architecture

### Core Modules

- **`src/adaptive_pipeline.py`** - Main pipeline orchestrator. Detects workflow from config and chains stages automatically.
- **`src/adaptive_config.py`** - Pydantic config models with validation, path resolution, and conflict detection.
- **`src/cli.py`** - Typer-based CLI entrypoint.

### Data Module (`src/data/`)

- **`augmentation.py`** - Orchestrates augmentation strategies
- **`augmentations/`** - 14 edge case strategies organized by solution type folder:
  - `approx_coef/` - 5 strategies for approximate coefficient solutions
  - `discrete_points/` - 2 strategies for point-only solutions
  - `series/` - Neumann series solutions
  - `family/` - Non-unique solution families (resonance)
  - `regularized/` - Ill-posed problems (Fredholm 1st kind)
  - `none_solution/` - 4 strategies where no solution exists
- **`formatters/`** - Convert equations between formats (LaTeX, RPN, infix, Python, tokenized)
- **`splitter.py`** - Stratified train/val/test splitting
- **`validator.py`** - Schema and consistency validation

### Prompts Module (`src/prompts/`)

- **`base.py`** - `EquationData`, `GeneratedPrompt`, `PromptStyle` base class
- **`factory.py`** - Creates prompt style instances by name
- **`styles/`** - 4 prompt styles: basic, chain-of-thought, few-shot, tool-assisted
- **`batch_processor.py`** - Batch processing of datasets to prompts

### LLM Module (`src/llm/`)

- **`model_runner.py`** - Multi-provider model inference (OpenAI, OpenRouter, local)
- **`postprocess.py`** - Parse LLM outputs
- **`evaluate.py`** - Symbolic and numeric evaluation metrics

### Key Data Structures

**7 Solution Types**: `exact_symbolic`, `approx_coef`, `discrete_points`, `series`, `family`, `regularized`, `none`

**3 Edge Case Modes** in prompts: `none` (pure inference), `guardrails` (add handling instructions), `hints` (include solution type hints)

### Adding New Components

**New augmentation strategy**: Create class in `src/data/augmentations/{solution_type}/` extending `BaseAugmentation`. Implement `augment()` and `strategy_name`. Register in folder's `__init__.py`.

**New prompt style**: Create class in `src/prompts/styles/` extending `PromptStyle`. Implement `get_system_prompt()` and `get_user_prompt()`. Register in `src/prompts/factory.py`.

**New formatter**: Create class in `src/data/formatters/` extending `BaseFormatter`. Implement `format()` method.

## Configuration

YAML configs control the pipeline. Key sections:

```yaml
dataset:
  raw:
    path: data/raw/dataset.csv  # Start from raw CSV
  prompting:
    style: chain-of-thought
    edge_case_mode: none  # none, guardrails, hints

model:
  provider: openai  # openai, openrouter, local
  name: gpt-4
```

The pipeline auto-detects format (infix/latex/rpn) and chains outputs between stages.

## Code Style

- Line length: 88 characters (Ruff default)
- Type hints required for public APIs
- Google-style docstrings
- Always update `docs/FEATURES.md` when completing features
