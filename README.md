# Fred-LLM

**Solving Fredholm Integral Equations using Large Language Models**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-134%20passing-brightgreen.svg)](tests/)

Train and evaluate LLMs on solving Fredholm integral equations with realistic edge cases.

## Overview

This project provides a complete pipeline for training LLMs to solve Fredholm integral equations:

**Fredholm Equation (Second Kind)**:
$(x) - \lambda \int_a^b K(x, t) u(t) \, dt = f(x)README.md.backup

**Fredholm Equation (First Kind - Ill-posed)**:
$$\int_a^b K(x, t) u(t) \, dt = g(x)README.md.backup

### Key Features

- 🤖 **Multi-provider LLM support**: OpenAI, OpenRouter, local models
- 📝 **4 prompting styles**: Basic, Chain-of-Thought, Few-Shot, Tool-Assisted  
- 🎯 **14 edge case strategies**: Teach LLMs to recognize when equations require special handling
- 📊 **8 solution types**: From exact symbolic to no solution
- 🔄 **Data augmentation**: Generate realistic edge cases automatically
- ⚖️ **Stratified splitting**: Maintain dataset balance across train/val/test
- 📈 **Comprehensive evaluation**: Symbolic, numeric, and edge case recognition metrics
- 🔢 **Multiple formats**: LaTeX, RPN, Infix, Python, Tokenized

## Quick Start

**5-minute setup** → [Full Quick Start Guide](docs/QUICKSTART.md)

```bash
# 1. Install
git clone https://github.com/ammahmoudi/fred-llm.git && cd fred-llm
uv venv && uv pip install -e ".[dev]"

# 2. Download dataset (5K sample)
python -m src.cli dataset download --variant sample

# 3. Prepare data with edge cases (specify strategies explicitly)
python scripts/prepare_dataset.py \
  --input data/raw/Fredholm_Dataset_Sample.csv \
  --augment --augment-multiplier 1.15 \
  --augment-strategies approx_coef discrete_points series family regularized none_solution \
  --validate --split --convert
# This applies all 6 edge case folders (14 strategies, 42 variants total)

# 4. Generate prompts (output format matches input format)
python scripts/run_prompt_generation.py \
  --input data/processed/training_data/formatted/ \
  --output data/prompts --styles all \
  --pattern "*_infix.csv"  # or *_latex.csv, *_rpn.csv
```

→ **Detailed instructions**: [Quick Start Guide](docs/QUICKSTART.md)

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/QUICKSTART.md) | 5-minute setup guide |
| [Augmentation Guide](docs/AUGMENTATION.md) | Edge case strategies and solution types |
| [Development Guide](docs/DEVELOPMENT.md) | Contributing and development workflow |
| [Pipeline Diagram](docs/pipeline-diagram.md) | System architecture overview |
| [Features Tracking](docs/FEATURES.md) | Implementation status (60% complete) |
| [Edge Cases](docs/EDGE_CASES.md) | Detailed edge case documentation |

## Solution Types (8 Categories)

| Type | Description | Example |
|------|-------------|---------|
| `exact_symbolic` | Closed-form solution | u(x) = sin(x) + x² |
| `exact_coef` | Exact with coefficients | u(x) = c₁sin(x) + c₂cos(x) |
| `approx_coef` | Approximate with coefficients | u(x) ≈ a₀ + a₁x + a₂x² |
| `discrete_points` | Solution at discrete points only | [(0, 1.2), (0.5, 3.4), ...] |
| `series` | Infinite series solution | u(x) = Σ aₙxⁿ |
| `family` | Non-unique solution family | u(x) = f(x) + C·φ(x) |
| `regularized` | Requires regularization | Tikhonov, Landweber |
| `none` | No solution exists | - |

→ **Full taxonomy**: [Augmentation Guide](docs/AUGMENTATION.md)

## Edge Case Strategies (14 Strategies, 42 Variants)

Organized by solution type folders:

- **`approx_coef`** (5 strategies): Weakly singular, boundary layer, oscillatory, mixed type, compact support
- **`discrete_points`** (2 strategies): Complex kernels, near-resonance
- **`series`** (1 strategy): Neumann series
- **`family`** (1 strategy): Exact resonance
- **`regularized`** (1 strategy): Ill-posed (Fredholm 1st kind)
- **`none_solution`** (4 strategies): Eigenvalue issue, range violation, divergent kernel, disconnected support

→ **Detailed guide**: [Augmentation Guide](docs/AUGMENTATION.md)

## Dataset

**Fredholm-LLM Dataset** from Zenodo (DOI: [10.5281/zenodo.16784707](https://doi.org/10.5281/zenodo.16784707))

- **Sample**: 5,000 equations (37 MB)
- **Full**: 500,000 equations (3.7 GB)

All equations are Fredholm second kind with verified solutions. Edge cases (first kind, no solution, etc.) are generated via augmentation.

## Pipeline Architecture

Visual overview → [Full Pipeline Diagram](docs/pipeline-diagram.md)

**4 Main Modules**:
1. **Dataset Preparation** - Augmentation (14 strategies), formatting (8 types), validation, splitting
2. **Prompt Engineering** - 4 styles (basic, CoT, few-shot, tool-assisted), 3 edge case modes
3. **LLM Methods** - Fine-tuning, in-context learning, tool use
4. **Evaluation** - Symbolic matching, numeric metrics, edge case recognition

## Project Structure

```
fred-llm/
├── src/                   # Core source code
│   ├── data/             # Data loading, augmentation, formatting
│   ├── prompts/          # Prompt generation system
│   ├── llm/              # Model runners and evaluation
│   └── utils/            # Utilities
├── scripts/              # CLI scripts
├── tests/                # Unit tests (134 tests passing)
├── docs/                 # Documentation
├── configs/              # Configuration presets
└── data/                 # Data directory
```

→ **Development guide**: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

## Configuration

Simple YAML configuration with 6 presets in `configs/`:

```yaml
model:
  provider: openai
  name: gpt-4

prompting:
  style: chain-of-thought
  edge_case_mode: none

data:
  augment_multiplier: 1.15
  augment_strategies: [approx_coef, discrete_points, series, family, regularized, none_solution]
```

→ **Full options**: [configs/README.md](configs/README.md)

## Testing

```bash
# Run all 134 tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src
```

**All tests passing** ✅

→ **Development workflow**: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes + add tests
4. Format code (`ruff format .`)
5. Submit pull request

→ **Development guide**: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

## Citation

```bibtex

```

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [Zenodo](https://zenodo.org/) - Dataset hosting
- [SymPy](https://www.sympy.org/) - Symbolic mathematics
- [OpenAI](https://openai.com/) / [OpenRouter](https://openrouter.ai/) - LLM access
