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

## Features

- 🤖 **Multi-provider LLM support**: OpenAI API, local models (HuggingFace, vLLM)
- 📝 **Multiple prompting styles**: Basic, Chain-of-Thought, Few-Shot, Tool-Assisted
- 🔢 **Symbolic & numeric evaluation**: SymPy-based parsing and verification
- 📊 **Format conversion**: LaTeX, RPN, tokenized representations
- 📈 **Comprehensive metrics**: Symbolic equivalence, numeric accuracy, solution verification

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

4. Set up environment variables:
```bash
# Create .env file or set directly
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### CLI

Run the main pipeline:
```bash
python -m src.cli run --config config.yaml
```

Evaluate results:
```bash
python -m src.cli evaluate data/processed/results.json --mode both
```

Convert equation formats:
```bash
python -m src.cli convert data/raw/equations.json --format rpn
```

Generate prompts:
```bash
python -m src.cli prompt "u(x) - ∫_0^1 x*t*u(t)dt = x" --style chain-of-thought
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
python scripts/prepare_dataset.py --input data/raw/fie_500k.json --output data/processed/ --split
```

Convert to RPN:
```bash
python scripts/convert_to_rpn.py --input data/processed/train.json --output data/processed/train_rpn.json
```

Generate prompts:
```bash
python scripts/generate_prompts.py --input data/processed/test.json --style chain-of-thought
```

## Project Structure

```
fred-llm/
├── config.yaml              # Runtime configuration
├── pyproject.toml           # Dependencies and project metadata
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
