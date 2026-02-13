# Development Guide

Guide for contributing to and developing Fred-LLM.

## Project Structure

```
fred-llm/
├── src/
│   ├── adaptive_config.py   # Adaptive config schema (current)
│   ├── adaptive_pipeline.py # Adaptive pipeline orchestrator
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Legacy config loader
│   ├── llm/                 # LLM-related modules
│   │   ├── model_runner.py  # Model inference
│   │   ├── postprocess.py   # Output parsing
│   │   └── evaluate.py      # Evaluation metrics
│   ├── data/                # Data handling
│   │   ├── loader.py
│   │   ├── augmentation.py
│   │   ├── format_converter.py
│   │   └── validator.py
│   ├── prompts/             # Prompt generation system
│   │   ├── base.py          # Base classes
│   │   ├── factory.py       # Factory pattern
│   │   ├── batch_processor.py
│   │   └── styles/          # Prompt style implementations
│   └── utils/               # Utilities
│       ├── math_utils.py
│       ├── logging_utils.py
│       └── visualization.py
├── scripts/                 # Internal pipeline scripts (called by CLI)
│   ├── prepare_dataset.py
│   ├── run_prompt_generation.py
├── notebooks/               # Jupyter notebooks
│   ├── explore_data.ipynb
│   ├── prompt_design.ipynb
│   └── evaluate_results.ipynb
├── tests/                   # Unit tests
│   ├── test_loader.py
│   ├── test_augmentation.py
│   ├── test_formatters.py
│   └── test_prompting.py
├── docs/                    # Documentation
│   ├── QUICKSTART.md
│   ├── AUGMENTATION.md
│   ├── FEATURES.md
│   └── pipeline-diagram.md
├── configs/                 # Configuration presets
│   ├── development.yaml
│   ├── production.yaml
│   └── README.md
└── data/                    # Data directory
    ├── raw/
    ├── processed/
    └── prompts/
```

## Setup Development Environment

```bash
# Clone repository
git clone https://github.com/ammahmoudi/fred-llm.git
cd fred-llm

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install with development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_augmentation.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Test Organization

- `test_loader.py` - Data loading tests (13 tests)
- `test_augmentation.py` - Augmentation strategies (22 tests)
- `test_formatters.py` - Format converters (18 tests)
- `test_splitting.py` - Dataset splitting (19 tests)
- `test_validation.py` - Data validation (5 tests)
- `test_prompting.py` - Prompt generation (30 tests)
- `test_model_runner.py` - LLM integration (scaffolded)

**Total**: See tests/ for current coverage (run pytest for an accurate count)

## Reproducibility

### Setting Global Random Seeds

Fred-LLM provides a global seed management system to ensure reproducible results across all random operations:

```bash
# Use seed from config (default: 42)
uv run python -m src.cli run --config config.yaml

# Override seed from CLI (takes precedence)
uv run python -m src.cli run --config config.yaml --seed 12345
```

### What the Seed Controls

The global seed (`src/utils/random_seed.py::set_global_seed()`) manages:
- **Data augmentation**: Edge case generation (strategies like oscillatory, boundary layer, etc.)
- **Dataset splitting**: Train/val/test allocation
- **Evaluation points**: Point sampling for numeric evaluation
- **All random libraries**: Python's `random`, NumPy, PyTorch, TensorFlow

### Config-Based Seed

Set seed in your YAML configuration:

```yaml
dataset:
  raw:
    seed: 42  # Applied globally on pipeline initialization
    augment: true
    split: true
```

### Using in Code

If you're not using the CLI, set the seed manually:

```python
from src.utils.random_seed import set_global_seed

set_global_seed(42)  # Call once at program start
```

The seed is automatically applied when the CLI loads a config.

## Code Style

### Formatting with Ruff

```bash
# Format all code
ruff format .

# Check formatting without changes
ruff format --check .

# Lint code
ruff check .

# Fix linting issues automatically
ruff check --fix .
```

### Style Guidelines

- **Line length**: 100 characters
- **Imports**: Organized with isort (automatic)
- **Type hints**: Use throughout, especially for public APIs
- **Docstrings**: Google style for all public functions/classes

Example:
```python
def augment_equation(
    equation: dict[str, Any],
    strategy: str,
    multiplier: float = 1.15,
) -> list[dict[str, Any]]:
    """Augment equation with edge cases.
    
    Args:
        equation: Original equation dictionary with u, f, kernel, etc.
        strategy: Augmentation strategy name (e.g., 'weakly_singular')
        multiplier: Target dataset size multiplier (default: 1.15)
        
    Returns:
        List of augmented equation dictionaries with metadata.
        
    Raises:
        ValueError: If strategy is unknown or equation is invalid.
    """
    ...
```

## Configuration

Edit `config.yaml` or use presets in `configs/`:

```yaml
dataset:
    raw:
        path: data/raw/Fredholm_Dataset_Sample.csv
        max_samples: 5000
        augment: true
        augment_multiplier: 1.15
        augment_strategies:
            - approx_coef
            - discrete_points
            - series
            - family
            - regularized
            - none_solution
        split: true
        convert_formats: [infix, latex, rpn]

    prompting:
        style: chain-of-thought  # basic, chain-of-thought, few-shot, tool-assisted
        edge_case_mode: none     # none, guardrails, hints
        include_examples: true
        num_examples: 3

model:
    provider: openai  # openai, openrouter, local
    name: gpt-4o-mini
    temperature: 0.1
    max_tokens: 2048

evaluation:
    mode: both  # symbolic, numeric, or both
    symbolic_tolerance: 1e-10
    numeric_tolerance: 1e-6
    num_test_points: 100

output:
    dir: outputs
```

### Configuration Presets

- `development.yaml` - Fast iteration, small samples
- `production.yaml` - Full evaluation, all strategies
- `local.yaml` - Local models (HuggingFace, vLLM)
- `openrouter.yaml` - OpenRouter with popular models
- `fine_tuning.yaml` - Fine-tuning configuration

## Adding New Features

### 1. Adding a New Augmentation Strategy

Create new file in `src/data/augmentations/{solution_type}/`:

```python
from src.data.augmentations.base import BaseAugmentation

class MyNewAugmentation(BaseAugmentation):
    @property
    def strategy_name(self) -> str:
        return "my_strategy"
    
    @property
    def description(self) -> str:
        return "Brief description of what this does"
    
    def augment(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate edge case variants."""
        # Implementation here
        result = {
            "u": "",  # or actual solution
            "f": modified_f,
            "kernel": modified_kernel,
            "lambda_val": item["lambda_val"],
            "a": item["a"],
            "b": item["b"],
            "augmentation_type": self.strategy_name,
            "has_solution": False,  # or True
            "solution_type": "none",  # or appropriate type
            "edge_case": "Brief description",
            "reason": "Mathematical reason",
            "recommended_methods": ["method1", "method2"],
            # ... other required fields
        }
        return [result]
```

Register in `src/data/augmentations/{folder}/__init__.py`:
```python
from .my_strategy import MyNewAugmentation

__all__ = ["MyNewAugmentation", ...]
```

### 2. Adding a New Prompt Style

Create file in `src/prompts/styles/`:

```python
from src.prompts.base import PromptStyle, EquationData

class MyPromptStyle(PromptStyle):
    def __init__(self, **kwargs):
        super().__init__(style_name="my-style", **kwargs)
    
    def get_system_prompt(self) -> str:
        return """Your system prompt here..."""
    
    def get_user_prompt(self, eq: EquationData) -> str:
        return f"""Solve this equation: {eq.format()}"""
```

Register in `src/prompts/factory.py`:
```python
STYLE_MAP = {
    "my-style": MyPromptStyle,
    ...
}
```

### 3. Adding a New Formatter

Create file in `src/data/formatters/`:

```python
from src.data.formatters.base import BaseFormatter

class MyFormatter(BaseFormatter):
    @property
    def name(self) -> str:
        return "my_format"
    
    def format(self, equation: dict[str, Any]) -> str:
        """Convert equation to your format."""
        # Implementation
        return formatted_string
```

### 4. Adding Tests

Create test file in `tests/`:

```python
import pytest
from src.data.augmentations.my_folder.my_strategy import MyNewAugmentation

def test_my_strategy_basic():
    """Test basic functionality."""
    aug = MyNewAugmentation()
    item = {
        "u": "x",
        "f": "x",
        "kernel": "1",
        "lambda_val": 0.5,
        "a": 0,
        "b": 1,
    }
    results = aug.augment(item)
    
    assert len(results) > 0
    assert results[0]["augmentation_type"] == "my_strategy"
    assert "has_solution" in results[0]
    assert "solution_type" in results[0]

def test_my_strategy_schema():
    """Test output schema compliance."""
    aug = MyNewAugmentation()
    # ... test all 18 required fields present
```

Run new tests:
```bash
pytest tests/test_my_feature.py -v
```

## Documentation Updates

When adding features, update:

1. **`docs/FEATURES.md`** - Mark feature as complete, add description
2. **`docs/AUGMENTATION.md`** - If adding augmentation strategy
3. **`src/{module}/README.md`** - Module-specific documentation
4. **This file** - If adding new development patterns

## Pull Request Checklist

Before submitting a PR:

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Code formatted (`ruff format .`)
- [ ] No linting errors (`ruff check .`)
- [ ] Type hints added for new functions
- [ ] Docstrings added (Google style)
- [ ] Tests added for new features
- [ ] Documentation updated
- [ ] `FEATURES.md` updated
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages clear and descriptive

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with changes
3. Run full test suite
4. Tag release: `git tag v0.x.0`
5. Push tag: `git push origin v0.x.0`
6. Create GitHub release with notes

## Common Development Tasks

### Rebuild Dataset

```bash
# Full rebuild with all edge cases
python scripts/prepare_dataset.py \
    --input data/raw/Fredholm_Dataset_Sample.csv \
    --output data/processed/training_data \
    --augment --validate --split --convert
```

### Regenerate Prompts

```bash
uv run python -m src.cli prompt batch \
    data/processed/training_data/formatted \
    --output data/prompts \
    --styles all
```

### Validate Data Quality

```bash
# Validation integrated
python scripts/prepare_dataset.py \
    --input data/raw/Fredholm_Dataset_Sample.csv \
    --output data/processed/training_data \
    --augment --validate
```

### Run Pipeline

```bash
python -m src.cli run --config config.yaml
```

## Troubleshooting

**Import errors after adding new module**
- Reinstall package: `uv pip install -e .`

**Tests fail after refactoring**
- Check test fixtures match new signatures
- Update mock objects if interfaces changed

**Ruff formatting conflicts**
- Use `ruff format .` to auto-fix
- Check `.ruff.toml` for project-specific rules

**Type checking errors**
- Run `mypy src/` to see all type issues
- Add type hints incrementally

**YAML config encoding errors (Windows)**
- Error: `UnicodeDecodeError: 'cp1252' codec can't decode byte...`
- Cause: Unicode characters (✅, ⚠️, ℹ️, →, box-drawing) in YAML files
- Solution: Use ASCII-only characters in YAML configs
- Fixed in: `configs/prepare_data.yaml` (February 11, 2026)

**RuntimeWarning: overflow in exp/cosh during augmentation**
- Error: `RuntimeWarning: overflow encountered in exp` during evaluation point generation
- Cause: Large exponents (e.g., `exp(100*x)`) producing inf/nan values
- Solution: Non-finite filtering in `BaseAugmentation._generate_evaluation_points()`
- Fixed in: `src/data/augmentations/base.py` (February 11, 2026)

**SympifyError: could not parse augmentation kernel string**
- Error: `SympifyError: could not parse 'Piecewise: nonzero in...'` or `'t if t <= x else x'`
- Cause: Placeholder strings or Python ternary operators not parseable by SymPy
- Solution: Use valid Piecewise expressions with logical operators (&, |, ~)
- Fixed in: 5 augmentation files (disconnected_support, mixed_type, compact_support, neumann_series)
- Example: `"Piecewise((expr, condition), (0, True))"` with `(x>=a) & (x<=b)` conditions

## Resources

- [SymPy Documentation](https://docs.sympy.org/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

## License

MIT License - see [LICENSE](LICENSE) for details.

