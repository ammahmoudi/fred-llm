# Prompt Generation Module

Complete prompt engineering system for Fredholm integral equations using OOP design with base classes and style inheritance.

## Architecture

### Base Classes (`src/prompts/base.py`)

**PromptStyle (Abstract Base Class)**
- Defines interface for all prompt styles
- Methods: `get_system_prompt()`, `get_user_prompt()`, `generate()`, `generate_batch()`
- Handles metadata and ground truth management

**Data Classes**
- `EquationData` - Container for equation parameters (u, f, kernel, lambda_val, a, b)
- `GeneratedPrompt` - Result object with prompt, style, metadata, ground_truth

### Style Implementations (`src/prompts/styles/`)

Each style extends `PromptStyle` base class:

1. **BasicPromptStyle** - Simple direct prompts
2. **ChainOfThoughtPromptStyle** - Step-by-step reasoning  
3. **FewShotPromptStyle** - Includes worked examples
4. **ToolAssistedPromptStyle** - Enables computational tools

### Factory Pattern (`src/prompts/factory.py`)

```python
create_prompt_style(
    style: str,
    include_examples: bool = True,
    num_examples: int = 2
) -> PromptStyle
```

### Batch Processing (`src/prompts/batch_processor.py`)

**BatchPromptProcessor**
- CSV → JSONL pipeline
- Progress tracking
- Format inference (infix/latex/rpn)
- Metadata preservation

## Usage

### Direct Style Usage

```python
from src.prompts import create_prompt_style, EquationData

# Create style
style = create_prompt_style("chain-of-thought")

# Create equation
eq = EquationData(
    u="x**2",
    f="x**2 + 2*x",
    kernel="x*t",
    lambda_val=0.5,
    a=0.0,
    b=1.0
)

# Generate prompt
prompt = style.generate(eq, include_ground_truth=True)
print(prompt.prompt)
```

### Batch Processing

```python
from src.prompts import create_processor

processor = create_processor(
    style="few-shot",
    output_dir="data/prompts",
    include_ground_truth=True
)

output_file = processor.process_dataset(
    "data/processed/train_infix.csv",
    format_type="infix"
)
```

### CLI Commands

```bash
# Generate prompts for dataset
uv run python -m src.cli prompt generate train.csv \
    --style chain-of-thought \
    --output data/prompts

# Batch process multiple files
uv run python -m src.cli prompt batch data/processed/ \
    --styles "basic,few-shot" \
    --pattern "train_*.csv"
```

## Module Structure

```
src/prompts/
├── __init__.py          # Public exports
├── base.py              # Base classes (PromptStyle, EquationData, GeneratedPrompt)
├── factory.py           # Factory function (create_prompt_style)
├── batch_processor.py   # BatchPromptProcessor class
├── templates.py         # Shared templates and examples
└── styles/
    ├── __init__.py
    ├── basic.py              # BasicPromptStyle
    ├── chain_of_thought.py   # ChainOfThoughtPromptStyle
    ├── few_shot.py           # FewShotPromptStyle
    └── tool_assisted.py      # ToolAssistedPromptStyle
```

## Design Principles

✅ **Single Responsibility** - Each style class handles one prompt type
✅ **Open/Closed** - Easy to add new styles by extending `PromptStyle`
✅ **Liskov Substitution** - All styles interchangeable via base class
✅ **Interface Segregation** - Clear abstract methods in base class
✅ **Dependency Inversion** - Depend on `PromptStyle` abstraction

## Adding New Styles

```python
from src.prompts.base import PromptStyle, EquationData

class CustomPromptStyle(PromptStyle):
    def __init__(self, **kwargs):
        super().__init__(style_name="custom", **kwargs)
    
    def get_system_prompt(self) -> str:
        return "Your custom system prompt..."
    
    def get_user_prompt(self, equation: EquationData, format_type: str) -> str:
        return f"Solve: {equation.f} with kernel {equation.kernel}"
```

Then register in `factory.py`:

```python
def create_prompt_style(style: str, ...) -> PromptStyle:
    styles = {
        ...
        "custom": CustomPromptStyle,
    }
```

## Tests

All 18 tests passing:
- EquationData creation
- All 4 prompt styles
- Batch processing
- JSONL export
- Factory functions

Run tests:
```bash
uv run pytest tests/test_prompt_generation.py -v
```

**Backward Compatibility**: None - this is a breaking refactor
