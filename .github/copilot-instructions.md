# Fred-LLM Project Instructions

This project solves and approximates Fredholm integral equations of the second kind using LLMs.

## Project Structure

- `src/` - Core Python source code
  - `src/data/` - Data loading, augmentation, format conversion
  - `src/llm/` - Model runners, prompt templates, evaluation
  - `src/utils/` - Logging, math utilities, visualization
- `scripts/` - Dataset preparation and utility scripts
- `notebooks/` - Jupyter notebooks for exploration
- `data/` - Raw, processed, and prompt data
- `tests/` - Unit tests
- `docs/` - Documentation including pipeline diagrams
- `configs/` - Configuration presets (development, production, etc.)

## Dataset

The project uses the Fredholm-LLM dataset from Zenodo (DOI: 10.5281/zenodo.16784707).

Download with:
```bash
python -m src.cli dataset download --variant sample
```

## Pipeline Architecture

See `docs/pipeline-diagram.md` for the full 4-module pipeline:

1. **Dataset Preparation** - Data augmentation and format conversion
2. **Prompt Engineering** - Prompt design and output formatting
3. **LLM Methods** - Fine-tuning, in-context learning, and tool use
4. **Evaluation** - Symbolic, numeric, and robustness evaluation

## Development Guidelines

- Use `uv` for dependency management
- Python 3.11+ required
- Format code with Ruff
- Configuration via `config.yaml` or `configs/*.yaml`
- **Always update `docs/FEATURES.md`** when adding new features or completing TODO items

## CLI Usage

```bash
# Run the pipeline
python -m src.cli run --config config.yaml

# Download dataset
python -m src.cli dataset download --variant sample

# Evaluate results
python -m src.cli evaluate results.json

# Generate prompts
python -m src.cli prompt "u(x) - ∫K(x,t)u(t)dt = f(x)"
```
