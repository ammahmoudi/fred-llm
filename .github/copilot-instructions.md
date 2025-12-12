# Fred-LLM Project Instructions

This project solves and approximates Fredholm integral equations of the second kind using LLMs.

## Project Structure
- `src/` - Core Python source code
- `scripts/` - Dataset preparation and utility scripts
- `notebooks/` - Jupyter notebooks for exploration
- `data/` - Raw, processed, and prompt data
- `tests/` - Unit tests
- `docs/` - Documentation including pipeline diagrams

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
- Configuration via `config.yaml`

## CLI Usage
```bash
python -m src.cli run --config config.yaml
```
