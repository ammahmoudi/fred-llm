# Fred-LLM Project Instructions

This project solves and approximates Fredholm integral equations of the second kind using LLMs.

## Project Structure
- `src/` - Core Python source code
- `scripts/` - Dataset preparation and utility scripts
- `notebooks/` - Jupyter notebooks for exploration
- `data/` - Raw, processed, and prompt data
- `tests/` - Unit tests

## Development Guidelines
- Use `uv` for dependency management
- Python 3.11+ required
- Format code with Ruff
- Configuration via `config.yaml`

## CLI Usage
```bash
python -m src.cli run --config config.yaml
```
