# Configs Directory

This directory contains configuration presets for different use cases.

## Available Configurations

| Config | Description | Use Case |
|--------|-------------|----------|
| `default.yaml` | Standard balanced settings | General usage |
| `development.yaml` | Fast iteration settings | Local development & testing |
| `production.yaml` | Full evaluation settings | Final benchmarking |
| `local.yaml` | Local model settings | Self-hosted models (vLLM, Ollama) |
| `fine_tuning.yaml` | Training settings | Fine-tuning experiments |

## Usage

```bash
# Use a specific config
python -m src.cli run --config configs/development.yaml

# Override specific values
python -m src.cli run --config configs/default.yaml --model.temperature 0.5
```

## Creating Custom Configs

1. Copy an existing config as a starting point:
   ```bash
   cp configs/default.yaml configs/my_experiment.yaml
   ```

2. Edit the values as needed

3. Run with your custom config:
   ```bash
   python -m src.cli run --config configs/my_experiment.yaml
   ```

## Environment Variables

Configs can reference environment variables. See `.env.sample` for available options.

Sensitive values (API keys) should be set via environment variables, not in config files.
