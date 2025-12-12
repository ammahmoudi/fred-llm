# Fred-LLM Features

This document tracks all features - implemented and planned. Check off items as they are completed.

---

## Core Infrastructure

- [x] Project scaffolding - Complete Python project structure with uv/hatchling
- [x] CLI interface - Typer-based CLI with rich output formatting
- [x] Configuration system - YAML-based config with Pydantic validation
- [x] Logging utilities - Structured logging with file output support
- [ ] Experiment tracking - MLflow or Weights & Biases integration
- [ ] Caching layer - Cache LLM responses for reproducibility
- [ ] Parallel processing - Concurrent batch evaluation
- [ ] REST API - Serve model as web service

## Data Pipeline

- [x] Dataset fetcher - Download Fredholm-LLM dataset from Zenodo
- [x] Fredholm loader - Specialized loader with expression type inference
- [x] Generic data loader - JSON/JSONL file loading with filtering
- [x] Format converter - LaTeX and RPN conversion utilities
- [x] Data augmentation framework - Framework for dataset expansion
- [ ] Data validator - Validate equation syntax and solvability
- [ ] Special function augmentation - Add Bessel, Legendre equations
- [ ] No-solution cases - Generate equations without closed-form solutions
- [ ] Numeric ground truth - SciPy-based numerical solutions

## LLM Integration

- [x] OpenAI API support - GPT-4, GPT-3.5
- [x] OpenRouter API support - Claude, Llama, Mistral, etc.
- [x] Local model support - HuggingFace, vLLM placeholder
- [x] Batch generation - Process multiple equations efficiently
- [ ] Fine-tuning support - Train custom models (Phi, T5)
- [ ] Tool-assisted solving - LLM generates Python code for SymPy
- [ ] Iterative refinement - Multi-turn conversation for complex equations
- [ ] Confidence scoring - Estimate solution reliability

## Prompt Engineering

- [x] Basic/direct prompts - Simple equation-to-solution prompts
- [x] Chain-of-thought prompts - Step-by-step reasoning
- [x] Few-shot prompts - Include worked examples
- [ ] Approximation prompts - Request series/polynomial approximations
- [ ] Step-by-step breakdown - Decompose complex kernels
- [ ] Error correction prompts - Self-correction mechanisms
- [ ] Template optimization - A/B testing for prompt effectiveness

## Evaluation

- [x] Symbolic evaluation - SymPy-based expression comparison
- [x] Numeric evaluation - MAE, MSE, RMSE metrics
- [x] Postprocessing - Extract solutions from LLM responses
- [ ] BLEU / TeX-BLEU - Token-level similarity metrics
- [ ] Robustness testing - Prompt variation sensitivity
- [ ] Generalization testing - Performance on unseen function types
- [ ] Benchmark suite - Standardized evaluation dataset

## Output Formats

- [x] Symbolic output - Mathematical expressions
- [ ] Series expansion output - Taylor/Fourier series solutions
- [ ] Code output - Executable Python/SymPy code
- [ ] LaTeX export - Publication-ready formatted solutions
- [x] Interactive notebooks - Data exploration notebook with visualizations

## Data Exploration

- [x] Dataset overview tables - Column info, types, sample values
- [x] Expression length analysis - Min/max/mean/median for u, f, kernel
- [x] Expression type distribution - Polynomial, trig, hyperbolic, etc.
- [x] Numerical parameter analysis - λ, a, b statistics and histograms
- [x] Solution type classification - Trivial, constant, linear, polynomial, etc.
- [ ] Kernel complexity analysis - Nested function depth, term count
- [ ] Solvability assessment - Existence of closed-form solutions

## Configuration Presets

- [x] default.yaml - Standard balanced settings
- [x] development.yaml - Fast iteration for testing
- [x] production.yaml - Full evaluation settings
- [x] local.yaml - Self-hosted model settings
- [x] openrouter.yaml - OpenRouter with popular models
- [x] fine_tuning.yaml - Training configuration

## Testing & Documentation

- [x] Unit tests - 40+ tests covering core functionality
- [x] Pipeline diagram - Mermaid-based architecture visualization
- [x] README - Comprehensive project documentation
- [x] Config README - Configuration usage guide
- [x] Features tracking - This document
- [ ] API reference - Auto-generated from docstrings
- [ ] Tutorial notebooks - Step-by-step usage examples
- [ ] Contribution guide - How to add new features

---

## Progress Summary

| Category | Done | Total | Progress |
|----------|------|-------|----------|
| Core Infrastructure | 4 | 8 | 50% |
| Data Pipeline | 5 | 9 | 56% |
| LLM Integration | 4 | 8 | 50% |
| Prompt Engineering | 3 | 7 | 43% |
| Evaluation | 3 | 7 | 43% |
| Output Formats | 2 | 5 | 40% |
| Data Exploration | 5 | 7 | 71% |
| Configuration | 6 | 6 | 100% |
| Testing & Docs | 5 | 8 | 63% |
| **Total** | **37** | **65** | **57%** |

---

## Version History

### v0.1.0 (Current)

- Initial project scaffolding
- Dataset fetching from Zenodo
- Multi-provider LLM support (OpenAI, OpenRouter, Local)
- Basic prompt templates (direct, chain-of-thought, few-shot)
- Symbolic and numeric evaluation
- CLI with run, evaluate, convert, prompt, dataset commands
- 40 unit tests passing
