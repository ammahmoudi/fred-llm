# LLM Module - Model Integration & Evaluation

This module handles LLM API integration and solution evaluation for Fredholm integral equations.

## 📦 Module Structure

```
src/llm/
├── model_runner.py      # LLM API integrations (OpenAI, OpenRouter, local)
├── evaluate.py          # Solution evaluation (symbolic & numeric)
├── postprocess.py       # Extract solutions from LLM responses
└── README.md            # This file
```

## 🔗 Related Modules

- **Prompt Generation**: [src/prompts/](../prompts/README.md) - Generate prompts in various styles
- **Data Pipeline**: [src/data/](../data/) - Dataset loading and formatting

---

## 🎯 Model Integration

The model_runner.py provides interfaces for:

### Supported Providers

1. **OpenAI** (GPT-4, GPT-3.5-turbo)
   ```python
   from src.llm import ModelRunner
   
   runner = ModelRunner(
       provider="openai",
       model_name="gpt-4",
       temperature=0.1
   )
   response = runner.generate(prompt)
   ```

2. **OpenRouter** (Claude, Llama, Mistral, etc.)
   ```python
   runner = ModelRunner(
       provider="openrouter",
       model_name="anthropic/claude-3.5-sonnet",
       temperature=0.1
   )
   ```

3. **Local Models** (HuggingFace, vLLM)
   ```python
   runner = ModelRunner(
       provider="local",
       model_path="meta-llama/Llama-3-8b"
   )
   ```

---

## 📊 Evaluation System

After generating LLM responses, evaluate them:

```python
from src.llm import evaluate_solutions

results = evaluate_solutions(
    predictions_file="responses.jsonl",
    mode="both"  # symbolic + numeric
)

print(f"Exact match: {results['exact_matches']}/{results['total']}")
print(f"Mean absolute error: {results['mae']:.4f}")

# When using the adaptive pipeline, evaluated predictions are saved as:
# predictions_evaluated_<timestamp>.jsonl
# Each entry includes an `evaluation` field with series term metrics.
```

---

## 🚀 End-to-End Workflow

```python
from src.llm import create_processor, ModelRunner, evaluate_solutions

# 1. Generate prompts
processor = create_processor(style="chain-of-thought")
prompt_file = processor.process_dataset("data/processed/test_infix.csv")

# 2. Run inference
runner = ModelRunner(provider="openai", model_name="gpt-4")
responses = []
with open(prompt_file) as f:
    for line in f:
        data = json.loads(line)
        response = runner.generate(data["prompt"])
        responses.append({
            "equation_id": data["equation_id"],
            "prediction": response,
            "ground_truth": data["ground_truth"]
        })

# 3. Evaluate
results = evaluate_solutions(responses, mode="both")
print(results)
```

---

## 🎯 Best Practices

### Prompt Generation

1. **Use chain-of-thought for complex equations** - Better reasoning
2. **Use few-shot for consistent patterns** - Improves accuracy
3. **Include ground truth during development** - Enable evaluation
4. **Exclude ground truth for production** - Real inference scenarios
5. **Batch process for efficiency** - Use `process_multiple_datasets()`

### Performance

1. **Stream JSONL files** - Don't load entire file into memory
2. **Use progress bars** - Track long-running operations
3. **Cache generated prompts** - Reuse across experiments
4. **Validate CSV schema early** - Fail fast on missing columns

### Testing

1. **Test all prompt styles** - Ensure consistent behavior
2. **Validate JSONL format** - Each line must be valid JSON
3. **Check metadata preservation** - All 20 columns maintained
4. **Verify format detection** - Auto-detection works correctly

---

## 📖 References

- [Fredholm Integral Equations](https://en.wikipedia.org/wiki/Fredholm_integral_equation)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Few-Shot Learning](https://arxiv.org/abs/2005.14165)
- [JSONL Format Specification](https://jsonlines.org/)

---

## 🆘 Troubleshooting

### Issue: Missing columns error

**Solution:** Ensure CSV has all required columns: u, f, kernel, lambda_val, a, b

### Issue: Format not detected

**Solution:** Use explicit `--format` flag or rename files (train_latex.csv, train_rpn.csv)

### Issue: Out of memory on large datasets

**Solution:** Process in smaller batches or use streaming JSONL reader

### Issue: Prompts too long for model context

**Solution:** Use basic style or reduce num_examples in few-shot style

---

## 📝 Contributing

When adding new features:

1. Update this README with usage examples
2. Add unit tests in `tests/test_prompt_generation.py`
3. Update `docs/FEATURES.md` checklist
4. Follow the existing code style (dataclasses, type hints, docstrings)
5. Use rich console for CLI output

---

## 📜 License

Part of the Fred-LLM project. See main repository LICENSE file.
