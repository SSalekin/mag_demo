# Mamba vs Titan Comparison v2

This folder contains the corrected comparison benchmark for:

- `elwen/mamba/mamba_implementation.py`
- `elwen/titan/titan_implementation.py`

## Why v2?

The first benchmark produced several false negatives. The v2 benchmark fixes:

- `NEW-CODE-1022` being incorrectly detected as containing old `CODE-1022`;
- `Java` being incorrectly detected inside `JavaScript`;
- retrieval-only forgetting tests expecting a generated sentence like `I do not know`;
- synthetic facts using first names only, which created artificial ambiguity at larger scales.

## Recommended commands

From the project root:

```powershell
python elwen\comparison\compare_memory_models.py --scales small medium
```

Large dataset:

```powershell
python elwen\comparison\compare_memory_models.py --scales small medium large
```

Full LLM mode with Ollama/Gemma:

```powershell
python elwen\comparison\compare_memory_models.py --scales small --full-llm
```

## Output

The script creates:

```text
elwen/comparison/results/comparison_v2_YYYYMMDD_HHMMSS.csv
elwen/comparison/results/comparison_v2_report_YYYYMMDD_HHMMSS.md
```

## Interpretation

Use retrieval-only mode for memory-backend comparison.
Use `--full-llm` for final chatbot-answer comparison.


## Important model fixes to apply

The comparison v2 exposed two real model issues:

1. Full-name possessives were not normalized correctly:
   - `Sarah Nguyen's favorite color is yellow`
   - `Sarah Nguyen's favorite color is now black`
   were not treated as the same subject because the old parser only supported one-word possessives.

2. Partial identity matches were too permissive for single-value properties:
   - a question about `CarlosBeta Wilson's secret code` could retrieve
     `MarcoAlpha Wilson's secret code` because the last name matched.

Use the fixed Mamba and Titan files before rerunning this benchmark.
