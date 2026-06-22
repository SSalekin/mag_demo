# Stress Benchmark — Mamba vs Titan

This benchmark is harder than the baseline comparison.

It tests:
- Top-1 / Top-3 / Top-5 retrieval accuracy;
- paraphrased questions;
- noisy questions;
- multiple successive updates;
- same first-name identity collisions;
- hard distractors;
- targeted forgetting;
- retention after forgetting;
- mixed French/English questions;
- profile summary retrieval.

## Install

Put the Python file here:

```text
elwen/comparison/compare_memory_models_stress.py
```

## Commands

Small + medium:

```powershell
python elwen\comparison\compare_memory_models_stress.py --scales small medium
```

Large:

```powershell
python elwen\comparison\compare_memory_models_stress.py --scales large
```

Large but lighter:

```powershell
python elwen\comparison\compare_memory_models_stress.py --scales large --large-profiles 120 --large-distractors 300
```

Full LLM mode:

```powershell
python elwen\comparison\compare_memory_models_stress.py --scales small --full-llm
```

## Interpretation

Use retrieval-only mode to compare the memory backends.

Important metrics:
- Top-1: the first retrieved memory is already enough;
- Top-3: the answer is in the first three retrieved memories;
- Top-5: the answer is somewhere in the context sent to the LLM.
