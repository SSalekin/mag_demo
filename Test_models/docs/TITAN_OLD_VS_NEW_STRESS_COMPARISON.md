# Old Titan vs New Titan — Stress Comparison

This benchmark compares the old `Legacy/titan_implementation.py` prototype against the new `Test_models/models/titan_model.py` on the same stress cases used for the Mamba/Titan stress benchmark.

## Why this benchmark exists

The standard benchmark checks basic operations such as simple recall, update, forget, distractor resistance and multi-profile separation.

The stress benchmark is harder. It checks:

- paraphrased questions;
- noisy questions;
- multiple successive updates;
- identity collisions;
- hard distractors;
- forgetting near similar memories;
- retention after forgetting;
- mixed French/English queries;
- profile summary.

## Important legacy note

The old Titan did not use `llama3` by default. It used:

```python
RESPONSE_MODEL_ID = os.getenv("RESPONSE_MODEL_ID", "gemma2:2b")
```

For memory-only comparison, the stress script defaults to:

```text
--legacy-mode facts
```

This mode compares the memory content directly and avoids depending on the legacy response model.

## Commands

Quick smoke test:

```powershell
python -m compileall Test_models/benchmark/compare_titan_versions_stress.py
python Test_models/benchmark/compare_titan_versions_stress.py --scales small --debug
```

Main stress comparison:

```powershell
$env:OLLAMA_MODEL="llama3.2:1b"
$env:RESPONSE_MODEL_ID="gemma2:2b"
python Test_models/benchmark/compare_titan_versions_stress.py --scales small medium --new-ollama-model llama3.2:1b --legacy-response-model gemma2:2b
```

Optional large test:

```powershell
python Test_models/benchmark/compare_titan_versions_stress.py --scales large --large-profiles 120 --large-distractors 300 --new-ollama-model llama3.2:1b
```

## Expected interpretation

The new Titan is expected to be stronger on:

- identity collision;
- multiple updates;
- targeted forget;
- distractor resistance;
- mixed-language queries;
- profile-level retrieval.

The old Titan may appear faster in `facts` mode because it has almost no structured update/retrieval logic. Speed should therefore be interpreted with caution: the main comparison is robustness and correctness, not only raw latency.
