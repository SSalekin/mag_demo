# Old Titan vs New Titan Comparison

## Goal

Compare the original legacy Titan prototype in `Legacy/titan_implementation.py` with the new Titan model in `Test_models/models/titan_model.py`.

The important point is that the legacy Titan implementation did **not** use `llama3` by default. It used:

```python
RESPONSE_MODEL_ID = os.getenv("RESPONSE_MODEL_ID", "gemma2:2b")
```

The new Titan should remain configurable through:

```powershell
$env:OLLAMA_MODEL="llama3.2:1b"
```

## Added files

- `Test_models/models/legacy_titan_model.py`
- `Test_models/benchmark/compare_titan_versions.py`
- `Test_models/docs/TITAN_OLD_VS_NEW_COMPARISON.md`

## Benchmark command

Fast deterministic comparison:

```powershell
$env:OLLAMA_MODEL="llama3.2:1b"
$env:RESPONSE_MODEL_ID="gemma2:2b"
python Test_models/benchmark/compare_titan_versions.py --scales small medium --new-ollama-model llama3.2:1b --legacy-response-model gemma2:2b
```

Small only, with detailed failures:

```powershell
python Test_models/benchmark/compare_titan_versions.py --scales small --debug
```

Optional slow neural legacy path:

```powershell
python Test_models/benchmark/compare_titan_versions.py --scales small --legacy-mode neural --legacy-train-steps 1 --legacy-max-decode-len 256
```

## How to interpret the comparison

The default `legacy-mode=facts` compares what the legacy memory keeps explicitly in its `facts` list. This is fast and deterministic.

The old demo originally used a char-level neural memory decoder plus an Agno/Ollama response layer. That path is available through `--legacy-mode neural`, but it is slow and often unstable for benchmark-style retrieval.

## Expected qualitative result

The new Titan should normally be stronger because it adds:

- structured memory items;
- subject/property extraction;
- active/inactive memory states;
- targeted update handling;
- targeted forgetting;
- Top-k retrieval;
- Aedelon-inspired neural long-term memory;
- direct memory answer fallback for small local LLMs.

The legacy model is expected to fail more often on:

- update cases;
- forget cases;
- multi-profile separation;
- distractor resistance;
- profile-specific retrieval.
