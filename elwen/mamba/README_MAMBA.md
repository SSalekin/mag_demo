# Mamba External Memory Agent

## Location

This folder contains the Mamba implementation:

```text
mag_demo/
└── elwen/
    └── mamba/
        ├── mamba_implementation.py
        ├── benchmark.py
        └── README_MAMBA.md
```

## Goal

This implementation is part of the project:

**R&D on Memory Neural Network for AI Agent**

The current implementation uses:

- **Mamba** as an external memory encoder.
- **Ollama/Gemma** as the LLM brain that generates the final response.
- A hybrid retrieval system combining:
  - Mamba semantic similarity,
  - lexical overlap,
  - entity/name matching,
  - property matching,
  - recency.

## Main idea

This implementation is not a fully fine-tuned Mamba neural memory network. It is a practical prototype of:

**Mamba-backed external vector memory for an LLM agent.**

The workflow is:

```text
User statement
→ atomization into memory facts
→ Mamba embedding
→ storage with metadata
→ hybrid retrieval
→ retrieved memories sent to Gemma/Ollama
→ final answer
```

## Run the chatbot

From the project root:

```powershell
python elwen\mamba\mamba_implementation.py
```

## Useful commands inside the chatbot

```text
/help
/debug on
/debug off
/store <text>
/ask <question>
/search <query>
/forget <query>
/memories
/memories all
/reindex
/save
/load
/reset
/exit
```

## Run the benchmark

From the project root:

```powershell
python elwen\mamba\benchmark.py
```

To test only retrieval without calling Ollama:

```powershell
python elwen\mamba\benchmark.py --retrieval-only
```

## Expected current results

On the current benchmark, this implementation should normally reach:

```text
Retrieval-only benchmark: 10/10
Full benchmark with Ollama/Gemma: 10/10
```

Results may vary slightly depending on the local Ollama model, machine performance, and memory file state.

## Local memory files

The runtime memory is saved next to the Mamba files:

```text
elwen/mamba/mamba_memory_store.pt
elwen/mamba/benchmark_memory_store.pt
```

These files should normally **not** be pushed to GitHub because they contain local test memories.

Recommended `.gitignore` entries:

```gitignore
# Local memory files
**/mamba_memory_store.pt
**/benchmark_memory_store.pt

# Python cache
__pycache__/
*.pyc
```

## Files

```text
mamba_implementation.py  Main chatbot and Mamba memory implementation
benchmark.py             Automatic benchmark for read/write/update/forget tests
README_MAMBA.md          Documentation for this Mamba implementation
```

## Next step

The next model can follow the same structure:

```text
elwen/
├── mamba/
├── titan/
└── transformer/
```

Each model should have its own implementation, benchmark, and README so that results can be compared cleanly.


## v8 fixes

- Fixed full-name possessive updates such as `Sarah Nguyen's favorite color is now black`.
- Added subject-aware entity matching.
- Added stricter filtering for single-value properties such as secret codes, favorite colors, languages, and office rooms.
