# Aedelon Titans Integration Plan for `mag_demo`

## Goal

The internship supervisor asked us to focus on this repository:

`https://github.com/Aedelon/titans-pytorch-mlx`

The important clarification is: **we do not need to replace Ollama or build a full Titans language model**. We only need the **memory component**.

Therefore, the next step is to isolate a Titans-style long-term memory module and integrate it into our existing external-memory framework.

## What matters in the Aedelon repo

Priority files:

- `src/titans/memory.py`: PyTorch Neural Long-Term Memory.
- `src/titans/config.py`: configuration for memory dimensions, learning rate, momentum and decay.
- `tests/test_memory.py`: sanity checks for the memory module.
- `src/titans/models.py`: useful to understand MAC/MAG/MAL/LMM, but not the first integration target.

Less important for our current Windows/PyTorch setup:

- `src/titans_mlx/`: MLX backend mainly useful for Apple Silicon.
- `scripts/pretrain*.py`: full model training scripts; too heavy for our current project.
- `triton_kernels.py`, Flash Attention, distributed training: not needed for the first memory-only prototype.

## Short-term vs long-term memory in our project

### Short-term memory

We keep our current explicit memory items:

- text fact;
- subject;
- property;
- active/inactive state;
- update handling;
- safe forgetting;
- Top-k retrieval metadata.

This part is precise and interpretable.

### Long-term memory

We add a Titans-style neural memory:

- key/value/query projections;
- associative loss `||M(k) - v||²`;
- test-time gradient update;
- surprise momentum;
- decay/forgetting;
- retrieve-only mode.

This part is closer to the Titans paper and to the Aedelon implementation.

## Added local file

This patch adds:

```text
Test_models/models/titans_ltm.py
```

It is a lightweight, standalone PyTorch module inspired by the Aedelon memory implementation. It does not depend on MLX, Triton, Flash Attention or the full Titans model.

## Added test

```text
Test_models/benchmark/test_titans_ltm.py
```

Run:

```bash
python Test_models/benchmark/test_titans_ltm.py
```

Expected output:

```text
Titans LTM focused checks: 5/5 passed
```

This checks that:

1. the memory forward pass works;
2. retrieve-only mode does not mutate memory;
3. update mode changes memory weights;
4. repeated test-time updates reduce associative loss;
5. the parameter count stays tiny.

## Next integration step

After the isolated test passes, we should connect `NeuralLongTermMemory` to `TitanExternalMemory` in `titan_model.py`:

1. Keep explicit text memories for exact retrieval and explainability.
2. For every stored memory unit, create a vector `x`.
3. Send `x` into `NeuralLongTermMemory` with `update=True`.
4. Store the returned `MemoryState` as the long-term memory state.
5. During retrieval, compute a query vector and call `retrieve(query, state)`.
6. Combine the neural score with the existing lexical/entity/property scores.

## Recommended architecture

```text
User fact/query
    ↓
Text parsing / atomization
    ↓
Short-term explicit memory
    - text
    - subject/property
    - active/inactive
    - update/forget
    ↓
Long-term Titans memory
    - M(k) -> v
    - surprise update
    - momentum
    - decay
    ↓
Hybrid retrieval Top-k
    ↓
Ollama LLM answer generation
```

## Why this is better aligned with the supervisor request

The previous Titan prototype was only Titan-inspired. This patch begins a cleaner move toward the actual Titans memory mechanism while keeping our current framework intact.

It also respects the constraint **under 2B parameters**, because the long-term memory module has only thousands or millions of parameters depending on `dim` and `num_memory_layers`, while the Ollama backbone remains configurable separately.
