# Titan Architecture Proposal — Lightweight External Memory for LLM Agents

## 1. Objective

The next step of the project is no longer only to benchmark existing memory backends. The goal is to define a cleaner Titan-based memory architecture that can run locally on a normal PC and stay below a **2B-parameter total budget**.

The proposed direction is:

- keep the LLM backbone configurable and small by default;
- use a separate Titan-inspired neural memory module;
- train only the memory module at test time;
- keep the existing `Test_models/` API compatible with `main.py` and the benchmark scripts;
- keep Mamba, Transformer, LSTM, GRU and GNN as baselines.

## 2. Constraints

| Constraint | Design decision |
|---|---|
| `< 2B` parameters | Default Ollama backend is configurable and set to a small model name (`llama3.2:1b` by default). The Titan memory module is tiny compared with the LLM. |
| Normal PC | Memory defaults run on CPU if CUDA is unavailable. No heavy extra dependency was added. |
| Keep benchmarks compatible | `TitanModel` still exposes `add_user_message`, `generate_response_stream`, `clear_memory`, `save_memory`, `load_memory`, `get_active_tokens_count`, `get_dropped_tokens_count`. |
| Honest comparison | Existing benchmark files are not modified to favor Titan. A separate Titan-focused test was added. |
| Configurable | Main parameters are centralized in `models/titan_config.py` and can be overridden with environment variables. |

## 3. What the current Titan version did

The previous `Test_models/models/titan_model.py` was already Titan-inspired:

1. user facts were split into atomic memories;
2. each memory was encoded as hashed text vectors;
3. a small MLP learned an associative mapping `M(key) -> value` at test time;
4. retrieval used a hybrid score combining neural readout, key similarity, lexical overlap, entity matching, property matching and recency;
5. active/inactive memories were used for update and forget operations.

This was a good prototype, but it remained closer to a hybrid symbolic/vector retriever than to a clean Titan-style memory architecture.

## 4. What was missing compared with Titans

The Titans paper introduces a neural long-term memory that learns to memorize at test time. The key elements are:

- associative memory objective: `loss = ||M(k) - v||²`;
- test-time memory update;
- surprise-based memorization;
- momentum over surprise;
- adaptive forgetting / weight decay;
- separation between short-term context and long-term neural memory;
- possible integration variants such as memory as context, memory as layer or memory as gated branch.

The previous project version had the associative objective and test-time updates, but only a simplified surprise mechanism. It also had a major practical weakness: subject resolution could collapse two people with the same first name, for example mapping `Sarah Martin` to an existing `Sarah Nguyen` subject. This created identity collisions.

## 5. Proposed architecture: Titan-Lite External Memory

### 5.1 High-level design

```text
User message
    |
    |-- Intent classifier: store / ask / forget / chat
    |
    |-- If store:
    |      normalize instruction wrapper
    |      atomize facts
    |      extract subject + property
    |      encode key/value
    |      compute surprise
    |      update neural long-term memory M(key)->value
    |      mark superseded memories inactive for single-value properties
    |
    |-- If ask:
    |      extract query subject + query property
    |      retrieve Top-k with neural + lexical + metadata score
    |      send retrieved memories to configurable Ollama LLM
    |
    |-- If forget:
           retrieve best forget candidate non-interactively for benchmarks/main.py
           mark memory inactive
```

### 5.2 Modules

| File | Role |
|---|---|
| `models/titan_config.py` | Central configuration: dimensions, top-k, learning rate, surprise momentum, max memory items, Ollama model. |
| `models/titan_model.py` | Titan memory implementation and `TitanModel` adapter kept compatible with the old framework. |
| `benchmark/test_titan_architecture.py` | Focused retrieval-only test for identity collisions, updates, forgetting, paraphrases and noisy memory. |

### 5.3 Parameter budget

Default Titan memory shape:

- `d_model = 128`
- `hidden_dim = 256`
- 3 frozen projections: `W_K`, `W_V`, `W_Q`
- 1 trainable MLP: `Linear(d_model, hidden_dim) -> GELU -> Linear(hidden_dim, d_model)`

Approximate memory module parameters:

```text
3 * d_model * d_model
+ d_model * hidden_dim + hidden_dim
+ hidden_dim * d_model + d_model

For d_model=128, hidden_dim=256:
≈ 115,072 parameters
```

This is negligible compared with the LLM. With a `1B` local Ollama model, the estimated total remains far below `2B` parameters:

```text
1,000,000,000 + 115,072 ≈ 1.0001B parameters
```

The project should avoid hardcoding `llama3` because many `llama3` tags are 8B and exceed the target scope. The default is now configurable through `OLLAMA_MODEL`, with `llama3.2:1b` as a small default name.

## 6. Memory storage

Each memory item stores:

- `id`
- original atomic `text`
- `key` vector
- `value` vector
- `subject`, for example `sarah nguyen`
- `property`, for example `favorite_color`, `location`, `secret_code`
- `active` / inactive flag
- `surprise`
- timestamps and access metadata
- deactivation reason when superseded or forgotten

This keeps the memory interpretable for debugging while still using a Titan-style neural memory module for associative learning.

## 7. Test-time update

For a new fact:

1. compute `k = W_K(x)` and `v = W_V(x)`;
2. read current prediction `M(k)`;
3. compute surprise with `MSE(M(k), v)`;
4. update a momentum surprise state;
5. train the memory MLP for a small adaptive number of steps;
6. replay a few recent active associations to reduce catastrophic overwriting.

The implemented update is intentionally lightweight. It is not a full end-to-end Titans reproduction, but it is closer to the paper than a pure vector retriever because the memory MLP is updated at test time.

## 8. Forgetting and decay

There are two forgetting levels:

1. **Semantic forgetting**: exact memories can be marked inactive through the forget API.
2. **Neural decay**: the memory optimizer uses configurable `weight_decay`, plus a small surprise-dependent decay factor.

Single-value properties such as `favorite_color`, `secret_code`, `office_room`, `location`, `password`, `student_id` automatically supersede previous active memories for the same subject/property.

## 9. Retrieval

Retrieval uses a hybrid Top-k score:

- neural readout similarity: `M(q)` vs memory value;
- key similarity: query key vs memory key;
- lexical overlap;
- entity match;
- strict subject match;
- property match;
- recency.

The important V2 change is stricter subject handling. If the query asks for a precise full-name subject and a matching subject exists in memory, other subjects with the same first name are filtered out for property-specific queries.

Example:

```text
Query: What is Sarah Nguyen's favorite color?
Stored:
- Sarah Nguyen's favorite color is purple.
- Sarah Martin's favorite color is green.
- Sarah Wilson's favorite color is orange.

Expected retrieval:
- Sarah Nguyen / favorite_color only
```

## 10. Main differences versus previous Titan prototype

| Aspect | Previous Titan | Titan-Lite V2 |
|---|---|---|
| Config | Mostly hardcoded values | `TitanMemoryConfig` + environment overrides |
| Ollama model | `llama3` was used in main config | configurable, default small model name |
| Subject resolution | Could collapse same-first-name identities | full names are never collapsed |
| Benchmark wrapper | `Please remember this information:` could pollute stored facts | instruction wrapper is stripped before storage |
| Surprise | basic surprise-to-steps rule | surprise + momentum + adaptive decay |
| Forget API | could prompt interactively | `TitanModel` API path is non-interactive for benchmarks |
| Tests | only global benchmarks | added Titan-specific focused test |

## 11. Commands

From the `mag_demo/` root:

```bash
# focused Titan memory test, no Ollama required
python Test_models/benchmark/test_titan_architecture.py

# standard clean benchmark, requires Ollama running
OLLAMA_MODEL=llama3.2:1b python Test_models/benchmark/compare_memory_models.py --models titan mamba --scales small medium

# stress benchmark, requires Ollama running
OLLAMA_MODEL=llama3.2:1b python Test_models/benchmark/compare_memory_models_stress.py --models titan mamba --scales small medium

# run the terminal application with a small model
OLLAMA_MODEL=llama3.2:1b python Test_models/main.py
```

Optional Titan overrides:

```bash
TITAN_D_MODEL=64 \
TITAN_HIDDEN_DIM=128 \
TITAN_TOP_K=5 \
TITAN_LR=0.01 \
TITAN_MAX_ITEMS=8192 \
OLLAMA_MODEL=llama3.2:1b \
python Test_models/main.py
```

## 12. Validation status

The focused Titan test passes locally in retrieval-only mode:

```text
Titan V2 focused checks: 10/10 passed
```

This validates the memory module without relying on Ollama generation. The full clean and stress benchmarks still need to be run in an environment where Ollama is installed and the selected local model has been pulled.

## 13. Limitations

- This is still a Titan-inspired external memory, not a full end-to-end trained Titans architecture.
- Text encoding currently uses deterministic hashed vectors, not a trained encoder.
- The neural memory MLP is small; it is appropriate for a local prototype, but not enough for real long-context language modeling alone.
- Retrieval still uses symbolic safeguards because the project needs reliable profile-level memory operations.
- Full LLM benchmark scores can vary with the chosen Ollama model.

## 14. Recommended next steps

1. Run the standard and stress benchmarks with `OLLAMA_MODEL=llama3.2:1b` or another local model below 2B.
2. Compare Titan V2 against the previous Titan checkpoint and Mamba on the same small/medium benchmark settings.
3. If Titan V2 keeps the speed advantage, run the large stress benchmark.
4. Replace hashed vectors with a lightweight local embedding encoder only if it does not break the PC/2B constraint.
5. Explore a gated branch version later: `final_context = gate * Titan_memory + (1-gate) * retrieved_text_context`.
