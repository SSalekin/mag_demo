# Detailed agent memory benchmark

This benchmark answers the next project question:

> How can we include Titan in an AI agent, and is it useful compared with simpler agent memory setups?

The benchmark is located at:

```text
Test_models/benchmark/compare_agent_memory_modes.py
```

## Compared agents

The benchmark compares three modes:

| Agent | Meaning |
|---|---|
| `no_memory_agent` | Stateless baseline. It does not persist anything. |
| `short_term_agent` | Working-memory baseline. It can use facts during the current session, but loses them after restart. |
| `titan_memory_agent` | Agent using the new Titan memory adapter as long-term memory. |

## Tested capabilities

The detailed benchmark covers:

- simple recall;
- value update;
- identity collision;
- targeted forgetting;
- paraphrased retrieval;
- noisy retrieval;
- mixed-language query;
- multiple updates;
- hard distractors;
- multi-session recall;
- consolidation;
- profile summary.

## Commands

Run the full benchmark:

```powershell
python Test_models/benchmark/compare_agent_memory_modes.py --scales small medium --save
```

Show answers for debugging:

```powershell
python Test_models/benchmark/compare_agent_memory_modes.py --scales small medium --show-answers
```

Compare only short-term vs Titan:

```powershell
python Test_models/benchmark/compare_agent_memory_modes.py --agents short_term titan --scales small medium --save
```

## Interpretation

This benchmark is deterministic and does not require Agno or Ollama. It tests the memory layer directly so we can isolate the benefit of Titan before evaluating complete agent behavior.

Expected interpretation:

- `no_memory_agent` should fail memory-dependent tasks.
- `short_term_agent` should pass simple current-session tasks but fail after simulated restart and consolidation-dependent cases.
- `titan_memory_agent` should perform best on long-term memory, update, forget and multi-session tasks.
