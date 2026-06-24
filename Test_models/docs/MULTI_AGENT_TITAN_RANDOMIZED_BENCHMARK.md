# Randomized adversarial benchmark for the Titan multi-agent team

This benchmark is intended to check that the high score on the previous holdout is not only a benchmark-specific effect.

It generates fresh cases at runtime from a seed and compares:

- `no_memory_agent`
- `single_titan_agent`
- `multi_agent_titan_team`

The benchmark includes:

- project conventions with distractors;
- latest-value updates;
- identity collisions;
- noisy queries;
- forget + retention;
- consolidation-style project rules;
- role routing tasks.

## Recommended commands

```powershell
python -m compileall Test_models/benchmark/compare_multi_agent_titan_randomized.py
python Test_models/benchmark/compare_multi_agent_titan_randomized.py --num-cases 150 --seed 20260624 --save
```

Run several seeds:

```powershell
python Test_models/benchmark/compare_multi_agent_titan_randomized.py --num-cases 150 --seed 101 --save
python Test_models/benchmark/compare_multi_agent_titan_randomized.py --num-cases 150 --seed 202 --save
python Test_models/benchmark/compare_multi_agent_titan_randomized.py --num-cases 150 --seed 303 --save
```

Stricter role-selection mode:

```powershell
python Test_models/benchmark/compare_multi_agent_titan_randomized.py --num-cases 150 --seed 404 --strict-agents --save
```

## Interpretation

A single perfect score is not enough. The useful signal is whether:

```text
multi_agent_titan_team > single_titan_agent > no_memory_agent
```

across multiple seeds.

If performance drops on new seeds, analyze the failure categories instead of tuning blindly.
