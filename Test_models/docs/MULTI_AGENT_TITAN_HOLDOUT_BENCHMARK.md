# Holdout / adversarial multi-agent Titan benchmark

This benchmark is a second, independent validation pass for the manager-led multi-agent Titan team.

It is intentionally different from `compare_multi_agent_titan_team.py`:

- different task names;
- different people;
- different modules;
- different distractors;
- additional update-like properties;
- additional noisy questions;
- additional consolidation and profile-summary checks.

The goal is to check whether the high score on the first large benchmark is robust or whether the implementation is overfitted to one test set.

## Command

```powershell
python Test_models/benchmark/compare_multi_agent_titan_holdout.py --save
```

For an even larger randomized run:

```powershell
python Test_models/benchmark/compare_multi_agent_titan_holdout.py --extra-random-cases 40 --seed 20260625 --save
```

For stricter role-selection validation:

```powershell
python Test_models/benchmark/compare_multi_agent_titan_holdout.py --strict-agents --save
```

## Expected interpretation

The target is not necessarily 100%.

A good result is that:

```text
multi_agent_titan_team > single_titan_agent > no_memory_agent
```

If the multi-agent team stays well above the single-agent baseline on this holdout benchmark, the result is much less likely to be a coincidence.

## What it tests

- project conventions;
- latest-value updates;
- identity collisions;
- noisy queries;
- forget + retention;
- consolidation;
- role routing;
- profile summaries;
- provider policy;
- UI and architecture policies.
