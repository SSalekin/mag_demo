# Multi-Agent Titan Team: detailed benchmark

This benchmark is the larger stress-oriented version of the first multi-agent
prototype comparison.

It compares three modes:

1. `no_memory_agent`: no shared memory.
2. `single_titan_agent`: Titan memory retrieval only, no team orchestration.
3. `multi_agent_titan_team`: manager-led multi-agent team with shared Titan memory.

The goal is to answer the research direction:

> How can we include the Titan model in an AI agent, and which agent architecture
> is best?

The benchmark checks both memory usage and orchestration:

- project conventions;
- DevWeb planning;
- DevSoft planning;
- DevOps/provider configuration;
- tester/evaluator roles;
- multi-session recall;
- noisy queries;
- hard distractors;
- identity collisions;
- multiple updates;
- consolidation;
- role selection.

## Commands

```powershell
python -m compileall Test_models/agents/multi_agent_titan_team.py Test_models/benchmark/compare_multi_agent_titan_team.py
python Test_models/benchmark/compare_multi_agent_titan_team.py --scales small medium stress --save
```

Show all answers:

```powershell
python Test_models/benchmark/compare_multi_agent_titan_team.py --scales small medium stress --show-answers
```

Strict role checking:

```powershell
python Test_models/benchmark/compare_multi_agent_titan_team.py --scales small medium stress --strict-agents --save
```

## Interpretation

The expected trend is not that every multi-agent answer is perfect. The expected
trend is that:

`multi_agent_titan_team > single_titan_agent > no_memory_agent`

on tasks where both long-term memory and specialist role coordination matter.
