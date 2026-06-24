# Large multi-agent Titan benchmark

This patch expands the multi-agent benchmark from 24 cases to 80 cases.

It keeps the same three agents/modes:

- `no_memory_agent`
- `single_titan_agent`
- `multi_agent_titan_team`

It adds a new `large` scale with many additional scenarios:

- project conventions with distractors
- role routing
- identity collisions
- multiple updates
- targeted forgetting with retained project rules
- consolidation after many rules
- noisy queries
- profile summaries

## Commands

Compile:

```powershell
python -m compileall Test_models/benchmark/compare_multi_agent_titan_team.py
```

Run all 80 cases:

```powershell
python Test_models/benchmark/compare_multi_agent_titan_team.py --scales small medium stress large --save
```

Show answers:

```powershell
python Test_models/benchmark/compare_multi_agent_titan_team.py --scales small medium stress large --show-answers
```

Strictly require expected specialist routing:

```powershell
python Test_models/benchmark/compare_multi_agent_titan_team.py --scales small medium stress large --strict-agents --save
```

Run only large cases:

```powershell
python Test_models/benchmark/compare_multi_agent_titan_team.py --scales large --save
```

## Interpretation

The goal is not to force a perfect 100% score.  The goal is to stress the team
architecture with a larger number of cases and identify real failure categories.
The most important comparison is whether `multi_agent_titan_team` remains above
`single_titan_agent`, especially on multi-step project tasks.
