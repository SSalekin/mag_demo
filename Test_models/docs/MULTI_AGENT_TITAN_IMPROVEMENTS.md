# Multi-Agent Titan Team Improvements

This patch improves the manager-led multi-agent Titan prototype after the large benchmark exposed three weaknesses:

1. **Distractor leakage**: noise words and distractor memories could appear in the final answer.
2. **Outdated values after updates**: old values such as Gemini/Claude, unittest/nose, or plain text could appear next to the current value.
3. **Weak role routing in large tasks**: the manager sometimes failed to select DevWeb/DevSoft/Evaluator when the task wording was vague but the retrieved memory clearly required them.

## Main changes

### 1. Task-aware memory filtering

`MultiAgentTitanTeam` now builds a curated memory context instead of directly printing raw Top-k retrieval.

It filters out:

- explicit distractors;
- noisy benchmark memories;
- multi-agent audit decisions when factual memories are available;
- nearby identity-collision memories that are not about the requested task.

### 2. Update-aware memory collapse

For update-heavy project properties, the team layer keeps only the most recent active value:

- LLM provider;
- testing framework;
- backend language;
- UI style;
- memory backend;
- deployment shell;
- reporting rule;
- benchmark output format.

This prevents the final answer from showing old values after a newer value has been stored.

### 3. Cleaner final answers

The manager no longer repeats noisy task tokens such as `banana`, `coffee`, `hotel`, or `router` in the final response.

### 4. Better role selection

The manager now routes from both:

- the user task;
- the selected Titan memory records.

For example, if the task says only “billing work” but Titan memory says “FastAPI, pytest, type hints”, the manager can still select DevWeb, DevSoft, Tester, and Evaluator.

## New regression test

Run:

```powershell
python Test_models/benchmark/test_multi_agent_routing_improvements.py
```

Expected result:

```text
Multi-agent routing improvement checks: 4/4 passed
```

## Recommended benchmark command

```powershell
python Test_models/benchmark/compare_multi_agent_titan_team.py --scales small medium stress large --save
```

A realistic improvement target is to raise `multi_agent_titan_team` from about 53.8% toward 70%+ on the 78-case benchmark.
