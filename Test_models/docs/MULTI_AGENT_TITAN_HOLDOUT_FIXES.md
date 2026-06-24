# Multi-Agent Titan Holdout Fixes

This patch targets the remaining weaknesses exposed by the holdout/adversarial benchmark.

## Fixes

1. **Unseen latest-value properties**
   - Added manager-level property recognition for `code_formatter` and `api_auth_method`.
   - The team now keeps only the most recent active value for these properties, just like it already did for LLM provider, testing framework, UI style and other update-heavy project settings.

2. **Forget + retention tasks**
   - If a task explicitly says to forget a temporary secret code before answering, the manager now applies that forget operation before retrieval.
   - The final retrieval then targets the remaining retention/project rule instead of leaking the forgotten secret.

3. **Consolidation command filtering**
   - Operational commands such as `Please consolidate the active project rules...` are no longer treated as useful memories.
   - The system retrieves the real consolidated rules instead of the instruction to consolidate.

4. **Policy summary routing**
   - Team-policy tasks now route more reliably to the evaluator.
   - Query expansion now includes `limitations`, `next steps`, `policy`, and consolidation-related terms.

## Tests

Run:

```powershell
python -m compileall Test_models/agents/multi_agent_titan_team.py Test_models/benchmark/test_multi_agent_holdout_improvements.py
python Test_models/benchmark/test_multi_agent_holdout_improvements.py
python Test_models/benchmark/compare_multi_agent_titan_holdout.py --save
```

Expected direction: the multi-agent Titan team should remain clearly above the single-agent baseline and improve on the previous holdout score of 69/91 (75.8%).
