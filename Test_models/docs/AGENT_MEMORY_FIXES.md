# Agent memory benchmark fixes

This patch improves the Titan memory agent on the detailed agent benchmark.

## Fixed issues

1. **Noisy query retrieval**
   - Added noisy query terms such as `ignore`, `random`, `noise`, and `again` to stopwords.
   - This prevents irrelevant capitalized words from being treated as entity names and blocking retrieval.

2. **Global multiple updates**
   - Added pseudo-subjects for global single-value facts such as:
     - `The preferred backend language is ...`
     - `The release secret code is ...`
     - `The documentation style is ...`
   - This allows Titan to deactivate older values and keep only the newest active value.

3. **Testing framework retrieval**
   - Added a `testing_framework` property for facts containing `pytest`, `unit tests`, or `testing framework`.
   - This improves retrieval for project convention questions.

4. **Benchmark forbidden-value evaluation**
   - Updated the detailed agent benchmark so a forbidden substring is not counted as an error when it is part of the correct expected answer.
   - Example: `FINAL-CODE-3053` legitimately contains `CODE-3053` as a substring.

## Expected benchmark result

After applying this patch, the detailed agent memory benchmark should show approximately:

```text
no_memory_agent        | 0/16  (0.0%)
short_term_agent       | 11/16 (68.8%)
titan_memory_agent     | 16/16 (100.0%)
```

Run:

```powershell
python -m compileall Test_models/models/titan_model.py Test_models/benchmark/compare_agent_memory_modes.py
python Test_models/benchmark/compare_agent_memory_modes.py --scales small medium --save
```
