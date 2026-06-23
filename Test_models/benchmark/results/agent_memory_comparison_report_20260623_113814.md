# Agent memory comparison

- **no_memory_agent**: 0/5 (0.0%)
- **titan_memory_agent**: 4/5 (80.0%)

## Cases
- no_memory_agent / remember_secret_code: FAIL — `I do not know because no long-term memory is available.`
- no_memory_agent / update_secret_code: FAIL — `I do not know because no long-term memory is available.`
- no_memory_agent / identity_collision_sarah_lucas: FAIL — `I do not know because no long-term memory is available.`
- no_memory_agent / forget_one_profile_keep_other: FAIL — `I do not know because no long-term memory is available.`
- no_memory_agent / consolidated_recall: FAIL — `I do not know because no long-term memory is available.`
- titan_memory_agent / remember_secret_code: PASS — `Lucas Martin's secret code is 8392.`
- titan_memory_agent / update_secret_code: PASS — `Lucas Martin's secret code is 1245.`
- titan_memory_agent / identity_collision_sarah_lucas: PASS — `Sarah Martin's favorite color is green.`
- titan_memory_agent / forget_one_profile_keep_other: PASS — `Lucas Martin's favorite color is violet.`
- titan_memory_agent / consolidated_recall: FAIL — `The project convention is to use type hints in new Python files.`