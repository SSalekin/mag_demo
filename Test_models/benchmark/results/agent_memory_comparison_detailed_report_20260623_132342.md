# Detailed agent memory comparison

## Global results

| Agent | Score | Avg store | Avg query |
|---|---:|---:|---:|
| no_memory_agent | 0/16 (0.0%) | 0.000s | 0.000s |
| short_term_agent | 10/16 (62.5%) | 0.000s | 0.000s |
| titan_memory_agent | 12/16 (75.0%) | 0.088s | 0.004s |

## Results by category

| Agent | Category | Score |
|---|---|---:|
| no_memory_agent | consolidation | 0/2 (0.0%) |
| no_memory_agent | forget | 0/2 (0.0%) |
| no_memory_agent | hard_distractors | 0/1 (0.0%) |
| no_memory_agent | identity_collision | 0/1 (0.0%) |
| no_memory_agent | mixed_language_query | 0/1 (0.0%) |
| no_memory_agent | multi_session | 0/1 (0.0%) |
| no_memory_agent | multiple_updates | 0/2 (0.0%) |
| no_memory_agent | noisy_query | 0/1 (0.0%) |
| no_memory_agent | paraphrase_recall | 0/1 (0.0%) |
| no_memory_agent | profile_summary | 0/2 (0.0%) |
| no_memory_agent | simple_recall | 0/1 (0.0%) |
| no_memory_agent | update | 0/1 (0.0%) |
| short_term_agent | consolidation | 0/2 (0.0%) |
| short_term_agent | forget | 2/2 (100.0%) |
| short_term_agent | hard_distractors | 1/1 (100.0%) |
| short_term_agent | identity_collision | 1/1 (100.0%) |
| short_term_agent | mixed_language_query | 1/1 (100.0%) |
| short_term_agent | multi_session | 0/1 (0.0%) |
| short_term_agent | multiple_updates | 1/2 (50.0%) |
| short_term_agent | noisy_query | 1/1 (100.0%) |
| short_term_agent | paraphrase_recall | 1/1 (100.0%) |
| short_term_agent | profile_summary | 0/2 (0.0%) |
| short_term_agent | simple_recall | 1/1 (100.0%) |
| short_term_agent | update | 1/1 (100.0%) |
| titan_memory_agent | consolidation | 1/2 (50.0%) |
| titan_memory_agent | forget | 2/2 (100.0%) |
| titan_memory_agent | hard_distractors | 1/1 (100.0%) |
| titan_memory_agent | identity_collision | 1/1 (100.0%) |
| titan_memory_agent | mixed_language_query | 1/1 (100.0%) |
| titan_memory_agent | multi_session | 1/1 (100.0%) |
| titan_memory_agent | multiple_updates | 0/2 (0.0%) |
| titan_memory_agent | noisy_query | 0/1 (0.0%) |
| titan_memory_agent | paraphrase_recall | 1/1 (100.0%) |
| titan_memory_agent | profile_summary | 2/2 (100.0%) |
| titan_memory_agent | simple_recall | 1/1 (100.0%) |
| titan_memory_agent | update | 1/1 (100.0%) |

## Failures
- **no_memory_agent / small_simple_secret_recall**: missing one of expected_any: ['8392'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / small_update_secret_code**: missing one of expected_any: ['1245'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / small_identity_collision_same_surname**: missing one of expected_any: ['green'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / small_forget_one_profile_keep_other**: missing one of expected_any: ['violet'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / small_paraphrased_location_recall**: missing one of expected_any: ['Nantes'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / small_multi_session_recall**: missing one of expected_any: ['Python'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / small_consolidated_project_convention**: missing one of expected_any: ['pytest'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / small_profile_summary**: missing expected_all: ['database optimization', 'mobile accessibility project'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / medium_noisy_secret_recall**: missing one of expected_any: ['Delta-42'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / medium_mixed_language_location**: missing one of expected_any: ['Madrid'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / medium_multiple_language_updates**: missing one of expected_any: ['Python'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / medium_multiple_secret_code_updates**: missing one of expected_any: ['FINAL-CODE-3053'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / medium_hard_distractor_room**: missing one of expected_any: ['B-204'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / medium_forget_with_nearby_distractor**: missing one of expected_any: ['cloud monitoring dashboard'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / medium_consolidated_after_restart**: missing one of expected_any: ['short bullet points'] — answer: `I do not know because no memory is available.`
- **no_memory_agent / medium_profile_summary**: missing expected_all: ['cybersecurity', 'gesture recognition prototype'] — answer: `I do not know because no memory is available.`
- **short_term_agent / small_multi_session_recall**: missing one of expected_any: ['Python'] — answer: `I do not know based on short-term memory.`
- **short_term_agent / small_consolidated_project_convention**: missing one of expected_any: ['pytest'] — answer: `The project convention is to use type hints in new Python files.`
- **short_term_agent / small_profile_summary**: missing expected_all: ['database optimization', 'mobile accessibility project'] — answer: `Emma Laurent prefers concise technical reports.`
- **short_term_agent / medium_multiple_secret_code_updates**: contains forbidden: ['CODE-3053.'] — answer: `The release secret code is FINAL-CODE-3053.`
- **short_term_agent / medium_consolidated_after_restart**: missing one of expected_any: ['short bullet points'] — answer: `I do not know based on short-term memory.`
- **short_term_agent / medium_profile_summary**: missing expected_all: ['cybersecurity', 'gesture recognition prototype']; contains forbidden: ['orange'] — answer: `Lucas Martin's favorite color is orange.`
- **titan_memory_agent / small_consolidated_project_convention**: missing one of expected_any: ['pytest'] — answer: `The project convention is to use type hints in new Python files.`
- **titan_memory_agent / medium_noisy_secret_recall**: missing one of expected_any: ['Delta-42'] — answer: `I do not know based on Titan memory.`
- **titan_memory_agent / medium_multiple_language_updates**: missing one of expected_any: ['Python']; contains forbidden: ['C++'] — answer: `The preferred backend language is C++.`
- **titan_memory_agent / medium_multiple_secret_code_updates**: missing one of expected_any: ['FINAL-CODE-3053']; contains forbidden: ['CODE-3053.'] — answer: `The release secret code is CODE-3053.`