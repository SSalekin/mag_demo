# Detailed multi-agent Titan team comparison

Cases: 24 | Scales: small, medium, stress

## Global results

| Agent | Score |
|---|---:|
| no_memory_agent | 0/24 (0.0%) |
| single_titan_agent | 10/24 (41.7%) |
| multi_agent_titan_team | 19/24 (79.2%) |

## Failures

- **no_memory_agent / small / api_endpoint_uses_pytest_type_hints**: missing_all=['pytest', 'type hints', 'endpoint'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / small / api_endpoint_uses_pytest_type_hints**: missing_all=['pytest', 'type hints', 'endpoint'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / small / ollama_provider_config**: missing_all=['Ollama', 'configurable'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / small / benchmark_regression_plan**: missing_all=['test', 'benchmark'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / small / benchmark_regression_plan**: missing_all=['test', 'benchmark'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / small / preferred_architecture_shared_titan**: missing_all=['manager-led', 'shared Titan memory'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / small / single_file_titan_preserved**: missing_all=['Titan', 'single-file'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / small / web_route_contract**: missing_all=['route', 'score', 'subject', 'property'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / small / web_route_contract**: missing_all=['route', 'score', 'subject', 'property'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / small / honest_reporting**: missing_all=['failures', 'honestly'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / small / honest_reporting**: missing_all=['failures', 'honestly'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / small / second_session_project_rule**: missing_all=['colored', 'terminal panels'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / medium / french_question_ollama_default**: missing_any=['Ollama'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / medium / french_question_ollama_default**: missing_any=['Ollama'] — answer: `if we change LLM later, it must be better and free.`
- **no_memory_agent / medium / endpoint_real_convention_among_distractors**: missing_all=['endpoint', 'pytest'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / medium / endpoint_real_convention_among_distractors**: missing_all=['endpoint', 'pytest'] — answer: `No relevant Titan memory found.`
- **multi_agent_titan_team / medium / endpoint_real_convention_among_distractors**: forbidden=['coffee', 'hotel', 'router']; missing_agents=['devsoft'] — answer: `Manager final response
Task: Create an implementation plan for a new endpoint and include the testing rule.

Shared memory used:
Shared active Titan memories:
1. Distractor: the coffee machine uses orange capsules.
2. Distractor: the hotel room number was B-204.
3. Project rule: new endpoints must i`
- **no_memory_agent / medium / devops_only_installation**: missing_all=['Commands', 'PowerShell'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / medium / devops_only_installation**: missing_all=['Commands', 'PowerShell'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / medium / tester_evaluator_regression**: missing_all=['no-memory', 'failures', 'benchmark'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / medium / tester_evaluator_regression**: missing_all=['no-memory', 'failures', 'benchmark'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / medium / provider_policy_update**: missing_any=['Ollama'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / medium / provider_policy_update**: forbidden=['Gemini'] — answer: `The preferred LLM provider for agent tests is Ollama.
The preferred LLM provider for agent tests is Gemini.`
- **multi_agent_titan_team / medium / provider_policy_update**: forbidden=['Gemini'] — answer: `Manager final response
Task: Which LLM provider should the prototype use for tests right now?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.542; subject=?; property=?] The preferred LLM provider for agent tests is Gemini.
2. [score=0.540; subject=?; property=?] The preferred LL`
- **no_memory_agent / medium / testing_framework_update**: missing_any=['pytest'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / medium / testing_framework_update**: forbidden=['unittest'] — answer: `The project testing framework is unittest.
The project testing framework is pytest.`
- **multi_agent_titan_team / medium / testing_framework_update**: forbidden=['unittest'] — answer: `Manager final response
Task: Which testing framework should the team use for new tests?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.429; subject=?; property=testing_framework] The project testing framework is unittest.
2. [score=0.381; subject=?; property=testing_framework] T`
- **no_memory_agent / medium / team_policy_summary**: missing_all=['manager', 'tests', 'limitations'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / medium / team_policy_summary**: missing_all=['manager', 'tests', 'limitations'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / medium / forget_old_secret_keep_project_rule**: missing_any=['Ollama', 'configurable'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / medium / forget_old_secret_keep_project_rule**: missing_any=['Ollama', 'configurable'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / stress / noisy_question_about_titan_memory**: missing_all=['Titan'] — answer: `I do not know because no shared memory is available.`
- **multi_agent_titan_team / stress / noisy_question_about_titan_memory**: forbidden=['banana', 'router'] — answer: `Manager final response
Task: Ignore noise !!! banana router sticker ### which memory layer does the team use?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.288; subject=?; property=?] The shared memory layer for the team is Titan long-term memory.

Agent plan:
- MANAGER (Projec`
- **no_memory_agent / stress / many_project_distractors_keep_real_rule**: missing_all=['type hints', 'pytest'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / stress / same_surname_agent_owners**: missing_any=['Sarah Martin'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / stress / same_surname_other_owner**: missing_any=['Lucas Martin'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / stress / consolidated_demo_rules**: missing_all=['colorful', 'selected agents', 'CSV'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / stress / consolidated_demo_rules**: missing_all=['colorful', 'selected agents', 'CSV'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / stress / full_stack_task_all_roles**: missing_all=['FastAPI', 'pytest', 'Commands', 'Evaluation'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / stress / full_stack_task_all_roles**: missing_all=['Evaluation'] — answer: `The project uses FastAPI, pytest, PowerShell commands and honest benchmark reports.`
- **no_memory_agent / stress / paraphrased_best_architecture**: missing_all=['manager', 'Titan'] — answer: `I do not know because no shared memory is available.`
- **multi_agent_titan_team / stress / paraphrased_best_architecture**: missing_all=['Titan'] — answer: `Manager final response
Task: What setup did we decide is the strongest for coordinating specialist agents with long-term recall?

Shared memory used:
No relevant long-term memory found.

Agent plan:
- MANAGER (Project manager and orchestrator, confidence=0.85)
  action: coordinate
  action: store_de`
- **no_memory_agent / stress / free_llm_future_constraint**: missing_all=['better', 'free'] — answer: `I do not know because no shared memory is available.`
