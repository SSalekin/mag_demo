# Detailed multi-agent Titan team comparison

Cases: 78 | Total available: 78 | Scales: small, medium, stress, large

## Global results

| Agent | Score |
|---|---:|
| no_memory_agent | 0/78 (0.0%) |
| single_titan_agent | 28/78 (35.9%) |
| multi_agent_titan_team | 42/78 (53.8%) |

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
1. [score=0.489; subject=?; property=?] The preferred LLM provider for agent tests is Gemini.
2. [score=0.482; subject=?; property=?] The preferred LL`
- **no_memory_agent / medium / testing_framework_update**: missing_any=['pytest'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / medium / testing_framework_update**: forbidden=['unittest'] — answer: `The project testing framework is unittest.
The project testing framework is pytest.`
- **multi_agent_titan_team / medium / testing_framework_update**: forbidden=['unittest'] — answer: `Manager final response
Task: Which testing framework should the team use for new tests?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.443; subject=?; property=testing_framework] The project testing framework is unittest.
2. [score=0.411; subject=?; property=testing_framework] T`
- **no_memory_agent / medium / team_policy_summary**: missing_all=['manager', 'tests', 'limitations'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / medium / team_policy_summary**: missing_all=['manager', 'tests', 'limitations'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / medium / forget_old_secret_keep_project_rule**: missing_any=['Ollama', 'configurable'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / medium / forget_old_secret_keep_project_rule**: missing_any=['Ollama', 'configurable'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / stress / noisy_question_about_titan_memory**: missing_all=['Titan'] — answer: `I do not know because no shared memory is available.`
- **multi_agent_titan_team / stress / noisy_question_about_titan_memory**: forbidden=['banana', 'router'] — answer: `Manager final response
Task: Ignore noise !!! banana router sticker ### which memory layer does the team use?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.282; subject=?; property=?] The shared memory layer for the team is Titan long-term memory.

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
- **no_memory_agent / large / large_convention_billing**: missing_all=['FastAPI', 'pytest']; missing_any=['type hints'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_convention_billing**: missing_all=['FastAPI', 'pytest']; missing_any=['type hints'] — answer: `No relevant Titan memory found.`
- **multi_agent_titan_team / large / large_convention_billing**: missing_all=['FastAPI']; missing_any=['type hints']; missing_agents=['devweb', 'devsoft'] — answer: `Manager final response
Task: Create a plan for the billing work and preserve the critical convention.

Shared memory used:
No relevant long-term memory found.

Agent plan:
- MANAGER (Project manager and orchestrator, confidence=0.85)
  action: coordinate
  action: store_decision
- TESTER (Test and b`
- **no_memory_agent / large / large_convention_analytics**: missing_all=['REST', 'pytest']; missing_any=['typed schemas'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_convention_analytics**: missing_all=['REST', 'pytest']; missing_any=['typed schemas'] — answer: `No relevant Titan memory found.`
- **multi_agent_titan_team / large / large_convention_analytics**: forbidden=['hotel room', 'coffee capsule', 'TRAVEL']; missing_agents=['devweb', 'devsoft'] — answer: `Manager final response
Task: Create a plan for the analytics work and preserve the critical convention.

Shared memory used:
Shared active Titan memories:
1. Distractor 2-A: hotel room B-202 is unrelated.
2. Distractor 2-B: coffee capsule color is orange.
3. Critical project convention for analytics`
- **no_memory_agent / large / large_convention_memory_search**: missing_all=['FastAPI', 'unit tests']; missing_any=['score field'] — answer: `I do not know because no shared memory is available.`
- **multi_agent_titan_team / large / large_convention_memory_search**: missing_all=['FastAPI', 'unit tests']; missing_any=['score field']; missing_agents=['devweb', 'devsoft'] — answer: `Manager final response
Task: Create a plan for the memory-search work and preserve the critical convention.

Shared memory used:
No relevant long-term memory found.

Agent plan:
- MANAGER (Project manager and orchestrator, confidence=0.85)
  action: coordinate
  action: store_decision
- TESTER (Test`
- **no_memory_agent / large / large_convention_deployment**: missing_all=['PowerShell', 'environment variables']; missing_any=['Ollama'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_convention_benchmark**: missing_all=['CSV reports', 'failure analysis']; missing_any=['honest reporting'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_convention_dashboard**: missing_all=['colored terminal panels', 'selected agents']; missing_any=['memory context'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_convention_adapter**: missing_all=['single-file Titan', 'clean API']; missing_any=['save/load'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_convention_provider**: missing_all=['configurable LLM', 'Ollama default']; missing_any=['free alternative later'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_convention_documentation**: missing_all=['limitations', 'commands']; missing_any=['benchmark results'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_convention_evaluation**: missing_all=['baseline comparison', 'stress categories']; missing_any=['risk summary'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_identity_collision_1**: missing_any=['Sarah Martin'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_identity_collision_2**: missing_any=['Lucas Martin'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_identity_collision_3**: missing_any=['Emma Laurent'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_identity_collision_3**: forbidden=['Ethan Laurent'] — answer: `Emma Laurent owns the benchmark report task.
Ethan Laurent owns the database migration task.
Distractor: both people attended the same meeting.`
- **multi_agent_titan_team / large / large_identity_collision_3**: forbidden=['Ethan Laurent'] — answer: `Manager final response
Task: Who owns the benchmark report task?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.457; subject=?; property=?] Emma Laurent owns the benchmark report task.
2. [score=0.236; subject=?; property=?] Ethan Laurent owns the database migration task.

Agent`
- **no_memory_agent / large / large_identity_collision_4**: missing_any=['Ethan Laurent'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_identity_collision_4**: forbidden=['Emma Laurent'] — answer: `Ethan Laurent owns the database migration task.
Emma Laurent owns the benchmark report task.
Distractor: both people attended the same meeting.`
- **multi_agent_titan_team / large / large_identity_collision_4**: forbidden=['Emma Laurent'] — answer: `Manager final response
Task: Who owns the database migration task?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.446; subject=?; property=?] Ethan Laurent owns the database migration task.
2. [score=0.256; subject=?; property=?] Emma Laurent owns the benchmark report task.

Age`
- **no_memory_agent / large / large_identity_collision_5**: missing_any=['Noah Bernard'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_identity_collision_5**: forbidden=['Nina Bernard'] — answer: `Noah Bernard owns the provider configuration task.
Nina Bernard owns the UI polish task.
Distractor: both people attended the same meeting.`
- **multi_agent_titan_team / large / large_identity_collision_5**: forbidden=['Nina Bernard'] — answer: `Manager final response
Task: Who owns the provider configuration task?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.488; subject=?; property=?] Noah Bernard owns the provider configuration task.
2. [score=0.296; subject=?; property=?] Nina Bernard owns the UI polish task.

Age`
- **no_memory_agent / large / large_identity_collision_6**: missing_any=['Nina Bernard'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_identity_collision_6**: forbidden=['Noah Bernard'] — answer: `Nina Bernard owns the UI polish task.
Noah Bernard owns the provider configuration task.
Distractor: both people attended the same meeting.`
- **multi_agent_titan_team / large / large_identity_collision_6**: forbidden=['Noah Bernard'] — answer: `Manager final response
Task: Who owns the UI polish task?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.404; subject=?; property=?] Nina Bernard owns the UI polish task.
2. [score=0.273; subject=?; property=?] Noah Bernard owns the provider configuration task.

Agent plan:
- MA`
- **no_memory_agent / large / large_identity_collision_7**: missing_any=['Hugo Morel'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_identity_collision_7**: forbidden=['Hugo Martin'] — answer: `Hugo Morel owns the risk evaluation task.
Hugo Martin owns the API route task.
Distractor: both people attended the same meeting.`
- **multi_agent_titan_team / large / large_identity_collision_7**: forbidden=['Hugo Martin'] — answer: `Manager final response
Task: Who owns the risk evaluation task?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.450; subject=?; property=?] Hugo Morel owns the risk evaluation task.
2. [score=0.237; subject=?; property=?] Hugo Martin owns the API route task.

Agent plan:
- MANAGE`
- **no_memory_agent / large / large_identity_collision_8**: missing_any=['Hugo Martin'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_update_1**: missing_any=['Python'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_update_2**: missing_any=['Ollama'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_update_2**: forbidden=['Gemini', 'Claude'] — answer: `The preferred LLM provider for current tests is Ollama.
The preferred LLM provider for current tests is Claude.
The preferred LLM provider for current tests is Gemini.`
- **multi_agent_titan_team / large / large_update_2**: forbidden=['Gemini', 'Claude']; missing_agents=['evaluator'] — answer: `Manager final response
Task: What is the current preferred LLM provider for current tests?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.684; subject=?; property=?] The preferred LLM provider for current tests is Ollama.
2. [score=0.679; subject=?; property=?] The preferred LLM`
- **no_memory_agent / large / large_update_3**: missing_any=['pytest'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_update_3**: forbidden=['unittest', 'nose'] — answer: `The testing framework is unittest.
The testing framework is pytest.
The testing framework is nose.`
- **multi_agent_titan_team / large / large_update_3**: forbidden=['unittest', 'nose'] — answer: `Manager final response
Task: What is the current testing framework?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.528; subject=?; property=testing_framework] The testing framework is pytest.
2. [score=0.525; subject=?; property=testing_framework] The testing framework is nose.
`
- **no_memory_agent / large / large_update_4**: missing_any=['colored terminal panels'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_update_4**: forbidden=['plain text', 'markdown only'] — answer: `The demo UI style is plain text.
The demo UI style is markdown only.
The demo UI style is colored terminal panels.`
- **multi_agent_titan_team / large / large_update_4**: forbidden=['plain text', 'markdown only']; missing_agents=['evaluator'] — answer: `Manager final response
Task: What is the current demo UI style?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.352; subject=?; property=?] The demo UI style is plain text.
2. [score=0.336; subject=?; property=?] The demo UI style is colored terminal panels.
3. [score=0.329; subj`
- **no_memory_agent / large / large_update_5**: missing_any=['shared Titan memory'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_update_5**: forbidden=['short-term memory', 'vector store'] — answer: `The memory backend is short-term memory.
The memory backend is shared Titan memory.
The memory backend is vector store.`
- **multi_agent_titan_team / large / large_update_5**: forbidden=['short-term memory', 'vector store']; missing_agents=['devsoft', 'evaluator'] — answer: `Manager final response
Task: What is the current memory backend?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.346; subject=?; property=?] The memory backend is vector store.
2. [score=0.345; subject=?; property=?] The memory backend is short-term memory.
3. [score=0.321; subje`
- **no_memory_agent / large / large_update_6**: missing_any=['PowerShell'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_update_6**: forbidden=['bash', 'cmd'] — answer: `The deployment shell is bash.
The deployment shell is PowerShell.
The deployment shell is cmd.`
- **multi_agent_titan_team / large / large_update_6**: forbidden=['bash', 'cmd'] — answer: `Manager final response
Task: What is the current deployment shell?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.350; subject=?; property=?] The deployment shell is PowerShell.
2. [score=0.325; subject=?; property=?] The deployment shell is cmd.
3. [score=0.321; subject=?; prop`
- **no_memory_agent / large / large_update_7**: missing_any=['report failures honestly'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_update_7**: forbidden=['hide failures', 'only show wins'] — answer: `The reporting rule is hide failures.
The reporting rule is only show wins.
The reporting rule is report failures honestly.`
- **multi_agent_titan_team / large / large_update_7**: forbidden=['hide failures', 'only show wins'] — answer: `Manager final response
Task: What is the current reporting rule?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.342; subject=?; property=?] The reporting rule is hide failures.
2. [score=0.327; subject=?; property=?] The reporting rule is only show wins.
3. [score=0.323; subject`
- **no_memory_agent / large / large_update_8**: missing_any=['CSV and markdown report'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_update_8**: forbidden=['text only', 'screenshots only'] — answer: `The benchmark output format is text only.
The benchmark output format is screenshots only.
The benchmark output format is CSV and markdown report.`
- **multi_agent_titan_team / large / large_update_8**: forbidden=['text only', 'screenshots only']; missing_agents=['evaluator'] — answer: `Manager final response
Task: What is the current benchmark output format?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.383; subject=?; property=?] The benchmark output format is screenshots only.
2. [score=0.373; subject=?; property=?] The benchmark output format is text only.`
- **no_memory_agent / large / large_forget_keep_rule_1**: missing_all=['benchmark', 'limitations'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_forget_keep_rule_1**: missing_all=['benchmark', 'limitations'] — answer: `Distractor: Lucas Martin's old favorite color was cyan-1.`
- **multi_agent_titan_team / large / large_forget_keep_rule_1**: missing_all=['limitations'] — answer: `Manager final response
Task: Forget Lucas Martin's temporary secret code, then explain the remaining project rule.

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.333; subject=?; property=favorite_color] Distractor: Lucas Martin's old favorite color was cyan-1.

Agent plan:
- MAN`
- **no_memory_agent / large / large_forget_keep_rule_2**: missing_all=['benchmark', 'limitations'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_forget_keep_rule_2**: missing_all=['benchmark', 'limitations'] — answer: `Distractor: Lucas Martin's old favorite color was cyan-2.`
- **multi_agent_titan_team / large / large_forget_keep_rule_2**: missing_all=['limitations'] — answer: `Manager final response
Task: Forget Lucas Martin's temporary secret code, then explain the remaining project rule.

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.319; subject=?; property=favorite_color] Distractor: Lucas Martin's old favorite color was cyan-2.

Agent plan:
- MAN`
- **no_memory_agent / large / large_forget_keep_rule_3**: missing_all=['benchmark', 'limitations'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_forget_keep_rule_3**: missing_all=['benchmark', 'limitations'] — answer: `Distractor: Lucas Martin's old favorite color was cyan-3.`
- **multi_agent_titan_team / large / large_forget_keep_rule_3**: missing_all=['limitations'] — answer: `Manager final response
Task: Forget Lucas Martin's temporary secret code, then explain the remaining project rule.

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.348; subject=?; property=favorite_color] Distractor: Lucas Martin's old favorite color was cyan-3.

Agent plan:
- MAN`
- **no_memory_agent / large / large_forget_keep_rule_4**: missing_all=['benchmark', 'limitations'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_forget_keep_rule_4**: missing_all=['benchmark', 'limitations'] — answer: `Distractor: Lucas Martin's old favorite color was cyan-4.`
- **multi_agent_titan_team / large / large_forget_keep_rule_4**: missing_all=['limitations'] — answer: `Manager final response
Task: Forget Lucas Martin's temporary secret code, then explain the remaining project rule.

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.360; subject=?; property=favorite_color] Distractor: Lucas Martin's old favorite color was cyan-4.

Agent plan:
- MAN`
- **no_memory_agent / large / large_forget_keep_rule_5**: missing_all=['benchmark', 'limitations'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_forget_keep_rule_5**: missing_all=['benchmark', 'limitations'] — answer: `Distractor: Lucas Martin's old favorite color was cyan-5.`
- **multi_agent_titan_team / large / large_forget_keep_rule_5**: missing_all=['limitations'] — answer: `Manager final response
Task: Forget Lucas Martin's temporary secret code, then explain the remaining project rule.

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.338; subject=?; property=favorite_color] Distractor: Lucas Martin's old favorite color was cyan-5.

Agent plan:
- MAN`
- **no_memory_agent / large / large_forget_keep_rule_6**: missing_all=['benchmark', 'limitations'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_forget_keep_rule_6**: missing_all=['benchmark', 'limitations'] — answer: `Distractor: Lucas Martin's old favorite color was cyan-6.`
- **multi_agent_titan_team / large / large_forget_keep_rule_6**: missing_all=['limitations'] — answer: `Manager final response
Task: Forget Lucas Martin's temporary secret code, then explain the remaining project rule.

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.344; subject=?; property=favorite_color] Distractor: Lucas Martin's old favorite color was cyan-6.

Agent plan:
- MAN`
- **no_memory_agent / large / large_forget_keep_rule_7**: missing_all=['benchmark', 'limitations'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_forget_keep_rule_7**: missing_all=['benchmark', 'limitations'] — answer: `Distractor: Lucas Martin's old favorite color was cyan-7.`
- **multi_agent_titan_team / large / large_forget_keep_rule_7**: missing_all=['limitations'] — answer: `Manager final response
Task: Forget Lucas Martin's temporary secret code, then explain the remaining project rule.

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.399; subject=?; property=favorite_color] Distractor: Lucas Martin's old favorite color was cyan-7.

Agent plan:
- MAN`
- **no_memory_agent / large / large_forget_keep_rule_8**: missing_all=['benchmark', 'limitations'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_forget_keep_rule_8**: missing_all=['benchmark', 'limitations'] — answer: `Distractor: Lucas Martin's old favorite color was cyan-8.`
- **multi_agent_titan_team / large / large_forget_keep_rule_8**: missing_all=['limitations'] — answer: `Manager final response
Task: Forget Lucas Martin's temporary secret code, then explain the remaining project rule.

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.321; subject=?; property=favorite_color] Distractor: Lucas Martin's old favorite color was cyan-8.

Agent plan:
- MAN`
- **no_memory_agent / large / large_consolidation_1**: missing_all=['terminal', 'manager']; missing_any=['results'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_consolidation_1**: missing_all=['terminal', 'manager']; missing_any=['results'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / large / large_consolidation_2**: missing_all=['Ollama', 'future']; missing_any=['provider'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_consolidation_2**: missing_all=['Ollama', 'future']; missing_any=['provider'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / large / large_consolidation_3**: missing_all=['Tester', 'Evaluator']; missing_any=['Manager'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_consolidation_3**: missing_all=['Tester', 'Evaluator']; missing_any=['Manager'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / large / large_consolidation_4**: missing_all=['DevSoft', 'DevWeb']; missing_any=['DevOps'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_consolidation_4**: missing_all=['DevSoft', 'DevWeb']; missing_any=['DevOps'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / large / large_consolidation_5**: missing_all=['profile', 'forget']; missing_any=['consolidation'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_consolidation_5**: missing_all=['profile', 'forget']; missing_any=['consolidation'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / large / large_consolidation_6**: missing_all=['large', 'stress']; missing_any=['reports'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_consolidation_6**: missing_all=['large', 'stress']; missing_any=['reports'] — answer: `No relevant Titan memory found.`
- **no_memory_agent / large / large_noisy_1**: missing_all=['Titan'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_noisy_1**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Critical memory 1: the team uses shared Titan memory for long-term project recall.
Noise distractor 1: banana router sticker coffee hotel orange.`
- **multi_agent_titan_team / large / large_noisy_1**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Manager final response
Task: Ignore banana !!! router ### coffee ??? What memory does the team use for long-term recall in case 1?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.272; subject=?; property=project] Critical memory 1: the team uses shared Titan memory for long-term `
- **no_memory_agent / large / large_noisy_2**: missing_all=['Titan'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_noisy_2**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Critical memory 2: the team uses shared Titan memory for long-term project recall.
Noise distractor 2: banana router sticker coffee hotel orange.`
- **multi_agent_titan_team / large / large_noisy_2**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Manager final response
Task: Ignore banana !!! router ### coffee ??? What memory does the team use for long-term recall in case 2?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.251; subject=?; property=project] Critical memory 2: the team uses shared Titan memory for long-term `
- **no_memory_agent / large / large_noisy_3**: missing_all=['Titan'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_noisy_3**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Critical memory 3: the team uses shared Titan memory for long-term project recall.
Noise distractor 3: banana router sticker coffee hotel orange.`
- **multi_agent_titan_team / large / large_noisy_3**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Manager final response
Task: Ignore banana !!! router ### coffee ??? What memory does the team use for long-term recall in case 3?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.262; subject=?; property=project] Critical memory 3: the team uses shared Titan memory for long-term `
- **no_memory_agent / large / large_noisy_4**: missing_all=['Titan'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_noisy_4**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Critical memory 4: the team uses shared Titan memory for long-term project recall.
Noise distractor 4: banana router sticker coffee hotel orange.`
- **multi_agent_titan_team / large / large_noisy_4**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Manager final response
Task: Ignore banana !!! router ### coffee ??? What memory does the team use for long-term recall in case 4?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.234; subject=?; property=project] Critical memory 4: the team uses shared Titan memory for long-term `
- **no_memory_agent / large / large_noisy_5**: missing_all=['Titan'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_noisy_5**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Critical memory 5: the team uses shared Titan memory for long-term project recall.
Noise distractor 5: banana router sticker coffee hotel orange.`
- **multi_agent_titan_team / large / large_noisy_5**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Manager final response
Task: Ignore banana !!! router ### coffee ??? What memory does the team use for long-term recall in case 5?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.241; subject=?; property=project] Critical memory 5: the team uses shared Titan memory for long-term `
- **no_memory_agent / large / large_noisy_6**: missing_all=['Titan'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_noisy_6**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Critical memory 6: the team uses shared Titan memory for long-term project recall.
Noise distractor 6: banana router sticker coffee hotel orange.`
- **multi_agent_titan_team / large / large_noisy_6**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Manager final response
Task: Ignore banana !!! router ### coffee ??? What memory does the team use for long-term recall in case 6?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.274; subject=?; property=project] Critical memory 6: the team uses shared Titan memory for long-term `
- **no_memory_agent / large / large_noisy_7**: missing_all=['Titan'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_noisy_7**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Critical memory 7: the team uses shared Titan memory for long-term project recall.
Noise distractor 7: banana router sticker coffee hotel orange.`
- **multi_agent_titan_team / large / large_noisy_7**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Manager final response
Task: Ignore banana !!! router ### coffee ??? What memory does the team use for long-term recall in case 7?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.283; subject=?; property=project] Critical memory 7: the team uses shared Titan memory for long-term `
- **no_memory_agent / large / large_noisy_8**: missing_all=['Titan'] — answer: `I do not know because no shared memory is available.`
- **single_titan_agent / large / large_noisy_8**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Critical memory 8: the team uses shared Titan memory for long-term project recall.
Noise distractor 8: banana router sticker coffee hotel orange.`
- **multi_agent_titan_team / large / large_noisy_8**: forbidden=['banana', 'coffee', 'hotel'] — answer: `Manager final response
Task: Ignore banana !!! router ### coffee ??? What memory does the team use for long-term recall in case 8?

Shared memory used:
Relevant long-term Titan memories:
1. [score=0.259; subject=?; property=project] Critical memory 8: the team uses shared Titan memory for long-term `
- **no_memory_agent / large / large_profile_summary_1**: missing_all=['database optimization', 'mobile accessibility project']; missing_any=['concise'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_profile_summary_2**: missing_all=['database optimization', 'mobile accessibility project']; missing_any=['concise'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_profile_summary_3**: missing_all=['database optimization', 'mobile accessibility project']; missing_any=['concise'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_profile_summary_4**: missing_all=['database optimization', 'mobile accessibility project']; missing_any=['concise'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_profile_summary_5**: missing_all=['database optimization', 'mobile accessibility project']; missing_any=['concise'] — answer: `I do not know because no shared memory is available.`
- **no_memory_agent / large / large_profile_summary_6**: missing_all=['database optimization', 'mobile accessibility project']; missing_any=['concise'] — answer: `I do not know because no shared memory is available.`
