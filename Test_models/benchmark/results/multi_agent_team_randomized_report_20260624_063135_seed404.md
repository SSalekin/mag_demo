# Randomized adversarial multi-agent Titan team comparison

============================================================================================
RANDOMIZED ADVERSARIAL MULTI-AGENT TITAN TEAM COMPARISON
============================================================================================
Cases: 150 | Seed: 404

GLOBAL RESULTS
--------------------------------------------------------------------------------------------
no_memory_agent          | 0/150 (0.0%)
single_titan_agent       | 17/150 (11.3%)
multi_agent_titan_team   | 125/150 (83.3%)

RESULTS BY CATEGORY
--------------------------------------------------------------------------------------------
no_memory_agent          | consolidation            | 0/18 (0.0%)
no_memory_agent          | forget_retention         | 0/18 (0.0%)
no_memory_agent          | identity_collision       | 0/36 (0.0%)
no_memory_agent          | latest_value_update      | 0/17 (0.0%)
no_memory_agent          | noisy_query              | 0/19 (0.0%)
no_memory_agent          | policy_recall            | 0/4 (0.0%)
no_memory_agent          | project_convention       | 0/18 (0.0%)
no_memory_agent          | role_routing             | 0/20 (0.0%)
single_titan_agent       | consolidation            | 0/18 (0.0%)
single_titan_agent       | forget_retention         | 0/18 (0.0%)
single_titan_agent       | identity_collision       | 0/36 (0.0%)
single_titan_agent       | latest_value_update      | 1/17 (5.9%)
single_titan_agent       | noisy_query              | 0/19 (0.0%)
single_titan_agent       | policy_recall            | 3/4 (75.0%)
single_titan_agent       | project_convention       | 13/18 (72.2%)
single_titan_agent       | role_routing             | 0/20 (0.0%)
multi_agent_titan_team   | consolidation            | 18/18 (100.0%)
multi_agent_titan_team   | forget_retention         | 18/18 (100.0%)
multi_agent_titan_team   | identity_collision       | 34/36 (94.4%)
multi_agent_titan_team   | latest_value_update      | 10/17 (58.8%)
multi_agent_titan_team   | noisy_query              | 4/19 (21.1%)
multi_agent_titan_team   | policy_recall            | 3/4 (75.0%)
multi_agent_titan_team   | project_convention       | 18/18 (100.0%)
multi_agent_titan_team   | role_routing             | 20/20 (100.0%)

FAILURES
--------------------------------------------------------------------------------------------
- no_memory_agent / role_routing / random_role_3: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_3: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / latest_value_update / random_update_7_current_code_formatter: missing_any=['black'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_7_current_code_formatter: forbidden=['ruff format', 'yapf'] — answer=`The current code formatter is black. The current code formatter is yapf. The current code formatter is ruff format.`
- no_memory_agent / identity_collision / random_owner_6_b: missing_any=['Ivan Lemoine'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_6_b: forbidden=['Iris Lemoine', 'release checklist', 'standup'] — answer=`Ivan Lemoine owns the regression suite task. Iris Lemoine owns the release checklist task. Distractor: Iris Lemoine and Ivan Lemoine attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_21: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_21: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-20.`
- no_memory_agent / consolidation / random_consolidation_14: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_14: missing_all=['Ollama', 'future'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / latest_value_update / random_update_21_current_llm_provider_for_tests: missing_any=['Ollama'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_21_current_llm_provider_for_tests: forbidden=['Claude', 'Gemini'] — answer=`The current LLM provider for tests is Ollama. The current LLM provider for tests is Gemini. The current LLM provider for tests is Claude.`
- no_memory_agent / role_routing / random_role_8: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_8: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / role_routing / random_role_2: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_2: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / latest_value_update / random_update_17_current_serialization_format: missing_any=['pickle'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_17_current_serialization_format: forbidden=['JSON', 'XML'] — answer=`The current serialization format is XML. The current serialization format is JSON. The current serialization format is pickle.`
- multi_agent_titan_team / latest_value_update / random_update_17_current_serialization_format: forbidden=['JSON', 'XML'] — answer=`Manager final response Task: What is the current serialization format right now Shared memory used: Selected shared Titan memories: 1. The current serialization format is pickle. 2. The current serialization format is JSON. 3. The current serialization format is XML. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate...`
- no_memory_agent / project_convention / random_convention_8_audit-log: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / role_routing / random_role_19: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_19: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / consolidation / random_consolidation_10: missing_all=['DevSoft', 'DevWeb'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_10: missing_all=['DevSoft', 'DevWeb'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / noisy_query / random_noisy_6: missing_any=['QZ-4-005'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_6: forbidden=['hotel', 'sticker', 'capsule', 'orange', 'travel ad', 'karaoke'] — answer=`Critical incident memory 5: the escalation key is QZ-4-005. Noise distractor 5: travel ad karaoke orange sticker hotel capsule should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_6: forbidden=['beach'] — answer=`Manager final response Task: Ignore beach what is the escalation key for incident 5 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 5: the escalation key is QZ-4-005. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confide...`
- no_memory_agent / forget_retention / random_forget_retention_12: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_12: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-11.`
- no_memory_agent / consolidation / random_consolidation_13: missing_all=['terminal', 'manager'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_13: missing_all=['terminal', 'manager'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / consolidation / random_consolidation_6: missing_all=['large', 'stress'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_6: missing_all=['large', 'stress'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / noisy_query / random_noisy_14: missing_any=['QZ-4-013'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_14: forbidden=['banana', 'hotel', 'sticker', 'capsule', 'orange', 'beach'] — answer=`Noise distractor 13: orange capsule banana hotel sticker beach should be ignored. Critical incident memory 13: the escalation key is QZ-4-013.`
- multi_agent_titan_team / noisy_query / random_noisy_14: forbidden=['beach'] — answer=`Manager final response Task: Ignore beach what is the escalation key for incident 13 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 13: the escalation key is QZ-4-013. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confi...`
- no_memory_agent / noisy_query / random_noisy_16: missing_any=['QZ-4-015'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_16: forbidden=['banana', 'coffee', 'router', 'orange', 'travel ad', 'beach'] — answer=`Critical incident memory 15: the escalation key is QZ-4-015. Noise distractor 15: travel ad coffee beach banana router orange should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_16: forbidden=['beach', 'karaoke'] — answer=`Manager final response Task: Ignore karaoke beach what is the escalation key for incident 15 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 15: the escalation key is QZ-4-015. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python develope...`
- no_memory_agent / noisy_query / random_noisy_10: missing_any=['QZ-4-009'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_10: forbidden=['banana', 'router', 'capsule', 'orange', 'beach', 'karaoke'] — answer=`Critical incident memory 9: the escalation key is QZ-4-009. Noise distractor 9: banana beach orange karaoke router capsule should be ignored.`
- no_memory_agent / identity_collision / random_owner_20_b: missing_any=['Corentin Masson'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_20_b: forbidden=['Camille Masson', 'memory adapter', 'standup'] — answer=`Corentin Masson owns the benchmark report task. Camille Masson owns the memory adapter task. Distractor: Camille Masson and Corentin Masson attended the same standup meeting.`
- no_memory_agent / project_convention / random_convention_21_inventory: missing_all=['RBAC', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / identity_collision / random_owner_17_a: missing_any=['Yuna Petit'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_17_a: forbidden=['Yanis Petit', 'monitoring setup', 'standup'] — answer=`Yuna Petit owns the security audit task. Yanis Petit owns the monitoring setup task. Distractor: Yuna Petit and Yanis Petit attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_3_a: missing_any=['Mila Bernard'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_3_a: forbidden=['Mathis Bernard', 'regression suite', 'standup'] — answer=`Mila Bernard owns the risk summary task. Mathis Bernard owns the regression suite task. Distractor: Mila Bernard and Mathis Bernard attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_5_b: missing_any=['Nathan Garnier'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_5_b: forbidden=['Nora Garnier', 'release checklist', 'standup'] — answer=`Nathan Garnier owns the regression suite task. Nora Garnier owns the release checklist task. Distractor: Nora Garnier and Nathan Garnier attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_9: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_9: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-8.`
- no_memory_agent / latest_value_update / random_update_15_current_queue_backend: missing_any=['Redis streams'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_15_current_queue_backend: forbidden=['Kafka', 'RabbitMQ'] — answer=`The current queue backend is Kafka. The current queue backend is RabbitMQ. The current queue backend is Redis streams.`
- multi_agent_titan_team / latest_value_update / random_update_15_current_queue_backend: forbidden=['Kafka', 'RabbitMQ'] — answer=`Manager final response Task: What is the current queue backend right now Shared memory used: Selected shared Titan memories: 1. The current queue backend is Redis streams. 2. The current queue backend is RabbitMQ. 3. The current queue backend is Kafka. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_...`
- no_memory_agent / consolidation / random_consolidation_15: missing_all=['Tester', 'Evaluator'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_15: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 14: preserve Tester, Evaluator, and Manager.`
- no_memory_agent / noisy_query / random_noisy_19: missing_any=['QZ-4-018'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_19: forbidden=['banana', 'coffee', 'sticker', 'capsule', 'beach', 'karaoke'] — answer=`Critical incident memory 18: the escalation key is QZ-4-018. Noise distractor 18: beach sticker karaoke capsule coffee banana should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_19: forbidden=['karaoke'] — answer=`Manager final response Task: Ignore karaoke what is the escalation key for incident 18 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 18: the escalation key is QZ-4-018. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVWEB (Web/API developer, confidence=0...`
- no_memory_agent / consolidation / random_consolidation_21: missing_all=['Tester', 'Evaluator'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_21: missing_all=['Tester', 'Evaluator'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / policy_recall / random_policy_5: missing_all=['manager', 'shared Titan memory'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / role_routing / random_role_10: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_10: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / forget_retention / random_forget_retention_15: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_15: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-14.`
- no_memory_agent / identity_collision / random_owner_20_a: missing_any=['Camille Masson'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_20_a: forbidden=['Corentin Masson', 'benchmark report', 'standup'] — answer=`Camille Masson owns the memory adapter task. Corentin Masson owns the benchmark report task. Distractor: Camille Masson and Corentin Masson attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_15_b: missing_any=['Nathan Garnier'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_15_b: forbidden=['Nora Garnier', 'provider evaluation', 'standup'] — answer=`Nathan Garnier owns the documentation pass task. Nora Garnier owns the provider evaluation task. Distractor: Nora Garnier and Nathan Garnier attended the same standup meeting.`
- no_memory_agent / role_routing / random_role_12: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_12: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / identity_collision / random_owner_14_a: missing_any=['Lina Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_14_a: forbidden=['Leo Robert', 'monitoring setup', 'standup'] — answer=`Lina Robert owns the report export task. Leo Robert owns the monitoring setup task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- no_memory_agent / noisy_query / random_noisy_20: missing_any=['QZ-4-019'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_20: forbidden=['banana', 'hotel', 'sticker', 'orange', 'travel ad', 'karaoke'] — answer=`Noise distractor 19: orange travel ad hotel sticker banana karaoke should be ignored. Critical incident memory 19: the escalation key is QZ-4-019.`
- multi_agent_titan_team / noisy_query / random_noisy_20: forbidden=['karaoke'] — answer=`Manager final response Task: Ignore karaoke what is the escalation key for incident 19 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 19: the escalation key is QZ-4-019. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, con...`
- no_memory_agent / consolidation / random_consolidation_17: missing_all=['profile', 'forget'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_17: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 16: preserve profile, forget, and consolidation.`
- no_memory_agent / identity_collision / random_owner_18_b: missing_any=['Elias Roux'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_18_b: forbidden=['Elise Roux', 'risk summary', 'standup'] — answer=`Elias Roux owns the database backup task. Elise Roux owns the risk summary task. Distractor: Elise Roux and Elias Roux attended the same standup meeting.`
- no_memory_agent / latest_value_update / random_update_8_current_cache_eviction_policy: missing_any=['least recently used'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_8_current_cache_eviction_policy: forbidden=['random eviction', 'oldest-first'] — answer=`The current cache eviction policy is oldest-first. The current cache eviction policy is random eviction. The current cache eviction policy is least recently used.`
- multi_agent_titan_team / latest_value_update / random_update_8_current_cache_eviction_policy: forbidden=['random eviction', 'oldest-first'] — answer=`Manager final response Task: What is the current cache eviction policy right now Shared memory used: Selected shared Titan memories: 1. The current cache eviction policy is random eviction. 2. The current cache eviction policy is least recently used. 3. The current cache eviction policy is oldest-first. Agent plan: - MANAGER (Project manager and orchestrator...`
- no_memory_agent / forget_retention / random_forget_retention_5: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_5: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-4.`
- no_memory_agent / role_routing / random_role_5: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_5: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / forget_retention / random_forget_retention_3: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_3: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-2.`
- no_memory_agent / policy_recall / random_policy_1: missing_all=['Ollama', 'better', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / policy_recall / random_policy_1: missing_all=['better', 'free'] — answer=`For the current prototype, Ollama is the default LLM provider.`
- multi_agent_titan_team / policy_recall / random_policy_1: missing_all=['better'] — answer=`Manager final response Task: Which LLM provider rule should we keep for tests Shared memory used: Selected shared Titan memories: 1. For the current prototype, Ollama is the default LLM provider. [property=llm_provider] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVOPS (DevOps and env...`
- no_memory_agent / identity_collision / random_owner_16_b: missing_any=['Ivan Lemoine'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_16_b: forbidden=['Iris Lemoine', 'memory adapter', 'standup'] — answer=`Ivan Lemoine owns the documentation pass task. Iris Lemoine owns the memory adapter task. Distractor: Iris Lemoine and Ivan Lemoine attended the same standup meeting.`
- multi_agent_titan_team / identity_collision / random_owner_16_b: forbidden=['memory adapter'] — answer=`Manager final response Task: Who owns the documentation pass task Shared memory used: Selected shared Titan memories: 1. Ivan Lemoine owns the documentation pass task. [property=owner_task:documentation pass] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python develope...`
- no_memory_agent / role_routing / random_role_6: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_6: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / project_convention / random_convention_15_document: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_15_document: missing_all=['FastAPI', 'pytest'] — answer=`No relevant Titan memory found.`
- no_memory_agent / latest_value_update / random_update_5_current_testing_framework: missing_any=['pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_5_current_testing_framework: forbidden=['unittest', 'nose'] — answer=`The current testing framework is unittest. The current testing framework is nose. The current testing framework is pytest.`
- no_memory_agent / project_convention / random_convention_14_profile: missing_all=['REST', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / forget_retention / random_forget_retention_19: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_19: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-18.`
- no_memory_agent / consolidation / random_consolidation_11: missing_all=['profile', 'forget'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_11: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 10: preserve profile, forget, and consolidation.`
- no_memory_agent / consolidation / random_consolidation_8: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_8: missing_all=['Ollama', 'future'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / identity_collision / random_owner_19_a: missing_any=['Manon Leroy'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_19_a: forbidden=['Marius Leroy', 'benchmark report', 'standup'] — answer=`Manon Leroy owns the risk summary task. Marius Leroy owns the benchmark report task. Distractor: Manon Leroy and Marius Leroy attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_19_b: missing_any=['Marius Leroy'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_19_b: forbidden=['Manon Leroy', 'risk summary', 'standup'] — answer=`Marius Leroy owns the benchmark report task. Manon Leroy owns the risk summary task. Distractor: Manon Leroy and Marius Leroy attended the same standup meeting.`
- no_memory_agent / consolidation / random_consolidation_1: missing_all=['terminal', 'manager'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_1: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 0: preserve terminal, manager, and results.`
- no_memory_agent / noisy_query / random_noisy_18: missing_any=['QZ-4-017'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_18: forbidden=['coffee', 'capsule', 'orange', 'travel ad', 'beach', 'karaoke'] — answer=`Critical incident memory 17: the escalation key is QZ-4-017. Noise distractor 17: karaoke travel ad orange coffee beach capsule should be ignored.`
- no_memory_agent / forget_retention / random_forget_retention_14: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_14: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-13.`
- no_memory_agent / consolidation / random_consolidation_3: missing_all=['Tester', 'Evaluator'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_3: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 2: preserve Tester, Evaluator, and Manager.`
- no_memory_agent / latest_value_update / random_update_14_current_preferred_backend_language: missing_any=['Rust'] — answer=`I do not know because no shared memory is available.`
- multi_agent_titan_team / latest_value_update / random_update_14_current_preferred_backend_language: forbidden=['Python'] — answer=`Manager final response Task: What is the current preferred backend language right now Shared memory used: Selected shared Titan memories: 1. The current preferred backend language is Rust. [subject=global:preferred_backend_language; property=backend_language] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action:...`
- no_memory_agent / consolidation / random_consolidation_2: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_2: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 1: preserve Ollama, future, and provider.`
- no_memory_agent / latest_value_update / random_update_13_current_deployment_shell: missing_any=['PowerShell'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_13_current_deployment_shell: forbidden=['cmd', 'bash'] — answer=`The current deployment shell is bash. The current deployment shell is PowerShell. The current deployment shell is cmd.`
- no_memory_agent / project_convention / random_convention_16_support: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / role_routing / random_role_4: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_4: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / identity_collision / random_owner_1_b: missing_any=['Axel Durand'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_1_b: forbidden=['Aline Durand', 'provider evaluation', 'standup'] — answer=`Axel Durand owns the database backup task. Aline Durand owns the provider evaluation task. Distractor: Aline Durand and Axel Durand attended the same standup meeting.`
- no_memory_agent / project_convention / random_convention_9_export: missing_all=['typed responses', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_9_export: missing_all=['typed responses', 'pytest'] — answer=`No relevant Titan memory found.`
- no_memory_agent / role_routing / random_role_7: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_7: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / noisy_query / random_noisy_21: missing_any=['QZ-4-020'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_21: forbidden=['banana', 'coffee', 'router', 'capsule', 'orange', 'travel ad'] — answer=`Critical incident memory 20: the escalation key is QZ-4-020. Noise distractor 20: travel ad capsule orange coffee banana router should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_21: forbidden=['karaoke'] — answer=`Manager final response Task: Ignore karaoke what is the escalation key for incident 20 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 20: the escalation key is QZ-4-020. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, con...`
- no_memory_agent / noisy_query / random_noisy_2: missing_any=['QZ-4-001'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_2: forbidden=['router', 'sticker', 'capsule', 'orange', 'beach', 'karaoke'] — answer=`Noise distractor 1: router karaoke capsule orange sticker beach should be ignored. Critical incident memory 1: the escalation key is QZ-4-001.`
- multi_agent_titan_team / noisy_query / random_noisy_2: forbidden=['beach', 'karaoke'] — answer=`Manager final response Task: Ignore beach karaoke what is the escalation key for incident 1 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 1: the escalation key is QZ-4-001. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer,...`
- no_memory_agent / identity_collision / random_owner_7_b: missing_any=['Yanis Petit'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_7_b: forbidden=['Yuna Petit', 'route review', 'standup'] — answer=`Yanis Petit owns the deployment script task. Yuna Petit owns the route review task. Distractor: Yuna Petit and Yanis Petit attended the same standup meeting.`
- no_memory_agent / project_convention / random_convention_3_billing: missing_all=['FastAPI', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / policy_recall / random_policy_3: missing_all=['test', 'benchmark'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / identity_collision / random_owner_17_b: missing_any=['Yanis Petit'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_17_b: forbidden=['Yuna Petit', 'security audit', 'standup'] — answer=`Yanis Petit owns the monitoring setup task. Yuna Petit owns the security audit task. Distractor: Yuna Petit and Yanis Petit attended the same standup meeting.`
- no_memory_agent / latest_value_update / random_update_18_current_demo_ui_style: missing_any=['markdown only'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_18_current_demo_ui_style: forbidden=['colored terminal panels', 'plain text'] — answer=`The current demo UI style is markdown only. The current demo UI style is plain text. The current demo UI style is colored terminal panels.`
- no_memory_agent / noisy_query / random_noisy_3: missing_any=['QZ-4-002'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_3: forbidden=['banana', 'coffee', 'sticker', 'capsule', 'orange', 'travel ad'] — answer=`Noise distractor 2: capsule coffee sticker orange banana travel ad should be ignored. Critical incident memory 2: the escalation key is QZ-4-002.`
- no_memory_agent / forget_retention / random_forget_retention_20: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_20: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-19.`
- no_memory_agent / latest_value_update / random_update_10_current_demo_ui_style: missing_any=['colored terminal panels'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_10_current_demo_ui_style: forbidden=['plain text', 'markdown only'] — answer=`The current demo UI style is markdown only. The current demo UI style is colored terminal panels. The current demo UI style is plain text.`
- no_memory_agent / role_routing / random_role_1: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_1: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / role_routing / random_role_21: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
... 148 more failures omitted from console report.
