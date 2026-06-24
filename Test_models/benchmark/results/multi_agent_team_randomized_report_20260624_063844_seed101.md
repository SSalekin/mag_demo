# Randomized adversarial multi-agent Titan team comparison

============================================================================================
RANDOMIZED ADVERSARIAL MULTI-AGENT TITAN TEAM COMPARISON
============================================================================================
Cases: 150 | Seed: 101

GLOBAL RESULTS
--------------------------------------------------------------------------------------------
no_memory_agent          | 0/150 (0.0%)
single_titan_agent       | 17/150 (11.3%)
multi_agent_titan_team   | 126/150 (84.0%)

RESULTS BY CATEGORY
--------------------------------------------------------------------------------------------
no_memory_agent          | consolidation            | 0/20 (0.0%)
no_memory_agent          | forget_retention         | 0/20 (0.0%)
no_memory_agent          | identity_collision       | 0/37 (0.0%)
no_memory_agent          | latest_value_update      | 0/17 (0.0%)
no_memory_agent          | noisy_query              | 0/17 (0.0%)
no_memory_agent          | policy_recall            | 0/5 (0.0%)
no_memory_agent          | project_convention       | 0/16 (0.0%)
no_memory_agent          | role_routing             | 0/18 (0.0%)
single_titan_agent       | consolidation            | 0/20 (0.0%)
single_titan_agent       | forget_retention         | 0/20 (0.0%)
single_titan_agent       | identity_collision       | 0/37 (0.0%)
single_titan_agent       | latest_value_update      | 1/17 (5.9%)
single_titan_agent       | noisy_query              | 0/17 (0.0%)
single_titan_agent       | policy_recall            | 4/5 (80.0%)
single_titan_agent       | project_convention       | 12/16 (75.0%)
single_titan_agent       | role_routing             | 0/18 (0.0%)
multi_agent_titan_team   | consolidation            | 20/20 (100.0%)
multi_agent_titan_team   | forget_retention         | 20/20 (100.0%)
multi_agent_titan_team   | identity_collision       | 34/37 (91.9%)
multi_agent_titan_team   | latest_value_update      | 10/17 (58.8%)
multi_agent_titan_team   | noisy_query              | 4/17 (23.5%)
multi_agent_titan_team   | policy_recall            | 4/5 (80.0%)
multi_agent_titan_team   | project_convention       | 16/16 (100.0%)
multi_agent_titan_team   | role_routing             | 18/18 (100.0%)

FAILURES
--------------------------------------------------------------------------------------------
- no_memory_agent / consolidation / random_consolidation_7: missing_all=['terminal', 'manager'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_7: missing_all=['terminal', 'manager'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / consolidation / random_consolidation_6: missing_all=['large', 'stress'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_6: missing_all=['large', 'stress'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / project_convention / random_convention_8_fraud: missing_all=['RBAC', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / forget_retention / random_forget_retention_5: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_5: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-4.`
- no_memory_agent / role_routing / random_role_19: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_19: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / role_routing / random_role_16: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_16: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / policy_recall / random_policy_3: missing_all=['test', 'benchmark'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / noisy_query / random_noisy_8: missing_any=['QZ-1-007'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_8: forbidden=['banana', 'coffee', 'router', 'sticker', 'capsule', 'orange'] — answer=`Critical incident memory 7: the escalation key is QZ-1-007. Noise distractor 7: capsule orange sticker banana router coffee should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_8: forbidden=['beach', 'karaoke'] — answer=`Manager final response Task: Ignore karaoke beach what is the escalation key for incident 7 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 7: the escalation key is QZ-1-007. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer,...`
- no_memory_agent / consolidation / random_consolidation_18: missing_all=['large', 'stress'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_18: missing_all=['large', 'stress'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / project_convention / random_convention_20_export: missing_all=['REST', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / consolidation / random_consolidation_16: missing_all=['DevSoft', 'DevWeb'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_16: missing_all=['DevSoft', 'DevWeb'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / role_routing / random_role_2: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_2: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / identity_collision / random_owner_1_a: missing_any=['Aline Durand'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_1_a: forbidden=['Axel Durand', 'auth refactor', 'standup'] — answer=`Aline Durand owns the security audit task. Axel Durand owns the auth refactor task. Distractor: Aline Durand and Axel Durand attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_5_a: missing_any=['Nora Garnier'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_5_a: forbidden=['Nathan Garnier', 'frontend polish', 'standup'] — answer=`Nora Garnier owns the UI accessibility task. Nathan Garnier owns the frontend polish task. Distractor: Nora Garnier and Nathan Garnier attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_14_a: missing_any=['Lina Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_14_a: forbidden=['Leo Robert', 'database backup', 'standup'] — answer=`Lina Robert owns the route review task. Leo Robert owns the database backup task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- no_memory_agent / consolidation / random_consolidation_15: missing_all=['Tester', 'Evaluator'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_15: missing_all=['Tester', 'Evaluator'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / latest_value_update / random_update_9_current_preferred_backend_language: missing_any=['Python'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / noisy_query / random_noisy_13: missing_any=['QZ-1-012'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_13: forbidden=['hotel', 'router', 'sticker', 'orange', 'travel ad', 'beach'] — answer=`Noise distractor 12: travel ad router beach sticker orange hotel should be ignored. Critical incident memory 12: the escalation key is QZ-1-012.`
- multi_agent_titan_team / noisy_query / random_noisy_13: forbidden=['beach'] — answer=`Manager final response Task: Ignore beach what is the escalation key for incident 12 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 12: the escalation key is QZ-1-012. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confi...`
- no_memory_agent / noisy_query / random_noisy_6: missing_any=['QZ-1-005'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_6: forbidden=['banana', 'coffee', 'hotel', 'router', 'orange', 'beach'] — answer=`Critical incident memory 5: the escalation key is QZ-1-005. Noise distractor 5: orange coffee beach router banana hotel should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_6: forbidden=['karaoke'] — answer=`Manager final response Task: Ignore karaoke what is the escalation key for incident 5 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 5: the escalation key is QZ-1-005. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confi...`
- no_memory_agent / forget_retention / random_forget_retention_18: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_18: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-17.`
- no_memory_agent / identity_collision / random_owner_21_a: missing_any=['Aline Durand'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_21_a: forbidden=['Axel Durand', 'pricing rules', 'standup'] — answer=`Aline Durand owns the UI accessibility task. Axel Durand owns the pricing rules task. Distractor: Aline Durand and Axel Durand attended the same standup meeting.`
- no_memory_agent / latest_value_update / random_update_4_current_llm_provider_for_tests: missing_any=['Claude'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_4_current_llm_provider_for_tests: forbidden=['Ollama', 'Gemini'] — answer=`The current LLM provider for tests is Claude. The current LLM provider for tests is Gemini. The current LLM provider for tests is Ollama.`
- multi_agent_titan_team / latest_value_update / random_update_4_current_llm_provider_for_tests: forbidden=['Ollama'] — answer=`Manager final response Task: What is the current LLM provider for tests right now Shared memory used: Selected shared Titan memories: 1. The current LLM provider for tests is Claude. [property=llm_provider] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVOPS (DevOps and environment engi...`
- no_memory_agent / consolidation / random_consolidation_14: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_14: missing_all=['Ollama', 'future'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / role_routing / random_role_17: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_17: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / latest_value_update / random_update_19_current_testing_framework: missing_any=['nose'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_19_current_testing_framework: forbidden=['pytest', 'unittest'] — answer=`The current testing framework is nose. The current testing framework is unittest. The current testing framework is pytest.`
- no_memory_agent / forget_retention / random_forget_retention_1: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_1: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-0.`
- no_memory_agent / forget_retention / random_forget_retention_7: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_7: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-6.`
- no_memory_agent / forget_retention / random_forget_retention_2: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_2: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-1.`
- no_memory_agent / identity_collision / random_owner_17_a: missing_any=['Yuna Petit'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_17_a: forbidden=['Yanis Petit', 'regression suite', 'standup'] — answer=`Yuna Petit owns the risk summary task. Yanis Petit owns the regression suite task. Distractor: Yuna Petit and Yanis Petit attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_10: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_10: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-9.`
- no_memory_agent / project_convention / random_convention_1_search: missing_all=['API route', 'integration tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / latest_value_update / random_update_11_current_queue_backend: missing_any=['Kafka'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_11_current_queue_backend: forbidden=['Redis streams', 'RabbitMQ'] — answer=`The current queue backend is RabbitMQ. The current queue backend is Kafka. The current queue backend is Redis streams.`
- multi_agent_titan_team / latest_value_update / random_update_11_current_queue_backend: forbidden=['Redis streams', 'RabbitMQ'] — answer=`Manager final response Task: What is the current queue backend right now Shared memory used: Selected shared Titan memories: 1. The current queue backend is RabbitMQ. 2. The current queue backend is Redis streams. 3. The current queue backend is Kafka. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_...`
- no_memory_agent / forget_retention / random_forget_retention_13: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_13: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-12.`
- no_memory_agent / identity_collision / random_owner_10_a: missing_any=['Camille Masson'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_10_a: forbidden=['Corentin Masson', 'pricing rules', 'standup'] — answer=`Camille Masson owns the release checklist task. Corentin Masson owns the pricing rules task. Distractor: Camille Masson and Corentin Masson attended the same standup meeting.`
- no_memory_agent / noisy_query / random_noisy_17: missing_any=['QZ-1-016'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_17: forbidden=['banana', 'coffee', 'sticker', 'capsule', 'orange', 'beach'] — answer=`Critical incident memory 16: the escalation key is QZ-1-016. Noise distractor 16: capsule sticker coffee banana orange beach should be ignored.`
- no_memory_agent / identity_collision / random_owner_14_b: missing_any=['Leo Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_14_b: forbidden=['Lina Robert', 'route review', 'standup'] — answer=`Leo Robert owns the database backup task. Lina Robert owns the route review task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_18_b: missing_any=['Elias Roux'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_18_b: forbidden=['Elise Roux', 'UI accessibility', 'standup'] — answer=`Elias Roux owns the pricing rules task. Elise Roux owns the UI accessibility task. Distractor: Elise Roux and Elias Roux attended the same standup meeting.`
- no_memory_agent / consolidation / random_consolidation_12: missing_all=['large', 'stress'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_12: missing_all=['large', 'stress'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / noisy_query / random_noisy_5: missing_any=['QZ-1-004'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_5: forbidden=['banana', 'coffee', 'sticker', 'capsule', 'orange', 'beach'] — answer=`Critical incident memory 4: the escalation key is QZ-1-004. Noise distractor 4: sticker capsule banana beach coffee orange should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_5: forbidden=['beach', 'karaoke'] — answer=`Manager final response Task: Ignore karaoke beach what is the escalation key for incident 4 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 4: the escalation key is QZ-1-004. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer,...`
- no_memory_agent / role_routing / random_role_21: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_21: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / project_convention / random_convention_13_analytics: missing_all=['RBAC', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / forget_retention / random_forget_retention_12: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_12: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-11.`
- no_memory_agent / identity_collision / random_owner_15_b: missing_any=['Nathan Garnier'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_15_b: forbidden=['Nora Garnier', 'memory adapter', 'standup'] — answer=`Nathan Garnier owns the pricing rules task. Nora Garnier owns the memory adapter task. Distractor: Nora Garnier and Nathan Garnier attended the same standup meeting.`
- multi_agent_titan_team / identity_collision / random_owner_15_b: forbidden=['memory adapter'] — answer=`Manager final response Task: Who owns the pricing rules task Shared memory used: Selected shared Titan memories: 1. Nathan Garnier owns the pricing rules task. [property=owner_task:pricing rules] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confidence...`
- no_memory_agent / forget_retention / random_forget_retention_20: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_20: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-19.`
- no_memory_agent / identity_collision / random_owner_12_a: missing_any=['Clara Moreau'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_12_a: forbidden=['Clement Moreau', 'database backup', 'standup'] — answer=`Clara Moreau owns the memory adapter task. Clement Moreau owns the database backup task. Distractor: Clara Moreau and Clement Moreau attended the same standup meeting.`
- no_memory_agent / role_routing / random_role_15: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_15: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / noisy_query / random_noisy_10: missing_any=['QZ-1-009'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_10: forbidden=['banana', 'hotel', 'sticker', 'capsule', 'travel ad', 'karaoke'] — answer=`Critical incident memory 9: the escalation key is QZ-1-009. Noise distractor 9: banana karaoke hotel travel ad sticker capsule should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_10: forbidden=['karaoke'] — answer=`Manager final response Task: Ignore karaoke what is the escalation key for incident 9 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 9: the escalation key is QZ-1-009. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVWEB (Web/API developer, confidence=0.8...`
- no_memory_agent / forget_retention / random_forget_retention_6: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_6: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-5.`
- no_memory_agent / identity_collision / random_owner_8_b: missing_any=['Elias Roux'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_8_b: forbidden=['Elise Roux', 'memory adapter', 'standup'] — answer=`Elias Roux owns the frontend polish task. Elise Roux owns the memory adapter task. Distractor: Elise Roux and Elias Roux attended the same standup meeting.`
- no_memory_agent / role_routing / random_role_10: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_10: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / policy_recall / random_policy_4: missing_all=['colored', 'panels'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / identity_collision / random_owner_7_b: missing_any=['Yanis Petit'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_7_b: forbidden=['Yuna Petit', 'risk summary', 'standup'] — answer=`Yanis Petit owns the deployment script task. Yuna Petit owns the risk summary task. Distractor: Yuna Petit and Yanis Petit attended the same standup meeting.`
- no_memory_agent / project_convention / random_convention_14_notification: missing_all=['RBAC', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_14_notification: missing_all=['RBAC', 'unit tests'] — answer=`No relevant Titan memory found.`
- no_memory_agent / identity_collision / random_owner_4_a: missing_any=['Lina Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_4_a: forbidden=['Leo Robert', 'monitoring setup', 'standup'] — answer=`Lina Robert owns the security audit task. Leo Robert owns the monitoring setup task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_4_b: missing_any=['Leo Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_4_b: forbidden=['Lina Robert', 'security audit', 'standup'] — answer=`Leo Robert owns the monitoring setup task. Lina Robert owns the security audit task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- no_memory_agent / consolidation / random_consolidation_20: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_20: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 19: preserve Ollama, future, and provider.`
- no_memory_agent / identity_collision / random_owner_2_b: missing_any=['Clement Moreau'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_2_b: forbidden=['Clara Moreau', 'route review', 'standup'] — answer=`Clement Moreau owns the benchmark report task. Clara Moreau owns the route review task. Distractor: Clara Moreau and Clement Moreau attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_11: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_11: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-10.`
- no_memory_agent / identity_collision / random_owner_7_a: missing_any=['Yuna Petit'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_7_a: forbidden=['Yanis Petit', 'deployment script', 'standup'] — answer=`Yuna Petit owns the risk summary task. Yanis Petit owns the deployment script task. Distractor: Yuna Petit and Yanis Petit attended the same standup meeting.`
- no_memory_agent / consolidation / random_consolidation_8: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_8: missing_all=['Ollama', 'future'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / identity_collision / random_owner_15_a: missing_any=['Nora Garnier'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_15_a: forbidden=['Nathan Garnier', 'pricing rules', 'standup'] — answer=`Nora Garnier owns the memory adapter task. Nathan Garnier owns the pricing rules task. Distractor: Nora Garnier and Nathan Garnier attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_9_b: missing_any=['Marius Leroy'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_9_b: forbidden=['Manon Leroy', 'release checklist', 'standup'] — answer=`Marius Leroy owns the deployment script task. Manon Leroy owns the release checklist task. Distractor: Manon Leroy and Marius Leroy attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_15: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_15: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-14.`
- no_memory_agent / identity_collision / random_owner_11_b: missing_any=['Axel Durand'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_11_b: forbidden=['Aline Durand', 'provider evaluation', 'standup'] — answer=`Axel Durand owns the regression suite task. Aline Durand owns the provider evaluation task. Distractor: Aline Durand and Axel Durand attended the same standup meeting.`
- no_memory_agent / consolidation / random_consolidation_11: missing_all=['profile', 'forget'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_11: missing_all=['profile', 'forget'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / noisy_query / random_noisy_18: missing_any=['QZ-1-017'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_18: forbidden=['hotel', 'router', 'sticker', 'capsule', 'travel ad', 'beach'] — answer=`Critical incident memory 17: the escalation key is QZ-1-017. Noise distractor 17: sticker travel ad beach hotel capsule router should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_18: forbidden=['beach'] — answer=`Manager final response Task: Ignore beach what is the escalation key for incident 17 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 17: the escalation key is QZ-1-017. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confi...`
- no_memory_agent / policy_recall / random_policy_5: missing_all=['manager', 'shared Titan memory'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / policy_recall / random_policy_2: missing_all=['manager', 'Titan memory'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / project_convention / random_convention_5_document: missing_all=['REST', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / latest_value_update / random_update_10_current_api_auth_method: missing_any=['API key'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_10_current_api_auth_method: forbidden=['JWT', 'basic auth'] — answer=`The current API auth method is API key. The current API auth method is JWT. The current API auth method is basic auth.`
- no_memory_agent / consolidation / random_consolidation_17: missing_all=['profile', 'forget'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_17: missing_all=['profile', 'forget'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / identity_collision / random_owner_5_b: missing_any=['Nathan Garnier'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_5_b: forbidden=['Nora Garnier', 'UI accessibility', 'standup'] — answer=`Nathan Garnier owns the frontend polish task. Nora Garnier owns the UI accessibility task. Distractor: Nora Garnier and Nathan Garnier attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_21_b: missing_any=['Axel Durand'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_21_b: forbidden=['Aline Durand', 'UI accessibility', 'standup'] — answer=`Axel Durand owns the pricing rules task. Aline Durand owns the UI accessibility task. Distractor: Aline Durand and Axel Durand attended the same standup meeting.`
- no_memory_agent / role_routing / random_role_11: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_11: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / latest_value_update / random_update_12_current_queue_backend: missing_any=['Kafka'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_12_current_queue_backend: forbidden=['RabbitMQ', 'Redis streams'] — answer=`The current queue backend is Kafka. The current queue backend is Redis streams. The current queue backend is RabbitMQ.`
- multi_agent_titan_team / latest_value_update / random_update_12_current_queue_backend: forbidden=['RabbitMQ', 'Redis streams'] — answer=`Manager final response Task: What is the current queue backend right now Shared memory used: Selected shared Titan memories: 1. The current queue backend is Kafka. 2. The current queue backend is Redis streams. 3. The current queue backend is RabbitMQ. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_...`
- no_memory_agent / role_routing / random_role_12: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_12: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / identity_collision / random_owner_19_a: missing_any=['Manon Leroy'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_19_a: forbidden=['Marius Leroy', 'regression suite', 'standup'] — answer=`Manon Leroy owns the load testing task. Marius Leroy owns the regression suite task. Distractor: Manon Leroy and Marius Leroy attended the same standup meeting.`
- no_memory_agent / role_routing / random_role_1: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_1: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / role_routing / random_role_9: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_9: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / role_routing / random_role_3: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_3: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
... 147 more failures omitted from console report.
