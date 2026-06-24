# Randomized adversarial multi-agent Titan team comparison

============================================================================================
RANDOMIZED ADVERSARIAL MULTI-AGENT TITAN TEAM COMPARISON
============================================================================================
Cases: 150 | Seed: 202

GLOBAL RESULTS
--------------------------------------------------------------------------------------------
no_memory_agent          | 0/150 (0.0%)
single_titan_agent       | 18/150 (12.0%)
multi_agent_titan_team   | 121/150 (80.7%)

RESULTS BY CATEGORY
--------------------------------------------------------------------------------------------
no_memory_agent          | consolidation            | 0/15 (0.0%)
no_memory_agent          | forget_retention         | 0/19 (0.0%)
no_memory_agent          | identity_collision       | 0/38 (0.0%)
no_memory_agent          | latest_value_update      | 0/19 (0.0%)
no_memory_agent          | noisy_query              | 0/17 (0.0%)
no_memory_agent          | policy_recall            | 0/4 (0.0%)
no_memory_agent          | project_convention       | 0/18 (0.0%)
no_memory_agent          | role_routing             | 0/20 (0.0%)
single_titan_agent       | consolidation            | 0/15 (0.0%)
single_titan_agent       | forget_retention         | 0/19 (0.0%)
single_titan_agent       | identity_collision       | 0/38 (0.0%)
single_titan_agent       | latest_value_update      | 5/19 (26.3%)
single_titan_agent       | noisy_query              | 0/17 (0.0%)
single_titan_agent       | policy_recall            | 3/4 (75.0%)
single_titan_agent       | project_convention       | 10/18 (55.6%)
single_titan_agent       | role_routing             | 0/20 (0.0%)
multi_agent_titan_team   | consolidation            | 15/15 (100.0%)
multi_agent_titan_team   | forget_retention         | 19/19 (100.0%)
multi_agent_titan_team   | identity_collision       | 35/38 (92.1%)
multi_agent_titan_team   | latest_value_update      | 9/19 (47.4%)
multi_agent_titan_team   | noisy_query              | 2/17 (11.8%)
multi_agent_titan_team   | policy_recall            | 3/4 (75.0%)
multi_agent_titan_team   | project_convention       | 18/18 (100.0%)
multi_agent_titan_team   | role_routing             | 20/20 (100.0%)

FAILURES
--------------------------------------------------------------------------------------------
- no_memory_agent / identity_collision / random_owner_2_b: missing_any=['Clement Moreau'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_2_b: forbidden=['Clara Moreau', 'security audit', 'standup'] — answer=`Clement Moreau owns the documentation pass task. Clara Moreau owns the security audit task. Distractor: Clara Moreau and Clement Moreau attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_6_b: missing_any=['Ivan Lemoine'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_6_b: forbidden=['Iris Lemoine', 'route review', 'standup'] — answer=`Ivan Lemoine owns the deployment script task. Iris Lemoine owns the route review task. Distractor: Iris Lemoine and Ivan Lemoine attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_6: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_6: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-5.`
- no_memory_agent / consolidation / random_consolidation_15: missing_all=['Tester', 'Evaluator'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_15: missing_all=['Tester', 'Evaluator'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / project_convention / random_convention_7_calendar: missing_all=['typed responses', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / latest_value_update / random_update_11_current_deployment_shell: missing_any=['bash'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_11_current_deployment_shell: forbidden=['cmd', 'PowerShell'] — answer=`The current deployment shell is PowerShell. The current deployment shell is bash. The current deployment shell is cmd.`
- no_memory_agent / identity_collision / random_owner_15_b: missing_any=['Nathan Garnier'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_15_b: forbidden=['Nora Garnier', 'schema migration', 'standup'] — answer=`Nathan Garnier owns the pricing rules task. Nora Garnier owns the schema migration task. Distractor: Nora Garnier and Nathan Garnier attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_13_b: missing_any=['Mathis Bernard'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_13_b: forbidden=['Mila Bernard', 'route review', 'standup'] — answer=`Mathis Bernard owns the frontend polish task. Mila Bernard owns the route review task. Distractor: Mila Bernard and Mathis Bernard attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_14_a: missing_any=['Lina Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_14_a: forbidden=['Leo Robert', 'auth refactor', 'standup'] — answer=`Lina Robert owns the memory adapter task. Leo Robert owns the auth refactor task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- no_memory_agent / project_convention / random_convention_5_search: missing_all=['typed responses', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / role_routing / random_role_5: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_5: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / identity_collision / random_owner_16_b: missing_any=['Ivan Lemoine'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_16_b: forbidden=['Iris Lemoine', 'UI accessibility', 'standup'] — answer=`Ivan Lemoine owns the deployment script task. Iris Lemoine owns the UI accessibility task. Distractor: Iris Lemoine and Ivan Lemoine attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_15: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_15: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-14.`
- no_memory_agent / project_convention / random_convention_16_document: missing_all=['RBAC', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / forget_retention / random_forget_retention_8: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_8: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-7.`
- no_memory_agent / latest_value_update / random_update_12_current_demo_ui_style: missing_any=['markdown only'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_12_current_demo_ui_style: forbidden=['plain text', 'colored terminal panels'] — answer=`The current demo UI style is plain text. The current demo UI style is markdown only. The current demo UI style is colored terminal panels.`
- no_memory_agent / forget_retention / random_forget_retention_9: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_9: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-8.`
- no_memory_agent / project_convention / random_convention_17_document: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_17_document: missing_all=['FastAPI', 'pytest'] — answer=`No relevant Titan memory found.`
- no_memory_agent / identity_collision / random_owner_10_a: missing_any=['Camille Masson'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_10_a: forbidden=['Corentin Masson', 'documentation pass', 'standup'] — answer=`Camille Masson owns the memory adapter task. Corentin Masson owns the documentation pass task. Distractor: Camille Masson and Corentin Masson attended the same standup meeting.`
- no_memory_agent / consolidation / random_consolidation_19: missing_all=['terminal', 'manager'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_19: missing_all=['terminal', 'manager'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / consolidation / random_consolidation_2: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_2: missing_all=['Ollama', 'future'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / noisy_query / random_noisy_14: missing_any=['QZ-2-013'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_14: forbidden=['banana', 'coffee', 'hotel', 'router', 'sticker', 'capsule'] — answer=`Critical incident memory 13: the escalation key is QZ-2-013. Noise distractor 13: coffee capsule sticker hotel router banana should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_14: forbidden=['beach'] — answer=`Manager final response Task: Ignore beach what is the escalation key for incident 13 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 13: the escalation key is QZ-2-013. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVWEB (Web/API developer, confidence=0.8...`
- no_memory_agent / role_routing / random_role_1: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_1: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / forget_retention / random_forget_retention_10: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_10: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-9.`
- no_memory_agent / forget_retention / random_forget_retention_12: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_12: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-11.`
- no_memory_agent / role_routing / random_role_10: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_10: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / role_routing / random_role_9: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_9: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / noisy_query / random_noisy_3: missing_any=['QZ-2-002'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_3: forbidden=['coffee', 'router', 'sticker', 'capsule', 'beach', 'karaoke'] — answer=`Critical incident memory 2: the escalation key is QZ-2-002. Noise distractor 2: beach router sticker karaoke coffee capsule should be ignored.`
- no_memory_agent / role_routing / random_role_14: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_14: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / consolidation / random_consolidation_21: missing_all=['Tester', 'Evaluator'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_21: missing_all=['Tester', 'Evaluator'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / latest_value_update / random_update_9_current_queue_backend: missing_any=['Redis streams'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_9_current_queue_backend: forbidden=['Kafka', 'RabbitMQ'] — answer=`The current queue backend is RabbitMQ. The current queue backend is Kafka. The current queue backend is Redis streams.`
- multi_agent_titan_team / latest_value_update / random_update_9_current_queue_backend: forbidden=['Kafka', 'RabbitMQ'] — answer=`Manager final response Task: What is the current queue backend right now Shared memory used: Selected shared Titan memories: 1. The current queue backend is RabbitMQ. 2. The current queue backend is Kafka. 3. The current queue backend is Redis streams. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_...`
- no_memory_agent / latest_value_update / random_update_13_current_preferred_backend_language: missing_any=['Python'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / project_convention / random_convention_18_billing: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_18_billing: missing_all=['FastAPI', 'pytest'] — answer=`No relevant Titan memory found.`
- no_memory_agent / latest_value_update / random_update_17_current_llm_provider_for_tests: missing_any=['Gemini'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_17_current_llm_provider_for_tests: forbidden=['Ollama', 'Claude'] — answer=`The current LLM provider for tests is Claude. The current LLM provider for tests is Ollama. The current LLM provider for tests is Gemini.`
- multi_agent_titan_team / latest_value_update / random_update_17_current_llm_provider_for_tests: forbidden=['Ollama'] — answer=`Manager final response Task: What is the current LLM provider for tests right now Shared memory used: Selected shared Titan memories: 1. The current LLM provider for tests is Gemini. [property=llm_provider] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVOPS (DevOps and environment engi...`
- no_memory_agent / noisy_query / random_noisy_17: missing_any=['QZ-2-016'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_17: forbidden=['coffee', 'hotel', 'capsule', 'orange', 'beach', 'karaoke'] — answer=`Critical incident memory 16: the escalation key is QZ-2-016. Noise distractor 16: beach capsule karaoke hotel coffee orange should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_17: forbidden=['karaoke'] — answer=`Manager final response Task: Ignore karaoke what is the escalation key for incident 16 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 16: the escalation key is QZ-2-016. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVWEB (Web/API developer, confidence=0...`
- no_memory_agent / identity_collision / random_owner_11_b: missing_any=['Axel Durand'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_11_b: forbidden=['Aline Durand', 'security audit', 'standup'] — answer=`Axel Durand owns the auth refactor task. Aline Durand owns the security audit task. Distractor: Aline Durand and Axel Durand attended the same standup meeting.`
- no_memory_agent / latest_value_update / random_update_10_current_benchmark_output_format: missing_any=['text only'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_10_current_benchmark_output_format: forbidden=['CSV and markdown report', 'screenshots only'] — answer=`The current benchmark output format is screenshots only. The current benchmark output format is text only. The current benchmark output format is CSV and markdown report.`
- no_memory_agent / noisy_query / random_noisy_9: missing_any=['QZ-2-008'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_9: forbidden=['banana', 'coffee', 'router', 'orange', 'beach', 'karaoke'] — answer=`Critical incident memory 8: the escalation key is QZ-2-008. Noise distractor 8: router beach karaoke orange banana coffee should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_9: forbidden=['karaoke'] — answer=`Manager final response Task: Ignore karaoke what is the escalation key for incident 8 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 8: the escalation key is QZ-2-008. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confi...`
- no_memory_agent / consolidation / random_consolidation_5: missing_all=['profile', 'forget'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_5: missing_all=['profile', 'forget'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / latest_value_update / random_update_19_current_llm_provider_for_tests: missing_any=['Ollama'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_19_current_llm_provider_for_tests: forbidden=['Claude', 'Gemini'] — answer=`The current LLM provider for tests is Claude. The current LLM provider for tests is Ollama. The current LLM provider for tests is Gemini.`
- no_memory_agent / forget_retention / random_forget_retention_21: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_21: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-20.`
- no_memory_agent / consolidation / random_consolidation_16: missing_all=['DevSoft', 'DevWeb'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_16: missing_all=['DevSoft', 'DevWeb'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / forget_retention / random_forget_retention_19: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_19: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-18.`
- no_memory_agent / identity_collision / random_owner_19_b: missing_any=['Marius Leroy'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_19_b: forbidden=['Manon Leroy', 'memory adapter', 'standup'] — answer=`Marius Leroy owns the benchmark report task. Manon Leroy owns the memory adapter task. Distractor: Manon Leroy and Marius Leroy attended the same standup meeting.`
- no_memory_agent / role_routing / random_role_12: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_12: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / latest_value_update / random_update_2_current_testing_framework: missing_any=['nose'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_2_current_testing_framework: forbidden=['pytest', 'unittest'] — answer=`The current testing framework is pytest. The current testing framework is unittest. The current testing framework is nose.`
- no_memory_agent / consolidation / random_consolidation_14: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_14: missing_all=['Ollama', 'future'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / noisy_query / random_noisy_18: missing_any=['QZ-2-017'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_18: forbidden=['banana', 'coffee', 'hotel', 'router', 'sticker', 'capsule'] — answer=`Critical incident memory 17: the escalation key is QZ-2-017. Noise distractor 17: banana hotel capsule router sticker coffee should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_18: forbidden=['beach'] — answer=`Manager final response Task: Ignore beach what is the escalation key for incident 17 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 17: the escalation key is QZ-2-017. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confi...`
- no_memory_agent / policy_recall / random_policy_2: missing_all=['manager', 'Titan memory'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / identity_collision / random_owner_6_a: missing_any=['Iris Lemoine'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_6_a: forbidden=['Ivan Lemoine', 'deployment script', 'standup'] — answer=`Iris Lemoine owns the route review task. Ivan Lemoine owns the deployment script task. Distractor: Iris Lemoine and Ivan Lemoine attended the same standup meeting.`
- no_memory_agent / noisy_query / random_noisy_7: missing_any=['QZ-2-006'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_7: forbidden=['coffee', 'router', 'sticker', 'travel ad', 'beach', 'karaoke'] — answer=`Noise distractor 6: beach karaoke router coffee travel ad sticker should be ignored. Critical incident memory 6: the escalation key is QZ-2-006.`
- multi_agent_titan_team / noisy_query / random_noisy_7: forbidden=['beach', 'karaoke'] — answer=`Manager final response Task: Ignore karaoke beach what is the escalation key for incident 6 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 6: the escalation key is QZ-2-006. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer,...`
- no_memory_agent / role_routing / random_role_8: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_8: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / identity_collision / random_owner_17_b: missing_any=['Yanis Petit'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_17_b: forbidden=['Yuna Petit', 'provider evaluation', 'standup'] — answer=`Yanis Petit owns the pricing rules task. Yuna Petit owns the provider evaluation task. Distractor: Yuna Petit and Yanis Petit attended the same standup meeting.`
- no_memory_agent / policy_recall / random_policy_1: missing_all=['Ollama', 'better', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / policy_recall / random_policy_1: missing_all=['better', 'free'] — answer=`For the current prototype, Ollama is the default LLM provider.`
- multi_agent_titan_team / policy_recall / random_policy_1: missing_all=['better'] — answer=`Manager final response Task: Which LLM provider rule should we keep for tests Shared memory used: Selected shared Titan memories: 1. For the current prototype, Ollama is the default LLM provider. [property=llm_provider] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVOPS (DevOps and env...`
- no_memory_agent / identity_collision / random_owner_7_a: missing_any=['Yuna Petit'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_7_a: forbidden=['Yanis Petit', 'pricing rules', 'standup'] — answer=`Yuna Petit owns the schema migration task. Yanis Petit owns the pricing rules task. Distractor: Yuna Petit and Yanis Petit attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_19_a: missing_any=['Manon Leroy'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_19_a: forbidden=['Marius Leroy', 'benchmark report', 'standup'] — answer=`Manon Leroy owns the memory adapter task. Marius Leroy owns the benchmark report task. Distractor: Manon Leroy and Marius Leroy attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_4_a: missing_any=['Lina Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_4_a: forbidden=['Leo Robert', 'monitoring setup', 'standup'] — answer=`Lina Robert owns the memory adapter task. Leo Robert owns the monitoring setup task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- no_memory_agent / role_routing / random_role_20: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_20: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / identity_collision / random_owner_20_a: missing_any=['Camille Masson'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_20_a: forbidden=['Corentin Masson', 'pricing rules', 'standup'] — answer=`Camille Masson owns the release checklist task. Corentin Masson owns the pricing rules task. Distractor: Camille Masson and Corentin Masson attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_3_b: missing_any=['Mathis Bernard'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_3_b: forbidden=['Mila Bernard', 'risk summary', 'standup'] — answer=`Mathis Bernard owns the pricing rules task. Mila Bernard owns the risk summary task. Distractor: Mila Bernard and Mathis Bernard attended the same standup meeting.`
- no_memory_agent / consolidation / random_consolidation_7: missing_all=['terminal', 'manager'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_7: missing_all=['terminal', 'manager'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / role_routing / random_role_16: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_16: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / noisy_query / random_noisy_15: missing_any=['QZ-2-014'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_15: forbidden=['banana', 'coffee', 'router', 'capsule', 'orange', 'beach'] — answer=`Critical incident memory 14: the escalation key is QZ-2-014. Noise distractor 14: coffee banana orange capsule router beach should be ignored.`
- no_memory_agent / identity_collision / random_owner_8_b: missing_any=['Elias Roux'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_8_b: forbidden=['Elise Roux', 'security audit', 'standup'] — answer=`Elias Roux owns the auth refactor task. Elise Roux owns the security audit task. Distractor: Elise Roux and Elias Roux attended the same standup meeting.`
- no_memory_agent / project_convention / random_convention_1_pricing: missing_all=['FastAPI', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / latest_value_update / random_update_14_current_queue_backend: missing_any=['Kafka'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_14_current_queue_backend: forbidden=['RabbitMQ', 'Redis streams'] — answer=`The current queue backend is Kafka. The current queue backend is RabbitMQ. The current queue backend is Redis streams.`
- multi_agent_titan_team / latest_value_update / random_update_14_current_queue_backend: forbidden=['RabbitMQ', 'Redis streams'] — answer=`Manager final response Task: What is the current queue backend right now Shared memory used: Selected shared Titan memories: 1. The current queue backend is RabbitMQ. 2. The current queue backend is Kafka. 3. The current queue backend is Redis streams. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_...`
- no_memory_agent / noisy_query / random_noisy_6: missing_any=['QZ-2-005'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_6: forbidden=['banana', 'coffee', 'hotel', 'sticker', 'travel ad', 'karaoke'] — answer=`Critical incident memory 5: the escalation key is QZ-2-005. Noise distractor 5: sticker travel ad hotel banana coffee karaoke should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_6: forbidden=['beach', 'karaoke'] — answer=`Manager final response Task: Ignore karaoke beach what is the escalation key for incident 5 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 5: the escalation key is QZ-2-005. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVWEB (Web/API developer, confiden...`
- no_memory_agent / identity_collision / random_owner_4_b: missing_any=['Leo Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_4_b: forbidden=['Lina Robert', 'memory adapter', 'standup'] — answer=`Leo Robert owns the monitoring setup task. Lina Robert owns the memory adapter task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- multi_agent_titan_team / identity_collision / random_owner_4_b: forbidden=['memory adapter'] — answer=`Manager final response Task: Who owns the monitoring setup task Shared memory used: Selected shared Titan memories: 1. Leo Robert owns the monitoring setup task. [property=owner_task:monitoring setup] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confi...`
- no_memory_agent / identity_collision / random_owner_14_b: missing_any=['Leo Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_14_b: forbidden=['Lina Robert', 'memory adapter', 'standup'] — answer=`Leo Robert owns the auth refactor task. Lina Robert owns the memory adapter task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- multi_agent_titan_team / identity_collision / random_owner_14_b: forbidden=['memory adapter'] — answer=`Manager final response Task: Who owns the auth refactor task Shared memory used: Selected shared Titan memories: 1. Leo Robert owns the auth refactor task. [property=owner_task:auth refactor] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confidence=0.8...`
- no_memory_agent / identity_collision / random_owner_21_b: missing_any=['Axel Durand'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_21_b: forbidden=['Aline Durand', 'security audit', 'standup'] — answer=`Axel Durand owns the deployment script task. Aline Durand owns the security audit task. Distractor: Aline Durand and Axel Durand attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_5: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_5: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-4.`
- no_memory_agent / project_convention / random_convention_13_search: missing_all=['webhooks', 'regression tests'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_13_search: missing_all=['webhooks', 'regression tests'] — answer=`No relevant Titan memory found.`
- no_memory_agent / consolidation / random_consolidation_4: missing_all=['DevSoft', 'DevWeb'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_4: missing_all=['DevSoft', 'DevWeb'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / role_routing / random_role_4: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_4: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / latest_value_update / random_update_21_current_preferred_backend_language: missing_any=['Python'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / latest_value_update / random_update_16_current_cache_eviction_policy: missing_any=['oldest-first'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_16_current_cache_eviction_policy: forbidden=['least recently used', 'random eviction'] — answer=`The current cache eviction policy is random eviction. The current cache eviction policy is oldest-first. The current cache eviction policy is least recently used.`
- multi_agent_titan_team / latest_value_update / random_update_16_current_cache_eviction_policy: forbidden=['least recently used', 'random eviction'] — answer=`Manager final response Task: What is the current cache eviction policy right now Shared memory used: Selected shared Titan memories: 1. The current cache eviction policy is oldest-first. 2. The current cache eviction policy is random eviction. 3. The current cache eviction policy is least recently used. Agent plan: - MANAGER (Project manager and orchestrator...`
- no_memory_agent / forget_retention / random_forget_retention_20: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_20: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-19.`
... 151 more failures omitted from console report.
