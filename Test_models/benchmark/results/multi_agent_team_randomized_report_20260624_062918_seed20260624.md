# Randomized adversarial multi-agent Titan team comparison

============================================================================================
RANDOMIZED ADVERSARIAL MULTI-AGENT TITAN TEAM COMPARISON
============================================================================================
Cases: 150 | Seed: 20260624

GLOBAL RESULTS
--------------------------------------------------------------------------------------------
no_memory_agent          | 0/150 (0.0%)
single_titan_agent       | 15/150 (10.0%)
multi_agent_titan_team   | 135/150 (90.0%)

RESULTS BY CATEGORY
--------------------------------------------------------------------------------------------
no_memory_agent          | consolidation            | 0/16 (0.0%)
no_memory_agent          | forget_retention         | 0/17 (0.0%)
no_memory_agent          | identity_collision       | 0/39 (0.0%)
no_memory_agent          | latest_value_update      | 0/18 (0.0%)
no_memory_agent          | noisy_query              | 0/17 (0.0%)
no_memory_agent          | policy_recall            | 0/5 (0.0%)
no_memory_agent          | project_convention       | 0/18 (0.0%)
no_memory_agent          | role_routing             | 0/20 (0.0%)
single_titan_agent       | consolidation            | 0/16 (0.0%)
single_titan_agent       | forget_retention         | 0/17 (0.0%)
single_titan_agent       | identity_collision       | 0/39 (0.0%)
single_titan_agent       | latest_value_update      | 0/18 (0.0%)
single_titan_agent       | noisy_query              | 0/17 (0.0%)
single_titan_agent       | policy_recall            | 4/5 (80.0%)
single_titan_agent       | project_convention       | 11/18 (61.1%)
single_titan_agent       | role_routing             | 0/20 (0.0%)
multi_agent_titan_team   | consolidation            | 16/16 (100.0%)
multi_agent_titan_team   | forget_retention         | 17/17 (100.0%)
multi_agent_titan_team   | identity_collision       | 39/39 (100.0%)
multi_agent_titan_team   | latest_value_update      | 12/18 (66.7%)
multi_agent_titan_team   | noisy_query              | 9/17 (52.9%)
multi_agent_titan_team   | policy_recall            | 4/5 (80.0%)
multi_agent_titan_team   | project_convention       | 18/18 (100.0%)
multi_agent_titan_team   | role_routing             | 20/20 (100.0%)

FAILURES
--------------------------------------------------------------------------------------------
- no_memory_agent / forget_retention / random_forget_retention_8: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_8: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-7.`
- no_memory_agent / latest_value_update / random_update_18_current_testing_framework: missing_any=['unittest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_18_current_testing_framework: forbidden=['nose', 'pytest'] — answer=`The current testing framework is nose. The current testing framework is unittest. The current testing framework is pytest.`
- no_memory_agent / forget_retention / random_forget_retention_6: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_6: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-5.`
- no_memory_agent / latest_value_update / random_update_8_current_benchmark_output_format: missing_any=['text only'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_8_current_benchmark_output_format: forbidden=['screenshots only', 'CSV and markdown report'] — answer=`The current benchmark output format is screenshots only. The current benchmark output format is text only. The current benchmark output format is CSV and markdown report.`
- no_memory_agent / latest_value_update / random_update_7_current_llm_provider_for_tests: missing_any=['Claude'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_7_current_llm_provider_for_tests: forbidden=['Gemini', 'Ollama'] — answer=`The current LLM provider for tests is Gemini. The current LLM provider for tests is Claude. The current LLM provider for tests is Ollama.`
- multi_agent_titan_team / latest_value_update / random_update_7_current_llm_provider_for_tests: forbidden=['Ollama'] — answer=`Manager final response Task: What is the current LLM provider for tests right now Shared memory used: Selected shared Titan memories: 1. The current LLM provider for tests is Claude. [property=llm_provider] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVOPS (DevOps and environment engi...`
- no_memory_agent / identity_collision / random_owner_4_a: missing_any=['Lina Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_4_a: forbidden=['Leo Robert', 'frontend polish', 'standup'] — answer=`Lina Robert owns the risk summary task. Leo Robert owns the frontend polish task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- no_memory_agent / noisy_query / random_noisy_12: missing_any=['QZ-24-011'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_12: forbidden=['coffee', 'hotel', 'sticker', 'orange', 'beach', 'karaoke'] — answer=`Noise distractor 11: coffee beach orange hotel sticker karaoke should be ignored. Critical incident memory 11: the escalation key is QZ-24-011.`
- no_memory_agent / identity_collision / random_owner_11_b: missing_any=['Axel Durand'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_11_b: forbidden=['Aline Durand', 'security audit', 'standup'] — answer=`Axel Durand owns the pricing rules task. Aline Durand owns the security audit task. Distractor: Aline Durand and Axel Durand attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_6_b: missing_any=['Ivan Lemoine'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_6_b: forbidden=['Iris Lemoine', 'security audit', 'standup'] — answer=`Ivan Lemoine owns the monitoring setup task. Iris Lemoine owns the security audit task. Distractor: Iris Lemoine and Ivan Lemoine attended the same standup meeting.`
- no_memory_agent / latest_value_update / random_update_20_current_deployment_shell: missing_any=['cmd'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_20_current_deployment_shell: forbidden=['bash', 'PowerShell'] — answer=`The current deployment shell is PowerShell. The current deployment shell is bash. The current deployment shell is cmd.`
- no_memory_agent / consolidation / random_consolidation_7: missing_all=['terminal', 'manager'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_7: missing_all=['terminal', 'manager'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / consolidation / random_consolidation_14: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_14: missing_all=['Ollama', 'future'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / identity_collision / random_owner_4_b: missing_any=['Leo Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_4_b: forbidden=['Lina Robert', 'risk summary', 'standup'] — answer=`Leo Robert owns the frontend polish task. Lina Robert owns the risk summary task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- no_memory_agent / latest_value_update / random_update_21_current_deployment_shell: missing_any=['cmd'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_21_current_deployment_shell: forbidden=['PowerShell', 'bash'] — answer=`The current deployment shell is cmd. The current deployment shell is bash. The current deployment shell is PowerShell.`
- no_memory_agent / role_routing / random_role_3: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_3: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / project_convention / random_convention_13_audit-log: missing_all=['RBAC', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / noisy_query / random_noisy_15: missing_any=['QZ-24-014'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_15: forbidden=['hotel', 'capsule', 'orange', 'travel ad', 'beach', 'karaoke'] — answer=`Critical incident memory 14: the escalation key is QZ-24-014. Noise distractor 14: karaoke travel ad beach hotel orange capsule should be ignored.`
- no_memory_agent / forget_retention / random_forget_retention_19: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_19: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-18.`
- no_memory_agent / identity_collision / random_owner_1_b: missing_any=['Axel Durand'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_1_b: forbidden=['Aline Durand', 'schema migration', 'standup'] — answer=`Axel Durand owns the pricing rules task. Aline Durand owns the schema migration task. Distractor: Aline Durand and Axel Durand attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_14_a: missing_any=['Lina Robert'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_14_a: forbidden=['Leo Robert', 'frontend polish', 'standup'] — answer=`Lina Robert owns the schema migration task. Leo Robert owns the frontend polish task. Distractor: Lina Robert and Leo Robert attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_3_b: missing_any=['Mathis Bernard'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_3_b: forbidden=['Mila Bernard', 'release checklist', 'standup'] — answer=`Mathis Bernard owns the auth refactor task. Mila Bernard owns the release checklist task. Distractor: Mila Bernard and Mathis Bernard attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_9_b: missing_any=['Marius Leroy'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_9_b: forbidden=['Manon Leroy', 'load testing', 'standup'] — answer=`Marius Leroy owns the regression suite task. Manon Leroy owns the load testing task. Distractor: Manon Leroy and Marius Leroy attended the same standup meeting.`
- no_memory_agent / noisy_query / random_noisy_5: missing_any=['QZ-24-004'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_5: forbidden=['hotel', 'router', 'sticker', 'orange', 'beach', 'karaoke'] — answer=`Noise distractor 4: router karaoke sticker hotel beach orange should be ignored. Critical incident memory 4: the escalation key is QZ-24-004.`
- no_memory_agent / identity_collision / random_owner_20_b: missing_any=['Corentin Masson'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_20_b: forbidden=['Camille Masson', 'load testing', 'standup'] — answer=`Corentin Masson owns the monitoring setup task. Camille Masson owns the load testing task. Distractor: Camille Masson and Corentin Masson attended the same standup meeting.`
- no_memory_agent / latest_value_update / random_update_11_current_benchmark_output_format: missing_any=['CSV and markdown report'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_11_current_benchmark_output_format: forbidden=['text only', 'screenshots only'] — answer=`The current benchmark output format is screenshots only. The current benchmark output format is text only. The current benchmark output format is CSV and markdown report.`
- no_memory_agent / project_convention / random_convention_10_permission: missing_all=['webhooks', 'regression tests'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_10_permission: missing_all=['webhooks', 'regression tests'] — answer=`No relevant Titan memory found.`
- no_memory_agent / consolidation / random_consolidation_20: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_20: missing_all=['Ollama', 'future'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / role_routing / random_role_9: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_9: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / noisy_query / random_noisy_19: missing_any=['QZ-24-018'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_19: forbidden=['hotel', 'router', 'sticker', 'capsule', 'beach', 'karaoke'] — answer=`Critical incident memory 18: the escalation key is QZ-24-018. Noise distractor 18: sticker karaoke hotel beach capsule router should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_19: forbidden=['beach', 'karaoke'] — answer=`Manager final response Task: Ignore beach karaoke what is the escalation key for incident 18 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 18: the escalation key is QZ-24-018. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python develop...`
- no_memory_agent / consolidation / random_consolidation_4: missing_all=['DevSoft', 'DevWeb'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_4: missing_all=['DevSoft', 'DevWeb'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / forget_retention / random_forget_retention_12: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_12: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-11.`
- no_memory_agent / latest_value_update / random_update_3_current_api_auth_method: missing_any=['JWT'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_3_current_api_auth_method: forbidden=['API key', 'basic auth'] — answer=`The current API auth method is JWT. The current API auth method is API key. The current API auth method is basic auth.`
- no_memory_agent / noisy_query / random_noisy_9: missing_any=['QZ-24-008'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_9: forbidden=['coffee', 'hotel', 'router', 'sticker', 'capsule', 'karaoke'] — answer=`Critical incident memory 8: the escalation key is QZ-24-008. Noise distractor 8: karaoke hotel coffee router capsule sticker should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_9: forbidden=['karaoke'] — answer=`Manager final response Task: Ignore karaoke what is the escalation key for incident 8 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 8: the escalation key is QZ-24-008. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVWEB (Web/API developer, confidence=0....`
- no_memory_agent / consolidation / random_consolidation_6: missing_all=['large', 'stress'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_6: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 5: preserve large, stress, and reports.`
- no_memory_agent / noisy_query / random_noisy_13: missing_any=['QZ-24-012'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_13: forbidden=['banana', 'hotel', 'router', 'sticker', 'travel ad', 'beach'] — answer=`Critical incident memory 12: the escalation key is QZ-24-012. Noise distractor 12: router hotel banana beach sticker travel ad should be ignored.`
- no_memory_agent / consolidation / random_consolidation_17: missing_all=['profile', 'forget'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_17: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 16: preserve profile, forget, and consolidation.`
- no_memory_agent / noisy_query / random_noisy_8: missing_any=['QZ-24-007'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_8: forbidden=['banana', 'hotel', 'router', 'orange', 'travel ad', 'karaoke'] — answer=`Critical incident memory 7: the escalation key is QZ-24-007. Noise distractor 7: karaoke travel ad router hotel banana orange should be ignored.`
- no_memory_agent / identity_collision / random_owner_8_a: missing_any=['Elise Roux'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_8_a: forbidden=['Elias Roux', 'API contract', 'standup'] — answer=`Elise Roux owns the provider evaluation task. Elias Roux owns the API contract task. Distractor: Elise Roux and Elias Roux attended the same standup meeting.`
- no_memory_agent / project_convention / random_convention_12_support: missing_all=['typed responses', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / latest_value_update / random_update_9_current_cache_eviction_policy: missing_any=['random eviction'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_9_current_cache_eviction_policy: forbidden=['least recently used', 'oldest-first'] — answer=`The current cache eviction policy is random eviction. The current cache eviction policy is oldest-first. The current cache eviction policy is least recently used.`
- multi_agent_titan_team / latest_value_update / random_update_9_current_cache_eviction_policy: forbidden=['least recently used', 'oldest-first'] — answer=`Manager final response Task: What is the current cache eviction policy right now Shared memory used: Selected shared Titan memories: 1. The current cache eviction policy is random eviction. 2. The current cache eviction policy is oldest-first. 3. The current cache eviction policy is least recently used. Agent plan: - MANAGER (Project manager and orchestrator...`
- no_memory_agent / consolidation / random_consolidation_12: missing_all=['large', 'stress'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_12: missing_all=['large', 'stress'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / role_routing / random_role_10: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_10: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / project_convention / random_convention_3_reporting: missing_all=['typed responses', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / project_convention / random_convention_4_document: missing_all=['REST', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / role_routing / random_role_16: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_16: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / identity_collision / random_owner_7_a: missing_any=['Yuna Petit'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_7_a: forbidden=['Yanis Petit', 'benchmark report', 'standup'] — answer=`Yuna Petit owns the report export task. Yanis Petit owns the benchmark report task. Distractor: Yuna Petit and Yanis Petit attended the same standup meeting.`
- no_memory_agent / role_routing / random_role_8: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_8: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / noisy_query / random_noisy_3: missing_any=['QZ-24-002'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_3: forbidden=['banana', 'hotel', 'router', 'sticker', 'travel ad', 'beach'] — answer=`Critical incident memory 2: the escalation key is QZ-24-002. Noise distractor 2: router beach travel ad sticker banana hotel should be ignored.`
- no_memory_agent / forget_retention / random_forget_retention_15: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_15: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-14.`
- no_memory_agent / latest_value_update / random_update_2_current_cache_eviction_policy: missing_any=['least recently used'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_2_current_cache_eviction_policy: forbidden=['random eviction', 'oldest-first'] — answer=`The current cache eviction policy is random eviction. The current cache eviction policy is oldest-first. The current cache eviction policy is least recently used.`
- multi_agent_titan_team / latest_value_update / random_update_2_current_cache_eviction_policy: forbidden=['random eviction', 'oldest-first'] — answer=`Manager final response Task: What is the current cache eviction policy right now Shared memory used: Selected shared Titan memories: 1. The current cache eviction policy is random eviction. 2. The current cache eviction policy is oldest-first. 3. The current cache eviction policy is least recently used. Agent plan: - MANAGER (Project manager and orchestrator...`
- no_memory_agent / identity_collision / random_owner_5_b: missing_any=['Nathan Garnier'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_5_b: forbidden=['Nora Garnier', 'provider evaluation', 'standup'] — answer=`Nathan Garnier owns the regression suite task. Nora Garnier owns the provider evaluation task. Distractor: Nora Garnier and Nathan Garnier attended the same standup meeting.`
- no_memory_agent / noisy_query / random_noisy_6: missing_any=['QZ-24-005'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_6: forbidden=['banana', 'coffee', 'capsule', 'orange', 'beach', 'karaoke'] — answer=`Noise distractor 5: karaoke coffee orange capsule banana beach should be ignored. Critical incident memory 5: the escalation key is QZ-24-005.`
- multi_agent_titan_team / noisy_query / random_noisy_6: forbidden=['beach', 'karaoke'] — answer=`Manager final response Task: Ignore karaoke beach what is the escalation key for incident 5 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 5: the escalation key is QZ-24-005. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer...`
- no_memory_agent / identity_collision / random_owner_2_a: missing_any=['Clara Moreau'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_2_a: forbidden=['Clement Moreau', 'pricing rules', 'standup'] — answer=`Clara Moreau owns the route review task. Clement Moreau owns the pricing rules task. Distractor: Clara Moreau and Clement Moreau attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_5_a: missing_any=['Nora Garnier'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_5_a: forbidden=['Nathan Garnier', 'regression suite', 'standup'] — answer=`Nora Garnier owns the provider evaluation task. Nathan Garnier owns the regression suite task. Distractor: Nora Garnier and Nathan Garnier attended the same standup meeting.`
- no_memory_agent / role_routing / random_role_17: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_17: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / latest_value_update / random_update_14_current_benchmark_output_format: missing_any=['text only'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_14_current_benchmark_output_format: forbidden=['screenshots only', 'CSV and markdown report'] — answer=`The current benchmark output format is text only. The current benchmark output format is screenshots only. The current benchmark output format is CSV and markdown report.`
- no_memory_agent / identity_collision / random_owner_16_b: missing_any=['Ivan Lemoine'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_16_b: forbidden=['Iris Lemoine', 'UI accessibility', 'standup'] — answer=`Ivan Lemoine owns the database backup task. Iris Lemoine owns the UI accessibility task. Distractor: Iris Lemoine and Ivan Lemoine attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_15_a: missing_any=['Nora Garnier'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_15_a: forbidden=['Nathan Garnier', 'frontend polish', 'standup'] — answer=`Nora Garnier owns the route review task. Nathan Garnier owns the frontend polish task. Distractor: Nora Garnier and Nathan Garnier attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_11_a: missing_any=['Aline Durand'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_11_a: forbidden=['Axel Durand', 'pricing rules', 'standup'] — answer=`Aline Durand owns the security audit task. Axel Durand owns the pricing rules task. Distractor: Aline Durand and Axel Durand attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_2: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_2: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-1.`
- no_memory_agent / policy_recall / random_policy_2: missing_all=['manager', 'Titan memory'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / project_convention / random_convention_2_notification: missing_all=['API route', 'integration tests'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_2_notification: missing_all=['API route', 'integration tests'] — answer=`No relevant Titan memory found.`
- no_memory_agent / noisy_query / random_noisy_17: missing_any=['QZ-24-016'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_17: forbidden=['coffee', 'sticker', 'capsule', 'orange', 'beach', 'karaoke'] — answer=`Critical incident memory 16: the escalation key is QZ-24-016. Noise distractor 16: orange sticker capsule beach coffee karaoke should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_17: forbidden=['beach'] — answer=`Manager final response Task: Ignore beach what is the escalation key for incident 16 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 16: the escalation key is QZ-24-016. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, conf...`
- no_memory_agent / noisy_query / random_noisy_16: missing_any=['QZ-24-015'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_16: forbidden=['banana', 'coffee', 'sticker', 'capsule', 'beach', 'karaoke'] — answer=`Critical incident memory 15: the escalation key is QZ-24-015. Noise distractor 15: sticker capsule coffee karaoke banana beach should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_16: forbidden=['beach', 'karaoke'] — answer=`Manager final response Task: Ignore beach karaoke what is the escalation key for incident 15 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 15: the escalation key is QZ-24-015. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python develop...`
- no_memory_agent / project_convention / random_convention_17_import: missing_all=['API route', 'integration tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / forget_retention / random_forget_retention_5: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_5: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-4.`
- no_memory_agent / identity_collision / random_owner_12_b: missing_any=['Clement Moreau'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_12_b: forbidden=['Clara Moreau', 'route review', 'standup'] — answer=`Clement Moreau owns the auth refactor task. Clara Moreau owns the route review task. Distractor: Clara Moreau and Clement Moreau attended the same standup meeting.`
- no_memory_agent / role_routing / random_role_7: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_7: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / policy_recall / random_policy_1: missing_all=['Ollama', 'better', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / policy_recall / random_policy_1: missing_all=['better', 'free'] — answer=`For the current prototype, Ollama is the default LLM provider.`
- multi_agent_titan_team / policy_recall / random_policy_1: missing_all=['better'] — answer=`Manager final response Task: Which LLM provider rule should we keep for tests Shared memory used: Selected shared Titan memories: 1. For the current prototype, Ollama is the default LLM provider. [property=llm_provider] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVOPS (DevOps and env...`
- no_memory_agent / identity_collision / random_owner_10_a: missing_any=['Camille Masson'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_10_a: forbidden=['Corentin Masson', 'benchmark report', 'standup'] — answer=`Camille Masson owns the security audit task. Corentin Masson owns the benchmark report task. Distractor: Camille Masson and Corentin Masson attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_13_b: missing_any=['Mathis Bernard'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_13_b: forbidden=['Mila Bernard', 'load testing', 'standup'] — answer=`Mathis Bernard owns the frontend polish task. Mila Bernard owns the load testing task. Distractor: Mila Bernard and Mathis Bernard attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_9: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_9: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-8.`
- no_memory_agent / project_convention / random_convention_16_analytics: missing_all=['webhooks', 'regression tests'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_16_analytics: missing_all=['webhooks', 'regression tests'] — answer=`No relevant Titan memory found.`
- no_memory_agent / role_routing / random_role_12: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_12: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / noisy_query / random_noisy_4: missing_any=['QZ-24-003'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_4: forbidden=['router', 'sticker', 'capsule', 'orange', 'travel ad', 'beach'] — answer=`Critical incident memory 3: the escalation key is QZ-24-003. Noise distractor 3: sticker beach orange router travel ad capsule should be ignored.`
- no_memory_agent / role_routing / random_role_4: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_4: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / identity_collision / random_owner_6_a: missing_any=['Iris Lemoine'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_6_a: forbidden=['Ivan Lemoine', 'monitoring setup', 'standup'] — answer=`Iris Lemoine owns the security audit task. Ivan Lemoine owns the monitoring setup task. Distractor: Iris Lemoine and Ivan Lemoine attended the same standup meeting.`
- no_memory_agent / policy_recall / random_policy_5: missing_all=['manager', 'shared Titan memory'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / identity_collision / random_owner_7_b: missing_any=['Yanis Petit'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_7_b: forbidden=['Yuna Petit', 'report export', 'standup'] — answer=`Yanis Petit owns the benchmark report task. Yuna Petit owns the report export task. Distractor: Yuna Petit and Yanis Petit attended the same standup meeting.`
... 140 more failures omitted from console report.
