# Randomized adversarial multi-agent Titan team comparison

============================================================================================
RANDOMIZED ADVERSARIAL MULTI-AGENT TITAN TEAM COMPARISON
============================================================================================
Cases: 150 | Seed: 303

GLOBAL RESULTS
--------------------------------------------------------------------------------------------
no_memory_agent          | 0/150 (0.0%)
single_titan_agent       | 18/150 (12.0%)
multi_agent_titan_team   | 135/150 (90.0%)

RESULTS BY CATEGORY
--------------------------------------------------------------------------------------------
no_memory_agent          | consolidation            | 0/20 (0.0%)
no_memory_agent          | forget_retention         | 0/20 (0.0%)
no_memory_agent          | identity_collision       | 0/34 (0.0%)
no_memory_agent          | latest_value_update      | 0/17 (0.0%)
no_memory_agent          | noisy_query              | 0/18 (0.0%)
no_memory_agent          | policy_recall            | 0/5 (0.0%)
no_memory_agent          | project_convention       | 0/17 (0.0%)
no_memory_agent          | role_routing             | 0/19 (0.0%)
single_titan_agent       | consolidation            | 0/20 (0.0%)
single_titan_agent       | forget_retention         | 0/20 (0.0%)
single_titan_agent       | identity_collision       | 1/34 (2.9%)
single_titan_agent       | latest_value_update      | 2/17 (11.8%)
single_titan_agent       | noisy_query              | 0/18 (0.0%)
single_titan_agent       | policy_recall            | 4/5 (80.0%)
single_titan_agent       | project_convention       | 11/17 (64.7%)
single_titan_agent       | role_routing             | 0/19 (0.0%)
multi_agent_titan_team   | consolidation            | 20/20 (100.0%)
multi_agent_titan_team   | forget_retention         | 20/20 (100.0%)
multi_agent_titan_team   | identity_collision       | 34/34 (100.0%)
multi_agent_titan_team   | latest_value_update      | 14/17 (82.4%)
multi_agent_titan_team   | noisy_query              | 7/18 (38.9%)
multi_agent_titan_team   | policy_recall            | 4/5 (80.0%)
multi_agent_titan_team   | project_convention       | 17/17 (100.0%)
multi_agent_titan_team   | role_routing             | 19/19 (100.0%)

FAILURES
--------------------------------------------------------------------------------------------
- no_memory_agent / role_routing / random_role_16: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_16: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / role_routing / random_role_21: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_21: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / latest_value_update / random_update_9_current_demo_ui_style: missing_any=['markdown only'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_9_current_demo_ui_style: forbidden=['colored terminal panels', 'plain text'] — answer=`The current demo UI style is markdown only. The current demo UI style is plain text. The current demo UI style is colored terminal panels.`
- no_memory_agent / latest_value_update / random_update_15_current_deployment_shell: missing_any=['cmd'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_15_current_deployment_shell: forbidden=['bash', 'PowerShell'] — answer=`The current deployment shell is cmd. The current deployment shell is bash. The current deployment shell is PowerShell.`
- no_memory_agent / consolidation / random_consolidation_14: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_14: missing_all=['Ollama', 'future'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / identity_collision / random_owner_18_a: missing_any=['Elise Roux'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_18_a: forbidden=['Elias Roux', 'deployment script', 'standup'] — answer=`Elise Roux owns the security audit task. Elias Roux owns the deployment script task. Distractor: Elise Roux and Elias Roux attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_16: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_16: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-15.`
- no_memory_agent / project_convention / random_convention_5_inventory: missing_all=['FastAPI', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / noisy_query / random_noisy_1: missing_any=['QZ-3-000'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_1: forbidden=['banana', 'coffee', 'hotel', 'router', 'capsule', 'karaoke'] — answer=`Critical incident memory 0: the escalation key is QZ-3-000. Noise distractor 0: router hotel banana karaoke capsule coffee should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_1: forbidden=['karaoke'] — answer=`Manager final response Task: Ignore karaoke what is the escalation key for incident 0 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 0: the escalation key is QZ-3-000. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confi...`
- no_memory_agent / project_convention / random_convention_15_reporting: missing_all=['API route', 'integration tests'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_15_reporting: missing_all=['API route', 'integration tests'] — answer=`No relevant Titan memory found.`
- no_memory_agent / policy_recall / random_policy_1: missing_all=['Ollama', 'better', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / policy_recall / random_policy_1: missing_all=['better', 'free'] — answer=`For the current prototype, Ollama is the default LLM provider.`
- multi_agent_titan_team / policy_recall / random_policy_1: missing_all=['better'] — answer=`Manager final response Task: Which LLM provider rule should we keep for tests Shared memory used: Selected shared Titan memories: 1. For the current prototype, Ollama is the default LLM provider. [property=llm_provider] Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVOPS (DevOps and env...`
- no_memory_agent / noisy_query / random_noisy_11: missing_any=['QZ-3-010'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_11: forbidden=['coffee', 'hotel', 'orange', 'travel ad', 'beach', 'karaoke'] — answer=`Critical incident memory 10: the escalation key is QZ-3-010. Noise distractor 10: coffee karaoke orange travel ad hotel beach should be ignored.`
- no_memory_agent / role_routing / random_role_3: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_3: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / consolidation / random_consolidation_13: missing_all=['terminal', 'manager'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_13: missing_all=['terminal', 'manager'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / consolidation / random_consolidation_17: missing_all=['profile', 'forget'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_17: missing_all=['profile', 'forget'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / forget_retention / random_forget_retention_19: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_19: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-18.`
- no_memory_agent / noisy_query / random_noisy_20: missing_any=['QZ-3-019'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_20: forbidden=['banana', 'coffee', 'sticker', 'capsule', 'beach', 'karaoke'] — answer=`Critical incident memory 19: the escalation key is QZ-3-019. Noise distractor 19: beach banana capsule coffee karaoke sticker should be ignored.`
- no_memory_agent / forget_retention / random_forget_retention_14: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_14: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-13.`
- no_memory_agent / latest_value_update / random_update_5_current_deployment_shell: missing_any=['bash'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_5_current_deployment_shell: forbidden=['PowerShell', 'cmd'] — answer=`The current deployment shell is PowerShell. The current deployment shell is bash. The current deployment shell is cmd.`
- no_memory_agent / consolidation / random_consolidation_8: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_8: missing_all=['Ollama', 'future'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / role_routing / random_role_7: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_7: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / project_convention / random_convention_3_support: missing_all=['RBAC', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_3_support: missing_all=['RBAC', 'unit tests'] — answer=`No relevant Titan memory found.`
- no_memory_agent / consolidation / random_consolidation_4: missing_all=['DevSoft', 'DevWeb'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_4: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 3: preserve DevSoft, DevWeb, and DevOps.`
- no_memory_agent / noisy_query / random_noisy_5: missing_any=['QZ-3-004'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_5: forbidden=['banana', 'router', 'sticker', 'orange', 'beach', 'karaoke'] — answer=`Critical incident memory 4: the escalation key is QZ-3-004. Noise distractor 4: banana router orange karaoke beach sticker should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_5: forbidden=['beach', 'karaoke'] — answer=`Manager final response Task: Ignore karaoke beach what is the escalation key for incident 4 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 4: the escalation key is QZ-3-004. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer,...`
- no_memory_agent / consolidation / random_consolidation_19: missing_all=['terminal', 'manager'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_19: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 18: preserve terminal, manager, and results.`
- no_memory_agent / identity_collision / random_owner_20_b: missing_any=['Corentin Masson'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_20_b: forbidden=['Camille Masson', 'memory adapter', 'standup'] — answer=`Corentin Masson owns the frontend polish task. Camille Masson owns the memory adapter task. Distractor: Camille Masson and Corentin Masson attended the same standup meeting.`
- no_memory_agent / noisy_query / random_noisy_3: missing_any=['QZ-3-002'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_3: forbidden=['hotel', 'router', 'sticker', 'capsule', 'travel ad', 'beach'] — answer=`Noise distractor 2: router sticker capsule travel ad hotel beach should be ignored. Critical incident memory 2: the escalation key is QZ-3-002.`
- no_memory_agent / noisy_query / random_noisy_15: missing_any=['QZ-3-014'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_15: forbidden=['banana', 'hotel', 'router', 'sticker', 'orange', 'karaoke'] — answer=`Critical incident memory 14: the escalation key is QZ-3-014. Noise distractor 14: hotel router sticker karaoke orange banana should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_15: forbidden=['beach'] — answer=`Manager final response Task: Ignore beach what is the escalation key for incident 14 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 14: the escalation key is QZ-3-014. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVWEB (Web/API developer, confidence=0.8...`
- no_memory_agent / identity_collision / random_owner_18_b: missing_any=['Elias Roux'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_18_b: forbidden=['Elise Roux', 'security audit', 'standup'] — answer=`Elias Roux owns the deployment script task. Elise Roux owns the security audit task. Distractor: Elise Roux and Elias Roux attended the same standup meeting.`
- no_memory_agent / project_convention / random_convention_8_reporting: missing_all=['REST', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / project_convention / random_convention_1_analytics: missing_all=['REST', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / consolidation / random_consolidation_1: missing_all=['terminal', 'manager'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_1: missing_all=['terminal', 'manager'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / noisy_query / random_noisy_2: missing_any=['QZ-3-001'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_2: forbidden=['banana', 'hotel', 'capsule', 'orange', 'travel ad', 'beach'] — answer=`Critical incident memory 1: the escalation key is QZ-3-001. Noise distractor 1: travel ad hotel capsule orange banana beach should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_2: forbidden=['karaoke'] — answer=`Manager final response Task: Ignore karaoke what is the escalation key for incident 1 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 1: the escalation key is QZ-3-001. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVSOFT (Software/Python developer, confi...`
- no_memory_agent / forget_retention / random_forget_retention_11: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_11: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-10.`
- no_memory_agent / latest_value_update / random_update_19_current_memory_backend: missing_any=['short-term memory'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_19_current_memory_backend: forbidden=['shared Titan memory', 'vector store'] — answer=`The current memory backend is shared Titan memory. The current memory backend is vector store. The current memory backend is short-term memory.`
- no_memory_agent / latest_value_update / random_update_12_current_memory_backend: missing_any=['shared Titan memory'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_12_current_memory_backend: forbidden=['short-term memory', 'vector store'] — answer=`The current memory backend is short-term memory. The current memory backend is vector store. The current memory backend is shared Titan memory.`
- no_memory_agent / noisy_query / random_noisy_13: missing_any=['QZ-3-012'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_13: forbidden=['banana', 'router', 'capsule', 'orange', 'travel ad', 'beach'] — answer=`Critical incident memory 12: the escalation key is QZ-3-012. Noise distractor 12: router beach capsule travel ad orange banana should be ignored.`
- multi_agent_titan_team / noisy_query / random_noisy_13: forbidden=['beach'] — answer=`Manager final response Task: Ignore beach what is the escalation key for incident 12 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 12: the escalation key is QZ-3-012. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVWEB (Web/API developer, confidence=0.8...`
- no_memory_agent / project_convention / random_convention_11_fraud: missing_all=['typed responses', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / project_convention / random_convention_14_import: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_14_import: missing_all=['FastAPI', 'pytest'] — answer=`No relevant Titan memory found.`
- no_memory_agent / role_routing / random_role_2: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_2: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / latest_value_update / random_update_1_current_api_auth_method: missing_any=['basic auth'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_1_current_api_auth_method: forbidden=['API key', 'JWT'] — answer=`The current API auth method is basic auth. The current API auth method is JWT. The current API auth method is API key.`
- no_memory_agent / identity_collision / random_owner_8_b: missing_any=['Elias Roux'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_8_b: forbidden=['Elise Roux', 'load testing', 'standup'] — answer=`Elias Roux owns the regression suite task. Elise Roux owns the load testing task. Distractor: Elise Roux and Elias Roux attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_5: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_5: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-4.`
- no_memory_agent / project_convention / random_convention_6_profile: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / project_convention / random_convention_16_audit-log: missing_all=['FastAPI', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / role_routing / random_role_17: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_17: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / forget_retention / random_forget_retention_15: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_15: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-14.`
- no_memory_agent / project_convention / random_convention_9_inventory: missing_all=['FastAPI', 'unit tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / latest_value_update / random_update_13_current_testing_framework: missing_any=['nose'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_13_current_testing_framework: forbidden=['unittest', 'pytest'] — answer=`The current testing framework is nose. The current testing framework is unittest. The current testing framework is pytest.`
- no_memory_agent / noisy_query / random_noisy_12: missing_any=['QZ-3-011'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_12: forbidden=['banana', 'coffee', 'router', 'capsule', 'travel ad', 'karaoke'] — answer=`Noise distractor 11: coffee karaoke banana router travel ad capsule should be ignored. Critical incident memory 11: the escalation key is QZ-3-011.`
- multi_agent_titan_team / noisy_query / random_noisy_12: forbidden=['karaoke'] — answer=`Manager final response Task: Ignore karaoke what is the escalation key for incident 11 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 11: the escalation key is QZ-3-011. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVWEB (Web/API developer, confidence=0...`
- no_memory_agent / identity_collision / random_owner_2_a: missing_any=['Clara Moreau'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_2_a: forbidden=['Clement Moreau', 'documentation pass', 'standup'] — answer=`Clara Moreau owns the risk summary task. Clement Moreau owns the documentation pass task. Distractor: Clara Moreau and Clement Moreau attended the same standup meeting.`
- no_memory_agent / consolidation / random_consolidation_20: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_20: missing_all=['Ollama', 'future'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / identity_collision / random_owner_19_a: missing_any=['Manon Leroy'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_19_a: forbidden=['Marius Leroy', 'API contract', 'standup'] — answer=`Manon Leroy owns the memory adapter task. Marius Leroy owns the API contract task. Distractor: Manon Leroy and Marius Leroy attended the same standup meeting.`
- no_memory_agent / project_convention / random_convention_20_fraud: missing_all=['REST', 'pytest'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / forget_retention / random_forget_retention_21: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_21: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-20.`
- no_memory_agent / forget_retention / random_forget_retention_2: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_2: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-1.`
- no_memory_agent / identity_collision / random_owner_17_a: missing_any=['Yuna Petit'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_17_a: forbidden=['Yanis Petit', 'auth refactor', 'standup'] — answer=`Yuna Petit owns the risk summary task. Yanis Petit owns the auth refactor task. Distractor: Yuna Petit and Yanis Petit attended the same standup meeting.`
- no_memory_agent / forget_retention / random_forget_retention_6: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_6: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-5.`
- no_memory_agent / identity_collision / random_owner_1_b: missing_any=['Axel Durand'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_1_b: forbidden=['Aline Durand', 'report export', 'standup'] — answer=`Axel Durand owns the database backup task. Aline Durand owns the report export task. Distractor: Aline Durand and Axel Durand attended the same standup meeting.`
- no_memory_agent / latest_value_update / random_update_7_current_preferred_backend_language: missing_any=['Python'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / forget_retention / random_forget_retention_7: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_7: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-6.`
- no_memory_agent / forget_retention / random_forget_retention_3: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_3: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-2.`
- no_memory_agent / forget_retention / random_forget_retention_18: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_18: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-17.`
- no_memory_agent / latest_value_update / random_update_6_current_cache_eviction_policy: missing_any=['oldest-first'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_6_current_cache_eviction_policy: forbidden=['random eviction', 'least recently used'] — answer=`The current cache eviction policy is random eviction. The current cache eviction policy is oldest-first. The current cache eviction policy is least recently used.`
- multi_agent_titan_team / latest_value_update / random_update_6_current_cache_eviction_policy: forbidden=['random eviction', 'least recently used'] — answer=`Manager final response Task: What is the current cache eviction policy right now Shared memory used: Selected shared Titan memories: 1. The current cache eviction policy is random eviction. 2. The current cache eviction policy is oldest-first. 3. The current cache eviction policy is least recently used. Agent plan: - MANAGER (Project manager and orchestrator...`
- no_memory_agent / policy_recall / random_policy_2: missing_all=['manager', 'Titan memory'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / role_routing / random_role_1: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_1: missing_all=['free'] — answer=`Provider rule: Ollama is default for current tests.`
- no_memory_agent / consolidation / random_consolidation_15: missing_all=['Tester', 'Evaluator'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_15: missing_all=['Tester', 'Evaluator'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / noisy_query / random_noisy_6: missing_any=['QZ-3-005'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / noisy_query / random_noisy_6: forbidden=['coffee', 'router', 'sticker', 'capsule', 'beach', 'karaoke'] — answer=`Noise distractor 5: router beach capsule karaoke coffee sticker should be ignored. Critical incident memory 5: the escalation key is QZ-3-005.`
- multi_agent_titan_team / noisy_query / random_noisy_6: forbidden=['beach', 'karaoke'] — answer=`Manager final response Task: Ignore karaoke beach what is the escalation key for incident 5 Shared memory used: Selected shared Titan memories: 1. Critical incident memory 5: the escalation key is QZ-3-005. Agent plan: - MANAGER (Project manager and orchestrator, confidence=0.85) action: coordinate action: store_decision - DEVWEB (Web/API developer, confiden...`
- no_memory_agent / project_convention / random_convention_4_fraud: missing_all=['API route', 'integration tests'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / consolidation / random_consolidation_2: missing_all=['Ollama', 'future'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_2: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 1: preserve Ollama, future, and provider.`
- no_memory_agent / identity_collision / random_owner_13_a: missing_any=['Mila Bernard'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_13_a: forbidden=['Mathis Bernard', 'documentation pass', 'standup'] — answer=`Mila Bernard owns the report export task. Mathis Bernard owns the documentation pass task. Distractor: Mila Bernard and Mathis Bernard attended the same standup meeting.`
- no_memory_agent / identity_collision / random_owner_12_b: missing_any=['Clement Moreau'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / identity_collision / random_owner_12_b: forbidden=['Clara Moreau', 'security audit', 'standup'] — answer=`Clement Moreau owns the benchmark report task. Clara Moreau owns the security audit task. Distractor: Clara Moreau and Clement Moreau attended the same standup meeting.`
- no_memory_agent / role_routing / random_role_18: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_18: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / latest_value_update / random_update_3_current_testing_framework: missing_any=['unittest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / latest_value_update / random_update_3_current_testing_framework: forbidden=['pytest', 'nose'] — answer=`The current testing framework is nose. The current testing framework is unittest. The current testing framework is pytest.`
- no_memory_agent / role_routing / random_role_11: missing_all=['benchmark', 'failures'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_11: missing_all=['benchmark', 'failures'] — answer=`No relevant Titan memory found.`
- no_memory_agent / consolidation / random_consolidation_12: missing_all=['large', 'stress'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_12: forbidden=['Please consolidate'] — answer=`Please consolidate the active project rules after storing them. Consolidation random rule 11: preserve large, stress, and reports.`
- no_memory_agent / identity_collision / random_owner_19_b: missing_any=['Marius Leroy'] — answer=`I do not know because no shared memory is available.`
- no_memory_agent / consolidation / random_consolidation_6: missing_all=['large', 'stress'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / consolidation / random_consolidation_6: missing_all=['large', 'stress'] — answer=`Please consolidate the active project rules after storing them.`
- no_memory_agent / project_convention / random_convention_19_fraud: missing_all=['API route', 'integration tests'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / project_convention / random_convention_19_fraud: missing_all=['API route', 'integration tests'] — answer=`No relevant Titan memory found.`
- no_memory_agent / forget_retention / random_forget_retention_9: missing_all=['limitations', 'next steps'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / forget_retention / random_forget_retention_9: missing_all=['limitations', 'next steps'] — answer=`Distractor: Lucas Martin's archived color was cyan-8.`
- no_memory_agent / role_routing / random_role_9: missing_all=['FastAPI', 'pytest'] — answer=`I do not know because no shared memory is available.`
- single_titan_agent / role_routing / random_role_9: missing_all=['FastAPI', 'pytest'] — answer=`Team rule: DevWeb handles API routes, DevSoft handles Python internals, DevOps handles commands, Tester handles benchmarks, Evaluator reports limitations.`
- no_memory_agent / role_routing / random_role_15: missing_all=['Ollama', 'free'] — answer=`I do not know because no shared memory is available.`
... 137 more failures omitted from console report.
