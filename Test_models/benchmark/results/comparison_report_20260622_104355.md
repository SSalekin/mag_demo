# Antoine Models Comparison Report

- Date: 2026-06-22 10:51:49
- Scales: small, medium

## Global results

| Model | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---:|---:|---:|---:|---:|
| mamba | 2 | 14 | 14.3% | 30.402s | 0.307s |
| titan | 13 | 14 | 92.9% | 1.137s | 0.048s |

## Results by scale

| Model | Scale | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| mamba | medium | 1 | 8 | 12.5% | 51.451s | 0.210s |
| mamba | small | 1 | 6 | 16.7% | 2.337s | 0.435s |
| titan | medium | 7 | 8 | 87.5% | 1.925s | 0.080s |
| titan | small | 6 | 6 | 100.0% | 0.087s | 0.004s |

## Results by category

| Model | Category | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| mamba | atomic_update | 0 | 1 | 0.0% | 3.292s | 0.423s |
| mamba | distractors | 0 | 1 | 0.0% | 0.237s | 0.189s |
| mamba | forget_secret_code | 1 | 1 | 100.0% | 9.549s | 0.011s |
| mamba | large_distractors | 0 | 1 | 0.0% | 45.464s | 0.201s |
| mamba | large_forget_secret_code | 1 | 1 | 100.0% | 101.842s | 0.205s |
| mamba | large_retention_after_forget | 0 | 1 | 0.0% | 65.942s | 0.212s |
| mamba | large_secret_code | 0 | 1 | 0.0% | 39.059s | 0.195s |
| mamba | large_simple_recall | 0 | 1 | 0.0% | 39.361s | 0.254s |
| mamba | large_update | 0 | 3 | 0.0% | 39.979s | 0.204s |
| mamba | multi_profile | 0 | 1 | 0.0% | 0.461s | 0.178s |
| mamba | simple_recall | 0 | 1 | 0.0% | 0.001s | 1.577s |
| mamba | single_value_update | 0 | 1 | 0.0% | 0.483s | 0.233s |
| titan | atomic_update | 1 | 1 | 100.0% | 0.074s | 0.003s |
| titan | distractors | 1 | 1 | 100.0% | 0.043s | 0.004s |
| titan | forget_secret_code | 1 | 1 | 100.0% | 0.015s | 0.008s |
| titan | large_distractors | 1 | 1 | 100.0% | 1.845s | 0.074s |
| titan | large_forget_secret_code | 1 | 1 | 100.0% | 1.979s | 0.074s |
| titan | large_retention_after_forget | 0 | 1 | 0.0% | 2.084s | 0.080s |
| titan | large_secret_code | 1 | 1 | 100.0% | 1.837s | 0.077s |
| titan | large_simple_recall | 1 | 1 | 100.0% | 1.903s | 0.089s |
| titan | large_update | 3 | 3 | 100.0% | 1.918s | 0.083s |
| titan | multi_profile | 1 | 1 | 100.0% | 0.060s | 0.004s |
| titan | simple_recall | 1 | 1 | 100.0% | 0.207s | 0.004s |
| titan | single_value_update | 1 | 1 | 100.0% | 0.120s | 0.003s |

## Failures

### mamba - small - simple_location
- Category: simple_recall
- Reason: missing expected keyword(s): ['hanoi']
- Question: Where is Sarah Nguyen from?
- Expected: `["hanoi"]`
- Forbidden: `["da nang", "danang", "rennes", "lyon"]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`

### mamba - small - study_retained_after_location_update
- Category: atomic_update
- Reason: missing expected keyword(s): ['natural language processing', 'nlp']
- Question: What does Sarah Nguyen study?
- Expected: `["natural language processing", "nlp"]`
- Forbidden: `[]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`

### mamba - small - favorite_color_update
- Category: single_value_update
- Reason: missing expected keyword(s): ['black']
- Question: What is Sarah Nguyen's favorite color?
- Expected: `["black"]`
- Forbidden: `["yellow"]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`

### mamba - small - multi_profile_separation
- Category: multi_profile
- Reason: missing expected keyword(s): ['french', 'english', 'spanish']
- Question: Which languages does Maya Dubois speak?
- Expected: `["french", "english", "spanish"]`
- Forbidden: `["osaka", "robotics", "nantes"]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`

### mamba - small - distractor_resistance
- Category: distractors
- Reason: missing expected keyword(s): ['medvision']
- Question: What is Sofia Garcia's research group called?
- Expected: `["medvision"]`
- Forbidden: `["blue sofa", "chicken rice", "weather"]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`

### mamba - medium - medium_location_recall
- Category: large_simple_recall
- Reason: missing expected keyword(s): ['marseille']
- Question: Where is HanaAlpha Bernard from?
- Expected: `["marseille"]`
- Forbidden: `["blue sofa", "chicken rice"]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`

### mamba - medium - medium_location_update
- Category: large_update
- Reason: missing expected keyword(s): ['rome']
- Question: Where is HanaAlpha Bernard from?
- Expected: `["rome"]`
- Forbidden: `["marseille"]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`

### mamba - medium - medium_favorite_color_update
- Category: large_update
- Reason: missing expected keyword(s): ['cyan']
- Question: What is PriyaAlpha Meyer's favorite color?
- Expected: `["cyan"]`
- Forbidden: `["yellow"]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`

### mamba - medium - medium_favorite_language_update
- Category: large_update
- Reason: missing expected keyword(s): ['python']
- Question: What is PriyaAlpha Meyer's favorite programming language?
- Expected: `["python"]`
- Forbidden: `["javascript"]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`

### mamba - medium - medium_secret_code_update
- Category: large_secret_code
- Reason: missing expected keyword(s): ['new-code-1022']
- Question: What is WendyAlpha Rossi's secret code?
- Expected: `["new-code-1022"]`
- Forbidden: `["code-1022"]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`

### titan - medium - medium_forget_target_only_retains_project
- Category: large_retention_after_forget
- Reason: missing expected keyword(s): ['medical diagnosis assistant']
- Question: What does CarlosBeta Wilson work on?
- Expected: `["medical diagnosis assistant"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: `Based on memory: Carlosbeta Wilson is from Manchester, United Kingdom.`

### mamba - medium - medium_forget_target_only_retains_project
- Category: large_retention_after_forget
- Reason: missing expected keyword(s): ['medical diagnosis assistant']
- Question: What does CarlosBeta Wilson work on?
- Expected: `["medical diagnosis assistant"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`

### mamba - medium - medium_distractor_resistance
- Category: large_distractors
- Reason: missing expected keyword(s): ['r122']
- Question: What is WendyAlpha Rossi's office room?
- Expected: `["r122"]`
- Forbidden: `["blue sofa", "chicken rice", "weather"]`
- Answer/retrieved text: ` | [Error communicating with Ollama: model 'llama3' not found (status code: 404)]`
