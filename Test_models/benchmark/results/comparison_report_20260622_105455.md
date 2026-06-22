# Antoine Models Comparison Report

- Date: 2026-06-22 11:02:29
- Scales: small, medium

## Global results

| Model | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---:|---:|---:|---:|---:|
| mamba | 2 | 14 | 14.3% | 28.520s | 0.281s |
| titan | 13 | 14 | 92.9% | 1.512s | 0.052s |

## Results by scale

| Model | Scale | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| mamba | medium | 1 | 8 | 12.5% | 48.934s | 0.258s |
| mamba | small | 1 | 6 | 16.7% | 1.301s | 0.311s |
| titan | medium | 7 | 8 | 87.5% | 2.462s | 0.087s |
| titan | small | 6 | 6 | 100.0% | 0.247s | 0.006s |

## Results by category

| Model | Category | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| mamba | atomic_update | 0 | 1 | 0.0% | 2.354s | 0.172s |
| mamba | distractors | 0 | 1 | 0.0% | 0.269s | 0.249s |
| mamba | forget_secret_code | 1 | 1 | 100.0% | 3.960s | 0.006s |
| mamba | large_distractors | 0 | 1 | 0.0% | 40.796s | 0.343s |
| mamba | large_forget_secret_code | 1 | 1 | 100.0% | 69.018s | 0.281s |
| mamba | large_retention_after_forget | 0 | 1 | 0.0% | 75.906s | 0.210s |
| mamba | large_secret_code | 0 | 1 | 0.0% | 39.653s | 0.196s |
| mamba | large_simple_recall | 0 | 1 | 0.0% | 44.086s | 0.406s |
| mamba | large_update | 0 | 3 | 0.0% | 40.670s | 0.210s |
| mamba | multi_profile | 0 | 1 | 0.0% | 0.742s | 0.490s |
| mamba | simple_recall | 0 | 1 | 0.0% | 0.001s | 0.683s |
| mamba | single_value_update | 0 | 1 | 0.0% | 0.478s | 0.266s |
| titan | atomic_update | 1 | 1 | 100.0% | 0.045s | 0.004s |
| titan | distractors | 1 | 1 | 100.0% | 0.109s | 0.007s |
| titan | forget_secret_code | 1 | 1 | 100.0% | 0.021s | 0.011s |
| titan | large_distractors | 1 | 1 | 100.0% | 1.903s | 0.077s |
| titan | large_forget_secret_code | 1 | 1 | 100.0% | 2.288s | 0.093s |
| titan | large_retention_after_forget | 0 | 1 | 0.0% | 3.690s | 0.104s |
| titan | large_secret_code | 1 | 1 | 100.0% | 1.871s | 0.073s |
| titan | large_simple_recall | 1 | 1 | 100.0% | 3.089s | 0.092s |
| titan | large_update | 3 | 3 | 100.0% | 2.284s | 0.085s |
| titan | multi_profile | 1 | 1 | 100.0% | 0.076s | 0.005s |
| titan | simple_recall | 1 | 1 | 100.0% | 1.221s | 0.008s |
| titan | single_value_update | 1 | 1 | 100.0% | 0.012s | 0.003s |

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
