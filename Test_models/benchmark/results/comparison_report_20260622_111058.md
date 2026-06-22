# Antoine Models Comparison Report

- Date: 2026-06-22 11:19:13
- Scales: small, medium

## Global results

| Model | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---:|---:|---:|---:|---:|
| mamba | 7 | 14 | 50.0% | 29.829s | 1.762s |
| titan | 12 | 14 | 85.7% | 1.369s | 0.106s |

## Results by scale

| Model | Scale | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| mamba | medium | 4 | 8 | 50.0% | 45.525s | 1.327s |
| mamba | small | 3 | 6 | 50.0% | 8.900s | 2.343s |
| titan | medium | 7 | 8 | 87.5% | 2.223s | 0.090s |
| titan | small | 5 | 6 | 83.3% | 0.230s | 0.127s |

## Results by category

| Model | Category | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| mamba | atomic_update | 1 | 1 | 100.0% | 3.344s | 1.015s |
| mamba | distractors | 0 | 1 | 0.0% | 0.262s | 0.797s |
| mamba | forget_secret_code | 1 | 1 | 100.0% | 48.774s | 0.762s |
| mamba | large_distractors | 1 | 1 | 100.0% | 39.216s | 1.173s |
| mamba | large_forget_secret_code | 0 | 1 | 0.0% | 45.725s | 1.976s |
| mamba | large_retention_after_forget | 0 | 1 | 0.0% | 69.569s | 1.266s |
| mamba | large_secret_code | 1 | 1 | 100.0% | 45.028s | 1.660s |
| mamba | large_simple_recall | 0 | 1 | 0.0% | 39.888s | 1.102s |
| mamba | large_update | 2 | 3 | 66.7% | 41.592s | 1.146s |
| mamba | multi_profile | 0 | 1 | 0.0% | 0.547s | 0.842s |
| mamba | simple_recall | 0 | 1 | 0.0% | 0.001s | 9.481s |
| mamba | single_value_update | 1 | 1 | 100.0% | 0.474s | 1.160s |
| titan | atomic_update | 0 | 1 | 0.0% | 0.154s | 0.004s |
| titan | distractors | 1 | 1 | 100.0% | 0.042s | 0.006s |
| titan | forget_secret_code | 1 | 1 | 100.0% | 0.009s | 0.718s |
| titan | large_distractors | 1 | 1 | 100.0% | 1.939s | 0.069s |
| titan | large_forget_secret_code | 1 | 1 | 100.0% | 3.471s | 0.085s |
| titan | large_retention_after_forget | 0 | 1 | 0.0% | 2.140s | 0.090s |
| titan | large_secret_code | 1 | 1 | 100.0% | 2.073s | 0.089s |
| titan | large_simple_recall | 1 | 1 | 100.0% | 2.015s | 0.082s |
| titan | large_update | 3 | 3 | 100.0% | 2.049s | 0.101s |
| titan | multi_profile | 1 | 1 | 100.0% | 0.088s | 0.005s |
| titan | simple_recall | 1 | 1 | 100.0% | 1.073s | 0.025s |
| titan | single_value_update | 1 | 1 | 100.0% | 0.015s | 0.004s |

## Failures

### mamba - small - simple_location
- Category: simple_recall
- Reason: missing expected keyword(s): ['hanoi']
- Question: Where is Sarah Nguyen from?
- Expected: `["hanoi"]`
- Forbidden: `["da nang", "danang", "rennes", "lyon"]`
- Answer/retrieved text: `I don't know based on memory.`

### titan - small - study_retained_after_location_update
- Category: atomic_update
- Reason: missing expected keyword(s): ['natural language processing', 'nlp']
- Question: What does Sarah Nguyen study?
- Expected: `["natural language processing", "nlp"]`
- Forbidden: `[]`
- Answer/retrieved text: `Based on memory: Sarah Nguyen is a 23-year-old AI student.`

### mamba - small - multi_profile_separation
- Category: multi_profile
- Reason: missing expected keyword(s): ['french', 'english', 'spanish']
- Question: Which languages does Maya Dubois speak?
- Expected: `["french", "english", "spanish"]`
- Forbidden: `["osaka", "robotics", "nantes"]`
- Answer/retrieved text: `I do not know based on memory.`

### mamba - small - distractor_resistance
- Category: distractors
- Reason: missing expected keyword(s): ['medvision']
- Question: What is Sofia Garcia's research group called?
- Expected: `["medvision"]`
- Forbidden: `["blue sofa", "chicken rice", "weather"]`
- Answer/retrieved text: `I know based on memory.`

### mamba - medium - medium_location_recall
- Category: large_simple_recall
- Reason: missing expected keyword(s): ['marseille']
- Question: Where is HanaAlpha Bernard from?
- Expected: `["marseille"]`
- Forbidden: `["blue sofa", "chicken rice"]`
- Answer/retrieved text: `Based on memory, I do not know where HanaAlpha Bernard from.`

### mamba - medium - medium_location_update
- Category: large_update
- Reason: missing expected keyword(s): ['rome']
- Question: Where is HanaAlpha Bernard from?
- Expected: `["rome"]`
- Forbidden: `["marseille"]`
- Answer/retrieved text: `Based on the provided memories, HanaAlpha Bernard from Marseille, France.`

### mamba - medium - medium_forget_secret_code
- Category: large_forget_secret_code
- Reason: retrieved an irrelevant secret-code memory after forgetting target
- Question: What is CarlosBeta Wilson's secret code?
- Expected: `["do not know", "don't know", "not know", "no memory"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: `Based on the provided memory, I do not know based on memory.  |  | The memory snippet does not contain a secret code or any information about CarlosBeta Wilson's personal life or preferences that could be used to deduce his secret code. The text only provides information about his office room and other professional affiliations.`

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
- Answer/retrieved text: `Based on the provided memories, I can infer that CarlosBeta Wilson works on data engineering projects.`
