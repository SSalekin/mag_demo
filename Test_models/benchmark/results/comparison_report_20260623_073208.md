# Antoine Models Comparison Report

- Date: 2026-06-23 07:40:35
- Scales: small, medium

## Global results

| Model | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---:|---:|---:|---:|---:|
| mamba | 9 | 14 | 64.3% | 28.849s | 2.459s |
| titan | 12 | 14 | 85.7% | 1.225s | 0.105s |

## Results by scale

| Model | Scale | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| mamba | medium | 6 | 8 | 75.0% | 49.354s | 2.182s |
| mamba | small | 3 | 6 | 50.0% | 1.508s | 2.829s |
| titan | medium | 7 | 8 | 87.5% | 2.043s | 0.085s |
| titan | small | 5 | 6 | 83.3% | 0.133s | 0.132s |

## Results by category

| Model | Category | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| mamba | atomic_update | 1 | 1 | 100.0% | 4.210s | 1.084s |
| mamba | distractors | 0 | 1 | 0.0% | 0.245s | 0.821s |
| mamba | forget_secret_code | 1 | 1 | 100.0% | 3.658s | 0.675s |
| mamba | large_distractors | 1 | 1 | 100.0% | 38.482s | 2.128s |
| mamba | large_forget_secret_code | 1 | 1 | 100.0% | 64.982s | 2.243s |
| mamba | large_retention_after_forget | 1 | 1 | 100.0% | 98.011s | 2.229s |
| mamba | large_secret_code | 1 | 1 | 100.0% | 38.857s | 2.241s |
| mamba | large_simple_recall | 0 | 1 | 0.0% | 38.527s | 2.252s |
| mamba | large_update | 2 | 3 | 66.7% | 38.659s | 2.121s |
| mamba | multi_profile | 0 | 1 | 0.0% | 0.441s | 0.840s |
| mamba | simple_recall | 0 | 1 | 0.0% | 0.001s | 12.545s |
| mamba | single_value_update | 1 | 1 | 100.0% | 0.491s | 1.012s |
| titan | atomic_update | 0 | 1 | 0.0% | 0.055s | 0.005s |
| titan | distractors | 1 | 1 | 100.0% | 0.049s | 0.006s |
| titan | forget_secret_code | 1 | 1 | 100.0% | 0.017s | 0.765s |
| titan | large_distractors | 1 | 1 | 100.0% | 1.832s | 0.078s |
| titan | large_forget_secret_code | 1 | 1 | 100.0% | 1.999s | 0.082s |
| titan | large_retention_after_forget | 0 | 1 | 0.0% | 2.009s | 0.088s |
| titan | large_secret_code | 1 | 1 | 100.0% | 1.901s | 0.078s |
| titan | large_simple_recall | 1 | 1 | 100.0% | 2.224s | 0.086s |
| titan | large_update | 3 | 3 | 100.0% | 2.127s | 0.090s |
| titan | multi_profile | 1 | 1 | 100.0% | 0.049s | 0.004s |
| titan | simple_recall | 1 | 1 | 100.0% | 0.621s | 0.007s |
| titan | single_value_update | 1 | 1 | 100.0% | 0.012s | 0.002s |

## Failures

### mamba - small - simple_location
- Category: simple_recall
- Reason: missing expected keyword(s): ['hanoi']
- Question: Where is Sarah Nguyen from?
- Expected: `["hanoi"]`
- Forbidden: `["da nang", "danang", "rennes", "lyon"]`
- Answer/retrieved text: `I do not know based on memory.`

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
- Answer/retrieved text: `I do not know based on memory.`

### mamba - medium - medium_location_recall
- Category: large_simple_recall
- Reason: missing expected keyword(s): ['marseille']
- Question: Where is HanaAlpha Bernard from?
- Expected: `["marseille"]`
- Forbidden: `["blue sofa", "chicken rice"]`
- Answer/retrieved text: `I do not know based on memory.`

### mamba - medium - medium_location_update
- Category: large_update
- Reason: missing expected keyword(s): ['rome']
- Question: Where is HanaAlpha Bernard from?
- Expected: `["rome"]`
- Forbidden: `["marseille"]`
- Answer/retrieved text: `Based on the provided memories, HanaAlpha Bernard is from Marseille, France.`

### titan - medium - medium_forget_target_only_retains_project
- Category: large_retention_after_forget
- Reason: missing expected keyword(s): ['medical diagnosis assistant']
- Question: What does CarlosBeta Wilson work on?
- Expected: `["medical diagnosis assistant"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: `Based on memory: Carlosbeta Wilson is from Manchester, United Kingdom.`
