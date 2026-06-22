# Antoine Models Comparison Report

- Date: 2026-06-18 09:14:26
- Scales: small, medium

## Global results

| Model | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---:|---:|---:|---:|---:|
| transformer | 11 | 14 | 78.6% | 1.926s | 1.757s |

## Results by scale

| Model | Scale | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| transformer | medium | 5 | 8 | 62.5% | 3.360s | 1.607s |
| transformer | small | 6 | 6 | 100.0% | 0.014s | 1.955s |

## Results by category

| Model | Category | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| transformer | atomic_update | 1 | 1 | 100.0% | 0.007s | 0.877s |
| transformer | distractors | 1 | 1 | 100.0% | 0.024s | 0.752s |
| transformer | forget_secret_code | 1 | 1 | 100.0% | 0.007s | 0.562s |
| transformer | large_distractors | 1 | 1 | 100.0% | 3.184s | 1.128s |
| transformer | large_forget_secret_code | 0 | 1 | 0.0% | 3.408s | 1.270s |
| transformer | large_retention_after_forget | 1 | 1 | 100.0% | 3.060s | 1.147s |
| transformer | large_secret_code | 0 | 1 | 0.0% | 3.334s | 1.829s |
| transformer | large_simple_recall | 1 | 1 | 100.0% | 3.396s | 3.283s |
| transformer | large_update | 2 | 3 | 66.7% | 3.498s | 1.401s |
| transformer | multi_profile | 1 | 1 | 100.0% | 0.013s | 0.733s |
| transformer | simple_recall | 1 | 1 | 100.0% | 0.023s | 8.230s |
| transformer | single_value_update | 1 | 1 | 100.0% | 0.008s | 0.579s |

## Failures

### transformer - medium - medium_location_update
- Category: large_update
- Reason: contains forbidden keyword(s): ['marseille']
- Question: Where is HanaAlpha Bernard from?
- Expected: `["rome"]`
- Forbidden: `["marseille"]`
- Answer/retrieved text: `HanaAlpha Bernard is from Rome, Italy. (Note: This information was previously remembered as being from Marseille, France. It has been updated to reflect the new information.)`

### transformer - medium - medium_secret_code_update
- Category: large_secret_code
- Reason: contains forbidden keyword(s): ['code-1022']
- Question: What is WendyAlpha Rossi's secret code?
- Expected: `["new-code-1022"]`
- Forbidden: `["code-1022"]`
- Answer/retrieved text: `According to the information I remembered, WendyAlpha Rossi's original secret code was CODE-1022. However, you recently informed me that her new secret code is actually NEW-CODE-1022.`

### transformer - medium - medium_forget_secret_code
- Category: large_forget_secret_code
- Reason: forgotten target still retrieved: ['code-1028']
- Question: What is CarlosBeta Wilson's secret code?
- Expected: `["do not know", "don't know", "not know", "no memory"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: `I've been reminded! CarlosBeta Wilson's secret code is CODE-1028.`
