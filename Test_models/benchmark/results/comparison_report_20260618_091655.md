# Antoine Models Comparison Report

- Date: 2026-06-18 09:29:10
- Scales: small, medium, large

## Global results

| Model | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---:|---:|---:|---:|---:|
| transformer | 11 | 22 | 50.0% | 26.212s | 6.587s |

## Results by scale

| Model | Scale | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| transformer | large | 1 | 8 | 12.5% | 68.881s | 16.038s |
| transformer | medium | 5 | 8 | 62.5% | 3.195s | 1.499s |
| transformer | small | 5 | 6 | 83.3% | 0.012s | 0.769s |

## Results by category

| Model | Category | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| transformer | atomic_update | 1 | 1 | 100.0% | 0.006s | 0.540s |
| transformer | distractors | 1 | 1 | 100.0% | 0.021s | 0.673s |
| transformer | forget_secret_code | 0 | 1 | 0.0% | 0.007s | 1.053s |
| transformer | large_distractors | 1 | 2 | 50.0% | 37.933s | 8.549s |
| transformer | large_forget_secret_code | 1 | 2 | 50.0% | 40.033s | 8.150s |
| transformer | large_retention_after_forget | 1 | 2 | 50.0% | 36.920s | 8.053s |
| transformer | large_secret_code | 0 | 2 | 0.0% | 33.677s | 8.523s |
| transformer | large_simple_recall | 1 | 2 | 50.0% | 31.251s | 9.548s |
| transformer | large_update | 2 | 6 | 33.3% | 36.163s | 9.108s |
| transformer | multi_profile | 1 | 1 | 100.0% | 0.009s | 0.674s |
| transformer | simple_recall | 1 | 1 | 100.0% | 0.020s | 1.051s |
| transformer | single_value_update | 1 | 1 | 100.0% | 0.008s | 0.623s |

## Failures

### transformer - small - forget_secret_code
- Category: forget_secret_code
- Reason: retrieved an irrelevant secret-code memory after forgetting target
- Question: What is Lina Moreau's secret code?
- Expected: `["do not know", "don't know", "not know", "no memory"]`
- Forbidden: `["la-2026"]`
- Answer/retrieved text: `I don't recall any information about Lina Moreau's secret code. My context window has been erased, and I no longer retain that information.`

### transformer - medium - medium_favorite_color_update
- Category: large_update
- Reason: contains forbidden keyword(s): ['yellow']
- Question: What is PriyaAlpha Meyer's favorite color?
- Expected: `["cyan"]`
- Forbidden: `["yellow"]`
- Answer/retrieved text: `According to the previous information, PriyaAlpha Meyer's favorite color is yellow. However, you have also informed me that PriyaAlpha Meyer's favorite color is now cyan. Therefore, PriyaAlpha Meyer's favorite color is actually cyan.`

### transformer - medium - medium_secret_code_update
- Category: large_secret_code
- Reason: contains forbidden keyword(s): ['code-1022']
- Question: What is WendyAlpha Rossi's secret code?
- Expected: `["new-code-1022"]`
- Forbidden: `["code-1022"]`
- Answer/retrieved text: `According to the information I remembered earlier, WendyAlpha Rossi's secret code is CODE-1022. However, you just informed me that it has been updated to NEW-CODE-1022. So, as of now, WendyAlpha Rossi's secret code is indeed NEW-CODE-1022.`

### transformer - medium - medium_forget_secret_code
- Category: large_forget_secret_code
- Reason: forgotten target still retrieved: ['code-1028']
- Question: What is CarlosBeta Wilson's secret code?
- Expected: `["do not know", "don't know", "not know", "no memory"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: `I've remembered! CarlosBeta Wilson's secret code is CODE-1028.`

### transformer - large - large_location_recall
- Category: large_simple_recall
- Reason: missing expected keyword(s): ['milan']
- Question: Where is EthanBeta Brown from?
- Expected: `["milan"]`
- Forbidden: `["blue sofa", "chicken rice"]`
- Answer/retrieved text: `I'm just an AI, I don't have any information about`

### transformer - large - large_location_update
- Category: large_update
- Reason: missing expected keyword(s): ['rome']
- Question: Where is EthanBeta Brown from?
- Expected: `["rome"]`
- Forbidden: `["milan"]`
- Answer/retrieved text: `According to the information you provided earlier, EthanBeta Brown is actually from Da N`

### transformer - large - large_favorite_color_update
- Category: large_update
- Reason: missing expected keyword(s): ['cyan']
- Question: What is IvanGamma Wilson's favorite color?
- Expected: `["cyan"]`
- Forbidden: `["black"]`
- Answer/retrieved text: `According to the information you provided earlier, IvanGamma Wilson's favorite color`

### transformer - large - large_favorite_language_update
- Category: large_update
- Reason: missing expected keyword(s): ['python']
- Question: What is IvanGamma Wilson's favorite programming language?
- Expected: `["python"]`
- Forbidden: `["java"]`
- Answer/retrieved text: `According to the information you provided earlier, IvanGamma Wilson's`

### transformer - large - large_secret_code_update
- Category: large_secret_code
- Reason: missing expected keyword(s): ['new-code-1090']
- Question: What is MarcoDelta Singh's secret code?
- Expected: `["new-code-1090"]`
- Forbidden: `["code-1090"]`
- Answer/retrieved text: `MarcoDelta Singh's secret code is NEW-C`

### transformer - large - large_forget_target_only_retains_project
- Category: large_retention_after_forget
- Reason: missing expected keyword(s): ['mobile accessibility project']
- Question: What does OwenEpsilon Rossi work on?
- Expected: `["mobile accessibility project"]`
- Forbidden: `["code-1118"]`
- Answer/retrieved text: `I've been trained on a vast amount of text data, including information about various individuals and their work.`

### transformer - large - large_distractor_resistance
- Category: large_distractors
- Reason: missing expected keyword(s): ['r190']
- Question: What is MarcoDelta Singh's office room?
- Expected: `["r190"]`
- Forbidden: `["blue sofa", "chicken rice", "weather"]`
- Answer/retrieved text: `I don't have any information about MarcoDelta Singh or`
