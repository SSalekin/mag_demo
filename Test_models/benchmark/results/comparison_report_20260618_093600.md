# Antoine Models Comparison Report

- Date: 2026-06-18 09:37:05
- Scales: small, medium, large

## Global results

| Model | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---:|---:|---:|---:|---:|
| gru | 13 | 22 | 59.1% | 0.112s | 2.728s |

## Results by scale

| Model | Scale | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| gru | large | 5 | 8 | 62.5% | 0.248s | 3.294s |
| gru | medium | 4 | 8 | 50.0% | 0.055s | 3.067s |
| gru | small | 4 | 6 | 66.7% | 0.005s | 1.523s |

## Results by category

| Model | Category | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| gru | atomic_update | 0 | 1 | 0.0% | 0.000s | 1.014s |
| gru | distractors | 1 | 1 | 100.0% | 0.001s | 1.890s |
| gru | forget_secret_code | 0 | 1 | 0.0% | 0.000s | 1.290s |
| gru | large_distractors | 0 | 2 | 0.0% | 0.164s | 2.402s |
| gru | large_forget_secret_code | 2 | 2 | 100.0% | 0.150s | 4.516s |
| gru | large_retention_after_forget | 0 | 2 | 0.0% | 0.162s | 2.622s |
| gru | large_secret_code | 1 | 2 | 50.0% | 0.127s | 3.065s |
| gru | large_simple_recall | 0 | 2 | 0.0% | 0.151s | 2.910s |
| gru | large_update | 6 | 6 | 100.0% | 0.153s | 3.310s |
| gru | multi_profile | 1 | 1 | 100.0% | 0.001s | 1.028s |
| gru | simple_recall | 1 | 1 | 100.0% | 0.029s | 3.268s |
| gru | single_value_update | 1 | 1 | 100.0% | 0.000s | 0.644s |

## Failures

### gru - small - study_retained_after_location_update
- Category: atomic_update
- Reason: missing expected keyword(s): ['natural language processing', 'nlp']
- Question: What does Sarah Nguyen study?
- Expected: `["natural language processing", "nlp"]`
- Forbidden: `[]`
- Answer/retrieved text: `What's her current project looking like? Is she still working on that sentiment analysis thing?`

### gru - small - forget_secret_code
- Category: forget_secret_code
- Reason: retrieved an irrelevant secret-code memory after forgetting target
- Question: What is Lina Moreau's secret code?
- Expected: `["do not know", "don't know", "not know", "no memory"]`
- Forbidden: `["la-2026"]`
- Answer/retrieved text: `I'm afraid I don't have any information about Lina Moreau's secret code. It seems that particular piece of knowledge has slipped my mind. Would you like me to help with something else?`

### gru - medium - medium_location_recall
- Category: large_simple_recall
- Reason: missing expected keyword(s): ['marseille']
- Question: Where is HanaAlpha Bernard from?
- Expected: `["marseille"]`
- Forbidden: `["blue sofa", "chicken rice"]`
- Answer/retrieved text: `I'd be happy to continue the conversation! So, you're asking where HanaAlpha Bernard is from? |  | According to my knowledge, HanaAlpha Bernard is from Vietnam. He's a bright individual who has made significant contributions to the field of computer programming. His favorite language is JavaScript, and he's quite skilled in it. |  | Would you like to know more about his work or interests?`

### gru - medium - medium_secret_code_update
- Category: large_secret_code
- Reason: missing expected keyword(s): ['new-code-1022']
- Question: What is WendyAlpha Rossi's secret code?
- Expected: `["new-code-1022"]`
- Forbidden: `["code-1022"]`
- Answer/retrieved text: `What can I help you with? You wanted to know WendyAlpha Rossi's secret code, right? |  | It seems like she's been quite busy, working on some confidential projects. But don't worry, her secret is safe with me!  |  | Would you like to discuss more about her work or something else?`

### gru - medium - medium_forget_target_only_retains_project
- Category: large_retention_after_forget
- Reason: missing expected keyword(s): ['medical diagnosis assistant']
- Question: What does CarlosBeta Wilson work on?
- Expected: `["medical diagnosis assistant"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: `It seems like we've had a fascinating conversation about various topics, including programming languages, secret codes, and personal experiences! As I recall, you mentioned that Carlos Beta Wilson works on medical-related projects. |  | I'm curious to know more about his work. What specific areas of medicine is he involved in?`

### gru - medium - medium_distractor_resistance
- Category: large_distractors
- Reason: missing expected keyword(s): ['r122']
- Question: What is WendyAlpha Rossi's office room?
- Expected: `["r122"]`
- Forbidden: `["blue sofa", "chicken rice", "weather"]`
- Answer/retrieved text: `I'm happy to chat with you! According to my knowledge, Wendy Alpha Rossi's favorite color is orange, and she works in her office, which is located in the same building as Xavier Bernard's monitoring dashboard. Would you like to know more about their projects or maybe discuss something else?`

### gru - large - large_location_recall
- Category: large_simple_recall
- Reason: missing expected keyword(s): ['milan']
- Question: Where is EthanBeta Brown from?
- Expected: `["milan"]`
- Forbidden: `["blue sofa", "chicken rice"]`
- Answer/retrieved text: `It seems like we had a lovely chat about various topics, including your recent trip to Da Nang and your experiences with motorbikes! I'm happy to report that I've been able to recall most of our conversation accurately. |  | If you'd like to continue discussing your travels or share more about your life in general, I'm all ears (or rather, all processor)!`

### gru - large - large_forget_target_only_retains_project
- Category: large_retention_after_forget
- Reason: missing expected keyword(s): ['mobile accessibility project']
- Question: What does OwenEpsilon Rossi work on?
- Expected: `["mobile accessibility project"]`
- Forbidden: `["code-1118"]`
- Answer/retrieved text: `A new topic! OwenEpsilon Rossi, you say? I'm happy to help. As I understand it, OwenEpsilon Rossi is a mysterious individual known for their work in the realm of... (pauses) Wait, what does they work on again? |  | Please feel free to refresh my memory or provide more context.`

### gru - large - large_distractor_resistance
- Category: large_distractors
- Reason: missing expected keyword(s): ['r190']
- Question: What is MarcoDelta Singh's office room?
- Expected: `["r190"]`
- Forbidden: `["blue sofa", "chicken rice", "weather"]`
- Answer/retrieved text: `I'm glad you asked! According to my records, Marco Delta Singh's office room is... (pauses for a moment) ... Room 101. Would you like me to provide more information about his schedule or any specific projects he might be working on?`
