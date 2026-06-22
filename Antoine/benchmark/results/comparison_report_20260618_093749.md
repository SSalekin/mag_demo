# Antoine Models Comparison Report

- Date: 2026-06-18 09:38:55
- Scales: small, medium, large

## Global results

| Model | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---:|---:|---:|---:|---:|
| gnn | 13 | 22 | 59.1% | 0.160s | 2.754s |

## Results by scale

| Model | Scale | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| gnn | large | 3 | 8 | 37.5% | 0.376s | 3.753s |
| gnn | medium | 5 | 8 | 62.5% | 0.061s | 2.997s |
| gnn | small | 5 | 6 | 83.3% | 0.004s | 1.097s |

## Results by category

| Model | Category | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| gnn | atomic_update | 1 | 1 | 100.0% | 0.000s | 1.332s |
| gnn | distractors | 1 | 1 | 100.0% | 0.001s | 1.181s |
| gnn | forget_secret_code | 0 | 1 | 0.0% | 0.000s | 1.316s |
| gnn | large_distractors | 1 | 2 | 50.0% | 0.203s | 3.572s |
| gnn | large_forget_secret_code | 0 | 2 | 0.0% | 0.239s | 3.131s |
| gnn | large_retention_after_forget | 0 | 2 | 0.0% | 0.211s | 3.688s |
| gnn | large_secret_code | 2 | 2 | 100.0% | 0.235s | 3.053s |
| gnn | large_simple_recall | 0 | 2 | 0.0% | 0.218s | 3.780s |
| gnn | large_update | 5 | 6 | 83.3% | 0.215s | 3.258s |
| gnn | multi_profile | 1 | 1 | 100.0% | 0.000s | 0.782s |
| gnn | simple_recall | 1 | 1 | 100.0% | 0.023s | 1.316s |
| gnn | single_value_update | 1 | 1 | 100.0% | 0.000s | 0.658s |

## Failures

### gnn - small - forget_secret_code
- Category: forget_secret_code
- Reason: retrieved an irrelevant secret-code memory after forgetting target
- Question: What is Lina Moreau's secret code?
- Expected: `["do not know", "don't know", "not know", "no memory"]`
- Forbidden: `["la-2026"]`
- Answer/retrieved text: `I'm happy to help! However, I need more context to answer your question accurately. Can you please provide more information about Lina Moreau or what you're trying to achieve with her secret code?`

### gnn - medium - medium_location_recall
- Category: large_simple_recall
- Reason: missing expected keyword(s): ['marseille']
- Question: Where is HanaAlpha Bernard from?
- Expected: `["marseille"]`
- Forbidden: `["blue sofa", "chicken rice"]`
- Answer/retrieved text: `I'm happy to help! So, you're asking about HanaAlpha Bernard's origin? Let me see... Ah yes, I remember! HanaAlpha Bernard is from France.`

### gnn - medium - medium_forget_secret_code
- Category: large_forget_secret_code
- Reason: retrieved an irrelevant secret-code memory after forgetting target
- Question: What is CarlosBeta Wilson's secret code?
- Expected: `["do not know", "don't know", "not know", "no memory"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: `I'm happy to continue our conversation! However, I need to clarify that I don't have any information about CarlosBeta Wilson's secret code since you asked me to forget it earlier. |  | If you'd like to share some new information or ask a question, I'm here to assist you. Please feel free to start a new topic or follow up on our previous conversation.`

### gnn - medium - medium_forget_target_only_retains_project
- Category: large_retention_after_forget
- Reason: missing expected keyword(s): ['medical diagnosis assistant']
- Question: What does CarlosBeta Wilson work on?
- Expected: `["medical diagnosis assistant"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: `I think there might be some confusion! Since you asked me to remember information about Carlos Beta Wilson, but then told me to forget it, I'm a bit puzzled. |  | However, before we move forward, let's clarify that we didn't actually forget anything, did we? |  | So, if you're ready to continue the conversation, I'd be happy to help. What would you like to talk about or remember?`

### gnn - large - large_location_recall
- Category: large_simple_recall
- Reason: missing expected keyword(s): ['milan']
- Question: Where is EthanBeta Brown from?
- Expected: `["milan"]`
- Forbidden: `["blue sofa", "chicken rice"]`
- Answer/retrieved text: `What a fascinating topic! Gabriel, Wilson, and Ivan Rossi were all quite knowledgeable about programming languages like Java and Python, weren't they? Rosa was particularly fond of coding, I recall. |  | Wendy, would you like to share some of your favorite programming concepts or tips with us? Khan, Meyer, Singh, and Priya have all been known to discuss the intricacies of code optimization.  |  | Ah, yes! Marco, can you tell us more about your experiences with robotic systems and artificial intelligence? Dub, Lena, and Kar were all very interested in exploring those areas. |  | It seems we've had quite a lively discussion on various programming topics. Would anyone like to share some new insights or discoveries they've made recently?`

### gnn - large - large_favorite_color_update
- Category: large_update
- Reason: missing expected keyword(s): ['cyan']
- Question: What is IvanGamma Wilson's favorite color?
- Expected: `["cyan"]`
- Forbidden: `["black"]`
- Answer/retrieved text: `I think there may be some confusion here! IvanGamma Wilson's favorite color was actually white, wasn't it?`

### gnn - large - large_forget_secret_code
- Category: large_forget_secret_code
- Reason: retrieved an irrelevant secret-code memory after forgetting target
- Question: What is OwenEpsilon Rossi's secret code?
- Expected: `["do not know", "don't know", "not know", "no memory"]`
- Forbidden: `["code-1118"]`
- Answer/retrieved text: `I'm happy to help! However, I must inform you that OwenEpsilon Rossi's secret code seems to be missing from my knowledge base. Can you please remind me what it is? |  | (And by the way, how are you doing today?)`

### gnn - large - large_forget_target_only_retains_project
- Category: large_retention_after_forget
- Reason: missing expected keyword(s): ['mobile accessibility project']
- Question: What does OwenEpsilon Rossi work on?
- Expected: `["mobile accessibility project"]`
- Forbidden: `["code-1118"]`
- Answer/retrieved text: `It seems we're in a bit of an interesting situation here! So, OwenEpsilon Rossi, you say? Well, let me see what I can find out about him... *ahem* Ah yes! According to my knowledge, OwenEpsilon Rossi is a team leader working on some top-secret projects. He's quite the brilliant mind, always tinkering with new codes and algorithms. |  | Would you like to know more about his work or perhaps discuss something else entirely?`

### gnn - large - large_distractor_resistance
- Category: large_distractors
- Reason: missing expected keyword(s): ['r190']
- Question: What is MarcoDelta Singh's office room?
- Expected: `["r190"]`
- Forbidden: `["blue sofa", "chicken rice", "weather"]`
- Answer/retrieved text: `I'm happy to help! However, I need a bit more context before I can provide an accurate answer. Could you please remind me of what we were discussing or what MarcoDelta Singh's office room refers to? Was it related to a specific project, person, or topic? |  | From my understanding, MarcoDelta Singh is not mentioned in the provided memory buffer, so I'm a bit unsure about how to proceed. If you could provide more context or clarify the question, I'll do my best to assist you!`
