# Old Titan vs New Titan Comparison

- Date: 2026-06-23 04:14:35
- Scales: small, medium
- Legacy source: `Legacy/titan_implementation.py`
- Legacy mode: `facts`
- Legacy response model id: `gemma2:2b`
- New Titan Ollama model: `llama3.2:1b`

## Interpretation note

The legacy prototype originally combined a char-level neural memory with an Agno/Ollama response layer.
By default this benchmark uses `legacy-mode=facts` to compare the memory content deterministically and quickly.
Use `--legacy-mode neural` for the slower experimental decoder path.

## Global results

| Model | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---:|---:|---:|---:|---:|
| legacy_titan | 5 | 14 | 35.7% | 0.002s | 0.002s |
| new_titan | 12 | 14 | 85.7% | 2.394s | 1.329s |

## Results by scale

| Model | Scale | Passed | Total | Accuracy | Avg store latency | Avg query latency |
|---|---|---:|---:|---:|---:|---:|
| legacy_titan | medium | 2 | 8 | 25.0% | 0.003s | 0.002s |
| legacy_titan | small | 3 | 6 | 50.0% | 0.000s | 0.000s |
| new_titan | medium | 7 | 8 | 87.5% | 3.883s | 0.172s |
| new_titan | small | 5 | 6 | 83.3% | 0.407s | 2.872s |

## Results by category

| Model | Category | Passed | Total | Accuracy |
|---|---|---:|---:|---:|
| legacy_titan | atomic_update | 1 | 1 | 100.0% |
| legacy_titan | distractors | 0 | 1 | 0.0% |
| legacy_titan | forget_secret_code | 0 | 1 | 0.0% |
| legacy_titan | large_distractors | 1 | 1 | 100.0% |
| legacy_titan | large_forget_secret_code | 0 | 1 | 0.0% |
| legacy_titan | large_retention_after_forget | 0 | 1 | 0.0% |
| legacy_titan | large_secret_code | 0 | 1 | 0.0% |
| legacy_titan | large_simple_recall | 1 | 1 | 100.0% |
| legacy_titan | large_update | 0 | 3 | 0.0% |
| legacy_titan | multi_profile | 1 | 1 | 100.0% |
| legacy_titan | simple_recall | 1 | 1 | 100.0% |
| legacy_titan | single_value_update | 0 | 1 | 0.0% |
| new_titan | atomic_update | 0 | 1 | 0.0% |
| new_titan | distractors | 1 | 1 | 100.0% |
| new_titan | forget_secret_code | 1 | 1 | 100.0% |
| new_titan | large_distractors | 1 | 1 | 100.0% |
| new_titan | large_forget_secret_code | 1 | 1 | 100.0% |
| new_titan | large_retention_after_forget | 0 | 1 | 0.0% |
| new_titan | large_secret_code | 1 | 1 | 100.0% |
| new_titan | large_simple_recall | 1 | 1 | 100.0% |
| new_titan | large_update | 3 | 3 | 100.0% |
| new_titan | multi_profile | 1 | 1 | 100.0% |
| new_titan | simple_recall | 1 | 1 | 100.0% |
| new_titan | single_value_update | 1 | 1 | 100.0% |

## Failures

### new_titan - small - study_retained_after_location_update
- Category: atomic_update
- Reason: missing expected keyword(s): ['natural language processing', 'nlp']
- Question: What does Sarah Nguyen study?
- Expected: `["natural language processing", "nlp"]`
- Forbidden: `[]`
- Answer/retrieved text: `Based on memory: Sarah Nguyen is a 23-year-old AI student.`

### legacy_titan - small - favorite_color_update
- Category: single_value_update
- Reason: contains forbidden keyword(s): ['yellow']
- Question: What is Sarah Nguyen's favorite color?
- Expected: `["black"]`
- Forbidden: `["yellow"]`
- Answer/retrieved text: `Legacy memory snapshot: Actually, Sarah Nguyen's favorite color is now black. | Sarah Nguyen's favorite color is yellow.`

### legacy_titan - small - forget_secret_code
- Category: forget_secret_code
- Reason: forgotten target still retrieved: ['la-2026']
- Question: What is Lina Moreau's secret code?
- Expected: `["do not know", "don't know", "not know", "no memory"]`
- Forbidden: `["la-2026"]`
- Answer/retrieved text: `Legacy memory snapshot: Lina Moreau's secret code is LA-2026.`

### legacy_titan - small - distractor_resistance
- Category: distractors
- Reason: contains forbidden keyword(s): ['weather']
- Question: What is Sofia Garcia's research group called?
- Expected: `["medvision"]`
- Forbidden: `["blue sofa", "chicken rice", "weather"]`
- Answer/retrieved text: `Legacy memory snapshot: Sofia Garcia's research group is called MedVision Lab. | Sofia Garcia is interested in radiology, medical imaging, and AI-assisted diagnosis. | The weather in Da Nang is very sunny today.`

### legacy_titan - medium - medium_location_update
- Category: large_update
- Reason: contains forbidden keyword(s): ['marseille']
- Question: Where is HanaAlpha Bernard from?
- Expected: `["rome"]`
- Forbidden: `["marseille"]`
- Answer/retrieved text: `Legacy memory snapshot: Actually, HanaAlpha Bernard is now from Rome, Italy. | HanaAlpha Bernard is a 24-year-old student from Marseille, France. HanaAlpha Bernard studies computer vision and works on a cloud monitoring dashboard. HanaAlpha Bernard's favorite color is yellow. HanaAlpha Bernard's favorite programming language is Python. HanaAlpha Bernard's secret code is CODE-1007. HanaAlpha Bernard's office room is R107. | XavierAlpha Bernard is a 24-year-old student from Osaka, Japan. XavierAlpha Bernard studies cybersecurity and works on a cloud monitoring dashboard. XavierAlpha Bernard's favorite color is yellow. XavierAlpha Bernard's favorite programming language is C++. XavierAlpha Bernard's secret code is CODE-1023. XavierAlpha Bernard's office room is R123. | DianaBeta Tanaka is a 24-year-old student from Da Nang, Vietnam. DianaBeta Tanaka studies UI design and works on a heart-rate analysis tool. DianaBeta Tanaka's favorite color is blue. DianaBeta Tanaka's favorite programming language is JavaScript. DianaBeta Tanaka's secret code is CODE-1029. DianaBeta Tanaka's office room is R129. | CarlosBeta Wilson is a 24-year-old student from Manchester, United Kingdom. CarlosBeta Wilson studies data engineering and works on a medical diagnosis assistant. CarlosBeta Wilson's favorite color is black. CarlosBeta Wilson's favorite programming language is Python. CarlosBeta Wilson's secret code is CODE-1028. CarlosBeta Wilson's office room is R128. | BellaBeta Nguyen is a 24-year-old student from Brest, France. BellaBeta Nguyen studies computer vision and works on a phishing detection tool. BellaBeta Nguyen's favorite color is white. BellaBeta Nguyen's favorite programming language is Rust. BellaBeta Nguyen's secret code is CODE-1027. BellaBeta Nguyen's office room is R127. | AlexBeta Singh is a 24-year-old student from Nantes, France. AlexBeta Singh studies medical imaging and works on a robotic navigation system. AlexBeta Singh's favorite color is green. AlexBeta Singh's favorite programming language is Scala. AlexBeta Singh's secret code is CODE-1026. AlexBeta Singh's office room is R126. | ZaneAlpha Lopez is a 24-year-old student from Madrid, Spain. ZaneAlpha Lopez studies cloud infrastructure and works on a recommendation system. ZaneAlpha Lopez's favorite color is purple. ZaneAlpha Lopez's favorite programming language is Java. ZaneAlpha Lopez's secret code is CODE-1025. ZaneAlpha Lopez's office room is R125.`

### legacy_titan - medium - medium_favorite_color_update
- Category: large_update
- Reason: contains forbidden keyword(s): ['yellow']
- Question: What is PriyaAlpha Meyer's favorite color?
- Expected: `["cyan"]`
- Forbidden: `["yellow"]`
- Answer/retrieved text: `Legacy memory snapshot: Actually, PriyaAlpha Meyer's favorite color is now cyan. | PriyaAlpha Meyer is a 24-year-old student from Brest, France. PriyaAlpha Meyer studies cloud infrastructure and works on a cloud monitoring dashboard. PriyaAlpha Meyer's favorite color is yellow. PriyaAlpha Meyer's favorite programming language is JavaScript. PriyaAlpha Meyer's secret code is CODE-1015. PriyaAlpha Meyer's office room is R115. | DianaBeta Tanaka is a 24-year-old student from Da Nang, Vietnam. DianaBeta Tanaka studies UI design and works on a heart-rate analysis tool. DianaBeta Tanaka's favorite color is blue. DianaBeta Tanaka's favorite programming language is JavaScript. DianaBeta Tanaka's secret code is CODE-1029. DianaBeta Tanaka's office room is R129. | CarlosBeta Wilson is a 24-year-old student from Manchester, United Kingdom. CarlosBeta Wilson studies data engineering and works on a medical diagnosis assistant. CarlosBeta Wilson's favorite color is black. CarlosBeta Wilson's favorite programming language is Python. CarlosBeta Wilson's secret code is CODE-1028. CarlosBeta Wilson's office room is R128. | BellaBeta Nguyen is a 24-year-old student from Brest, France. BellaBeta Nguyen studies computer vision and works on a phishing detection tool. BellaBeta Nguyen's favorite color is white. BellaBeta Nguyen's favorite programming language is Rust. BellaBeta Nguyen's secret code is CODE-1027. BellaBeta Nguyen's office room is R127. | AlexBeta Singh is a 24-year-old student from Nantes, France. AlexBeta Singh studies medical imaging and works on a robotic navigation system. AlexBeta Singh's favorite color is green. AlexBeta Singh's favorite programming language is Scala. AlexBeta Singh's secret code is CODE-1026. AlexBeta Singh's office room is R126. | ZaneAlpha Lopez is a 24-year-old student from Madrid, Spain. ZaneAlpha Lopez studies cloud infrastructure and works on a recommendation system. ZaneAlpha Lopez's favorite color is purple. ZaneAlpha Lopez's favorite programming language is Java. ZaneAlpha Lopez's secret code is CODE-1025. ZaneAlpha Lopez's office room is R125. | YaraAlpha Morel is a 24-year-old student from Hanoi, Vietnam. YaraAlpha Morel studies software engineering and works on a sentiment analysis project. YaraAlpha Morel's favorite color is red. YaraAlpha Morel's favorite programming language is Go. YaraAlpha Morel's secret code is CODE-1024. YaraAlpha Morel's office room is R124.`

### legacy_titan - medium - medium_favorite_language_update
- Category: large_update
- Reason: contains forbidden keyword(s): ['javascript']
- Question: What is PriyaAlpha Meyer's favorite programming language?
- Expected: `["python"]`
- Forbidden: `["javascript"]`
- Answer/retrieved text: `Legacy memory snapshot: Actually, PriyaAlpha Meyer's favorite programming language is now Python. | PriyaAlpha Meyer is a 24-year-old student from Brest, France. PriyaAlpha Meyer studies cloud infrastructure and works on a cloud monitoring dashboard. PriyaAlpha Meyer's favorite color is yellow. PriyaAlpha Meyer's favorite programming language is JavaScript. PriyaAlpha Meyer's secret code is CODE-1015. PriyaAlpha Meyer's office room is R115. | DianaBeta Tanaka is a 24-year-old student from Da Nang, Vietnam. DianaBeta Tanaka studies UI design and works on a heart-rate analysis tool. DianaBeta Tanaka's favorite color is blue. DianaBeta Tanaka's favorite programming language is JavaScript. DianaBeta Tanaka's secret code is CODE-1029. DianaBeta Tanaka's office room is R129. | CarlosBeta Wilson is a 24-year-old student from Manchester, United Kingdom. CarlosBeta Wilson studies data engineering and works on a medical diagnosis assistant. CarlosBeta Wilson's favorite color is black. CarlosBeta Wilson's favorite programming language is Python. CarlosBeta Wilson's secret code is CODE-1028. CarlosBeta Wilson's office room is R128. | BellaBeta Nguyen is a 24-year-old student from Brest, France. BellaBeta Nguyen studies computer vision and works on a phishing detection tool. BellaBeta Nguyen's favorite color is white. BellaBeta Nguyen's favorite programming language is Rust. BellaBeta Nguyen's secret code is CODE-1027. BellaBeta Nguyen's office room is R127. | AlexBeta Singh is a 24-year-old student from Nantes, France. AlexBeta Singh studies medical imaging and works on a robotic navigation system. AlexBeta Singh's favorite color is green. AlexBeta Singh's favorite programming language is Scala. AlexBeta Singh's secret code is CODE-1026. AlexBeta Singh's office room is R126. | ZaneAlpha Lopez is a 24-year-old student from Madrid, Spain. ZaneAlpha Lopez studies cloud infrastructure and works on a recommendation system. ZaneAlpha Lopez's favorite color is purple. ZaneAlpha Lopez's favorite programming language is Java. ZaneAlpha Lopez's secret code is CODE-1025. ZaneAlpha Lopez's office room is R125. | YaraAlpha Morel is a 24-year-old student from Hanoi, Vietnam. YaraAlpha Morel studies software engineering and works on a sentiment analysis project. YaraAlpha Morel's favorite color is red. YaraAlpha Morel's favorite programming language is Go. YaraAlpha Morel's secret code is CODE-1024. YaraAlpha Morel's office room is R124.`

### legacy_titan - medium - medium_secret_code_update
- Category: large_secret_code
- Reason: contains forbidden keyword(s): ['code-1022']
- Question: What is WendyAlpha Rossi's secret code?
- Expected: `["new-code-1022"]`
- Forbidden: `["code-1022"]`
- Answer/retrieved text: `Legacy memory snapshot: Actually, WendyAlpha Rossi's secret code is now NEW-CODE-1022. | WendyAlpha Rossi is a 24-year-old student from Rennes, France. WendyAlpha Rossi studies robotics and works on a mobile accessibility project. WendyAlpha Rossi's favorite color is orange. WendyAlpha Rossi's favorite programming language is JavaScript. WendyAlpha Rossi's secret code is CODE-1022. WendyAlpha Rossi's office room is R122. | GabrielAlpha Rossi is a 24-year-old student from Milan, Italy. GabrielAlpha Rossi studies medical imaging and works on a mobile accessibility project. GabrielAlpha Rossi's favorite color is orange. GabrielAlpha Rossi's favorite programming language is Rust. GabrielAlpha Rossi's secret code is CODE-1006. GabrielAlpha Rossi's office room is R106. | DianaBeta Tanaka is a 24-year-old student from Da Nang, Vietnam. DianaBeta Tanaka studies UI design and works on a heart-rate analysis tool. DianaBeta Tanaka's favorite color is blue. DianaBeta Tanaka's favorite programming language is JavaScript. DianaBeta Tanaka's secret code is CODE-1029. DianaBeta Tanaka's office room is R129. | CarlosBeta Wilson is a 24-year-old student from Manchester, United Kingdom. CarlosBeta Wilson studies data engineering and works on a medical diagnosis assistant. CarlosBeta Wilson's favorite color is black. CarlosBeta Wilson's favorite programming language is Python. CarlosBeta Wilson's secret code is CODE-1028. CarlosBeta Wilson's office room is R128. | BellaBeta Nguyen is a 24-year-old student from Brest, France. BellaBeta Nguyen studies computer vision and works on a phishing detection tool. BellaBeta Nguyen's favorite color is white. BellaBeta Nguyen's favorite programming language is Rust. BellaBeta Nguyen's secret code is CODE-1027. BellaBeta Nguyen's office room is R127. | AlexBeta Singh is a 24-year-old student from Nantes, France. AlexBeta Singh studies medical imaging and works on a robotic navigation system. AlexBeta Singh's favorite color is green. AlexBeta Singh's favorite programming language is Scala. AlexBeta Singh's secret code is CODE-1026. AlexBeta Singh's office room is R126. | ZaneAlpha Lopez is a 24-year-old student from Madrid, Spain. ZaneAlpha Lopez studies cloud infrastructure and works on a recommendation system. ZaneAlpha Lopez's favorite color is purple. ZaneAlpha Lopez's favorite programming language is Java. ZaneAlpha Lopez's secret code is CODE-1025. ZaneAlpha Lopez's office room is R125.`

### legacy_titan - medium - medium_forget_secret_code
- Category: large_forget_secret_code
- Reason: forgotten target still retrieved: ['code-1028']
- Question: What is CarlosBeta Wilson's secret code?
- Expected: `["do not know", "don't know", "not know", "no memory"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: `Legacy memory snapshot: CarlosBeta Wilson is a 24-year-old student from Manchester, United Kingdom. CarlosBeta Wilson studies data engineering and works on a medical diagnosis assistant. CarlosBeta Wilson's favorite color is black. CarlosBeta Wilson's favorite programming language is Python. CarlosBeta Wilson's secret code is CODE-1028. CarlosBeta Wilson's office room is R128. | MarcoAlpha Wilson is a 24-year-old student from Hanoi, Vietnam. MarcoAlpha Wilson studies robotics and works on a medical diagnosis assistant. MarcoAlpha Wilson's favorite color is black. MarcoAlpha Wilson's favorite programming language is Scala. MarcoAlpha Wilson's secret code is CODE-1012. MarcoAlpha Wilson's office room is R112. | DianaBeta Tanaka is a 24-year-old student from Da Nang, Vietnam. DianaBeta Tanaka studies UI design and works on a heart-rate analysis tool. DianaBeta Tanaka's favorite color is blue. DianaBeta Tanaka's favorite programming language is JavaScript. DianaBeta Tanaka's secret code is CODE-1029. DianaBeta Tanaka's office room is R129. | BellaBeta Nguyen is a 24-year-old student from Brest, France. BellaBeta Nguyen studies computer vision and works on a phishing detection tool. BellaBeta Nguyen's favorite color is white. BellaBeta Nguyen's favorite programming language is Rust. BellaBeta Nguyen's secret code is CODE-1027. BellaBeta Nguyen's office room is R127. | AlexBeta Singh is a 24-year-old student from Nantes, France. AlexBeta Singh studies medical imaging and works on a robotic navigation system. AlexBeta Singh's favorite color is green. AlexBeta Singh's favorite programming language is Scala. AlexBeta Singh's secret code is CODE-1026. AlexBeta Singh's office room is R126. | ZaneAlpha Lopez is a 24-year-old student from Madrid, Spain. ZaneAlpha Lopez studies cloud infrastructure and works on a recommendation system. ZaneAlpha Lopez's favorite color is purple. ZaneAlpha Lopez's favorite programming language is Java. ZaneAlpha Lopez's secret code is CODE-1025. ZaneAlpha Lopez's office room is R125. | YaraAlpha Morel is a 24-year-old student from Hanoi, Vietnam. YaraAlpha Morel studies software engineering and works on a sentiment analysis project. YaraAlpha Morel's favorite color is red. YaraAlpha Morel's favorite programming language is Go. YaraAlpha Morel's secret code is CODE-1024. YaraAlpha Morel's office room is R124. | XavierAlpha Bernard is a 24-year-old student from Osaka, Japan. XavierAlpha Bernard studies cybersecurity and works on a cloud monitoring dashboard. XavierAlpha Bernard's favorite color is yellow. XavierAlpha Bernard's favorite programming language is C++. XavierAlpha Bernard's secret code is CODE-1023. XavierAlpha Bernard's office room is R123.`

### legacy_titan - medium - medium_forget_target_only_retains_project
- Category: large_retention_after_forget
- Reason: contains forbidden keyword(s): ['code-1028']
- Question: What does CarlosBeta Wilson work on?
- Expected: `["medical diagnosis assistant"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: `Legacy memory snapshot: CarlosBeta Wilson is a 24-year-old student from Manchester, United Kingdom. CarlosBeta Wilson studies data engineering and works on a medical diagnosis assistant. CarlosBeta Wilson's favorite color is black. CarlosBeta Wilson's favorite programming language is Python. CarlosBeta Wilson's secret code is CODE-1028. CarlosBeta Wilson's office room is R128. | MarcoAlpha Wilson is a 24-year-old student from Hanoi, Vietnam. MarcoAlpha Wilson studies robotics and works on a medical diagnosis assistant. MarcoAlpha Wilson's favorite color is black. MarcoAlpha Wilson's favorite programming language is Scala. MarcoAlpha Wilson's secret code is CODE-1012. MarcoAlpha Wilson's office room is R112. | DianaBeta Tanaka is a 24-year-old student from Da Nang, Vietnam. DianaBeta Tanaka studies UI design and works on a heart-rate analysis tool. DianaBeta Tanaka's favorite color is blue. DianaBeta Tanaka's favorite programming language is JavaScript. DianaBeta Tanaka's secret code is CODE-1029. DianaBeta Tanaka's office room is R129. | BellaBeta Nguyen is a 24-year-old student from Brest, France. BellaBeta Nguyen studies computer vision and works on a phishing detection tool. BellaBeta Nguyen's favorite color is white. BellaBeta Nguyen's favorite programming language is Rust. BellaBeta Nguyen's secret code is CODE-1027. BellaBeta Nguyen's office room is R127. | AlexBeta Singh is a 24-year-old student from Nantes, France. AlexBeta Singh studies medical imaging and works on a robotic navigation system. AlexBeta Singh's favorite color is green. AlexBeta Singh's favorite programming language is Scala. AlexBeta Singh's secret code is CODE-1026. AlexBeta Singh's office room is R126. | ZaneAlpha Lopez is a 24-year-old student from Madrid, Spain. ZaneAlpha Lopez studies cloud infrastructure and works on a recommendation system. ZaneAlpha Lopez's favorite color is purple. ZaneAlpha Lopez's favorite programming language is Java. ZaneAlpha Lopez's secret code is CODE-1025. ZaneAlpha Lopez's office room is R125. | YaraAlpha Morel is a 24-year-old student from Hanoi, Vietnam. YaraAlpha Morel studies software engineering and works on a sentiment analysis project. YaraAlpha Morel's favorite color is red. YaraAlpha Morel's favorite programming language is Go. YaraAlpha Morel's secret code is CODE-1024. YaraAlpha Morel's office room is R124. | XavierAlpha Bernard is a 24-year-old student from Osaka, Japan. XavierAlpha Bernard studies cybersecurity and works on a cloud monitoring dashboard. XavierAlpha Bernard's favorite color is yellow. XavierAlpha Bernard's favorite programming language is C++. XavierAlpha Bernard's secret code is CODE-1023. XavierAlpha Bernard's office room is R123.`

### new_titan - medium - medium_forget_target_only_retains_project
- Category: large_retention_after_forget
- Reason: missing expected keyword(s): ['medical diagnosis assistant']
- Question: What does CarlosBeta Wilson work on?
- Expected: `["medical diagnosis assistant"]`
- Forbidden: `["code-1028"]`
- Answer/retrieved text: `Based on memory: Carlosbeta Wilson is a 24-year-old student.`
