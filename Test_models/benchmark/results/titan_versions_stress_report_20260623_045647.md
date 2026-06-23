# Stress Comparison — Old Titan vs New Titan

- Date: 2026-06-23 04:56:47
- Scales: small
- Legacy source: `Legacy/titan_implementation.py`
- Legacy mode: `facts`
- Legacy response model id: `gemma2:2b`
- New Titan Ollama model: `llama3.2:1b`
- Capacity: `8192`

## Interpretation note

This script compares the old and new Titan versions on the same stress cases used for the Mamba/Titan stress benchmark.
By default, the legacy version runs in `facts` mode to compare memory content directly and avoid depending on its old LLM response layer.
Because the current project evaluates the generated/retrieved answer text, Top1, Top3 and Top5 are identical in this script.

## Global results

| Model | Top1 | Top3 | Top5 | Total | Avg store | Avg query |
|---|---:|---:|---:|---:|---:|---:|
| legacy_titan | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 11 | 0.003s | 0.002s |
| new_titan | 10 (90.9%) | 10 (90.9%) | 10 (90.9%) | 11 | 1.724s | 1.415s |

## Results by scale

| Model | Scale | Top1 | Top5 | Total | Avg store | Avg query |
|---|---|---:|---:|---:|---:|---:|
| legacy_titan | small | 0 (0.0%) | 0 (0.0%) | 11 | 0.003s | 0.002s |
| new_titan | small | 10 (90.9%) | 10 (90.9%) | 11 | 1.724s | 1.415s |

## Results by category

| Model | Category | Top1 | Top5 | Total |
|---|---|---:|---:|---:|
| legacy_titan | forget_with_nearby_distractors | 0 (0.0%) | 0 (0.0%) | 1 |
| legacy_titan | hard_distractors | 0 (0.0%) | 0 (0.0%) | 1 |
| legacy_titan | identity_collision | 0 (0.0%) | 0 (0.0%) | 1 |
| legacy_titan | mixed_language_query | 0 (0.0%) | 0 (0.0%) | 1 |
| legacy_titan | multiple_updates | 0 (0.0%) | 0 (0.0%) | 3 |
| legacy_titan | noisy_query | 0 (0.0%) | 0 (0.0%) | 1 |
| legacy_titan | paraphrase_recall | 0 (0.0%) | 0 (0.0%) | 1 |
| legacy_titan | profile_summary | 0 (0.0%) | 0 (0.0%) | 1 |
| legacy_titan | retention_after_forget | 0 (0.0%) | 0 (0.0%) | 1 |
| new_titan | forget_with_nearby_distractors | 1 (100.0%) | 1 (100.0%) | 1 |
| new_titan | hard_distractors | 1 (100.0%) | 1 (100.0%) | 1 |
| new_titan | identity_collision | 1 (100.0%) | 1 (100.0%) | 1 |
| new_titan | mixed_language_query | 1 (100.0%) | 1 (100.0%) | 1 |
| new_titan | multiple_updates | 3 (100.0%) | 3 (100.0%) | 3 |
| new_titan | noisy_query | 1 (100.0%) | 1 (100.0%) | 1 |
| new_titan | paraphrase_recall | 1 (100.0%) | 1 (100.0%) | 1 |
| new_titan | profile_summary | 1 (100.0%) | 1 (100.0%) | 1 |
| new_titan | retention_after_forget | 0 (0.0%) | 0 (0.0%) | 1 |

## Top-5 failures

### legacy_titan - small - small_paraphrased_location
- Category: paraphrase_recall
- Reason: contains forbidden value(s): ['travel advertisement']
- Question: In which city is CarlosAlpha Singh currently based?
- Expected any: `["Nantes"]`
- Expected all: `[]`
- Forbidden: `["blue sofa", "router sticker", "travel advertisement"]`
- Retrieved/answer: `Legacy memory snapshot: CarlosAlpha Singh is a 24-year-old student from Nantes, France. CarlosAlpha Singh studies database optimization and works on a mobile accessibility project. CarlosAlpha Singh's favorite color is black. CarlosAlpha Singh's favorite programming language is C++. CarlosAlpha Singh's secret code is CODE-3002. CarlosAlpha Singh's office room is R402. | Hanoi is mentioned in a travel advertisement. | A note says that Garcia is a common surname in the dataset. | A note says that Khan is a common surname in the dataset. | Madrid is mentioned in a travel advertisement. | A note says that Lopez is a common surname in the dataset. | Nantes is mentioned in a travel advertisement. | A note says that Tanaka is a common surname in the dataset.`

### legacy_titan - small - small_noisy_office_room
- Category: noisy_query
- Reason: contains forbidden value(s): ['hotel has a room']
- Question: hey, remind me plz, what's the office room for CarlosAlpha Singh??
- Expected any: `["R402"]`
- Expected all: `[]`
- Forbidden: `["hotel has a room"]`
- Retrieved/answer: `Legacy memory snapshot: CarlosAlpha Singh is a 24-year-old student from Nantes, France. CarlosAlpha Singh studies database optimization and works on a mobile accessibility project. CarlosAlpha Singh's favorite color is black. CarlosAlpha Singh's favorite programming language is C++. CarlosAlpha Singh's secret code is CODE-3002. CarlosAlpha Singh's office room is R402. | The hotel has a room named R412 but it is not an office. | The hotel has a room named R404 but it is not an office. | The hotel has a room named R416 but it is not an office. | The hotel has a room named R408 but it is not an office. | The hotel has a room named R400 but it is not an office. | SamuelAlpha Singh is a 24-year-old student from Hanoi, Vietnam. SamuelAlpha Singh studies data engineering and works on a robotic navigation system. SamuelAlpha Singh's favorite color is orange. SamuelAlpha Singh's favorite programming language is C++. SamuelAlpha Singh's secret code is CODE-3018. SamuelAlpha Singh's office room is R418. | TaraAlpha Bernard is a 24-year-old student from Madrid, Spain. TaraAlpha Bernard studies cloud infrastructure and works on a heart-rate analysis tool. TaraAlpha Bernard's favorite color is yellow. TaraAlpha Bernard's favorite programming language is Scala. TaraAlpha Bernard's secret code is CODE-3019. TaraAlpha Bernard's office room is R419.`

### legacy_titan - small - small_multiple_color_updates
- Category: multiple_updates
- Reason: contains forbidden value(s): ['green', 'magenta']
- Question: What is GabrielAlpha Brown's favorite color?
- Expected any: `["turquoise"]`
- Expected all: `[]`
- Forbidden: `["green", "magenta"]`
- Retrieved/answer: `Legacy memory snapshot: Correction: GabrielAlpha Brown's favorite color is now turquoise. | Actually, GabrielAlpha Brown's favorite color is now magenta. | GabrielAlpha Brown is a 24-year-old student from Hanoi, Vietnam. GabrielAlpha Brown studies data engineering and works on a robotic navigation system. GabrielAlpha Brown's favorite color is green. GabrielAlpha Brown's favorite programming language is Go. GabrielAlpha Brown's secret code is CODE-3006. GabrielAlpha Brown's office room is R406. | TaraAlpha Bernard is a 24-year-old student from Madrid, Spain. TaraAlpha Bernard studies cloud infrastructure and works on a heart-rate analysis tool. TaraAlpha Bernard's favorite color is yellow. TaraAlpha Bernard's favorite programming language is Scala. TaraAlpha Bernard's secret code is CODE-3019. TaraAlpha Bernard's office room is R419. | SamuelAlpha Singh is a 24-year-old student from Hanoi, Vietnam. SamuelAlpha Singh studies data engineering and works on a robotic navigation system. SamuelAlpha Singh's favorite color is orange. SamuelAlpha Singh's favorite programming language is C++. SamuelAlpha Singh's secret code is CODE-3018. SamuelAlpha Singh's office room is R418. | RosaAlpha Tanaka is a 24-year-old student from Nantes, France. RosaAlpha Tanaka studies sports analytics and works on a cloud monitoring dashboard. RosaAlpha Tanaka's favorite color is turquoise. RosaAlpha Tanaka's favorite programming language is JavaScript. RosaAlpha Tanaka's secret code is CODE-3017. RosaAlpha Tanaka's office room is R417. | QuentinAlpha Martin is a 24-year-old student from Madrid, Spain. QuentinAlpha Martin studies software engineering and works on a medical diagnosis assistant. QuentinAlpha Martin's favorite color is green. QuentinAlpha Martin's favorite programming language is Python. QuentinAlpha Martin's secret code is CODE-3016. QuentinAlpha Martin's office room is R416. | PriyaAlpha Smith is a 24-year-old student from Hanoi, Vietnam. PriyaAlpha Smith studies computer vision and works on a recommendation system. PriyaAlpha Smith's favorite color is purple. PriyaAlpha Smith's favorite programming language is Java. PriyaAlpha Smith's secret code is CODE-3015. PriyaAlpha Smith's office room is R415.`

### legacy_titan - small - small_multiple_language_updates
- Category: multiple_updates
- Reason: contains forbidden value(s): ['C++', 'Rust']
- Question: What programming language does KarimAlpha Dubois prefer now?
- Expected any: `["Python"]`
- Expected all: `[]`
- Forbidden: `["C++", "Rust"]`
- Retrieved/answer: `Legacy memory snapshot: Update: KarimAlpha Dubois's favorite programming language is now Python. | Actually, KarimAlpha Dubois's favorite programming language is now Rust. | KarimAlpha Dubois is a 24-year-old student from Madrid, Spain. KarimAlpha Dubois studies medical imaging and works on a database indexing experiment. KarimAlpha Dubois's favorite color is red. KarimAlpha Dubois's favorite programming language is C++. KarimAlpha Dubois's secret code is CODE-3010. KarimAlpha Dubois's office room is R410. | TaraAlpha Bernard is a 24-year-old student from Madrid, Spain. TaraAlpha Bernard studies cloud infrastructure and works on a heart-rate analysis tool. TaraAlpha Bernard's favorite color is yellow. TaraAlpha Bernard's favorite programming language is Scala. TaraAlpha Bernard's secret code is CODE-3019. TaraAlpha Bernard's office room is R419. | SamuelAlpha Singh is a 24-year-old student from Hanoi, Vietnam. SamuelAlpha Singh studies data engineering and works on a robotic navigation system. SamuelAlpha Singh's favorite color is orange. SamuelAlpha Singh's favorite programming language is C++. SamuelAlpha Singh's secret code is CODE-3018. SamuelAlpha Singh's office room is R418. | RosaAlpha Tanaka is a 24-year-old student from Nantes, France. RosaAlpha Tanaka studies sports analytics and works on a cloud monitoring dashboard. RosaAlpha Tanaka's favorite color is turquoise. RosaAlpha Tanaka's favorite programming language is JavaScript. RosaAlpha Tanaka's secret code is CODE-3017. RosaAlpha Tanaka's office room is R417. | QuentinAlpha Martin is a 24-year-old student from Madrid, Spain. QuentinAlpha Martin studies software engineering and works on a medical diagnosis assistant. QuentinAlpha Martin's favorite color is green. QuentinAlpha Martin's favorite programming language is Python. QuentinAlpha Martin's secret code is CODE-3016. QuentinAlpha Martin's office room is R416. | PriyaAlpha Smith is a 24-year-old student from Hanoi, Vietnam. PriyaAlpha Smith studies computer vision and works on a recommendation system. PriyaAlpha Smith's favorite color is purple. PriyaAlpha Smith's favorite programming language is Java. PriyaAlpha Smith's secret code is CODE-3015. PriyaAlpha Smith's office room is R415.`

### legacy_titan - small - small_multiple_secret_code_updates
- Category: multiple_updates
- Reason: contains forbidden value(s): ['CODE-3013', 'TEMP-CODE-3013']
- Question: What is NoraAlpha Lopez's latest secret code?
- Expected any: `["FINAL-CODE-3013"]`
- Expected all: `[]`
- Forbidden: `["CODE-3013", "TEMP-CODE-3013"]`
- Retrieved/answer: `Legacy memory snapshot: Correction: NoraAlpha Lopez's secret code is now FINAL-CODE-3013. | Actually, NoraAlpha Lopez's secret code is now TEMP-CODE-3013. | NoraAlpha Lopez is a 24-year-old student from Madrid, Spain. NoraAlpha Lopez studies human-computer interaction and works on a phishing detection tool. NoraAlpha Lopez's favorite color is blue. NoraAlpha Lopez's favorite programming language is TypeScript. NoraAlpha Lopez's secret code is CODE-3013. NoraAlpha Lopez's office room is R413. | TaraAlpha Bernard is a 24-year-old student from Madrid, Spain. TaraAlpha Bernard studies cloud infrastructure and works on a heart-rate analysis tool. TaraAlpha Bernard's favorite color is yellow. TaraAlpha Bernard's favorite programming language is Scala. TaraAlpha Bernard's secret code is CODE-3019. TaraAlpha Bernard's office room is R419. | SamuelAlpha Singh is a 24-year-old student from Hanoi, Vietnam. SamuelAlpha Singh studies data engineering and works on a robotic navigation system. SamuelAlpha Singh's favorite color is orange. SamuelAlpha Singh's favorite programming language is C++. SamuelAlpha Singh's secret code is CODE-3018. SamuelAlpha Singh's office room is R418. | RosaAlpha Tanaka is a 24-year-old student from Nantes, France. RosaAlpha Tanaka studies sports analytics and works on a cloud monitoring dashboard. RosaAlpha Tanaka's favorite color is turquoise. RosaAlpha Tanaka's favorite programming language is JavaScript. RosaAlpha Tanaka's secret code is CODE-3017. RosaAlpha Tanaka's office room is R417. | QuentinAlpha Martin is a 24-year-old student from Madrid, Spain. QuentinAlpha Martin studies software engineering and works on a medical diagnosis assistant. QuentinAlpha Martin's favorite color is green. QuentinAlpha Martin's favorite programming language is Python. QuentinAlpha Martin's secret code is CODE-3016. QuentinAlpha Martin's office room is R416. | PriyaAlpha Smith is a 24-year-old student from Hanoi, Vietnam. PriyaAlpha Smith studies computer vision and works on a recommendation system. PriyaAlpha Smith's favorite color is purple. PriyaAlpha Smith's favorite programming language is Java. PriyaAlpha Smith's secret code is CODE-3015. PriyaAlpha Smith's office room is R415.`

### legacy_titan - small - small_forget_secret_code_no_other_secret_code
- Category: forget_with_nearby_distractors
- Reason: contains forgotten value(s): ['CODE-3017']
- Question: What is RosaAlpha Tanaka's secret code?
- Expected any: `["do not know", "don't know", "not know", "no memory"]`
- Expected all: `[]`
- Forbidden: `["CODE-3017"]`
- Retrieved/answer: `Legacy memory snapshot: RosaAlpha Tanaka is a 24-year-old student from Nantes, France. RosaAlpha Tanaka studies sports analytics and works on a cloud monitoring dashboard. RosaAlpha Tanaka's favorite color is turquoise. RosaAlpha Tanaka's favorite programming language is JavaScript. RosaAlpha Tanaka's secret code is CODE-3017. RosaAlpha Tanaka's office room is R417. | BellaAlpha Tanaka is a 24-year-old student from Madrid, Spain. BellaAlpha Tanaka studies human-computer interaction and works on a phishing detection tool. BellaAlpha Tanaka's favorite color is white. BellaAlpha Tanaka's favorite programming language is JavaScript. BellaAlpha Tanaka's secret code is CODE-3001. BellaAlpha Tanaka's office room is R401. | TaraAlpha Bernard is a 24-year-old student from Madrid, Spain. TaraAlpha Bernard studies cloud infrastructure and works on a heart-rate analysis tool. TaraAlpha Bernard's favorite color is yellow. TaraAlpha Bernard's favorite programming language is Scala. TaraAlpha Bernard's secret code is CODE-3019. TaraAlpha Bernard's office room is R419. | SamuelAlpha Singh is a 24-year-old student from Hanoi, Vietnam. SamuelAlpha Singh studies data engineering and works on a robotic navigation system. SamuelAlpha Singh's favorite color is orange. SamuelAlpha Singh's favorite programming language is C++. SamuelAlpha Singh's secret code is CODE-3018. SamuelAlpha Singh's office room is R418. | QuentinAlpha Martin is a 24-year-old student from Madrid, Spain. QuentinAlpha Martin studies software engineering and works on a medical diagnosis assistant. QuentinAlpha Martin's favorite color is green. QuentinAlpha Martin's favorite programming language is Python. QuentinAlpha Martin's secret code is CODE-3016. QuentinAlpha Martin's office room is R416. | PriyaAlpha Smith is a 24-year-old student from Hanoi, Vietnam. PriyaAlpha Smith studies computer vision and works on a recommendation system. PriyaAlpha Smith's favorite color is purple. PriyaAlpha Smith's favorite programming language is Java. PriyaAlpha Smith's secret code is CODE-3015. PriyaAlpha Smith's office room is R415. | OwenAlpha Rossi is a 24-year-old student from Nantes, France. OwenAlpha Rossi studies database optimization and works on a mobile accessibility project. OwenAlpha Rossi's favorite color is cyan. OwenAlpha Rossi's favorite programming language is Go. OwenAlpha Rossi's secret code is CODE-3014. OwenAlpha Rossi's office room is R414. | NoraAlpha Lopez is a 24-year-old student from Madrid, Spain. NoraAlpha Lopez studies human-computer interaction and works on a phishing detection tool. NoraAlpha Lopez's favorite color is blue. NoraAlpha Lopez's favorite programming language is TypeScript. NoraAlpha Lopez's secret code is CODE-3013. NoraAlpha Lopez's office room is R413.`

### legacy_titan - small - small_retains_project_after_forget
- Category: retention_after_forget
- Reason: contains forbidden value(s): ['CODE-3017']
- Question: What does RosaAlpha Tanaka work on?
- Expected any: `["cloud monitoring dashboard"]`
- Expected all: `[]`
- Forbidden: `["CODE-3017"]`
- Retrieved/answer: `Legacy memory snapshot: RosaAlpha Tanaka is a 24-year-old student from Nantes, France. RosaAlpha Tanaka studies sports analytics and works on a cloud monitoring dashboard. RosaAlpha Tanaka's favorite color is turquoise. RosaAlpha Tanaka's favorite programming language is JavaScript. RosaAlpha Tanaka's secret code is CODE-3017. RosaAlpha Tanaka's office room is R417. | BellaAlpha Tanaka is a 24-year-old student from Madrid, Spain. BellaAlpha Tanaka studies human-computer interaction and works on a phishing detection tool. BellaAlpha Tanaka's favorite color is white. BellaAlpha Tanaka's favorite programming language is JavaScript. BellaAlpha Tanaka's secret code is CODE-3001. BellaAlpha Tanaka's office room is R401. | The code CODE-9035 is printed on a router sticker. | The code CODE-9027 is printed on a router sticker. | The code CODE-9019 is printed on a router sticker. | The code CODE-9011 is printed on a router sticker. | The code CODE-9003 is printed on a router sticker. | A note says that Tanaka is a common surname in the dataset.`

### new_titan - small - small_retains_project_after_forget
- Category: retention_after_forget
- Reason: missing one of expected_any: ['cloud monitoring dashboard']
- Question: What does RosaAlpha Tanaka work on?
- Expected any: `["cloud monitoring dashboard"]`
- Expected all: `[]`
- Forbidden: `["CODE-3017"]`
- Retrieved/answer: `Based on memory: Rosaalpha Tanaka is from Nantes, France.`

### legacy_titan - small - small_hard_distractor_room
- Category: hard_distractors
- Reason: contains forbidden value(s): ['hotel has a room']
- Question: What is the real office room of NoraAlpha Lopez?
- Expected any: `["R413"]`
- Expected all: `[]`
- Forbidden: `["hotel has a room"]`
- Retrieved/answer: `Legacy memory snapshot: NoraAlpha Lopez is a 24-year-old student from Madrid, Spain. NoraAlpha Lopez studies human-computer interaction and works on a phishing detection tool. NoraAlpha Lopez's favorite color is blue. NoraAlpha Lopez's favorite programming language is TypeScript. NoraAlpha Lopez's secret code is CODE-3013. NoraAlpha Lopez's office room is R413. | The hotel has a room named R412 but it is not an office. | The hotel has a room named R404 but it is not an office. | The hotel has a room named R416 but it is not an office. | The hotel has a room named R408 but it is not an office. | The hotel has a room named R400 but it is not an office. | A note says that Lopez is a common surname in the dataset. | TaraAlpha Bernard is a 24-year-old student from Madrid, Spain. TaraAlpha Bernard studies cloud infrastructure and works on a heart-rate analysis tool. TaraAlpha Bernard's favorite color is yellow. TaraAlpha Bernard's favorite programming language is Scala. TaraAlpha Bernard's secret code is CODE-3019. TaraAlpha Bernard's office room is R419.`

### legacy_titan - small - small_mixed_language_location
- Category: mixed_language_query
- Reason: contains forbidden value(s): ['travel advertisement']
- Question: En quelle ville does KarimAlpha Dubois live / is based?
- Expected any: `["Madrid"]`
- Expected all: `[]`
- Forbidden: `["travel advertisement", "blue sofa"]`
- Retrieved/answer: `Legacy memory snapshot: KarimAlpha Dubois is a 24-year-old student from Madrid, Spain. KarimAlpha Dubois studies medical imaging and works on a database indexing experiment. KarimAlpha Dubois's favorite color is red. KarimAlpha Dubois's favorite programming language is C++. KarimAlpha Dubois's secret code is CODE-3010. KarimAlpha Dubois's office room is R410. | Hanoi is mentioned in a travel advertisement. | The code CODE-9035 is printed on a router sticker. | The hotel has a room named R412 but it is not an office. | The code CODE-9027 is printed on a router sticker. | A note says that Garcia is a common surname in the dataset. | The hotel has a room named R404 but it is not an office. | The code CODE-9019 is printed on a router sticker.`

### legacy_titan - small - small_profile_summary
- Category: profile_summary
- Reason: contains forbidden value(s): ['router sticker', 'blue sofa']
- Question: Tell me the key things you know about CarlosAlpha Singh.
- Expected any: `[]`
- Expected all: `["Nantes", "database optimization", "mobile accessibility project"]`
- Forbidden: `["router sticker", "blue sofa"]`
- Retrieved/answer: `Legacy memory snapshot: CarlosAlpha Singh is a 24-year-old student from Nantes, France. CarlosAlpha Singh studies database optimization and works on a mobile accessibility project. CarlosAlpha Singh's favorite color is black. CarlosAlpha Singh's favorite programming language is C++. CarlosAlpha Singh's secret code is CODE-3002. CarlosAlpha Singh's office room is R402. | PriyaAlpha saw a blue sofa near the coffee machine. | The code CODE-9035 is printed on a router sticker. | The color blue appears in a poster near the lab. | The hotel has a room named R412 but it is not an office. | TaraAlpha saw a blue sofa near the coffee machine. | The code CODE-9027 is printed on a router sticker. | A note says that Garcia is a common surname in the dataset.`

### legacy_titan - small - small_same_first_name_full_identity
- Category: identity_collision
- Reason: contains forbidden value(s): ['green', 'orange']
- Question: What is Sarah Nguyen's favorite color?
- Expected any: `["purple"]`
- Expected all: `[]`
- Forbidden: `["green", "orange"]`
- Retrieved/answer: `Legacy memory snapshot: Sarah Nguyen's favorite color is purple. | HanaAlpha Nguyen is a 24-year-old student from Madrid, Spain. HanaAlpha Nguyen studies cloud infrastructure and works on a heart-rate analysis tool. HanaAlpha Nguyen's favorite color is turquoise. HanaAlpha Nguyen's favorite programming language is Java. HanaAlpha Nguyen's secret code is CODE-3007. HanaAlpha Nguyen's office room is R407. | Sarah Wilson's favorite color is orange. | Sarah Martin's favorite color is green. | TaraAlpha Bernard is a 24-year-old student from Madrid, Spain. TaraAlpha Bernard studies cloud infrastructure and works on a heart-rate analysis tool. TaraAlpha Bernard's favorite color is yellow. TaraAlpha Bernard's favorite programming language is Scala. TaraAlpha Bernard's secret code is CODE-3019. TaraAlpha Bernard's office room is R419. | SamuelAlpha Singh is a 24-year-old student from Hanoi, Vietnam. SamuelAlpha Singh studies data engineering and works on a robotic navigation system. SamuelAlpha Singh's favorite color is orange. SamuelAlpha Singh's favorite programming language is C++. SamuelAlpha Singh's secret code is CODE-3018. SamuelAlpha Singh's office room is R418. | RosaAlpha Tanaka is a 24-year-old student from Nantes, France. RosaAlpha Tanaka studies sports analytics and works on a cloud monitoring dashboard. RosaAlpha Tanaka's favorite color is turquoise. RosaAlpha Tanaka's favorite programming language is JavaScript. RosaAlpha Tanaka's secret code is CODE-3017. RosaAlpha Tanaka's office room is R417. | QuentinAlpha Martin is a 24-year-old student from Madrid, Spain. QuentinAlpha Martin studies software engineering and works on a medical diagnosis assistant. QuentinAlpha Martin's favorite color is green. QuentinAlpha Martin's favorite programming language is Python. QuentinAlpha Martin's secret code is CODE-3016. QuentinAlpha Martin's office room is R416.`
