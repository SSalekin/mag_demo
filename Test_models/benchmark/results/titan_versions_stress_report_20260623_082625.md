# Stress Comparison — Old Titan vs New Titan

- Date: 2026-06-23 08:26:25
- Scales: small, medium
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
| legacy_titan | 1 (4.5%) | 1 (4.5%) | 1 (4.5%) | 22 | 0.004s | 0.003s |
| new_titan | 18 (81.8%) | 18 (81.8%) | 18 (81.8%) | 22 | 3.688s | 0.136s |

## Results by scale

| Model | Scale | Top1 | Top5 | Total | Avg store | Avg query |
|---|---|---:|---:|---:|---:|---:|
| legacy_titan | medium | 1 (9.1%) | 1 (9.1%) | 11 | 0.006s | 0.004s |
| legacy_titan | small | 0 (0.0%) | 0 (0.0%) | 11 | 0.002s | 0.002s |
| new_titan | medium | 9 (81.8%) | 9 (81.8%) | 11 | 6.169s | 0.207s |
| new_titan | small | 9 (81.8%) | 9 (81.8%) | 11 | 1.207s | 0.064s |

## Results by category

| Model | Category | Top1 | Top5 | Total |
|---|---|---:|---:|---:|
| legacy_titan | forget_with_nearby_distractors | 0 (0.0%) | 0 (0.0%) | 2 |
| legacy_titan | hard_distractors | 0 (0.0%) | 0 (0.0%) | 2 |
| legacy_titan | identity_collision | 0 (0.0%) | 0 (0.0%) | 2 |
| legacy_titan | mixed_language_query | 1 (50.0%) | 1 (50.0%) | 2 |
| legacy_titan | multiple_updates | 0 (0.0%) | 0 (0.0%) | 6 |
| legacy_titan | noisy_query | 0 (0.0%) | 0 (0.0%) | 2 |
| legacy_titan | paraphrase_recall | 0 (0.0%) | 0 (0.0%) | 2 |
| legacy_titan | profile_summary | 0 (0.0%) | 0 (0.0%) | 2 |
| legacy_titan | retention_after_forget | 0 (0.0%) | 0 (0.0%) | 2 |
| new_titan | forget_with_nearby_distractors | 2 (100.0%) | 2 (100.0%) | 2 |
| new_titan | hard_distractors | 2 (100.0%) | 2 (100.0%) | 2 |
| new_titan | identity_collision | 2 (100.0%) | 2 (100.0%) | 2 |
| new_titan | mixed_language_query | 2 (100.0%) | 2 (100.0%) | 2 |
| new_titan | multiple_updates | 6 (100.0%) | 6 (100.0%) | 6 |
| new_titan | noisy_query | 2 (100.0%) | 2 (100.0%) | 2 |
| new_titan | paraphrase_recall | 2 (100.0%) | 2 (100.0%) | 2 |
| new_titan | profile_summary | 0 (0.0%) | 0 (0.0%) | 2 |
| new_titan | retention_after_forget | 0 (0.0%) | 0 (0.0%) | 2 |

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

### new_titan - small - small_profile_summary
- Category: profile_summary
- Reason: missing expected_all: ['database optimization', 'mobile accessibility project']
- Question: Tell me the key things you know about CarlosAlpha Singh.
- Expected any: `[]`
- Expected all: `["Nantes", "database optimization", "mobile accessibility project"]`
- Forbidden: `["router sticker", "blue sofa"]`
- Retrieved/answer: `Based on memory: Carlosalpha Singh is from Nantes, France.`

### legacy_titan - small - small_same_first_name_full_identity
- Category: identity_collision
- Reason: contains forbidden value(s): ['green', 'orange']
- Question: What is Sarah Nguyen's favorite color?
- Expected any: `["purple"]`
- Expected all: `[]`
- Forbidden: `["green", "orange"]`
- Retrieved/answer: `Legacy memory snapshot: Sarah Nguyen's favorite color is purple. | HanaAlpha Nguyen is a 24-year-old student from Madrid, Spain. HanaAlpha Nguyen studies cloud infrastructure and works on a heart-rate analysis tool. HanaAlpha Nguyen's favorite color is turquoise. HanaAlpha Nguyen's favorite programming language is Java. HanaAlpha Nguyen's secret code is CODE-3007. HanaAlpha Nguyen's office room is R407. | Sarah Wilson's favorite color is orange. | Sarah Martin's favorite color is green. | TaraAlpha Bernard is a 24-year-old student from Madrid, Spain. TaraAlpha Bernard studies cloud infrastructure and works on a heart-rate analysis tool. TaraAlpha Bernard's favorite color is yellow. TaraAlpha Bernard's favorite programming language is Scala. TaraAlpha Bernard's secret code is CODE-3019. TaraAlpha Bernard's office room is R419. | SamuelAlpha Singh is a 24-year-old student from Hanoi, Vietnam. SamuelAlpha Singh studies data engineering and works on a robotic navigation system. SamuelAlpha Singh's favorite color is orange. SamuelAlpha Singh's favorite programming language is C++. SamuelAlpha Singh's secret code is CODE-3018. SamuelAlpha Singh's office room is R418. | RosaAlpha Tanaka is a 24-year-old student from Nantes, France. RosaAlpha Tanaka studies sports analytics and works on a cloud monitoring dashboard. RosaAlpha Tanaka's favorite color is turquoise. RosaAlpha Tanaka's favorite programming language is JavaScript. RosaAlpha Tanaka's secret code is CODE-3017. RosaAlpha Tanaka's office room is R417. | QuentinAlpha Martin is a 24-year-old student from Madrid, Spain. QuentinAlpha Martin studies software engineering and works on a medical diagnosis assistant. QuentinAlpha Martin's favorite color is green. QuentinAlpha Martin's favorite programming language is Python. QuentinAlpha Martin's secret code is CODE-3016. QuentinAlpha Martin's office room is R416.`

### legacy_titan - medium - medium_paraphrased_location
- Category: paraphrase_recall
- Reason: contains forbidden value(s): ['travel advertisement']
- Question: In which city is LenaAlpha Meyer currently based?
- Expected any: `["Nantes"]`
- Expected all: `[]`
- Forbidden: `["blue sofa", "router sticker", "travel advertisement"]`
- Retrieved/answer: `Legacy memory snapshot: LenaAlpha Meyer is a 24-year-old student from Nantes, France. LenaAlpha Meyer studies cybersecurity and works on a gesture recognition prototype. LenaAlpha Meyer's favorite color is white. LenaAlpha Meyer's favorite programming language is Scala. LenaAlpha Meyer's secret code is CODE-3011. LenaAlpha Meyer's office room is R411. | Madrid is mentioned in a travel advertisement. | Nantes is mentioned in a travel advertisement. | A note says that Khan is a common surname in the dataset. | Hanoi is mentioned in a travel advertisement. | A note says that Tanaka is a common surname in the dataset. | XavierGamma Meyer is a 24-year-old student from Hanoi, Vietnam. XavierGamma Meyer studies computer vision and works on a recommendation system. XavierGamma Meyer's favorite color is purple. XavierGamma Meyer's favorite programming language is Scala. XavierGamma Meyer's secret code is CODE-3075. XavierGamma Meyer's office room is R475. | HanaGamma Meyer is a 24-year-old student from Nantes, France. HanaGamma Meyer studies cybersecurity and works on a gesture recognition prototype. HanaGamma Meyer's favorite color is yellow. HanaGamma Meyer's favorite programming language is Scala. HanaGamma Meyer's secret code is CODE-3059. HanaGamma Meyer's office room is R459.`

### legacy_titan - medium - medium_noisy_office_room
- Category: noisy_query
- Reason: contains forbidden value(s): ['hotel has a room']
- Question: hey, remind me plz, what's the office room for LenaAlpha Meyer??
- Expected any: `["R411"]`
- Expected all: `[]`
- Forbidden: `["hotel has a room"]`
- Retrieved/answer: `Legacy memory snapshot: LenaAlpha Meyer is a 24-year-old student from Nantes, France. LenaAlpha Meyer studies cybersecurity and works on a gesture recognition prototype. LenaAlpha Meyer's favorite color is white. LenaAlpha Meyer's favorite programming language is Scala. LenaAlpha Meyer's secret code is CODE-3011. LenaAlpha Meyer's office room is R411. | The hotel has a room named R472 but it is not an office. | The hotel has a room named R464 but it is not an office. | The hotel has a room named R456 but it is not an office. | The hotel has a room named R448 but it is not an office. | The hotel has a room named R440 but it is not an office. | The hotel has a room named R432 but it is not an office. | The hotel has a room named R424 but it is not an office.`

### legacy_titan - medium - medium_multiple_color_updates
- Category: multiple_updates
- Reason: contains forbidden value(s): ['green', 'magenta']
- Question: What is AlexBeta Dubois's favorite color?
- Expected any: `["turquoise"]`
- Expected all: `[]`
- Forbidden: `["green", "magenta"]`
- Retrieved/answer: `Legacy memory snapshot: Correction: AlexBeta Dubois's favorite color is now turquoise. | Actually, AlexBeta Dubois's favorite color is now magenta. | AlexBeta Dubois is a 24-year-old student from Nantes, France. AlexBeta Dubois studies database optimization and works on a mobile accessibility project. AlexBeta Dubois's favorite color is green. AlexBeta Dubois's favorite programming language is C++. AlexBeta Dubois's secret code is CODE-3026. AlexBeta Dubois's office room is R426. | WendyGamma Dubois is a 24-year-old student from Nantes, France. WendyGamma Dubois studies database optimization and works on a mobile accessibility project. WendyGamma Dubois's favorite color is cyan. WendyGamma Dubois's favorite programming language is C++. WendyGamma Dubois's secret code is CODE-3074. WendyGamma Dubois's office room is R474. | GabrielGamma Dubois is a 24-year-old student from Madrid, Spain. GabrielGamma Dubois studies medical imaging and works on a database indexing experiment. GabrielGamma Dubois's favorite color is orange. GabrielGamma Dubois's favorite programming language is C++. GabrielGamma Dubois's secret code is CODE-3058. GabrielGamma Dubois's office room is R458. | QuentinBeta Dubois is a 24-year-old student from Hanoi, Vietnam. QuentinBeta Dubois studies data engineering and works on a robotic navigation system. QuentinBeta Dubois's favorite color is black. QuentinBeta Dubois's favorite programming language is C++. QuentinBeta Dubois's secret code is CODE-3042. QuentinBeta Dubois's office room is R442. | KarimAlpha Dubois is a 24-year-old student from Madrid, Spain. KarimAlpha Dubois studies medical imaging and works on a database indexing experiment. KarimAlpha Dubois's favorite color is red. KarimAlpha Dubois's favorite programming language is C++. KarimAlpha Dubois's secret code is CODE-3010. KarimAlpha Dubois's office room is R410. | BellaDelta Smith is a 24-year-old student from Madrid, Spain. BellaDelta Smith studies cloud infrastructure and works on a heart-rate analysis tool. BellaDelta Smith's favorite color is yellow. BellaDelta Smith's favorite programming language is Java. BellaDelta Smith's secret code is CODE-3079. BellaDelta Smith's office room is R479.`

### legacy_titan - medium - medium_multiple_language_updates
- Category: multiple_updates
- Reason: contains forbidden value(s): ['Rust']
- Question: What programming language does OwenBeta Morel prefer now?
- Expected any: `["Python"]`
- Expected all: `[]`
- Forbidden: `["Python", "Rust"]`
- Retrieved/answer: `Legacy memory snapshot: Update: OwenBeta Morel's favorite programming language is now Python. | Actually, OwenBeta Morel's favorite programming language is now Rust. | OwenBeta Morel is a 24-year-old student from Madrid, Spain. OwenBeta Morel studies software engineering and works on a medical diagnosis assistant. OwenBeta Morel's favorite color is red. OwenBeta Morel's favorite programming language is Python. OwenBeta Morel's secret code is CODE-3040. OwenBeta Morel's office room is R440. | UmaGamma Morel is a 24-year-old student from Hanoi, Vietnam. UmaGamma Morel studies natural language processing and works on a sentiment analysis project. UmaGamma Morel's favorite color is black. UmaGamma Morel's favorite programming language is Python. UmaGamma Morel's secret code is CODE-3072. UmaGamma Morel's office room is R472. | EthanGamma Morel is a 24-year-old student from Nantes, France. EthanGamma Morel studies robotics and works on an image classification pipeline. EthanGamma Morel's favorite color is green. EthanGamma Morel's favorite programming language is Python. EthanGamma Morel's secret code is CODE-3056. EthanGamma Morel's office room is R456. | YaraAlpha Morel is a 24-year-old student from Hanoi, Vietnam. YaraAlpha Morel studies natural language processing and works on a sentiment analysis project. YaraAlpha Morel's favorite color is cyan. YaraAlpha Morel's favorite programming language is Python. YaraAlpha Morel's secret code is CODE-3024. YaraAlpha Morel's office room is R424. | IvanAlpha Morel is a 24-year-old student from Nantes, France. IvanAlpha Morel studies robotics and works on an image classification pipeline. IvanAlpha Morel's favorite color is orange. IvanAlpha Morel's favorite programming language is Python. IvanAlpha Morel's secret code is CODE-3008. IvanAlpha Morel's office room is R408. | BellaDelta Smith is a 24-year-old student from Madrid, Spain. BellaDelta Smith studies cloud infrastructure and works on a heart-rate analysis tool. BellaDelta Smith's favorite color is yellow. BellaDelta Smith's favorite programming language is Java. BellaDelta Smith's secret code is CODE-3079. BellaDelta Smith's office room is R479.`

### legacy_titan - medium - medium_multiple_secret_code_updates
- Category: multiple_updates
- Reason: contains forbidden value(s): ['CODE-3053', 'TEMP-CODE-3053']
- Question: What is BellaGamma Garcia's latest secret code?
- Expected any: `["FINAL-CODE-3053"]`
- Expected all: `[]`
- Forbidden: `["CODE-3053", "TEMP-CODE-3053"]`
- Retrieved/answer: `Legacy memory snapshot: Correction: BellaGamma Garcia's secret code is now FINAL-CODE-3053. | Actually, BellaGamma Garcia's secret code is now TEMP-CODE-3053. | BellaGamma Garcia is a 24-year-old student from Nantes, France. BellaGamma Garcia studies sports analytics and works on a cloud monitoring dashboard. BellaGamma Garcia's favorite color is blue. BellaGamma Garcia's favorite programming language is TypeScript. BellaGamma Garcia's secret code is CODE-3053. BellaGamma Garcia's office room is R453. | RosaGamma Garcia is a 24-year-old student from Hanoi, Vietnam. RosaGamma Garcia studies UI design and works on a multilingual chatbot. RosaGamma Garcia's favorite color is yellow. RosaGamma Garcia's favorite programming language is TypeScript. RosaGamma Garcia's secret code is CODE-3069. RosaGamma Garcia's office room is R469. | LenaBeta Garcia is a 24-year-old student from Madrid, Spain. LenaBeta Garcia studies human-computer interaction and works on a phishing detection tool. LenaBeta Garcia's favorite color is turquoise. LenaBeta Garcia's favorite programming language is TypeScript. LenaBeta Garcia's secret code is CODE-3037. LenaBeta Garcia's office room is R437. | VictorAlpha Garcia is a 24-year-old student from Hanoi, Vietnam. VictorAlpha Garcia studies UI design and works on a multilingual chatbot. VictorAlpha Garcia's favorite color is white. VictorAlpha Garcia's favorite programming language is TypeScript. VictorAlpha Garcia's secret code is CODE-3021. VictorAlpha Garcia's office room is R421. | FatimaAlpha Garcia is a 24-year-old student from Nantes, France. FatimaAlpha Garcia studies sports analytics and works on a cloud monitoring dashboard. FatimaAlpha Garcia's favorite color is purple. FatimaAlpha Garcia's favorite programming language is TypeScript. FatimaAlpha Garcia's secret code is CODE-3005. FatimaAlpha Garcia's office room is R405. | BellaDelta Smith is a 24-year-old student from Madrid, Spain. BellaDelta Smith studies cloud infrastructure and works on a heart-rate analysis tool. BellaDelta Smith's favorite color is yellow. BellaDelta Smith's favorite programming language is Java. BellaDelta Smith's secret code is CODE-3079. BellaDelta Smith's office room is R479.`

### legacy_titan - medium - medium_forget_secret_code_no_other_secret_code
- Category: forget_with_nearby_distractors
- Reason: contains forgotten value(s): ['CODE-3077']
- Question: What is ZaneGamma Lopez's secret code?
- Expected any: `["do not know", "don't know", "not know", "no memory"]`
- Expected all: `[]`
- Forbidden: `["CODE-3077"]`
- Retrieved/answer: `Legacy memory snapshot: ZaneGamma Lopez is a 24-year-old student from Nantes, France. ZaneGamma Lopez studies sports analytics and works on a cloud monitoring dashboard. ZaneGamma Lopez's favorite color is turquoise. ZaneGamma Lopez's favorite programming language is TypeScript. ZaneGamma Lopez's secret code is CODE-3077. ZaneGamma Lopez's office room is R477. | JuliaGamma Lopez is a 24-year-old student from Madrid, Spain. JuliaGamma Lopez studies human-computer interaction and works on a phishing detection tool. JuliaGamma Lopez's favorite color is white. JuliaGamma Lopez's favorite programming language is TypeScript. JuliaGamma Lopez's secret code is CODE-3061. JuliaGamma Lopez's office room is R461. | TaraBeta Lopez is a 24-year-old student from Hanoi, Vietnam. TaraBeta Lopez studies UI design and works on a multilingual chatbot. TaraBeta Lopez's favorite color is purple. TaraBeta Lopez's favorite programming language is TypeScript. TaraBeta Lopez's secret code is CODE-3045. TaraBeta Lopez's office room is R445. | DianaBeta Lopez is a 24-year-old student from Nantes, France. DianaBeta Lopez studies sports analytics and works on a cloud monitoring dashboard. DianaBeta Lopez's favorite color is yellow. DianaBeta Lopez's favorite programming language is TypeScript. DianaBeta Lopez's secret code is CODE-3029. DianaBeta Lopez's office room is R429. | NoraAlpha Lopez is a 24-year-old student from Madrid, Spain. NoraAlpha Lopez studies human-computer interaction and works on a phishing detection tool. NoraAlpha Lopez's favorite color is blue. NoraAlpha Lopez's favorite programming language is TypeScript. NoraAlpha Lopez's secret code is CODE-3013. NoraAlpha Lopez's office room is R413. | BellaDelta Smith is a 24-year-old student from Madrid, Spain. BellaDelta Smith studies cloud infrastructure and works on a heart-rate analysis tool. BellaDelta Smith's favorite color is yellow. BellaDelta Smith's favorite programming language is Java. BellaDelta Smith's secret code is CODE-3079. BellaDelta Smith's office room is R479. | AlexDelta Rossi is a 24-year-old student from Hanoi, Vietnam. AlexDelta Rossi studies data engineering and works on a robotic navigation system. AlexDelta Rossi's favorite color is orange. AlexDelta Rossi's favorite programming language is Go. AlexDelta Rossi's secret code is CODE-3078. AlexDelta Rossi's office room is R478. | YaraGamma Wilson is a 24-year-old student from Madrid, Spain. YaraGamma Wilson studies software engineering and works on a medical diagnosis assistant. YaraGamma Wilson's favorite color is green. YaraGamma Wilson's favorite programming language is Rust. YaraGamma Wilson's secret code is CODE-3076. YaraGamma Wilson's office room is R476.`

### legacy_titan - medium - medium_retains_project_after_forget
- Category: retention_after_forget
- Reason: contains forbidden value(s): ['CODE-3077']
- Question: What does ZaneGamma Lopez work on?
- Expected any: `["cloud monitoring dashboard"]`
- Expected all: `[]`
- Forbidden: `["CODE-3077"]`
- Retrieved/answer: `Legacy memory snapshot: ZaneGamma Lopez is a 24-year-old student from Nantes, France. ZaneGamma Lopez studies sports analytics and works on a cloud monitoring dashboard. ZaneGamma Lopez's favorite color is turquoise. ZaneGamma Lopez's favorite programming language is TypeScript. ZaneGamma Lopez's secret code is CODE-3077. ZaneGamma Lopez's office room is R477. | JuliaGamma Lopez is a 24-year-old student from Madrid, Spain. JuliaGamma Lopez studies human-computer interaction and works on a phishing detection tool. JuliaGamma Lopez's favorite color is white. JuliaGamma Lopez's favorite programming language is TypeScript. JuliaGamma Lopez's secret code is CODE-3061. JuliaGamma Lopez's office room is R461. | TaraBeta Lopez is a 24-year-old student from Hanoi, Vietnam. TaraBeta Lopez studies UI design and works on a multilingual chatbot. TaraBeta Lopez's favorite color is purple. TaraBeta Lopez's favorite programming language is TypeScript. TaraBeta Lopez's secret code is CODE-3045. TaraBeta Lopez's office room is R445. | DianaBeta Lopez is a 24-year-old student from Nantes, France. DianaBeta Lopez studies sports analytics and works on a cloud monitoring dashboard. DianaBeta Lopez's favorite color is yellow. DianaBeta Lopez's favorite programming language is TypeScript. DianaBeta Lopez's secret code is CODE-3029. DianaBeta Lopez's office room is R429. | NoraAlpha Lopez is a 24-year-old student from Madrid, Spain. NoraAlpha Lopez studies human-computer interaction and works on a phishing detection tool. NoraAlpha Lopez's favorite color is blue. NoraAlpha Lopez's favorite programming language is TypeScript. NoraAlpha Lopez's secret code is CODE-3013. NoraAlpha Lopez's office room is R413. | The code CODE-9179 is printed on a router sticker. | The code CODE-9171 is printed on a router sticker. | The code CODE-9163 is printed on a router sticker.`

### new_titan - medium - medium_retains_project_after_forget
- Category: retention_after_forget
- Reason: missing one of expected_any: ['cloud monitoring dashboard']
- Question: What does ZaneGamma Lopez work on?
- Expected any: `["cloud monitoring dashboard"]`
- Expected all: `[]`
- Forbidden: `["CODE-3077"]`
- Retrieved/answer: `Based on memory: Zanegamma Lopez is from Nantes, France.`

### legacy_titan - medium - medium_hard_distractor_room
- Category: hard_distractors
- Reason: contains forbidden value(s): ['hotel has a room']
- Question: What is the real office room of BellaGamma Garcia?
- Expected any: `["R453"]`
- Expected all: `[]`
- Forbidden: `["hotel has a room"]`
- Retrieved/answer: `Legacy memory snapshot: BellaGamma Garcia is a 24-year-old student from Nantes, France. BellaGamma Garcia studies sports analytics and works on a cloud monitoring dashboard. BellaGamma Garcia's favorite color is blue. BellaGamma Garcia's favorite programming language is TypeScript. BellaGamma Garcia's secret code is CODE-3053. BellaGamma Garcia's office room is R453. | The hotel has a room named R472 but it is not an office. | The hotel has a room named R464 but it is not an office. | The hotel has a room named R456 but it is not an office. | The hotel has a room named R448 but it is not an office. | The hotel has a room named R440 but it is not an office. | The hotel has a room named R432 but it is not an office. | The hotel has a room named R424 but it is not an office.`

### legacy_titan - medium - medium_profile_summary
- Category: profile_summary
- Reason: contains forbidden value(s): ['router sticker']
- Question: Tell me the key things you know about LenaAlpha Meyer.
- Expected any: `[]`
- Expected all: `["Nantes", "cybersecurity", "gesture recognition prototype"]`
- Forbidden: `["router sticker", "blue sofa"]`
- Retrieved/answer: `Legacy memory snapshot: LenaAlpha Meyer is a 24-year-old student from Nantes, France. LenaAlpha Meyer studies cybersecurity and works on a gesture recognition prototype. LenaAlpha Meyer's favorite color is white. LenaAlpha Meyer's favorite programming language is Scala. LenaAlpha Meyer's secret code is CODE-3011. LenaAlpha Meyer's office room is R411. | The code CODE-9179 is printed on a router sticker. | The code CODE-9171 is printed on a router sticker. | The code CODE-9163 is printed on a router sticker. | The code CODE-9155 is printed on a router sticker. | The code CODE-9147 is printed on a router sticker. | The code CODE-9139 is printed on a router sticker. | The code CODE-9131 is printed on a router sticker.`

### new_titan - medium - medium_profile_summary
- Category: profile_summary
- Reason: missing expected_all: ['Nantes', 'cybersecurity', 'gesture recognition prototype']
- Question: Tell me the key things you know about LenaAlpha Meyer.
- Expected any: `[]`
- Expected all: `["Nantes", "cybersecurity", "gesture recognition prototype"]`
- Forbidden: `["router sticker", "blue sofa"]`
- Retrieved/answer: `Based on memory: Lenaalpha Meyer is a 24-year-old student.`

### legacy_titan - medium - medium_same_first_name_full_identity
- Category: identity_collision
- Reason: contains forbidden value(s): ['green', 'orange']
- Question: What is Sarah Nguyen's favorite color?
- Expected any: `["purple"]`
- Expected all: `[]`
- Forbidden: `["green", "orange"]`
- Retrieved/answer: `Legacy memory snapshot: Sarah Nguyen's favorite color is purple. | HanaAlpha Nguyen is a 24-year-old student from Madrid, Spain. HanaAlpha Nguyen studies cloud infrastructure and works on a heart-rate analysis tool. HanaAlpha Nguyen's favorite color is turquoise. HanaAlpha Nguyen's favorite programming language is Java. HanaAlpha Nguyen's secret code is CODE-3007. HanaAlpha Nguyen's office room is R407. | Sarah Wilson's favorite color is orange. | Sarah Martin's favorite color is green. | TaraAlpha Bernard is a 24-year-old student from Madrid, Spain. TaraAlpha Bernard studies cloud infrastructure and works on a heart-rate analysis tool. TaraAlpha Bernard's favorite color is yellow. TaraAlpha Bernard's favorite programming language is Scala. TaraAlpha Bernard's secret code is CODE-3019. TaraAlpha Bernard's office room is R419. | SamuelAlpha Singh is a 24-year-old student from Hanoi, Vietnam. SamuelAlpha Singh studies data engineering and works on a robotic navigation system. SamuelAlpha Singh's favorite color is orange. SamuelAlpha Singh's favorite programming language is C++. SamuelAlpha Singh's secret code is CODE-3018. SamuelAlpha Singh's office room is R418. | RosaAlpha Tanaka is a 24-year-old student from Nantes, France. RosaAlpha Tanaka studies sports analytics and works on a cloud monitoring dashboard. RosaAlpha Tanaka's favorite color is turquoise. RosaAlpha Tanaka's favorite programming language is JavaScript. RosaAlpha Tanaka's secret code is CODE-3017. RosaAlpha Tanaka's office room is R417. | QuentinAlpha Martin is a 24-year-old student from Madrid, Spain. QuentinAlpha Martin studies software engineering and works on a medical diagnosis assistant. QuentinAlpha Martin's favorite color is green. QuentinAlpha Martin's favorite programming language is Python. QuentinAlpha Martin's secret code is CODE-3016. QuentinAlpha Martin's office room is R416.`
