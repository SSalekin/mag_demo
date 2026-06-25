\# Project Documentation — Memory Neural Network for AI Agent



\## 1. Project Overview



This project is called \*\*R\&D on Memory Neural Network for AI Agent\*\*.



The goal is to study how to add a reliable \*\*long-term memory\*\* to an AI agent based on a Large Language Model. Classical LLMs do not naturally keep persistent memory across sessions. They mainly rely on a limited context window.



The project therefore focuses on building an external memory system able to:



\* store information;

\* retrieve information;

\* update existing information;

\* forget targeted information;

\* consolidate long-term memory;

\* support single-agent and multi-agent AI systems.



The current implementation is located mainly in:



```text

Test\_models/

```



\---



\## 2. Main Technologies



The project uses:



\* Python;

\* PyTorch;

\* Ollama for local LLM inference;

\* Agno for agent integration;

\* a custom Titan-based memory model;

\* terminal-based interfaces;

\* benchmark scripts for evaluation.



The default local LLM for testing is:



```text

llama3.2:1b

```



This model is used because it is local, free, and compatible with normal PC constraints.



\---



\## 3. Memory Models



Several memory architectures were tested:



\* Transformer;

\* LSTM;

\* GRU;

\* GNN;

\* Mamba;

\* Titan.



The baseline models are stored in:



```text

Test\_models/models/

```



The most important current model is:



```text

Test\_models/models/titan\_model.py

```



This file contains the current Titan memory implementation, including:



\* structured memory items;

\* subject/property extraction;

\* active/inactive memory states;

\* key/value-style long-term memory;

\* targeted forgetting;

\* update handling;

\* consolidation;

\* retrieval and ranking.



Older and legacy implementations are kept for comparison, especially:



```text

Legacy/titan\_implementation.py

Test\_models/models/legacy\_titan\_model.py

```



\---



\## 4. Titan Memory



Titan was selected as the main direction because it showed the best compromise between accuracy, speed, and long-term memory behavior.



The current Titan memory supports:



\* storing facts;

\* retrieving relevant facts;

\* updating old values;

\* forgetting targeted memories;

\* separating similar identities;

\* consolidating active memories into long-term memory;

\* resisting distractors and noisy queries.



The consolidation method is important because it allows the system to reinforce long-term memories while keeping the explicit memory structure available for update and forget operations.



\---



\## 5. Agent Integration



Titan was first integrated into a simple agent through an adapter:



```text

Test\_models/agents/titan\_agent\_memory.py

```



This adapter exposes a clean API:



```text

store()

recall()

forget()

consolidate()

build\_context()

save()

load()

clear()

stats()

```



This makes Titan usable by different types of AI agents.



A simple local agent is available in:



```text

Test\_models/agents/simple\_titan\_agent.py

```



An Agno-based agent is available in:



```text

Test\_models/agents/agno\_titan\_agent.py

```



\---



\## 6. Multi-Agent System



The current final prototype is a manager-led multi-agent system with shared Titan memory.



It is implemented in:



```text

Test\_models/agents/multi\_agent\_titan\_team.py

```



The team contains six specialized agents:



\* \*\*Manager\*\*: coordinates the task and decides what should be stored in Titan memory;

\* \*\*DevWeb\*\*: handles web, API, and frontend-related tasks;

\* \*\*DevSoft\*\*: handles Python code, software architecture, and business logic;

\* \*\*DevOps\*\*: handles environment setup, commands, installation, and configuration;

\* \*\*Tester\*\*: handles tests, benchmarks, and regression checks;

\* \*\*Evaluator\*\*: analyzes results, limitations, risks, and gives an honest conclusion.



The Manager uses Titan as a shared long-term memory layer. This avoids allowing every agent to write freely into memory, which would make memory noisy and unreliable.



\---



\## 7. Main Benchmarks



The project includes several benchmark scripts.



\### Memory model benchmarks



```text

Test\_models/benchmark/compare\_memory\_models.py

Test\_models/benchmark/compare\_memory\_models\_stress.py

```



These compare Mamba, Titan, and other memory models.



\### Old Titan vs New Titan



```text

Test\_models/benchmark/compare\_titan\_versions.py

Test\_models/benchmark/compare\_titan\_versions\_stress.py

```



These compare the legacy Titan implementation with the new Titan memory.



\### Agent memory benchmarks



```text

Test\_models/benchmark/compare\_agent\_memory\_modes.py

```



This compares:



\* no-memory agent;

\* short-term agent;

\* Titan memory agent.



\### Multi-agent benchmarks



```text

Test\_models/benchmark/compare\_multi\_agent\_titan\_team.py

Test\_models/benchmark/compare\_multi\_agent\_titan\_holdout.py

Test\_models/benchmark/compare\_multi\_agent\_titan\_randomized.py

```



These compare:



\* no-memory agent;

\* single Titan agent;

\* multi-agent Titan team.



\---



\## 8. Final Multi-Agent Results



The randomized adversarial benchmark was run with 5 different seeds and 150 cases per seed.



Average results:



```text

no\_memory\_agent        : 0/750   = 0.0%

single\_titan\_agent     : 85/750  = 11.3%

multi\_agent\_titan\_team : 642/750 = 85.6%

```



These results show that the manager-led multi-agent architecture with shared Titan memory is currently the best option tested.



\---



\## 9. Current Conclusion



The project demonstrates that Titan memory can improve both single-agent and multi-agent systems.



The strongest architecture so far is:



```text

Manager-led multi-agent team

\+ shared Titan long-term memory

\+ configurable LLM provider

```



For now, the project uses Ollama because it is local and free. Later, the LLM provider could be changed if a better free option is available.



Potential future providers include:



\* a stronger local Ollama model;

\* Gemini free tier;

\* Groq free tier;

\* OpenRouter free models.



\---



\## 10. How to Run



Activate the virtual environment:



```powershell

.\\.venv\\Scripts\\activate

```



Set the Ollama model:



```powershell

$env:OLLAMA\_MODEL="llama3.2:1b"

```



Run the main interface:



```powershell

python Test\_models/main.py

```



Run the simple Titan agent:



```powershell

python Test\_models/agents/simple\_titan\_agent.py

```



Run the Agno Titan agent:



```powershell

python Test\_models/agents/agno\_titan\_agent.py --ollama-model llama3.2:1b

```



Run the multi-agent Titan team:



```powershell

python Test\_models/agents/multi\_agent\_titan\_team.py --ollama-model llama3.2:1b

```



Run the final randomized benchmark:



```powershell

python Test\_models/benchmark/compare\_multi\_agent\_titan\_randomized.py --num-cases 150 --seed 20260624 --save

```



\---



\## 11. Recommended Next Steps



The next steps are:



1\. clean the project structure;

2\. keep only final benchmark summaries;

3\. keep Ollama for current tests;

4\. make the LLM provider configurable;

5\. prepare a short presentation/demo;

6\. later test a better free LLM provider if needed.



