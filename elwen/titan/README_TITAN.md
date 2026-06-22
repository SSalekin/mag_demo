# Elwen - Titan External Memory Agent

## Goal

This folder contains a Titan-inspired external memory agent for the project:

**R&D on Memory Neural Network for AI Agent**

It is designed to be comparable with the Mamba external memory implementation.

## Main idea

This implementation uses:

- a **Titan-inspired neural long-term memory** backend;
- **Ollama/Gemma** as the LLM brain;
- a test-time trainable MLP memory using the associative objective:

```text
loss = || M(k) - v ||²
```

Workflow:

```text
User statement
→ atomization into memory facts
→ deterministic text vector
→ key/value projection
→ test-time neural memory update
→ hybrid retrieval
→ retrieved facts passed to Gemma/Ollama
→ final answer
```

## Difference with Mamba

Mamba version:

```text
Text → pretrained Mamba encoder → vector memory → hybrid retrieval
```

Titan version:

```text
Text → key/value vectors → neural MLP memory trained at test time → hybrid retrieval
```

So this Titan version is closer to the idea of a memory module that learns during inference.

## Run the chatbot

From the project root:

```powershell
python elwen\titan\titan_implementation.py
```

## Commands

```text
/help
/debug on
/debug off
/store <text>
/ask <question>
/search <query>
/forget <query>
/memories
/memories all
/reindex
/save
/load
/reset
/exit
```

## Benchmark

Retrieval-only benchmark:

```powershell
python elwen\titan\benchmark.py --retrieval-only
```

Full benchmark with Ollama/Gemma:

```powershell
python elwen\titan\benchmark.py
```

## Limitations

This is not the complete Google Titans architecture. It is a practical prototype:

- the text encoder is a lightweight deterministic hashed encoder;
- the Titan memory is a trainable MLP updated at test time;
- retrieval combines neural memory scoring with lexical/entity/property safeguards;
- the goal is to compare this Titan-inspired approach with the Mamba prototype.


## v2 fixes

- Fixed full-name possessive updates such as `Sarah Nguyen's favorite color is now black`.
- Added subject-aware entity matching.
- Added stricter filtering for single-value properties such as secret codes, favorite colors, languages, and office rooms.
