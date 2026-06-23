# Agno + Titan Memory Agent Integration

## Objective

The goal of this step is to answer:

> How can we include our Titan model in an AI agent, and which AI agent design is the best?

The recommended first design is a **single Agno agent with Titan as an external long-term memory layer**.

We do not replace the LLM with Titan. Titan is used as memory:

```text
User
  -> Agno Agent
  -> Titan memory tools: remember / recall / forget / consolidate
  -> Ollama LLM response
```

## Why single-agent first?

A single-agent design is easier to debug and benchmark than a multi-agent team. It also directly tests the value of our project: whether Titan improves long-term memory behavior.

Multi-agent systems can be tested later after the Titan memory layer is stable.

## Added files

```text
Test_models/agents/agno_titan_agent.py
Test_models/benchmark/test_agno_titan_agent_adapter.py
Test_models/docs/AGNO_TITAN_AGENT_INTEGRATION.md
```

## Titan tools exposed to Agno

The agent receives these tools:

```text
remember(text)
recall_memory(query, top_k=5)
forget_memory(query)
consolidate_memory(keep_existing_ltm=False)
memory_stats()
```

These tools wrap the existing `TitanAgentMemory` adapter.

## Install Agno

Inside the project virtual environment:

```powershell
python -m pip install -U agno ollama
```

Ollama must also have a local model installed, for example:

```powershell
ollama pull llama3.2:1b
```

## Tests

This test does not require Agno. It validates the tool adapter used by Agno:

```powershell
python -m compileall Test_models/agents Test_models/benchmark/test_agno_titan_agent_adapter.py
python Test_models/benchmark/test_agno_titan_agent_adapter.py
```

Expected result:

```text
Agno Titan adapter checks: 7/7 passed
```

## Manual Agno run

```powershell
$env:OLLAMA_MODEL="llama3.2:1b"
python Test_models/agents/agno_titan_agent.py --ollama-model llama3.2:1b
```

Example prompts:

```text
Remember that Lucas Martin's secret code is 8392.
What is Lucas Martin's secret code?
Remember that Lucas Martin's secret code is 1245.
What is Lucas Martin's secret code now?
Forget Lucas Martin's secret code.
What is Lucas Martin's secret code?
Consolidate memory.
```

## Next benchmark idea

Compare three agents:

1. No memory agent.
2. Simple short-term memory agent.
3. Agno agent with Titan long-term memory.

Suggested evaluation cases:

- user preference retention;
- project convention recall;
- update handling;
- targeted forgetting;
- identity collision;
- multi-session continuity;
- profile summary;
- coding convention recall.

## Current conclusion

The best next architecture is:

```text
Agno Level-3-style single agent + Titan long-term memory tools
```

A coding agent inspired by Agno/Gcode can be tested after the simple Agno Titan agent is validated.
