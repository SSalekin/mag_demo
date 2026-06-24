# Multi-agent Titan team prototype

## Objective

The next research question is:

> How can we include our Titan memory model in a multi-agent AI system, and which agent architecture is best?

The current prototype implements a manager-led multi-agent team with a shared Titan long-term memory.

## Agents

The team contains six roles:

1. **Manager**: orchestrates the task, retrieves Titan memory, selects agents, merges outputs and stores decisions.
2. **DevWeb**: web, frontend, backend API, routes and UI.
3. **DevSoft**: Python, architecture, classes, refactoring and Titan integration.
4. **DevOps**: environment, dependencies, Ollama/provider configuration and deployment commands.
5. **Tester**: unit tests, integration tests, benchmarks and regressions.
6. **Evaluator**: quality analysis, risks, trade-offs and final recommendation.

## Memory design

Titan is used as a **shared long-term memory**. The specialists do not write freely into memory. The Manager stores compact decisions after each task. This avoids memory pollution.

Stored examples:

- project conventions;
- selected agent decisions;
- benchmark results;
- validated technical choices;
- recurring errors and fixes.

## LLM policy

For now, the prototype stays on **Ollama** for tests. The model is configurable with:

```powershell
$env:OLLAMA_MODEL="llama3.2:1b"
```

The team code keeps the provider configurable so a future free/better model can replace Ollama if needed. We do **not** hardcode a cloud provider yet.

## Why deterministic role agents for the first prototype?

Running six local LLM agents with a small model can be slow and unreliable. The first objective is to validate the architecture:

```text
User -> Manager -> Shared Titan Memory -> Specialists -> Evaluator -> Manager response
```

Once the orchestration is stable, each role can be replaced by an Agno Agent or another LLM-backed agent.

## Commands

Run the focused test:

```powershell
python Test_models/benchmark/test_multi_agent_titan_team.py
```

Run the prototype:

```powershell
$env:OLLAMA_MODEL="llama3.2:1b"
python Test_models/agents/multi_agent_titan_team.py --ollama-model llama3.2:1b
```

Example interaction:

```text
/store This project uses pytest and type hints in new Python files.
/task Create a plan to add a FastAPI endpoint for Titan memory search.
```

Run the comparison:

```powershell
python Test_models/benchmark/compare_multi_agent_titan_team.py --save
```

## Current conclusion

The best architecture to test next is a **manager-led multi-agent team with shared Titan long-term memory**. The Manager controls memory writes, while specialized agents contribute implementation, testing, DevOps and evaluation perspectives.
