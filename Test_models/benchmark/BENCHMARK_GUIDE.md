# MAG Demo benchmark guide

This folder currently contains historical research benchmarks, step-by-step regression tests, and the new BMAD coding-workflow tests.  Keeping every file in the daily workflow makes the project hard to use, so this guide separates what should be run often from what should be kept only for research/history.

## Recommended daily command

From `Test_models/`:

```powershell
python benchmark\run_benchmark_suite.py --suite core
```

For a shorter coding-only check:

```powershell
python benchmark\run_benchmark_suite.py --suite bmad
```

For the current end-to-end coding workflow check:

```powershell
python benchmark\run_benchmark_suite.py --suite full-coding --stop-on-fail
```

To see all available suites:

```powershell
python benchmark\run_benchmark_suite.py --list
```

## Keep as active regression tests

These files are useful for the current goal: a Titan-backed BMAD coding agent that generates, validates, archives, remembers, and reuses code.

| File | Why keep it |
|---|---|
| `test_main_chat_interface.py` | Validates the unified natural chat interface and routing. |
| `test_coding_tools.py` | Validates staging/workspace file utilities. |
| `test_bmad_multi_prompt_generation.py` | Main coverage for multi-prompt generation and strict generic-output rejection. |
| `test_bmad_docker_execution.py` | Validates Docker/sandbox path and safe fallback. |
| `test_bmad_memory_validation.py` | Validates controlled Titan write gate. |
| `test_bmad_project_archives.py` | Validates persistent project archives and manifests. |
| `test_bmad_project_reuse.py` | Validates archive search and reuse context. |
| `test_bmad_project_versioning_strict.py` | Best current regression for project reuse/versioning. |
| `test_bmad_project_memory_indexing.py` | Validates smart Titan records for generated projects. |
| `test_project_memory_policy.py` | Validates what generated-project records Titan may store, reject, deduplicate, and consolidate. |
| `test_bmad_full_coding_workflow.py` | End-to-end workflow: generate code, run QA, archive, store compact memory records, search, reuse and version a project. |
| `test_bmad_open_llm_quality.py` | Validates open-ended LLM output quality and repair-loop diagnostics without calling Ollama. |
| `test_titan_agent_memory.py` | Core Titan adapter smoke test. |
| `test_titan_consolidation_scheduler.py` | Validates automatic consolidation policy/scheduler. |
| `test_titan_memory_presets.py` | Validates small/medium/large/research memory presets. |
| `test_titan_project_memory_deduplication.py` | Validates deduplication of old project-memory pointers during consolidation. |

## Keep, but run only when changing legacy agent/memory code

| File | Why keep it |
|---|---|
| `test_agno_titan_direct_router.py` | Useful for deterministic Agno/Titan routing. |
| `test_agno_titan_agent_adapter.py` | Useful if the Agno adapter changes. |
| `test_multi_agent_titan_team.py` | Smoke test for the old Titan multi-agent prototype. |
| `test_multi_agent_routing_improvements.py` | Legacy multi-agent regression. |
| `test_multi_agent_holdout_improvements.py` | Legacy holdout regression. |
| `test_titan_model_with_ltm.py` | Useful when changing `titan_model.py`. |
| `test_titan_ltm_consolidation.py` | Useful when changing LTM consolidation internals. |

Run these with:

```powershell
python benchmark\run_benchmark_suite.py --suite agent --suite memory
```

## Keep as research benchmarks, not daily tests

These are important historically because they produced the original comparisons, but they are longer and more report-oriented.

| File | Purpose |
|---|---|
| `compare_memory_models.py` | Mamba/Titan/baseline comparison. |
| `compare_memory_models_stress.py` | Stress comparison across memory models. |
| `compare_titan_versions.py` | Legacy Titan vs current Titan comparison. |
| `compare_titan_versions_stress.py` | Stress comparison for Titan versions. |
| `compare_agent_memory_modes.py` | no-memory vs STM vs Titan agent modes. |
| `compare_multi_agent_titan_team.py` | old multi-agent Titan benchmark. |
| `compare_multi_agent_titan_holdout.py` | holdout/adversarial multi-agent benchmark. |
| `compare_multi_agent_titan_randomized.py` | randomized adversarial benchmark. |

Do not delete these yet; they are useful for the report and for proving why Titan/BMAD was chosen.  They should not be part of the fast development loop.

## Candidates to archive later

These files were useful during incremental development, but their coverage is now mostly included in newer tests.  They can be moved later to something like `benchmark/legacy_step_tests/` once the project is stable.

| File | Superseded by |
|---|---|
| `test_bmad_coding_team.py` | `test_bmad_multi_prompt_generation.py` + `test_main_chat_interface.py` |
| `test_bmad_useful_generation.py` | `test_bmad_multi_prompt_generation.py` |
| `test_bmad_llm_integration.py` | `test_bmad_multi_prompt_generation.py` + future LLM-specific tests |
| `test_bmad_auto_repair.py` | future repair-loop benchmark + `test_bmad_repair_diagnostics.py` |
| `test_bmad_generation_quality.py` | `test_bmad_multi_prompt_generation.py` |
| `test_bmad_runtime_validation.py` | `test_bmad_multi_prompt_generation.py` |
| `test_bmad_titan_readonly.py` | `test_bmad_memory_validation.py` + `test_bmad_project_memory_indexing.py` |
| `test_bmad_repair_diagnostics.py` | Keep until repair-loop benchmark is stronger. |
| `test_bmad_project_versioning.py` | `test_bmad_project_versioning_strict.py` |
| `test_bmad_project_versioning_hotfix.py` | `test_bmad_project_versioning_strict.py` |
| `test_bmad_project_memory_recall_hotfix.py` | `test_bmad_project_memory_indexing.py` + archive search tests |

## Suggested policy

1. Do not delete old benchmarks immediately.
2. Use `run_benchmark_suite.py --suite core` for everyday validation.
3. Move superseded step-specific files to a legacy folder only after the report/presentation no longer needs them.
4. Keep research `compare_*.py` scripts for the final report, but run them manually only when collecting results.
5. Any new feature should add or update one of the active regression tests instead of creating another one-off hotfix test.
6. Generated-project memory should store compact pointers, not full source code. Use `test_project_memory_policy.py` when changing `tools/project_memory.py` or consolidation/deduplication behavior.
7. Use `test_bmad_full_coding_workflow.py` or the `full-coding` suite before demos, because it verifies the whole pipeline rather than one isolated component.
