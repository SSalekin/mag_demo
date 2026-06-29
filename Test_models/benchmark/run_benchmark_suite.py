"""Run curated benchmark suites for the MAG Demo project.

This file intentionally does not replace the historical benchmark scripts.  It
provides a small, explicit entry point for the tests that are still useful in
normal development, so the benchmark/ directory can keep research history
without forcing every old step-specific script to be run manually.

Usage from Test_models/:
    python benchmark\run_benchmark_suite.py --suite core
    python benchmark\run_benchmark_suite.py --suite bmad
    python benchmark\run_benchmark_suite.py --list
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BenchmarkScript:
    path: str
    reason: str


SUITES: dict[str, list[BenchmarkScript]] = {
    "core": [
        BenchmarkScript("benchmark/test_main_chat_interface.py", "Unified natural chat interface and command routing."),
        BenchmarkScript("benchmark/test_coding_tools.py", "staging/workspace file tools."),
        BenchmarkScript("benchmark/test_bmad_multi_prompt_generation.py", "multi-prompt code generation and strict template coverage."),
        BenchmarkScript("benchmark/test_bmad_docker_execution.py", "Docker/sandbox integration with local fallback."),
        BenchmarkScript("benchmark/test_bmad_memory_validation.py", "controlled Titan write gate."),
        BenchmarkScript("benchmark/test_bmad_project_archives.py", "persistent project archives and manifests."),
        BenchmarkScript("benchmark/test_bmad_project_reuse.py", "project search and reuse context."),
        BenchmarkScript("benchmark/test_bmad_project_versioning_strict.py", "strict reuse/versioning regression checks."),
        BenchmarkScript("benchmark/test_bmad_project_memory_indexing.py", "Titan records for generated projects."),
        BenchmarkScript("benchmark/test_project_memory_policy.py", "policy for what generated-project memories Titan may store."),
        BenchmarkScript("benchmark/test_bmad_full_coding_workflow.py", "end-to-end coding workflow: generate, QA, archive, memory, reuse/version."),
        BenchmarkScript("benchmark/test_bmad_open_llm_quality.py", "open-ended LLM quality gate and repair loop."),
        BenchmarkScript("benchmark/test_titan_agent_memory.py", "Titan agent memory adapter."),
        BenchmarkScript("benchmark/test_titan_consolidation_scheduler.py", "nightly consolidation scheduler."),
        BenchmarkScript("benchmark/test_titan_memory_presets.py", "Titan small/medium/large/research presets."),
        BenchmarkScript("benchmark/test_titan_project_memory_deduplication.py", "project-memory deduplication during consolidation."),
    ],
    "bmad": [
        BenchmarkScript("benchmark/test_bmad_multi_prompt_generation.py", "multi-prompt generation."),
        BenchmarkScript("benchmark/test_bmad_docker_execution.py", "Docker/sandbox validation."),
        BenchmarkScript("benchmark/test_bmad_project_archives.py", "project archive manifests."),
        BenchmarkScript("benchmark/test_bmad_project_reuse.py", "generated project discovery."),
        BenchmarkScript("benchmark/test_bmad_project_versioning_strict.py", "versioning and complex-root regression."),
        BenchmarkScript("benchmark/test_bmad_project_memory_indexing.py", "project memories written to Titan."),
        BenchmarkScript("benchmark/test_project_memory_policy.py", "generated-project Titan memory storage policy."),
        BenchmarkScript("benchmark/test_bmad_open_llm_quality.py", "open-ended LLM quality and repair."),
        BenchmarkScript("benchmark/test_bmad_full_coding_workflow.py", "full BMAD coding workflow regression."),
        BenchmarkScript("benchmark/test_bmad_memory_validation.py", "controlled memory writes."),
    ],
    "memory": [
        BenchmarkScript("benchmark/test_titan_agent_memory.py", "Titan memory adapter."),
        BenchmarkScript("benchmark/test_titan_model_with_ltm.py", "TitanExternalMemory + standalone LTM."),
        BenchmarkScript("benchmark/test_titan_ltm_consolidation.py", "LTM consolidation behavior."),
        BenchmarkScript("benchmark/test_titan_consolidation_scheduler.py", "policy-based scheduler."),
        BenchmarkScript("benchmark/test_titan_memory_presets.py", "memory presets and parameter estimates."),
        BenchmarkScript("benchmark/test_titan_project_memory_deduplication.py", "project memory deduplication."),
        BenchmarkScript("benchmark/test_project_memory_policy.py", "generated-project memory policy."),
    ],
    "agent": [
        BenchmarkScript("benchmark/test_agno_titan_direct_router.py", "deterministic Agno/Titan route checks."),
        BenchmarkScript("benchmark/test_agno_titan_agent_adapter.py", "Agno/Titan adapter layer."),
        BenchmarkScript("benchmark/test_multi_agent_titan_team.py", "legacy Titan multi-agent team smoke test."),
        BenchmarkScript("benchmark/test_multi_agent_routing_improvements.py", "legacy multi-agent routing regressions."),
        BenchmarkScript("benchmark/test_multi_agent_holdout_improvements.py", "legacy holdout regressions."),
    ],
    # Kept for history only.  These are usually superseded by the core suite.
    "legacy-step": [
        BenchmarkScript("benchmark/test_bmad_coding_team.py", "early BMAD deterministic workflow."),
        BenchmarkScript("benchmark/test_bmad_useful_generation.py", "step 5 template checks, now covered by multi-prompt."),
        BenchmarkScript("benchmark/test_bmad_llm_integration.py", "step 6 LLM connector checks."),
        BenchmarkScript("benchmark/test_bmad_auto_repair.py", "step 7 repair loop checks."),
        BenchmarkScript("benchmark/test_bmad_generation_quality.py", "step 8 quality gate checks."),
        BenchmarkScript("benchmark/test_bmad_runtime_validation.py", "step 9 behavior checks, now covered by multi-prompt."),
        BenchmarkScript("benchmark/test_bmad_titan_readonly.py", "step 3 read-only Titan integration."),
        BenchmarkScript("benchmark/test_bmad_repair_diagnostics.py", "step 12 diagnostics checks."),
        BenchmarkScript("benchmark/test_bmad_project_versioning.py", "older project versioning checks."),
        BenchmarkScript("benchmark/test_bmad_project_versioning_hotfix.py", "superseded hotfix regression."),
        BenchmarkScript("benchmark/test_bmad_project_memory_recall_hotfix.py", "superseded recall hotfix regression."),
    ],
    # These are research benchmarks, not daily regression tests. They can be
    # slow/noisy and may write result files.
    "research": [
        BenchmarkScript("benchmark/compare_memory_models.py", "Mamba/Titan/baseline memory comparison."),
        BenchmarkScript("benchmark/compare_memory_models_stress.py", "stress comparison across memory models."),
        BenchmarkScript("benchmark/compare_titan_versions.py", "legacy vs current Titan comparison."),
        BenchmarkScript("benchmark/compare_titan_versions_stress.py", "stress comparison for Titan versions."),
        BenchmarkScript("benchmark/compare_agent_memory_modes.py", "no-memory vs STM vs Titan agent modes."),
        BenchmarkScript("benchmark/compare_multi_agent_titan_team.py", "multi-agent Titan benchmark."),
        BenchmarkScript("benchmark/compare_multi_agent_titan_holdout.py", "holdout/adversarial multi-agent benchmark."),
        BenchmarkScript("benchmark/compare_multi_agent_titan_randomized.py", "randomized adversarial multi-agent benchmark."),
    ],
}

SUITES["full-coding"] = [
    BenchmarkScript("benchmark/test_bmad_full_coding_workflow.py", "end-to-end generate -> QA -> archive -> memory -> reuse/version."),
    BenchmarkScript("benchmark/test_bmad_multi_prompt_generation.py", "multi-prompt generation and strict generic-output rejection."),
    BenchmarkScript("benchmark/test_bmad_docker_execution.py", "Docker/sandbox validation with local fallback."),
    BenchmarkScript("benchmark/test_bmad_project_archives.py", "persistent project archives and manifests."),
    BenchmarkScript("benchmark/test_bmad_project_reuse_v2.py", "exact archived project reuse workflow."),
    BenchmarkScript("benchmark/test_bmad_project_memory_indexing.py", "Titan records for generated projects."),
    BenchmarkScript("benchmark/test_project_memory_policy.py", "generated-project Titan memory storage policy."),
    BenchmarkScript("benchmark/test_project_archive_integrity.py", "workspace/projects and project_index consistency."),
]

# Useful when you want almost all active, non-research tests.
SUITES["all-safe"] = list(dict.fromkeys(SUITES["core"] + SUITES["agent"]))


def _repo_root() -> Path:
    # This file lives in Test_models/benchmark/.
    return Path(__file__).resolve().parents[1]


def _iter_scripts(suite_names: Iterable[str]) -> list[BenchmarkScript]:
    scripts: list[BenchmarkScript] = []
    seen: set[str] = set()
    for suite_name in suite_names:
        if suite_name not in SUITES:
            raise SystemExit(f"Unknown suite: {suite_name}. Use --list to see valid suites.")
        for script in SUITES[suite_name]:
            if script.path not in seen:
                scripts.append(script)
                seen.add(script.path)
    return scripts


def list_suites() -> None:
    print("Available benchmark suites:\n")
    for name, scripts in SUITES.items():
        print(f"{name} ({len(scripts)} script(s))")
        for script in scripts:
            print(f"  - {script.path}: {script.reason}")
        print()


def run_script(root: Path, script: BenchmarkScript, timeout: int) -> int:
    path = root / script.path
    if not path.exists():
        print(f"[MISSING] {script.path}")
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n=== RUN {script.path} ===")
    print(f"Reason: {script.reason}")
    result = subprocess.run(
        [sys.executable, script.path],
        cwd=root,
        env=env,
        text=True,
        timeout=timeout,
    )
    if result.returncode == 0:
        print(f"=== PASS {script.path} ===")
    else:
        print(f"=== FAIL {script.path} (exit={result.returncode}) ===")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run curated MAG Demo benchmark suites.")
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Suite to run. Can be repeated. Default: core.",
    )
    parser.add_argument("--list", action="store_true", help="List suites and exit.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop after the first failing benchmark.")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per script in seconds.")
    args = parser.parse_args()

    if args.list:
        list_suites()
        return 0

    suite_names = args.suite or ["core"]
    scripts = _iter_scripts(suite_names)
    root = _repo_root()

    failures: list[str] = []
    print(f"Running suite(s): {', '.join(suite_names)}")
    print(f"Root: {root}")
    print(f"Scripts: {len(scripts)}")

    for script in scripts:
        code = run_script(root, script, timeout=args.timeout)
        if code != 0:
            failures.append(script.path)
            if args.stop_on_fail:
                break

    print("\n=== SUMMARY ===")
    print(f"Passed: {len(scripts) - len(failures)}/{len(scripts)}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All selected benchmarks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
