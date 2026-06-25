#!/usr/bin/env python3
"""Minimal checks for the coding workflow tools.

This test intentionally does not use Agno, Ollama or Titan. It validates only the
staging/workspace/Docker-file tool layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.docker_tools import create_docker_compose, create_dockerfile
from tools.file_tools import (
    STAGING_DIR,
    WORKSPACE_DIR,
    clear_staging,
    clear_workspace,
    list_staging_files,
    list_workspace_files,
    publish_to_workspace,
    read_staging_file,
    write_code_to_staging,
    write_file_to_staging,
)


def check(condition: bool, label: str, failures: list[str]) -> None:
    if not condition:
        failures.append(label)


def main() -> int:
    failures: list[str] = []

    clear_staging()
    clear_workspace()

    r1 = write_file_to_staging("src/app.py", "def add(a, b):\n    return a + b\n")
    r2 = write_code_to_staging(
        "tests/test_app.py",
        "from src.app import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    )
    check("Success" in r1, "write_file_to_staging should write src/app.py", failures)
    check("Success" in r2, "write_code_to_staging alias should write tests/test_app.py", failures)

    listing = list_staging_files()
    check("src/app.py" in listing, "list_staging_files should include src/app.py", failures)
    check("tests/test_app.py" in listing, "list_staging_files should include tests/test_app.py", failures)

    content = read_staging_file("src/app.py")
    check("def add" in content, "read_staging_file should read generated code", failures)

    dockerfile = create_dockerfile("FROM python:3.11-slim\nCMD [\"python\", \"-m\", \"pytest\", \"tests\"]\n")
    compose = create_docker_compose(
        "services:\n"
        "  app:\n"
        "    build: .\n"
        "    command: python -m pytest tests\n"
    )
    check("Success" in dockerfile, "create_dockerfile should create staging/Dockerfile", failures)
    check("Success" in compose, "create_docker_compose should create staging/docker-compose.yml", failures)
    check((STAGING_DIR / "Dockerfile").exists(), "Dockerfile should physically exist", failures)
    check((STAGING_DIR / "docker-compose.yml").exists(), "docker-compose.yml should physically exist", failures)

    publish_result = publish_to_workspace(["src", "tests"])
    check("Success" in publish_result, "publish_to_workspace should publish selected useful files", failures)
    check((WORKSPACE_DIR / "src" / "app.py").exists(), "workspace should contain src/app.py", failures)
    check((WORKSPACE_DIR / "tests" / "test_app.py").exists(), "workspace should contain tests/test_app.py", failures)
    check(not (WORKSPACE_DIR / "Dockerfile").exists(), "workspace should not contain Dockerfile", failures)
    check("empty" in list_staging_files().lower(), "staging should be empty after publishing", failures)
    check("src/app.py" in list_workspace_files(), "list_workspace_files should include published file", failures)

    # Path traversal protection is important because agents receive user text.
    bad_write = write_file_to_staging("../outside.py", "print('bad')")
    check("Error" in bad_write, "write_file_to_staging should reject path traversal", failures)
    check(not (ROOT.parent / "outside.py").exists(), "path traversal should not create outside.py", failures)

    clear_staging()
    clear_workspace()

    if failures:
        print("Coding tools checks failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Coding tools checks: 17/17 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
