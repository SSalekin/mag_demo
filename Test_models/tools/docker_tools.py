"""Docker tools for the coding-agent workflow.

The functions are safe to import on machines without Docker. Docker execution is
only attempted when ``run_tests_in_docker`` is called.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

try:  # Package import when called from Test_models as a package.
    from .file_tools import STAGING_DIR, ensure_coding_dirs, write_file_to_staging
except ImportError:  # Direct script-style import fallback.
    from file_tools import STAGING_DIR, ensure_coding_dirs, write_file_to_staging  # type: ignore


def create_dockerfile(content: str) -> str:
    """Create ``Test_models/staging/Dockerfile``."""

    return write_file_to_staging("Dockerfile", content)


def create_docker_compose(content: str) -> str:
    """Create ``Test_models/staging/docker-compose.yml``."""

    return write_file_to_staging("docker-compose.yml", content)


def _docker_compose_command() -> list[str] | None:
    """Return the available Docker Compose command, if any."""

    docker = shutil.which("docker")
    if docker:
        version = subprocess.run(
            [docker, "compose", "version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if version.returncode == 0:
            return [docker, "compose"]

    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return [docker_compose]

    return None


def _truncate(text: str, max_chars: int = 4_000) -> str:
    if not text:
        return ""
    return text[-max_chars:] if len(text) > max_chars else text


def run_tests_in_docker(timeout_seconds: int = 300) -> str:
    """Run tests from the staging folder with Docker.

    Preferred mode: ``docker compose up --build --abort-on-container-exit`` when a
    docker-compose file exists. If only a Dockerfile exists, a basic image build
    and container run are attempted.
    """

    ensure_coding_dirs()
    compose_file = STAGING_DIR / "docker-compose.yml"
    dockerfile = STAGING_DIR / "Dockerfile"

    if not compose_file.exists() and not dockerfile.exists():
        return "Error: no Dockerfile or docker-compose.yml found in staging."

    compose_cmd = _docker_compose_command()
    docker = shutil.which("docker")

    if compose_file.exists():
        if compose_cmd is None:
            return "Docker execution error: Docker Compose is not installed or not available in PATH."
        command = compose_cmd + ["up", "--build", "--abort-on-container-exit"]
        cleanup_command = compose_cmd + ["down"]
    else:
        if docker is None:
            return "Docker execution error: Docker is not installed or not available in PATH."
        image_name = "titan-coding-staging-test"
        build = subprocess.run(
            [docker, "build", "-t", image_name, "."],
            cwd=str(STAGING_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        if build.returncode != 0:
            return (
                f"Return code: {build.returncode}\n\n"
                f"--- STDOUT (truncated) ---\n{_truncate(build.stdout)}\n"
                f"--- STDERR (truncated) ---\n{_truncate(build.stderr)}"
            )
        command = [docker, "run", "--rm", image_name]
        cleanup_command = None

    try:
        result = subprocess.run(
            command,
            cwd=str(STAGING_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )

        output = f"Return code: {result.returncode}\n\n"
        output += f"--- STDOUT (truncated) ---\n{_truncate(result.stdout)}\n"
        if result.stderr:
            output += f"--- STDERR (truncated) ---\n{_truncate(result.stderr)}\n"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Docker execution exceeded the {timeout_seconds}-second time limit."
    except Exception as exc:  # pragma: no cover
        return f"Docker execution error: {exc}"
    finally:
        if cleanup_command is not None:
            subprocess.run(
                cleanup_command,
                cwd=str(STAGING_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
