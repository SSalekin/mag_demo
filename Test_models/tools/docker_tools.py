"""Docker tools for the coding-agent workflow.

The functions are safe to import on machines without Docker. Docker execution is
only attempted when explicitly requested by QA. When Docker is unavailable, the
helpers return a structured ``skipped`` result so the BMAD workflow can fall back
to local validation instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

try:  # Package import when called from Test_models as a package.
    from .file_tools import STAGING_DIR, ensure_coding_dirs, write_file_to_staging
except ImportError:  # Direct script-style import fallback.
    from file_tools import STAGING_DIR, ensure_coding_dirs, write_file_to_staging  # type: ignore


@dataclass(frozen=True)
class DockerCommandResult:
    """Result for one command executed inside the Docker sandbox."""

    name: str
    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass(frozen=True)
class DockerBatchResult:
    """Structured result for a Docker build plus one or more sandbox commands."""

    success: bool
    skipped: bool
    reason: str
    image_name: str
    build_returncode: int | None = None
    build_stdout: str = ""
    build_stderr: str = ""
    commands: list[DockerCommandResult] = field(default_factory=list)

    def to_text(self, max_chars: int = 4_000) -> str:
        """Return a compact backward-compatible text report."""

        if self.skipped:
            return f"Docker execution skipped: {self.reason}"

        lines = [
            f"Docker success: {self.success}",
            f"Docker image: {self.image_name}",
            f"Docker build return code: {self.build_returncode}",
        ]
        if self.build_stdout:
            lines.append("--- BUILD STDOUT (truncated) ---")
            lines.append(_truncate(self.build_stdout, max_chars=max_chars))
        if self.build_stderr:
            lines.append("--- BUILD STDERR (truncated) ---")
            lines.append(_truncate(self.build_stderr, max_chars=max_chars))
        for command in self.commands:
            lines.append(f"--- COMMAND {command.name}: return code {command.returncode} ---")
            lines.append("$ " + " ".join(command.command))
            if command.stdout:
                lines.append(_truncate(command.stdout, max_chars=max_chars))
            if command.stderr:
                lines.append(_truncate(command.stderr, max_chars=max_chars))
            if command.timed_out:
                lines.append("Command timed out.")
        return "\n".join(lines)


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
        try:
            version = subprocess.run(
                [docker, "compose", "version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
            if version.returncode == 0:
                return [docker, "compose"]
        except Exception:
            pass

    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return [docker_compose]

    return None


def _truncate(text: str, max_chars: int = 4_000) -> str:
    if not text:
        return ""
    return text[-max_chars:] if len(text) > max_chars else text


def _docker_cli() -> str | None:
    return shutil.which("docker")


def _docker_daemon_available(docker: str, timeout_seconds: int = 20) -> tuple[bool, str]:
    """Return whether the Docker CLI can reach a running daemon."""

    try:
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, "Docker daemon check timed out."
    except Exception as exc:
        return False, f"Docker daemon check failed: {exc}"

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "Docker daemon is not reachable.").strip()
        return False, message
    return True, "Docker daemon is available."


def run_commands_in_docker(
    commands: Sequence[tuple[str, Sequence[str]]],
    *,
    timeout_seconds: int = 300,
    image_name: str = "titan-bmad-staging-test",
    mount_staging: bool = True,
) -> DockerBatchResult:
    """Build the staging Docker image once and execute commands in it.

    Args:
        commands: Sequence of ``(name, command_argv)`` tuples. Each command is
            executed from ``/app`` inside the container.
        timeout_seconds: Timeout for build and each command.
        image_name: Local temporary Docker image name.
        mount_staging: Mount the staging folder into ``/app`` when running
            commands. This lets stateful CLI behavior checks share files across
            sequential commands while still using the container Python runtime.

    Returns:
        A structured ``DockerBatchResult``. Missing Docker or a stopped daemon
        returns ``skipped=True`` and ``success=True`` so callers can continue
        with local QA fallback. Build or command failures return
        ``skipped=False`` and ``success=False``.
    """

    ensure_coding_dirs()
    dockerfile = STAGING_DIR / "Dockerfile"
    if not dockerfile.exists():
        return DockerBatchResult(
            success=False,
            skipped=False,
            reason="No Dockerfile found in staging.",
            image_name=image_name,
        )

    docker = _docker_cli()
    if docker is None:
        return DockerBatchResult(
            success=True,
            skipped=True,
            reason="Docker CLI is not installed or not available in PATH.",
            image_name=image_name,
        )

    daemon_ok, daemon_reason = _docker_daemon_available(docker)
    if not daemon_ok:
        return DockerBatchResult(
            success=True,
            skipped=True,
            reason=daemon_reason,
            image_name=image_name,
        )

    try:
        build = subprocess.run(
            [docker, "build", "-t", image_name, "."],
            cwd=str(STAGING_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return DockerBatchResult(
            success=False,
            skipped=False,
            reason=f"Docker image build exceeded the {timeout_seconds}-second time limit.",
            image_name=image_name,
        )
    except Exception as exc:
        return DockerBatchResult(
            success=False,
            skipped=False,
            reason=f"Docker image build failed before completion: {exc}",
            image_name=image_name,
        )

    if build.returncode != 0:
        return DockerBatchResult(
            success=False,
            skipped=False,
            reason="Docker image build failed.",
            image_name=image_name,
            build_returncode=build.returncode,
            build_stdout=build.stdout,
            build_stderr=build.stderr,
        )

    results: list[DockerCommandResult] = []
    for name, command in commands:
        argv = [str(part) for part in command]
        run_command = [docker, "run", "--rm"]
        if mount_staging:
            run_command.extend(["-v", f"{STAGING_DIR.resolve()}:/app", "-w", "/app"])
        run_command.extend([image_name, *argv])
        try:
            completed = subprocess.run(
                run_command,
                cwd=str(STAGING_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
            results.append(
                DockerCommandResult(
                    name=name,
                    command=argv,
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            )
        except subprocess.TimeoutExpired as exc:
            results.append(
                DockerCommandResult(
                    name=name,
                    command=argv,
                    returncode=124,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or f"Command exceeded {timeout_seconds}-second timeout.",
                    timed_out=True,
                )
            )

    return DockerBatchResult(
        success=all(result.success for result in results),
        skipped=False,
        reason="Docker sandbox commands executed.",
        image_name=image_name,
        build_returncode=build.returncode,
        build_stdout=build.stdout,
        build_stderr=build.stderr,
        commands=results,
    )


def run_tests_in_docker(timeout_seconds: int = 300) -> str:
    """Backward-compatible Docker test runner returning a text report.

    This keeps the Step 1 public API intact while the BMAD QA now uses the more
    structured ``run_commands_in_docker`` function internally.
    """

    command: list[str]
    if (STAGING_DIR / "test_app.py").exists():
        command = ["python", "-m", "unittest", "test_app.py"]
    else:
        command = ["python", "-c", "import app; print('app import ok')"]
    result = run_commands_in_docker([("docker_tests", command)], timeout_seconds=timeout_seconds)
    return result.to_text()
