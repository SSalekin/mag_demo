"""File tools for the coding-agent workflow.

The staging folder is a temporary area where generated files are written and
tested. The workspace folder contains only validated outputs that were published
by the manager/evaluator workflow.

This module does not depend on Agno, Ollama or Titan. It can be tested alone.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = PROJECT_ROOT / "staging"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

# Technical files are needed for Docker execution but should not be published to
# the final workspace by default.
DEFAULT_PUBLISH_IGNORES = {
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitkeep",
}


def ensure_coding_dirs() -> None:
    """Create staging and workspace folders if they do not exist."""

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_relative_path(relative_path: str | Path) -> Path:
    """Validate a user-provided relative path.

    The coding agents should never be able to write outside staging/workspace.
    Absolute paths and path traversal with ".." are rejected.
    """

    if relative_path is None:
        raise ValueError("A relative path is required.")

    path = Path(str(relative_path).replace("\\", "/"))
    if path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {relative_path}")
    if not path.parts or str(path) in {".", ""}:
        raise ValueError("The path cannot be empty.")
    if any(part in {"..", ""} for part in path.parts):
        raise ValueError(f"Path traversal is not allowed: {relative_path}")
    return path


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _list_files(base_dir: Path) -> list[str]:
    ensure_coding_dirs()
    files: list[str] = []
    for path in sorted(base_dir.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            files.append(path.relative_to(base_dir).as_posix())
    return files


def write_file_to_staging(filename: str, content: str) -> str:
    """Write a generated file into ``Test_models/staging``.

    Args:
        filename: Relative file path, for example ``src/app.py``.
        content: File content to write.
    """

    try:
        ensure_coding_dirs()
        safe_path = _safe_relative_path(filename)
        target = STAGING_DIR / safe_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Success: file written to staging/{safe_path.as_posix()}"
    except Exception as exc:  # pragma: no cover - returned for agent readability
        return f"Error writing to staging: {exc}"


def write_code_to_staging(filename: str, content: str) -> str:
    """Backward-compatible alias for Antoine's original tool name."""

    return write_file_to_staging(filename, content)


def read_staging_file(filename: str, max_chars: int = 12_000) -> str:
    """Read a file from staging, truncating long files for agent context safety."""

    try:
        ensure_coding_dirs()
        safe_path = _safe_relative_path(filename)
        target = STAGING_DIR / safe_path
        if not target.exists() or not target.is_file():
            return f"Error: staging/{safe_path.as_posix()} does not exist."
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n... [truncated to {max_chars} characters]"
        return text
    except Exception as exc:  # pragma: no cover
        return f"Error reading staging file: {exc}"


def list_staging_files() -> str:
    """List all generated files currently present in staging."""

    try:
        files = _list_files(STAGING_DIR)
        if not files:
            return "The staging folder is empty."
        return "Files in staging:\n" + "\n".join(files)
    except Exception as exc:  # pragma: no cover
        return f"Error listing staging folder: {exc}"


def list_workspace_files() -> str:
    """List all validated files currently present in workspace."""

    try:
        files = _list_files(WORKSPACE_DIR)
        if not files:
            return "The workspace folder is empty."
        return "Files in workspace:\n" + "\n".join(files)
    except Exception as exc:  # pragma: no cover
        return f"Error listing workspace folder: {exc}"


def clear_staging() -> str:
    """Remove all generated files from staging while keeping the folder itself."""

    try:
        ensure_coding_dirs()
        for item in STAGING_DIR.iterdir():
            if item.name == ".gitkeep":
                continue
            _remove_path(item)
        return "The staging folder has been cleaned."
    except Exception as exc:  # pragma: no cover
        return f"Error cleaning staging: {exc}"


def clear_workspace() -> str:
    """Remove all validated files from workspace while keeping the folder itself."""

    try:
        ensure_coding_dirs()
        for item in WORKSPACE_DIR.iterdir():
            if item.name == ".gitkeep":
                continue
            _remove_path(item)
        return "The workspace folder has been cleaned."
    except Exception as exc:  # pragma: no cover
        return f"Error cleaning workspace: {exc}"


def _publish_one(relative_path: Path) -> list[str]:
    source = STAGING_DIR / relative_path
    destination = WORKSPACE_DIR / relative_path

    if not source.exists():
        raise FileNotFoundError(f"staging/{relative_path.as_posix()} does not exist")

    if destination.exists():
        _remove_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        shutil.copytree(source, destination)
        return [p.relative_to(WORKSPACE_DIR).as_posix() for p in destination.rglob("*") if p.is_file()]

    shutil.copy2(source, destination)
    return [relative_path.as_posix()]


def publish_to_workspace(useful_files: Sequence[str] | None = None) -> str:
    """Publish validated files from staging to workspace and then clean staging.

    Args:
        useful_files: Optional list of relative files or folders to publish. If not
            provided, every non-Docker technical item from staging is published.
    """

    try:
        ensure_coding_dirs()

        if useful_files is None:
            candidates = [
                item.relative_to(STAGING_DIR)
                for item in sorted(STAGING_DIR.iterdir())
                if item.name not in DEFAULT_PUBLISH_IGNORES
            ]
        else:
            candidates = [_safe_relative_path(item) for item in useful_files]

        if not candidates:
            return "No useful files to publish from staging."

        published: list[str] = []
        for relative_path in candidates:
            published.extend(_publish_one(relative_path))

        clear_staging()
        return (
            f"Success: {len(published)} file(s) published to workspace. "
            f"Published files: {', '.join(sorted(published))}. Staging cleaned."
        )
    except Exception as exc:  # pragma: no cover
        return f"Error publishing to workspace: {exc}"


# Ensure folders exist as soon as the tools are imported.
ensure_coding_dirs()
