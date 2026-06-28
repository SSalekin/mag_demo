"""Persistent registry for validated generated coding projects.

The BMAD workflow publishes the latest accepted artifact to ``workspace/`` for
quick manual testing.  This module additionally snapshots each accepted project
under ``workspace/projects/<project_id>/`` with a ``project_manifest.json``.

Titan memory should store a compact reference to the project, not the full code:
project id, path, prompt, main functions, QA status and manual commands.  The
agent can later retrieve that memory, open the project folder and reuse or
modify the generated files instead of starting from scratch.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tools.file_tools import WORKSPACE_DIR, ensure_coding_dirs

PROJECTS_DIR = WORKSPACE_DIR / "projects"
PROJECT_INDEX_PATH = PROJECTS_DIR / "project_index.json"
MANIFEST_NAME = "project_manifest.json"


@dataclass(frozen=True)
class GeneratedProjectManifest:
    """Metadata describing one validated generated project."""

    project_id: str
    original_prompt: str
    project_kind: str
    project_path: str
    files: list[str]
    main_functions: list[str]
    manual_test_commands: list[str]
    unit_test_command: str | None
    qa_status: str
    docker_status: str
    created_at: str
    updated_at: str
    source: str = "bmad_coding_team"
    code_hash: str | None = None
    memory_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_project_registry() -> None:
    ensure_coding_dirs()
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PROJECT_INDEX_PATH.exists():
        PROJECT_INDEX_PATH.write_text("[]\n", encoding="utf-8")


def _slugify(text: str, default: str = "generated_project") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (slug or default)[:56]


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:10]


def _read_index() -> list[dict[str, Any]]:
    ensure_project_registry()
    try:
        data = json.loads(PROJECT_INDEX_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_index(index: list[dict[str, Any]]) -> None:
    ensure_project_registry()
    PROJECT_INDEX_PATH.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _extract_python_functions(path: Path) -> list[str]:
    if not path.exists() or path.suffix != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return names[:30]


def _copy_workspace_files(project_dir: Path, files: Sequence[str]) -> list[str]:
    copied: list[str] = []
    for relative in files:
        rel = Path(str(relative).replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            continue
        if rel.parts[0] == "projects":
            continue
        source = WORKSPACE_DIR / rel
        if not source.exists() or not source.is_file():
            continue
        destination = project_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(rel.as_posix())
    return sorted(copied)


def archive_generated_project(
    *,
    task: str,
    project_kind: str,
    workspace_files: Sequence[str],
    qa_passed: bool,
    docker_output: str | None = None,
    manual_test_commands: Sequence[str] | None = None,
    unit_test_command: str | None = "cd workspace; python -m unittest test_app.py; cd ..",
    metadata: dict[str, Any] | None = None,
) -> GeneratedProjectManifest:
    """Snapshot a validated workspace artifact into ``workspace/projects``.

    The archive is intentionally created after publication, because only
    published files have passed QA and Evaluator review.
    """

    ensure_project_registry()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base_slug = _slugify(project_kind if project_kind != "generic" else task)
    project_id = f"{base_slug}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{_short_hash(task)}"
    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=False)

    copied_files = _copy_workspace_files(project_dir, workspace_files)
    app_path = project_dir / "app.py"
    functions = _extract_python_functions(app_path)
    code_hash = None
    if app_path.exists():
        code_hash = _short_hash(app_path.read_text(encoding="utf-8", errors="replace"))

    docker_status = "not_run"
    if docker_output:
        lower = docker_output.lower()
        if "skipped" in lower:
            docker_status = "skipped"
        elif "failed" in lower or "error" in lower:
            docker_status = "failed"
        else:
            docker_status = "passed"

    manifest = GeneratedProjectManifest(
        project_id=project_id,
        original_prompt=task,
        project_kind=project_kind,
        project_path=project_dir.relative_to(WORKSPACE_DIR.parent).as_posix(),
        files=copied_files,
        main_functions=functions,
        manual_test_commands=list(manual_test_commands or []),
        unit_test_command=unit_test_command,
        qa_status="passed" if qa_passed else "failed",
        docker_status=docker_status,
        created_at=now,
        updated_at=now,
        code_hash=code_hash,
        memory_summary=build_project_memory_summary(
            project_id=project_id,
            task=task,
            project_kind=project_kind,
            project_path=project_dir.relative_to(WORKSPACE_DIR.parent).as_posix(),
            files=copied_files,
            functions=functions,
            qa_status="passed" if qa_passed else "failed",
            manual_test_commands=list(manual_test_commands or []),
        ),
        metadata=dict(metadata or {}),
    )
    (project_dir / MANIFEST_NAME).write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    index = _read_index()
    index.append({
        "project_id": manifest.project_id,
        "project_kind": manifest.project_kind,
        "original_prompt": manifest.original_prompt,
        "project_path": manifest.project_path,
        "created_at": manifest.created_at,
        "qa_status": manifest.qa_status,
        "files": manifest.files,
        "main_functions": manifest.main_functions,
    })
    _write_index(index)
    return manifest


def build_project_memory_summary(
    *,
    project_id: str,
    task: str,
    project_kind: str,
    project_path: str,
    files: Sequence[str],
    functions: Sequence[str],
    qa_status: str,
    manual_test_commands: Sequence[str],
) -> str:
    functions_text = ", ".join(functions) if functions else "unknown functions"
    files_text = ", ".join(files) if files else "no files"
    commands_text = "; ".join(manual_test_commands[:3]) if manual_test_commands else "see README.md"
    return (
        f"Validated generated project {project_id}: kind={project_kind}; prompt={task!r}; "
        f"path={project_path}; files={files_text}; main_functions={functions_text}; "
        f"QA={qa_status}; manual_tests={commands_text}."
    )


def list_generated_projects(limit: int = 20) -> str:
    """Return a readable list of archived generated projects."""

    index = _read_index()
    if not index:
        return "No generated project archives found."
    rows = index[-limit:]
    lines = [f"Generated project archives ({len(rows)} shown / {len(index)} total):"]
    for item in reversed(rows):
        functions = ", ".join(item.get("main_functions") or []) or "functions unknown"
        lines.append(
            f"- {item.get('project_id')} [{item.get('project_kind')}] "
            f"QA={item.get('qa_status')} path={item.get('project_path')} functions={functions}"
        )
    return "\n".join(lines)


def read_project_manifest(project_id: str) -> dict[str, Any] | None:
    """Read one project manifest by id."""

    safe_id = _slugify(project_id, default="")
    if not safe_id:
        return None
    manifest_path = PROJECTS_DIR / safe_id / MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None



def _project_search_score(item: dict[str, Any], query: str) -> int:
    """Simple deterministic score for matching a user request to archived projects."""

    query_words = set(re.findall(r"[a-zA-Z0-9_]{3,}", (query or "").lower()))
    if not query_words:
        return 0
    haystack_parts = [
        str(item.get("project_id", "")),
        str(item.get("project_kind", "")),
        str(item.get("original_prompt", "")),
        " ".join(str(x) for x in item.get("files", []) or []),
        " ".join(str(x) for x in item.get("main_functions", []) or []),
    ]
    haystack = " ".join(haystack_parts).lower()
    score = 0
    for word in query_words:
        if word in haystack:
            score += 2
    # Strong boosts for common semantic aliases.
    aliases = {
        "quadratic": ["quadratic", "solve_quadratic", "discriminant", "roots"],
        "equation": ["quadratic", "solve_quadratic", "discriminant"],
        "csv": ["csv", "statistics", "calculate_statistics"],
        "memory": ["memory", "memories", "store_memory", "retrieve"],
        "todo": ["todo", "task"],
        "email": ["email", "phone", "validator"],
        "phone": ["email", "phone", "validator"],
        "caesar": ["caesar", "cipher", "encrypt", "decrypt"],
        "cipher": ["caesar", "cipher", "encrypt", "decrypt"],
        "organize": ["organizer", "extension", "files"],
        "files": ["organizer", "extension", "files"],
    }
    for word in query_words:
        for alias in aliases.get(word, []):
            if alias in haystack:
                score += 3
    return score


def find_generated_projects(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Find archived projects relevant to a natural-language query."""

    index = _read_index()
    scored: list[tuple[int, dict[str, Any]]] = []
    for item in index:
        score = _project_search_score(item, query)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda pair: (pair[0], str(pair[1].get("created_at", ""))), reverse=True)
    return [dict(item) | {"match_score": score} for score, item in scored[: max(0, int(limit))]]


def format_project_search_results(query: str, limit: int = 5) -> str:
    """Return readable project search results for the chat interface."""

    matches = find_generated_projects(query, limit=limit)
    if not matches:
        return f"No generated project archive matched: {query!r}"
    lines = [f"Generated project matches for {query!r}:"]
    for item in matches:
        functions = ", ".join(item.get("main_functions") or []) or "functions unknown"
        lines.append(
            f"- {item.get('project_id')} [{item.get('project_kind')}] "
            f"score={item.get('match_score')} QA={item.get('qa_status')} "
            f"path={item.get('project_path')} functions={functions}"
        )
    return "\n".join(lines)


def format_project_detail(project_id: str) -> str:
    """Return readable details for one archived project."""

    manifest = read_project_manifest(project_id)
    if not manifest:
        return f"No project manifest found for: {project_id}"
    lines = [f"Generated project: {manifest.get('project_id')}"]
    lines.append(f"- kind: {manifest.get('project_kind')}")
    lines.append(f"- path: {manifest.get('project_path')}")
    lines.append(f"- prompt: {manifest.get('original_prompt')}")
    lines.append(f"- QA: {manifest.get('qa_status')} | Docker: {manifest.get('docker_status')}")
    lines.append("- files: " + (", ".join(manifest.get("files") or []) or "none"))
    lines.append("- functions: " + (", ".join(manifest.get("main_functions") or []) or "unknown"))
    commands = manifest.get("manual_test_commands") or []
    if commands:
        lines.append("Manual test commands:")
        lines.extend(f"- {command}" for command in commands)
    return "\n".join(lines)


def read_project_file(project_id: str, relative_path: str) -> str | None:
    """Read one safe file from an archived project."""

    manifest = read_project_manifest(project_id)
    if not manifest:
        return None
    rel = Path(str(relative_path).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        return None
    project_path = WORKSPACE_DIR.parent / str(manifest.get("project_path", ""))
    target = project_path / rel
    try:
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def build_project_reuse_context(query: str, limit: int = 3, max_chars_per_file: int = 1800) -> str:
    """Build compact context that lets BMAD reuse existing validated projects.

    This context is intentionally a summary plus short file previews.  Titan or
    the LLM should not receive entire large files by default; the manifest path
    remains the stable pointer for future deeper reuse.
    """

    matches = find_generated_projects(query, limit=limit)
    if not matches:
        return ""
    blocks: list[str] = ["Relevant generated project archives that may be reused:"]
    for item in matches:
        project_id = str(item.get("project_id"))
        manifest = read_project_manifest(project_id) or item
        blocks.append(
            f"Project {project_id}: kind={manifest.get('project_kind')}; "
            f"path={manifest.get('project_path')}; QA={manifest.get('qa_status')}; "
            f"functions={', '.join(manifest.get('main_functions') or []) or 'unknown'}; "
            f"prompt={manifest.get('original_prompt')!r}."
        )
        for filename in ["README.md", "app.py", "test_app.py"]:
            if filename not in (manifest.get("files") or []):
                continue
            content = read_project_file(project_id, filename)
            if not content:
                continue
            preview = content[:max_chars_per_file]
            if len(content) > max_chars_per_file:
                preview += "\n... [truncated]"
            blocks.append(f"--- {project_id}/{filename} preview ---\n{preview}")
    return "\n\n".join(blocks)
