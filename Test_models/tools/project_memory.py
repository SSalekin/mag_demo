"""Titan memory records for validated generated projects.

Project archives keep the real files under ``workspace/projects``. Titan should
not store full source code. It should store compact, searchable pointers that
let agents later find the project, read its manifest and reuse the files.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Protocol, Sequence


class ProjectMemoryStore(Protocol):
    """Small protocol used by BMAD to write validated project memories."""

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> Sequence[Any]: ...


@dataclass(frozen=True)
class ProjectMemoryRecord:
    """One compact Titan memory record about a generated project."""

    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _join(values: Sequence[Any] | None, default: str = "none") -> str:
    cleaned = [str(value) for value in (values or []) if str(value).strip()]
    return ", ".join(cleaned) if cleaned else default



def _project_aliases(project_kind: str, prompt: str) -> list[str]:
    """Return natural-language aliases that make project memories searchable.

    Titan's lexical retrieval works best when the memory text contains ordinary
    words as well as machine slugs.  For example, ``email_phone_validator`` is a
    good stable id, but a user will later ask for "email validator projects".
    These aliases prevent the memory from depending only on underscores or exact
    project ids.
    """

    lower = f"{project_kind} {prompt}".lower()
    aliases: list[str] = []
    if "email" in lower or "phone" in lower or "validator" in lower:
        aliases.extend(["email validator", "phone validator", "email and phone validator", "validator project"])
    if "csv" in lower or "statistics" in lower:
        aliases.extend(["csv statistics", "csv reader", "statistics calculator", "data analysis project"])
    if "caesar" in lower or "cipher" in lower:
        aliases.extend(["caesar cipher", "encryption project", "decrypt messages", "encrypt messages"])
    if "quadratic" in lower or "equation" in lower or "roots" in lower:
        aliases.extend(["quadratic solver", "equation solver", "root calculator", "complex roots project"])
    if "memory" in lower or "memories" in lower:
        aliases.extend(["memory store", "user memories", "retrieve memories", "memory project"])
    if "todo" in lower or "task" in lower:
        aliases.extend(["todo list", "task manager", "todo project"])
    if "organize" in lower or "extension" in lower or "folder" in lower:
        aliases.extend(["file organizer", "folder organizer", "extension sorter", "organize files project"])
    if "chatbot" in lower:
        aliases.extend(["chatbot", "chatbot with memory", "conversation bot"])
    if "test runner" in lower or "manual and unit tests" in lower:
        aliases.extend(["test runner", "unit test runner", "manual test runner"])

    # Preserve order while removing duplicates and empty strings.
    seen: set[str] = set()
    out: list[str] = []
    for alias in aliases:
        alias = alias.strip()
        if alias and alias not in seen:
            seen.add(alias)
            out.append(alias)
    return out

def _project_metadata(project_archive: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "bmad_project_archive",
        "project_id": str(project_archive.get("project_id", "")),
        "project_kind": str(project_archive.get("project_kind", "")),
        "project_path": str(project_archive.get("project_path", "")),
        "qa_status": str(project_archive.get("qa_status", "")),
        "docker_status": str(project_archive.get("docker_status", "")),
        "memory_type": "validated_generated_project",
    }


def build_project_memory_records(project_archive: dict[str, Any] | None) -> list[ProjectMemoryRecord]:
    """Build useful Titan memory records from one archived project manifest.

    The records are intentionally small and searchable. They preserve the stable
    pointer to the archived files, the original request, the project kind, main
    functions, test commands and version lineage. Full code stays on disk.
    """

    if not project_archive or project_archive.get("error"):
        return []

    project_id = str(project_archive.get("project_id") or "").strip()
    project_path = str(project_archive.get("project_path") or "").strip()
    prompt = str(project_archive.get("original_prompt") or "").strip()
    project_kind = str(project_archive.get("project_kind") or "generated_project").strip()
    files = [str(file) for file in (project_archive.get("files") or [])]
    functions = [str(function) for function in (project_archive.get("main_functions") or [])]
    commands = [str(command) for command in (project_archive.get("manual_test_commands") or [])]
    qa_status = str(project_archive.get("qa_status") or "unknown")
    docker_status = str(project_archive.get("docker_status") or "unknown")
    metadata = _project_metadata(project_archive)
    aliases = _project_aliases(project_kind, prompt)
    aliases_text = ", ".join(aliases) if aliases else project_kind.replace("_", " ")

    if not project_id or not project_path:
        return []

    records: list[ProjectMemoryRecord] = []
    records.append(
        ProjectMemoryRecord(
            text=(
                f"Validated generated project {project_id} is stored at {project_path}. "
                f"It was created for prompt: {prompt!r}. Project kind: {project_kind}. "
                f"Search aliases: {aliases_text}. "
                f"QA status: {qa_status}. Docker status: {docker_status}."
            ),
            metadata={**metadata, "memory_subtype": "project_summary", "aliases": aliases},
        )
    )
    records.append(
        ProjectMemoryRecord(
            text=(
                f"Generated project {project_id} contains files: {_join(files)}. "
                f"Use path {project_path} to read or restore these files for future modifications."
            ),
            metadata={**metadata, "memory_subtype": "project_files"},
        )
    )
    if functions:
        records.append(
            ProjectMemoryRecord(
                text=(
                    f"Generated project {project_id} exposes main Python functions: {_join(functions)}. "
                    f"Reuse these functions when a future request mentions {project_kind}, {aliases_text}, updates, extensions or modifications."
                ),
                metadata={**metadata, "memory_subtype": "project_functions", "aliases": aliases},
            )
        )
    if commands:
        records.append(
            ProjectMemoryRecord(
                text=(
                    f"Generated project {project_id} can be manually tested with: {_join(commands[:4], default='see README.md')}."
                ),
                metadata={**metadata, "memory_subtype": "project_tests", "aliases": aliases},
            )
        )

    archive_metadata = project_archive.get("metadata") if isinstance(project_archive.get("metadata"), dict) else {}
    base_project_id = archive_metadata.get("base_project_id") or archive_metadata.get("reused_project_ids")
    if base_project_id:
        if isinstance(base_project_id, list):
            lineage = _join([str(item) for item in base_project_id])
        else:
            lineage = str(base_project_id)
        records.append(
            ProjectMemoryRecord(
                text=(
                    f"Generated project {project_id} is a new version or reuse of archived project(s): {lineage}. "
                    f"For future changes, prefer the newest validated project {project_id} unless the user asks for an older version."
                ),
                metadata={**metadata, "memory_subtype": "project_lineage"},
            )
        )

    return records


def store_project_memory_records(memory: ProjectMemoryStore, records: Sequence[ProjectMemoryRecord]) -> int:
    """Store project records in Titan-compatible memory and return stored count."""

    stored = 0
    for record in records:
        result = memory.store(record.text, metadata=record.metadata)
        try:
            stored += len(result)
        except TypeError:
            stored += 1
    return stored
