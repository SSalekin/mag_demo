#!/usr/bin/env python3
"""Optional LLM code-generation helpers for the BMAD coding team.

This module is intentionally optional. It imports Agno and the Ollama provider
only inside ``AgnoOllamaCodeGenerator.generate`` so the deterministic BMAD
benchmarks can still run on machines where Agno, Ollama, or a local model are
not available.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class GeneratedCodeFile:
    """One source artifact proposed by an LLM coding provider."""

    path: str
    content: str
    purpose: str = "generated file"


@dataclass(frozen=True)
class CodeGenerationResult:
    """Result returned by an optional code-generation provider."""

    success: bool
    provider: str
    summary: str
    files: list[GeneratedCodeFile] = field(default_factory=list)
    error: str | None = None
    raw_response: str | None = None


class LLMCodeGenerator(Protocol):
    """Small protocol used by BMAD's Dev Agent."""

    def generate(self, spec: Any) -> CodeGenerationResult: ...


_ALLOWED_FILENAMES = {
    "app.py",
    "test_app.py",
    "README.md",
    "requirements.txt",
}


def _response_to_text(response: Any) -> str:
    """Extract text from common Agno response object shapes."""

    if response is None:
        return ""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
        if content is not None:
            return str(content)
    if isinstance(response, dict):
        if "content" in response:
            return str(response["content"])
        if isinstance(response.get("message"), dict):
            return str(response["message"].get("content", ""))
    return str(response)


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Parse a strict JSON object, accepting markdown fenced JSON if needed."""

    text = (text or "").strip()
    if not text:
        raise ValueError("LLM returned an empty response.")

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON payload must be an object.")
    return payload


def _validate_relative_path(path: str) -> str:
    """Validate a generated path before it is written to staging/."""

    cleaned = str(path or "").replace("\\", "/").strip()
    if not cleaned:
        raise ValueError("Generated file path is empty.")
    posix = PurePosixPath(cleaned)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"Unsafe generated path rejected: {path!r}")
    if cleaned not in _ALLOWED_FILENAMES:
        raise ValueError(
            f"Generated path {cleaned!r} is not allowed in this first LLM integration. "
            f"Allowed files: {sorted(_ALLOWED_FILENAMES)}"
        )
    return cleaned


def parse_code_generation_response(raw_response: str, provider: str = "llm") -> CodeGenerationResult:
    """Convert the LLM JSON response into validated generated files."""

    try:
        payload = _extract_json_payload(raw_response)
        raw_files = payload.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError("LLM JSON must contain a non-empty 'files' list.")

        files: list[GeneratedCodeFile] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_files, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"files[{index}] must be an object.")
            path = _validate_relative_path(str(item.get("path", "")))
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"files[{index}] has empty content.")
            if path in seen:
                raise ValueError(f"Duplicate generated path: {path}")
            seen.add(path)
            files.append(
                GeneratedCodeFile(
                    path=path,
                    content=content,
                    purpose=str(item.get("purpose") or "generated file"),
                )
            )

        required = {"app.py", "README.md"}
        missing = sorted(required - {file.path for file in files})
        if missing:
            raise ValueError(f"LLM output is missing required files: {missing}")

        summary = str(payload.get("summary") or "LLM generated code files.")
        return CodeGenerationResult(success=True, provider=provider, summary=summary, files=files, raw_response=raw_response)
    except Exception as exc:
        return CodeGenerationResult(
            success=False,
            provider=provider,
            summary="LLM output could not be used safely.",
            files=[],
            error=str(exc),
            raw_response=raw_response,
        )


class AgnoOllamaCodeGenerator:
    """Generate BMAD code artifacts with an Agno Agent backed by Ollama."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        provider_name: str = "agno_ollama",
        temperature: float = 0.0,
        markdown: bool = False,
    ) -> None:
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.provider_name = provider_name
        self.temperature = temperature
        self.markdown = markdown

    def _import_agno(self) -> tuple[Any, Any]:
        try:
            from agno.agent import Agent  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "Agno is not installed. Install/update it with: python -m pip install -U agno ollama"
            ) from exc

        try:
            from agno.models.ollama import Ollama  # type: ignore
        except Exception:
            try:
                from agno.models.ollama.chat import Ollama  # type: ignore
            except Exception as exc:  # pragma: no cover - depends on Agno version
                raise RuntimeError(
                    "Agno is installed, but the Ollama provider import path was not found. "
                    "Try: python -m pip install -U agno ollama"
                ) from exc
        return Agent, Ollama

    def build_prompt(self, spec: Any) -> str:
        requirements = "\n".join(f"- {item}" for item in getattr(spec, "requirements", []))
        criteria = "\n".join(f"- {item}" for item in getattr(spec, "acceptance_criteria", []))
        memory_context = getattr(spec, "memory_context", "") or "No Titan memory context provided."
        return f"""
You are the Dev Agent inside a BMAD software factory.
Generate a small, runnable Python project for the request below.

Return ONLY valid JSON. Do not wrap it in prose.
The JSON schema is:
{{
  "summary": "one short sentence",
  "files": [
    {{"path": "app.py", "purpose": "main implementation", "content": "..."}},
    {{"path": "README.md", "purpose": "usage documentation", "content": "..."}},
    {{"path": "test_app.py", "purpose": "unittest tests", "content": "..."}}
  ]
}}

Hard rules:
- Only generate these file paths: app.py, README.md, test_app.py, requirements.txt.
- app.py and README.md are mandatory.
- Create test_app.py when tests are requested.
- Prefer Python standard library only.
- Tests must use unittest so they run with: python -m unittest test_app.py
- app.py must expose pure functions that test_app.py can import.
- app.py must also have a CLI entry point guarded by if __name__ == "__main__".
- Do not use markdown fences around the JSON.
- Escape newlines correctly inside JSON strings.

User request:
{getattr(spec, "original_request", "")}

Functional goal:
{getattr(spec, "goal", "")}

Requirements:
{requirements}

Acceptance criteria:
{criteria}

Read-only Titan project memory context:
{memory_context}
""".strip()

    def build_repair_prompt(
        self,
        spec: Any,
        *,
        qa_failures: Sequence[str],
        current_files: Sequence[GeneratedCodeFile],
        attempt: int,
    ) -> str:
        failures = "\n".join(f"- {failure}" for failure in qa_failures) or "- Unknown QA failure"
        files_text = "\n\n".join(
            f"### {file.path}\n```\n{file.content}\n```" for file in current_files
        ) or "No current files were available."
        return f"""
You are the Dev Agent inside a BMAD software factory.
The first generated project failed QA. Repair the files.

Return ONLY valid JSON. Do not wrap it in prose.
The JSON schema is the same as the initial generation schema:
{{
  "summary": "one short sentence describing the repair",
  "files": [
    {{"path": "app.py", "purpose": "main implementation", "content": "..."}},
    {{"path": "README.md", "purpose": "usage documentation", "content": "..."}},
    {{"path": "test_app.py", "purpose": "unittest tests", "content": "..."}}
  ]
}}

Hard rules:
- Only generate these file paths: app.py, README.md, test_app.py, requirements.txt.
- app.py and README.md are mandatory.
- Prefer Python standard library only.
- Tests must use unittest so they run with: python -m unittest test_app.py
- app.py must expose pure functions that test_app.py can import.
- app.py must also have a CLI entry point guarded by if __name__ == "__main__".
- Fix the QA failures. Do not reintroduce the same errors.
- Do not use markdown fences around the JSON.

Repair attempt: {attempt}

Original user request:
{getattr(spec, "original_request", "")}

Functional goal:
{getattr(spec, "goal", "")}

QA failures to fix:
{failures}

Current staged files:
{files_text}
""".strip()

    def repair(
        self,
        *,
        spec: Any,
        qa_failures: Sequence[str],
        current_files: Sequence[GeneratedCodeFile],
        attempt: int = 1,
    ) -> CodeGenerationResult:
        try:
            Agent, Ollama = self._import_agno()
            model_kwargs = {"id": self.model_name}
            try:
                model = Ollama(**model_kwargs, options={"temperature": self.temperature})
            except TypeError:
                model = Ollama(**model_kwargs)

            agent = Agent(
                name="BMAD Dev LLM Repair Agent",
                role="Repair Python project files after BMAD QA failure.",
                model=model,
                instructions=[
                    "Return only valid JSON matching the requested schema.",
                    "Do not call tools. Do not include explanations outside JSON.",
                ],
                markdown=self.markdown,
            )
            response = agent.run(
                self.build_repair_prompt(
                    spec,
                    qa_failures=qa_failures,
                    current_files=current_files,
                    attempt=attempt,
                )
            )
            raw_text = _response_to_text(response)
            return parse_code_generation_response(raw_text, provider=f"{self.provider_name}_repair")
        except Exception as exc:  # pragma: no cover - depends on local env
            return CodeGenerationResult(
                success=False,
                provider=f"{self.provider_name}_repair",
                summary="Agno/Ollama repair failed before producing usable files.",
                files=[],
                error=str(exc),
                raw_response=None,
            )

    def generate(self, spec: Any) -> CodeGenerationResult:
        try:
            Agent, Ollama = self._import_agno()
            model_kwargs = {"id": self.model_name}
            try:
                model = Ollama(**model_kwargs, options={"temperature": self.temperature})
            except TypeError:
                model = Ollama(**model_kwargs)

            agent = Agent(
                name="BMAD Dev LLM Code Generator",
                role="Generate safe Python project files as JSON for BMAD staging.",
                model=model,
                instructions=[
                    "Return only valid JSON matching the requested schema.",
                    "Do not call tools. Do not include explanations outside JSON.",
                ],
                markdown=self.markdown,
            )
            response = agent.run(self.build_prompt(spec))
            raw_text = _response_to_text(response)
            return parse_code_generation_response(raw_text, provider=self.provider_name)
        except Exception as exc:  # pragma: no cover - depends on local env
            return CodeGenerationResult(
                success=False,
                provider=self.provider_name,
                summary="Agno/Ollama generation failed before producing usable files.",
                files=[],
                error=str(exc),
                raw_response=None,
            )
