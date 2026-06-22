#!/usr/bin/env python3
"""
Mamba External Memory for LLM Agent - v7
=======================================

Expected location in the project:
    mag_demo/elwen/mamba/mamba_implementation.py

Goal:
- Use a Mamba model as an external memory encoder.
- Use Ollama/Gemma as the LLM brain that writes the final response.
- Keep the chat interface natural and fluid by default.
- Keep debug/inspection tools available when needed.

Main improvements over the previous prototype:
1. Natural chat mode by default:
   - Statements are stored automatically.
   - Questions are answered naturally.
   - Retrieval details are hidden unless debug mode is enabled.

2. Safer memory management:
   - New memories are no longer overwritten just because their embeddings are similar.
   - Exact duplicates are ignored.
   - Old facts are only deactivated when the new statement is clearly an update
     about the same subject and property.

3. Better retrieval:
   - Retrieval combines Mamba semantic similarity, lexical overlap, entity/name match,
     property match, and slight recency.
   - This fixes common failures such as asking about "Hugo Bernard" and retrieving Elwen.

4. Safer forgetting:
   - /forget shows candidates and asks confirmation.
   - By default, it deletes only the best safe match.
   - /forget --all requires typing YES.

5. Cleaner interface:
   - No "[MEMORY SNIPPET: id]" in normal answers.
   - No retrieved-memory block unless debug mode is ON.
   - Compact colored status lines.

Commands:
- /help
- /debug on | /debug off
- /store <text>
- /ask <question>
- /search <query>
- /forget <query>
- /forget --all <query>
- /memories
- /memories all
- /stats
- /save
- /load
- /reset
- /exit
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]


# =============================================================================
# Console UI
# =============================================================================

USE_COLOR = True


class Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    GRAY = "\033[90m"


def color(text: str, code: str = "") -> str:
    if not USE_COLOR or not code:
        return text
    return f"{code}{text}{Ansi.RESET}"


def ok(text: str) -> None:
    print(color(f"✓ {text}", Ansi.GREEN))


def warn(text: str) -> None:
    print(color(f"⚠ {text}", Ansi.YELLOW))


def error(text: str) -> None:
    print(color(f"✗ {text}", Ansi.RED))


def info(text: str) -> None:
    print(color(f"• {text}", Ansi.BLUE))


def assistant_print(text: str) -> None:
    """Print the assistant answer in a clean chatbot-style box."""
    print()
    box("ASSISTANT", text, Ansi.CYAN)


def memory_print(text: str) -> None:
    """Print memory operations in a compact green box."""
    print()
    box("MEMORY", text, Ansi.GREEN)


def system_print(text: str) -> None:
    """Print system status in a compact blue box."""
    print()
    box("SYSTEM", text, Ansi.BLUE)


def terminal_width(default: int = 100) -> int:
    try:
        return min(max(shutil.get_terminal_size((default, 20)).columns, 72), 120)
    except Exception:
        return default


def wrap_text(text: str, width: int) -> List[str]:
    if not text:
        return [""]
    lines: List[str] = []
    for raw_line in str(text).splitlines():
        words = raw_line.split(" ")
        current = ""
        for word in words:
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= width:
                current += " " + word
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines or [""]


def box(title: str, body: str = "", code: str = "") -> None:
    width = terminal_width()
    inner = width - 4
    label = f" {title} "
    top = "┌" + label + "─" * max(0, inner - len(label) + 2) + "┐"
    bottom = "└" + "─" * (width - 2) + "┘"
    print(color(top, code))
    for line in wrap_text(body, inner):
        print(color("│ ", code) + line.ljust(inner) + color(" │", code))
    print(color(bottom, code))


def truncate(text: str, max_len: int = 95) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# =============================================================================
# Text helpers
# =============================================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "from",
    "is", "are", "am", "was", "were", "be", "been", "being", "as",
    "i", "me", "my", "mine", "you", "your", "he", "she", "it", "they", "we",
    "his", "her", "their", "this", "that", "these", "those",
    "do", "does", "did", "what", "where", "when", "who", "why", "how", "which",
    "tell", "give", "show", "more", "about", "please", "can", "could", "would",
    "should", "now", "currently", "actually", "learn", "remember", "that",
}

QUESTION_STARTERS = (
    "what", "where", "when", "who", "why", "how", "which",
    "do", "does", "did", "can", "could", "would", "should", "is", "are", "am",
    "tell me", "explain", "give me", "show me",
)

MEMORY_MARKERS = (
    "my name is", "i am", "i'm", "i live", "i study", "i work", "i like", "i love",
    "my favorite", "my favourite", "i prefer", "i want", "i need", "i was born",
    "i am currently", "remember that", "note that", "save this", "learn that",
    "lives in", "likes", "favorite color", "favourite color", "favorite programming language",
    "favourite programming language", "programming language", "secret code", "temporary access code",
    "student id", "office room", "portfolio password", "research group", "operating system",
    "was born", "is from", "works on", "studies", "is currently", "wants to",
)

UPDATE_MARKERS = (
    "actually", "now", "changed", "instead", "no longer", "correction", "update:",
    "is now", "currently",
)

FORGET_MARKERS = (
    "forget", "delete", "remove", "do not remember", "don't remember", "erase",
)

BROAD_MEMORY_PATTERNS = (
    "what do you know about me",
    "what do you remember about me",
    "summarize my profile",
    "summarise my profile",
    "who am i",
)


PROPERTY_PATTERNS: List[Tuple[str, str]] = [
    ("secret_code", r"\bsecret code\b|\btemporary access code\b|\baccess code\b"),
    ("student_id", r"\bstudent id\b|\bid number\b"),
    ("office_room", r"\boffice room\b|\broom\b"),
    ("password", r"\bpassword\b"),
    ("research_group", r"\bresearch group\b|\blab\b|\blaboratory\b"),
    ("favorite_color", r"\bfavo[u]?rite color\b|\bcolor\b|\bcolour\b"),
    ("favorite_food", r"\bfavo[u]?rite food\b|\bfood\b"),
    ("favorite_language", r"\bfavo[u]?rite programming language\b|\bprogramming language\b|\bfavo[u]?rite language\b"),
    ("favorite_os", r"\bfavo[u]?rite operating system\b|\boperating system\b|\bos\b"),
    ("favorite_opening", r"\bfavo[u]?rite chess opening\b|\bchess opening\b"),
    # Put study before location: a question like "Where does Sarah study?"
    # should retrieve Sarah's study facts, not only Sarah's origin/location.
    ("study", r"\bstud(?:y|ies|ying)\b|\bstudent\b|\bschool\b|\buniversity\b"),
    ("project", r"\bproject\b|\bworking on\b|\bworks on\b|\bbuild(?:ing)?\b|\bcreate\b"),
    ("work", r"\bwork[s]?\b|\bjob\b|\bengineer\b"),
    ("current_location", r"\bcurrently\b|\bcurrent location\b|\bwhere .* currently\b"),
    ("location", r"\blive[s]?\b|\blives in\b|\bwhere .* live\b|\bfrom\b"),
    ("age", r"\bage\b|\byears old\b"),
    ("name", r"\bname\b"),
]

SINGLE_VALUE_PROPERTIES = {
    "secret_code", "student_id", "office_room", "password", "research_group",
    "favorite_color", "favorite_food", "favorite_language", "favorite_os",
    "favorite_opening", "location", "current_location", "age", "name",
}


def now() -> float:
    return time.time()


def fmt_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))


def words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_À-ÖØ-öø-ÿ']+", text.lower())


def keywords(text: str) -> set[str]:
    return {w for w in words(text) if len(w) >= 3 and w not in STOPWORDS}


def lexical_overlap(query: str, memory_text: str) -> float:
    q = keywords(query)
    if not q:
        return 0.0
    m = keywords(memory_text)
    return len(q & m) / len(q)


def extract_entities(text: str) -> set[str]:
    """
    Lightweight entity extraction.
    Good enough for names like "Hugo Bernard", "Lucas", "Elwen Coroller".
    """
    found: set[str] = set()

    # Capitalized sequences.
    for match in re.finditer(r"\b[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+(?:\s+[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+){0,2}\b", text):
        phrase = match.group(0)
        for token in words(phrase):
            if len(token) >= 3 and token not in STOPWORDS:
                found.add(token)

    # Possessive names: Lucas's, Maya's
    for match in re.finditer(r"\b([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+)'s\b", text):
        token = match.group(1).lower()
        if len(token) >= 3:
            found.add(token)

    return found


def entity_match_score(query: str, memory_text: str) -> float:
    q_entities = extract_entities(query)
    if not q_entities:
        return 0.0
    m_entities = extract_entities(memory_text)
    if not m_entities:
        return 0.0
    return len(q_entities & m_entities) / len(q_entities)


def subject_entity_match_score(q_entities: set[str], subject: Optional[str]) -> float:
    """
    Compare the query entities with the stored canonical subject.

    This avoids retrieving "MarcoAlpha Wilson's secret code" for a question about
    "CarlosBeta Wilson's secret code" just because they share the last name Wilson.
    """
    if not q_entities or not subject:
        return 0.0
    subject_tokens = set(subject.split())
    if not subject_tokens:
        return 0.0
    return len(q_entities & subject_tokens) / len(q_entities)


def property_key(text: str) -> Optional[str]:
    lower = text.lower()
    for key, pattern in PROPERTY_PATTERNS:
        if re.search(pattern, lower):
            return key
    return None


def property_match_score(query: str, memory_text: str) -> float:
    q_key = property_key(query)
    if not q_key:
        return 0.0
    return 1.0 if property_key(memory_text) == q_key else 0.0


def is_broad_memory_question(question: str) -> bool:
    lower = question.lower().strip()
    return any(pattern in lower for pattern in BROAD_MEMORY_PATTERNS)


def extract_subject(text: str) -> Optional[str]:
    """
    Extract a canonical subject key.

    Fixed in this version:
    - "Sarah Nguyen's favorite color..." -> "sarah nguyen"
    - "PriyaAlpha Meyer's secret code..." -> "priyaalpha meyer"

    The previous implementation only handled one-word possessives, so updates
    with full names did not deactivate the old fact.
    """
    stripped = normalize_statement_for_storage(text)
    lower = stripped.lower()

    if lower.startswith(("my ", "i ", "i'm", "i am")):
        return "user"

    # Multi-word possessive: "Sarah Nguyen's ...", "PriyaAlpha Meyer's ..."
    possessive = re.match(
        r"^([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+(?:\s+[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+){0,2})'s\b",
        stripped,
    )
    if possessive:
        return " ".join(words(possessive.group(1)))

    named = re.match(
        r"^([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+(?:\s+[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+){0,2})\s+"
        r"(?:is|lives|likes|works|studies|has|wants|enjoys|speaks)\b",
        stripped,
    )
    if named:
        return " ".join(words(named.group(1)))

    return None


def has_update_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in UPDATE_MARKERS)


def looks_like_memory_statement(text: str) -> bool:
    """
    Conservative natural classification for storing.
    It should catch:
    - Lucas's secret code is 8392.
    - Hugo Bernard is a cybersecurity student...
    - Learn that Lucas's secret code is 8392.
    """
    stripped = clean_text(text)
    lower = stripped.lower()

    if not stripped or stripped.endswith("?"):
        return False

    if any(marker in lower for marker in MEMORY_MARKERS):
        return True

    # Named entity declarative facts.
    if re.match(
        r"^[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+(?:\s+[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+){0,2}\s+"
        r"(?:is|lives|likes|works|studies|has|wants|enjoys|speaks)\b",
        stripped,
    ):
        return True

    # Possessive named facts: Lucas's secret code is...
    if re.match(r"^[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+'s\s+.+\s+(?:is|are|was|were)\b", stripped):
        return True

    return False


def normalize_statement_for_storage(text: str) -> str:
    """
    Strip conversational/update prefixes used to teach the memory.

    Important:
    - "Actually, Sarah's favorite color is now black." becomes
      "Sarah's favorite color is now black."
    - The update intent is detected before this normalization in store().
    """
    stripped = clean_text(text)
    stripped = re.sub(r"(?i)^(learn|remember|note)\s+that\s+", "", stripped).strip()
    stripped = re.sub(r"(?i)^save\s+this\s*:\s*", "", stripped).strip()
    stripped = re.sub(r"(?i)^(actually|correction|update)\s*[:,]?\s*", "", stripped).strip()
    return stripped


def subject_display(subject: Optional[str]) -> Optional[str]:
    if not subject:
        return None
    if subject == "user":
        return "I"
    return " ".join(part.capitalize() for part in subject.split())


def split_sentences(text: str) -> List[str]:
    """Split a paragraph into sentence-like chunks while keeping short facts readable."""
    cleaned = normalize_statement_for_storage(text)
    cleaned = cleaned.replace(";", ".")
    chunks = re.split(r"(?<=[.!?])\s+", cleaned)
    out: List[str] = []
    for chunk in chunks:
        chunk = chunk.strip(" \t\r\n")
        if not chunk:
            continue
        out.append(chunk)
    return out or ([cleaned] if cleaned else [])


def split_memory_units(text: str) -> List[str]:
    """
    Atomize a long profile into smaller facts.

    Why:
    Storing a full paragraph as one memory is bad for updates. If only the location
    changes, we do not want to deactivate the whole paragraph and lose study/project
    facts. This function turns a paragraph into smaller memory units.

    Example:
        "Sarah Nguyen is a 23-year-old AI student from Hanoi, Vietnam.
         She studies NLP and works on a sentiment analysis project."

    becomes:
        "Sarah Nguyen is a 23-year-old AI student."
        "Sarah Nguyen is from Hanoi, Vietnam."
        "Sarah Nguyen studies NLP."
        "Sarah Nguyen works on a sentiment analysis project."
    """
    raw_sentences = split_sentences(text)
    units: List[str] = []
    current_subject: Optional[str] = None
    current_display: Optional[str] = None

    for raw in raw_sentences:
        sentence = raw.strip().rstrip(".")
        if not sentence:
            continue

        # Resolve leading pronouns using the last known subject.
        if current_display:
            sentence = re.sub(r"(?i)^(he|she|they)\s+", f"{current_display} ", sentence)
            sentence = re.sub(r"(?i)^his\s+", f"{current_display}'s ", sentence)
            sentence = re.sub(r"(?i)^her\s+", f"{current_display}'s ", sentence)
            sentence = re.sub(r"(?i)^their\s+", f"{current_display}'s ", sentence)

        detected_subject = extract_subject(sentence)
        if detected_subject:
            current_subject = detected_subject
            current_display = subject_display(current_subject)

        display = current_display or subject_display(extract_subject(sentence))

        # Pattern: "Sarah Nguyen is a 23-year-old AI student from Hanoi, Vietnam"
        if display:
            m = re.match(rf"^({re.escape(display)})\s+is\s+(.+?)\s+from\s+(.+)$", sentence, flags=re.I)
            if m:
                description = m.group(2).strip(" ,")
                place = m.group(3).strip(" ,")
                if description:
                    units.append(f"{display} is {description}.")
                if place:
                    units.append(f"{display} is from {place}.")
                continue

            # Pattern: "Sarah Nguyen studies NLP and works on a sentiment analysis project"
            m = re.match(rf"^({re.escape(display)})\s+stud(?:ies|y|ying)\s+(.+?)\s+and\s+works?\s+on\s+(.+)$", sentence, flags=re.I)
            if m:
                study = m.group(2).strip(" ,")
                project = m.group(3).strip(" ,")
                if study:
                    units.append(f"{display} studies {study}.")
                if project:
                    units.append(f"{display} works on {project}.")
                continue

            # Pattern: "Sarah Nguyen works with SQL and builds dashboards" stays one fact
            # because it describes a coherent work/project area.

        # Split only very clear "X. Y" was handled above; avoid over-splitting preferences.
        final = sentence.strip()
        if final:
            if not final.endswith("."):
                final += "."
            units.append(final)

    # Remove exact duplicates while preserving order.
    seen = set()
    deduped: List[str] = []
    for unit in units:
        key = normalize_text(unit)
        if key not in seen:
            seen.add(key)
            deduped.append(unit)

    return deduped


def classify_user_message(message: str) -> str:
    """
    Return one of: exit, command, ask, store, forget, chat.
    """
    text = clean_text(message)
    lower = text.lower()

    if lower in {"/exit", "exit", "quit", "/quit"}:
        return "exit"
    if lower.startswith("/"):
        return "command"
    if any(marker in lower for marker in FORGET_MARKERS):
        return "forget"
    if lower.endswith("?") or lower.startswith(QUESTION_STARTERS):
        return "ask"

    normalized_for_memory = normalize_statement_for_storage(text)
    if looks_like_memory_statement(normalized_for_memory):
        return "store"
    return "chat"


# =============================================================================
# Memory data model
# =============================================================================

@dataclass
class MemoryItem:
    id: int
    text: str
    embedding: torch.Tensor
    subject: Optional[str] = None
    property: Optional[str] = None
    active: bool = True
    created_at: float = field(default_factory=now)
    updated_at: float = field(default_factory=now)
    access_count: int = 0
    last_accessed_at: Optional[float] = None
    deactivated_reason: Optional[str] = None

    def to_serializable(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding,
            "subject": self.subject,
            "property": self.property,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at,
            "deactivated_reason": self.deactivated_reason,
        }

    @staticmethod
    def from_serializable(data: Dict[str, object]) -> "MemoryItem":
        return MemoryItem(
            id=int(data["id"]),
            text=str(data["text"]),
            embedding=data["embedding"].float().cpu(),  # type: ignore[union-attr]
            subject=data.get("subject"),  # type: ignore[arg-type]
            property=data.get("property"),  # type: ignore[arg-type]
            active=bool(data.get("active", True)),
            created_at=float(data.get("created_at", now())),
            updated_at=float(data.get("updated_at", now())),
            access_count=int(data.get("access_count", 0)),
            last_accessed_at=data.get("last_accessed_at"),  # type: ignore[arg-type]
            deactivated_reason=data.get("deactivated_reason"),  # type: ignore[arg-type]
        )


# =============================================================================
# Mamba external memory
# =============================================================================

class MambaVectorMemory:
    """
    External memory whose embeddings come from a Mamba language model.
    """

    def __init__(
        self,
        hf_model: str,
        device: str,
        memory_path: Optional[Path] = None,
        max_length: int = 512,
        max_items: int = 5000,
        exact_duplicate_only: bool = True,
    ) -> None:
        self.hf_model = hf_model
        self.device = device
        self.memory_path = memory_path
        self.max_length = max_length
        self.max_items = max_items
        self.exact_duplicate_only = exact_duplicate_only

        info(f"Loading tokenizer: {hf_model}")
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)

        dtype = torch.float16 if device == "cuda" else torch.float32
        info(f"Loading Mamba model on {device}: {hf_model}")
        self._model = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=dtype)
        self._model.to(device)
        self._model.eval()

        self._items: List[MemoryItem] = []
        self._next_id = 1
        self._matrix_cache: Optional[torch.Tensor] = None
        self._active_cache_ids: List[int] = []
        self._dirty = True

        if self.memory_path and self.memory_path.exists():
            try:
                self.load(self.memory_path)
                ok(f"Loaded {len(self.active_items)} active memory item(s) from {self.memory_path}")
            except Exception as exc:
                warn(f"Could not load existing memory file: {exc}")

    @property
    def items(self) -> List[MemoryItem]:
        return self._items

    @property
    def active_items(self) -> List[MemoryItem]:
        return [item for item in self._items if item.active]

    @property
    def inactive_items(self) -> List[MemoryItem]:
        return [item for item in self._items if not item.active]

    @torch.inference_mode()
    def embed(self, text: str) -> torch.Tensor:
        text = clean_text(text)
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self._model(**inputs, output_hidden_states=True, use_cache=False)
        last_hidden = outputs.hidden_states[-1][0]

        if "attention_mask" in inputs:
            mask = inputs["attention_mask"][0].to(last_hidden.dtype).unsqueeze(-1)
        else:
            mask = torch.ones(last_hidden.shape[0], 1, dtype=last_hidden.dtype, device=last_hidden.device)

        pooled = (last_hidden * mask).sum(dim=0) / mask.sum().clamp_min(1.0)
        return l2_normalize(pooled.detach().float().cpu())

    def store(self, text: str) -> Tuple[str, MemoryItem, List[MemoryItem]]:
        """
        Store a new memory.

        Update rule:
        - Exact duplicates are ignored.
        - If a new fact has the same subject + same single-value property,
          the older active fact is deactivated.
        - This fixes cases like:
            Sarah's favorite color is yellow.
            Actually, Sarah's favorite color is now black.
          or:
            Nathan's favorite programming language is Scala.
            Actually, Nathan's favorite programming language is now Python.
        """
        raw_text = clean_text(text)
        update_intent = has_update_marker(raw_text)

        text = normalize_statement_for_storage(raw_text)
        if not text:
            raise ValueError("Cannot store an empty memory.")

        normalized = normalize_text(text)
        for item in self.active_items:
            if normalize_text(item.text) == normalized:
                item.updated_at = now()
                return "duplicate_ignored", item, []

        subject = self._resolve_subject(extract_subject(text))
        prop = property_key(text)
        emb = self.embed(text)

        deactivated: List[MemoryItem] = []
        same_single_value_fact = subject and prop and prop in SINGLE_VALUE_PROPERTIES

        if same_single_value_fact:
            for item in self.active_items:
                if item.subject == subject and item.property == prop:
                    item.active = False
                    if update_intent:
                        item.deactivated_reason = f"explicitly updated by newer memory about {subject}/{prop}"
                    else:
                        item.deactivated_reason = f"replaced by newer single-value memory about {subject}/{prop}"
                    item.updated_at = now()
                    deactivated.append(item)

        item = MemoryItem(
            id=self._next_id,
            text=text,
            embedding=emb,
            subject=subject,
            property=prop,
            active=True,
        )
        self._next_id += 1
        self._items.append(item)
        self._dirty = True
        self._enforce_capacity()
        return "stored", item, deactivated

    def store_text(self, text: str) -> List[Tuple[str, MemoryItem, List[MemoryItem]]]:
        """
        Store a user message after atomizing it into smaller facts.
        """
        units = split_memory_units(text)
        if not units:
            return []
        results: List[Tuple[str, MemoryItem, List[MemoryItem]]] = []
        for unit in units:
            results.append(self.store(unit))
        return results

    def atomize_existing_memories(self) -> Tuple[int, int]:
        """
        Convert old broad paragraph memories into atomic facts.
        Useful after upgrading from older versions.
        """
        originals = list(self.active_items)
        changed = 0
        created = 0

        for item in originals:
            units = split_memory_units(item.text)
            if len(units) <= 1:
                continue

            item.active = False
            item.deactivated_reason = "reindexed into atomic facts"
            item.updated_at = now()
            changed += 1

            for unit in units:
                action, _, _ = self.store(unit)
                if action == "stored":
                    created += 1

        if changed:
            self._dirty = True
        return changed, created

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.12,
    ) -> List[Tuple[float, Dict[str, float], MemoryItem]]:
        """
        Hybrid retrieval:
        - Mamba semantic similarity
        - lexical overlap
        - entity/name match
        - property match
        - slight recency bonus

        This is much more robust than semantic-only retrieval for profile memory.
        """
        active = self.active_items
        if not active:
            return []

        if is_broad_memory_question(query):
            return [
                (1.0, {"semantic": 1.0, "lexical": 0.0, "entity": 0.0, "property": 0.0, "recency": 0.0}, item)
                for item in sorted(active, key=lambda item: item.updated_at, reverse=True)[: max(k, 8)]
            ]

        q_embedding = self.embed(query)
        matrix, active_items = self._active_matrix()
        semantic_scores = matrix @ q_embedding

        q_entities = extract_entities(query)
        q_property = property_key(query)
        current_time = now()

        scored: List[Tuple[float, Dict[str, float], MemoryItem]] = []
        for idx, item in enumerate(active_items):
            semantic = float(semantic_scores[idx].item())
            lexical = lexical_overlap(query, item.text)
            entity = max(
                entity_match_score(query, item.text),
                subject_entity_match_score(q_entities, item.subject),
            )
            prop = 1.0 if q_property and item.property == q_property else property_match_score(query, item.text)

            # Small recency bonus: useful for updated facts, but not enough to override relevance.
            age_hours = max(0.0, (current_time - item.updated_at) / 3600.0)
            recency = 1.0 / (1.0 + age_hours)

            # If the query explicitly contains a name, penalize memories that do not contain it.
            entity_penalty = -0.25 if q_entities and entity == 0.0 else 0.0

            # If the query explicitly asks for a property, boost matching properties.
            property_bonus = 0.12 if q_property and prop > 0 else 0.0

            hybrid = (
                0.42 * semantic
                + 0.28 * lexical
                + 0.22 * entity
                + 0.08 * prop
                + 0.03 * recency
                + entity_penalty
                + property_bonus
            )

            details = {
                "semantic": semantic,
                "lexical": lexical,
                "entity": entity,
                "property": prop,
                "recency": recency,
            }

            # Keep useful candidates. If an entity is specified, require entity match OR very high lexical/property match.
            if q_entities and entity == 0.0 and lexical < 0.34 and prop == 0.0:
                continue

            # For precise single-value questions such as secret code, favorite color,
            # office room, etc., a partial entity match is dangerous.
            # Example: "CarlosBeta Wilson's secret code" must not retrieve
            # "MarcoAlpha Wilson's secret code" just because both share "Wilson".
            if q_entities and q_property in SINGLE_VALUE_PROPERTIES and prop > 0 and entity < 1.0:
                continue

            if hybrid >= min_score:
                scored.append((hybrid, details, item))

        scored.sort(key=lambda row: row[0], reverse=True)
        results = scored[: max(1, min(k, len(scored)))]

        access_time = now()
        for _, _, item in results:
            item.access_count += 1
            item.last_accessed_at = access_time

        return results

    def search_forget_candidates(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[float, Dict[str, float], MemoryItem]]:
        """
        Forget is stricter than normal retrieval.
        """
        candidates = self.retrieve(query, k=max(k, 8), min_score=-1.0)
        safe: List[Tuple[float, Dict[str, float], MemoryItem]] = []
        for hybrid, details, item in candidates:
            lexical = details["lexical"]
            entity = details["entity"]
            prop = details["property"]
            semantic = details["semantic"]

            if lexical >= 0.34 or (entity > 0 and prop > 0) or semantic >= 0.72:
                safe.append((hybrid, details, item))

        safe.sort(key=lambda row: row[0], reverse=True)
        return safe[:k]

    def deactivate_ids(self, ids: Iterable[int], reason: str = "forgotten") -> List[MemoryItem]:
        ids_set = set(ids)
        changed: List[MemoryItem] = []
        for item in self._items:
            if item.id in ids_set and item.active:
                item.active = False
                item.deactivated_reason = reason
                item.updated_at = now()
                changed.append(item)
        if changed:
            self._dirty = True
        return changed

    def reset(self) -> None:
        self._items.clear()
        self._next_id = 1
        self._dirty = True

    def stats(self) -> Dict[str, object]:
        return {
            "active_items": len(self.active_items),
            "inactive_items": len(self.inactive_items),
            "total_items": len(self.items),
            "hf_model": self.hf_model,
            "device": self.device,
            "max_items": self.max_items,
            "memory_path": str(self.memory_path) if self.memory_path else None,
        }

    def save(self, path: Optional[Path] = None) -> None:
        target = path or self.memory_path
        if target is None:
            raise ValueError("No memory path configured.")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 7,
            "hf_model": self.hf_model,
            "next_id": self._next_id,
            "items": [item.to_serializable() for item in self._items],
        }
        torch.save(payload, target)

    def load(self, path: Optional[Path] = None) -> None:
        target = path or self.memory_path
        if target is None:
            raise ValueError("No memory path configured.")
        payload = torch.load(target, map_location="cpu")
        self._items = [MemoryItem.from_serializable(raw) for raw in payload.get("items", [])]
        self._next_id = int(payload.get("next_id", len(self._items) + 1))
        self._dirty = True

    def _resolve_subject(self, subject: Optional[str]) -> Optional[str]:
        """
        Resolve short subjects to an existing canonical subject.

        Example:
        if "sarah nguyen" already exists and the user writes "Sarah is from Danang",
        we map subject "sarah" to "sarah nguyen" so the location update targets
        the right person.
        """
        if not subject or subject == "user":
            return subject

        active_subjects = sorted({item.subject for item in self.active_items if item.subject})
        if subject in active_subjects:
            return subject

        first = subject.split()[0]
        matches = [s for s in active_subjects if s and s.split()[0] == first]
        if len(matches) == 1:
            return matches[0]

        return subject

    def _active_matrix(self) -> Tuple[torch.Tensor, List[MemoryItem]]:
        active = self.active_items
        ids = [item.id for item in active]
        if not self._dirty and self._matrix_cache is not None and self._active_cache_ids == ids:
            return self._matrix_cache, active

        if not active:
            raise RuntimeError("No active memory.")
        self._matrix_cache = torch.stack([item.embedding for item in active], dim=0)
        self._active_cache_ids = ids
        self._dirty = False
        return self._matrix_cache, active

    def _enforce_capacity(self) -> None:
        if len(self._items) <= self.max_items:
            return

        # Remove inactive memories first.
        inactive = self.inactive_items
        if inactive:
            inactive_sorted = sorted(inactive, key=lambda item: item.updated_at)
            to_remove = len(self._items) - self.max_items
            remove_ids = {item.id for item in inactive_sorted[:to_remove]}
            self._items = [item for item in self._items if item.id not in remove_ids]
            self._dirty = True

        if len(self._items) <= self.max_items:
            return

        # If still too large, remove least recently updated active memories.
        self._items.sort(key=lambda item: (item.active, item.updated_at), reverse=True)
        self._items = self._items[: self.max_items]
        self._dirty = True


# =============================================================================
# Ollama + prompt
# =============================================================================

def ollama_chat(model: str, system: str, user: str) -> str:
    try:
        import ollama  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import `ollama`. Install it with `pip install ollama`, "
            "and make sure Ollama for Windows is running."
        ) from exc

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    return (content or "").strip()


def build_prompt(
    question: str,
    retrieved: List[Tuple[float, Dict[str, float], MemoryItem]],
    show_citations: bool = False,
) -> Tuple[str, str]:
    citation_rule = (
        "At the end of the answer, cite the memory id like [memory: id]."
        if show_citations
        else "Do not mention memory snippets, memory IDs, or retrieval scores in the final answer."
    )

    system = (
        "You are a helpful assistant connected to an external Mamba memory.\n"
        "Use the provided MEMORY only when it is relevant.\n"
        "If the memory does not contain the answer, say: I do not know based on memory.\n"
        "If memories conflict, prefer active, recent, and specific memories.\n"
        "Do not invent personal facts.\n"
        f"{citation_rule}\n"
        "Answer naturally and concisely."
    )

    if not retrieved:
        user = f"Question: {question}\n\nMEMORY:\n(none)\n\nAnswer:"
        return system, user

    lines = []
    for rank, (score, details, item) in enumerate(retrieved, start=1):
        lines.append(
            f"[{rank}] id={item.id}; score={score:.3f}; updated={fmt_time(item.updated_at)}; "
            f"subject={item.subject}; property={item.property}; text={item.text}"
        )
    memory_block = "\n".join(lines)
    user = f"Question: {question}\n\nMEMORY:\n{memory_block}\n\nAnswer:"
    return system, user


def ask_with_memory(
    question: str,
    memory: MambaVectorMemory,
    ollama_model: str,
    topk: int,
    min_score: float,
    debug: bool,
    show_citations: bool,
) -> str:
    retrieved = memory.retrieve(question, k=topk, min_score=min_score)

    if debug:
        if retrieved:
            lines = []
            for score, details, item in retrieved:
                lines.append(
                    f"#{item.id} | score={score:.3f} | sem={details['semantic']:.3f} "
                    f"lex={details['lexical']:.3f} ent={details['entity']:.3f} "
                    f"prop={details['property']:.3f} | {item.text}"
                )
            box("DEBUG RETRIEVED MEMORY", "\n".join(lines), Ansi.MAGENTA)
        else:
            box("DEBUG RETRIEVED MEMORY", "No relevant memory found.", Ansi.YELLOW)

    system, user = build_prompt(question, retrieved, show_citations=show_citations)
    return ollama_chat(model=ollama_model, system=system, user=user) or "(no response)"


# =============================================================================
# Commands
# =============================================================================

@dataclass
class RuntimeConfig:
    debug: bool = False
    show_citations: bool = False


def print_help() -> None:
    body = (
        "Natural mode:\n"
        "  Lucas's secret code is 8392.                -> stored automatically\n"
        "  What is Lucas's secret code?                -> answered using memory\n"
        "\n"
        "Commands:\n"
        "  /store <text>          Force storing a memory\n"
        "  /ask <question>        Force asking a question\n"
        "  /search <query>        Show retrieval results without asking Ollama\n"
        "  /forget <query>        Safely forget the best matching memory only\n"
        "  /forget --all <query>  Forget all safe matches after typing YES\n"
        "  /memories              Show active memories\n"
        "  /memories all          Show active and inactive memories\n"
        "  /debug on | off        Show or hide retrieval details\n"
        "  /citations on | off    Show or hide memory IDs in final answers\n"
        "  /stats                 Show memory statistics\n"
        "  /save                  Save memory\n"
        "  /load                  Load memory\n"
        "  /reset                 Clear all memories\n"
        "  /exit                  Quit"
    )
    box("HELP", body, Ansi.CYAN)


def print_memories(memory: MambaVectorMemory, include_inactive: bool = False, limit: int = 80) -> None:
    items = memory.items if include_inactive else memory.active_items
    if not items:
        warn("No memory stored.")
        return

    lines = []
    for item in sorted(items, key=lambda x: x.updated_at, reverse=True)[:limit]:
        status = "active" if item.active else f"inactive ({item.deactivated_reason or 'no reason'})"
        lines.append(
            f"#{item.id} | {status} | subject={item.subject} | property={item.property} | "
            f"updated={fmt_time(item.updated_at)} | {item.text}"
        )
    box("MEMORIES", "\n".join(lines), Ansi.CYAN)


def print_store_results(results: List[Tuple[str, MemoryItem, List[MemoryItem]]]) -> None:
    if not results:
        warn("Nothing was stored.")
        return

    lines: List[str] = []
    for action, item, deactivated in results:
        if action == "duplicate_ignored":
            lines.append(f"Already remembered: {truncate(item.text, 130)}")
        elif deactivated:
            old_ids = ", ".join(f"#{old.id}" for old in deactivated)
            lines.append(f"Updated memory: {truncate(item.text, 130)}")
            lines.append(f"Deactivated older related fact(s): {old_ids}")
        else:
            lines.append(f"Saved: {truncate(item.text, 130)}")

    memory_print("\n".join(lines))


def print_search(memory: MambaVectorMemory, query: str, topk: int, min_score: float) -> None:
    results = memory.retrieve(query, k=topk, min_score=min_score)
    if not results:
        warn("No relevant memory found.")
        return
    lines = []
    for score, details, item in results:
        lines.append(
            f"#{item.id} | score={score:.3f} | sem={details['semantic']:.3f} "
            f"lex={details['lexical']:.3f} ent={details['entity']:.3f} "
            f"prop={details['property']:.3f} | {item.text}"
        )
    box("SEARCH RESULTS", "\n".join(lines), Ansi.MAGENTA)


def handle_forget(query: str, memory: MambaVectorMemory, forget_all: bool) -> None:
    query = clean_text(query)
    if not query:
        warn("Give a query to forget, for example: /forget Lucas secret code")
        return

    candidates = memory.search_forget_candidates(query, k=8 if forget_all else 5)
    if not candidates:
        warn("No safe matching memory found. Nothing was deleted.")
        return

    lines = []
    for score, details, item in candidates:
        lines.append(
            f"#{item.id} | score={score:.3f} | sem={details['semantic']:.3f} "
            f"lex={details['lexical']:.3f} ent={details['entity']:.3f} "
            f"prop={details['property']:.3f} | {item.text}"
        )
    box("FORGET CANDIDATES", "\n".join(lines), Ansi.YELLOW)

    if forget_all:
        confirm = input(color("Deactivate ALL these memories? Type YES to confirm: ", Ansi.RED)).strip()
        if confirm != "YES":
            warn("Cancelled. No memory was deactivated.")
            return
        ids = [item.id for _, _, item in candidates]
    else:
        best = candidates[0][2]
        confirm = input(color(f"Deactivate only best match #{best.id}? [y/N]: ", Ansi.RED)).strip().lower()
        if confirm not in {"y", "yes"}:
            warn("Cancelled. No memory was deactivated.")
            return
        ids = [best.id]

    changed = memory.deactivate_ids(ids, reason=f"forgotten by query: {query}")
    try:
        memory.save()
    except Exception:
        pass

    if changed:
        box("MEMORY", "Memory deactivated: " + ", ".join(f"#{item.id}" for item in changed), Ansi.GREEN)
    else:
        warn("Nothing was changed.")


def handle_command(
    line: str,
    memory: MambaVectorMemory,
    ollama_model: str,
    topk: int,
    min_score: float,
    runtime: RuntimeConfig,
) -> bool:
    lower = line.lower().strip()

    if lower in {"/exit", "/quit"}:
        return False

    if lower == "/help":
        print_help()
        return True

    if lower == "/debug on":
        runtime.debug = True
        system_print("Debug mode enabled. Retrieval details will be shown.")
        return True

    if lower == "/debug off":
        runtime.debug = False
        system_print("Debug mode disabled. Chat is now cleaner.")
        return True

    if lower == "/citations on":
        runtime.show_citations = True
        system_print("Citations enabled.")
        return True

    if lower == "/citations off":
        runtime.show_citations = False
        system_print("Citations disabled.")
        return True

    if lower == "/stats":
        box("STATS", "\n".join(f"{k}: {v}" for k, v in memory.stats().items()), Ansi.CYAN)
        return True

    if lower == "/memories":
        print_memories(memory, include_inactive=False)
        return True

    if lower == "/memories all":
        print_memories(memory, include_inactive=True)
        return True

    if lower == "/reindex":
        changed, created = memory.atomize_existing_memories()
        memory.save()
        memory_print(
            f"Reindexed broad memories.\n"
            f"Deactivated broad items: {changed}\n"
            f"Created atomic facts: {created}"
        )
        return True

    if lower == "/save":
        memory.save()
        system_print("Memory saved.")
        return True

    if lower == "/load":
        memory.load()
        system_print("Memory loaded.")
        return True

    if lower == "/reset":
        confirm = input(color("Clear ALL memories? Type YES to confirm: ", Ansi.RED)).strip()
        if confirm == "YES":
            memory.reset()
            try:
                memory.save()
            except Exception:
                pass
            system_print("Memory reset.")
        else:
            warn("Reset cancelled.")
        return True

    if line.startswith("/store "):
        text = line[len("/store ") :].strip()
        try:
            results = memory.store_text(text)
            memory.save()
            print_store_results(results)
        except Exception as exc:
            error(f"Could not store memory: {exc}")
        return True

    if line.startswith("/ask "):
        question = line[len("/ask ") :].strip()
        try:
            answer = ask_with_memory(
                question,
                memory,
                ollama_model,
                topk,
                min_score,
                debug=runtime.debug,
                show_citations=runtime.show_citations,
            )
            assistant_print(answer)
        except Exception as exc:
            error(f"Ollama error: {exc}")
            info(f"Make sure the model exists: ollama pull {ollama_model}")
        return True

    if line.startswith("/search "):
        query = line[len("/search ") :].strip()
        print_search(memory, query, topk=topk, min_score=min_score)
        return True

    if line.startswith("/forget "):
        raw = line[len("/forget ") :].strip()
        forget_all = False
        if raw.startswith("--all "):
            forget_all = True
            raw = raw[len("--all ") :].strip()
        handle_forget(raw, memory, forget_all=forget_all)
        return True

    warn("Unknown command. Type /help.")
    return True


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Mamba external memory for LLM agent")
    parser.add_argument("--mamba-model", default="state-spaces/mamba-130m-hf", help="Hugging Face Mamba model id")
    parser.add_argument("--ollama-model", default="gemma2:2b", help="Ollama model name, must be pulled locally")
    parser.add_argument("--topk", type=int, default=5, help="Number of memory snippets to retrieve")
    parser.add_argument("--min-score", type=float, default=0.12, help="Minimum hybrid retrieval score")
    parser.add_argument("--max-items", type=int, default=5000, help="Maximum number of total memory items")
    parser.add_argument(
        "--memory-file",
        default=str(base_dir / "mamba_memory_store.pt"),
        help="Local memory save file",
    )
    parser.add_argument("--debug", action="store_true", help="Show retrieval details")
    parser.add_argument("--citations", action="store_true", help="Show memory IDs in answers")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Torch device for Mamba",
    )
    args = parser.parse_args(argv)

    global USE_COLOR
    USE_COLOR = not args.no_color

    runtime = RuntimeConfig(debug=args.debug, show_citations=args.citations)

    memory = MambaVectorMemory(
        hf_model=args.mamba_model,
        device=args.device,
        memory_path=Path(args.memory_file),
        max_items=args.max_items,
    )

    box(
        "READY",
        "Natural chat mode is enabled.\n"
        "Write normally to chat or teach the memory.\n"
        "Use /help for commands and /debug on to inspect retrieval.",
        Ansi.GREEN,
    )

    while True:
        try:
            line = input(color("\nYou> ", Ansi.BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            try:
                memory.save()
                system_print("Memory saved. Bye.")
            except Exception:
                system_print("Bye.")
            return 0

        if not line:
            continue

        intent = classify_user_message(line)

        if intent == "exit":
            try:
                memory.save()
                system_print("Memory saved. Bye.")
            except Exception as exc:
                warn(f"Could not save memory: {exc}")
            return 0

        if intent == "command":
            keep_going = handle_command(
                line,
                memory,
                args.ollama_model,
                args.topk,
                args.min_score,
                runtime,
            )
            if not keep_going:
                try:
                    memory.save()
                    system_print("Memory saved. Bye.")
                except Exception as exc:
                    warn(f"Could not save memory: {exc}")
                return 0
            continue

        if intent == "store":
            try:
                results = memory.store_text(line)
                memory.save()
                print_store_results(results)
            except Exception as exc:
                error(f"Could not store memory: {exc}")
            continue

        if intent == "forget":
            query = re.sub(
                r"(?i)\b(forget|delete|remove|erase|do not remember|don't remember)\b",
                "",
                line,
            ).strip()
            handle_forget(query or line, memory, forget_all=False)
            continue

        # ask/chat
        try:
            answer = ask_with_memory(
                line,
                memory,
                args.ollama_model,
                args.topk,
                args.min_score,
                debug=runtime.debug,
                show_citations=runtime.show_citations,
            )
            assistant_print(answer)
        except Exception as exc:
            error(f"Ollama error: {exc}")
            info(f"Make sure the model exists: ollama pull {args.ollama_model}")


if __name__ == "__main__":
    raise SystemExit(main())



class MambaModel:
    def __init__(self, model_name: str = "llama3", max_capacity: int = 5000):
        self.model_name = model_name
        self.max_capacity = max_capacity
        self.memory = MambaVectorMemory(
            hf_model="state-spaces/mamba-130m-hf",
            device="cuda" if torch.cuda.is_available() else "cpu",
            memory_path=Path("memory_mamba.pt"),
            max_items=max_capacity
        )
        self.history = []
        self._set_system_prompt()

    def _set_system_prompt(self):
        system_prompt = (
            "You are a highly intelligent, concise, and helpful AI assistant. "
            "You always communicate in English. "
            "Your underlying architecture relies on a Mamba semantic vector memory."
        )
        self.history.append({"role": "system", "content": system_prompt})

    def get_active_tokens_count(self) -> int:
        return len(self.memory.active_items)

    def get_dropped_tokens_count(self) -> int:
        return len(self.memory.inactive_items)

    def get_memory_dump(self) -> str:
        active = self.memory.active_items
        lines = [f"[bold cyan]Architecture:[/bold cyan] [bold]Mamba (Semantic Vector Memory)[/bold]",
                 f"[bold yellow]Active Memories:[/bold yellow] {len(active)} / {self.max_capacity}",
                 f"[bold red]Inactive Memories:[/bold red] {len(self.memory.inactive_items)}",
                 ""]
        if active:
            lines.append("[bold green]Recent Active Memories:[/bold green]")
            for item in sorted(active, key=lambda x: x.updated_at, reverse=True)[:10]:
                lines.append(f"  • #{item.id} | Subj: {item.subject} | Prop: {item.property} | {item.text}")
        return "\n".join(lines)

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})

        intent = classify_user_message(message)
        if intent == "store":
            self.memory.store_text(message)
        elif intent == "forget":
            query = re.sub(r"(?i)\b(forget|delete|remove|erase|do not remember|don't remember)\b", "", message).strip()
            handle_forget(query or message, self.memory, forget_all=False)

    def add_assistant_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})

    def generate_response_stream(self):
        last_user_message = self.history[-1]['content']
        retrieved = self.memory.retrieve(last_user_message, k=5, min_score=0.12)
        
        system, user = build_prompt(last_user_message, retrieved, show_citations=False)
        
        inference_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        try:
            import ollama
            response = ollama.chat(
                model=self.model_name,
                messages=inference_messages,
                stream=True,
            )
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            yield f"\n[Error communicating with Ollama: {str(e)}]"

    def save_memory(self, filepath: str = "memory_mamba.pt"):
        self.memory.memory_path = Path(filepath)
        self.memory.save()
        import pickle
        history_path = filepath + "_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(self.history, f)

    def load_memory(self, filepath: str = "memory_mamba.pt") -> bool:
        self.memory.memory_path = Path(filepath)
        if not self.memory.memory_path.exists():
            return False
        try:
            self.memory.load()
            import pickle, os
            history_path = filepath + "_history.pkl"
            if os.path.exists(history_path):
                with open(history_path, 'rb') as f:
                    self.history = pickle.load(f)
            return True
        except Exception:
            return False

    def clear_memory(self):
        self.history = []
        self._set_system_prompt()
        self.memory.reset()
