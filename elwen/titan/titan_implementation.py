#!/usr/bin/env python3
"""
Titan External Memory for LLM Agent - v1

Titan-inspired prototype:
- facts are atomized into small memory units;
- each fact is encoded into key/value vectors;
- a neural MLP memory learns key -> value at test time;
- retrieval combines neural score + lexical/entity/property safeguards;
- Ollama/Gemma generates the final answer from retrieved facts.

This is not the full Google Titans architecture trained end-to-end.
"""

from __future__ import annotations

import argparse, hashlib, re, shutil, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


USE_COLOR = True


class Ansi:
    RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[36m"; GREEN="\033[32m"
    YELLOW="\033[33m"; RED="\033[31m"; BLUE="\033[34m"; MAGENTA="\033[35m"


def color(text: str, code: str="") -> str:
    return f"{code}{text}{Ansi.RESET}" if USE_COLOR and code else text


def terminal_width(default: int=100) -> int:
    try: return min(max(shutil.get_terminal_size((default, 20)).columns, 72), 120)
    except Exception: return default


def wrap_text(text: str, width: int) -> List[str]:
    if not text: return [""]
    out=[]
    for raw in str(text).splitlines():
        cur=""
        for w in raw.split(" "):
            if not cur: cur=w
            elif len(cur)+1+len(w)<=width: cur += " " + w
            else: out.append(cur); cur=w
        out.append(cur)
    return out or [""]


def box(title: str, body: str="", code: str="") -> None:
    width=terminal_width(); inner=width-4; label=f" {title} "
    print(color("┌"+label+"─"*max(0, inner-len(label)+2)+"┐", code))
    for line in wrap_text(body, inner):
        print(color("│ ", code)+line.ljust(inner)+color(" │", code))
    print(color("└"+"─"*(width-2)+"┘", code))


def warn(text: str) -> None: print(color(f"⚠ {text}", Ansi.YELLOW))
def error(text: str) -> None: print(color(f"✗ {text}", Ansi.RED))
def info(text: str) -> None: print(color(f"• {text}", Ansi.BLUE))
def assistant_print(text: str) -> None: print(); box("ASSISTANT", text, Ansi.CYAN)
def memory_print(text: str) -> None: print(); box("TITAN MEMORY", text, Ansi.GREEN)
def system_print(text: str) -> None: print(); box("SYSTEM", text, Ansi.BLUE)


def truncate(text: str, n: int=120) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text)<=n else text[:n-3]+"..."


STOPWORDS = {
    "a","an","the","and","or","of","to","in","on","for","with","from","is","are","am",
    "was","were","be","been","being","as","i","me","my","mine","you","your","he","she",
    "it","they","we","his","her","their","this","that","these","those","do","does","did",
    "what","where","when","who","why","how","which","tell","give","show","more","about",
    "please","can","could","would","should","now","currently","actually","learn","remember","that"
}
QUESTION_STARTERS = ("what","where","when","who","why","how","which","do","does","did","can","could","would","should","is","are","am","tell me","explain","give me","show me")
MEMORY_MARKERS = ("my name is","i am","i'm","i live","i study","i work","i like","i love","my favorite","my favourite","i prefer","i want","i need","i was born","remember that","note that","save this","learn that","lives in","likes","favorite color","favourite color","favorite programming language","programming language","secret code","temporary access code","student id","office room","portfolio password","research group","operating system","was born","is from","works on","studies","is currently","wants to")
UPDATE_MARKERS = ("actually","now","changed","instead","no longer","correction","update:","is now","currently")
FORGET_MARKERS = ("forget","delete","remove","do not remember","don't remember","erase")
BROAD_MEMORY_PATTERNS = ("what do you know about me","what do you remember about me","summarize my profile","summarise my profile","who am i")

PROPERTY_PATTERNS: List[Tuple[str,str]] = [
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
    ("study", r"\bstud(?:y|ies|ying)\b|\bstudent\b|\bschool\b|\buniversity\b"),
    ("project", r"\bproject\b|\bworking on\b|\bworks on\b|\bbuild(?:ing)?\b|\bcreate\b"),
    ("work", r"\bwork[s]?\b|\bjob\b|\bengineer\b"),
    ("current_location", r"\bcurrently\b|\bcurrent location\b|\bwhere .* currently\b"),
    ("location", r"\blive[s]?\b|\blives in\b|\bwhere .* live\b|\bfrom\b"),
    ("age", r"\bage\b|\byears old\b"),
    ("name", r"\bname\b"),
]
SINGLE_VALUE_PROPERTIES = {"secret_code","student_id","office_room","password","research_group","favorite_color","favorite_food","favorite_language","favorite_os","favorite_opening","location","current_location","age","name"}


def now() -> float: return time.time()
def fmt_time(ts: float) -> str: return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
def clean_text(text: str) -> str: return re.sub(r"\s+", " ", str(text)).strip()
def normalize_text(text: str) -> str: return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
def words(text: str) -> List[str]: return re.findall(r"[a-zA-Z0-9_À-ÖØ-öø-ÿ']+", text.lower())
def keywords(text: str) -> set[str]: return {w for w in words(text) if len(w)>=3 and w not in STOPWORDS}
def l2_normalize(x: torch.Tensor, eps: float=1e-12) -> torch.Tensor: return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def lexical_overlap(query: str, memory_text: str) -> float:
    q=keywords(query)
    return 0.0 if not q else len(q & keywords(memory_text)) / len(q)


def extract_entities(text: str) -> set[str]:
    found=set()
    for m in re.finditer(r"\b[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+(?:\s+[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+){0,2}\b", text):
        for token in words(m.group(0)):
            if len(token)>=3 and token not in STOPWORDS: found.add(token)
    for m in re.finditer(r"\b([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+)'s\b", text):
        token=m.group(1).lower()
        if len(token)>=3: found.add(token)
    return found


def entity_match_score(query: str, memory_text: str) -> float:
    q=extract_entities(query)
    return 0.0 if not q else len(q & extract_entities(memory_text)) / len(q)


def subject_entity_match_score(q_entities: set[str], subject: Optional[str]) -> float:
    """
    Compare the query entities with the stored canonical subject.
    Prevents retrieving a memory for another person with the same last name.
    """
    if not q_entities or not subject:
        return 0.0
    subject_tokens=set(subject.split())
    return 0.0 if not subject_tokens else len(q_entities & subject_tokens) / len(q_entities)


def property_key(text: str) -> Optional[str]:
    lower=text.lower()
    for key, pattern in PROPERTY_PATTERNS:
        if re.search(pattern, lower): return key
    return None


def property_match_score(query: str, memory_text: str) -> float:
    q=property_key(query)
    return 0.0 if not q else (1.0 if property_key(memory_text)==q else 0.0)


def is_broad_memory_question(question: str) -> bool:
    return any(p in question.lower().strip() for p in BROAD_MEMORY_PATTERNS)


def normalize_statement_for_storage(text: str) -> str:
    s=clean_text(text)
    s=re.sub(r"(?i)^(learn|remember|note)\s+that\s+", "", s).strip()
    s=re.sub(r"(?i)^save\s+this\s*:\s*", "", s).strip()
    s=re.sub(r"(?i)^(actually|correction|update)\s*[:,]?\s*", "", s).strip()
    return s


def extract_subject(text: str) -> Optional[str]:
    s=normalize_statement_for_storage(text); low=s.lower()
    if low.startswith(("my ","i ","i'm","i am")): return "user"
    # Multi-word possessive: "Sarah Nguyen's ...", "PriyaAlpha Meyer's ..."
    m=re.match(r"^([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+(?:\s+[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+){0,2})'s\b", s)
    if m: return " ".join(words(m.group(1)))
    m=re.match(r"^([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+(?:\s+[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+){0,2})\s+(?:is|lives|likes|works|studies|has|wants|enjoys|speaks)\b", s)
    return " ".join(words(m.group(1))) if m else None


def has_update_marker(text: str) -> bool: return any(m in text.lower() for m in UPDATE_MARKERS)


def looks_like_memory_statement(text: str) -> bool:
    s=clean_text(text); low=s.lower()
    if not s or s.endswith("?"): return False
    if any(m in low for m in MEMORY_MARKERS): return True
    if re.match(r"^[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+(?:\s+[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+){0,2}\s+(?:is|lives|likes|works|studies|has|wants|enjoys|speaks)\b", s): return True
    if re.match(r"^[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ']+'s\s+.+\s+(?:is|are|was|were)\b", s): return True
    return False


def subject_display(subject: Optional[str]) -> Optional[str]:
    if not subject: return None
    return "I" if subject=="user" else " ".join(p.capitalize() for p in subject.split())


def split_sentences(text: str) -> List[str]:
    cleaned=normalize_statement_for_storage(text).replace(";", ".")
    chunks=[c.strip() for c in re.split(r"(?<=[.!?])\s+", cleaned) if c.strip()]
    return chunks or ([cleaned] if cleaned else [])


def split_memory_units(text: str) -> List[str]:
    units=[]; current_display=None
    for raw in split_sentences(text):
        sentence=raw.strip().rstrip(".")
        if not sentence: continue
        if current_display:
            sentence=re.sub(r"(?i)^(he|she|they)\s+", f"{current_display} ", sentence)
            sentence=re.sub(r"(?i)^his\s+", f"{current_display}'s ", sentence)
            sentence=re.sub(r"(?i)^her\s+", f"{current_display}'s ", sentence)
            sentence=re.sub(r"(?i)^their\s+", f"{current_display}'s ", sentence)
        subj=extract_subject(sentence)
        if subj: current_display=subject_display(subj)
        display=current_display or subject_display(extract_subject(sentence))
        if display:
            m=re.match(rf"^({re.escape(display)})\s+is\s+(.+?)\s+from\s+(.+)$", sentence, flags=re.I)
            if m:
                desc=m.group(2).strip(" ,"); place=m.group(3).strip(" ,")
                if desc: units.append(f"{display} is {desc}.")
                if place: units.append(f"{display} is from {place}.")
                continue
            m=re.match(rf"^({re.escape(display)})\s+stud(?:ies|y|ying)\s+(.+?)\s+and\s+works?\s+on\s+(.+)$", sentence, flags=re.I)
            if m:
                study=m.group(2).strip(" ,"); project=m.group(3).strip(" ,")
                if study: units.append(f"{display} studies {study}.")
                if project: units.append(f"{display} works on {project}.")
                continue
        final=sentence.strip()
        if final:
            units.append(final if final.endswith(".") else final+".")
    seen=set(); out=[]
    for u in units:
        key=normalize_text(u)
        if key not in seen:
            seen.add(key); out.append(u)
    return out


def classify_user_message(message: str) -> str:
    text=clean_text(message); low=text.lower()
    if low in {"/exit","exit","quit","/quit"}: return "exit"
    if low.startswith("/"): return "command"
    if any(m in low for m in FORGET_MARKERS): return "forget"
    if low.endswith("?") or low.startswith(QUESTION_STARTERS): return "ask"
    return "store" if looks_like_memory_statement(normalize_statement_for_storage(text)) else "chat"


def hashed_text_vector(text: str, dim: int=128) -> torch.Tensor:
    vec=torch.zeros(dim, dtype=torch.float32)
    toks=keywords(text) or set(words(text))
    for tok in toks:
        d=hashlib.sha256(tok.encode("utf-8")).digest()
        for off in range(0, 12, 2):
            idx=int.from_bytes(d[off:off+2], "little") % dim
            vec[idx] += 1.0 if d[off] % 2 == 0 else -1.0
    return l2_normalize(vec)


class TitanMemoryNetwork(nn.Module):
    def __init__(self, d_model: int=128, hidden_dim: int=256):
        super().__init__()
        self.d_model=d_model
        self.proj_k=nn.Linear(d_model, d_model, bias=False)
        self.proj_v=nn.Linear(d_model, d_model, bias=False)
        self.proj_q=nn.Linear(d_model, d_model, bias=False)
        self.memory_mlp=nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model))
        self._init_eye(self.proj_k); self._init_eye(self.proj_v); self._init_eye(self.proj_q)
        for p in list(self.proj_k.parameters())+list(self.proj_v.parameters())+list(self.proj_q.parameters()):
            p.requires_grad=False

    @staticmethod
    def _init_eye(layer: nn.Linear) -> None:
        with torch.no_grad():
            layer.weight.zero_(); n=min(layer.weight.shape); layer.weight[:n,:n]=torch.eye(n)

    def key(self, x: torch.Tensor) -> torch.Tensor: return l2_normalize(self.proj_k(x))
    def value(self, x: torch.Tensor) -> torch.Tensor: return l2_normalize(self.proj_v(x))
    def query(self, x: torch.Tensor) -> torch.Tensor: return l2_normalize(self.proj_q(x))
    def retrieve_value(self, q: torch.Tensor) -> torch.Tensor: return l2_normalize(self.memory_mlp(q))
    def associative_loss(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor: return F.mse_loss(self.memory_mlp(k), v)


@dataclass
class MemoryItem:
    id: int
    text: str
    key: torch.Tensor
    value: torch.Tensor
    subject: Optional[str]=None
    property: Optional[str]=None
    active: bool=True
    surprise: float=0.0
    created_at: float=field(default_factory=now)
    updated_at: float=field(default_factory=now)
    access_count: int=0
    last_accessed_at: Optional[float]=None
    deactivated_reason: Optional[str]=None

    def to_serializable(self) -> Dict[str, object]:
        return self.__dict__.copy()

    @staticmethod
    def from_serializable(data: Dict[str, object]) -> "MemoryItem":
        return MemoryItem(
            id=int(data["id"]), text=str(data["text"]),
            key=data["key"].float().cpu(), value=data["value"].float().cpu(), # type: ignore[union-attr]
            subject=data.get("subject"), property=data.get("property"), active=bool(data.get("active", True)),
            surprise=float(data.get("surprise", 0.0)), created_at=float(data.get("created_at", now())),
            updated_at=float(data.get("updated_at", now())), access_count=int(data.get("access_count", 0)),
            last_accessed_at=data.get("last_accessed_at"), deactivated_reason=data.get("deactivated_reason")
        )


class TitanExternalMemory:
    def __init__(self, memory_path: Optional[Path]=None, d_model: int=128, hidden_dim: int=256,
                 max_items: int=5000, device: str="cpu", learning_rate: float=0.01,
                 weight_decay: float=0.001, replay_items: int=8) -> None:
        self.memory_path=memory_path; self.d_model=d_model; self.hidden_dim=hidden_dim
        self.max_items=max_items; self.device=device; self.learning_rate=learning_rate
        self.weight_decay=weight_decay; self.replay_items=replay_items
        info(f"Loading Titan neural memory on {device}.")
        self.network=TitanMemoryNetwork(d_model, hidden_dim).to(device)
        self.optimizer=torch.optim.AdamW(self.network.memory_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self._items: List[MemoryItem]=[]; self._next_id=1
        if self.memory_path and self.memory_path.exists():
            try: self.load(self.memory_path); memory_print(f"Loaded {len(self.active_items)} active memory item(s).")
            except Exception as exc: warn(f"Could not load existing memory file: {exc}")

    @property
    def items(self) -> List[MemoryItem]: return self._items
    @property
    def active_items(self) -> List[MemoryItem]: return [i for i in self._items if i.active]
    @property
    def inactive_items(self) -> List[MemoryItem]: return [i for i in self._items if not i.active]

    def encode(self, text: str) -> torch.Tensor: return hashed_text_vector(text, self.d_model).to(self.device)

    @torch.no_grad()
    def _make_key_value(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        x=self.encode(text).unsqueeze(0)
        return self.network.key(x)[0].detach().cpu(), self.network.value(x)[0].detach().cpu()

    def _surprise(self, key: torch.Tensor, value: torch.Tensor) -> float:
        with torch.no_grad():
            pred=self.network.retrieve_value(key.to(self.device).unsqueeze(0))
            return float(F.mse_loss(pred, value.to(self.device).unsqueeze(0)).item())

    def _train_association(self, key: torch.Tensor, value: torch.Tensor, surprise: float, base_steps: int=12) -> None:
        steps=base_steps + min(28, int(surprise*400))
        pairs=[(key,value)] + [(i.key,i.value) for i in sorted(self.active_items, key=lambda i: i.updated_at, reverse=True)[:self.replay_items]]
        self.network.train()
        for _ in range(max(1, steps)):
            self.optimizer.zero_grad(set_to_none=True)
            losses=[self.network.associative_loss(k.to(self.device).unsqueeze(0), v.to(self.device).unsqueeze(0)) for k,v in pairs]
            loss=torch.stack(losses).mean(); loss.backward()
            nn.utils.clip_grad_norm_(self.network.memory_mlp.parameters(), max_norm=1.0)
            self.optimizer.step()

    def store(self, text: str) -> Tuple[str, MemoryItem, List[MemoryItem]]:
        raw=clean_text(text); update_intent=has_update_marker(raw); text=normalize_statement_for_storage(raw)
        if not text: raise ValueError("Cannot store an empty memory.")
        norm=normalize_text(text)
        for item in self.active_items:
            if normalize_text(item.text)==norm:
                item.updated_at=now(); return "duplicate_ignored", item, []
        subject=self._resolve_subject(extract_subject(text)); prop=property_key(text)
        key,value=self._make_key_value(text); surprise=self._surprise(key,value)
        deactivated=[]
        if subject and prop and prop in SINGLE_VALUE_PROPERTIES:
            for item in self.active_items:
                if item.subject==subject and item.property==prop:
                    item.active=False; item.updated_at=now()
                    item.deactivated_reason = ("explicitly updated" if update_intent else "replaced") + f" by newer memory about {subject}/{prop}"
                    deactivated.append(item)
        self._train_association(key, value, surprise)
        item=MemoryItem(self._next_id, text, key, value, subject=subject, property=prop, surprise=surprise)
        self._next_id+=1; self._items.append(item); self._enforce_capacity()
        return "stored", item, deactivated

    def store_text(self, text: str) -> List[Tuple[str, MemoryItem, List[MemoryItem]]]:
        return [self.store(unit) for unit in split_memory_units(text)]

    def atomize_existing_memories(self) -> Tuple[int,int]:
        changed=created=0
        for item in list(self.active_items):
            units=split_memory_units(item.text)
            if len(units)<=1: continue
            item.active=False; item.deactivated_reason="reindexed into atomic facts"; item.updated_at=now(); changed+=1
            for u in units:
                action,_,_=self.store(u)
                if action=="stored": created+=1
        return changed, created

    def retrieve(self, query: str, k: int=5, min_score: float=0.12) -> List[Tuple[float,Dict[str,float],MemoryItem]]:
        active=self.active_items
        if not active: return []
        if is_broad_memory_question(query):
            return [(1.0, {"neural":1.0,"key":0.0,"lexical":0.0,"entity":0.0,"property":0.0,"recency":0.0}, i) for i in sorted(active, key=lambda i: i.updated_at, reverse=True)[:max(k,8)]]
        qkey,_=self._make_key_value(query); qents=extract_entities(query); qprop=property_key(query)
        with torch.no_grad():
            nread=self.network.retrieve_value(qkey.to(self.device).unsqueeze(0))[0].detach().cpu()
        scored=[]; t=now()
        for item in active:
            key_score=float(torch.dot(l2_normalize(qkey), l2_normalize(item.key)).item())
            neural=float(torch.dot(l2_normalize(nread), l2_normalize(item.value)).item())
            lex=lexical_overlap(query, item.text)
            ent=max(entity_match_score(query, item.text), subject_entity_match_score(qents, item.subject))
            prop=1.0 if qprop and item.property==qprop else property_match_score(query, item.text)
            rec=1.0/(1.0+max(0.0,(t-item.updated_at)/3600.0))
            ent_pen=-0.25 if qents and ent==0.0 else 0.0
            prop_bonus=0.12 if qprop and prop>0 else 0.0
            hybrid=0.24*neural + 0.20*key_score + 0.25*lex + 0.22*ent + 0.09*prop + 0.03*rec + ent_pen + prop_bonus
            details={"neural":neural,"key":key_score,"lexical":lex,"entity":ent,"property":prop,"recency":rec}
            if qents and ent==0.0 and lex<0.34 and prop==0.0: continue
            # For precise single-value properties, partial identity matches are unsafe.
            # Example: do not retrieve another person's secret code just because they
            # share the same last name.
            if qents and qprop in SINGLE_VALUE_PROPERTIES and prop>0 and ent<1.0: continue
            if hybrid>=min_score: scored.append((hybrid, details, item))
        scored.sort(key=lambda row: row[0], reverse=True); results=scored[:max(1,min(k,len(scored)))]
        for _,_,item in results:
            item.access_count+=1; item.last_accessed_at=now()
        return results

    def search_forget_candidates(self, query: str, k: int=5) -> List[Tuple[float,Dict[str,float],MemoryItem]]:
        candidates=self.retrieve(query, k=max(k,8), min_score=-1.0); safe=[]
        for score,details,item in candidates:
            if details["lexical"]>=0.34 or (details["entity"]>0 and details["property"]>0) or details["neural"]>=0.72 or details["key"]>=0.72:
                safe.append((score,details,item))
        safe.sort(key=lambda r: r[0], reverse=True); return safe[:k]

    def deactivate_ids(self, ids: Iterable[int], reason: str="forgotten") -> List[MemoryItem]:
        ids=set(ids); out=[]
        for item in self._items:
            if item.id in ids and item.active:
                item.active=False; item.deactivated_reason=reason; item.updated_at=now(); out.append(item)
        return out

    def reset(self) -> None:
        self._items.clear(); self._next_id=1
        self.network=TitanMemoryNetwork(self.d_model, self.hidden_dim).to(self.device)
        self.optimizer=torch.optim.AdamW(self.network.memory_mlp.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def stats(self) -> Dict[str,object]:
        return {"active_items":len(self.active_items),"inactive_items":len(self.inactive_items),"total_items":len(self.items),"d_model":self.d_model,"hidden_dim":self.hidden_dim,"device":self.device,"memory_path":str(self.memory_path) if self.memory_path else None}

    def save(self, path: Optional[Path]=None) -> None:
        target=path or self.memory_path
        if target is None: raise ValueError("No memory path configured.")
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version":1,"next_id":self._next_id,"d_model":self.d_model,"hidden_dim":self.hidden_dim,"items":[i.to_serializable() for i in self._items],"network_state":self.network.state_dict(),"optimizer_state":self.optimizer.state_dict()}, target)

    def load(self, path: Optional[Path]=None) -> None:
        target=path or self.memory_path
        if target is None: raise ValueError("No memory path configured.")
        payload=torch.load(target, map_location="cpu")
        self._items=[MemoryItem.from_serializable(x) for x in payload.get("items", [])]
        self._next_id=int(payload.get("next_id", len(self._items)+1))
        self.network.load_state_dict(payload["network_state"]); self.network.to(self.device)
        if payload.get("optimizer_state"): self.optimizer.load_state_dict(payload["optimizer_state"])

    def _resolve_subject(self, subject: Optional[str]) -> Optional[str]:
        if not subject or subject=="user": return subject
        subjects=sorted({i.subject for i in self.active_items if i.subject})
        if subject in subjects: return subject
        first=subject.split()[0]; matches=[s for s in subjects if s and s.split()[0]==first]
        return matches[0] if len(matches)==1 else subject

    def _enforce_capacity(self) -> None:
        if len(self._items)<=self.max_items: return
        inactive=sorted(self.inactive_items, key=lambda i:i.updated_at)
        remove={i.id for i in inactive[:max(0,len(self._items)-self.max_items)]}
        self._items=[i for i in self._items if i.id not in remove]
        if len(self._items)>self.max_items:
            self._items.sort(key=lambda i:(i.active,i.updated_at), reverse=True)
            self._items=self._items[:self.max_items]


def ollama_chat(model: str, system: str, user: str) -> str:
    try:
        import ollama  # type: ignore
    except Exception as exc:
        raise RuntimeError("Failed to import `ollama`. Install with `pip install ollama` and ensure Ollama is running.") from exc
    resp=ollama.chat(model=model, messages=[{"role":"system","content":system},{"role":"user","content":user}])
    msg=resp.get("message", {}) if isinstance(resp, dict) else getattr(resp, "message", {})
    content=msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    return (content or "").strip()


def build_prompt(question: str, retrieved: List[Tuple[float,Dict[str,float],MemoryItem]], show_citations: bool=False) -> Tuple[str,str]:
    cite = "At the end, cite the memory id like [memory: id]." if show_citations else "Do not mention memory IDs or retrieval scores."
    system=("You are a helpful assistant connected to an external Titan-inspired neural memory.\n"
            "Use the provided MEMORY only when relevant.\n"
            "If the memory does not contain the answer, say: I do not know based on memory.\n"
            "If memories conflict, prefer active, recent, and specific memories.\n"
            "Do not invent personal facts.\n"+cite+"\nAnswer naturally and concisely.")
    if not retrieved: return system, f"Question: {question}\n\nMEMORY:\n(none)\n\nAnswer:"
    lines=[f"[{rank}] id={item.id}; score={score:.3f}; subject={item.subject}; property={item.property}; text={item.text}" for rank,(score,_,item) in enumerate(retrieved,1)]
    return system, f"Question: {question}\n\nMEMORY:\n"+"\n".join(lines)+"\n\nAnswer:"


def ask_with_memory(question: str, memory: TitanExternalMemory, ollama_model: str, topk: int, min_score: float, debug: bool, show_citations: bool) -> str:
    retrieved=memory.retrieve(question, k=topk, min_score=min_score)
    if debug:
        if retrieved:
            lines=[f"#{i.id} | score={s:.3f} | neural={d['neural']:.3f} key={d['key']:.3f} lex={d['lexical']:.3f} ent={d['entity']:.3f} prop={d['property']:.3f} | {i.text}" for s,d,i in retrieved]
            box("DEBUG RETRIEVED TITAN MEMORY", "\n".join(lines), Ansi.MAGENTA)
        else:
            box("DEBUG RETRIEVED TITAN MEMORY", "No relevant memory found.", Ansi.YELLOW)
    system,user=build_prompt(question, retrieved, show_citations)
    return ollama_chat(ollama_model, system, user) or "(no response)"


@dataclass
class RuntimeConfig:
    debug: bool=False
    show_citations: bool=False


def print_help() -> None:
    body=("Natural mode:\n"
          "  Lucas's secret code is 8392.                -> stored automatically\n"
          "  What is Lucas's secret code?                -> answered using Titan memory\n\n"
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
          "  /save /load /reset /exit")
    box("HELP", body, Ansi.CYAN)


def print_memories(memory: TitanExternalMemory, include_inactive: bool=False, limit: int=80) -> None:
    items=memory.items if include_inactive else memory.active_items
    if not items: warn("No memory stored."); return
    lines=[]
    for item in sorted(items, key=lambda x:x.updated_at, reverse=True)[:limit]:
        status="active" if item.active else f"inactive ({item.deactivated_reason or 'no reason'})"
        lines.append(f"#{item.id} | {status} | subject={item.subject} | property={item.property} | surprise={item.surprise:.4f} | {item.text}")
    box("TITAN MEMORIES", "\n".join(lines), Ansi.CYAN)


def print_store_results(results: List[Tuple[str,MemoryItem,List[MemoryItem]]]) -> None:
    if not results: warn("Nothing was stored."); return
    lines=[]
    for action,item,deactivated in results:
        if action=="duplicate_ignored": lines.append(f"Already remembered: {truncate(item.text)}")
        elif deactivated:
            lines.append(f"Updated memory: {truncate(item.text)}")
            lines.append("Deactivated older related fact(s): "+", ".join(f"#{x.id}" for x in deactivated))
        else: lines.append(f"Saved: {truncate(item.text)} | surprise={item.surprise:.4f}")
    memory_print("\n".join(lines))


def print_search(memory: TitanExternalMemory, query: str, topk: int, min_score: float) -> None:
    res=memory.retrieve(query, k=topk, min_score=min_score)
    if not res: warn("No relevant memory found."); return
    lines=[f"#{i.id} | score={s:.3f} | neural={d['neural']:.3f} key={d['key']:.3f} lex={d['lexical']:.3f} ent={d['entity']:.3f} prop={d['property']:.3f} | {i.text}" for s,d,i in res]
    box("TITAN SEARCH RESULTS", "\n".join(lines), Ansi.MAGENTA)


def handle_forget(query: str, memory: TitanExternalMemory, forget_all: bool) -> None:
    query=clean_text(query)
    if not query: warn("Give a query to forget, for example: /forget Lucas secret code"); return
    candidates=memory.search_forget_candidates(query, k=8 if forget_all else 5)
    if not candidates: warn("No safe matching memory found. Nothing was deleted."); return
    lines=[f"#{i.id} | score={s:.3f} | neural={d['neural']:.3f} key={d['key']:.3f} lex={d['lexical']:.3f} ent={d['entity']:.3f} prop={d['property']:.3f} | {i.text}" for s,d,i in candidates]
    box("FORGET CANDIDATES", "\n".join(lines), Ansi.YELLOW)
    if forget_all:
        if input(color("Deactivate ALL these memories? Type YES to confirm: ", Ansi.RED)).strip()!="YES":
            warn("Cancelled. No memory was deactivated."); return
        ids=[i.id for _,_,i in candidates]
    else:
        best=candidates[0][2]
        if input(color(f"Deactivate only best match #{best.id}? [y/N]: ", Ansi.RED)).strip().lower() not in {"y","yes"}:
            warn("Cancelled. No memory was deactivated."); return
        ids=[best.id]
    changed=memory.deactivate_ids(ids, reason=f"forgotten by query: {query}")
    try: memory.save()
    except Exception: pass
    memory_print("Memory deactivated: "+", ".join(f"#{i.id}" for i in changed)) if changed else warn("Nothing was changed.")


def handle_command(line: str, memory: TitanExternalMemory, ollama_model: str, topk: int, min_score: float, runtime: RuntimeConfig) -> bool:
    lower=line.lower().strip()
    if lower in {"/exit","/quit"}: return False
    if lower=="/help": print_help(); return True
    if lower=="/debug on": runtime.debug=True; system_print("Debug mode enabled."); return True
    if lower=="/debug off": runtime.debug=False; system_print("Debug mode disabled."); return True
    if lower=="/citations on": runtime.show_citations=True; system_print("Citations enabled."); return True
    if lower=="/citations off": runtime.show_citations=False; system_print("Citations disabled."); return True
    if lower=="/stats": box("STATS", "\n".join(f"{k}: {v}" for k,v in memory.stats().items()), Ansi.CYAN); return True
    if lower=="/memories": print_memories(memory, False); return True
    if lower=="/memories all": print_memories(memory, True); return True
    if lower=="/reindex":
        changed,created=memory.atomize_existing_memories(); memory.save()
        memory_print(f"Reindexed broad memories.\nDeactivated broad items: {changed}\nCreated atomic facts: {created}"); return True
    if lower=="/save": memory.save(); system_print("Titan memory saved."); return True
    if lower=="/load": memory.load(); system_print("Titan memory loaded."); return True
    if lower=="/reset":
        if input(color("Clear ALL Titan memories? Type YES to confirm: ", Ansi.RED)).strip()=="YES":
            memory.reset(); memory.save(); system_print("Titan memory reset.")
        else: warn("Reset cancelled.")
        return True
    if line.startswith("/store "):
        try: res=memory.store_text(line[len("/store "):].strip()); memory.save(); print_store_results(res)
        except Exception as exc: error(f"Could not store memory: {exc}")
        return True
    if line.startswith("/ask "):
        try: assistant_print(ask_with_memory(line[len("/ask "):].strip(), memory, ollama_model, topk, min_score, runtime.debug, runtime.show_citations))
        except Exception as exc: error(f"Ollama error: {exc}"); info(f"Make sure the model exists: ollama pull {ollama_model}")
        return True
    if line.startswith("/search "): print_search(memory, line[len("/search "):].strip(), topk, min_score); return True
    if line.startswith("/forget "):
        raw=line[len("/forget "):].strip(); forget_all=False
        if raw.startswith("--all "): forget_all=True; raw=raw[len("--all "):].strip()
        handle_forget(raw, memory, forget_all); return True
    warn("Unknown command. Type /help."); return True


def main(argv: Optional[List[str]]=None) -> int:
    base_dir=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description="Titan external memory for LLM agent")
    parser.add_argument("--ollama-model", default="gemma2:2b")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.12)
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--memory-file", default=str(base_dir/"titan_memory_store.pt"))
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--citations", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu","cuda"])
    args=parser.parse_args(argv)

    global USE_COLOR; USE_COLOR=not args.no_color
    runtime=RuntimeConfig(debug=args.debug, show_citations=args.citations)
    memory=TitanExternalMemory(memory_path=Path(args.memory_file), d_model=args.d_model, hidden_dim=args.hidden_dim, max_items=args.max_items, device=args.device)

    box("READY", "Titan natural chat mode is enabled.\nWrite normally to chat or teach the memory.\nUse /help for commands and /debug on to inspect retrieval.", Ansi.GREEN)

    while True:
        try: line=input(color("\nYou> ", Ansi.BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            try: memory.save(); system_print("Titan memory saved. Bye.")
            except Exception: system_print("Bye.")
            return 0
        if not line: continue
        intent=classify_user_message(line)
        if intent=="exit":
            try: memory.save(); system_print("Titan memory saved. Bye.")
            except Exception as exc: warn(f"Could not save memory: {exc}")
            return 0
        if intent=="command":
            if not handle_command(line, memory, args.ollama_model, args.topk, args.min_score, runtime):
                try: memory.save(); system_print("Titan memory saved. Bye.")
                except Exception as exc: warn(f"Could not save memory: {exc}")
                return 0
            continue
        if intent=="store":
            try: res=memory.store_text(line); memory.save(); print_store_results(res)
            except Exception as exc: error(f"Could not store memory: {exc}")
            continue
        if intent=="forget":
            query=re.sub(r"(?i)\b(forget|delete|remove|erase|do not remember|don't remember)\b","",line).strip()
            handle_forget(query or line, memory, False); continue
        try: assistant_print(ask_with_memory(line, memory, args.ollama_model, args.topk, args.min_score, runtime.debug, runtime.show_citations))
        except Exception as exc: error(f"Ollama error: {exc}"); info(f"Make sure the model exists: ollama pull {args.ollama_model}")


if __name__ == "__main__":
    raise SystemExit(main())
