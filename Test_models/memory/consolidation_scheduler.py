#!/usr/bin/env python3
"""Policy-based automatic consolidation for Titan agent memory.

This module adds a small scheduler layer around :class:`TitanAgentMemory`.
It does not replace the existing manual ``consolidate()`` method; it wraps it
with a realistic policy:

- consolidate only active memory by default;
- run at a configurable nightly hour;
- avoid multiple runs during the same night;
- write JSONL logs for traceability;
- keep a tiny state file with the last successful run;
- stay testable without Windows Task Scheduler or a background daemon.

The scheduler is intentionally explicit. A caller such as ``main.py``, an agent,
or a future Windows scheduled task can call ``run_if_due()`` on startup or before
shutdown. This gives us automatic nightly behavior without hiding side effects.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_LOG_DIR = ROOT / "logs"
DEFAULT_STATE_PATH = DEFAULT_LOG_DIR / "consolidation_state.json"
DEFAULT_LOG_PATH = DEFAULT_LOG_DIR / "consolidation_log.jsonl"


class ConsolidatableMemory(Protocol):
    """Small protocol satisfied by TitanAgentMemory and test doubles."""

    def consolidate(
        self,
        keep_existing_ltm: bool = False,
        max_items: int | None = None,
        steps: int = 3,
        include_inactive: bool = False,
    ) -> dict[str, Any]: ...

    def save(self, path: str | Path | None = None) -> None: ...

    def stats(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ConsolidationPolicy:
    """Configuration for scheduled Titan memory consolidation."""

    enabled: bool = True
    schedule_hour: int = 2
    schedule_minute: int = 0
    min_hours_between_runs: float = 20.0
    max_items: int | None = None
    steps: int = 3
    keep_existing_ltm: bool = False
    include_inactive: bool = False
    save_after: bool = True
    log_dir: Path = field(default_factory=lambda: DEFAULT_LOG_DIR)
    state_path: Path = field(default_factory=lambda: DEFAULT_STATE_PATH)
    log_path: Path = field(default_factory=lambda: DEFAULT_LOG_PATH)

    def __post_init__(self) -> None:
        if not 0 <= int(self.schedule_hour) <= 23:
            raise ValueError("schedule_hour must be in [0, 23].")
        if not 0 <= int(self.schedule_minute) <= 59:
            raise ValueError("schedule_minute must be in [0, 59].")
        if float(self.min_hours_between_runs) < 0:
            raise ValueError("min_hours_between_runs must be non-negative.")
        if int(self.steps) < 1:
            raise ValueError("steps must be >= 1.")
        if self.max_items is not None and int(self.max_items) < 1:
            raise ValueError("max_items must be >= 1 when provided.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["log_dir"] = str(self.log_dir)
        data["state_path"] = str(self.state_path)
        data["log_path"] = str(self.log_path)
        return data


@dataclass(frozen=True)
class ConsolidationRunResult:
    """Structured result for one scheduled/manual consolidation attempt."""

    ran: bool
    status: str
    reason: str
    message: str
    run_at: str
    policy: dict[str, Any]
    stats_before: dict[str, Any] = field(default_factory=dict)
    stats_after: dict[str, Any] = field(default_factory=dict)
    consolidation: dict[str, Any] = field(default_factory=dict)
    log_path: str | None = None
    state_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConsolidationScheduler:
    """Run Titan consolidation when the nightly policy says it is due."""

    def __init__(
        self,
        memory: ConsolidatableMemory,
        policy: ConsolidationPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.memory = memory
        self.policy = policy or ConsolidationPolicy()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def now(self) -> datetime:
        current = self.clock()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current

    def last_run_at(self) -> datetime | None:
        state = self._load_state()
        raw = state.get("last_run_at")
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(str(raw))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def should_run_now(self, current_time: datetime | None = None) -> bool:
        """Return true when the nightly consolidation window is due."""

        if not self.policy.enabled:
            return False
        current = current_time or self.now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        if current.hour != int(self.policy.schedule_hour):
            return False
        if current.minute < int(self.policy.schedule_minute):
            return False
        last = self.last_run_at()
        if last is None:
            return True
        elapsed_hours = (current - last).total_seconds() / 3600.0
        return elapsed_hours >= float(self.policy.min_hours_between_runs)

    def run_if_due(self, current_time: datetime | None = None) -> ConsolidationRunResult:
        """Run consolidation only when the policy window is due."""

        current = current_time or self.now()
        if not self.should_run_now(current):
            return self._skipped_result(
                current,
                reason="not_due",
                message="Automatic consolidation skipped: policy window is not due or it already ran recently.",
            )
        return self.run_once(reason="scheduled_nightly", current_time=current, force=False)

    def run_once(
        self,
        *,
        reason: str = "manual",
        current_time: datetime | None = None,
        force: bool = True,
    ) -> ConsolidationRunResult:
        """Run one consolidation pass and append a JSONL log entry."""

        current = current_time or self.now()
        if not self.policy.enabled and not force:
            return self._skipped_result(
                current,
                reason="disabled",
                message="Automatic consolidation skipped: policy is disabled.",
            )

        stats_before = self._safe_stats()
        try:
            consolidation = self.memory.consolidate(
                keep_existing_ltm=bool(self.policy.keep_existing_ltm),
                max_items=self.policy.max_items,
                steps=int(self.policy.steps),
                include_inactive=bool(self.policy.include_inactive),
            )
            if self.policy.save_after:
                self.memory.save()
            stats_after = self._safe_stats()
            result = ConsolidationRunResult(
                ran=True,
                status="ok",
                reason=reason,
                message=str(consolidation.get("message", "Titan memory consolidated.")),
                run_at=current.isoformat(),
                policy=self.policy.to_dict(),
                stats_before=stats_before,
                stats_after=stats_after,
                consolidation=dict(consolidation),
                log_path=str(self.policy.log_path),
                state_path=str(self.policy.state_path),
            )
            self._write_state(current, result)
            self._append_log(result)
            return result
        except Exception as exc:  # pragma: no cover - defensive path
            result = ConsolidationRunResult(
                ran=False,
                status="error",
                reason=reason,
                message="Titan memory consolidation failed.",
                run_at=current.isoformat(),
                policy=self.policy.to_dict(),
                stats_before=stats_before,
                error=str(exc),
                log_path=str(self.policy.log_path),
                state_path=str(self.policy.state_path),
            )
            self._append_log(result)
            return result

    def _skipped_result(self, current: datetime, *, reason: str, message: str) -> ConsolidationRunResult:
        return ConsolidationRunResult(
            ran=False,
            status="skipped",
            reason=reason,
            message=message,
            run_at=current.isoformat(),
            policy=self.policy.to_dict(),
            stats_before=self._safe_stats(),
            log_path=str(self.policy.log_path),
            state_path=str(self.policy.state_path),
        )

    def _safe_stats(self) -> dict[str, Any]:
        try:
            return dict(self.memory.stats())
        except Exception as exc:  # pragma: no cover - defensive path
            return {"stats_error": str(exc)}

    def _load_state(self) -> dict[str, Any]:
        try:
            if not self.policy.state_path.exists():
                return {}
            data = json.loads(self.policy.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_state(self, current: datetime, result: ConsolidationRunResult) -> None:
        self.policy.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_run_at": current.isoformat(),
            "last_status": result.status,
            "last_reason": result.reason,
            "last_consolidated": result.consolidation,
        }
        self.policy.state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_log(self, result: ConsolidationRunResult) -> None:
        self.policy.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.policy.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def run_scheduled_consolidation(
    memory: ConsolidatableMemory,
    policy: ConsolidationPolicy | None = None,
) -> ConsolidationRunResult:
    """Convenience function for future app/agent startup integration."""

    return ConsolidationScheduler(memory=memory, policy=policy).run_if_due()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Titan memory consolidation with a nightly policy.")
    parser.add_argument("--memory-path", default="agent_titan_memory.pt", help="Titan memory .pt file to consolidate.")
    parser.add_argument("--hour", type=int, default=2, help="Nightly schedule hour, 0-23. Default: 2.")
    parser.add_argument("--minute", type=int, default=0, help="Nightly schedule minute. Default: 0.")
    parser.add_argument("--steps", type=int, default=3, help="LTM replay steps. Default: 3.")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum active memories to replay.")
    parser.add_argument("--force", action="store_true", help="Run now even if the nightly window is not due.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    args = parser.parse_args(argv)

    from agents.titan_agent_memory import TitanAgentMemory

    memory = TitanAgentMemory(memory_path=args.memory_path)
    memory.load(args.memory_path)
    policy = ConsolidationPolicy(
        schedule_hour=args.hour,
        schedule_minute=args.minute,
        steps=args.steps,
        max_items=args.max_items,
    )
    scheduler = ConsolidationScheduler(memory=memory, policy=policy)
    result = scheduler.run_once(reason="manual_force", force=True) if args.force else scheduler.run_if_due()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.message)
        print(f"status={result.status}; ran={result.ran}; log={result.log_path}")
    return 0 if result.status in {"ok", "skipped"} else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
