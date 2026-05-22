"""
Security Audit Logger — structured event logging for the AI CFO system.

All security-relevant events (auth failures, injection attempts, HITL decisions,
validation failures) are written here in addition to the pipeline audit_log.

Events are appended to a JSONL file alongside the DB for forensic review.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

_DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "security_audit.jsonl"
)


class EventSeverity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    AUTH_FAILURE         = "auth_failure"
    PROMPT_INJECTION     = "prompt_injection"
    INPUT_VALIDATION     = "input_validation_failure"
    LLM_VALIDATION       = "llm_output_validation"
    HITL_DECISION        = "hitl_decision"
    PIPELINE_ERROR       = "pipeline_error"
    RATE_LIMIT           = "rate_limit_exceeded"
    DATA_ACCESS          = "data_access"
    MEMORY_WRITE         = "memory_write"
    SCHEMA_VIOLATION     = "schema_violation"


class SecurityAuditLogger:
    """
    Thread-safe JSONL audit logger.

    Each line in security_audit.jsonl is a self-contained JSON event record:
    {
        "ts":          "2026-05-22T14:30:00.000Z",
        "event_type":  "prompt_injection",
        "severity":    "CRITICAL",
        "company":     "Acme Corp",
        "period":      "Q1 2026",
        "user":        "analyst@firm.com",
        "detail":      {...},
        "source":      "input_sanitizer"
    }
    """

    def __init__(self, log_path: Optional[str] = None):
        self._path = log_path or os.getenv("SECURITY_LOG_PATH", _DEFAULT_LOG_PATH)
        self._lock = threading.Lock()

    def log(
        self,
        event_type: EventType,
        severity: EventSeverity = EventSeverity.INFO,
        company: Optional[str] = None,
        period: Optional[str] = None,
        user: Optional[str] = None,
        source: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "ts":         datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "event_type": event_type.value if isinstance(event_type, EventType) else event_type,
            "severity":   severity.value if isinstance(severity, EventSeverity) else severity,
            "company":    company,
            "period":     period,
            "user":       user,
            "source":     source,
            "detail":     detail or {},
        }
        line = json.dumps(record, default=str)
        with self._lock:
            try:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                # Don't crash the pipeline if the audit log can't be written.
                pass

    # ── convenience methods ──────────────────────────────────────────────────

    def auth_failure(self, user: str, source: str = "api") -> None:
        self.log(
            EventType.AUTH_FAILURE,
            EventSeverity.CRITICAL,
            user=user, source=source,
            detail={"message": "API key validation failed"},
        )

    def prompt_injection(
        self,
        field: str,
        pattern: str,
        company: Optional[str] = None,
        user: Optional[str] = None,
    ) -> None:
        self.log(
            EventType.PROMPT_INJECTION,
            EventSeverity.CRITICAL,
            company=company, user=user,
            source="input_sanitizer",
            detail={"field": field, "matched_pattern": pattern[:200]},
        )

    def input_validation_failure(
        self,
        field: str,
        reason: str,
        company: Optional[str] = None,
        user: Optional[str] = None,
    ) -> None:
        self.log(
            EventType.INPUT_VALIDATION,
            EventSeverity.WARNING,
            company=company, user=user,
            source="input_sanitizer",
            detail={"field": field, "reason": reason},
        )

    def llm_validation_failure(
        self,
        errors: list,
        warnings: list,
        company: Optional[str] = None,
        period: Optional[str] = None,
    ) -> None:
        severity = EventSeverity.CRITICAL if errors else EventSeverity.WARNING
        self.log(
            EventType.LLM_VALIDATION,
            severity,
            company=company, period=period,
            source="llm_validator",
            detail={"errors": errors, "warnings": warnings},
        )

    def hitl_decision(
        self,
        decision: str,
        approver: str,
        company: Optional[str] = None,
        period: Optional[str] = None,
        triggers: Optional[list] = None,
    ) -> None:
        self.log(
            EventType.HITL_DECISION,
            EventSeverity.INFO,
            company=company, period=period,
            user=approver, source="human_review_node",
            detail={"decision": decision, "triggers": triggers or []},
        )

    def schema_violation(
        self,
        errors: list,
        company: Optional[str] = None,
        user: Optional[str] = None,
    ) -> None:
        self.log(
            EventType.SCHEMA_VIOLATION,
            EventSeverity.WARNING,
            company=company, user=user,
            source="pydantic_schema",
            detail={"schema_errors": errors},
        )

    # ── read-back (for dashboard/review) ────────────────────────────────────

    def read_recent(self, n: int = 100) -> list:
        """Return the last n security events as dicts (most recent last)."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return [json.loads(l) for l in lines[-n:] if l.strip()]
        except FileNotFoundError:
            return []
        except Exception:
            return []

    def read_by_severity(self, severity: str) -> list:
        """Return all events matching a given severity level."""
        return [e for e in self.read_recent(n=10_000) if e.get("severity") == severity]

    def read_by_type(self, event_type: str) -> list:
        return [e for e in self.read_recent(n=10_000) if e.get("event_type") == event_type]


# ── module singleton ─────────────────────────────────────────────────────────

_singleton: Optional[SecurityAuditLogger] = None


def get_security_logger(log_path: Optional[str] = None) -> SecurityAuditLogger:
    global _singleton
    if log_path:
        return SecurityAuditLogger(log_path=log_path)
    if _singleton is None:
        _singleton = SecurityAuditLogger()
    return _singleton
