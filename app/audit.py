"""Audit trail.

Every question produces one record: the decision, the full trace, the
retrieved sources and the settings used, and, when accounts are in use, the
user who asked.

Records are stored in the database (see app/db.py). If the database can't be
reached, the store falls back to an append-only JSONL file, and if that isn't
writable either, to memory only, so the app keeps working. An in-memory ring
buffer of recent records serves the dashboard quickly in every case. Set
AUDIT_PERSIST=false to keep records in memory only.

The raw question is never stored: records keep the PII-redacted question."""

from __future__ import annotations

import json
import logging
import secrets
from collections import OrderedDict
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from . import config, db, integrity
from .schemas import AskResponse

log = logging.getLogger("groundcheck.audit")


def _top_score(extras: dict[str, Any]) -> float | None:
    scores = []
    for item in extras.get("retrieved") or []:
        try:
            scores.append(float(item.get("score")))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else None


class AuditStore:
    def __init__(self, capacity: int, log_path: Path | None = None, persist: bool = True):
        self.capacity = capacity
        self.log_path = log_path
        self.persist = persist
        self._records: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._backend: str | None = None  # "database", "file", or None

    # --- Backend selection ---------------------------------------------------
    def backend(self) -> str | None:
        """Where records are persisted, chosen on first use."""
        if self._backend is not None or not self.persist:
            return self._backend
        if db.ready():
            self._backend = "database"
        elif self.log_path is not None and self._init_log():
            self._backend = "file"
            self._load_recent_from_file()
        else:
            self.persist = False
        return self._backend

    def _init_log(self) -> bool:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.touch(exist_ok=True)
            return True
        except OSError:
            return False

    def _load_recent_from_file(self) -> None:
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines[-self.capacity:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("audit_id"):
                self._remember(record)

    def _remember(self, record: dict[str, Any]) -> None:
        self._records[record["audit_id"]] = record
        self._records.move_to_end(record["audit_id"])
        while len(self._records) > self.capacity:
            self._records.popitem(last=False)

    # --- API ---------------------------------------------------------------
    def new_id(self) -> str:
        return secrets.token_hex(4)  # 8 hex characters

    def save(self, audit_id: str, response: AskResponse, extras: dict[str, Any],
             user_id: int | None = None, site_id: int | None = None) -> None:
        record = {
            "audit_id": audit_id,
            "user_id": user_id,
            "site_id": site_id,
            "response": response.model_dump(),
            **{k: v for k, v in extras.items() if k != "raw_query"},
        }
        self._remember(record)

        backend = self.backend()
        if backend == "database":
            try:
                with integrity.write_lock, db.session() as s:
                    integrity.append(s, db.AuditRecord(
                        audit_id=audit_id,
                        user_id=user_id,
                        decision=response.decision,
                        query=str(extras.get("redacted_query", "")),
                        total_ms=response.total_ms,
                        llm_used=response.llm_used,
                        record=record,
                        test_run=bool(extras.get("test_run")),
                        top_score=_top_score(extras),
                        site_id=site_id,
                    ))
            except Exception as exc:  # noqa: BLE001 - never fail a question over the audit
                log.error("Couldn't write audit record %s to the database: %s", audit_id, exc)
        elif backend == "file":
            try:
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                self._backend, self.persist = None, False

    def get(self, audit_id: str) -> dict[str, Any] | None:
        if audit_id in self._records:
            return self._records[audit_id]
        if self.backend() == "database":
            with db.session() as s:
                row = s.scalar(select(db.AuditRecord).where(db.AuditRecord.audit_id == audit_id))
                return row.record if row else None
        return None

    def recent(self, limit: int = 20, user_id: int | None = None, site_id: int | None = None) -> list[dict[str, Any]]:
        """Most recent records first, lightweight fields only. With user_id,
        only that user's records; with site_id, only that site's."""
        if self.backend() == "database":
            with db.session() as s:
                q = select(db.AuditRecord).order_by(db.AuditRecord.created_at.desc(), db.AuditRecord.id.desc())
                if user_id is not None:
                    q = q.where(db.AuditRecord.user_id == user_id)
                if site_id is not None:
                    q = q.where(db.AuditRecord.site_id == site_id)
                rows = s.scalars(q.limit(limit)).all()
                return [{
                    "audit_id": r.audit_id, "decision": r.decision, "query": r.query,
                    "total_ms": r.total_ms, "llm_used": r.llm_used, "user_id": r.user_id,
                    "created_at": r.created_at.isoformat(),
                } for r in rows]

        items = [r for r in self._records.values() if user_id is None or r.get("user_id") == user_id]
        return [{
            "audit_id": r.get("audit_id"),
            "decision": r.get("response", {}).get("decision"),
            "query": r.get("redacted_query", ""),
            "total_ms": r.get("response", {}).get("total_ms"),
            "llm_used": r.get("response", {}).get("llm_used"),
            "user_id": r.get("user_id"),
        } for r in items[-limit:][::-1]]

    def count(self) -> int:
        if self.backend() == "database":
            with db.session() as s:
                return s.scalar(select(func.count(db.AuditRecord.id))) or 0
        return len(self._records)


store = AuditStore(config.AUDIT_RING_SIZE, config.AUDIT_LOG_PATH, config.AUDIT_PERSIST)
