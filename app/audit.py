"""Audit trail.

An in-memory ring buffer keeps the most recent records for fast display, and
each record is also appended to a JSONL file so the trail survives a restart.
Persistence is best-effort: if the log path is not writable (for example a
read-only container), the store silently falls back to memory only and the app
keeps working. Set AUDIT_PERSIST=false to disable disk persistence."""

from __future__ import annotations

import json
import secrets
from collections import OrderedDict
from pathlib import Path
from typing import Any

from . import config
from .schemas import AskResponse


class AuditStore:
    def __init__(self, capacity: int, log_path: Path | None = None,
                 persist: bool = True):
        self.capacity = capacity
        self.log_path = log_path
        self.persist = persist and log_path is not None
        self._records: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        if self.persist:
            self._init_log()
            self._load_recent()

    # --- persistence -------------------------------------------------------
    def _init_log(self) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Touch the file so a missing log is not an error on first read.
            self.log_path.touch(exist_ok=True)
        except OSError:
            # Not writable: degrade to memory only.
            self.persist = False

    def _load_recent(self) -> None:
        """Reload the last `capacity` records from the log into memory so the UI
        can show history after a restart."""
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return
        for line in lines[-self.capacity:]:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = record.get("audit_id")
            if aid:
                self._records[aid] = record
        # Keep only the most recent within capacity.
        while len(self._records) > self.capacity:
            self._records.popitem(last=False)

    def _append(self, record: dict[str, Any]) -> None:
        if not self.persist:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # Stop trying to persist if the disk becomes unavailable; the
            # in-memory buffer continues to serve the UI.
            self.persist = False

    # --- API ---------------------------------------------------------------
    def new_id(self) -> str:
        return secrets.token_hex(4)  # 8 hex characters

    def save(self, audit_id: str, response: AskResponse, extras: dict[str, Any]) -> None:
        record = {
            "audit_id": audit_id,
            "response": response.model_dump(),
            **extras,
        }
        self._records[audit_id] = record
        self._records.move_to_end(audit_id)
        while len(self._records) > self.capacity:
            self._records.popitem(last=False)
        self._append(record)

    def get(self, audit_id: str) -> dict[str, Any] | None:
        return self._records.get(audit_id)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent records first, lightweight fields only."""
        items = list(self._records.values())[-limit:][::-1]
        out = []
        for r in items:
            resp = r.get("response", {})
            out.append({
                "audit_id": r.get("audit_id"),
                "decision": resp.get("decision"),
                "query": r.get("redacted_query") or r.get("raw_query", ""),
                "total_ms": resp.get("total_ms"),
                "llm_used": resp.get("llm_used"),
            })
        return out

    def count(self) -> int:
        return len(self._records)


store = AuditStore(config.AUDIT_RING_SIZE, config.AUDIT_LOG_PATH, config.AUDIT_PERSIST)
