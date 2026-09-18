"""A tamper-evident audit trail.

Each audit record is numbered and chained to the one before it: its hash
covers its own content and the previous record's hash. Changing a record,
deleting one, reordering them or removing the newest ones breaks the chain,
and `verify()` says where.

With AUDIT_SIGNING_KEYS set, the hash is an HMAC-SHA256 keyed with a secret
kept outside the database, so someone who can write to the database can't
rewrite the chain to hide a change. Without a key, the chain uses SHA-256 and
detects changes made without recomputing it. Recording the chain head
(`python -m app.cli audit-head`) somewhere the database's administrators
can't change, such as a write-once log, detects even a complete rewrite.

Retention (app/retention.py) removes the oldest records in order and moves the
chain's anchor forward, so the records that remain still verify.

The hash covers the record's content in plain text, so rotating encryption
keys doesn't change it. Timestamps are covered to the second."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from datetime import timezone

from sqlalchemy import func, select, update

from . import config, db, encryption
from .db import AuditChain, AuditRecord

GENESIS = "0" * 64

# Serialises appends within a process. Across processes, the row lock on the
# chain head does the same.
write_lock = threading.Lock()


def signing_summary() -> dict:
    keys = list(_signing_keys())
    return {"enabled": bool(keys), "primary_key_id": keys[0] if keys else None, "verify_key_ids": keys[1:]}


def _signing_keys() -> dict[str, bytes]:
    texts = [t for t in config.AUDIT_SIGNING_KEYS.split(",") if t.strip()]
    keys = [encryption._decode_key(t) for t in texts]
    return {encryption.key_id(k): k for k in keys}


def _primary_signing_key() -> tuple[str, bytes] | None:
    keys = _signing_keys()
    return next(iter(keys.items()), None)


# Tells `canonical` to leave a field out entirely, rather than write it as null.
_LEGACY = object()


def canonical(seq: int, audit_id: str, created_at, user_id, decision: str, query: str,
              total_ms: int, llm_used: bool, record: dict, site_id: int | None = _LEGACY) -> bytes:
    """The bytes a record's hash covers. `site_id` decides which site's people
    can see a record, so moving one between sites has to break the chain; it
    joined the hashed content after the first release, and passing nothing
    reproduces the earlier format so records written then still verify."""
    fields = {
        "seq": seq,
        "audit_id": audit_id,
        "created_at": created_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user_id": user_id,
        "decision": decision,
        "query": query,
        "total_ms": int(total_ms),
        "llm_used": bool(llm_used),
        "record": record,
    }
    if site_id is not _LEGACY:
        fields["site_id"] = site_id
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def compute(prev_hash: str, content: bytes, alg: str | None = None) -> tuple[str, str]:
    """Returns (algorithm, hash). With alg=None, uses the primary signing key
    if one is configured, else SHA-256."""
    message = prev_hash.encode("ascii") + b"\n" + content
    if alg is None:
        primary = _primary_signing_key()
        alg = f"hmac-sha256:{primary[0]}" if primary else "sha256"
    if alg == "sha256":
        return alg, hashlib.sha256(message).hexdigest()
    if alg.startswith("hmac-sha256:"):
        key = _signing_keys().get(alg.split(":", 1)[1])
        if key is None:
            raise KeyError(alg)
        return alg, hmac.new(key, message, hashlib.sha256).hexdigest()
    raise KeyError(alg)


def _row_content(row: AuditRecord) -> bytes:
    return canonical(row.seq, row.audit_id, row.created_at, row.user_id, row.decision, row.query,
                     row.total_ms, row.llm_used, row.record, row.site_id)


def _legacy_row_content(row: AuditRecord) -> bytes:
    """The content format used before `site_id` was covered."""
    return canonical(row.seq, row.audit_id, row.created_at, row.user_id, row.decision, row.query,
                     row.total_ms, row.llm_used, row.record)


def append(s, row: AuditRecord) -> None:
    """Chain a new audit record and add it to the session. Call inside
    `write_lock` and commit the session before releasing it."""
    row.created_at = row.created_at or db.utcnow()
    # Updating the head first takes its row lock (a write lock on SQLite), so
    # concurrent writers queue here and each reads the head the last one left.
    s.execute(update(AuditChain).where(AuditChain.id == 1).values(last_seq=AuditChain.last_seq + 1))
    head = s.execute(select(AuditChain.last_seq, AuditChain.last_hash).where(AuditChain.id == 1)).one_or_none()
    if head is None:
        raise RuntimeError("The audit chain head is missing. Run migrations.")
    seq, prev_hash = head
    row.seq = seq
    row.prev_hash = prev_hash
    row.chain_alg, row.entry_hash = compute(prev_hash, _row_content(row))
    s.execute(update(AuditChain).where(AuditChain.id == 1)
              .values(last_hash=row.entry_hash, updated_at=db.utcnow()))
    s.add(row)


def head() -> dict:
    with db.session() as s:
        chain = s.get(AuditChain, 1)
        return {"seq": chain.last_seq, "hash": chain.last_hash, "anchor_seq": chain.anchor_seq,
                "anchor_hash": chain.anchor_hash, "updated_at": chain.updated_at.isoformat()}


def verify(max_problems: int = 20) -> dict:
    """Walk the chain from its anchor and report every break found."""
    problems: list[dict] = []
    problem_count = 0

    def problem(seq, audit_id, text):
        nonlocal problem_count
        problem_count += 1
        if len(problems) < max_problems:
            problems.append({"seq": seq, "audit_id": audit_id, "problem": text})

    checked = 0
    unverifiable = 0
    earlier_format = 0
    error = None
    first_seq = last_seq = None
    with db.session() as s:
        chain = s.get(AuditChain, 1)
        if chain is None:
            raise RuntimeError("The audit chain head is missing. Run migrations.")
        expected_seq, prev_hash = chain.anchor_seq + 1, chain.anchor_hash
        unchained = s.scalar(select(func.count(AuditRecord.id)).where(AuditRecord.seq.is_(None))) or 0
        rows = s.scalars(select(AuditRecord).where(AuditRecord.seq.is_not(None))
                         .order_by(AuditRecord.seq).execution_options(yield_per=500))
        try:
            for row in rows:
                checked += 1
                first_seq = row.seq if first_seq is None else first_seq
                last_seq = row.seq
                if row.seq < expected_seq:
                    problem(row.seq, row.audit_id, "Numbered before the start of the chain.")
                elif row.seq > expected_seq:
                    missing = row.seq - expected_seq
                    problem(expected_seq, None, f"{missing} record{'s are' if missing > 1 else ' is'} missing "
                                                f"before number {row.seq}.")
                if row.prev_hash != prev_hash:
                    problem(row.seq, row.audit_id, "Doesn't follow the record before it.")
                try:
                    _, expected = compute(row.prev_hash, _row_content(row), row.chain_alg)
                    if not hmac.compare_digest(expected, row.entry_hash or ""):
                        # Records written before `site_id` was covered still
                        # verify under the earlier format. They are counted so
                        # an operator can see how many predate the change.
                        _, legacy = compute(row.prev_hash, _legacy_row_content(row), row.chain_alg)
                        if hmac.compare_digest(legacy, row.entry_hash or ""):
                            earlier_format += 1
                        else:
                            problem(row.seq, row.audit_id, "Its content has changed since it was written.")
                except KeyError:
                    unverifiable += 1
                    problem(row.seq, row.audit_id,
                            f"Signed with a key that isn't in AUDIT_SIGNING_KEYS ({row.chain_alg}).")
                prev_hash, expected_seq = row.entry_hash, row.seq + 1
        except encryption.EncryptionError as exc:
            error = f"Stopped at record {expected_seq}: {exc}"
        if error is None and (chain.last_seq != expected_seq - 1 or chain.last_hash != prev_hash):
            problem(chain.last_seq, None,
                    "The newest records don't match the chain head: records were removed or rewritten.")
    return {
        "ok": problem_count == 0 and error is None,
        "complete": error is None,
        "error": error,
        "checked": checked,
        "first_seq": first_seq,
        "last_seq": last_seq,
        "anchor_seq": chain.anchor_seq,
        "unchained": unchained,
        "unverifiable": unverifiable,
        "earlier_format": earlier_format,
        "signed": _primary_signing_key() is not None,
        "problem_count": problem_count,
        "problems": problems,
        "head": {"seq": chain.last_seq, "hash": chain.last_hash},
        "verified_at": db.utcnow().isoformat(),
    }
