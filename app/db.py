"""Database: engine, sessions, schema and migrations.

GroundCheck stores users, sign-in sessions and audit records in a relational
database through SQLAlchemy, so the same code runs on:

- SQLite, the default: a single file, nothing to install
- PostgreSQL: DATABASE_URL=postgresql+psycopg://user:pass@host/db
- MySQL or MariaDB: DATABASE_URL=mysql+pymysql://user:pass@host/db

The schema is managed with Alembic migrations in migrations/. On startup the
app upgrades the database to the latest migration (DB_AUTO_MIGRATE=true), so a
new install and an upgrade both need no manual step."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from . import config, encryption

log = logging.getLogger("groundcheck.db")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC timestamps on every backend. Values are converted to
    UTC when written, and SQLite and MySQL, which drop timezone information,
    get it back as UTC when read."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if isinstance(value, datetime):
            value = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
            if dialect.name in ("sqlite", "mysql"):
                value = value.replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if isinstance(value, datetime) and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


class EncryptedText(TypeDecorator):
    """Text encrypted with DATA_ENCRYPTION_KEYS when it's set (app/encryption.py).
    The context names the column and is bound to the ciphertext."""

    impl = Text
    cache_ok = True

    def __init__(self, context: str):
        super().__init__()
        self.context = context

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return encryption.keyring().encrypt(value, self.context)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return encryption.keyring().decrypt(value, self.context)


class EncryptedJSON(TypeDecorator):
    """A JSON document, stored as an encrypted string when DATA_ENCRYPTION_KEYS
    is set, and as plain JSON otherwise."""

    impl = JSON
    cache_ok = True

    def __init__(self, context: str):
        super().__init__()
        self.context = context

    def process_bind_param(self, value, dialect):
        ring = encryption.keyring()
        if value is None or not ring.enabled:
            return value
        return ring.encrypt(json.dumps(value, ensure_ascii=False, separators=(",", ":")), self.context)

    def process_result_value(self, value, dialect):
        if encryption.is_encrypted(value):
            return json.loads(encryption.keyring().decrypt(value, self.context))
        return value


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="clinician")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    mfa_secret: Mapped[str | None] = mapped_column(EncryptedText("users.mfa_secret"), nullable=True)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    failed_logins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    # Set for people who sign in through single sign-on (app/sso.py).
    sso_issuer: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sso_subject: Mapped[str | None] = mapped_column(String(255), nullable=True)

    __table_args__ = (UniqueConstraint("sso_issuer", "sso_subject", name="uq_users_sso"),)

    sessions: Mapped[list["AuthSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class AuthSession(Base):
    """A signed-in browser. Only a SHA-256 hash of the token is stored, so a
    database leak doesn't hand out working sessions."""

    __tablename__ = "auth_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    # True until the second factor is verified, for users with MFA enabled.
    mfa_pending: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False, default="")
    user_agent: Mapped[str] = mapped_column(String(300), nullable=False, default="")

    user: Mapped[User] = relationship(back_populates="sessions")

    __table_args__ = (Index("ix_auth_sessions_user_id", "user_id"),)


class AuditRecord(Base):
    """One question and everything the pipeline did with it."""

    __tablename__ = "audit_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    audit_id: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    decision: Mapped[str] = mapped_column(String(10), nullable=False)
    query: Mapped[str] = mapped_column(EncryptedText("audit_records.query"), nullable=False, default="")
    total_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    record: Mapped[dict] = mapped_column(EncryptedJSON("audit_records.record"), nullable=False)
    # Tamper-evident chain (app/integrity.py). Null only for records written
    # before the chain existed and not yet backfilled.
    seq: Mapped[int | None] = mapped_column(Integer, nullable=True, unique=True)
    prev_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entry_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_alg: Mapped[str | None] = mapped_column(String(40), nullable=True)

    __table_args__ = (
        Index("ix_audit_records_created_at", "created_at"),
        Index("ix_audit_records_user_id", "user_id"),
    )


class SsoLogin(Base):
    """A single sign-on attempt between sending someone to the identity
    provider and their return. Kept in the database so any app instance can
    finish it. The state is stored only as a hash."""

    __tablename__ = "sso_logins"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    nonce: Mapped[str] = mapped_column(String(64), nullable=False)
    code_verifier: Mapped[str] = mapped_column(EncryptedText("sso_logins.code_verifier"), nullable=False)
    next_path: Mapped[str] = mapped_column(String(300), nullable=False, default="/")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class EhrLaunch(Base):
    """A SMART on FHIR launch between sending the browser to the EHR's
    authorisation server and its return. The state is stored as a hash."""

    __tablename__ = "ehr_launches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    issuer: Mapped[str] = mapped_column(String(500), nullable=False)
    code_verifier: Mapped[str] = mapped_column(EncryptedText("ehr_launches.code_verifier"), nullable=False)
    token_endpoint: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class EhrContext(Base):
    """A patient loaded from an EHR for one browser, for a limited time. The
    patient details are encrypted when DATA_ENCRYPTION_KEYS is set; the browser
    holds only a random token, stored here as a hash."""

    __tablename__ = "ehr_contexts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    data: Mapped[dict] = mapped_column(EncryptedJSON("ehr_contexts.data"), nullable=False)


class RetentionRun(Base):
    """A record of each time retention deleted data: what, up to when, and who."""

    __tablename__ = "retention_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ran_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    ran_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    audit_cutoff: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    audit_deleted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    anchor_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    review_cutoff: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    reviews_deleted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sessions_deleted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class AuditChain(Base):
    """The audit chain's head: the last number and hash written, and the anchor
    the remaining records start from after retention removes older ones."""

    __tablename__ = "audit_chain"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    last_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    anchor_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    anchor_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)


class Source(Base):
    """One version of a document imported by the organisation. A new upload of
    the same document key becomes the next version; approving it retires the
    previous approved version."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_key: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    filename: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    media_type: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    owner: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    effective_from: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    expires_on: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    uploaded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    review_note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    evaluation: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    chunks: Mapped[list["SourceChunk"]] = relationship(
        back_populates="source", cascade="all, delete-orphan", order_by="SourceChunk.position")

    __table_args__ = (
        Index("ix_sources_document_key", "document_key"),
        Index("ix_sources_status", "status"),
    )


class SourceChunk(Base):
    __tablename__ = "source_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    section: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    text: Mapped[str] = mapped_column(Text, nullable=False)

    source: Mapped[Source] = relationship(back_populates="chunks")

    __table_args__ = (Index("ix_source_chunks_source_id", "source_id"),)


class ReviewCase(Base):
    """A refusal or a flagged answer that a clinician needs to look at.
    Repeats of the same open question are grouped into one case."""

    __tablename__ = "review_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)          # refusal, flagged
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open")  # open, resolved, dismissed
    priority: Mapped[str] = mapped_column(String(10), nullable=False, default="normal")  # normal, high
    query: Mapped[str] = mapped_column(EncryptedText("review_cases.query"), nullable=False)
    query_key: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(EncryptedText("review_cases.reason"), nullable=False, default="")
    reason_category: Mapped[str] = mapped_column(String(40), nullable=False, default="")
    first_audit_id: Mapped[str] = mapped_column(String(16), nullable=False)
    last_audit_id: Mapped[str] = mapped_column(String(16), nullable=False)
    occurrences: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    due_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    escalated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    assigned_to: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    flagged_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    resolved_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    outcome: Mapped[str] = mapped_column(String(30), nullable=False, default="")
    outcome_note: Mapped[str] = mapped_column(EncryptedText("review_cases.outcome_note"), nullable=False, default="")

    events: Mapped[list["ReviewEvent"]] = relationship(
        back_populates="case", cascade="all, delete-orphan", order_by="ReviewEvent.id")

    __table_args__ = (
        Index("ix_review_cases_status", "status"),
        Index("ix_review_cases_query_key", "query_key"),
        Index("ix_review_cases_created_at", "created_at"),
    )


class ReviewEvent(Base):
    """One entry in a case's timeline."""

    __tablename__ = "review_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("review_cases.id", ondelete="CASCADE"), nullable=False)
    at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    note: Mapped[str] = mapped_column(EncryptedText("review_events.note"), nullable=False, default="")

    case: Mapped[ReviewCase] = relationship(back_populates="events")

    __table_args__ = (Index("ix_review_events_case_id", "case_id"),)


class EvalCase(Base):
    """A test question added from a review, run alongside the golden set."""

    __tablename__ = "eval_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(EncryptedText("eval_cases.query"), nullable=False)
    expect: Mapped[str] = mapped_column(String(10), nullable=False)
    note: Mapped[str] = mapped_column(EncryptedText("eval_cases.note"), nullable=False, default="")
    from_case_id: Mapped[int | None] = mapped_column(ForeignKey("review_cases.id", ondelete="SET NULL"), nullable=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)


class Hazard(Base):
    """An entry in the clinical hazard log (DCB0129 / ISO 14971 style): what
    could go wrong, how bad and how likely, the controls, and the risk left."""

    __tablename__ = "hazards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    cause: Mapped[str] = mapped_column(Text, nullable=False, default="")
    effect: Mapped[str] = mapped_column(Text, nullable=False, default="")
    severity: Mapped[int] = mapped_column(Integer, nullable=False)
    likelihood: Mapped[int] = mapped_column(Integer, nullable=False)
    controls: Mapped[str] = mapped_column(Text, nullable=False, default="")
    residual_severity: Mapped[int] = mapped_column(Integer, nullable=False)
    residual_likelihood: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open")  # open, mitigated, closed
    owner: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    related_case_id: Mapped[int | None] = mapped_column(ForeignKey("review_cases.id", ondelete="SET NULL"), nullable=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)


# --------------------------------------------------------------------------
# Engine and sessions
# --------------------------------------------------------------------------

_engine: Engine | None = None
_sessionmaker: sessionmaker[Session] | None = None
_lock = threading.Lock()


def _make_engine(url: str) -> Engine:
    kwargs: dict = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    return engine


def engine() -> Engine:
    global _engine, _sessionmaker
    with _lock:
        if _engine is None:
            url = config.DATABASE_URL
            if url.startswith("sqlite:///") and ":memory:" not in url:
                # A relative SQLite path is relative to the repository root, and
                # its folder is created if needed.
                raw = url.removeprefix("sqlite:///")
                path = Path(raw) if raw.startswith("/") else config.ROOT_DIR / raw
                path.parent.mkdir(parents=True, exist_ok=True)
                url = f"sqlite:///{path}"
            _engine = _make_engine(url)
            _sessionmaker = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        return _engine


def reset_engine() -> None:
    """Forget the engine, so the next use reads DATABASE_URL again (tests)."""
    global _engine, _sessionmaker
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine, _sessionmaker = None, None


@contextmanager
def session() -> Iterator[Session]:
    engine()
    assert _sessionmaker is not None
    s = _sessionmaker()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def migrate() -> None:
    """Upgrade the database to the latest migration."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(config.ROOT_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(config.ROOT_DIR / "migrations"))
    with engine().begin() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")


_ready = False


def ready() -> bool:
    """Migrate once, on first use. Returns False (and the app runs without
    persistence) if the database can't be reached or written."""
    global _ready
    if _ready:
        return True
    try:
        if config.DB_AUTO_MIGRATE:
            migrate()
        _ready = True
    except Exception as exc:  # noqa: BLE001 - any failure means no database
        log.warning("Database unavailable, continuing without persistence: %s", exc)
        _ready = False
    return _ready
