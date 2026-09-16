"""Tamper-evident audit chain, and room for encrypted values.

Adds a number, previous hash, hash and algorithm to every audit record, and
the chain head. Existing records are chained in the order they were written.
Widens users.mfa_secret, which is encrypted when DATA_ENCRYPTION_KEYS is set,
and adds a log of retention runs.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app import integrity
from app.db import EncryptedJSON, EncryptedText, UTCDateTime

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("audit_records") as batch:
        batch.add_column(sa.Column("seq", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("prev_hash", sa.String(64), nullable=True))
        batch.add_column(sa.Column("entry_hash", sa.String(64), nullable=True))
        batch.add_column(sa.Column("chain_alg", sa.String(40), nullable=True))
        batch.create_unique_constraint("uq_audit_records_seq", ["seq"])
    with op.batch_alter_table("users") as batch:
        batch.alter_column("mfa_secret", type_=sa.Text(), existing_type=sa.String(64), existing_nullable=True)

    op.create_table(
        "retention_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ran_at", UTCDateTime(), nullable=False),
        sa.Column("ran_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("audit_cutoff", UTCDateTime(), nullable=True),
        sa.Column("audit_deleted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("anchor_seq", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("review_cutoff", UTCDateTime(), nullable=True),
        sa.Column("reviews_deleted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sessions_deleted", sa.Integer(), nullable=False, server_default="0"),
    )

    chain = op.create_table(
        "audit_chain",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("last_seq", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_hash", sa.String(64), nullable=False),
        sa.Column("anchor_seq", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("anchor_hash", sa.String(64), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
    )

    records = sa.table(
        "audit_records",
        sa.column("id", sa.Integer()), sa.column("audit_id", sa.String()),
        sa.column("created_at", UTCDateTime()), sa.column("user_id", sa.Integer()),
        sa.column("decision", sa.String()), sa.column("query", EncryptedText("audit_records.query")),
        sa.column("total_ms", sa.Integer()), sa.column("llm_used", sa.Boolean()),
        sa.column("record", EncryptedJSON("audit_records.record")),
        sa.column("seq", sa.Integer()), sa.column("prev_hash", sa.String()),
        sa.column("entry_hash", sa.String()), sa.column("chain_alg", sa.String()),
    )
    bind = op.get_bind()
    rows = bind.execute(sa.select(records.c.id, records.c.audit_id, records.c.created_at, records.c.user_id,
                                  records.c.decision, records.c.query, records.c.total_ms, records.c.llm_used,
                                  records.c.record)
                        .order_by(records.c.created_at, records.c.id)).all()
    seq, prev_hash = 0, integrity.GENESIS
    for row in rows:
        seq += 1
        content = integrity.canonical(seq, row.audit_id, row.created_at, row.user_id, row.decision,
                                      row.query, row.total_ms, row.llm_used, row.record)
        alg, entry_hash = integrity.compute(prev_hash, content)
        bind.execute(records.update().where(records.c.id == row.id)
                     .values(seq=seq, prev_hash=prev_hash, entry_hash=entry_hash, chain_alg=alg))
        prev_hash = entry_hash

    from datetime import datetime, timezone

    op.bulk_insert(chain, [{"id": 1, "last_seq": seq, "last_hash": prev_hash, "anchor_seq": 0,
                            "anchor_hash": integrity.GENESIS, "updated_at": datetime.now(timezone.utc)}])


def downgrade() -> None:
    op.drop_table("audit_chain")
    op.drop_table("retention_runs")
    with op.batch_alter_table("users") as batch:
        batch.alter_column("mfa_secret", type_=sa.String(64), existing_type=sa.Text(), existing_nullable=True)
    with op.batch_alter_table("audit_records") as batch:
        batch.drop_constraint("uq_audit_records_seq", type_="unique")
        batch.drop_column("chain_alg")
        batch.drop_column("entry_hash")
        batch.drop_column("prev_hash")
        batch.drop_column("seq")
