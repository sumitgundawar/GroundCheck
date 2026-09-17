"""Monitoring: alerts, and audit columns read without decrypting records.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import UTCDateTime

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("audit_records") as batch:
        batch.add_column(sa.Column("test_run", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch.add_column(sa.Column("top_score", sa.Float(), nullable=True))
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("rule", sa.String(40), nullable=False),
        sa.Column("severity", sa.String(10), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("first_seen", UTCDateTime(), nullable=False),
        sa.Column("last_seen", UTCDateTime(), nullable=False),
        sa.Column("resolved_at", UTCDateTime(), nullable=True),
        sa.Column("acknowledged_at", UTCDateTime(), nullable=True),
        sa.Column("acknowledged_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("acknowledged_by_name", sa.String(200), nullable=False),
        sa.Column("notified", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_alerts_rule_status", "alerts", ["rule", "status"])


def downgrade() -> None:
    op.drop_table("alerts")
    with op.batch_alter_table("audit_records") as batch:
        batch.drop_column("top_score")
        batch.drop_column("test_run")
