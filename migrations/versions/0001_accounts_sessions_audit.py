"""Users, sign-in sessions and audit records.

Revision ID: 0001
Revises:
Create Date: 2026-09-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import UTCDateTime

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("name", sa.String(200), nullable=False, server_default=""),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="clinician"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("mfa_secret", sa.String(64), nullable=True),
        sa.Column("mfa_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("failed_logins", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("locked_until", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("last_login_at", UTCDateTime(), nullable=True),
    )
    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("expires_at", UTCDateTime(), nullable=False),
        sa.Column("last_seen_at", UTCDateTime(), nullable=False),
        sa.Column("mfa_pending", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("ip_address", sa.String(45), nullable=False, server_default=""),
        sa.Column("user_agent", sa.String(300), nullable=False, server_default=""),
    )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])
    op.create_table(
        "audit_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("audit_id", sa.String(16), nullable=False, unique=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("decision", sa.String(10), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("total_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("llm_used", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("record", sa.JSON(), nullable=False),
    )
    op.create_index("ix_audit_records_created_at", "audit_records", ["created_at"])
    op.create_index("ix_audit_records_user_id", "audit_records", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_audit_records_user_id", table_name="audit_records")
    op.drop_index("ix_audit_records_created_at", table_name="audit_records")
    op.drop_table("audit_records")
    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_table("users")
