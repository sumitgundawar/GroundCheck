"""Service account API keys, daily usage rollups, and the key on audit records.

Revision ID: 0012
Revises: 0011
Create Date: 2026-11-24
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import UTCDateTime

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("prefix", sa.String(16), nullable=False, unique=True),
        sa.Column("salt", sa.String(32), nullable=False),
        sa.Column("hash", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False, server_default=""),
        sa.Column("site_id", sa.Integer(), sa.ForeignKey("sites.id", ondelete="SET NULL"), nullable=True),
        sa.Column("scopes", sa.String(200), nullable=False, server_default=""),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("expires_at", UTCDateTime(), nullable=True),
        sa.Column("last_used_at", UTCDateTime(), nullable=True),
        sa.Column("revoked_at", UTCDateTime(), nullable=True),
        sa.Column("rpm_limit", sa.Integer(), nullable=True),
        sa.Column("monthly_quota", sa.Integer(), nullable=True),
        sa.Column("quota_period", sa.String(7), nullable=False, server_default=""),
        sa.Column("quota_used", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_api_keys_site_id", "api_keys", ["site_id"])
    op.create_table(
        "usage_daily",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("day", sa.String(10), nullable=False),
        sa.Column("site_id", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("api_key_id", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("endpoint", sa.String(64), nullable=False),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
        sa.UniqueConstraint("day", "site_id", "api_key_id", "endpoint", "decision", name="uq_usage_daily"),
    )
    op.create_index("ix_usage_daily_day", "usage_daily", ["day"])
    with op.batch_alter_table("audit_records") as batch:
        batch.add_column(sa.Column("api_key_id", sa.Integer(), nullable=True))
        batch.create_foreign_key("fk_audit_records_api_key_id", "api_keys", ["api_key_id"], ["id"],
                                 ondelete="SET NULL")
        batch.create_index("ix_audit_records_api_key_id", ["api_key_id"])


def downgrade() -> None:
    with op.batch_alter_table("audit_records") as batch:
        batch.drop_index("ix_audit_records_api_key_id")
        batch.drop_constraint("fk_audit_records_api_key_id", type_="foreignkey")
        batch.drop_column("api_key_id")
    op.drop_index("ix_usage_daily_day", table_name="usage_daily")
    op.drop_table("usage_daily")
    op.drop_index("ix_api_keys_site_id", table_name="api_keys")
    op.drop_table("api_keys")
