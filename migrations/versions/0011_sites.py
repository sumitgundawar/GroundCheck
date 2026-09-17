"""Sites: several hospitals or clinics in one installation.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import UTCDateTime

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None

SCOPED = ("users", "audit_records", "review_cases", "incidents", "imaging_series")


def upgrade() -> None:
    op.create_table(
        "sites",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("key", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
    )
    for table in SCOPED:
        with op.batch_alter_table(table) as batch:
            batch.add_column(sa.Column("site_id", sa.Integer(), nullable=True))
            batch.create_foreign_key(f"fk_{table}_site_id", "sites", ["site_id"], ["id"], ondelete="SET NULL")
            batch.create_index(f"ix_{table}_site_id", ["site_id"])


def downgrade() -> None:
    for table in SCOPED:
        with op.batch_alter_table(table) as batch:
            batch.drop_index(f"ix_{table}_site_id")
            batch.drop_constraint(f"fk_{table}_site_id", type_="foreignkey")
            batch.drop_column("site_id")
    op.drop_table("sites")
