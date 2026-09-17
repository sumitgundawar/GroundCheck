"""Knowledge releases.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import UTCDateTime

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "releases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("number", sa.Integer(), nullable=False, unique=True),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("reason", sa.String(300), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_by_name", sa.String(200), nullable=False),
        sa.Column("passages", sa.Integer(), nullable=False),
        sa.Column("demo_passages", sa.Integer(), nullable=False),
        sa.Column("documents", sa.JSON(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("formulary_sha256", sa.String(64), nullable=False),
        sa.Column("check", sa.JSON(), nullable=True),
        sa.Column("live_at", UTCDateTime(), nullable=True),
        sa.Column("live_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("live_by_name", sa.String(200), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("releases")
