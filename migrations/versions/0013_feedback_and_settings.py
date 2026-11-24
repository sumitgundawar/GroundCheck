"""Clinician feedback on answers, and the install's own small settings.

Revision ID: 0013
Revises: 0012
Create Date: 2026-11-24
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import UTCDateTime

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "answer_feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("audit_id", sa.String(16), nullable=False),
        sa.Column("voter", sa.String(24), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("site_id", sa.Integer(), sa.ForeignKey("sites.id", ondelete="SET NULL"), nullable=True),
        sa.Column("rating", sa.String(4), nullable=False),
        sa.Column("reason", sa.String(20), nullable=False, server_default=""),
        sa.Column("comment", sa.Text(), nullable=False, server_default=""),
        sa.Column("case_id", sa.Integer(), sa.ForeignKey("review_cases.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.UniqueConstraint("audit_id", "voter", name="uq_answer_feedback_voter"),
    )
    op.create_index("ix_answer_feedback_created_at", "answer_feedback", ["created_at"])
    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("value", sa.String(500), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_table("app_settings")
    op.drop_index("ix_answer_feedback_created_at", table_name="answer_feedback")
    op.drop_table("answer_feedback")
