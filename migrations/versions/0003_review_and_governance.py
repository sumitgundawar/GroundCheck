"""Review cases, their timelines, evaluation cases from reviews, and the hazard log.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import UTCDateTime

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def _user_fk(name: str) -> sa.Column:
    return sa.Column(name, sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


def upgrade() -> None:
    op.create_table(
        "review_cases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),
        sa.Column("priority", sa.String(10), nullable=False, server_default="normal"),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("query_key", sa.String(64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("reason_category", sa.String(40), nullable=False, server_default=""),
        sa.Column("first_audit_id", sa.String(16), nullable=False),
        sa.Column("last_audit_id", sa.String(16), nullable=False),
        sa.Column("occurrences", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("last_seen_at", UTCDateTime(), nullable=False),
        sa.Column("due_at", UTCDateTime(), nullable=False),
        sa.Column("escalated_at", UTCDateTime(), nullable=True),
        _user_fk("assigned_to"),
        _user_fk("flagged_by"),
        sa.Column("resolved_at", UTCDateTime(), nullable=True),
        _user_fk("resolved_by"),
        sa.Column("outcome", sa.String(30), nullable=False, server_default=""),
        sa.Column("outcome_note", sa.Text(), nullable=False),
    )
    op.create_index("ix_review_cases_status", "review_cases", ["status"])
    op.create_index("ix_review_cases_query_key", "review_cases", ["query_key"])
    op.create_index("ix_review_cases_created_at", "review_cases", ["created_at"])
    op.create_table(
        "review_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("case_id", sa.Integer(), sa.ForeignKey("review_cases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("at", UTCDateTime(), nullable=False),
        _user_fk("user_id"),
        sa.Column("action", sa.String(20), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
    )
    op.create_index("ix_review_events_case_id", "review_events", ["case_id"])
    op.create_table(
        "eval_cases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("expect", sa.String(10), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("from_case_id", sa.Integer(), sa.ForeignKey("review_cases.id", ondelete="SET NULL"), nullable=True),
        _user_fk("created_by"),
        sa.Column("created_at", UTCDateTime(), nullable=False),
    )
    op.create_table(
        "hazards",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("cause", sa.Text(), nullable=False),
        sa.Column("effect", sa.Text(), nullable=False),
        sa.Column("severity", sa.Integer(), nullable=False),
        sa.Column("likelihood", sa.Integer(), nullable=False),
        sa.Column("controls", sa.Text(), nullable=False),
        sa.Column("residual_severity", sa.Integer(), nullable=False),
        sa.Column("residual_likelihood", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),
        sa.Column("owner", sa.String(200), nullable=False, server_default=""),
        sa.Column("related_case_id", sa.Integer(), sa.ForeignKey("review_cases.id", ondelete="SET NULL"), nullable=True),
        _user_fk("created_by"),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("hazards")
    op.drop_table("eval_cases")
    op.drop_index("ix_review_events_case_id", table_name="review_events")
    op.drop_table("review_events")
    op.drop_index("ix_review_cases_created_at", table_name="review_cases")
    op.drop_index("ix_review_cases_query_key", table_name="review_cases")
    op.drop_index("ix_review_cases_status", table_name="review_cases")
    op.drop_table("review_cases")
