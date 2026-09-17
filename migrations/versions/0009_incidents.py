"""Incident reporting.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import EncryptedText, UTCDateTime

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "incidents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", EncryptedText("incidents.title"), nullable=False),
        sa.Column("category", sa.String(20), nullable=False),
        sa.Column("harm", sa.String(10), nullable=False),
        sa.Column("status", sa.String(15), nullable=False),
        sa.Column("description", EncryptedText("incidents.description"), nullable=False),
        sa.Column("occurred_at", UTCDateTime(), nullable=True),
        sa.Column("reported_at", UTCDateTime(), nullable=False),
        sa.Column("aware_at", UTCDateTime(), nullable=False),
        sa.Column("reported_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("reporter_name", sa.String(200), nullable=False),
        sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("root_cause", EncryptedText("incidents.root_cause"), nullable=False),
        sa.Column("actions", EncryptedText("incidents.actions"), nullable=False),
        sa.Column("external_reference", EncryptedText("incidents.external_reference"), nullable=False),
        sa.Column("audit_id", sa.String(16), nullable=True),
        sa.Column("review_case_id", sa.Integer(), sa.ForeignKey("review_cases.id", ondelete="SET NULL"), nullable=True),
        sa.Column("alert_id", sa.Integer(), sa.ForeignKey("alerts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("imaging_series_id", sa.Integer(), sa.ForeignKey("imaging_series.id", ondelete="SET NULL"), nullable=True),
        sa.Column("hazard_id", sa.Integer(), sa.ForeignKey("hazards.id", ondelete="SET NULL"), nullable=True),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.Column("closed_at", UTCDateTime(), nullable=True),
    )
    op.create_index("ix_incidents_status", "incidents", ["status"])
    op.create_index("ix_incidents_reported_at", "incidents", ["reported_at"])
    op.create_table(
        "incident_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("incident_id", sa.Integer(), sa.ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("at", UTCDateTime(), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("user_name", sa.String(200), nullable=False),
        sa.Column("action", sa.String(20), nullable=False),
        sa.Column("note", EncryptedText("incident_events.note"), nullable=False),
    )
    op.create_index("ix_incident_events_incident_id", "incident_events", ["incident_id"])


def downgrade() -> None:
    op.drop_table("incident_events")
    op.drop_table("incidents")
