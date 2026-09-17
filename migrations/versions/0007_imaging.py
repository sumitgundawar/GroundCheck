"""CT and MRI imaging: imported series, model analyses and clinician reports.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import EncryptedJSON, EncryptedText, UTCDateTime

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "imaging_series",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("uid", sa.String(64), nullable=False, unique=True),
        sa.Column("study_uid", sa.String(64), nullable=False),
        sa.Column("source", sa.String(10), nullable=False),
        sa.Column("label", EncryptedText("imaging_series.label"), nullable=False),
        sa.Column("modality", sa.String(4), nullable=False),
        sa.Column("description", EncryptedText("imaging_series.description"), nullable=False),
        sa.Column("body_part", sa.String(64), nullable=False),
        sa.Column("slices", sa.Integer(), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("columns", sa.Integer(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("original", EncryptedJSON("imaging_series.original"), nullable=True),
        sa.Column("file", sa.String(80), nullable=False),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
    )
    op.create_table(
        "imaging_analyses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("series_id", sa.Integer(), sa.ForeignKey("imaging_series.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_id", sa.String(200), nullable=False),
        sa.Column("model_name", sa.String(200), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("requested_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
    )
    op.create_index("ix_imaging_analyses_series_id", "imaging_analyses", ["series_id"])
    op.create_table(
        "imaging_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("series_id", sa.Integer(), sa.ForeignKey("imaging_series.id", ondelete="CASCADE"), nullable=False),
        sa.Column("analysis_id", sa.Integer(), sa.ForeignKey("imaging_analyses.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("agreement", sa.String(12), nullable=False),
        sa.Column("findings", EncryptedText("imaging_reports.findings"), nullable=False),
        sa.Column("impression", EncryptedText("imaging_reports.impression"), nullable=False),
        sa.Column("author_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("author_name", sa.String(200), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.Column("signed_at", UTCDateTime(), nullable=True),
        sa.Column("replaces_id", sa.Integer(), sa.ForeignKey("imaging_reports.id", ondelete="SET NULL"), nullable=True),
        sa.Column("sr_uid", sa.String(64), nullable=False),
        sa.Column("sent_at", UTCDateTime(), nullable=True),
    )
    op.create_index("ix_imaging_reports_series_id", "imaging_reports", ["series_id"])


def downgrade() -> None:
    op.drop_table("imaging_reports")
    op.drop_table("imaging_analyses")
    op.drop_table("imaging_series")
