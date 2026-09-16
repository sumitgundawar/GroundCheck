"""Imported documents and their chunks.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import UTCDateTime

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_key", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("filename", sa.String(255), nullable=False, server_default=""),
        sa.Column("media_type", sa.String(100), nullable=False, server_default=""),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("owner", sa.String(200), nullable=False, server_default=""),
        sa.Column("effective_from", UTCDateTime(), nullable=True),
        sa.Column("expires_on", UTCDateTime(), nullable=True),
        sa.Column("uploaded_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("uploaded_at", UTCDateTime(), nullable=False),
        sa.Column("reviewed_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("reviewed_at", UTCDateTime(), nullable=True),
        sa.Column("review_note", sa.Text(), nullable=False),
        sa.Column("evaluation", sa.JSON(), nullable=True),
    )
    op.create_index("ix_sources_document_key", "sources", ["document_key"])
    op.create_index("ix_sources_status", "sources", ["status"])
    op.create_table(
        "source_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.String(40), nullable=False, unique=True),
        sa.Column("section", sa.String(300), nullable=False, server_default=""),
        sa.Column("text", sa.Text(), nullable=False),
    )
    op.create_index("ix_source_chunks_source_id", "source_chunks", ["source_id"])


def downgrade() -> None:
    op.drop_index("ix_source_chunks_source_id", table_name="source_chunks")
    op.drop_table("source_chunks")
    op.drop_index("ix_sources_status", table_name="sources")
    op.drop_index("ix_sources_document_key", table_name="sources")
    op.drop_table("sources")
