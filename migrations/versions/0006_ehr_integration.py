"""EHR integration: pending SMART launches, and patients loaded from an EHR.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import EncryptedJSON, EncryptedText, UTCDateTime

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ehr_launches",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("state_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("issuer", sa.String(500), nullable=False),
        sa.Column("code_verifier", EncryptedText("ehr_launches.code_verifier"), nullable=False),
        sa.Column("token_endpoint", sa.String(500), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("expires_at", UTCDateTime(), nullable=False),
    )
    op.create_table(
        "ehr_contexts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("expires_at", UTCDateTime(), nullable=False),
        sa.Column("data", EncryptedJSON("ehr_contexts.data"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("ehr_contexts")
    op.drop_table("ehr_launches")
