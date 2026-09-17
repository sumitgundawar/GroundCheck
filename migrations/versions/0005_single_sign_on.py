"""Single sign-on: link users to an identity provider, and pending sign-ins.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db import EncryptedText, UTCDateTime

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("sso_issuer", sa.String(255), nullable=True))
        batch.add_column(sa.Column("sso_subject", sa.String(255), nullable=True))
        batch.create_unique_constraint("uq_users_sso", ["sso_issuer", "sso_subject"])
    op.create_table(
        "sso_logins",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("state_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("nonce", sa.String(64), nullable=False),
        sa.Column("code_verifier", EncryptedText("sso_logins.code_verifier"), nullable=False),
        sa.Column("next_path", sa.String(300), nullable=False, server_default="/"),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("expires_at", UTCDateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("sso_logins")
    with op.batch_alter_table("users") as batch:
        batch.drop_constraint("uq_users_sso", type_="unique")
        batch.drop_column("sso_subject")
        batch.drop_column("sso_issuer")
