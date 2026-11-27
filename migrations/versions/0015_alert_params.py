"""Alerts keep the values their text was written from.

Revision ID: 0015
Revises: 0014
Create Date: 2026-11-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("alerts") as batch:
        batch.add_column(sa.Column("params", sa.JSON(), nullable=False, server_default="{}"))


def downgrade() -> None:
    with op.batch_alter_table("alerts") as batch:
        batch.drop_column("params")
