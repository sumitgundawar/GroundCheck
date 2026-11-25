"""Documents belong to a site, or to none, which means shared by the group.

Revision ID: 0014
Revises: 0013
Create Date: 2026-11-25
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("sources") as batch:
        batch.add_column(sa.Column("site_id", sa.Integer(), nullable=True))
        batch.create_foreign_key("fk_sources_site_id", "sites", ["site_id"], ["id"], ondelete="SET NULL")
        batch.create_index("ix_sources_site_id", ["site_id"])


def downgrade() -> None:
    with op.batch_alter_table("sources") as batch:
        batch.drop_index("ix_sources_site_id")
        batch.drop_constraint("fk_sources_site_id", type_="foreignkey")
        batch.drop_column("site_id")
