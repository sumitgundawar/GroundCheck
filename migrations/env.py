"""Alembic environment. Uses the app's engine, so migrations run against
DATABASE_URL. `app.db.migrate()` passes its own connection."""

from __future__ import annotations

from alembic import context

from app import db

target_metadata = db.Base.metadata


def run_migrations() -> None:
    connection = context.config.attributes.get("connection")
    if connection is not None:
        context.configure(connection=connection, target_metadata=target_metadata,
                          render_as_batch=connection.dialect.name == "sqlite")
        with context.begin_transaction():
            context.run_migrations()
        return
    with db.engine().connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata,
                          render_as_batch=conn.dialect.name == "sqlite")
        with context.begin_transaction():
            context.run_migrations()


run_migrations()
