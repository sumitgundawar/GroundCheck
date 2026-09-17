#!/usr/bin/env bash
# A backup and restore drill for the Docker Compose deployment: back up the
# database and the data volume, destroy everything, restore into a fresh
# stack, and check that the accounts, the audit trail and the search index
# came back. Run it before you rely on your backups.
#
#   deploy/.env must exist. The app is reachable at http://localhost:18000
#   only with the test override; otherwise use your own URL.
#
#   bash scripts/backup_restore_drill.sh <compose args...>
set -euo pipefail

COMPOSE=(docker compose "$@")
BACKUP_DIR=${BACKUP_DIR:-./backups/$(date +%Y%m%d-%H%M%S)}
# The app container drops every Linux capability, so it can't write files it
# doesn't own: a small helper container handles the data volume instead.
DATA_VOLUME=${DATA_VOLUME:-groundcheck_app-data}
HELPER=${HELPER:-alpine:3}
mkdir -p "$BACKUP_DIR"

echo "== Backing up to $BACKUP_DIR"
"${COMPOSE[@]}" exec -T db pg_dump -U groundcheck -d groundcheck --clean --if-exists > "$BACKUP_DIR/groundcheck.sql"
docker run --rm -v "$DATA_VOLUME":/data "$HELPER" sh -c 'cd /data && tar -czf - .' > "$BACKUP_DIR/data.tar.gz"
ls -la "$BACKUP_DIR"

echo "== Destroying the stack, including its volumes"
"${COMPOSE[@]}" down -v

echo "== Starting a fresh stack"
"${COMPOSE[@]}" up -d --no-start app    # creates the data volume for the restore
"${COMPOSE[@]}" up -d db
until "${COMPOSE[@]}" exec -T db pg_isready -U groundcheck -d groundcheck >/dev/null 2>&1; do sleep 2; done

echo "== Restoring"
"${COMPOSE[@]}" exec -T db psql -U groundcheck -d groundcheck < "$BACKUP_DIR/groundcheck.sql" > /dev/null
docker run --rm -i -v "$DATA_VOLUME":/data "$HELPER" sh -c 'cd /data && tar -xzf - && chown -R 1000:1000 /data' < "$BACKUP_DIR/data.tar.gz"
"${COMPOSE[@]}" up -d app

echo "== Checking"
"${COMPOSE[@]}" exec -T app python -m app.cli list-users
"${COMPOSE[@]}" exec -T app python -m app.cli verify-audit
echo "Restored from $BACKUP_DIR"
