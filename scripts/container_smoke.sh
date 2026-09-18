#!/usr/bin/env bash
# Check a built image the way a site would run it: one container, everything
# it writes on a mounted volume, accounts required.
#
#   docker build -t groundcheck:test .
#   scripts/container_smoke.sh groundcheck:test
#
# It signs in, asks a question that must be refused and one that must be
# answered, asks the answered one again to see the cache, reads the monitoring
# and metrics endpoints, then restarts the container and checks the data and
# the audit chain came back. Exits non-zero on the first thing that is wrong.

set -euo pipefail

IMAGE="${1:-groundcheck:test}"
NAME="${CONTAINER_NAME:-groundcheck-smoke}"
VOLUME="${VOLUME_NAME:-groundcheck-smoke-data}"
PORT="${PORT:-8099}"
EMAIL="admin@example.org"
PASSWORD="Smoke-test-9x!Passw0rd"
WORK="$(mktemp -d)"
COOKIES="$WORK/cookies.txt"
BASE="http://127.0.0.1:${PORT}"

cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker volume rm -f "$VOLUME" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# A fresh volume, so this also covers a first install.
cleanup
mkdir -p "$WORK"
docker volume create "$VOLUME" >/dev/null

docker run -d --name "$NAME" -p "${PORT}:7860" -v "$VOLUME":/data \
  -e AUTH_REQUIRED=true -e ADMIN_ACCESS=none -e SESSION_COOKIE_SECURE=false \
  -e DATABASE_URL=sqlite:////data/groundcheck.db \
  -e AUDIT_LOG_PATH=/data/audit/audit_log.jsonl \
  -e LOCAL_AI_STATE_PATH=/data/local_ai.json \
  -e INDEX_DIR=/data/index \
  -e MODEL_LIBRARY_DIR=/data/models/library \
  -e TRAINING_RUNS_DIR=/data/models/runs \
  -e IMAGING_DIR=/data/imaging \
  "$IMAGE" >/dev/null

# The first start builds the search index into the empty volume, and the port
# stays closed until it finishes.
wait_ready() {
  local limit=$1 waited=0
  until curl -fs "$BASE/healthz/ready" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    [ "$waited" -ge "$limit" ] && fail "not ready after ${limit}s: $(docker logs --tail 20 "$NAME" 2>&1)"
    docker ps --filter "name=^${NAME}$" --filter status=running -q | grep -q . \
      || fail "container stopped: $(docker logs --tail 20 "$NAME" 2>&1)"
  done
  echo "ready after ${waited}s"
}
wait_ready 900

docker exec -i -e SMOKE_EMAIL="$EMAIL" -e SMOKE_PASSWORD="$PASSWORD" "$NAME" python - >/dev/null <<'PY'
import os
from app import auth, db
db.migrate()
auth.create_user(os.environ["SMOKE_EMAIL"], os.environ["SMOKE_PASSWORD"], role="admin", name="Smoke test")
PY

python3 - "$EMAIL" "$PASSWORD" > "$WORK/login.json" <<'PY'
import json, sys
print(json.dumps({"email": sys.argv[1], "password": sys.argv[2]}))
PY

sign_in() {
  curl -s -o /dev/null -w '%{http_code}' -c "$COOKIES" -X POST "$BASE/api/auth/login" \
    -H 'content-type: application/json' --data @"$WORK/login.json"
}
[ "$(sign_in)" = "200" ] || fail "sign-in failed"

# $1 is the question, $2 the file to write the response to.
ask() {
  python3 - "$1" > "$WORK/question.json" <<'PY'
import json, sys
print(json.dumps({"query": sys.argv[1]}))
PY
  curl -s -b "$COOKIES" -X POST "$BASE/api/ask" -H 'content-type: application/json' \
    --data @"$WORK/question.json" -o "$2"
}

ask "What is the recommended dose of Zalortin for a patient with Veltris syndrome?" "$WORK/refused.json"
python3 - "$WORK/refused.json" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("decision") != "refuse":
    sys.exit(f"FAIL: a medicine in no source was {d.get('decision')}, not refused")
print("refused a medicine that is in no source")
PY

ask "What is the first-line medication for Veltris syndrome?" "$WORK/answered.json"
python3 - "$WORK/answered.json" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("decision") != "answer":
    sys.exit(f"FAIL: expected an answer, got {d.get('decision')}")
claims = d.get("claims") or []
if not claims:
    sys.exit("FAIL: answered with no claims")
if not all(c.get("source_ids") for c in claims):
    sys.exit("FAIL: a claim has no source")
if not all(c.get("grounded") for c in claims):
    sys.exit("FAIL: a claim was not grounded in its sources")
print(f"answered with {len(claims)} grounded claims in {d.get('total_ms')} ms")
PY

ask "What is the first-line medication for Veltris syndrome?" "$WORK/repeat.json"
python3 - "$WORK/repeat.json" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("decision") != "answer":
    sys.exit("FAIL: the repeat question was not answered")
print(f"repeat question served in {d.get('total_ms')} ms")
PY

curl -s -b "$COOKIES" "$BASE/api/monitoring" -o "$WORK/monitoring.json"
python3 - "$WORK/monitoring.json" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
i = d.get("instance") or {}
missing = [k for k in ("version", "cpus", "memory_gb", "accelerator", "embed_batch", "web_workers")
           if i.get(k) is None]
if missing:
    sys.exit(f"FAIL: the hardware panel is missing {', '.join(missing)}")
print(f"instance: {i['cpus']} cpus, {i['memory_gb']} GB, {i['accelerator']}, version {i['version']}")
PY

docker exec -i "$NAME" python - <<'PY' || fail "metrics are wrong"
import urllib.request
body = urllib.request.urlopen("http://127.0.0.1:7860/metrics", timeout=10).read().decode()
assert "groundcheck_http_requests_total" in body, "no request metric"
assert 'route="/api/ask"' in body, "questions are not counted"
print("metrics served from the machine itself")
PY

code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/metrics")
[ "$code" = "403" ] || fail "metrics from another machine returned $code, not 403"

docker restart "$NAME" >/dev/null
wait_ready 300

[ "$(sign_in)" = "200" ] || fail "sign-in after the restart failed: the account didn't survive"

docker exec -i "$NAME" python - <<'PY' || fail "the audit trail did not survive the restart"
from app import audit, db, integrity
db.migrate()
backend = audit.store.backend()
assert backend == "database", f"audit went to {backend}, not the database"
result = integrity.verify()
assert result["ok"] and result["complete"], result
assert result["checked"] >= 3, f"only {result['checked']} audit records survived the restart"
print(f"{result['checked']} audit records, chain intact to sequence {result['last_seq']}")
PY

echo "Container smoke test passed for $IMAGE"
