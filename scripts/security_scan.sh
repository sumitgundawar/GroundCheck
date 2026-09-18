#!/usr/bin/env bash
# Scan a running GroundCheck with OWASP ZAP: the passive baseline first, then
# the active API scan driven by the OpenAPI description the app publishes.
#
#   uvicorn app.main:app --port 8000
#   scripts/security_scan.sh http://127.0.0.1:8000
#
# Reports land in eval/security/. The script exits non-zero if either scan
# reports a failure, so it can gate a release.
#
# The active scan sends real requests, including ones designed to break things.
# Point it at a test instance, never at an instance holding patient data.

set -euo pipefail

TARGET="${1:-http://127.0.0.1:8000}"
OUT="${OUT_DIR:-eval/security}"
IMAGE="${ZAP_IMAGE:-ghcr.io/zaproxy/zaproxy:stable}"

command -v docker >/dev/null || { echo "This needs Docker to run ZAP." >&2; exit 2; }
curl -fs "$TARGET/healthz/ready" >/dev/null || { echo "Nothing is answering at $TARGET." >&2; exit 2; }

mkdir -p "$OUT"
# ZAP runs in its own container, so localhost has to be the host's localhost.
HOST_TARGET="${TARGET/127.0.0.1/host.docker.internal}"
HOST_TARGET="${HOST_TARGET/localhost/host.docker.internal}"

echo "Baseline scan (passive: headers, cookies, information leaks)"
docker run --rm --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/$OUT:/zap/wrk:rw" "$IMAGE" \
  zap-baseline.py -t "$HOST_TARGET" -J baseline.json -r baseline.html -I \
  | tee "$OUT/baseline.log" || true

echo
echo "API scan (active: injection, traversal, broken authorisation)"
docker run --rm --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/$OUT:/zap/wrk:rw" "$IMAGE" \
  zap-api-scan.py -t "$HOST_TARGET/openapi.json" -f openapi -J api.json -r api.html -I \
  | tee "$OUT/api.log" || true

echo
python3 - "$OUT" <<'PY'
import json, pathlib, sys

out = pathlib.Path(sys.argv[1])
total = 0
for name in ("baseline.json", "api.json"):
    path = out / name
    if not path.exists():
        print(f"{name}: no report written — the scan did not finish")
        total += 1
        continue
    report = json.loads(path.read_text())
    alerts = [a for site in report.get("site", []) for a in site.get("alerts", [])]
    # riskcode: 0 informational, 1 low, 2 medium, 3 high.
    serious = [a for a in alerts if int(a.get("riskcode", 0)) >= 2]
    print(f"{name}: {len(alerts)} alerts, {len(serious)} medium or high")
    for alert in serious:
        print(f"  {alert.get('riskdesc')}: {alert.get('name')} ({len(alert.get('instances', []))} places)")
    total += len(serious)
print(f"\n{total} finding(s) to fix" if total else "\nNo medium or high findings.")
sys.exit(1 if total else 0)
PY
