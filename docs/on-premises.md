# Running GroundCheckHealth on your own servers

This guide is for IT teams installing GroundCheckHealth inside a hospital or clinic
network. It covers the architecture, installation, sign-in, keys, backups,
upgrades, monitoring and a hardening checklist.

GroundCheckHealth is not a certified medical device. Your organisation is
responsible for clinical safety sign-off (for example DCB0129 and DCB0160 in
England), a data protection impact assessment, and any regulatory approval
for how you use it.

## Architecture

```
  clinicians' browsers
          │ HTTPS (443)
  ┌───────▼────────┐
  │  proxy (Caddy) │  TLS, compression, request size limits
  └───────┬────────┘
          │ internal network, no internet access
  ┌───────▼────────┐      ┌──────────────┐
  │   app          │──────▶  PostgreSQL  │  users, audit trail, reviews, documents
  │  (GroundCheckHealth) │      └──────────────┘
  │                │──────▶ Ollama (optional)  local language model
  │                │──────▶ Qdrant (optional)  vector database for large collections
  └────────────────┘
      /data volume: search index, trained models, training runs
      /datasets (read-only): image folders for the training studio
```

`deploy/docker-compose.yml` defines this stack. The app container:

- runs as a non-root user with a read-only filesystem, no Linux capabilities
  and `no-new-privileges`
- sits on an internal network with no route to the internet: the embedding
  model is built into the image, and nothing is downloaded at runtime
- requires sign-in (`AUTH_REQUIRED=true`), sends session cookies over HTTPS
  only, and disables management from the server itself (`ADMIN_ACCESS=none`)
- sends a strict Content Security Policy, HSTS, and headers that stop
  framing, content sniffing and access to cameras or microphones
- loads no fonts, scripts or styles from other sites

## Sizing (starting points)

| Deployment | CPU | Memory | Disk | Notes |
| --- | --- | --- | --- | --- |
| Pilot, up to 50 users, documents only | 4 cores | 8 GB | 40 GB | Extractive answers, no local model |
| Department, local language model | 8 cores | 32 GB | 100 GB | Llama 3.2 3B or Qwen 2.5 7B on CPU is slow; prefer a GPU |
| With a GPU | 8 cores | 32 GB | 200 GB | NVIDIA with 12 GB or more for 7B models and image training |

PostgreSQL grows with the audit trail: roughly 20 KB per question, so
100,000 questions a year need about 2 GB.

**Measured capacity.** On an 8-core laptop with four workers and PostgreSQL,
extractive mode sustained **30 questions a second** over 100,000 requests with
no errors: median 0.7 s, 95th percentile 1.9 s, and the audit trail complete
and verified afterwards (100,002 records, no gaps). That was a deliberately
saturating test with 40 users asking without pause; a ward of 40 clinicians
asks far less often. Size for your peak minute, add workers for more
throughput (`WEB_WORKERS`), and remember that a language model, local or
cloud, is much slower than the extractive path.

## Install

On a Linux host with Docker Engine 24 or later and Docker Compose v2:

```bash
git clone https://github.com/sumitgundawar/GroundCheck.git
cd GroundCheckHealth
cp deploy/.env.example deploy/.env
```

Edit `deploy/.env`:

1. `GROUNDCHECK_HOSTNAME`: the name in DNS that people will use.
2. `TLS_SETTING`: `internal` for a certificate from Caddy's own authority (then
   install its root certificate, from the `caddy-data` volume at
   `/data/caddy/pki/authorities/local/root.crt`, on client machines through
   group policy), or the paths to your organisation's certificate and key.
3. `POSTGRES_PASSWORD`: a long random value.
4. `DATA_ENCRYPTION_KEYS` and `AUDIT_SIGNING_KEYS`: generate each with
   `docker compose -f deploy/docker-compose.yml run --rm --no-deps app python -m app.cli generate-key`.
   Store copies in your secrets manager before going further. Without the
   encryption key, encrypted data can't be recovered.
5. Single sign-on settings, if you use it (below).

Build and start:

```bash
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build
docker compose -f deploy/docker-compose.yml --env-file deploy/.env ps
```

The first start creates the database schema and builds the search index.
Then create the first admin:

```bash
docker compose -f deploy/docker-compose.yml --env-file deploy/.env exec app \
  python -m app.cli create-user --email admin@your-trust.nhs.uk --role admin
```

Keep this account as the emergency account, even when everyone else signs in
through single sign-on.

### A single container, without Compose

Compose is the supported install, because it brings PostgreSQL and HTTPS with
it. For a small site that is happy with SQLite behind an existing reverse
proxy, the image runs on its own — as long as every path that holds state is
pointed at the mounted volume. The plain `docker run` in the README writes
inside the container, so `docker rm` takes the data with it.

```bash
docker volume create groundcheck-data
docker run -d --name groundcheck -p 7860:7860 \
  -v groundcheck-data:/data \
  -e AUTH_REQUIRED=true -e ADMIN_ACCESS=none \
  -e DATABASE_URL=sqlite:////data/groundcheck.db \
  -e AUDIT_LOG_PATH=/data/audit/audit_log.jsonl \
  -e LOCAL_AI_STATE_PATH=/data/local_ai.json \
  -e INDEX_DIR=/data/index \
  -e MODEL_LIBRARY_DIR=/data/models/library \
  -e TRAINING_RUNS_DIR=/data/models/runs \
  -e IMAGING_DIR=/data/imaging \
  -e DATA_ENCRYPTION_KEYS=... -e AUDIT_SIGNING_KEYS=... \
  groundcheck:1.0.0
docker exec -it groundcheck python -m app.cli create-user \
  --email admin@your-trust.nhs.uk --role admin
```

The first start builds the search index into the empty volume, which takes a
couple of minutes on a CPU; the port stays closed until it finishes, so give
health checks a start period of five minutes. Later starts are ready in about
twenty seconds. Terminate TLS in front of it and set
`SESSION_COOKIE_SECURE=true`.

Metrics are refused from other machines unless `METRICS_TOKEN` is set; from the
host itself, `docker exec groundcheck python -c "import urllib.request;
print(urllib.request.urlopen('http://127.0.0.1:7860/metrics').read().decode())"`
works without one.

## Sign-in

Register GroundCheckHealth with your identity provider (Entra ID, Okta, Keycloak,
ADFS or any OpenID Connect provider) as a web application with the redirect
address `https://<GROUNDCHECK_HOSTNAME>/api/auth/sso/callback`. Create app
roles or groups for admins and reviewers, and set the `OIDC_*` values in
`deploy/.env`. See "Single sign-on" in the README for each setting.

Recommended: `OIDC_REQUIRE_MFA=true`, `OIDC_ALLOWED_DOMAINS` set to your
domains, and `PASSWORD_SIGN_IN=false` once single sign-on works.

## Scheduled jobs

Run these from the host's scheduler (cron or systemd timers):

| When | Command | Why |
| --- | --- | --- |
| Hourly | `python -m app.cli escalate-reviews` | Escalate overdue review cases |
| Daily | `python -m app.cli purge-sessions` | Remove expired sign-in sessions |
| Daily | `python -m app.cli verify-audit` | Exits with 2 if the audit trail has changed |
| Daily | `python -m app.cli audit-head` | Record the chain head in a system the database administrators can't change |
| Weekly | `python -m app.cli retention --apply` | Delete records past their retention period |

Run each as
`docker compose -f deploy/docker-compose.yml --env-file deploy/.env exec -T app <command>`.
Alert on a non-zero exit from `verify-audit`.

## Backups

Back up three things, and test restoring them at least every quarter:

1. **The database.** `docker compose ... exec -T db pg_dump -U groundcheck -Fc groundcheck > groundcheck-$(date +%F).dump`
   Encrypt backups and keep them off the host.
2. **The data volume** (`app-data`): the search index, trained models and
   training runs. The index can be rebuilt; trained models can't.
3. **The keys** in `deploy/.env`, kept separately from the backups they
   protect.

To restore, start a fresh stack with the same keys, then
`pg_restore -U groundcheck -d groundcheck --clean groundcheck-<date>.dump`
and copy the data volume back. Run `python -m app.cli verify-audit` afterwards.

`scripts/backup_restore_drill.sh` does the whole drill against a running
stack: back up, destroy it, restore into a fresh one, and check the accounts
and the audit trail. Run it on a test deployment, not on production.

```bash
bash scripts/backup_restore_drill.sh -f deploy/docker-compose.yml --env-file deploy/.env
```

It uses a small helper container for the data volume, because the app
container drops every Linux capability and can't write files it doesn't own.
Set `DATA_VOLUME` if your project name isn't `groundcheck`. Back up while the
app is healthy: a backup taken while it's still building its index on a new
volume contains no index.

## Upgrades

1. Read the release notes.
2. Back up the database and the data volume.
3. `git pull`, then `docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build`.
   Database migrations run automatically when the app starts.
4. Check `docker compose ... ps` shows the app as healthy, and run
   `python -m app.cli verify-audit`.

To roll back, restore the backup taken in step 2 and check out the previous
release.

A data volume created by a version before 1.0 may be owned by root, which
stops the app from writing to it. Fix it once:

```bash
docker compose -f deploy/docker-compose.yml --env-file deploy/.env run --rm --user root --entrypoint chown app -R 1000:1000 /data
```

## Kubernetes

A Helm chart is in `deploy/helm/groundcheck`, with the same hardening: a
non-root, read-only container, readiness and liveness probes, and a volume
uninstalling never deletes. It needs an existing PostgreSQL database and
Secrets for the database URL and keys. See `deploy/helm/README.md`. The
chart passes `helm lint` and Kubernetes 1.30 schema validation.

## Rotating keys

- **Encryption:** put the new key first in `DATA_ENCRYPTION_KEYS` and the old
  one in `DATA_ENCRYPTION_RETIRED_KEYS`, restart, run
  `python -m app.cli reencrypt`, then remove the old key and restart.
- **Audit signing:** put the new key first in `AUDIT_SIGNING_KEYS` and keep
  the old one after it, so older records still verify.
- **Database password and client secret:** change them in the database or
  identity provider and in `deploy/.env`, then restart.

## Monitoring

- `GET /healthz/ready` returns 200 when the database answers and the search
  index is loaded, and 503 otherwise; the compose file uses it as the
  container health check. `GET /healthz/live` answers while the process runs.
- `GET /metrics` serves Prometheus metrics to requests with
  `Authorization: Bearer <METRICS_TOKEN>`.
- The **Monitoring** page shows alerts on refusal rates, response times,
  drift from the documents, overdue reviews, expiring documents, the audit
  trail and failed releases. Set `ALERT_WEBHOOK_URL` to post them to Slack,
  Teams or a paging service.
- Caddy writes JSON access logs to standard output, and the app writes its logs
  there too: collect them with your log platform.
- The **Usage** page shows questions, refusals and response times; **Review**
  shows open and overdue cases; **Data protection** verifies the audit trail.

## Network access

At runtime, nothing leaves your network unless you configure it:

| Destination | Needed for | When |
| --- | --- | --- |
| Hugging Face | Embedding model | Image build only |
| Your identity provider | Single sign-on | Every sign-in |
| ollama.com registry | Downloading local models | Only when an admin downloads one |
| An OpenAI-compatible API (`GROQ_BASE_URL`) | Cloud language model | Only if `GROQ_API_KEY` is set; leave it empty to keep questions in your network |

To pull Ollama models on an internal network, download them on a connected
machine and copy the Ollama data volume across.

## Hardening checklist

- [ ] `AUTH_REQUIRED=true`, and single sign-on with MFA required
- [ ] `PASSWORD_SIGN_IN=false`, with one emergency local admin whose password is in a safe
- [ ] `DATA_ENCRYPTION_KEYS` and `AUDIT_SIGNING_KEYS` set, with copies in a secrets manager
- [ ] `GROQ_API_KEY` empty, unless a cloud model has been approved by information governance
- [ ] `INCLUDE_DEMO_CORPUS=false`, so only your approved documents are cited
- [ ] `TRUSTED_PROXIES` set to your load balancer, if one sits in front, so
      rate limiting and audit records attribute requests to the real caller
      rather than to the proxy
- [ ] `LOG_FORMAT=json` if you collect logs, and an alert on
      `groundcheck_recording == 0`, which means the instance is answering
      questions with nothing being written to the audit trail. That cannot be
      an in-app alert, because in-app alerts live in the database that is
      unavailable when it fires
- [ ] `API_DOCS=false`
- [ ] `METRICS_TOKEN` set, and `/metrics` reachable only from your monitoring system
- [ ] `ALERT_WEBHOOK_URL` pointing at a channel someone watches
- [ ] Sites created and everyone assigned, if you run more than one hospital or clinic
- [ ] `RELEASE_CHECKS=true`, so a document change is checked before it goes live
- [ ] Retention periods agreed with information governance and set
- [ ] The host's disk encrypted (LUKS, BitLocker or your storage's encryption)
- [ ] Only ports 80 and 443 open, and only from your clinical network
- [ ] Scheduled jobs running, with alerts on `verify-audit` failures
- [ ] Backups encrypted, off the host, and a restore tested
- [ ] `TRAINING_DATA_DIRS` limited to approved de-identified dataset folders
- [ ] Clinical safety case and hazard log reviewed and signed off
- [ ] Penetration test before go-live, and after major changes
- [ ] `pip-audit` and the safety evaluation re-run on the version you're deploying
