# GroundCheckHealth Helm chart

Runs GroundCheckHealth on Kubernetes with the same hardening as the Docker Compose
deployment: a non-root, read-only container with no capabilities, probes on
`/healthz/live` and `/healthz/ready`, and data on a persistent volume that
uninstalling the chart never deletes.

## Before installing

1. **Build and push the image**, from the repository root:

   ```bash
   docker build -t registry.example.org/groundcheck:1.0.0 .
   docker push registry.example.org/groundcheck:1.0.0
   ```

2. **Create a PostgreSQL database** and a Secret with its URL:

   ```bash
   kubectl create secret generic groundcheck-db \
     --from-literal=DATABASE_URL='postgresql+psycopg://groundcheck:PASSWORD@postgres:5432/groundcheck'
   ```

3. **Create a Secret for the other sensitive settings.** Generate the keys
   with `python -m app.cli generate-key`, and keep a copy somewhere safe:
   without the encryption key, stored data can't be read.

   ```bash
   kubectl create secret generic groundcheck-secrets \
     --from-literal=DATA_ENCRYPTION_KEYS='...' \
     --from-literal=AUDIT_SIGNING_KEYS='...' \
     --from-literal=METRICS_TOKEN='...'
   ```

## Install

```bash
helm install groundcheck deploy/helm/groundcheck \
  --set image.repository=registry.example.org/groundcheck \
  --set database.existingSecret=groundcheck-db \
  --set existingSecret=groundcheck-secrets \
  --set ingress.enabled=true --set ingress.className=nginx \
  --set ingress.host=groundcheck.example.org
```

Plain settings go under `env` (see the README's Configuration table), for
example `--set env.OIDC_ISSUER=https://login.example.org`.

## Options

| Value | Default | Purpose |
| --- | --- | --- |
| `replicaCount` | `1` | More than one needs `persistence.accessMode=ReadWriteMany` and PostgreSQL. |
| `persistence.size` | `20Gi` | The index, trained models, imaging series and release snapshots. |
| `networkPolicy.enabled` | `false` | Allow traffic only from the ingress controller and Prometheus, and out only to DNS, the database and `networkPolicy.extraEgress`. |
| `metrics.serviceMonitor.enabled` | `false` | Scrape `/metrics` with the Prometheus Operator, using `METRICS_TOKEN`. |
| `metrics.prometheusRule.enabled` | `false` | Page on GroundCheckHealth's critical alerts, slow answers, and no instance up. |
| `podDisruptionBudget.enabled` | `false` | Keep an instance up during node maintenance, with several replicas. |

The chart refuses to render without a database Secret, and with several
replicas on a ReadWriteOnce volume.
