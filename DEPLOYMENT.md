# AI CFO System — Deployment Guide

Operational runbook for Docker and Kubernetes production deployments.

---

## Table of Contents

1. [Docker](#1-docker)
2. [Kubernetes — First Deployment](#2-kubernetes--first-deployment)
3. [Kubernetes — Ongoing Operations](#3-kubernetes--ongoing-operations)
4. [CD Pipeline](#4-cd-pipeline)
5. [Field Encryption Key Rotation](#5-field-encryption-key-rotation)
6. [Database Backups](#6-database-backups)
7. [TLS & Ingress](#7-tls--ingress)

---

## 1. Docker

### Build

```bash
docker build -t ai-cfo-system:latest .
```

The multi-stage build produces a ~350MB image (Python 3.11-slim base, non-root user uid 1001).

### Run (local dev)

```bash
docker run -p 8000:8000 \
  -e LLM_BACKEND=ollama \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e DATABASE_URL=sqlite:///./ai_cfo.db \
  ai-cfo-system:latest
```

### docker-compose (full stack)

```bash
docker-compose up -d
# Starts: api, postgres, redis, ollama
```

---

## 2. Kubernetes — First Deployment

### Prerequisites

- A cluster with a CNI that enforces NetworkPolicy (Calico, Cilium, or Weave)
- `kubectl` configured (`KUBECONFIG` pointing to the cluster)
- nginx ingress controller installed:
  ```bash
  kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.10.1/deploy/static/provider/cloud/deploy.yaml
  ```
- cert-manager installed (for automatic TLS):
  ```bash
  kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.5/cert-manager.yaml
  ```

### Apply Order

```bash
# 1. Namespace (must be first — everything else references it)
kubectl apply -f k8s/namespace.yaml

# 2. Configuration
kubectl apply -f k8s/configmap.yaml

# 3. Secrets — copy the template and fill in real values
cp k8s/secret.yaml.example k8s/secret.yaml
# Edit k8s/secret.yaml with your real values
kubectl apply -f k8s/secret.yaml

# 4. Storage
kubectl apply -f k8s/backup-pvc.yaml

# 5. Application
kubectl apply -f k8s/deployment.yaml     # runs alembic upgrade head as init-container
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/networkpolicy.yaml

# 6. Ingress (after nginx + cert-manager are ready)
# Edit k8s/ingress.yaml — replace ai-cfo.example.com with your real domain
kubectl apply -f k8s/ingress.yaml

# 7. Backup schedule
kubectl apply -f k8s/backup-cronjob.yaml
```

### Verify

```bash
kubectl get pods -n ai-cfo
# NAME                          READY   STATUS    RESTARTS
# ai-cfo-api-7d9f8b6c4-xk2pq   1/1     Running   0
# ai-cfo-api-7d9f8b6c4-mn8tz   1/1     Running   0

kubectl get ingress -n ai-cfo
# NAME              CLASS   HOSTS                   ADDRESS         PORTS
# ai-cfo-ingress    nginx   ai-cfo.example.com      203.0.113.42   80, 443

curl https://ai-cfo.example.com/health
# {"status": "healthy", "db": "ok", ...}
```

---

## 3. Kubernetes — Ongoing Operations

### Rolling Update (manual)

```bash
# After pushing a new image tag:
IMAGE="ghcr.io/moussdiop240-source/ai_cfo_system:sha-<commit>"
kubectl set image deployment/ai-cfo-api api="$IMAGE" migrate="$IMAGE" -n ai-cfo
kubectl rollout status deployment/ai-cfo-api -n ai-cfo --timeout=300s
```

### Rollback

```bash
kubectl rollout undo deployment/ai-cfo-api -n ai-cfo
kubectl rollout status deployment/ai-cfo-api -n ai-cfo
```

### Scale

```bash
# Manual scale (HPA will take over based on CPU/memory)
kubectl scale deployment/ai-cfo-api --replicas=4 -n ai-cfo
```

### View Logs

```bash
kubectl logs -l app=ai-cfo-api -n ai-cfo --follow
# Each line is a JSON structured log with request_id, method, path, status_code, latency_ms
```

### Run Database Migration Manually

```bash
kubectl run alembic-upgrade --rm -it --restart=Never \
  --image=ghcr.io/moussdiop240-source/ai_cfo_system:latest \
  --env-from=configmap/ai-cfo-config \
  --env-from=secret/ai-cfo-secrets \
  --namespace=ai-cfo \
  -- python -m alembic upgrade head
```

---

## 4. CD Pipeline

The GitHub Actions `deploy` job runs automatically on every `master` merge **when the `KUBECONFIG` secret is set**.

### Enable CD

1. Go to your repo → **Settings → Secrets and variables → Actions**
2. Add secret `KUBECONFIG` — paste the contents of your `~/.kube/config` (or a service-account kubeconfig scoped to the `ai-cfo` namespace)

### What the deploy job does

1. Writes `KUBECONFIG` secret to `~/.kube/config`
2. Runs `kubectl set image` to pin both containers to `sha-<commit>`
3. Waits up to 5 minutes for the rollout to complete
4. Verifies `/health` from inside the cluster via a temporary pod

### Disable CD (CI-only mode)

Leave the `KUBECONFIG` secret unset. The `deploy` job is skipped automatically.

---

## 5. Field Encryption Key Rotation

When `FIELD_ENCRYPTION_KEY` needs to change (key compromise, scheduled rotation), use the provided rotation script. It re-encrypts all sensitive columns in a single atomic transaction.

### Step-by-step

```bash
# 1. Generate a new key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# → save this as NEW_KEY

# 2. Dry run — prints counts, touches nothing
OLD_FIELD_ENCRYPTION_KEY=<current-key> \
NEW_FIELD_ENCRYPTION_KEY=<new-key> \
DATABASE_URL=postgresql://user:pass@host:5432/ai_cfo \
DRY_RUN=1 \
  python scripts/rotate_encryption_key.py

# Expected output:
# INFO Rotated 0 field(s), skipped 0 already on new key.  ← (dry run)

# 3. Execute rotation
OLD_FIELD_ENCRYPTION_KEY=<current-key> \
NEW_FIELD_ENCRYPTION_KEY=<new-key> \
DATABASE_URL=postgresql://user:pass@host:5432/ai_cfo \
  python scripts/rotate_encryption_key.py

# 4. Update the secret in Kubernetes
kubectl create secret generic ai-cfo-secrets \
  --from-literal=FIELD_ENCRYPTION_KEY=<new-key> \
  --dry-run=client -o yaml | kubectl apply -f -

# 5. Restart pods to pick up the new secret
kubectl rollout restart deployment/ai-cfo-api -n ai-cfo
```

**The script is idempotent** — rows already encrypted with the new key are detected and skipped. Safe to re-run if interrupted.

---

## 6. Database Backups

### Automatic (CronJob)

The `backup-cronjob.yaml` runs `pg_dump` daily at 02:00 UTC and stores compressed archives on a 10Gi PVC. Backups older than 7 days are pruned automatically.

```bash
# Trigger a manual backup
kubectl create job --from=cronjob/ai-cfo-db-backup manual-backup-$(date +%s) -n ai-cfo

# List backup files
kubectl exec -n ai-cfo $(kubectl get pods -n ai-cfo -l job-name -o name | head -1) \
  -- ls -lh /backups/
```

### Restore

```bash
# 1. Copy the backup file out of the PVC
kubectl cp ai-cfo/<backup-pod>:/backups/ai_cfo_20260523_020000.sql.gz ./restore.sql.gz

# 2. Decompress and restore
gunzip restore.sql.gz
psql postgresql://user:pass@host:5432/ai_cfo < restore.sql
```

### Adjust Retention

Edit `backup-cronjob.yaml`, line:
```bash
find /backups -name "*.sql.gz" -mtime +7 -delete
```
Change `+7` to the desired retention in days, then `kubectl apply -f k8s/backup-cronjob.yaml`.

---

## 7. TLS & Ingress

### cert-manager ClusterIssuer

Create a `ClusterIssuer` for Let's Encrypt before applying `ingress.yaml`:

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your@email.com
    privateKeySecretRef:
      name: letsencrypt-prod-account-key
    solvers:
      - http01:
          ingress:
            class: nginx
```

```bash
kubectl apply -f clusterissuer.yaml
```

### Staging (test first)

Use `letsencrypt-staging` for testing to avoid rate limits. Change `cert-manager.io/cluster-issuer: "letsencrypt-prod"` in `ingress.yaml` to `letsencrypt-staging`.

### Custom Domain

Edit `k8s/ingress.yaml` — replace every occurrence of `ai-cfo.example.com` with your domain before applying.
