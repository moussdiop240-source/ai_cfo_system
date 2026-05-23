# AI CFO System — Failure Handling & Scalability Guide

Practical, system-specific guidance for operating the AI CFO pipeline reliably at production scale.
Every recommendation is grounded in the actual codebase — specific file and line references are included.

---

## Table of Contents

1. [Failure Taxonomy](#1-failure-taxonomy)
2. [Current Gaps & Immediate Fixes](#2-current-gaps--immediate-fixes)
3. [Retry & Circuit Breaker Patterns](#3-retry--circuit-breaker-patterns)
4. [Graceful Degradation](#4-graceful-degradation)
5. [Scalability Analysis by Layer](#5-scalability-analysis-by-layer)
6. [Production Tuning](#6-production-tuning)
7. [SLO Definitions](#7-slo-definitions)
8. [Capacity Planning](#8-capacity-planning)
9. [Incident Runbooks](#9-incident-runbooks)

---

## 1. Failure Taxonomy

Failures fall into five categories, ordered by blast radius.

### 1.1 LLM Backend Failures

| Failure | Trigger | Current behaviour | Risk |
|---|---|---|---|
| Ollama timeout | Model too large for CPU; host overloaded | `httpx.ReadTimeout` after 600 s; task → `error` | Pipeline blocked for up to 10 min per request |
| Ollama unreachable | Process not running | `Connection refused` immediately; task → `error` | All LLM-dependent agents fail |
| Anthropic 429 | API rate limit exceeded | Exception propagates; task → `error` | Burst traffic fails silently |
| Anthropic 402 | Credits exhausted | Exception propagates; task → `error` | All cloud-backend tasks fail permanently |
| Malformed LLM JSON | Model outputs free text | `instructor` retries 3×; then exception | Task fails after ~30 s of retries |
| Hallucinated numbers | Model fabricates figures | **Caught by Layer 1+2** — math engine runs first; LLM only writes narrative | No financial impact — by design |

### 1.2 Pipeline / Agent Failures

| Failure | Current guard | Gap |
|---|---|---|
| Agent throws unhandled exception | `errors > 5` → supervisor hard-stops (`supervisor.py:62`) | No per-agent retry; one transient error counts against the error budget |
| Infinite routing loop | `max_iterations=20` hard stop (`supervisor.py:65`) | State is lost on pod restart mid-pipeline |
| State corruption | LangGraph `MemorySaver` in-process checkpoint | Checkpoint lost on pod restart; task resumes from scratch |
| HITL approval timeout | None — waits indefinitely | Tasks stuck in `awaiting_approval` forever if approver is offline |

### 1.3 Database Failures

| Failure | Current behaviour | Gap |
|---|---|---|
| Connection pool exhaustion | SQLAlchemy default: 5 + 10 overflow = 15 max (`session.py`) | No pool size configured; high concurrency → `QueuePool limit exceeded` |
| PostgreSQL restart | `OperationalError` on next query | No automatic reconnect (`pool_pre_ping` not set) |
| SQLite file lock | Multiple writers block each other | SQLite is single-writer; breaks with `--workers 2+` |
| Migration race | Two init-containers race to run `alembic upgrade head` | `alembic_version` write conflict possible |
| Migration failure | Init-container fails; pod never becomes Ready | Rollout blocked until fixed manually |

### 1.4 Infrastructure Failures

| Failure | Current guard | Gap |
|---|---|---|
| Pod OOMKill | HPA limit: 2 Gi per pod | `sentence-transformers` load spikes to ~1.5 Gi; tight margin |
| Redis unavailable | Falls back to in-memory rate limiting (`rate_limiter.py:32`) | In-memory limits reset on pod restart; effective rate limit is multiplied by pod count |
| Ingress controller crash | None | Single ingress controller is a SPOF unless deployed in HA mode |
| Backup PVC full | None — `pg_dump` silently fails | Backups stop; data loss risk undetected |

### 1.5 Security / Operational Failures

| Failure | Risk |
|---|---|
| `FIELD_ENCRYPTION_KEY` lost | Encrypted rows permanently unreadable — no recovery path |
| `RBAC_KEYS` rotated without updating clients | All API clients receive 403 simultaneously |
| `ANTHROPIC_API_KEY` leaked in logs | Credential exposure — log scrubbing required |

---

## 2. Current Gaps & Immediate Fixes

### Gap 1 — No connection pool configuration

**Problem:** `backend/database/session.py` uses SQLAlchemy default pool settings (5 connections, 10 overflow). Under concurrent pipeline execution this exhausts immediately.

**Fix — `backend/database/session.py`:**

```python
from sqlalchemy.pool import NullPool, QueuePool

is_sqlite = "sqlite" in DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if is_sqlite else {},
    poolclass=NullPool if is_sqlite else QueuePool,
    pool_size=10,          # base connections per pod (PostgreSQL only)
    max_overflow=20,       # burst headroom
    pool_pre_ping=True,    # discard stale connections; handles DB restarts
    pool_recycle=1800,     # recycle every 30 min to prevent silent drops
)
```

**Impact:** Eliminates `QueuePool limit exceeded` under concurrent load. `pool_pre_ping` handles PostgreSQL restarts transparently.

---

### Gap 2 — No LLM retry on transient failure

**Problem:** A single `httpx.ConnectError` during an LLM call increments the error counter toward the 5-error hard stop (`supervisor.py:62`). One Ollama process restart fails an entire pipeline.

**Fix — wrap calls in `backend/llm/adapter.py`:**

```python
import time

def _with_retry(fn, max_attempts=3, base_delay=2.0):
    for attempt in range(max_attempts):
        try:
            return fn()
        except (httpx.TimeoutException, httpx.ConnectError):
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))   # 2 s, 4 s, 8 s
```

Usage:
```python
result = _with_retry(lambda: self._ollama_complete(prompt, system))
```

---

### Gap 3 — No HITL approval timeout

**Problem:** Tasks in `awaiting_approval` wait indefinitely. The queue grows unboundedly.

**Fix — `scripts/expire_stale_approvals.py` (run as CronJob):**

```python
from datetime import datetime, timedelta
from backend.database.session import get_db
from backend.database.models import Approval, Task

TIMEOUT_HOURS = 48

with get_db() as db:
    cutoff = datetime.utcnow() - timedelta(hours=TIMEOUT_HOURS)
    stale = db.query(Approval).filter(
        Approval.status == "pending",
        Approval.created_at < cutoff,
    ).all()
    for approval in stale:
        approval.status = "auto_rejected"
        approval.decision = "rejected"
        approval.feedback = f"Auto-rejected: no response within {TIMEOUT_HOURS} h."
        task = db.query(Task).get(approval.task_id)
        if task:
            task.status = "error"
            task.errors = ["HITL approval timed out"]
```

---

### Gap 4 — Alembic concurrent migration race

**Problem:** When a Deployment scales from 0→2 replicas both init-containers race to run `alembic upgrade head`. The second may fail with a write conflict on `alembic_version`.

**Fix — add a PostgreSQL advisory lock in `alembic/env.py`:**

```python
with connectable.connect() as connection:
    if "postgresql" in str(connection.engine.url):
        connection.execute(text("SELECT pg_advisory_lock(12345678)"))
    try:
        context.configure(connection=connection, ...)
        with context.begin_transaction():
            context.run_migrations()
    finally:
        if "postgresql" in str(connection.engine.url):
            connection.execute(text("SELECT pg_advisory_unlock(12345678)"))
```

---

### Gap 5 — Ollama 600 s timeout blocks background workers

**Problem:** `adapter.py:138` sets `read=600.0`. FastAPI `BackgroundTasks` shares the Uvicorn thread pool with request handling. A hung Ollama call for 10 minutes starves all other background work.

**Fix — reduce Ollama timeout and add a pipeline deadline:**

```python
# backend/llm/adapter.py — reduce read timeout
timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
```

```python
# backend/api/routes/tasks.py — pipeline-level deadline
import asyncio

async def _run_with_deadline(graph, state, config, timeout_s=300):
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, graph.invoke, state, config),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        return {**state, "status": "error",
                "errors": [f"Pipeline timed out after {timeout_s} s"]}
```

---

## 3. Retry & Circuit Breaker Patterns

### 3.1 LLM Circuit Breaker

Without a circuit breaker, every task queued during an Ollama outage burns through its error budget and fails. A circuit breaker opens after N consecutive failures and blocks further calls until the backend recovers.

```python
# backend/llm/circuit_breaker.py
import time
from threading import Lock

class CircuitBreaker:
    """CLOSED → OPEN (failing) → HALF_OPEN (recovery probe)."""

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at = None
        self._lock = Lock()

    @property
    def is_open(self):
        with self._lock:
            if self._opened_at is None:
                return False
            if time.time() - self._opened_at > self.recovery_timeout:
                return False  # half-open: allow one probe
            return True

    def record_success(self):
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self):
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.time()

_llm_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

def call_with_breaker(fn):
    if _llm_breaker.is_open:
        raise RuntimeError("LLM circuit breaker OPEN — backend unavailable")
    try:
        result = fn()
        _llm_breaker.record_success()
        return result
    except Exception:
        _llm_breaker.record_failure()
        raise
```

Wrap every `adapter.complete()` call: `call_with_breaker(lambda: adapter.complete(...))`.

**Tuning:** Start with `failure_threshold=5`, `recovery_timeout=60 s`. Ollama typically restarts in 15–30 s; 60 s gives ample recovery margin before the breaker closes.

### 3.2 Retry Decision Matrix

| Operation | Retry? | Max attempts | Backoff | Never retry |
|---|---|---|---|---|
| LLM `complete()` — network error | Yes | 3 | 2 s, 4 s, 8 s | 4xx auth errors |
| LLM `complete()` — timeout | Yes | 2 | 5 s | — |
| DB query — transient disconnect | Yes | 3 | 0.5 s, 1 s, 2 s | Schema errors, constraint violations |
| `pg_dump` CronJob | Yes (k8s restart) | 3 | 5 min | — |
| Alembic migration | No — fix forward | 1 | — | — |
| 429 rate limit response | No — surface to client | 1 | — | — |

---

## 4. Graceful Degradation

The 4-layer architecture is designed to degrade gracefully. Define explicit fallback behaviour at each layer:

### Degradation Ladder

```
Level 0  Full pipeline — math + GAAP/IFRS + RAG + LLM narrative
         ↓ LLM backend unavailable
Level 1  Math-only — deterministic summary, no LLM narrative
         ↓ Database unavailable
Level 2  Stateless compute — results returned but not persisted
         ↓ Everything unavailable
Level 3  Maintenance — HTTP 503 with Retry-After header
```

### Implementing Level 1 Fallback (Analysis Agent)

```python
# backend/agents/analysis_agent.py
try:
    narrative = adapter.complete(prompt)
except (RuntimeError, httpx.TimeoutException) as exc:
    kpis = state.get("kpi_metrics", {})
    narrative = (
        "[LLM unavailable — deterministic summary] "
        f"Gross margin: {kpis.get('gross_margin_pct', 'N/A')}%. "
        f"EBITDA margin: {kpis.get('ebitda_margin_pct', 'N/A')}%. "
        f"Current ratio: {kpis.get('current_ratio', 'N/A')}. "
        f"Full LLM analysis unavailable: {exc}"
    )
    state["warnings"].append(f"LLM degraded: {exc}")
```

### Degradation Signal in `/health`

```json
{
  "status": "degraded",
  "degradation_level": 1,
  "degradation_reason": "LLM circuit breaker OPEN",
  "available_features": ["kpi_computation", "gaap_check", "ifrs_check", "pdf_export"],
  "unavailable_features": ["ai_narrative", "debate"]
}
```

---

## 5. Scalability Analysis by Layer

### 5.1 Math Engine (Layer 1 — stateless, CPU-bound)

Scales linearly with CPU cores. No external dependencies.

| Concurrent tasks | Cores needed | Notes |
|---|---|---|
| 1–10 | 1 | Trivially handled |
| 10–100 | 2–4 | Increase `--workers` in Uvicorn |
| 100–1 000 | 4–8 + HPA | Consider Celery at this range — BackgroundTasks is not a job queue |
| 1 000+ | Celery + Redis | FastAPI BackgroundTasks has no persistence or retry |

**Key constraint:** FastAPI `BackgroundTasks` shares the Uvicorn thread pool with request handlers. One slow LLM call blocks all other background work in that worker. At >10 concurrent users, migrate to Celery.

### 5.2 Database (Layer 3)

**SQLite — use only for single-process local dev:**

| Constraint | Limit | Migration trigger |
|---|---|---|
| Concurrent writers | 1 (file lock) | >1 Uvicorn worker |
| Concurrent reads | Unlimited (WAL mode) | — |

**Switch to PostgreSQL immediately when running `--workers 2+`.**

**PostgreSQL connection pool sizing formula:**

```
pool_size     = num_workers × avg_concurrent_db_ops_per_worker
              = 2 workers × 5 = 10

max_overflow  = pool_size × 2 = 20

Total per pod = 30 connections
With 4 pods (HPA max) = 120 total connections

Set postgresql.conf:  max_connections = 200   (headroom above 120)
                      shared_buffers = 512MB  (25% of RAM on a 2 GB instance)
```

**Add pgBouncer when pods > 4:**

```yaml
# k8s/pgbouncer-deployment.yaml
pgbouncer:
  image: pgbouncer/pgbouncer:1.22
  env:
    DATABASES_HOST: postgres-service
    PGBOUNCER_POOL_MODE: transaction
    PGBOUNCER_MAX_CLIENT_CONN: "1000"
    PGBOUNCER_DEFAULT_POOL_SIZE: "25"
```

Point `DATABASE_URL` at pgBouncer. Each pod now uses pgBouncer's multiplexed pool instead of holding direct PostgreSQL connections.

### 5.3 RAG / Vector Store (Layer 4)

| Metric | SQLite (default) | pgvector |
|---|---|---|
| Documents indexed | ~50 K practical | Millions |
| Query latency (1 K docs) | 10–50 ms | 5–20 ms |
| Query latency (100 K docs) | 200–500 ms | 10–50 ms with HNSW index |

**Switch to pgvector when knowledge base exceeds 10 000 documents or RAG latency exceeds 100 ms.** Since PostgreSQL is already in the stack, this is zero additional infrastructure.

```sql
-- Add HNSW index for sub-millisecond search at scale
CREATE INDEX ON rag_documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

### 5.4 LLM Backend

**Ollama — local GPU scaling:**

Ollama does not support concurrent inference on one GPU. Run multiple instances on separate GPUs behind a load balancer:

```nginx
upstream ollama_pool {
    least_conn;
    server ollama-gpu-1:11434;
    server ollama-gpu-2:11434;
}
```

**Anthropic API tier guidance:**

| Tier | Tokens/min | RPM | Suitable for |
|---|---|---|---|
| Tier 1 | 40 K | 50 | Dev / <50 analyses/day |
| Tier 2 | 80 K | 1 000 | Small FP&A team (<200 analyses/day) |
| Tier 3 | 160 K | 2 000 | Mid-size teams |
| Tier 4+ | 400 K+ | 4 000+ | Enterprise |

At Tier 1, the existing 5/min debate rate limit is the correct constraint — without it, 10 simultaneous debates would saturate the API limit.

### 5.5 Rate Limiter

**In-memory (default):** Resets on pod restart. With 4 pods, each has an independent 200/min limit — effective global limit is 800/min, 4× the intended cap.

**Redis-backed (production requirement):** Set `REDIS_URL` to share state across all pods.

```env
REDIS_URL=redis://redis-sentinel:26379/0
```

**Redis sizing for rate limiting only:** A single `t3.micro` (1 GB RAM) handles >100 K ops/second for the counter workload. Rate limit state is trivially small.

---

## 6. Production Tuning

### 6.1 Uvicorn Workers

```bash
uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 2 \            # 2 workers for 2-vCPU pod; pipeline is blocking I/O
  --loop uvloop \          # pip install uvloop — 2-4× faster event loop
  --timeout-keep-alive 75
```

**Why 2 workers on 2 cores?** The LLM pipeline blocks threads. Two workers let HTTP request handling continue while one worker's thread pool is busy with a long-running pipeline.

### 6.2 HPA Thresholds

Current HPA targets CPU 70% and memory 80%. Recommended adjustments:

```yaml
metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60    # LLM tasks spike CPU briefly on load/unload
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70    # sentence-transformers baseline ~800 MB
```

### 6.3 Resource Requests / Limits

Actual baseline memory after model load:
- `sentence-transformers` model: ~800 MB
- Python runtime: ~150 MB
- FastAPI + SQLAlchemy: ~50 MB
- **Total: ~1 GB baseline**

Recommended `k8s/deployment.yaml` update:

```yaml
resources:
  requests:
    cpu: "500m"     # was 250m — insufficient for pipeline bursts
    memory: "1Gi"   # was 512Mi — below baseline; OOMKill risk
  limits:
    cpu: "2000m"
    memory: "3Gi"   # was 2Gi; headroom for model + concurrent tasks
```

### 6.4 Probe Timing

`sentence-transformers` loads on first request, not at startup (10–20 s cold load). Tighten probes accordingly:

```yaml
livenessProbe:
  initialDelaySeconds: 30    # was 20 — allow model load
  periodSeconds: 30
  failureThreshold: 3

readinessProbe:
  initialDelaySeconds: 20    # was 10
  periodSeconds: 10          # was 15 — more frequent during startup
  failureThreshold: 3        # was 2 — less aggressive removal from pool
```

### 6.5 Ingress — SSE Streaming

SSE streams (`GET /tasks/{id}/stream`) can run longer than the current 120 s `proxy-read-timeout`. Add a path-specific annotation:

```yaml
# k8s/ingress.yaml — add SSE annotation
nginx.ingress.kubernetes.io/proxy-read-timeout: "600"   # 10 min for SSE paths
nginx.ingress.kubernetes.io/proxy-buffering: "off"      # required for SSE
```

---

## 7. SLO Definitions

### Tier 1 — Deterministic Pipeline (no LLM)

| Operation | P50 | P95 | P99 | Error rate |
|---|---|---|---|---|
| Math engine (17 KPIs) | < 100 ms | < 500 ms | < 1 s | 0% |
| GAAP engine (12 checks) | < 50 ms | < 200 ms | < 500 ms | 0% |
| IFRS engine (12 checks) | < 50 ms | < 200 ms | < 500 ms | 0% |
| PDF generation | < 300 ms | < 1 s | < 2 s | < 0.1% |
| RAG retrieval (SQLite) | < 100 ms | < 300 ms | < 1 s | < 0.1% |
| `POST /tasks` (submit) | < 200 ms | < 500 ms | < 1 s | < 0.1% |
| `GET /tasks/{id}` | < 50 ms | < 200 ms | < 500 ms | 0% |

### Tier 2 — LLM Pipeline

| Operation | P50 | P95 | Hard timeout |
|---|---|---|---|
| Full pipeline (Anthropic) | < 30 s | < 90 s | 300 s |
| Full pipeline (Ollama CPU, llama3.2:1b) | < 10 min | < 20 min | 600 s |
| 3-round debate | < 60 s | < 180 s | 300 s |
| Analysis narrative only | < 20 s | < 60 s | 120 s |

### Availability Targets

| Endpoint / feature | Target | How to measure |
|---|---|---|
| `GET /health` | 99.9% (43 min/month downtime) | Synthetic probe every 30 s |
| `POST /tasks` (task submission) | 99.5% | Error rate from `/metrics/prometheus` |
| Full pipeline completion | 95% | `tasks.status = complete` / total submitted |
| HITL approval SLA | 99% within 48 h | Age of `approvals.status = pending` rows |

### Error Budget

At 99.5% availability for `POST /tasks`: **3.6 hours/month** allowable downtime.
Alert at **0.3% error rate** (60% of budget consumed) to allow investigation before the budget is exhausted.

---

## 8. Capacity Planning

### Small deployment — internal FP&A team (<50 analysts)

```
API pods:       2 (HPA min=2, max=4)
PostgreSQL:     t3.medium (2 vCPU, 4 GB), 10 Gi storage
Redis:          t3.micro (1 GB) — rate limiter only
LLM:            1 GPU node (RTX 3090 / A10) for Ollama  OR  Anthropic Tier 2
Daily analyses: 50
Concurrent:     5

Database sizing per year:
  tasks table:  50 rows/day × 365 = 18 K rows (~500 MB with encrypted blobs)
  mem_kpi_ts:   ~300 K rows — no action needed
```

### Mid-size deployment (<500 analysts)

```
API pods:       4–8 (HPA)
PostgreSQL:     r5.large (2 vCPU, 16 GB), 100 Gi + read replica
pgBouncer:      2 instances (connection multiplexing)
Redis:          Sentinel HA (3 nodes)
LLM:            2–4 GPU nodes  OR  Anthropic Tier 3
Task queue:     Celery + Redis (replace BackgroundTasks at this scale)

Migrate to Celery when:
  - >10 analyses submitted per minute
  - Users wait >5 min just for task acknowledgement
  - Pipeline failures need automatic retry
  - Task cancellation is required
```

### Institutional memory table growth

| Daily analyses | `mem_kpi_ts` rows/year | Action |
|---|---|---|
| 50 | 300 K | No action |
| 500 | 3 M | Add index on `(company_name, period)` |
| 5 000 | 30 M | Partition by year; archive cold data |

Partition SQL (add as an Alembic migration):
```sql
ALTER TABLE mem_kpi_ts PARTITION BY RANGE (recorded_at);
CREATE TABLE mem_kpi_ts_2026 PARTITION OF mem_kpi_ts
  FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
```

### When to migrate from BackgroundTasks to Celery

| Signal | Action |
|---|---|
| >10 analyses/min submitted | Migrate |
| Users wait >5 min for acknowledgement | Migrate |
| Pipeline failures need automatic retry | Migrate |
| Task cancellation needed | Migrate |
| Pod restart loses in-flight tasks | Migrate |

Celery migration touchpoints:
- `backend/api/routes/tasks.py` — replace `background_tasks.add_task(run_pipeline, ...)` with `run_pipeline.delay(...)`
- Add `backend/celery_app.py` with Redis broker
- Add `k8s/celery-worker-deployment.yaml`

---

## 9. Incident Runbooks

### INC-001 — All tasks failing with "LLM unavailable"

**Symptoms:** Tasks immediately entering `error` state; LLM probe in `/health` fails.

```bash
# 1. Check Ollama from inside the cluster
kubectl exec -n ai-cfo deployment/ai-cfo-api -- \
  curl -s http://ollama-service:11434/api/tags | python -m json.tool

# 2. Restart Ollama if unresponsive
kubectl rollout restart deployment/ollama -n ai-cfo

# 3. If model is not loaded
kubectl exec -n ai-cfo deployment/ollama -- ollama pull llama3.2

# 4. Switch to Anthropic fallback (if key is set)
kubectl set env deployment/ai-cfo-api LLM_BACKEND=anthropic -n ai-cfo
kubectl rollout status deployment/ai-cfo-api -n ai-cfo --timeout=120s

# 5. Verify recovery
curl https://ai-cfo.example.com/health
```

**Prevention:** Implement the circuit breaker (Section 3.1) so the system degrades to Level 1 automatically during Ollama outages.

---

### INC-002 — Connection pool exhaustion

**Symptoms:** `QueuePool limit of size X overflow Y reached`; `GET /tasks/{id}` returns 500; `/health` returns 503.

```bash
# 1. Check active DB connections
kubectl exec -n ai-cfo deployment/ai-cfo-api -- python -c "
from backend.database.session import engine
from sqlalchemy import text
with engine.connect() as c:
    n = c.execute(text(\"SELECT count(*) FROM pg_stat_activity WHERE state='active'\")).scalar()
    print('Active connections:', n)
"

# 2. Emergency: kill idle connections in PostgreSQL
# SELECT pg_terminate_backend(pid) FROM pg_stat_activity
# WHERE state = 'idle' AND query_start < NOW() - INTERVAL '5 minutes';

# 3. Temporarily reduce pod count to lower connection pressure
kubectl scale deployment/ai-cfo-api --replicas=1 -n ai-cfo

# 4. Apply Gap 1 fix (pool_pre_ping, pool_size=10, max_overflow=20), redeploy
kubectl rollout restart deployment/ai-cfo-api -n ai-cfo
```

**Prevention:** Apply the connection pool fix from Section 2, Gap 1 before going to production.

---

### INC-003 — Backup PVC full / backups silently failing

**Symptoms:** Recent CronJob history shows failures; `df -h /backups` shows 100% usage.

```bash
# 1. Trigger a test job to see the failure
kubectl create job backup-debug-$(date +%s) \
  --from=cronjob/ai-cfo-db-backup -n ai-cfo

# 2. Check PVC usage
kubectl exec -n ai-cfo <backup-pod> -- df -h /backups
kubectl exec -n ai-cfo <backup-pod> -- ls -lt /backups/ | head -20

# 3. Delete oldest backups manually
kubectl exec -n ai-cfo <backup-pod> -- \
  find /backups -name "*.sql.gz" -mtime +3 -delete

# 4. Expand PVC (if StorageClass supports volume expansion)
kubectl patch pvc ai-cfo-backup-pvc -n ai-cfo \
  -p '{"spec": {"resources": {"requests": {"storage": "20Gi"}}}}'
```

**Prevention:** Add a Prometheus alert on PVC usage > 80%:
```yaml
- alert: BackupPVCNearFull
  expr: kubelet_volume_stats_used_bytes{persistentvolumeclaim="ai-cfo-backup-pvc"}
        / kubelet_volume_stats_capacity_bytes{persistentvolumeclaim="ai-cfo-backup-pvc"} > 0.8
  for: 1h
  annotations:
    summary: "Backup PVC is {{ $value | humanizePercentage }} full"
```

---

### INC-004 — FIELD_ENCRYPTION_KEY lost or corrupted

**Symptoms:** Reads on `final_report`, `analysis_narrative`, or `feedback` return `InvalidToken` errors or garbled data.

**This is the most severe incident. Act immediately.**

```bash
# 1. Stop the application to prevent further writes with the wrong key
kubectl scale deployment/ai-cfo-api --replicas=0 -n ai-cfo

# 2. Restore the key from your secrets manager
#    (HashiCorp Vault / AWS Secrets Manager / GCP Secret Manager)
#    This is why FIELD_ENCRYPTION_KEY must be stored in a secrets manager,
#    not only in the Kubernetes Secret.

# 3. Verify the recovered key decrypts correctly
python -c "
from cryptography.fernet import Fernet
f = Fernet(b'<recovered-key>')
# Paste a known ciphertext from the DB
print(f.decrypt(b'<ciphertext>').decode()[:50])
"

# 4. Update the Kubernetes secret
kubectl create secret generic ai-cfo-secrets \
  --from-literal=FIELD_ENCRYPTION_KEY=<recovered-key> \
  --dry-run=client -o yaml | kubectl apply -f -

# 5. Restart
kubectl scale deployment/ai-cfo-api --replicas=2 -n ai-cfo
kubectl rollout status deployment/ai-cfo-api -n ai-cfo
```

**Prevention:**
- Store `FIELD_ENCRYPTION_KEY` in a dedicated secrets manager with versioning enabled.
- Rotate quarterly using `scripts/rotate_encryption_key.py` — a planned rotation is safer than an emergency one.
- Test decryption after every key rotation with a known plaintext.

---

### INC-005 — HPA scaled to 8 pods but latency still high

**Symptoms:** 6–8 pods running; tasks still queuing; CPU low; memory below limits.

**Root cause:** FastAPI `BackgroundTasks` thread pool is saturated by long-running LLM calls. More pods don't help if each pod's threads are all blocked.

```bash
# 1. Confirm threads are blocked (not CPU-bound)
kubectl exec -n ai-cfo deployment/ai-cfo-api -- python -c "
import threading
blocked = [t for t in threading.enumerate() if not t.is_alive() or 'worker' in t.name.lower()]
print(f'{len(threading.enumerate())} threads total')
"

# 2. Short-term: increase Uvicorn thread pool
kubectl set env deployment/ai-cfo-api UVICORN_WORKERS=4 -n ai-cfo

# 3. Long-term: migrate to Celery (see Section 8 capacity planning)
```

---

*Document version: 2026-05-23 — reflects codebase at master (commit 3e2cf32)*
*Update whenever pipeline architecture, infrastructure, or SLOs change.*
