# AI CFO System — Multi-Agent Financial Intelligence Platform

[![CI](https://github.com/moussdiop240-source/ai_cfo_system/actions/workflows/ci.yml/badge.svg)](https://github.com/moussdiop240-source/ai_cfo_system/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-941%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **Zero-hallucination financial analysis for CFOs, Controllers, and FP&A teams.**  
> Built on deterministic math engines, GAAP/IFRS compliance automation, and LangGraph multi-agent orchestration. Every number is exact. No LLM generates financial figures.

---

## Table of Contents

1. [Goal](#1-goal)
2. [Architecture](#2-architecture)
3. [Agent Roster](#3-agent-roster)
4. [Production Features](#4-production-features)
5. [API Reference](#5-api-reference)
6. [Quick Start](#6-quick-start)
7. [Configuration](#7-configuration)
8. [Deployment](#8-deployment)
9. [Security](#9-security)
10. [Observability](#10-observability)
11. [Testing](#11-testing)
12. [Company Datasets](#12-company-datasets)
13. [GAAP & IFRS Standards](#13-gaap--ifrs-standards)
14. [Project Structure](#14-project-structure)
15. [Debugging](#15-debugging)

---

## 1. Goal

The **AI CFO System** is a production-grade multi-agent financial intelligence platform for Chief Financial Officers, Controllers, and FP&A analysts.

| Capability | Detail |
|---|---|
| **Automated KPI computation** | 17 exact financial KPIs from raw accounting data |
| **GAAP compliance audit** | 12 ASC standards checked deterministically |
| **IFRS compliance audit** | 12 IASB standards checked deterministically |
| **Budget vs Actuals** | SAB 99 materiality flagging (≥5% threshold) |
| **Revenue forecasting** | Linear Regression + Holt-Winters ensemble (40/60) |
| **Anomaly detection** | IQR statistical method, no ML model needed |
| **Cash runway** | Burn rate + runway months from cash flow data |
| **AI debate** | GAAP agent vs IFRS agent structured 3-round LLM debate |
| **PDF board reports** | Download board-ready PDF from any completed analysis |
| **CFO dashboards** | 4 HTML dashboards per company, browser-native |
| **Institutional memory** | KPIs, insights, and HITL decisions persisted across periods |
| **HITL approvals** | Human-in-the-loop gate for anomalous analyses |
| **CSV/Excel ingestion** | Upload financial data directly from spreadsheets |

**Why zero-LLM math?** LLMs hallucinate numbers. A CFO cannot present a board with a gross margin figure that was "estimated" by a language model. Every metric in this system is computed by Pandas/NumPy arithmetic from source data — the LLM is used only for narrative interpretation and structured debate, never for calculation.

---

## 2. Architecture

### 4-Layer Anti-Hallucination Stack

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Deterministic Math Engine                             │
│  Pandas · NumPy · Scikit-learn · Statsmodels                     │
│  All KPIs, variances, forecasts, anomalies computed exactly.     │
│  Zero rounding errors. Reproducible to the cent.                 │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  LAYER 2 — Pydantic v2 Schema Validation                         │
│  AnalysisOutput · FinancialData schemas                          │
│  Rejects any LLM output that doesn't match the declared schema.  │
│  min_length / max_length enforced on all list fields.            │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  LAYER 3 — Secure Local Infrastructure                           │
│  FastAPI · SQLAlchemy · Alembic migrations · SHA-256 hashes      │
│  RBAC (5 roles) · Fernet field encryption · Rate limiting        │
│  IRS $75 Rule · $31M UNICAP threshold enforced.                  │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│  LAYER 4 — RAG Knowledge Retrieval                               │
│  Sentence-Transformers · pgvector (SQLite fallback)              │
│  Standards documents retrieved and cited in agent analysis.      │
│  Prevents LLM from fabricating regulatory citations.             │
└──────────────────────────────────────────────────────────────────┘
```

### LangGraph Pipeline

```
Raw Financial Data
        │
        ▼
[Supervisor] ──── deterministic routing via LangGraph conditional edges
        │
        ├──▶ [Data Agent]        validates & normalises input
        ├──▶ [Math Engine]       17 KPIs, variance, forecast, runway
        ├──▶ [RAG Agent]         retrieves regulatory context
        ├──▶ [GAAP Agent]        12 ASC standard checks
        ├──▶ [IFRS Agent]        12 IASB standard checks
        ├──▶ [Analysis Agent]    LLM narrative (Anthropic or Ollama)
        ├──▶ [HITL Node]         human approval gate (anomaly-triggered)
        └──▶ [Reporting Agent]   assembles final report + PDF
```

### Production Stack

```
Internet → nginx Ingress (TLS) → FastAPI (2 replicas, HPA 2–8)
                                      │
                         ┌────────────┴────────────┐
                    PostgreSQL                    Redis
                  (Alembic schema)            (rate limiter)
```

---

## 3. Agent Roster

| Agent | File | Role | Uses LLM |
|---|---|---|---|
| **Supervisor** | `backend/agents/supervisor.py` | Deterministic routing; OTel span per iteration | No |
| **Data Agent** | `backend/agents/data_agent.py` | Validates and normalises input | No |
| **Math Engine** | `backend/agents/math_engine.py` | All KPIs, variance, forecast | No |
| **GAAP Agent** | `backend/agents/gaap_agent.py` | 12 ASC standard checks | No |
| **IFRS Agent** | `backend/agents/ifrs_agent.py` | 12 IASB standard checks | No |
| **RAG Agent** | `backend/agents/rag_agent.py` | Retrieves regulatory context | No |
| **Analysis Agent** | `backend/agents/analysis_agent.py` | LLM narrative summary | **Yes** |
| **Debate Agent** | `backend/agents/debate_agent.py` | 3-round GAAP vs IFRS debate | **Yes** |
| **HITL Node** | `backend/agents/human_loop_node.py` | Human-in-the-loop approval gate | No |
| **Reporting Agent** | `backend/agents/reporting_agent.py` | Final report + PDF | No |

---

## 4. Production Features

### Security
| Feature | Detail |
|---|---|
| **RBAC** | 5-role hierarchy: `analyst < manager < vp < cfo < admin`; `X-API-Key` header; per-key expiry dates |
| **Field encryption** | Fernet symmetric encryption on `tasks.final_report`, `tasks.analysis_narrative`, `approvals.feedback`; backward-compatible |
| **Rate limiting** | slowapi: `POST /tasks` 10/min, `POST /debate/run` 5/min, `POST /approvals/*` 30/min, global 200/min |
| **CORS lockdown** | `CORS_ORIGINS` env var; defaults to `*` in dev, restrict in production |
| **Secrets validation** | Advisory startup check — warns on weak/missing `ADMIN_API_KEY`, errors on `LLM_BACKEND=anthropic` without key |
| **NetworkPolicy** | k8s policy restricts API pod ingress to ingress-nginx; egress to DB/Redis/Ollama/HTTPS only |

### Observability
| Feature | Detail |
|---|---|
| **Structured logging** | JSON log line per request with `request_id`, `method`, `path`, `status_code`, `latency_ms` |
| **`X-Request-ID`** | UUID header on every response; trace requests across services |
| **`GET /metrics`** | JSON: `uptime_seconds`, `requests_total`, `errors_total` |
| **`GET /metrics/prometheus`** | Prometheus text/plain 0.0.4 format; scrape-compatible |
| **`GET /health`** | DB liveness check; returns `503` if unreachable; includes `data_residency` field |
| **`GET /config/backend`** | Declares LLM backend, data residency, rate limit backend |
| **OpenTelemetry** | OTLP trace export when `OTEL_EXPORTER_OTLP_ENDPOINT` is set; pure no-op otherwise |

### Deployment
| Feature | Detail |
|---|---|
| **Docker** | Multi-stage Python 3.11-slim; non-root user (uid 1001); `/health` HEALTHCHECK |
| **Kubernetes** | Namespace, Deployment (init-container migration, liveness/readiness probes), Service, HPA (2–8 pods), ConfigMap, Ingress (nginx + TLS), NetworkPolicy, backup CronJob + PVC |
| **Alembic migrations** | Database schema versioned; init-container runs `alembic upgrade head` before app starts |
| **CD pipeline** | GitHub Actions `deploy` job pins the running image to `sha-<commit>`; waits for rollout; verifies `/health` |
| **Nightly backups** | CronJob runs `pg_dump` at 02:00 UTC; 7-day retention on a 10Gi PVC |

---

## 5. API Reference

All endpoints are documented interactively at `http://localhost:8000/docs`.

### Tasks

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/tasks` | — | Submit a financial analysis task (10/min rate limit) |
| `GET` | `/tasks/{id}` | — | Get task status and results |
| `GET` | `/tasks/{id}/stream` | — | SSE stream for real-time pipeline progress |
| `GET` | `/tasks/{id}/report` | — | Get final board report as JSON |
| `GET` | `/tasks/{id}/report/pdf` | — | Download board report as PDF |

### Approvals (HITL)

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/approvals/pending` | analyst+ | List tasks awaiting CFO approval |
| `POST` | `/approvals/{id}` | cfo+ | Submit approval/rejection decision (30/min) |

### Debate

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/debate/run` | — | Run 3-round GAAP vs IFRS debate (5/min) |

### RAG

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/rag/index` | — | Index a new document into the knowledge base |
| `POST` | `/rag/search` | — | Semantic search over the knowledge base |

### System

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | DB liveness + LLM status; `503` when DB unreachable |
| `GET` | `/metrics` | JSON runtime counters |
| `GET` | `/metrics/prometheus` | Prometheus text/plain scrape endpoint |
| `GET` | `/config/backend` | LLM backend, data residency, CORS origins |

---

## 6. Quick Start

### Option A — Windows One-Click Launcher

Double-click **`Launch AI CFO.bat`** (or `Launch AI CFO (Admin).bat` for UAC elevation).

The launcher automatically:
1. Verifies Python 3.10+ is installed
2. Creates an isolated `.venv` on first run
3. Installs all dependencies from `requirements.txt`
4. Starts the Streamlit UI and opens your browser at `http://localhost:8501`

### Option B — Manual (local dev)

```bash
git clone https://github.com/moussdiop240-source/ai_cfo_system.git
cd ai_cfo_system
pip install -r requirements.txt
cp .env.example .env          # edit with your API keys / settings
streamlit run streamlit_app.py
```

Streamlit UI opens at `http://localhost:8501`.

### Option C — Docker

```bash
docker build -t ai-cfo-system .
docker run -p 8000:8000 \
  -e LLM_BACKEND=ollama \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  ai-cfo-system
```

API at `http://localhost:8000` · Docs at `http://localhost:8000/docs`

### Option D — Kubernetes

See [Deployment](#8-deployment) and [`DEPLOYMENT.md`](DEPLOYMENT.md) for the full guide.

### Run the FastAPI Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### Run Tests

```bash
pytest tests/ --ignore=tests/test_integration_ollama.py -q
# 941 passed, 6 skipped
```

---

## 7. Configuration

Copy `.env.example` to `.env` and set values before running.

### LLM Backend

```env
# "ollama"     → 100% local, no API key, data never leaves the machine
# "anthropic"  → cloud, requires ANTHROPIC_API_KEY
# "auto"       → Anthropic if key is set, else Ollama
LLM_BACKEND=ollama
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-6
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### Database

```env
DATABASE_URL=sqlite:///./ai_cfo.db            # local dev (default)
# DATABASE_URL=postgresql://user:pass@host:5432/ai_cfo  # production
PGVECTOR_URL=                                  # leave blank for SQLite vector store
```

### Security

```env
# RBAC key map — JSON: {"key": "role"} or {"key": {"role": "cfo", "expires": "2027-12-31"}}
RBAC_KEYS=
ADMIN_API_KEY=change-me-to-at-least-32-random-characters

# CORS — comma-separated allowed origins; leave blank for * (dev only)
CORS_ORIGINS=https://yourapp.example.com

# Field encryption — Fernet key for database columns at rest
# Generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
FIELD_ENCRYPTION_KEY=
```

### Streamlit Auth

```env
# Option A: single shared password
STREAMLIT_PASSWORD=

# Option B: per-user bcrypt (takes precedence over STREAMLIT_PASSWORD)
# Generate hash: python -c "import bcrypt; print(bcrypt.hashpw(b'mypass', bcrypt.gensalt()).decode())"
STREAMLIT_USERS={"alice": "$2b$12$...", "bob": "$2b$12$..."}
```

### Rate Limiting & Observability

```env
REDIS_URL=redis://localhost:6379/0    # leave blank for in-memory (single instance)

# OpenTelemetry — leave blank to disable tracing (zero overhead)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_SERVICE_NAME=ai-cfo-system

LOG_LEVEL=INFO
```

---

## 8. Deployment

See **[`DEPLOYMENT.md`](DEPLOYMENT.md)** for the complete production deployment guide covering Docker, Kubernetes, TLS, and key rotation.

### Kubernetes Quick Reference

```bash
# 1. Create namespace first
kubectl apply -f k8s/namespace.yaml

# 2. Apply config and secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml          # copy from secret.yaml.example

# 3. Storage
kubectl apply -f k8s/backup-pvc.yaml

# 4. Application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml         # requires nginx + cert-manager
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/networkpolicy.yaml

# 5. Backup schedule
kubectl apply -f k8s/backup-cronjob.yaml
```

### CD Pipeline

The `deploy` job in `.github/workflows/ci.yml` runs automatically on every `master` merge when the `KUBECONFIG` secret is set in GitHub repo settings. It pins the deployment to the exact commit SHA and waits for rollout completion.

---

## 9. Security

### RBAC

Protect endpoints by adding the `X-API-Key` header. Keys are configured via `RBAC_KEYS` (JSON) or `ADMIN_API_KEY`.

```
analyst  → GET /approvals/pending
manager  → all analyst endpoints
vp       → all manager endpoints
cfo      → POST /approvals/{id}  (approve or reject)
admin    → all endpoints
```

Key expiry example:
```json
{"my-key": {"role": "cfo", "expires": "2027-12-31"}}
```

### Field Encryption

When `FIELD_ENCRYPTION_KEY` is set, the following columns are encrypted at rest with Fernet symmetric encryption:
- `tasks.final_report`
- `tasks.analysis_narrative`
- `approvals.feedback`

Existing plaintext rows are read back correctly after enabling encryption (backward-compatible).

### Key Rotation

To rotate the encryption key without downtime:

```bash
# Dry run — prints counts, touches nothing
DRY_RUN=1 \
  OLD_FIELD_ENCRYPTION_KEY=<old-key> \
  NEW_FIELD_ENCRYPTION_KEY=<new-key> \
  DATABASE_URL=postgresql://... \
  python scripts/rotate_encryption_key.py

# Execute
OLD_FIELD_ENCRYPTION_KEY=<old-key> \
NEW_FIELD_ENCRYPTION_KEY=<new-key> \
DATABASE_URL=postgresql://... \
  python scripts/rotate_encryption_key.py
```

The script is idempotent — rows already encrypted with the new key are skipped.

---

## 10. Observability

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "db": "ok", "llm_backend": "ollama", "data_residency": "local", ...}
```

Returns `503` if the database is unreachable.

### Metrics

```bash
curl http://localhost:8000/metrics
# {"uptime_seconds": 142.3, "requests_total": 87, "errors_total": 0, ...}

curl http://localhost:8000/metrics/prometheus
# Prometheus text/plain 0.0.4 — scrape-compatible
```

### Distributed Tracing

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to export traces to any OTLP-compatible backend (Jaeger, Grafana Tempo, Honeycomb, Datadog):

```env
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_SERVICE_NAME=ai-cfo-system
```

When unset, tracing is a pure no-op with zero overhead.

### Request Tracing

Every response includes an `X-Request-ID` UUID header. Pass it through to the backend for correlated log search:

```bash
curl -v http://localhost:8000/health 2>&1 | grep X-Request-ID
# X-Request-ID: a3f2c1d4-...
```

---

## 11. Testing

```bash
# Full suite (excludes live Ollama/Anthropic tests)
pytest tests/ --ignore=tests/test_integration_ollama.py -q
# 941 passed, 6 skipped

# With coverage
pytest tests/ --ignore=tests/test_integration_ollama.py \
  --cov=backend --cov-report=term-missing

# Specific suites
pytest tests/test_math_engine.py         # deterministic math
pytest tests/test_gaap_engine.py         # 12 ASC checks
pytest tests/test_ifrs_engine.py         # 12 IASB checks
pytest tests/test_rbac.py                # RBAC + key expiry
pytest tests/test_rate_limiting.py       # slowapi limits
pytest tests/test_field_encryption.py    # Fernet TypeDecorator
pytest tests/test_tracing.py             # OTel no-op behaviour
pytest tests/test_key_rotation.py        # rotation script
pytest tests/test_audit_comprehensive.py # SQL/prompt injection, RBAC matrix
pytest tests/test_performance.py         # SLO benchmarks
pytest tests/test_pdf_generation.py      # PDF board report
```

### CI Jobs

| Job | Trigger | What it does |
|---|---|---|
| `lint` | PR + master | ruff — E, F, W rules |
| `unit-tests` | after lint | 941 tests, SQLite, coverage report |
| `smoke-tests` | after unit-tests | deterministic end-to-end, no LLM |
| `integration-pgvector` | after unit-tests | RAG tests against real pgvector |
| `docker` | after lint | Dockerfile build (PR) + push to ghcr.io (master) |
| `security-scan` | after lint | bandit static analysis |
| `deploy` | after docker + tests | `kubectl set image` → rollout → health check (opt-in via `KUBECONFIG` secret) |

---

## 12. Company Datasets

Three pre-built datasets ship in `data/sample_companies.py`:

| Key | Company | Sector | Period | Revenue | GM | EBITDA |
|---|---|---|---|---|---|---|
| `novatech` | NovaTech Solutions Inc. | SaaS | Q1 2026 | $12.84M | 72.0% | 24.4% |
| `meridian` | Meridian Manufacturing Co. | Industrial | Q1 2026 | $28.40M | 30.0% | 12.0% |
| `horizon` | Horizon Community Foundation | Non-Profit 501(c)(3) | FY 2025 | $6.24M | 80.0% | 20.0% |

### Custom Company Minimum Fields

```python
{
    "_meta": {"name": "My Company", "sector": "SaaS", "period": "Q1 2026"},
    "revenue": 10_000_000,
    "cogs": 3_000_000,
    "gross_profit": 7_000_000,
    "ebitda": 2_000_000,
    "net_income": 1_200_000,
    "cash": 5_000_000,
    "current_assets": 8_000_000,
    "current_liabilities": 3_000_000,
    "total_assets": 20_000_000,
    "total_equity": 12_000_000,
    "budget": {"revenue": 9_500_000, "cogs": 3_200_000},
    "historical_revenue": [7.2e6, 7.8e6, 8.4e6, 9.1e6, 9.8e6, 10.0e6],
}
```

---

## 13. GAAP & IFRS Standards

### GAAP (ASC — 2026)

| Standard | Check | Key Threshold |
|---|---|---|
| ASC 205-40 | Going Concern | Cash runway < 12 months → disclosure required |
| ASC 230 | Cash Flows | Operating / investing / financing classification |
| ASC 260 | EPS | Basic and diluted EPS computed |
| ASC 280 | Segments | Revenue segments disclosed (≥10% rule) |
| ASC 310/326 | CECL Credit Losses | Allowance vs DSO, ECL model required |
| ASC 350 | Goodwill | Annual impairment test flag |
| ASC 450 | Contingencies | Probable loss accrual |
| ASC 606 | Revenue Recognition | Performance obligations, contract liability |
| ASC 740 | Income Taxes | Effective tax rate, deferred tax |
| ASC 820 | Fair Value | Level 1/2/3 hierarchy |
| ASC 842 | Leases | ROU asset + lease liability recognition |
| SAB 99 | Materiality | 5% threshold — items above flagged as Material |

### IFRS (IASB — 2026)

| Standard | Check | Key Difference from GAAP |
|---|---|---|
| IAS 1 | Presentation | Statement of comprehensive income required |
| IAS 2 | Inventories | **LIFO strictly prohibited** — FIFO or weighted avg only |
| IAS 7 | Cash Flows | Direct or indirect method |
| IAS 12 | Income Taxes | Deferred tax on all temporary differences |
| IAS 16 | PPE | Revaluation model allowed (vs cost-only under GAAP) |
| IAS 33 | EPS | Same as ASC 260 |
| IAS 36 | Impairment | CGU recoverable amount; reversals allowed |
| IAS 37 | Provisions | Constructive obligations included |
| IAS 38 | Intangibles | Development costs capitalised if criteria met |
| IFRS 9 | ECL | Expected credit loss 3-stage model |
| IFRS 15 | Revenue | Same 5-step model as ASC 606 |
| IFRS 16 | Leases | No operating lease exemption (unlike ASC 842) |

---

## 14. Project Structure

```
ai_cfo_system/
├── Dockerfile                        # Multi-stage Python 3.11-slim build
├── docker-compose.yml                # Local dev stack
├── streamlit_app.py                  # Streamlit UI (per-user bcrypt auth)
├── requirements.txt                  # Pinned dependencies
├── .env.example                      # All environment variables documented
├── Launch AI CFO.bat                 # Windows one-click launcher
├── Launch AI CFO (Admin).bat         # Windows launcher with UAC elevation
│
├── backend/
│   ├── main.py                       # FastAPI app, middleware, all core endpoints
│   ├── tracing.py                    # OpenTelemetry setup + agent_span()
│   │
│   ├── agents/
│   │   ├── supervisor.py             # LangGraph orchestrator + OTel spans
│   │   ├── math_engine.py            # FinancialCalculationEngine (zero LLM)
│   │   ├── gaap_agent.py
│   │   ├── ifrs_agent.py
│   │   ├── rag_agent.py
│   │   ├── analysis_agent.py         # LLM narrative (Anthropic / Ollama)
│   │   ├── debate_agent.py           # 3-round GAAP vs IFRS debate
│   │   ├── human_loop_node.py        # HITL approval gate
│   │   ├── reporting_agent.py        # Final report assembly
│   │   └── state.py                  # CFOAgentState TypedDict
│   │
│   ├── api/routes/
│   │   ├── tasks.py                  # POST/GET /tasks (rate-limited)
│   │   ├── approvals.py              # HITL approval endpoints (RBAC-gated)
│   │   ├── debate.py                 # POST /debate/run (rate-limited)
│   │   ├── rag.py                    # RAG index + search
│   │   └── stream.py                 # SSE streaming
│   │
│   ├── compliance/
│   │   ├── gaap_engine.py            # 12 ASC standard checks
│   │   └── ifrs_engine.py            # 12 IASB standard checks
│   │
│   ├── database/
│   │   ├── models.py                 # SQLAlchemy ORM (EncryptedText columns)
│   │   └── session.py                # Session management
│   │
│   ├── memory/                       # Institutional memory (4 tables)
│   │   ├── models.py                 # mem_company, mem_period_snapshot, etc.
│   │   └── engine.py                 # Memory accumulation logic
│   │
│   ├── middleware/
│   │   ├── rate_limiter.py           # slowapi (Redis-backed or in-memory)
│   │   └── request_logger.py         # JSON structured logging + X-Request-ID
│   │
│   ├── security/
│   │   ├── rbac.py                   # 5-role RBAC + key expiry
│   │   ├── field_encryption.py       # Fernet EncryptedText TypeDecorator
│   │   └── secrets_validator.py      # Startup advisory validation
│   │
│   ├── reporting/
│   │   └── pdf_generator.py          # fpdf2 board report PDF
│   │
│   ├── rag/
│   │   ├── pipeline.py               # Embedding + retrieval
│   │   └── knowledge_base.py         # 20 regulatory standards documents
│   │
│   └── llm/
│       └── adapter.py                # Anthropic / Ollama unified client
│
├── alembic/
│   ├── env.py                        # Wires AppBase + MemoryBase; render_as_batch
│   └── versions/
│       └── 0001_initial_schema.py    # All 7 tables baseline migration
│
├── k8s/
│   ├── namespace.yaml                # Create ai-cfo namespace first
│   ├── configmap.yaml                # Non-secret configuration
│   ├── secret.yaml.example           # Secret template (never commit secret.yaml)
│   ├── deployment.yaml               # 2 replicas, init-container migration, probes
│   ├── service.yaml                  # ClusterIP → port 80
│   ├── ingress.yaml                  # nginx + cert-manager TLS
│   ├── hpa.yaml                      # HPA: 2–8 replicas (CPU 70%, memory 80%)
│   ├── networkpolicy.yaml            # Ingress: ingress-nginx only; egress: DB/Redis/Ollama
│   ├── backup-pvc.yaml               # 10Gi PVC for nightly backups
│   └── backup-cronjob.yaml           # pg_dump at 02:00 UTC, 7-day retention
│
├── scripts/
│   ├── index_knowledge_base.py       # Index RAG documents into pgvector
│   └── rotate_encryption_key.py      # Atomic FIELD_ENCRYPTION_KEY rotation
│
├── dashboards/
│   └── html_generators.py            # 4 HTML dashboard generators
│
├── data/
│   ├── sample_companies.py           # NovaTech / Meridian / Horizon datasets
│   ├── novatech/                     # Generated HTML dashboards
│   ├── meridian/
│   └── horizon/
│
├── tests/                            # 941 tests, 0 ruff errors
│   ├── test_math_engine.py
│   ├── test_gaap_engine.py
│   ├── test_ifrs_engine.py
│   ├── test_agents.py
│   ├── test_rbac.py                  # RBAC matrix + key expiry
│   ├── test_rbac_expiry.py
│   ├── test_rate_limiting.py
│   ├── test_field_encryption.py      # Fernet TypeDecorator
│   ├── test_tracing.py               # OTel no-op behaviour
│   ├── test_key_rotation.py          # Rotation script
│   ├── test_health_observability.py  # /health, /metrics, X-Request-ID
│   ├── test_cors_prometheus.py
│   ├── test_llm_mocked_pipeline.py   # CI-safe mocked LLM pipeline
│   ├── test_pdf_generation.py
│   ├── test_performance.py           # SLO benchmarks
│   ├── test_audit_comprehensive.py   # Injection + RBAC matrix
│   └── test_integration_ollama.py    # Live LLM (skipped in CI)
│
└── .github/workflows/
    └── ci.yml                        # 7-job CI/CD pipeline
```

---

## 15. Debugging

### Installation

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'anthropic'` | `pip install -r requirements.txt` |
| `pip install` hangs | `pip install --timeout 120 -r requirements.txt` |
| `cryptography` build fails on Windows | Install Visual C++ Build Tools or use a wheel: `pip install cryptography --only-binary :all:` |

### Database

| Error | Fix |
|---|---|
| `no such table` | Tables are auto-created on startup. If using Alembic: `python -m alembic upgrade head` |
| `OperationalError` on ALTER TABLE | Ensure `render_as_batch=True` in `alembic/env.py` (required for SQLite) |

### LLM / Ollama

| Error | Fix |
|---|---|
| `httpx.ReadTimeout` | Model too large for CPU — use `OLLAMA_MODEL=llama3.2:1b` |
| `Connection refused` at `localhost:11434` | Start Ollama: `ollama serve` |
| `model not found` | `ollama pull llama3.2` |
| Anthropic `AuthenticationError` | Set `ANTHROPIC_API_KEY` in `.env` |
| Anthropic `402 Payment Required` | Add credits at console.anthropic.com or switch to Ollama |

### Security / Auth

| Error | Fix |
|---|---|
| `403 Forbidden` on `/approvals/pending` | Add `X-API-Key` header with an `analyst` or higher key |
| `403 API key expired or invalid` | Check `expires` date in `RBAC_KEYS`; update or remove expiry |
| Streamlit login loop | `STREAMLIT_USERS` JSON must be valid; test with `python -c "import json, os; json.loads(os.environ['STREAMLIT_USERS'])"` |

### Kubernetes

| Error | Fix |
|---|---|
| `namespaces "ai-cfo" not found` | Apply `k8s/namespace.yaml` first |
| Pods in `ImagePullBackOff` | Check `ghcr.io` package visibility; ensure CI `docker` job ran successfully on master |
| Backup CronJob pod fails | Verify `ai-cfo-backup-pvc` is `Bound`: `kubectl get pvc -n ai-cfo` |
| NetworkPolicy blocks internal traffic | Ensure your CNI supports NetworkPolicy (Calico/Cilium); not all clusters do |

### Performance Notes

- **Math engine + GAAP + IFRS**: < 1 second for all checks
- **Dashboard generation**: < 2 seconds for 4 HTML files
- **Ollama llama3.2:1b on CPU**: ~10–15 minutes for LLM narrative
- **Anthropic claude-sonnet-4-6**: ~10–20 seconds for LLM narrative
- **PDF generation**: < 500ms

---

## License

MIT — built for educational and professional use.

---

*Built with [Claude Code](https://claude.ai/code) · Anthropic claude-sonnet-4-6 · 941 tests · Zero hallucination guarantee on all numerical outputs*
