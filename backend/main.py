"""
AI CFO System — FastAPI entry point.
Production-grade, multi-agent financial intelligence platform.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .api.routes import approvals, debate, rag, stream, tasks
from .database.session import create_tables
from .middleware.rate_limiter import limiter
from .middleware.request_logger import (
    RequestLoggerMiddleware,
    get_uptime_seconds,
)
from .security.secrets_validator import validate as validate_secrets


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    validate_secrets()
    yield


app = FastAPI(
    title="AI CFO System",
    description=(
        "Production-grade multi-agent financial intelligence platform. "
        "GAAP (12 ASC standards) + IFRS (12 IAS/IFRS standards) compliance. "
        "LangGraph orchestration + RAG + HITL approvals."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware (order matters — first added = outermost wrapper) ───────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestLoggerMiddleware)
_cors_raw = os.environ.get("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()] if _cors_raw else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(tasks.router)
app.include_router(approvals.router)
app.include_router(debate.router)
app.include_router(stream.router)
app.include_router(rag.router)


# ── Core endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Enhanced health check — verifies DB connectivity, LLM backend status."""
    from .database.session import engine
    from sqlalchemy import text

    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    llm_backend = os.environ.get("LLM_BACKEND", "ollama").lower()
    data_residency = "local" if llm_backend == "ollama" else "cloud"

    body = {
        "status": "healthy" if db_ok else "degraded",
        "version": "1.0.0",
        "db": "ok" if db_ok else "unreachable",
        "llm_backend": llm_backend,
        "data_residency": data_residency,
        "compliance": {
            "gaap_standards": 12,
            "ifrs_standards": 12,
            "rag_documents": 20,
        },
    }
    status_code = 200 if db_ok else 503
    return JSONResponse(content=body, status_code=status_code)


@app.get("/metrics")
def metrics():
    """Runtime counters — requests, errors, uptime."""
    import backend.middleware.request_logger as _rl

    return {
        "uptime_seconds": round(get_uptime_seconds(), 1),
        "requests_total": _rl.requests_total,
        "errors_total": _rl.errors_total,
        "llm_backend": os.environ.get("LLM_BACKEND", "ollama"),
        "version": "1.0.0",
    }


@app.get("/config/backend")
def config_backend():
    """Data-residency declaration — shows whether data leaves the machine."""
    llm_backend = os.environ.get("LLM_BACKEND", "ollama").lower()
    model = os.environ.get("OLLAMA_MODEL", "llama3.2") if llm_backend == "ollama" \
        else os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {
        "llm_backend": llm_backend,
        "model": model,
        "data_residency": "local" if llm_backend == "ollama" else "cloud",
        "cloud_calls_enabled": bool(anthropic_key),
        "rate_limit_backend": "redis" if os.environ.get("REDIS_URL") else "memory",
        "cors_origins": _cors_origins,
    }


@app.get("/metrics/prometheus", response_class=Response)
def metrics_prometheus():
    """Prometheus text-format metrics endpoint (scrape-compatible)."""
    import backend.middleware.request_logger as _rl
    from fastapi.responses import Response

    uptime = round(get_uptime_seconds(), 1)
    lines = [
        "# HELP ai_cfo_requests_total Total HTTP requests processed",
        "# TYPE ai_cfo_requests_total counter",
        f"ai_cfo_requests_total {_rl.requests_total}",
        "# HELP ai_cfo_errors_total Total HTTP 5xx responses",
        "# TYPE ai_cfo_errors_total counter",
        f"ai_cfo_errors_total {_rl.errors_total}",
        "# HELP ai_cfo_uptime_seconds Seconds since process start",
        "# TYPE ai_cfo_uptime_seconds gauge",
        f"ai_cfo_uptime_seconds {uptime}",
        "",
    ]
    return Response(content="\n".join(lines), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/")
def root():
    return {
        "service": "AI CFO System",
        "architecture": "4-Layer: Deterministic Math → Schema Enforcers → Secure Infra → RAG",
        "endpoints": {
            "POST /tasks":                  "Submit financial analysis task",
            "GET  /tasks/{id}":             "Get task status + results",
            "GET  /tasks/{id}/stream":      "SSE stream for real-time updates",
            "GET  /tasks/{id}/report":      "Get final board report (JSON)",
            "GET  /tasks/{id}/report/pdf":  "Download board report as PDF",
            "GET  /approvals/pending":      "Get tasks awaiting CFO approval",
            "POST /approvals/{id}":         "Submit CFO approval decision",
            "POST /debate/run":             "Run 3-round IFRS vs GAAP debate",
            "POST /rag/index":              "Index new document",
            "POST /rag/search":             "Search knowledge base",
            "GET  /health":                 "System health (DB + LLM status)",
            "GET  /metrics":                "Runtime counters",
            "GET  /config/backend":         "LLM backend + data residency info",
        },
    }
