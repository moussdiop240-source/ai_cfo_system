"""
Structured JSON request logging middleware.

Emits one log line per request:
  {"ts": "...", "method": "POST", "path": "/tasks", "status": 200,
   "latency_ms": 42.1, "request_id": "uuid4", "client_ip": "127.0.0.1"}

Attaches X-Request-ID response header for traceability.
LOG_LEVEL env var controls output (default INFO).
"""
import json
import logging
import os
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Module-level counters incremented by this middleware
requests_total: int = 0
errors_total: int = 0
_start_time: float = time.time()

logger = logging.getLogger("ai_cfo.requests")
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        global requests_total, errors_total

        request_id = str(uuid.uuid4())
        t0 = time.time()

        response: Response = await call_next(request)

        latency_ms = (time.time() - t0) * 1000
        requests_total += 1
        if response.status_code >= 500:
            errors_total += 1

        log_entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 1),
            "client_ip": request.client.host if request.client else "unknown",
        }
        logger.info(json.dumps(log_entry))

        response.headers["X-Request-ID"] = request_id
        return response


def get_uptime_seconds() -> float:
    return time.time() - _start_time
