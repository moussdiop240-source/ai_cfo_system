"""
Rate limiting via slowapi.

Uses Redis-backed storage when REDIS_URL is set (production, multi-instance safe).
Falls back to in-memory storage when REDIS_URL is absent (local dev, single-instance).

Limits:
  POST /tasks        — 10/minute  (expensive pipeline)
  POST /debate/run   — 5/minute   (3-round LLM debate)
  POST /approvals/*  — 30/minute  (human approval clicks)
  Global default     — 200/minute
"""
import logging
import os

from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger("ai_cfo.rate_limiter")

_redis_url = os.environ.get("REDIS_URL", "")

if _redis_url:
    try:
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=["200/minute"],
            storage_uri=_redis_url,
        )
        logger.info("Rate limiter: Redis backend at %s", _redis_url.split("@")[-1])
    except Exception as exc:
        logger.warning("Rate limiter: Redis init failed (%s) — falling back to in-memory", exc)
        limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
else:
    limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
    logger.info("Rate limiter: in-memory (set REDIS_URL for multi-instance deployments)")
