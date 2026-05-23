"""
In-memory rate limiting via slowapi.

Limits:
  POST /tasks        — 10/minute  (expensive pipeline)
  POST /debate/run   — 5/minute   (3-round LLM debate)
  POST /approvals/*  — 30/minute  (human approval clicks)
  Global default     — 200/minute
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
