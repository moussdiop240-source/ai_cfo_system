"""
LLM circuit breaker — prevents cascade failures when the LLM backend is down.

States:
  CLOSED   — normal operation; failures counted
  OPEN     — backend tripped; calls rejected immediately
  HALF_OPEN — one probe allowed after recovery_timeout; resets or re-opens

Usage:
    from backend.llm.circuit_breaker import call_with_breaker, _llm_breaker

    result = call_with_breaker(lambda: adapter.complete(system, user))
"""
import time
from threading import Lock


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "CLOSED"
            elapsed = time.time() - self._opened_at
            if elapsed > self.recovery_timeout:
                return "HALF_OPEN"
            return "OPEN"

    @property
    def is_open(self) -> bool:
        return self.state == "OPEN"

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.time()


_llm_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)


def call_with_breaker(fn, breaker: CircuitBreaker = _llm_breaker):
    """
    Call fn() through the circuit breaker.
    Raises RuntimeError immediately when the breaker is OPEN.
    Records success/failure and transitions state automatically.
    """
    if breaker.is_open:
        raise RuntimeError(
            "LLM circuit breaker OPEN — backend unavailable. "
            f"Will retry after {breaker.recovery_timeout}s."
        )
    try:
        result = fn()
        breaker.record_success()
        return result
    except Exception:
        breaker.record_failure()
        raise
