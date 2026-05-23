"""
OpenTelemetry distributed tracing.

- No-op when OTEL_EXPORTER_OTLP_ENDPOINT is not set (dev/local).
- Exports spans to any OTLP-compatible backend (Jaeger, Tempo, Honeycomb, etc.)
  when OTEL_EXPORTER_OTLP_ENDPOINT is set.
- FastAPI is auto-instrumented when setup_tracing(app) is called in main.py.
- Agents get per-node spans via get_tracer("ai_cfo.agents").

Environment variables:
  OTEL_EXPORTER_OTLP_ENDPOINT  e.g. http://localhost:4318
  OTEL_SERVICE_NAME             defaults to "ai-cfo-system"
"""
import logging
import os
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger("ai_cfo.tracing")

_tracer = None
_tracing_enabled = False


def setup_tracing(app) -> bool:
    """
    Configure OpenTelemetry and instrument the FastAPI app.
    Returns True if tracing is active, False if running in no-op mode.
    """
    global _tracer, _tracing_enabled

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        logger.info("OpenTelemetry: no OTEL_EXPORTER_OTLP_ENDPOINT set — tracing disabled (no-op)")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        service_name = os.environ.get("OTEL_SERVICE_NAME", "ai-cfo-system")
        resource = Resource(attributes={SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        _tracer = trace.get_tracer("ai_cfo.core")
        _tracing_enabled = True
        logger.info("OpenTelemetry: tracing enabled → %s (service=%s)", endpoint, service_name)
        return True

    except Exception as exc:
        logger.warning("OpenTelemetry: setup failed (%s) — continuing without tracing", exc)
        return False


def get_tracer(name: str = "ai_cfo"):
    """Return the configured tracer, or a no-op tracer if tracing is disabled."""
    if _tracing_enabled:
        from opentelemetry import trace
        return trace.get_tracer(name)

    class _NoOpTracer:
        @contextmanager
        def start_as_current_span(self, name, **kwargs):
            yield _NoOpSpan()

    return _NoOpTracer()


class _NoOpSpan:
    def set_attribute(self, key, value): pass
    def set_status(self, status): pass
    def record_exception(self, exc): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


@contextmanager
def agent_span(agent_name: str, task_id: Optional[str] = None):
    """Context manager for a single agent node span."""
    tracer = get_tracer("ai_cfo.agents")
    with tracer.start_as_current_span(f"agent.{agent_name}") as span:
        try:
            span.set_attribute("agent.name", agent_name)
            if task_id:
                span.set_attribute("task.id", task_id)
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise
