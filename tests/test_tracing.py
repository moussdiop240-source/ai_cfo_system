"""Tests for OpenTelemetry tracing setup — verifies no-op behavior when endpoint unset."""
import os
from unittest.mock import MagicMock, patch


class TestTracingNoOp:
    def test_get_tracer_returns_noop_when_no_endpoint(self):
        import sys
        for m in list(sys.modules):
            if "backend.tracing" in m:
                del sys.modules[m]
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": ""}, clear=False):
            from backend.tracing import get_tracer
            tracer = get_tracer("test")
            # Must support start_as_current_span without raising
            with tracer.start_as_current_span("test-span") as span:
                span.set_attribute("k", "v")  # no-op, no exception

    def test_setup_tracing_returns_false_when_no_endpoint(self):
        import sys
        for m in list(sys.modules):
            if "backend.tracing" in m:
                del sys.modules[m]
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": ""}, clear=False):
            import backend.tracing as t
            app = MagicMock()
            result = t.setup_tracing(app)
            assert result is False

    def test_agent_span_context_manager_noop(self):
        import sys
        for m in list(sys.modules):
            if "backend.tracing" in m:
                del sys.modules[m]
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": ""}, clear=False):
            from backend.tracing import agent_span
            # Must be usable as context manager without raising
            with agent_span("test_agent", task_id="task-123") as span:
                span.set_attribute("test", True)

    def test_noop_span_methods_do_not_raise(self):
        import sys
        for m in list(sys.modules):
            if "backend.tracing" in m:
                del sys.modules[m]
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": ""}, clear=False):
            from backend.tracing import _NoOpSpan
            span = _NoOpSpan()
            span.set_attribute("key", "value")
            span.set_status("ok")
            span.record_exception(Exception("test"))


class TestAgentSpanPropagation:
    def test_agent_span_reraises_exceptions(self):
        import sys
        for m in list(sys.modules):
            if "backend.tracing" in m:
                del sys.modules[m]
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": ""}, clear=False):
            from backend.tracing import agent_span
            import pytest
            with pytest.raises(ValueError):
                with agent_span("failing_agent"):
                    raise ValueError("agent error")
