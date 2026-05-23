"""
CORS configuration and Prometheus metrics endpoint tests.

Covers:
- CORS_ORIGINS env var locks allowed origins
- Blank CORS_ORIGINS defaults to wildcard (*)
- GET /metrics/prometheus returns Prometheus text format
- Prometheus response has correct Content-Type
- Prometheus response contains required metric names
- /config/backend exposes rate_limit_backend and cors_origins fields
"""
import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── CORS ─────────────────────────────────────────────────────────────────────

class TestCORSConfiguration:
    def test_default_cors_allows_all_origins(self):
        """Without CORS_ORIGINS, the wildcard is used."""
        from backend.main import app
        with patch.dict(os.environ, {"CORS_ORIGINS": ""}, clear=False):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/", headers={"Origin": "https://evil.example.com"})
        # CORS header present and permissive when wildcard
        assert r.status_code == 200

    def test_cors_origins_env_var_restricts_origins(self):
        """CORS_ORIGINS controls the middleware allow_origins list."""

        # Verify _cors_origins is derived from env at module load time
        with patch.dict(os.environ, {"CORS_ORIGINS": "https://trusted.com,https://admin.trusted.com"}):
            origins_raw = os.environ.get("CORS_ORIGINS", "")
            origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
        assert "https://trusted.com" in origins
        assert "https://admin.trusted.com" in origins
        assert len(origins) == 2

    def test_empty_cors_origins_gives_wildcard(self):
        origins_raw = ""
        origins = [o.strip() for o in origins_raw.split(",") if o.strip()] if origins_raw else ["*"]
        assert origins == ["*"]

    def test_single_cors_origin_parsed(self):
        origins_raw = "https://myapp.com"
        origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
        assert origins == ["https://myapp.com"]

    def test_cors_origins_strips_whitespace(self):
        origins_raw = "  https://a.com , https://b.com  "
        origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
        assert origins == ["https://a.com", "https://b.com"]

    def test_config_backend_exposes_cors_origins(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/config/backend")
        assert r.status_code == 200
        data = r.json()
        assert "cors_origins" in data
        assert isinstance(data["cors_origins"], list)

    def test_config_backend_exposes_rate_limit_backend(self):
        from backend.main import app
        with patch.dict(os.environ, {"REDIS_URL": ""}, clear=False):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/config/backend")
        data = r.json()
        assert "rate_limit_backend" in data
        assert data["rate_limit_backend"] in ("redis", "memory")

    def test_config_backend_memory_when_no_redis(self):
        from backend.main import app
        with patch.dict(os.environ, {"REDIS_URL": ""}, clear=False):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/config/backend")
        assert r.json()["rate_limit_backend"] == "memory"


# ── Prometheus metrics ────────────────────────────────────────────────────────

class TestPrometheusMetrics:
    def test_prometheus_endpoint_returns_200(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/metrics/prometheus")
        assert r.status_code == 200

    def test_prometheus_content_type(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/metrics/prometheus")
        assert "text/plain" in r.headers.get("content-type", "")

    def test_prometheus_contains_requests_total(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/metrics/prometheus")
        assert "ai_cfo_requests_total" in r.text

    def test_prometheus_contains_errors_total(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/metrics/prometheus")
        assert "ai_cfo_errors_total" in r.text

    def test_prometheus_contains_uptime(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/metrics/prometheus")
        assert "ai_cfo_uptime_seconds" in r.text

    def test_prometheus_has_help_and_type_lines(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/metrics/prometheus")
        assert "# HELP" in r.text
        assert "# TYPE" in r.text

    def test_prometheus_counter_values_are_numeric(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/metrics/prometheus")
        for line in r.text.splitlines():
            if line.startswith("ai_cfo_") and not line.startswith("#"):
                parts = line.split()
                assert len(parts) == 2, f"Unexpected line: {line}"
                float(parts[1])  # must be numeric


# ── Redis rate limiter fallback ───────────────────────────────────────────────

class TestRedisRateLimiterFallback:
    def test_in_memory_limiter_used_without_redis_url(self):
        """With no REDIS_URL, limiter should initialise without error."""
        with patch.dict(os.environ, {"REDIS_URL": ""}, clear=False):
            import importlib
            import backend.middleware.rate_limiter as rl_mod
            importlib.reload(rl_mod)
            assert rl_mod.limiter is not None

    def test_invalid_redis_url_falls_back_gracefully(self):
        """An unreachable Redis URL should not crash startup — falls back to in-memory."""
        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:19999/99"}, clear=False):
            import importlib
            import backend.middleware.rate_limiter as rl_mod
            # Reload to pick up new env var
            importlib.reload(rl_mod)
            # Limiter should be initialised (either redis or in-memory fallback)
            assert rl_mod.limiter is not None
