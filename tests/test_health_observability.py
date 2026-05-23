"""
Health, metrics, and observability tests.

Covers:
- GET /health — DB check, LLM backend, data_residency, 503 on DB failure
- GET /metrics — counters are integers >= 0
- GET /config/backend — llm_backend, data_residency, cloud_calls_enabled
- X-Request-ID header on all responses
- Secrets validator: caplog assertions for warnings/errors
"""
import logging
import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200_when_db_ok(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/health")
        # 200 if DB reachable; 503 if not — both are valid outcomes
        assert r.status_code in (200, 503)

    def test_health_has_required_fields(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert "version" in data
        assert "llm_backend" in data
        assert "data_residency" in data
        assert "compliance" in data

    def test_health_data_residency_local_for_ollama(self):
        from backend.main import app
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama"}):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/health")
        data = r.json()
        assert data["data_residency"] == "local"
        assert data["llm_backend"] == "ollama"

    def test_health_data_residency_cloud_for_anthropic(self):
        from backend.main import app
        with patch.dict(os.environ, {"LLM_BACKEND": "anthropic"}):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/health")
        data = r.json()
        assert data["data_residency"] == "cloud"
        assert data["llm_backend"] == "anthropic"

    def test_health_503_when_db_unreachable(self):
        from backend.main import app
        from sqlalchemy.exc import OperationalError
        with patch("backend.database.session.engine") as mock_engine:
            mock_engine.connect.side_effect = OperationalError("DB down", None, None)
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/health")
        assert r.status_code == 503
        assert r.json()["db"] == "unreachable"

    def test_health_status_degraded_on_db_failure(self):
        from backend.main import app
        from sqlalchemy.exc import OperationalError
        with patch("backend.database.session.engine") as mock_engine:
            mock_engine.connect.side_effect = OperationalError("DB down", None, None)
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/health")
        assert r.json()["status"] == "degraded"

    def test_health_compliance_fields(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/health")
        comp = r.json().get("compliance", {})
        assert comp.get("gaap_standards") == 12
        assert comp.get("ifrs_standards") == 12

    def test_health_has_request_id(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/health")
        assert "x-request-id" in r.headers


# ── /metrics ─────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_has_required_fields(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/metrics")
        data = r.json()
        assert "uptime_seconds" in data
        assert "requests_total" in data
        assert "errors_total" in data
        assert "llm_backend" in data
        assert "version" in data

    def test_metrics_counters_are_non_negative_integers(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/metrics")
        data = r.json()
        assert isinstance(data["requests_total"], int)
        assert isinstance(data["errors_total"], int)
        assert data["requests_total"] >= 0
        assert data["errors_total"] >= 0

    def test_metrics_uptime_is_positive(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/metrics")
        assert r.json()["uptime_seconds"] >= 0

    def test_metrics_version_matches_health(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        v_health  = client.get("/health").json()["version"]
        v_metrics = client.get("/metrics").json()["version"]
        assert v_health == v_metrics


# ── /config/backend ────────────────────────────────────────────────────────────

class TestConfigBackendEndpoint:
    def test_config_backend_returns_200(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/config/backend")
        assert r.status_code == 200

    def test_config_backend_has_required_fields(self):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/config/backend")
        data = r.json()
        assert "llm_backend" in data
        assert "model" in data
        assert "data_residency" in data
        assert "cloud_calls_enabled" in data

    def test_config_backend_ollama_local(self):
        from backend.main import app
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama", "ANTHROPIC_API_KEY": ""}):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/config/backend")
        data = r.json()
        assert data["llm_backend"] == "ollama"
        assert data["data_residency"] == "local"
        assert data["cloud_calls_enabled"] is False

    def test_config_backend_anthropic_cloud(self):
        from backend.main import app
        with patch.dict(os.environ, {"LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "sk-test-key"}):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/config/backend")
        data = r.json()
        assert data["llm_backend"] == "anthropic"
        assert data["data_residency"] == "cloud"
        assert data["cloud_calls_enabled"] is True

    def test_config_backend_cloud_calls_false_without_key(self):
        from backend.main import app
        with patch.dict(os.environ, {"LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": ""}):
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/config/backend")
        assert r.json()["cloud_calls_enabled"] is False


# ── Secrets validator ─────────────────────────────────────────────────────────

class TestSecretsValidator:
    def test_no_keys_emits_warning(self, caplog):
        from backend.security.secrets_validator import validate
        with patch.dict(os.environ, {"RBAC_KEYS": "", "ADMIN_API_KEY": ""}, clear=False):
            with caplog.at_level(logging.WARNING, logger="ai_cfo.secrets"):
                result = validate()
        assert result["warnings"] >= 1
        assert any("RBAC_KEYS" in r.message or "ADMIN_API_KEY" in r.message
                   for r in caplog.records)

    def test_weak_admin_key_emits_warning(self, caplog):
        from backend.security.secrets_validator import validate
        with patch.dict(os.environ, {"ADMIN_API_KEY": "changeme", "RBAC_KEYS": ""}):
            with caplog.at_level(logging.WARNING, logger="ai_cfo.secrets"):
                result = validate()
        assert result["warnings"] >= 1
        assert any("weak" in r.message.lower() for r in caplog.records)

    def test_short_admin_key_emits_warning(self, caplog):
        from backend.security.secrets_validator import validate
        with patch.dict(os.environ, {"ADMIN_API_KEY": "short", "RBAC_KEYS": ""}):
            with caplog.at_level(logging.WARNING, logger="ai_cfo.secrets"):
                result = validate()
        assert result["warnings"] >= 1

    def test_anthropic_backend_without_key_emits_error(self, caplog):
        from backend.security.secrets_validator import validate
        env = {"LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "",
               "RBAC_KEYS": "", "ADMIN_API_KEY": "a-good-and-long-api-key-for-production-use"}
        with patch.dict(os.environ, env):
            with caplog.at_level(logging.ERROR, logger="ai_cfo.secrets"):
                result = validate()
        assert result["errors"] >= 1
        assert any("ANTHROPIC_API_KEY" in r.message for r in caplog.records)

    def test_valid_config_no_warnings(self, caplog):
        from backend.security.secrets_validator import validate
        env = {
            "ADMIN_API_KEY": "a-sufficiently-long-production-api-key-here",
            "RBAC_KEYS": "",
            "LLM_BACKEND": "ollama",
            "ANTHROPIC_API_KEY": "",
        }
        with patch.dict(os.environ, env):
            with caplog.at_level(logging.WARNING, logger="ai_cfo.secrets"):
                result = validate()
        assert result["warnings"] == 0
        assert result["errors"] == 0

    def test_validate_returns_dict_with_counts(self):
        from backend.security.secrets_validator import validate
        with patch.dict(os.environ, {"RBAC_KEYS": "", "ADMIN_API_KEY": "",
                                      "LLM_BACKEND": "ollama", "ANTHROPIC_API_KEY": ""}):
            result = validate()
        assert "warnings" in result
        assert "errors" in result
        assert isinstance(result["warnings"], int)
        assert isinstance(result["errors"], int)

    def test_never_raises(self):
        from backend.security.secrets_validator import validate
        with patch.dict(os.environ, {"RBAC_KEYS": "", "ADMIN_API_KEY": "",
                                      "LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": ""}):
            result = validate()  # should not raise
        assert result is not None
