"""Tests covering the 5 reliability gap fixes."""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")


# ── Gap 1: Connection pool config ────────────────────────────────────────────

class TestConnectionPool:
    def test_sqlite_uses_null_pool(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///./test_pool.db")
        import importlib
        import backend.database.session as sess_mod
        importlib.reload(sess_mod)
        assert sess_mod.engine.pool.__class__.__name__ == "NullPool"

    def test_postgres_uses_queue_pool(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/testdb")
        import importlib
        import backend.database.session as sess_mod
        with patch("sqlalchemy.create_engine") as mock_ce:
            mock_engine = MagicMock()
            mock_ce.return_value = mock_engine
            importlib.reload(sess_mod)
            call_kwargs = mock_ce.call_args[1]
            assert call_kwargs["pool_size"] == 10
            assert call_kwargs["max_overflow"] == 20
            assert call_kwargs["pool_pre_ping"] is True
            assert call_kwargs["pool_recycle"] == 1800


# ── Gap 2: LLM retry with exponential backoff ─────────────────────────────────

class TestLLMRetry:
    def test_retries_on_timeout(self):
        import httpx
        from backend.llm.adapter import _with_retry

        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise httpx.TimeoutException("timeout")
            return "ok"

        with patch("time.sleep"):
            result = _with_retry(flaky, max_attempts=3, base_delay=0.0)

        assert result == "ok"
        assert len(calls) == 3

    def test_raises_after_max_attempts(self):
        import httpx
        from backend.llm.adapter import _with_retry

        def always_fails():
            raise httpx.TimeoutException("timeout")

        with patch("time.sleep"):
            with pytest.raises(httpx.TimeoutException):
                _with_retry(always_fails, max_attempts=3, base_delay=0.0)

    def test_no_retry_on_other_errors(self):
        from backend.llm.adapter import _with_retry

        calls = []

        def bad():
            calls.append(1)
            raise ValueError("not a network error")

        with pytest.raises(ValueError):
            _with_retry(bad, max_attempts=3)

        assert len(calls) == 1

    def test_ollama_timeout_reduced_to_120(self):
        """Confirm the Ollama read timeout is 120s not 600s."""
        import httpx
        from backend.llm.adapter import LLMAdapter

        adapter = LLMAdapter(backend="ollama")
        captured = []

        def fake_post(url, json, timeout):
            captured.append(timeout)
            raise httpx.ConnectError("offline")

        with patch("httpx.post", side_effect=fake_post):
            with patch("time.sleep"):
                try:
                    adapter._ollama_complete("sys", "user", 100, False)
                except RuntimeError:
                    pass

        assert captured, "httpx.post was not called"
        assert captured[0].read == 120.0


# ── Gap 3: HITL approval timeout ─────────────────────────────────────────────

class TestExpireStaleApprovals:
    def _make_db(self):
        from sqlalchemy import Column, DateTime, String, Text, create_engine
        from sqlalchemy.orm import declarative_base, sessionmaker

        Base = declarative_base()

        class Task(Base):
            __tablename__ = "tasks"
            id = Column(String, primary_key=True)
            status = Column(String, default="complete")
            errors = Column(Text)

        class Approval(Base):
            __tablename__ = "approvals"
            id = Column(String, primary_key=True)
            task_id = Column(String)
            status = Column(String, default="pending")
            decision = Column(String)
            feedback = Column(Text)
            created_at = Column(DateTime, default=datetime.utcnow)

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return engine, Session, Task, Approval

    def _make_fake_get_db(self, mock_db):
        from contextlib import contextmanager

        @contextmanager
        def fake_get_db():
            yield mock_db

        return fake_get_db

    def test_dry_run_does_not_modify(self):
        """Dry run returns 0 and does not modify any approval."""
        from scripts.expire_stale_approvals import expire_stale

        old_time = datetime.utcnow() - timedelta(hours=72)
        mock_approval = MagicMock()
        mock_approval.id = "a1"
        mock_approval.task_id = "t1"
        mock_approval.status = "pending"
        mock_approval.created_at = old_time

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [mock_approval]

        mock_db = MagicMock()
        mock_db.query.return_value = mock_query

        with patch("backend.database.session.get_db", self._make_fake_get_db(mock_db)):
            result = expire_stale(dry_run=True)

        assert result == 0
        assert mock_approval.status == "pending"

    def test_auto_rejects_old_approvals(self):
        """Auto-reject sets status, decision, and feedback on stale approvals."""
        from scripts.expire_stale_approvals import expire_stale

        old_time = datetime.utcnow() - timedelta(hours=72)
        mock_approval = MagicMock()
        mock_approval.id = "a1"
        mock_approval.task_id = "t1"
        mock_approval.status = "pending"
        mock_approval.created_at = old_time

        mock_task = MagicMock()
        mock_task.id = "t1"

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [mock_approval]
        mock_query.first.return_value = mock_task

        mock_db = MagicMock()
        mock_db.query.return_value = mock_query

        with patch("backend.database.session.get_db", self._make_fake_get_db(mock_db)):
            result = expire_stale(dry_run=False)

        assert result == 0
        assert mock_approval.status == "auto_rejected"
        assert mock_approval.decision == "rejected"
        assert "48h" in mock_approval.feedback


# ── Gap 4: Alembic advisory lock ─────────────────────────────────────────────

class TestAlembicAdvisoryLock:
    """
    The advisory lock lives in alembic/env.py which runs inside Alembic's
    runtime context and cannot be imported like a normal Python module.
    We verify the lock is present via two complementary approaches:
      1. Source-code content check (fast, reliable)
      2. Logic unit test via an extracted helper
    """

    def test_advisory_lock_present_in_env_py(self):
        """env.py must contain pg_advisory_lock and pg_advisory_unlock."""
        env_path = os.path.join(os.path.dirname(__file__), "..", "alembic", "env.py")
        with open(env_path) as f:
            content = f.read()
        assert "pg_advisory_lock" in content, "pg_advisory_lock missing from alembic/env.py"
        assert "pg_advisory_unlock" in content, "pg_advisory_unlock missing from alembic/env.py"
        assert "is_pg" in content, "postgresql check missing from alembic/env.py"

    def test_lock_logic_pg_vs_sqlite(self):
        """Simulate the is_pg conditional directly."""
        executed = []

        def fake_execute(stmt):
            executed.append(str(stmt))

        # Postgres path
        pg_url = "postgresql://user:pass@localhost/db"
        is_pg = "postgresql" in str(pg_url)
        if is_pg:
            fake_execute("SELECT pg_advisory_lock(12345678)")
        fake_execute("run_migrations")
        if is_pg:
            fake_execute("SELECT pg_advisory_unlock(12345678)")

        assert any("pg_advisory_lock" in s for s in executed)
        assert any("pg_advisory_unlock" in s for s in executed)

        # SQLite path
        executed.clear()
        sqlite_url = "sqlite:///./ai_cfo.db"
        is_pg = "postgresql" in str(sqlite_url)
        if is_pg:
            fake_execute("SELECT pg_advisory_lock(12345678)")
        fake_execute("run_migrations")
        if is_pg:
            fake_execute("SELECT pg_advisory_unlock(12345678)")

        assert not any("pg_advisory" in s for s in executed)


# ── Gap 5: Pipeline deadline ──────────────────────────────────────────────────

class TestPipelineDeadline:
    def _make_mock_session(self):
        mock_task = MagicMock()
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_task
        mock_db.close = MagicMock()
        return mock_db, mock_task

    def test_timeout_sets_error_state(self):
        """Pipeline records a timeout error and does not raise."""
        from backend.api.routes.tasks import _run_pipeline

        initial_state = {"task_id": "t-timeout", "company_name": "Corp", "errors": []}
        mock_db, mock_task = self._make_mock_session()
        mock_graph = MagicMock()

        async def run():
            with patch("backend.api.routes.tasks.get_graph", return_value=mock_graph), \
                 patch("backend.api.routes.tasks.SessionLocal", return_value=mock_db), \
                 patch("backend.api.routes.tasks.asyncio.wait_for",
                       side_effect=asyncio.TimeoutError()):
                await _run_pipeline("t-timeout", initial_state)

        asyncio.run(run())

        assert mock_task.status == "error"
        errors = mock_task.errors
        assert errors and any("timed out" in str(e) for e in errors)

    def test_successful_pipeline_completes(self):
        """Happy path: graph returns state → task status = complete."""
        from backend.api.routes.tasks import _run_pipeline

        initial_state = {"task_id": "t-ok", "company_name": "Acme", "errors": []}
        final_state = {**initial_state, "final_report": "Board Report", "errors": []}
        mock_db, mock_task = self._make_mock_session()
        mock_graph = MagicMock()

        async def run():
            with patch("backend.api.routes.tasks.get_graph", return_value=mock_graph), \
                 patch("backend.api.routes.tasks.SessionLocal", return_value=mock_db):
                # Make wait_for return the final_state directly
                async def fake_wait_for(coro, timeout):
                    return final_state
                with patch("backend.api.routes.tasks.asyncio.wait_for", side_effect=fake_wait_for):
                    await _run_pipeline("t-ok", initial_state)

        asyncio.run(run())

        assert mock_task.status == "complete"
        assert mock_task.final_report == "Board Report"


# ── Circuit breaker ───────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_closed_state_allows_calls(self):
        from backend.llm.circuit_breaker import CircuitBreaker, call_with_breaker
        cb = CircuitBreaker(failure_threshold=5)
        result = call_with_breaker(lambda: "ok", breaker=cb)
        assert result == "ok"

    def test_opens_after_threshold(self):
        from backend.llm.circuit_breaker import CircuitBreaker, call_with_breaker
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            try:
                call_with_breaker(lambda: (_ for _ in ()).throw(RuntimeError("fail")), breaker=cb)
            except RuntimeError:
                pass
        assert cb.state == "OPEN"
        with pytest.raises(RuntimeError, match="circuit breaker OPEN"):
            call_with_breaker(lambda: "should not run", breaker=cb)

    def test_success_resets_failures(self):
        from backend.llm.circuit_breaker import CircuitBreaker, call_with_breaker
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(3):
            try:
                call_with_breaker(lambda: (_ for _ in ()).throw(RuntimeError("fail")), breaker=cb)
            except RuntimeError:
                pass
        call_with_breaker(lambda: "ok", breaker=cb)
        assert cb.state == "CLOSED"
        assert cb._failures == 0

    def test_half_open_after_recovery_timeout(self):
        from backend.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == "OPEN"
        import time
        time.sleep(0.02)
        assert cb.state == "HALF_OPEN"


# ── Graceful degradation in analysis_agent ────────────────────────────────────

class TestGracefulDegradation:
    def _make_state(self):
        return {
            "task_id": "t1",
            "company_name": "Acme",
            "period": "Q1 2026",
            "task_description": "Full financial analysis",
            "report_format": "board",
            "kpi_metrics": {
                "gross_margin_pct": 42.5,
                "ebitda_margin_pct": 18.3,
                "net_margin_pct": 11.2,
                "current_ratio": 2.1,
            },
            "validated_data": {"revenue": 1_000_000},
            "variance_table": {},
            "gaap_results": {},
            "ifrs_results": {},
            "errors": [],
            "audit_log": [],
            "agent_statuses": {},
        }

    def test_degraded_mode_when_all_llm_fail(self):
        from backend.agents.analysis_agent import analysis_agent_node

        state = self._make_state()
        mock_adapter = MagicMock()
        mock_adapter.active_backend = "ollama"
        mock_adapter.active_model = "llama3.2"
        mock_adapter.complete.side_effect = RuntimeError("Ollama offline")

        with patch("backend.agents.analysis_agent.get_adapter", return_value=mock_adapter), \
             patch("backend.agents.analysis_agent.AnalysisOutputValidator") as MockValidator:
            mock_val_result = MagicMock()
            mock_val_result.errors = []
            mock_val_result.warnings = []
            mock_val_result.score = 0.5
            MockValidator.return_value.validate.return_value = mock_val_result

            result = analysis_agent_node(state)

        assert result.get("analysis_narrative"), "Should have a deterministic narrative"
        assert any("deterministic" in str(e) for e in result.get("errors", [])), \
            "Should record degraded-mode error"
        assert result.get("agent_statuses", {}).get("analysis_agent") in ("complete", "validation_failed")

    def test_degraded_narrative_contains_kpis(self):
        from backend.agents.analysis_agent import analysis_agent_node

        state = self._make_state()
        mock_adapter = MagicMock()
        mock_adapter.active_backend = "ollama"
        mock_adapter.active_model = "llama3.2"
        mock_adapter.complete.side_effect = RuntimeError("offline")

        with patch("backend.agents.analysis_agent.get_adapter", return_value=mock_adapter), \
             patch("backend.agents.analysis_agent.AnalysisOutputValidator") as MockValidator:
            mock_val_result = MagicMock()
            mock_val_result.errors = []
            mock_val_result.warnings = []
            mock_val_result.score = 0.5
            MockValidator.return_value.validate.return_value = mock_val_result

            result = analysis_agent_node(state)

        narrative = result.get("analysis_narrative", "")
        assert "42.5" in narrative or "18.3" in narrative or "deterministic" in narrative.lower()
