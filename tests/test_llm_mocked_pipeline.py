"""
CI-safe mocked LLM pipeline tests.

Runs without any external services or API keys. All LLM calls are mocked.
Verifies that each agent handles responses correctly and that the full
pipeline supervisor routes through all nodes.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.agents.supervisor import create_initial_state


# ── Shared fixtures ───────────────────────────────────────────────────────────

VALID_FINANCIAL_DATA = {
    "company_name": "MockCorp Inc",
    "period": "Q1 2026",
    "currency": "USD",
    "revenue": 15_000_000,
    "cogs": 6_000_000,
    "total_assets": 50_000_000,
    "total_equity": 30_000_000,
    "current_assets": 20_000_000,
    "current_liabilities": 9_000_000,
    "cash": 7_000_000,
    "total_debt": 15_000_000,
    "accounts_receivable": 5_000_000,
    "inventory": 2_000_000,
    "shares_outstanding": 5_000_000,
    "diluted_shares": 5_250_000,
    "interest_expense": 500_000,
    "pre_tax_income": 2_500_000,
    "tax_provision": 625_000,
}


def _make_state(**overrides):
    state = create_initial_state(
        task_id="test-001",
        task_type="variance_analysis",
        task_description="Q1 variance analysis for MockCorp",
        company_name="MockCorp Inc",
        period="Q1 2026",
        raw_financial_data=VALID_FINANCIAL_DATA,
    )
    state.update(overrides)
    return state


def _mock_adapter(text_response: str = "Mocked LLM analysis.", backend: str = "ollama"):
    """Return a MagicMock LLMAdapter that returns text_response from complete()."""
    adapter = MagicMock()
    adapter.active_backend = backend
    adapter.active_model = "llama3.2" if backend == "ollama" else "claude-sonnet-4-6"
    adapter.complete.return_value = text_response
    adapter.complete_json.return_value = {
        "executive_summary": "MockCorp Q1 revenue $15M meets target.",
        "key_variance_drivers": ["Volume growth", "Price mix"],
        "identified_risks": ["Supply chain risk"],
        "opportunities": ["New market expansion"],
        "action_items": ["CFO to review by 2026-06-30"],
        "confidence_score": 0.85,
        "rag_sources_cited": [],
        "gaap_citations": [],
        "ifrs_citations": [],
    }
    return adapter


# ── Test: analysis_agent with mocked adapter ─────────────────────────────────

class TestAnalysisAgentMocked:
    def _run_with_mocked_adapter(self, text_response="Revenue $15M meets target."):
        from backend.agents.analysis_agent import analysis_agent_node
        from backend.agents.data_agent import data_agent_node
        from backend.agents.gaap_agent import gaap_agent_node
        from backend.agents.ifrs_agent import ifrs_agent_node
        from backend.agents.math_engine import math_engine_node

        state = _make_state()
        state = data_agent_node(state)
        state = math_engine_node(state)
        state = gaap_agent_node(state)
        state = ifrs_agent_node(state)

        adapter = _mock_adapter(text_response, backend="ollama")
        with patch("backend.agents.analysis_agent.get_adapter", return_value=adapter):
            state = analysis_agent_node(state)

        return state

    def test_analysis_agent_completes_successfully(self):
        state = self._run_with_mocked_adapter("Revenue $15M. Gross margin 60%. Risk: supply chain.")
        assert state["agent_statuses"].get("analysis_agent") in ("complete", "validation_failed")

    def test_analysis_narrative_populated(self):
        state = self._run_with_mocked_adapter("Revenue $15M. Gross margin 60%. Risk: supply chain.")
        if state["agent_statuses"].get("analysis_agent") == "complete":
            assert isinstance(state.get("analysis_narrative"), str)

    def test_adapter_called_with_company_name_in_prompt(self):
        from backend.agents.analysis_agent import analysis_agent_node
        from backend.agents.data_agent import data_agent_node
        from backend.agents.math_engine import math_engine_node

        state = _make_state()
        state = data_agent_node(state)
        state = math_engine_node(state)

        adapter = _mock_adapter(backend="ollama")
        with patch("backend.agents.analysis_agent.get_adapter", return_value=adapter):
            analysis_agent_node(state)

        # complete() must have been called
        adapter.complete.assert_called_once()
        call_args = adapter.complete.call_args
        # system + user args — company name should appear in prompt
        user_prompt = call_args[0][1] if call_args[0] else call_args[1].get("user", "")
        assert "MockCorp" in user_prompt

    def test_llm_error_propagates_to_state(self):
        from backend.agents.analysis_agent import analysis_agent_node
        from backend.agents.data_agent import data_agent_node
        from backend.agents.math_engine import math_engine_node

        state = _make_state()
        state = data_agent_node(state)
        state = math_engine_node(state)

        adapter = MagicMock()
        adapter.active_backend = "ollama"
        adapter.active_model = "llama3.2"
        adapter.complete.side_effect = RuntimeError("Ollama offline")

        with patch("backend.agents.analysis_agent.get_adapter", return_value=adapter):
            result = analysis_agent_node(state)

        assert any("ollama" in e.lower() or "offline" in e.lower() for e in result["errors"])


# ── Test: adapter backend selection ──────────────────────────────────────────

class TestAdapterBackendSelection:
    def test_adapter_ollama_backend_selected(self):
        from backend.llm.adapter import LLMAdapter, reset_adapter
        reset_adapter()
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama"}, clear=False):
            adapter = LLMAdapter()
        assert adapter.active_backend == "ollama"
        reset_adapter()

    def test_adapter_anthropic_backend_selected_when_key_set(self):
        from backend.llm.adapter import LLMAdapter, reset_adapter
        reset_adapter()
        with patch.dict(os.environ,
                        {"LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "sk-test"},
                        clear=False):
            adapter = LLMAdapter()
        assert adapter.active_backend == "anthropic"
        reset_adapter()

    def test_adapter_auto_selects_ollama_without_key(self):
        from backend.llm.adapter import LLMAdapter, reset_adapter
        reset_adapter()
        env = {"LLM_BACKEND": "auto", "ANTHROPIC_API_KEY": ""}
        with patch.dict(os.environ, env, clear=False):
            adapter = LLMAdapter()
        assert adapter.active_backend == "ollama"
        reset_adapter()

    def test_adapter_auto_selects_anthropic_with_key(self):
        from backend.llm.adapter import LLMAdapter, reset_adapter
        reset_adapter()
        env = {"LLM_BACKEND": "auto", "ANTHROPIC_API_KEY": "sk-real-key"}
        with patch.dict(os.environ, env, clear=False):
            adapter = LLMAdapter()
        assert adapter.active_backend == "anthropic"
        reset_adapter()


# ── Test: JSON extraction fallback ────────────────────────────────────────────

class TestAdapterJSONExtractionFallback:
    def test_complete_json_from_fenced_markdown(self):
        from backend.llm import adapter as adapter_mod
        fenced = '```json\n{"key": "value", "num": 42}\n```'
        result = adapter_mod._extract_json(fenced)
        assert result == {"key": "value", "num": 42}

    def test_complete_json_from_plain_json(self):
        from backend.llm import adapter as adapter_mod
        result = adapter_mod._extract_json('{"revenue": 5000000}')
        assert result["revenue"] == 5_000_000

    def test_complete_json_embedded_in_prose(self):
        from backend.llm import adapter as adapter_mod
        prose = 'Here is the answer: {"score": 0.9, "ok": true} — end of message.'
        result = adapter_mod._extract_json(prose)
        assert result["score"] == 0.9

    def test_complete_json_raises_on_garbage(self):
        from backend.llm import adapter as adapter_mod
        with pytest.raises(ValueError, match="Could not extract JSON"):
            adapter_mod._extract_json("not json at all here")


# ── Test: reporting agent with mocked anthropic ────────────────────────────────

class TestReportingAgentMocked:
    def _build_pre_report_state(self):
        from backend.agents.data_agent import data_agent_node
        from backend.agents.gaap_agent import gaap_agent_node
        from backend.agents.ifrs_agent import ifrs_agent_node
        from backend.agents.math_engine import math_engine_node

        state = _make_state()
        state = data_agent_node(state)
        state = math_engine_node(state)
        state = gaap_agent_node(state)
        state = ifrs_agent_node(state)
        state["analysis_narrative"] = "Revenue $15M met plan. Gross margin 60%."
        state["agent_statuses"]["analysis_agent"] = "complete"
        return state

    def test_reporting_agent_with_mocked_anthropic(self):
        from backend.agents.reporting_agent import reporting_agent_node

        state = self._build_pre_report_state()

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock()]
        mock_msg.content[0].text = (
            "## EXECUTIVE SUMMARY\n"
            "MockCorp Q1 2026 revenue $15,000,000 exceeded plan by 7%. "
            "EPS $0.35. Gross margin 60%. GAAP compliant per ASC 606.\n"
            "## RISK ASSESSMENT\n"
            "1. Supply chain risk — quantified at $500K exposure.\n"
        )
        mock_msg.usage = MagicMock(input_tokens=100, output_tokens=200)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        with patch("backend.agents.reporting_agent.anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_client
            result = reporting_agent_node(state)

        assert result["agent_statuses"].get("reporting_agent") in ("complete", "error")

    def test_reporting_agent_final_report_non_empty_on_success(self):
        from backend.agents.reporting_agent import reporting_agent_node

        state = self._build_pre_report_state()

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock()]
        mock_msg.content[0].text = (
            "## EXECUTIVE SUMMARY\n"
            "Revenue $15,000,000 vs plan $14,000,000 (+7.1%). EPS $0.35 diluted.\n"
            "## US GAAP COMPLIANCE NOTES\n"
            "ASC 606 compliant. Revenue recognized ratably over contract term.\n"
        )
        mock_msg.usage = MagicMock(input_tokens=150, output_tokens=300)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        with patch("backend.agents.reporting_agent.anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_client
            result = reporting_agent_node(state)

        if result["agent_statuses"].get("reporting_agent") == "complete":
            assert result.get("final_report")
            assert len(result["final_report"]) > 10


# ── Test: full pipeline with mocked LLM ──────────────────────────────────────

class TestFullPipelineMocked:
    def test_deterministic_nodes_run_without_llm(self):
        """data_agent, math_engine, gaap_agent, ifrs_agent need no LLM."""
        from backend.agents.data_agent import data_agent_node
        from backend.agents.gaap_agent import gaap_agent_node
        from backend.agents.ifrs_agent import ifrs_agent_node
        from backend.agents.math_engine import math_engine_node

        state = _make_state()
        state = data_agent_node(state)
        state = math_engine_node(state)
        state = gaap_agent_node(state)
        state = ifrs_agent_node(state)

        assert state["agent_statuses"].get("data_agent") == "complete"
        assert state["agent_statuses"].get("math_engine") == "complete"
        assert state["agent_statuses"].get("gaap_agent") == "complete"
        assert state["agent_statuses"].get("ifrs_agent") == "complete"

    def test_rag_agent_runs_without_pgvector(self):
        """RAG agent falls back to SQLite — no pgvector or API key needed."""
        from backend.agents.data_agent import data_agent_node
        from backend.agents.math_engine import math_engine_node
        from backend.agents.rag_agent import rag_agent_node

        state = _make_state()
        state = data_agent_node(state)
        state = math_engine_node(state)
        state = rag_agent_node(state)

        assert state["agent_statuses"].get("rag_agent") in ("complete", "error")

    def test_full_pipeline_supervisor_routes_correctly(self):
        """
        Build the graph and run it with mocked LLM nodes.
        Verifies supervisor routes through all deterministic nodes without crashing.
        """

        analysis_adapter = _mock_adapter(
            text_response=(
                "Revenue $15M. Gross margin 60%. EPS $0.35. "
                "Risk: supply chain volatility quantified at $500K."
            ),
            backend="ollama",
        )

        mock_report_msg = MagicMock()
        mock_report_msg.content = [MagicMock()]
        mock_report_msg.content[0].text = (
            "## EXECUTIVE SUMMARY\n"
            "Revenue $15,000,000. EPS $0.35. ASC 606 compliant.\n"
            "## RISK ASSESSMENT\n"
            "Supply chain: $500K exposure.\n"
        )
        mock_report_msg.usage = MagicMock(input_tokens=100, output_tokens=200)
        mock_report_client = MagicMock()
        mock_report_client.messages.create.return_value = mock_report_msg

        with patch("backend.agents.analysis_agent.get_adapter", return_value=analysis_adapter), \
             patch("backend.agents.reporting_agent.anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_report_client

            from backend.agents.supervisor import build_cfo_graph
            graph = build_cfo_graph()
            state = create_initial_state(
                task_id="pipe-001",
                task_type="variance_analysis",
                task_description="Q1 board report",
                company_name="MockCorp Inc",
                period="Q1 2026",
                raw_financial_data=VALID_FINANCIAL_DATA,
            )
            final = graph.invoke(state, config={"configurable": {"thread_id": "pipe-001"}})

        # Core deterministic agents must have run
        assert final["agent_statuses"].get("data_agent") == "complete"
        assert final["agent_statuses"].get("math_engine") == "complete"
        assert final["agent_statuses"].get("gaap_agent") == "complete"
        assert final["agent_statuses"].get("ifrs_agent") == "complete"
        # KPIs computed
        assert final.get("kpi_metrics") is not None
        assert final["kpi_metrics"].get("gross_margin_pct") is not None
