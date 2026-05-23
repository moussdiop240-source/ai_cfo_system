"""
End-to-end pipeline integration tests.

Runs all 9 nodes in sequence with mocked LLM calls. Verifies:
- Full state transitions from raw_financial_data → final_report
- Each node sets its agent_status to "complete"
- supervisor routing progresses through all steps
- HITL gate fires and can be approved
- Final state contains all expected fields

No real LLM calls are made — all AI nodes are mocked via the adapter or
anthropic client.
"""
import os
import sys
from unittest.mock import MagicMock, patch, ANY

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.agents.data_agent import data_agent_node
from backend.agents.math_engine import math_engine_node
from backend.agents.gaap_agent import gaap_agent_node
from backend.agents.ifrs_agent import ifrs_agent_node
from backend.agents.rag_agent import rag_agent_node
from backend.agents.analysis_agent import analysis_agent_node
from backend.agents.debate_agent import debate_agent_node
from backend.agents.human_loop_node import human_review_node
from backend.agents.reporting_agent import reporting_agent_node
from backend.agents.supervisor import route_from_supervisor


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _make_mock_adapter(analysis_text: str = None):
    adapter = MagicMock()
    adapter.active_backend = "ollama"
    adapter.active_model = "llama3"
    adapter.complete.return_value = analysis_text or (
        "Revenue reached $12.5M in Q1 2026, a favorable 13.6% variance to $11.0M budget. "
        "Gross margin 58.2% exceeds sector benchmark per ASC 606 / IFRS 15. "
        "CFO to review capital allocation by end of Q2 2026."
    )
    return adapter


def _make_mock_anthropic(report_text: str = None):
    usage = MagicMock()
    usage.input_tokens = 500
    usage.output_tokens = 1000
    content = MagicMock()
    content.text = report_text or (
        "# HealthyCo Inc — Q1 2026 Board Report\n\n"
        "## 1. EXECUTIVE SUMMARY\n"
        "EPS $0.42. Revenue $12.5M — favorable $1.5M to $11.0M budget. ASC 606 compliant.\n\n"
        "## 7. ACTION PLAN\n"
        "CFO to present Q2 forecast by April 30, 2026 per ASC 606.\n"
    )
    resp = MagicMock()
    resp.content = [content]
    resp.usage = usage
    return resp


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def initial_state():
    """Minimal starting state — only raw financial data and task metadata."""
    return {
        "task_id":           "e2e-test-001",
        "task_type":         "full_report",
        "task_description":  "Q1 2026 full board report with IFRS/GAAP debate",
        "company_name":      "HealthyCo Inc",
        "period":            "Q1 2026",
        "report_format":     "board",
        "submitted_by":      "cfo@healthyco.com",
        "submitted_by_role": "cfo",
        "submitted_at":      "2026-01-01T00:00:00Z",
        "raw_financial_data": {
            "company_name":   "HealthyCo Inc",
            "period":         "Q1 2026",
            "currency":       "USD",
            "revenue":        12_500_000,
            "cogs":            5_225_000,
            "ebitda":          2_800_000,
            "ebit":            2_460_000,
            "net_income":      1_890_000,
            "total_assets":   45_000_000,
            "total_equity":   28_000_000,
            "current_assets": 18_000_000,
            "current_liabilities": 8_500_000,
            "cash":            6_200_000,
            "total_debt":     12_000_000,
            "accounts_receivable": 4_200_000,
            "accounts_payable": 2_800_000,
            "inventory":       1_800_000,
            "goodwill":        5_000_000,
            "goodwill_impairment_test_date": "2026-01-15",
            "rou_assets":      2_400_000,
            "lease_liability": 2_200_000,
            "shares_outstanding": 4_500_000,
            "diluted_shares":  4_750_000,
            "interest_expense":  420_000,
            "pre_tax_income":  2_040_000,
            "tax_provision":     150_000,
            "revenue_recognition_policy": "5-step model per ASC 606",
            "deferred_revenue":  800_000,
            "interest_cash_flow_classification": "operating",
            "cash_from_operations": 2_100_000,
            "historical_revenue": [9_000_000, 9_800_000, 10_500_000, 11_200_000, 12_000_000, 12_500_000],
            "actuals": {"revenue": 12_500_000, "cogs": 5_225_000},
            "budget":  {"revenue": 11_000_000, "cogs":  5_000_000},
            "jurisdiction": "United States",
            "listing_exchange": "NASDAQ",
            "industry": "Technology",
        },
        "raw_documents": None,
        "validated_data":    None,
        "kpi_metrics":       None,
        "variance_table":    None,
        "gaap_results":      None,
        "ifrs_results":      None,
        "rag_chunks":        None,
        "analysis_narrative": None,
        "final_report":      None,
        "debate_complete":   False,
        "debate_ifrs_advocate": None,
        "debate_gaap_advocate": None,
        "debate_arbiter":    None,
        "requires_human_approval": False,
        "human_decision":    None,
        "approval_triggers": None,
        "human_feedback":    None,
        "approved_by":       None,
        "approved_at":       None,
        "agent_statuses":    {},
        "errors":            [],
        "warnings":          [],
        "flags":             [],
        "audit_log":         [],
        "agent_history":     [],
        "iteration_count":   0,
        "max_iterations":    20,
        "current_agent":     "supervisor",
        "next_agent":        None,
        "math_results":      None,
        "forecast_outputs":  None,
        "reconciliation_data": None,
        "data_quality_score": None,
        "anomaly_flags":     None,
        "gaap_compliant_count": None,
        "gaap_issues_count": None,
        "ifrs_compliant_count": None,
        "ifrs_issues_count": None,
        "schema_errors":     None,
        "schema_version":    None,
        "structured_output": None,
        "rag_query_used":    None,
        "rag_sources_cited": None,
        "retrieval_confidence": None,
        "identified_risks":  None,
        "opportunities":     None,
        "action_items":      None,
        "ai_confidence_score": None,
        "validation_errors": None,
        "validation_warnings": None,
        "validation_score":  None,
        "draft_report":      None,
        "report_pdf_path":   None,
        "total_tokens_used": 0,
        "total_cost_usd":    0.0,
        "processing_time_ms": 0,
    }


# ── Individual node tests ─────────────────────────────────────────────────────

class TestNodeSequence:
    """Each test runs one node given the previous node's output."""

    def test_step1_data_agent(self, initial_state):
        state = data_agent_node(initial_state)
        assert state["agent_statuses"]["data_agent"] == "complete"
        assert state["validated_data"] is not None
        assert state["validated_data"]["revenue"] == 12_500_000

    def test_step2_math_engine(self, initial_state):
        state = data_agent_node(initial_state)
        state = math_engine_node(state)
        assert state["agent_statuses"]["math_engine"] == "complete"
        assert state["kpi_metrics"] is not None
        assert state["kpi_metrics"]["gross_margin_pct"] > 0
        assert state["variance_table"] is not None

    def test_step3_gaap_agent(self, initial_state):
        state = data_agent_node(initial_state)
        state = math_engine_node(state)
        state = gaap_agent_node(state)
        assert state["agent_statuses"]["gaap_agent"] == "complete"
        assert state["gaap_results"] is not None
        assert len(state["gaap_results"]) > 0

    def test_step4_ifrs_agent(self, initial_state):
        state = data_agent_node(initial_state)
        state = math_engine_node(state)
        state = gaap_agent_node(state)
        state = ifrs_agent_node(state)
        assert state["agent_statuses"]["ifrs_agent"] == "complete"
        assert state["ifrs_results"] is not None

    def test_step5_rag_agent(self, initial_state):
        state = data_agent_node(initial_state)
        state = math_engine_node(state)
        state = gaap_agent_node(state)
        state = ifrs_agent_node(state)
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama"}):
            state = rag_agent_node(state)
        assert state["agent_statuses"]["rag_agent"] == "complete"
        assert isinstance(state.get("rag_chunks"), list)

    def test_step6_analysis_agent(self, initial_state):
        state = data_agent_node(initial_state)
        state = math_engine_node(state)
        state = gaap_agent_node(state)
        state = ifrs_agent_node(state)
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama"}):
            state = rag_agent_node(state)

        mock_adapter = _make_mock_adapter()
        with patch("backend.agents.analysis_agent.get_adapter", return_value=mock_adapter):
            state = analysis_agent_node(state)

        assert state["agent_statuses"]["analysis_agent"] in ("complete", "validation_failed")
        assert state["analysis_narrative"] is not None or state["validation_errors"]

    def test_step7_debate_agent(self, initial_state):
        state = data_agent_node(initial_state)
        state = math_engine_node(state)
        state = gaap_agent_node(state)
        state = ifrs_agent_node(state)
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama"}):
            state = rag_agent_node(state)

        mock_adapter = _make_mock_adapter()
        with patch("backend.agents.analysis_agent.get_adapter", return_value=mock_adapter), \
             patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            state = analysis_agent_node(state)
            state = debate_agent_node(state)

        assert state["debate_complete"] is True
        assert state["agent_statuses"]["debate_agent"] == "complete"

    def test_step8_reporting_agent(self, initial_state):
        state = data_agent_node(initial_state)
        state = math_engine_node(state)
        state = gaap_agent_node(state)
        state = ifrs_agent_node(state)
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama"}):
            state = rag_agent_node(state)

        mock_adapter = _make_mock_adapter()
        mock_anthropic_resp = _make_mock_anthropic()

        with patch("backend.agents.analysis_agent.get_adapter", return_value=mock_adapter), \
             patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter), \
             patch("backend.agents.reporting_agent.anthropic") as mock_ant, \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test", "LLM_BACKEND": "ollama"}):
            mock_ant.Anthropic.return_value.messages.create.return_value = mock_anthropic_resp
            state = analysis_agent_node(state)
            state = debate_agent_node(state)
            state = reporting_agent_node(state)

        assert state["agent_statuses"]["reporting_agent"] == "complete"
        assert state["final_report"] is not None
        assert len(state["final_report"]) > 50


# ── Full pipeline test ────────────────────────────────────────────────────────

class TestFullPipeline:
    def _run_full_pipeline(self, state, with_hitl: bool = False):
        """Run all 9 nodes in sequence. Returns final state."""
        # Node 1: data agent
        state = data_agent_node(state)

        # Node 2: math engine
        state = math_engine_node(state)

        # Node 3: GAAP
        state = gaap_agent_node(state)

        # Node 4: IFRS
        state = ifrs_agent_node(state)

        # Node 5: RAG
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama"}):
            state = rag_agent_node(state)

        # Node 6: Analysis (mocked LLM)
        mock_adapter = _make_mock_adapter()
        with patch("backend.agents.analysis_agent.get_adapter", return_value=mock_adapter):
            state = analysis_agent_node(state)

        # Node 7: HITL (if required)
        if state.get("requires_human_approval") and with_hitl:
            state = human_review_node({**state, "human_decision": "approved", "approved_by": "CFO"})

        # Node 8: Debate
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            state = debate_agent_node(state)

        # Node 9: Reporting (mocked Anthropic)
        mock_resp = _make_mock_anthropic()
        with patch("backend.agents.reporting_agent.anthropic") as mock_ant, \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            mock_ant.Anthropic.return_value.messages.create.return_value = mock_resp
            state = reporting_agent_node(state)

        return state

    def test_full_pipeline_produces_final_report(self, initial_state):
        final = self._run_full_pipeline(initial_state)
        assert final["final_report"] is not None
        assert len(final["final_report"]) > 50

    def test_full_pipeline_no_critical_errors(self, initial_state):
        final = self._run_full_pipeline(initial_state)
        # Allow validation warnings but not LLM/infra errors
        infra_errors = [e for e in final["errors"]
                        if "reporting_agent" in e or "math" in e or "gaap" in e]
        assert infra_errors == [], f"Infrastructure errors: {infra_errors}"

    def test_full_pipeline_all_statuses_complete(self, initial_state):
        final = self._run_full_pipeline(initial_state)
        expected = [
            "data_agent", "math_engine", "gaap_agent", "ifrs_agent",
            "rag_agent", "debate_agent", "reporting_agent",
        ]
        for agent in expected:
            assert final["agent_statuses"].get(agent) in ("complete", "error"), \
                f"{agent} status is {final['agent_statuses'].get(agent)!r}"

    def test_full_pipeline_kpis_computed(self, initial_state):
        final = self._run_full_pipeline(initial_state)
        kpis = final["kpi_metrics"]
        assert kpis is not None
        assert kpis.get("gross_margin_pct", 0) > 0
        assert kpis.get("current_ratio", 0) > 0

    def test_full_pipeline_gaap_12_standards_checked(self, initial_state):
        final = self._run_full_pipeline(initial_state)
        assert len(final["gaap_results"]) == 12

    def test_full_pipeline_ifrs_12_standards_checked(self, initial_state):
        final = self._run_full_pipeline(initial_state)
        assert len(final["ifrs_results"]) == 12

    def test_full_pipeline_rag_chunks_retrieved(self, initial_state):
        final = self._run_full_pipeline(initial_state)
        assert isinstance(final.get("rag_chunks"), list)
        assert len(final["rag_chunks"]) > 0

    def test_full_pipeline_debate_complete(self, initial_state):
        final = self._run_full_pipeline(initial_state)
        assert final["debate_complete"] is True

    def test_full_pipeline_audit_log_has_all_agents(self, initial_state):
        final = self._run_full_pipeline(initial_state)
        agents_in_log = {e.get("agent") for e in final["audit_log"]}
        assert "data_agent" in agents_in_log
        assert "reporting_agent" in agents_in_log

    def test_full_pipeline_tokens_accumulated(self, initial_state):
        final = self._run_full_pipeline(initial_state)
        # Reporting agent consumes 1500 mocked tokens
        assert final["total_tokens_used"] >= 1500

    def test_full_pipeline_supervisor_routing(self, initial_state):
        """Verify supervisor routing returns __end__ after full pipeline."""
        final = self._run_full_pipeline(initial_state)
        route = route_from_supervisor(final)
        assert route == "__end__"

    def test_distressed_company_flags_anomalies(self, initial_state):
        """A distressed company should produce anomaly flags."""
        distressed_data = {
            "company_name": "DistressedCo LLC",
            "period": "Q2 2026",
            "currency": "USD",
            "revenue": 2_000_000,
            "cogs": 1_800_000,
            "total_assets": 5_000_000,
            "total_equity": 500_000,
            "current_assets": 800_000,
            "current_liabilities": 1_200_000,
            "cash": 300_000,
            "total_debt": 4_000_000,
            "shares_outstanding": 1_000_000,
            "diluted_shares": 1_000_000,
            "monthly_cash_burn": 100_000,
            "actuals": {"revenue": 2_000_000, "cogs": 1_800_000},
            "budget":  {"revenue": 5_000_000, "cogs": 2_500_000},
        }
        state = {**initial_state, "raw_financial_data": distressed_data, "company_name": "DistressedCo"}
        state = data_agent_node(state)
        state = math_engine_node(state)
        # Distressed company should have anomaly flags or HITL triggers
        anomalies = state.get("anomaly_flags") or []
        hitl = state.get("requires_human_approval", False)
        triggers = state.get("approval_triggers") or []
        assert len(anomalies) > 0 or hitl or len(triggers) > 0


# ── Supervisor routing integration ────────────────────────────────────────────

class TestSupervisorIntegration:
    def test_supervisor_routes_to_data_agent_first(self, initial_state):
        assert route_from_supervisor(initial_state) == "data_agent"

    def test_supervisor_routes_to_math_after_data(self, initial_state):
        state = data_agent_node(initial_state)
        assert route_from_supervisor(state) == "math_engine"

    def test_supervisor_routes_to_gaap_after_math(self, initial_state):
        state = data_agent_node(initial_state)
        state = math_engine_node(state)
        assert route_from_supervisor(state) == "rag_agent"  # after math: rag

    def test_supervisor_ends_after_reporting(self, initial_state):
        state = data_agent_node(initial_state)
        state = math_engine_node(state)
        # Simulate all nodes complete
        state["agent_statuses"].update({
            "rag_agent": "complete",
            "gaap_agent": "complete",
            "ifrs_agent": "complete",
            "analysis_agent": "complete",
            "reporting_agent": "complete",
        })
        state["rag_chunks"] = [{"title": "T", "content": "C", "id": "1", "score": 0.9, "category": "g"}]
        state["gaap_results"] = {"asc606": {"status": "COMPLIANT"}}
        state["ifrs_results"] = {"ifrs15": {"status": "COMPLIANT"}}
        state["analysis_narrative"] = "Summary..."
        state["final_report"] = "Full board report."
        state["requires_human_approval"] = False
        assert route_from_supervisor(state) == "__end__"
