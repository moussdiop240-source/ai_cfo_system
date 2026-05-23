"""
Tests for reporting_agent_node.

All Anthropic API calls are mocked. Tests cover:
- Report prompt construction (_build_report_prompt)
- Successful report generation
- Missing API key graceful error
- API failure graceful error
- Audit log entries
- Token accounting
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.agents.reporting_agent import reporting_agent_node, _build_report_prompt


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def full_state(healthy_financials, healthy_kpis, healthy_variance, healthy_runway):
    from backend.compliance.gaap_engine import GAAPEngine
    from backend.compliance.ifrs_engine import IFRSEngine
    gaap = GAAPEngine().check_all(healthy_financials, healthy_kpis, healthy_variance, healthy_runway)
    ifrs = IFRSEngine().check_all(healthy_financials, healthy_kpis, healthy_variance, healthy_runway)
    return {
        "task_id": "t1",
        "task_type": "variance_analysis",
        "task_description": "Q1 board variance analysis",
        "company_name": "HealthyCo Inc",
        "period": "Q1 2026",
        "report_format": "board",
        "submitted_by": "cfo@healthyco.com",
        "submitted_by_role": "cfo",
        "submitted_at": "2026-01-01T00:00:00Z",
        "validated_data": healthy_financials,
        "kpi_metrics": healthy_kpis,
        "variance_table": healthy_variance,
        "gaap_results": gaap,
        "ifrs_results": ifrs,
        "analysis_narrative": (
            "HealthyCo delivered $12.5M revenue, a favorable $1.5M variance to $11.0M budget. "
            "Gross margin of 58.2% exceeds sector benchmark. ASC 606 revenue recognition compliant."
        ),
        "forecast_outputs": {"r2": 0.98, "method": "ensemble"},
        "rag_sources_cited": ["EBITDA Standards", "ASC 606 Guide"],
        "human_feedback": "Approved by CFO Jane Doe. Strong quarter.",
        "requires_human_approval": False,
        "human_decision": "approved",
        "agent_statuses": {
            "data_agent": "complete",
            "math_engine": "complete",
            "rag_agent": "complete",
            "gaap_agent": "complete",
            "ifrs_agent": "complete",
            "analysis_agent": "complete",
        },
        "errors": [],
        "warnings": [],
        "audit_log": [],
        "agent_history": [],
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        "processing_time_ms": 0,
    }


SAMPLE_REPORT = """# HealthyCo Inc — Q1 2026 Board Report

## 1. EXECUTIVE SUMMARY
EPS of $0.42 basic ($0.40 diluted). Revenue $12.5M — favorable $1.5M (13.6%) to $11.0M budget.
Per ASC 606 / IFRS 15 five-step model.

## 2. REVENUE PERFORMANCE
Revenue of $12.5M recognized per ASC 606 (IFRS 15). Variance: +$1.5M favorable.

## 3. COST & MARGIN ANALYSIS
Gross margin 58.2%. EBITDA margin 22.4%.

## 4. US GAAP COMPLIANCE NOTES
ASC 606 — COMPLIANT. ASC 842 — COMPLIANT.

## 5. IFRS COMPLIANCE NOTES
IFRS 15 — COMPLIANT. IFRS 16 — COMPLIANT.

## 6. RISK ASSESSMENT
1. Customer concentration: top 3 = 60% of revenue.

## 7. ACTION PLAN
CFO to present Q2 forecast to board by April 30, 2026.
"""


# ── Prompt construction ───────────────────────────────────────────────────────

class TestBuildReportPrompt:
    def test_prompt_contains_company_name(self, full_state):
        prompt = _build_report_prompt(full_state)
        assert "HealthyCo Inc" in prompt

    def test_prompt_contains_period(self, full_state):
        prompt = _build_report_prompt(full_state)
        assert "Q1 2026" in prompt

    def test_prompt_contains_revenue(self, full_state):
        prompt = _build_report_prompt(full_state)
        assert "12,500,000" in prompt

    def test_prompt_contains_gaap_status(self, full_state):
        prompt = _build_report_prompt(full_state)
        assert "GAAP" in prompt or "ASC" in prompt

    def test_prompt_contains_ifrs_status(self, full_state):
        prompt = _build_report_prompt(full_state)
        assert "IFRS" in prompt or "IAS" in prompt

    def test_prompt_contains_analysis_narrative(self, full_state):
        prompt = _build_report_prompt(full_state)
        assert "favorable" in prompt or "12.5M" in prompt

    def test_prompt_contains_human_feedback(self, full_state):
        prompt = _build_report_prompt(full_state)
        assert "Jane Doe" in prompt or "Approved by CFO" in prompt

    def test_prompt_contains_rag_sources(self, full_state):
        prompt = _build_report_prompt(full_state)
        assert "EBITDA Standards" in prompt or "ASC 606 Guide" in prompt

    def test_empty_gaap_handled(self, full_state):
        state = {**full_state, "gaap_results": {}}
        prompt = _build_report_prompt(state)
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_empty_variance_handled(self, full_state):
        state = {**full_state, "variance_table": {}}
        prompt = _build_report_prompt(state)
        assert isinstance(prompt, str) and len(prompt) > 0


# ── Agent node with mocked Anthropic ─────────────────────────────────────────

class TestReportingAgentNode:
    def _mock_response(self, text: str, input_tokens: int = 1000, output_tokens: int = 2000):
        usage = MagicMock()
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens
        content = MagicMock()
        content.text = text
        resp = MagicMock()
        resp.content = [content]
        resp.usage = usage
        return resp

    def test_successful_report_generation(self, full_state):
        mock_resp = self._mock_response(SAMPLE_REPORT)
        with patch("backend.agents.reporting_agent.anthropic") as mock_ant, \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            mock_ant.Anthropic.return_value.messages.create.return_value = mock_resp
            result = reporting_agent_node(full_state)

        assert result["agent_statuses"]["reporting_agent"] == "complete"
        assert result["final_report"] == SAMPLE_REPORT

    def test_token_usage_accumulated(self, full_state):
        mock_resp = self._mock_response(SAMPLE_REPORT, input_tokens=1000, output_tokens=2000)
        with patch("backend.agents.reporting_agent.anthropic") as mock_ant, \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            mock_ant.Anthropic.return_value.messages.create.return_value = mock_resp
            result = reporting_agent_node(full_state)

        assert result["total_tokens_used"] == 3000

    def test_audit_log_entry_added(self, full_state):
        mock_resp = self._mock_response(SAMPLE_REPORT)
        with patch("backend.agents.reporting_agent.anthropic") as mock_ant, \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            mock_ant.Anthropic.return_value.messages.create.return_value = mock_resp
            result = reporting_agent_node(full_state)

        agents = [e.get("agent") for e in result["audit_log"]]
        assert "reporting_agent" in agents

    def test_missing_api_key_returns_error(self, full_state):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            result = reporting_agent_node(full_state)
        assert result["agent_statuses"]["reporting_agent"] == "error"
        assert any("ANTHROPIC_API_KEY" in e for e in result["errors"])

    def test_api_exception_returns_error(self, full_state):
        with patch("backend.agents.reporting_agent.anthropic") as mock_ant, \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            mock_ant.Anthropic.return_value.messages.create.side_effect = RuntimeError("API timeout")
            result = reporting_agent_node(full_state)

        assert result["agent_statuses"]["reporting_agent"] == "error"
        assert any("reporting_agent" in e for e in result["errors"])

    def test_existing_errors_preserved(self, full_state):
        state = {**full_state, "errors": ["upstream error"]}
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            result = reporting_agent_node(state)
        assert "upstream error" in result["errors"]

    def test_existing_tokens_accumulated(self, full_state):
        state = {**full_state, "total_tokens_used": 500}
        mock_resp = self._mock_response(SAMPLE_REPORT, input_tokens=500, output_tokens=1500)
        with patch("backend.agents.reporting_agent.anthropic") as mock_ant, \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            mock_ant.Anthropic.return_value.messages.create.return_value = mock_resp
            result = reporting_agent_node(state)

        assert result["total_tokens_used"] == 500 + 2000

    def test_state_fields_preserved(self, full_state):
        """Unrelated state fields must pass through unchanged."""
        state = {**full_state, "iteration_count": 5}
        mock_resp = self._mock_response(SAMPLE_REPORT)
        with patch("backend.agents.reporting_agent.anthropic") as mock_ant, \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            mock_ant.Anthropic.return_value.messages.create.return_value = mock_resp
            result = reporting_agent_node(state)

        assert result.get("iteration_count") == 5

    def test_report_format_in_prompt(self, full_state):
        """Different report_format should produce different prompts."""
        prompts = set()
        for fmt in ("board", "investor", "internal", "audit"):
            state = {**full_state, "report_format": fmt}
            prompts.add(_build_report_prompt(state))
        # At least 2 distinct prompts (board vs audit differ)
        assert len(prompts) >= 2
