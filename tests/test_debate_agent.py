"""
Tests for debate_agent_node (3-round IFRS vs GAAP debate).

All LLM calls are mocked via the adapter. Tests cover:
- Prompt construction for IFRS advocate, GAAP advocate, arbiter
- All 3 rounds completing successfully
- Individual round failures handled gracefully
- state fields set correctly (debate_ifrs_advocate, debate_gaap_advocate, debate_arbiter, debate_complete)
- Audit log entries
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.agents.debate_agent import (
    debate_agent_node,
    _build_advocate_prompt,
    _build_arbiter_prompt,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def debate_state(healthy_financials, healthy_kpis, healthy_variance, healthy_runway):
    from backend.compliance.gaap_engine import GAAPEngine
    from backend.compliance.ifrs_engine import IFRSEngine
    gaap = GAAPEngine().check_all(healthy_financials, healthy_kpis, healthy_variance, healthy_runway)
    ifrs = IFRSEngine().check_all(healthy_financials, healthy_kpis, healthy_variance, healthy_runway)
    return {
        "task_id": "t1",
        "task_type": "variance_analysis",
        "task_description": "Q1 full report with IFRS/GAAP debate",
        "company_name": "HealthyCo Inc",
        "period": "Q1 2026",
        "report_format": "board",
        "submitted_by": "cfo@co.com",
        "submitted_by_role": "cfo",
        "submitted_at": "2026-01-01T00:00:00Z",
        "validated_data": {
            **healthy_financials,
            "jurisdiction": "United States",
            "listing_exchange": "NASDAQ",
            "industry": "Technology",
            "rd_dev_capitalizable_pct": 30,
        },
        "kpi_metrics": healthy_kpis,
        "variance_table": healthy_variance,
        "gaap_results": gaap,
        "ifrs_results": ifrs,
        "analysis_narrative": "Strong Q1 results.",
        "debate_ifrs_advocate": None,
        "debate_gaap_advocate": None,
        "debate_arbiter": None,
        "debate_complete": False,
        "requires_human_approval": False,
        "human_decision": None,
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


IFRS_ARG = "IFRS provides superior transparency for this international SaaS company..."
GAAP_ARG = "US GAAP is required for NASDAQ listing. ASC 606 ensures conservative recognition..."
ARBITER = "VERDICT: US GAAP is mandatory for NASDAQ listing per SEC rules. Top 5 reconciling items..."


# ── Prompt construction ───────────────────────────────────────────────────────

class TestDebatePrompts:
    def test_ifrs_advocate_prompt_contains_company(self, debate_state):
        prompt = _build_advocate_prompt(debate_state, "ifrs")
        assert "HealthyCo Inc" in prompt

    def test_ifrs_advocate_prompt_contains_period(self, debate_state):
        prompt = _build_advocate_prompt(debate_state, "ifrs")
        assert "Q1 2026" in prompt

    def test_ifrs_advocate_prompt_cites_ifrs16(self, debate_state):
        prompt = _build_advocate_prompt(debate_state, "ifrs")
        assert "IFRS 16" in prompt or "ifrs16" in prompt.lower()

    def test_gaap_advocate_prompt_cites_asc842(self, debate_state):
        prompt = _build_advocate_prompt(debate_state, "gaap")
        assert "ASC 842" in prompt or "asc842" in prompt.lower()

    def test_ifrs_and_gaap_prompts_different(self, debate_state):
        p_ifrs = _build_advocate_prompt(debate_state, "ifrs")
        p_gaap = _build_advocate_prompt(debate_state, "gaap")
        assert p_ifrs != p_gaap

    def test_arbiter_prompt_contains_both_arguments(self, debate_state):
        prompt = _build_arbiter_prompt(debate_state, IFRS_ARG, GAAP_ARG)
        assert IFRS_ARG[:30] in prompt
        assert GAAP_ARG[:30] in prompt

    def test_arbiter_prompt_contains_quantified_facts(self, debate_state):
        prompt = _build_arbiter_prompt(debate_state, IFRS_ARG, GAAP_ARG)
        assert "EBITDA" in prompt or "R&D" in prompt or "Goodwill" in prompt

    def test_empty_gaap_results_handled(self, debate_state):
        state = {**debate_state, "gaap_results": {}}
        prompt = _build_advocate_prompt(state, "gaap")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_missing_financial_data_handled(self, debate_state):
        state = {**debate_state, "validated_data": {}}
        prompt = _build_advocate_prompt(state, "ifrs")
        assert isinstance(prompt, str) and len(prompt) > 0


# ── Agent node with mocked adapter ───────────────────────────────────────────

class TestDebateAgentNode:
    def _mock_adapter(self, responses: list):
        """Return a mock adapter whose complete() cycles through responses."""
        adapter = MagicMock()
        adapter.active_backend = "ollama"
        adapter.active_model = "llama3"
        adapter.complete.side_effect = responses
        return adapter

    def test_all_three_rounds_complete(self, debate_state):
        mock_adapter = self._mock_adapter([IFRS_ARG, GAAP_ARG, ARBITER])
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            result = debate_agent_node(debate_state)

        assert result["debate_ifrs_advocate"] == IFRS_ARG
        assert result["debate_gaap_advocate"] == GAAP_ARG
        assert result["debate_arbiter"] == ARBITER

    def test_debate_complete_flag_set(self, debate_state):
        mock_adapter = self._mock_adapter([IFRS_ARG, GAAP_ARG, ARBITER])
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            result = debate_agent_node(debate_state)
        assert result["debate_complete"] is True

    def test_agent_status_set_complete(self, debate_state):
        mock_adapter = self._mock_adapter([IFRS_ARG, GAAP_ARG, ARBITER])
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            result = debate_agent_node(debate_state)
        assert result["agent_statuses"]["debate_agent"] == "complete"

    def test_audit_log_entry_added(self, debate_state):
        mock_adapter = self._mock_adapter([IFRS_ARG, GAAP_ARG, ARBITER])
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            result = debate_agent_node(debate_state)
        agents = [e.get("agent") for e in result["audit_log"]]
        assert "debate_agent" in agents

    def test_round1_failure_continues(self, debate_state):
        """If IFRS round fails, GAAP and arbiter should still run."""
        mock_adapter = self._mock_adapter([
            RuntimeError("IFRS round failed"),
            GAAP_ARG,
            ARBITER,
        ])
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            result = debate_agent_node(debate_state)

        assert result["debate_complete"] is True
        assert "Error" in result["debate_ifrs_advocate"]
        assert result["debate_gaap_advocate"] == GAAP_ARG

    def test_round2_failure_continues(self, debate_state):
        mock_adapter = self._mock_adapter([
            IFRS_ARG,
            RuntimeError("GAAP round failed"),
            ARBITER,
        ])
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            result = debate_agent_node(debate_state)

        assert result["debate_ifrs_advocate"] == IFRS_ARG
        assert "Error" in result["debate_gaap_advocate"]
        assert result["debate_arbiter"] == ARBITER

    def test_round3_failure_continues(self, debate_state):
        mock_adapter = self._mock_adapter([
            IFRS_ARG,
            GAAP_ARG,
            RuntimeError("Arbiter failed"),
        ])
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            result = debate_agent_node(debate_state)

        assert result["debate_ifrs_advocate"] == IFRS_ARG
        assert result["debate_gaap_advocate"] == GAAP_ARG
        assert "Error" in result["debate_arbiter"]

    def test_round_failures_add_errors(self, debate_state):
        mock_adapter = self._mock_adapter([
            RuntimeError("Round 1 error"),
            GAAP_ARG,
            ARBITER,
        ])
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            result = debate_agent_node(debate_state)
        assert any("debate" in e.lower() for e in result["errors"])

    def test_all_rounds_fail_still_completes(self, debate_state):
        mock_adapter = self._mock_adapter([
            RuntimeError("R1"), RuntimeError("R2"), RuntimeError("R3"),
        ])
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            result = debate_agent_node(debate_state)
        assert result["debate_complete"] is True
        assert result["agent_statuses"]["debate_agent"] == "complete"
        assert len(result["errors"]) == 3

    def test_existing_errors_preserved(self, debate_state):
        state = {**debate_state, "errors": ["upstream error"]}
        mock_adapter = self._mock_adapter([IFRS_ARG, GAAP_ARG, ARBITER])
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            result = debate_agent_node(state)
        assert "upstream error" in result["errors"]

    def test_adapter_called_three_times(self, debate_state):
        mock_adapter = self._mock_adapter([IFRS_ARG, GAAP_ARG, ARBITER])
        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            debate_agent_node(debate_state)
        assert mock_adapter.complete.call_count == 3

    def test_anthropic_backend_uses_larger_tokens(self, debate_state):
        """Anthropic backend should request more tokens than Ollama."""
        mock_adapter = MagicMock()
        mock_adapter.active_backend = "anthropic"
        mock_adapter.active_model = "claude-sonnet-4-6"
        mock_adapter.complete.side_effect = [IFRS_ARG, GAAP_ARG, ARBITER]

        with patch("backend.agents.debate_agent.get_adapter", return_value=mock_adapter):
            debate_agent_node(debate_state)

        calls = mock_adapter.complete.call_args_list
        # For anthropic: adv_tok=1500, arb_tok=2000
        assert calls[0][1]["max_tokens"] == 1500 or calls[0].args[-1] == 1500
        # Arbiter call (last) should use arb_tok
        last_call_tokens = calls[2][1].get("max_tokens") or calls[2].args[-1]
        assert last_call_tokens == 2000
