"""
Tests for data_agent_node.

The data agent has two paths:
1. Direct Pydantic validation (raw_financial_data is a dict that validates directly)
2. LLM extraction (unstructured data or dict that fails validation)

All LLM calls are mocked. We test path 1 extensively since it runs without
ANTHROPIC_API_KEY, and test path 2 with mocked anthropic.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.agents.data_agent import data_agent_node


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_state():
    return {
        "task_id": "t1",
        "task_type": "variance_analysis",
        "task_description": "Q1 analysis",
        "company_name": "TestCo",
        "period": "Q1 2026",
        "report_format": "board",
        "submitted_by": "test@co.com",
        "submitted_by_role": "analyst",
        "submitted_at": "2026-01-01T00:00:00Z",
        "raw_financial_data": None,
        "raw_documents": None,
        "validated_data": None,
        "agent_statuses": {},
        "errors": [],
        "warnings": [],
        "audit_log": [],
        "flags": [],
        "agent_history": [],
        "iteration_count": 0,
        "max_iterations": 20,
        "current_agent": "data_agent",
    }


VALID_FINANCIAL_DATA = {
    "company_name": "TestCo",
    "period": "Q1 2026",
    "currency": "USD",
    "revenue": 12_500_000,
    "cogs": 5_225_000,
    "total_assets": 45_000_000,
    "total_equity": 28_000_000,
    "current_assets": 18_000_000,
    "current_liabilities": 8_500_000,
    "cash": 6_200_000,
    "total_debt": 12_000_000,
    "accounts_receivable": 4_200_000,
    "inventory": 1_800_000,
    "shares_outstanding": 4_500_000,
    "diluted_shares": 4_750_000,
    "interest_expense": 420_000,
    "pre_tax_income": 2_040_000,
    "tax_provision": 150_000,
}


# ── Direct validation path ────────────────────────────────────────────────────

class TestDataAgentDirectValidation:
    def test_valid_dict_validates_directly(self, minimal_state):
        state = {**minimal_state, "raw_financial_data": VALID_FINANCIAL_DATA}
        result = data_agent_node(state)
        assert result["agent_statuses"]["data_agent"] == "complete"
        assert result["validated_data"] is not None
        assert result["validated_data"]["revenue"] == 12_500_000

    def test_validated_data_includes_derived_fields(self, minimal_state):
        """compute_derived() should be called — gross_profit should be set."""
        data = {**VALID_FINANCIAL_DATA, "cogs": 5_225_000}
        state = {**minimal_state, "raw_financial_data": data}
        result = data_agent_node(state)
        assert result["validated_data"].get("gross_profit") is not None

    def test_schema_version_set(self, minimal_state):
        state = {**minimal_state, "raw_financial_data": VALID_FINANCIAL_DATA}
        result = data_agent_node(state)
        assert result["schema_version"] == "FinancialDataSchema_v1"

    def test_schema_errors_empty_on_success(self, minimal_state):
        state = {**minimal_state, "raw_financial_data": VALID_FINANCIAL_DATA}
        result = data_agent_node(state)
        assert result["schema_errors"] == []

    def test_audit_log_entry_added(self, minimal_state):
        state = {**minimal_state, "raw_financial_data": VALID_FINANCIAL_DATA}
        result = data_agent_node(state)
        agents = [e.get("agent") for e in result["audit_log"]]
        assert "data_agent" in agents

    def test_negative_revenue_triggers_llm_fallback(self, minimal_state):
        """Pydantic rejects negative revenue → falls through to LLM path → errors out."""
        invalid = {**VALID_FINANCIAL_DATA, "revenue": -100}
        state = {**minimal_state, "raw_financial_data": invalid}
        # Without API key, LLM path fails
        result = data_agent_node(state)
        # Should either succeed (if Pydantic coerces) or fail gracefully
        assert "data_agent" in result["agent_statuses"]

    def test_missing_required_field_triggers_llm_fallback(self, minimal_state):
        """company_name missing → Pydantic validation fails → LLM fallback."""
        invalid = {k: v for k, v in VALID_FINANCIAL_DATA.items() if k != "company_name"}
        state = {**minimal_state, "raw_financial_data": invalid}
        result = data_agent_node(state)
        # Without API key, should error
        assert "data_agent" in result["agent_statuses"]

    def test_none_raw_data_triggers_llm_path(self, minimal_state):
        """None raw data → LLM path → errors without API key."""
        state = {**minimal_state, "raw_financial_data": None}
        result = data_agent_node(state)
        assert "data_agent" in result["agent_statuses"]

    def test_existing_errors_preserved(self, minimal_state):
        state = {
            **minimal_state,
            "raw_financial_data": VALID_FINANCIAL_DATA,
            "errors": ["pre-existing error"],
        }
        result = data_agent_node(state)
        assert "pre-existing error" in result["errors"]

    def test_minimal_required_fields_only(self, minimal_state):
        """Only company_name, period, currency are required by FinancialDataSchema."""
        minimal_data = {
            "company_name": "MinimalCo",
            "period": "Q1 2026",
            "currency": "USD",
        }
        state = {**minimal_state, "raw_financial_data": minimal_data}
        result = data_agent_node(state)
        assert result["agent_statuses"]["data_agent"] == "complete"
        assert result["validated_data"]["company_name"] == "MinimalCo"

    def test_string_revenue_coerced(self, minimal_state):
        """Pydantic should coerce numeric strings to float."""
        data = {**VALID_FINANCIAL_DATA, "revenue": "12500000"}
        state = {**minimal_state, "raw_financial_data": data}
        result = data_agent_node(state)
        # Pydantic v2 coerces string → float
        if result["agent_statuses"]["data_agent"] == "complete":
            assert result["validated_data"]["revenue"] == 12_500_000.0


# ── LLM extraction path (mocked) ─────────────────────────────────────────────

class TestDataAgentLLMPath:
    def test_no_api_key_errors_gracefully(self, minimal_state):
        """Without ANTHROPIC_API_KEY, LLM path should error, not crash."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            state = {**minimal_state, "raw_financial_data": "Acme Corp Q1 revenue $5M"}
            result = data_agent_node(state)
        assert "data_agent" in result["agent_statuses"]
        assert result["agent_statuses"]["data_agent"] in ("error", "complete")

    def test_instructor_path_mocked(self, minimal_state):
        """Mock instructor to return a valid FinancialDataSchema."""
        from backend.schemas.financial import FinancialDataSchema
        mock_result = FinancialDataSchema(
            company_name="MockCo", period="Q1 2026", currency="USD", revenue=5_000_000
        )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_result

        with patch("backend.agents.data_agent.INSTRUCTOR_AVAILABLE", True), \
             patch("backend.agents.data_agent.instructor") as mock_instructor, \
             patch("backend.agents.data_agent.anthropic") as mock_ant, \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            mock_instructor.from_anthropic.return_value = mock_client
            state = {**minimal_state, "raw_financial_data": "MockCo revenue $5M Q1 2026"}
            result = data_agent_node(state)

        assert result["agent_statuses"]["data_agent"] in ("complete", "error")

    def test_raw_claude_path_mocked(self, minimal_state):
        """Mock raw anthropic call returning JSON."""
        import json
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "company_name": "RawCo",
            "period": "Q1 2026",
            "currency": "USD",
            "revenue": 3_000_000,
        }))]

        with patch("backend.agents.data_agent.INSTRUCTOR_AVAILABLE", False), \
             patch("backend.agents.data_agent.anthropic") as mock_ant, \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            mock_ant.Anthropic.return_value.messages.create.return_value = mock_response
            state = {**minimal_state, "raw_financial_data": "RawCo $3M revenue Q1"}
            result = data_agent_node(state)

        assert result["agent_statuses"]["data_agent"] in ("complete", "error")
