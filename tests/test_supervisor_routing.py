"""
Tests for the LangGraph supervisor routing logic.

All 9 routing steps, hard stops, and state transitions are tested
without any LLM calls — routing is deterministic.
"""
import copy
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.agents.supervisor import route_from_supervisor


# ── Helpers ──────────────────────────────────────────────────────────────────

def _state_at_step(base, **overrides):
    """Build a minimal state dict for a specific routing scenario."""
    s = copy.deepcopy(base)
    s.update(overrides)
    return s


# ── Step-by-step routing tests ───────────────────────────────────────────────

class TestSupervisorRouting:
    """Each test verifies exactly one routing decision."""

    def test_step1_routes_to_data_agent_when_no_validated_data(self, minimal_cfo_state):
        assert route_from_supervisor(minimal_cfo_state) == "data_agent"

    def test_step1_routes_to_data_agent_when_status_not_complete(self, minimal_cfo_state):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           agent_statuses={"data_agent": "running"})
        assert route_from_supervisor(s) == "data_agent"

    def test_step2_routes_to_math_engine_after_data_validated(self, minimal_cfo_state):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           agent_statuses={"data_agent": "complete"})
        assert route_from_supervisor(s) == "math_engine"

    def test_step3_routes_to_rag_after_math_complete(self, minimal_cfo_state, healthy_kpis):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           kpi_metrics=healthy_kpis,
                           agent_statuses={
                               "data_agent": "complete",
                               "math_engine": "complete",
                           })
        assert route_from_supervisor(s) == "rag_agent"

    def test_step4_routes_to_gaap_after_rag_complete(self, minimal_cfo_state, healthy_kpis):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           kpi_metrics=healthy_kpis,
                           rag_chunks=[{"title": "ASC 606", "content": "..."}],
                           agent_statuses={
                               "data_agent": "complete",
                               "math_engine": "complete",
                               "rag_agent": "complete",
                           })
        assert route_from_supervisor(s) == "gaap_agent"

    def test_step5_routes_to_ifrs_after_gaap_complete(self, minimal_cfo_state, healthy_kpis):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           kpi_metrics=healthy_kpis,
                           rag_chunks=[{}],
                           gaap_results={"asc606": {"status": "COMPLIANT"}},
                           agent_statuses={
                               "data_agent": "complete",
                               "math_engine": "complete",
                               "rag_agent": "complete",
                               "gaap_agent": "complete",
                           })
        assert route_from_supervisor(s) == "ifrs_agent"

    def test_step6_routes_to_analysis_after_ifrs_complete(self, minimal_cfo_state, healthy_kpis):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           kpi_metrics=healthy_kpis,
                           rag_chunks=[{}],
                           gaap_results={"asc606": {"status": "COMPLIANT"}},
                           ifrs_results={"ifrs15": {"status": "COMPLIANT"}},
                           agent_statuses={
                               "data_agent": "complete",
                               "math_engine": "complete",
                               "rag_agent": "complete",
                               "gaap_agent": "complete",
                               "ifrs_agent": "complete",
                           })
        assert route_from_supervisor(s) == "analysis_agent"

    def test_step7_routes_to_human_review_when_hitl_required(self, minimal_cfo_state, healthy_kpis):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           kpi_metrics=healthy_kpis,
                           rag_chunks=[{}],
                           gaap_results={"asc606": {"status": "COMPLIANT"}},
                           ifrs_results={"ifrs15": {"status": "COMPLIANT"}},
                           analysis_narrative="Summary...",
                           requires_human_approval=True,
                           human_decision=None,
                           agent_statuses={
                               "data_agent": "complete",
                               "math_engine": "complete",
                               "rag_agent": "complete",
                               "gaap_agent": "complete",
                               "ifrs_agent": "complete",
                               "analysis_agent": "complete",
                           })
        assert route_from_supervisor(s) == "human_review"

    def test_step7_skipped_when_no_hitl_required(self, minimal_cfo_state, healthy_kpis):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           kpi_metrics=healthy_kpis,
                           rag_chunks=[{}],
                           gaap_results={"asc606": {"status": "COMPLIANT"}},
                           ifrs_results={"ifrs15": {"status": "COMPLIANT"}},
                           analysis_narrative="Summary...",
                           requires_human_approval=False,
                           agent_statuses={
                               "data_agent": "complete",
                               "math_engine": "complete",
                               "rag_agent": "complete",
                               "gaap_agent": "complete",
                               "ifrs_agent": "complete",
                               "analysis_agent": "complete",
                           })
        assert route_from_supervisor(s) == "reporting_agent"

    def test_step7_skipped_when_hitl_approved(self, minimal_cfo_state, healthy_kpis):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           kpi_metrics=healthy_kpis,
                           rag_chunks=[{}],
                           gaap_results={"asc606": {"status": "COMPLIANT"}},
                           ifrs_results={"ifrs15": {"status": "COMPLIANT"}},
                           analysis_narrative="Summary...",
                           requires_human_approval=True,
                           human_decision="approved",
                           agent_statuses={
                               "data_agent": "complete",
                               "math_engine": "complete",
                               "rag_agent": "complete",
                               "gaap_agent": "complete",
                               "ifrs_agent": "complete",
                               "analysis_agent": "complete",
                           })
        assert route_from_supervisor(s) == "reporting_agent"

    def test_step8_routes_to_reporting_agent(self, minimal_cfo_state, healthy_kpis):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           kpi_metrics=healthy_kpis,
                           rag_chunks=[{}],
                           gaap_results={"asc606": {"status": "COMPLIANT"}},
                           ifrs_results={"ifrs15": {"status": "COMPLIANT"}},
                           analysis_narrative="Summary...",
                           requires_human_approval=False,
                           agent_statuses={
                               "data_agent": "complete",
                               "math_engine": "complete",
                               "rag_agent": "complete",
                               "gaap_agent": "complete",
                               "ifrs_agent": "complete",
                               "analysis_agent": "complete",
                           })
        assert route_from_supervisor(s) == "reporting_agent"

    def test_step9_end_when_report_complete(self, minimal_cfo_state, healthy_kpis):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           kpi_metrics=healthy_kpis,
                           rag_chunks=[{}],
                           gaap_results={"asc606": {"status": "COMPLIANT"}},
                           ifrs_results={"ifrs15": {"status": "COMPLIANT"}},
                           analysis_narrative="Summary...",
                           final_report="Full board report...",
                           requires_human_approval=False,
                           agent_statuses={
                               "data_agent": "complete",
                               "math_engine": "complete",
                               "rag_agent": "complete",
                               "gaap_agent": "complete",
                               "ifrs_agent": "complete",
                               "analysis_agent": "complete",
                               "reporting_agent": "complete",
                           })
        assert route_from_supervisor(s) == "__end__"


# ── Hard stop conditions ─────────────────────────────────────────────────────

class TestSupervisorHardStops:
    def test_hard_stop_on_too_many_errors(self, minimal_cfo_state):
        s = _state_at_step(minimal_cfo_state,
                           errors=["e1", "e2", "e3", "e4", "e5", "e6"])
        assert route_from_supervisor(s) == "__end__"

    def test_hard_stop_at_exactly_6_errors(self, minimal_cfo_state):
        s = _state_at_step(minimal_cfo_state,
                           errors=["e"] * 6)
        assert route_from_supervisor(s) == "__end__"

    def test_5_errors_does_not_hard_stop(self, minimal_cfo_state):
        s = _state_at_step(minimal_cfo_state,
                           errors=["e"] * 5)
        # 5 errors should not trigger the hard stop (> 5 is the threshold)
        result = route_from_supervisor(s)
        assert result != "__end__"

    def test_hard_stop_on_max_iterations(self, minimal_cfo_state):
        s = _state_at_step(minimal_cfo_state,
                           iteration_count=20,
                           max_iterations=20)
        assert route_from_supervisor(s) == "__end__"

    def test_iteration_count_19_continues(self, minimal_cfo_state):
        s = _state_at_step(minimal_cfo_state,
                           iteration_count=19,
                           max_iterations=20)
        # Should NOT end — should route to data_agent (first pending step)
        result = route_from_supervisor(s)
        assert result == "data_agent"

    def test_custom_max_iterations_respected(self, minimal_cfo_state):
        s = _state_at_step(minimal_cfo_state,
                           iteration_count=5,
                           max_iterations=5)
        assert route_from_supervisor(s) == "__end__"


# ── HITL decision state transitions ─────────────────────────────────────────

class TestHITLStateTransitions:
    def test_hitl_pending_stays_at_human_review(self, minimal_cfo_state, healthy_kpis):
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           kpi_metrics=healthy_kpis,
                           rag_chunks=[{}],
                           gaap_results={"asc606": {"status": "COMPLIANT"}},
                           ifrs_results={"ifrs15": {"status": "COMPLIANT"}},
                           analysis_narrative="Summary...",
                           requires_human_approval=True,
                           human_decision="pending",
                           agent_statuses={
                               "data_agent": "complete",
                               "math_engine": "complete",
                               "rag_agent": "complete",
                               "gaap_agent": "complete",
                               "ifrs_agent": "complete",
                               "analysis_agent": "complete",
                           })
        assert route_from_supervisor(s) == "human_review"

    def test_hitl_rejected_still_routes_to_human_review(self, minimal_cfo_state, healthy_kpis):
        """A rejection doesn't approve the pipeline; it stays in human_review loop."""
        s = _state_at_step(minimal_cfo_state,
                           validated_data={"revenue": 1},
                           kpi_metrics=healthy_kpis,
                           rag_chunks=[{}],
                           gaap_results={"asc606": {"status": "COMPLIANT"}},
                           ifrs_results={"ifrs15": {"status": "COMPLIANT"}},
                           analysis_narrative="Summary...",
                           requires_human_approval=True,
                           human_decision="rejected",
                           agent_statuses={
                               "data_agent": "complete",
                               "math_engine": "complete",
                               "rag_agent": "complete",
                               "gaap_agent": "complete",
                               "ifrs_agent": "complete",
                               "analysis_agent": "complete",
                           })
        result = route_from_supervisor(s)
        # rejected != "approved", so HITL is still required
        assert result == "human_review"
