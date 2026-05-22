"""
Tests for the Human-in-the-Loop (HITL) approval node.

Covers:
- Pending / approved / rejected state transitions
- compute_approval_triggers behavior
- Audit log entries for each decision type
- Edge cases: no triggers, all triggers at once
"""
import copy
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.agents.human_loop_node import compute_approval_triggers, human_review_node


@pytest.fixture
def hitl_state_base(minimal_cfo_state, healthy_kpis, healthy_variance, healthy_runway):
    """State that has completed math/GAAP/IFRS and is ready for HITL evaluation."""
    from backend.compliance.gaap_engine import GAAPEngine
    from backend.compliance.ifrs_engine import IFRSEngine

    data = {
        "revenue": 12_500_000,
        "cogs": 5_225_000,
        "goodwill": 5_000_000,
        "goodwill_impairment_test_date": "2026-01-15",
        "rou_assets": 2_400_000,
        "lease_liability": 2_200_000,
        "interest_cash_flow_classification": "operating",
        "revenue_recognition_policy": "5-step ASC 606",
    }
    gaap = GAAPEngine().check_all(data, healthy_kpis, healthy_variance, healthy_runway)
    ifrs = IFRSEngine().check_all(data, healthy_kpis, healthy_variance, healthy_runway)

    state = copy.deepcopy(minimal_cfo_state)
    state.update({
        "kpi_metrics":    healthy_kpis,
        "variance_table": healthy_variance,
        "anomaly_flags":  [],
        "gaap_results":   gaap,
        "ifrs_results":   ifrs,
        "requires_human_approval": False,
        "approval_triggers": None,
        "human_decision":  None,
        "human_feedback":  None,
        "approved_by":     None,
    })
    return state


# ── compute_approval_triggers ────────────────────────────────────────────────

class TestComputeApprovalTriggers:
    def test_no_triggers_for_healthy_company(self, hitl_state_base):
        """Healthy variance (<10%) and good gross margin → no triggers (if GAAP all compliant)."""
        state = dict(hitl_state_base)
        state["variance_table"] = {
            "totals": {"variance_pct": 3.0},
            "line_items": {},
            "material_items": [],
        }
        # Use a kpis with good gross margin
        state["kpi_metrics"] = {**state["kpi_metrics"], "gross_margin_pct": 55.0}
        result = compute_approval_triggers(state)
        # Check: if any GAAP/IFRS is non-compliant, that's fine — we're testing variance/gm only
        # The function may still trigger on GAAP issues in the real engine
        # So just verify the variance and gross margin triggers are absent
        trigger_reasons = [t["reason"] for t in result.get("approval_triggers", [])]
        assert "variance_exceeds_10pct" not in trigger_reasons
        assert "gross_margin_below_30pct" not in trigger_reasons

    def test_high_variance_triggers_hitl(self, hitl_state_base):
        state = dict(hitl_state_base)
        state["variance_table"] = {
            "totals": {"variance_pct": 15.0},
            "line_items": {},
            "material_items": ["revenue"],
        }
        result = compute_approval_triggers(state)
        assert result["requires_human_approval"] is True
        reasons = [t["reason"] for t in result["approval_triggers"]]
        assert "variance_exceeds_10pct" in reasons

    def test_low_gross_margin_triggers_hitl(self, hitl_state_base, healthy_variance):
        state = dict(hitl_state_base)
        state["kpi_metrics"] = {**state["kpi_metrics"], "gross_margin_pct": 25.0}
        state["variance_table"] = {
            "totals": {"variance_pct": 3.0},
            "line_items": {},
            "material_items": [],
        }
        result = compute_approval_triggers(state)
        assert result["requires_human_approval"] is True
        reasons = [t["reason"] for t in result["approval_triggers"]]
        assert "gross_margin_below_30pct" in reasons

    def test_triggers_preserved_in_state(self, hitl_state_base):
        state = dict(hitl_state_base)
        state["variance_table"] = {
            "totals": {"variance_pct": 20.0},
            "line_items": {},
            "material_items": [],
        }
        result = compute_approval_triggers(state)
        # Original state fields are preserved
        assert result["kpi_metrics"] is not None
        assert result["gaap_results"] is not None

    def test_multiple_triggers_all_captured(self, hitl_state_base):
        state = dict(hitl_state_base)
        state["variance_table"] = {"totals": {"variance_pct": 20.0}, "line_items": {}, "material_items": []}
        state["kpi_metrics"] = {**state["kpi_metrics"], "gross_margin_pct": 20.0}
        state["anomaly_flags"] = ["Flag A", "Flag B", "Flag C"]
        result = compute_approval_triggers(state)
        assert len(result["approval_triggers"]) >= 2

    def test_missing_kpis_no_crash(self, hitl_state_base):
        state = dict(hitl_state_base)
        state["kpi_metrics"] = {}
        state["variance_table"] = {"totals": {}, "line_items": {}, "material_items": []}
        # Should not raise
        result = compute_approval_triggers(state)
        assert "approval_triggers" in result


# ── human_review_node state transitions ─────────────────────────────────────

class TestHumanReviewNode:
    def test_approved_clears_requires_human_approval(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision="approved",
                     approved_by="Jane CFO", human_feedback="Reviewed and approved.")
        result = human_review_node(state)
        assert result["requires_human_approval"] is False

    def test_approved_sets_agent_status(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision="approved",
                     approved_by="Jane CFO", human_feedback="OK")
        result = human_review_node(state)
        assert result["agent_statuses"]["human_review"] == "approved"

    def test_approved_appends_audit_entry(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision="approved",
                     approved_by="Jane CFO", human_feedback="OK")
        result = human_review_node(state)
        audit_actions = [e["action"] for e in result["audit_log"]]
        assert "approved" in audit_actions

    def test_approved_audit_entry_has_approver(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision="approved",
                     approved_by="Jane CFO", human_feedback="OK")
        result = human_review_node(state)
        approved_entries = [e for e in result["audit_log"] if e.get("action") == "approved"]
        assert len(approved_entries) == 1
        assert approved_entries[0]["approved_by"] == "Jane CFO"

    def test_rejected_adds_error(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision="rejected",
                     approved_by="John Manager", human_feedback="Numbers look wrong.")
        result = human_review_node(state)
        assert any("rejected" in e.lower() for e in result["errors"])

    def test_rejected_sets_agent_status(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision="rejected",
                     approved_by="John Manager", human_feedback="Rejected.")
        result = human_review_node(state)
        assert result["agent_statuses"]["human_review"] == "rejected"

    def test_rejected_audit_entry_captured(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision="rejected",
                     approved_by="John Manager", human_feedback="Rejected.")
        result = human_review_node(state)
        audit_actions = [e["action"] for e in result["audit_log"]]
        assert "rejected" in audit_actions

    def test_pending_sets_decision_to_pending(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision=None)
        result = human_review_node(state)
        assert result["human_decision"] == "pending"

    def test_pending_agent_status_is_pending(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision=None)
        result = human_review_node(state)
        assert result["agent_statuses"]["human_review"] == "pending"

    def test_pending_does_not_clear_requires_approval(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision=None, requires_human_approval=True)
        result = human_review_node(state)
        assert result["requires_human_approval"] is True

    def test_audit_log_grows_on_each_call(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision="approved",
                     approved_by="CFO", human_feedback="OK",
                     audit_log=[{"timestamp": "T", "agent": "math_engine", "action": "done"}])
        result = human_review_node(state)
        assert len(result["audit_log"]) == 2

    def test_empty_feedback_allowed(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision="approved",
                     approved_by="CFO", human_feedback="")
        result = human_review_node(state)
        assert result["requires_human_approval"] is False

    def test_no_approved_by_still_works(self, hitl_state_base):
        state = dict(hitl_state_base, human_decision="approved",
                     approved_by=None, human_feedback="OK")
        result = human_review_node(state)
        assert result["requires_human_approval"] is False
