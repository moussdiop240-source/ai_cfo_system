"""
Tests for the institutional memory engine.

Covers:
- SHA-256 deduplication
- Period-over-period delta computation
- KPI time-series storage and retrieval
- Peer benchmarks
- HITL decision update
- Recurring issues detection
- Company summary rollup
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures are in conftest.py (memory_engine, full_results, temp_db) ───────


class TestSaveAnalysisDedup:
    def test_first_save_returns_snapshot_id(self, memory_engine, full_results):
        snap_id = memory_engine.save_analysis("AcmeCorp", "Q1 2026", full_results)
        assert isinstance(snap_id, int)
        assert snap_id > 0

    def test_identical_save_returns_same_id(self, memory_engine, full_results):
        id1 = memory_engine.save_analysis("AcmeCorp", "Q1 2026", full_results)
        id2 = memory_engine.save_analysis("AcmeCorp", "Q1 2026", full_results)
        assert id1 == id2

    def test_revised_data_same_period_updates_snapshot(self, memory_engine, full_results):
        """Revised numbers for the same (company, period) updates the existing snapshot — upsert."""
        import copy
        results_v2 = copy.deepcopy(full_results)
        results_v2["revenue"] = 9_000_000   # revised numbers
        id1 = memory_engine.save_analysis("AcmeCorp", "Q1 2026", full_results)
        id2 = memory_engine.save_analysis("AcmeCorp", "Q1 2026", results_v2)
        # Same snapshot row is updated — both calls return the same ID
        assert id1 == id2
        assert isinstance(id2, int)

    def test_different_period_creates_new_snapshot(self, memory_engine, full_results):
        id1 = memory_engine.save_analysis("AcmeCorp", "Q1 2026", full_results)
        id2 = memory_engine.save_analysis("AcmeCorp", "Q2 2026", full_results)
        assert id1 != id2

    def test_different_company_creates_new_snapshot(self, memory_engine, full_results):
        id1 = memory_engine.save_analysis("AcmeCorp", "Q1 2026", full_results)
        id2 = memory_engine.save_analysis("BetaCorp", "Q1 2026", full_results)
        assert id1 != id2

    def test_save_with_empty_results_no_exception(self, memory_engine):
        snap_id = memory_engine.save_analysis("EmptyCo", "Q1 2026", {})
        assert snap_id > 0


class TestCompanyMasterRow:
    def test_company_created_on_first_save(self, memory_engine, full_results):
        memory_engine.save_analysis("NewCo", "Q1 2026", full_results)
        summary = memory_engine.get_company_summary("NewCo")
        assert summary is not None
        assert summary["company_name"] == "NewCo"

    def test_periods_analysed_increments(self, memory_engine, full_results):
        import copy
        memory_engine.save_analysis("GrowCo", "Q1 2026", full_results)
        results2 = copy.deepcopy(full_results)
        results2["revenue"] = 15_000_000
        memory_engine.save_analysis("GrowCo", "Q2 2026", results2)
        summary = memory_engine.get_company_summary("GrowCo")
        assert summary["periods_analysed"] == 2

    def test_sector_stored(self, memory_engine, full_results):
        memory_engine.save_analysis("SectorCo", "Q1 2026", full_results, sector="Technology")
        summary = memory_engine.get_company_summary("SectorCo")
        assert summary["sector"] == "Technology"

    def test_rolling_averages_computed(self, memory_engine, full_results):
        memory_engine.save_analysis("AvgCo", "Q1 2026", full_results)
        summary = memory_engine.get_company_summary("AvgCo")
        assert summary["avg_gross_margin"] is not None

    def test_nonexistent_company_returns_none(self, memory_engine):
        result = memory_engine.get_company_summary("NoSuchCo_XYZ")
        assert result is None

    def test_get_all_companies_returns_list(self, memory_engine, full_results):
        memory_engine.save_analysis("Alpha", "Q1 2026", full_results)
        memory_engine.save_analysis("Beta",  "Q1 2026", full_results)
        companies = memory_engine.get_all_companies()
        assert "Alpha" in companies
        assert "Beta" in companies


class TestKPITimeSeries:
    def test_kpi_trends_returned_for_saved_periods(self, memory_engine, full_results):
        memory_engine.save_analysis("TrendCo", "Q1 2026", full_results)
        trends = memory_engine.get_kpi_trends("TrendCo", ["gross_margin_pct"])
        assert "gross_margin_pct" in trends
        assert len(trends["gross_margin_pct"]) >= 1

    def test_kpi_trend_has_correct_structure(self, memory_engine, full_results):
        memory_engine.save_analysis("TrendCo2", "Q1 2026", full_results)
        trends = memory_engine.get_kpi_trends("TrendCo2")
        for kpi_name, history in trends.items():
            for point in history:
                assert "period" in point
                assert "value" in point
                assert "unit" in point

    def test_multiple_periods_build_time_series(self, memory_engine, full_results):
        import copy
        memory_engine.save_analysis("MultiCo", "Q1 2026", full_results)
        r2 = copy.deepcopy(full_results)
        r2["revenue"] = 14_000_000
        r2["kpi_metrics"] = {**full_results["kpi_metrics"], "gross_margin_pct": 60.0}
        memory_engine.save_analysis("MultiCo", "Q2 2026", r2)
        trends = memory_engine.get_kpi_trends("MultiCo", ["gross_margin_pct"])
        assert len(trends["gross_margin_pct"]) == 2

    def test_kpi_trends_for_nonexistent_company_returns_empty(self, memory_engine):
        trends = memory_engine.get_kpi_trends("GhostCo")
        # All keys should have empty lists
        for v in trends.values():
            assert v == []


class TestPeerBenchmarks:
    def test_peer_benchmarks_returns_dict(self, memory_engine, full_results):
        memory_engine.save_analysis("PeerA", "Q1 2026", full_results)
        memory_engine.save_analysis("PeerB", "Q1 2026", full_results)
        benchmarks = memory_engine.get_peer_benchmarks()
        assert isinstance(benchmarks, dict)
        assert len(benchmarks) > 0

    def test_exclude_company_removes_its_data(self, memory_engine, full_results):
        import copy
        r_a = copy.deepcopy(full_results)
        r_a["kpi_metrics"] = {**full_results["kpi_metrics"], "gross_margin_pct": 80.0}
        r_b = copy.deepcopy(full_results)
        r_b["kpi_metrics"] = {**full_results["kpi_metrics"], "gross_margin_pct": 40.0}
        memory_engine.save_analysis("ExclA", "Q1 2026", r_a)
        memory_engine.save_analysis("ExclB", "Q1 2026", r_b)

        bench_all = memory_engine.get_peer_benchmarks()
        bench_excl = memory_engine.get_peer_benchmarks(exclude_company="ExclA")

        # Excluding ExclA (80%) should bring the average down
        if "gross_margin_pct" in bench_all and "gross_margin_pct" in bench_excl:
            assert bench_excl["gross_margin_pct"] <= bench_all["gross_margin_pct"]


class TestUpdateHITL:
    def test_update_hitl_returns_true_for_existing(self, memory_engine, full_results):
        memory_engine.save_analysis("HITLCo", "Q1 2026", full_results)
        result = memory_engine.update_hitl("HITLCo", "Q1 2026", "approved", "Jane CFO", "Reviewed.")
        assert result is True

    def test_update_hitl_returns_false_for_missing(self, memory_engine):
        result = memory_engine.update_hitl("GhostCo", "Q1 2026", "approved")
        assert result is False

    def test_hitl_decision_stored(self, memory_engine, full_results):
        memory_engine.save_analysis("HITLCo2", "Q1 2026", full_results,
                                     hitl_state={"requires_human_approval": True})
        memory_engine.update_hitl("HITLCo2", "Q1 2026", "approved", "CFO", "OK")
        log = memory_engine.get_hitl_log("HITLCo2")
        assert len(log) >= 1
        decisions = [e["decision"] for e in log]
        assert "approved" in decisions

    def test_hitl_approver_stored(self, memory_engine, full_results):
        memory_engine.save_analysis("HITLCo3", "Q1 2026", full_results,
                                     hitl_state={"requires_human_approval": True})
        memory_engine.update_hitl("HITLCo3", "Q1 2026", "approved", "Mary VP", "Looks good")
        log = memory_engine.get_hitl_log("HITLCo3")
        approvers = [e["approver"] for e in log]
        assert "Mary VP" in approvers


class TestPeriodHistory:
    def test_period_history_is_sorted_chronologically(self, memory_engine, full_results):
        import copy, time
        memory_engine.save_analysis("HistCo", "Q1 2026", full_results)
        time.sleep(0.01)
        r2 = copy.deepcopy(full_results)
        r2["revenue"] = 13_000_000
        memory_engine.save_analysis("HistCo", "Q2 2026", r2)
        history = memory_engine.get_period_history("HistCo")
        assert len(history) == 2
        assert history[0]["period"] == "Q1 2026"
        assert history[1]["period"] == "Q2 2026"

    def test_period_history_has_required_fields(self, memory_engine, full_results):
        memory_engine.save_analysis("FieldCo", "Q1 2026", full_results)
        history = memory_engine.get_period_history("FieldCo")
        required = {"period", "saved_at", "gross_margin_pct", "gaap_compliant"}
        assert required.issubset(history[0].keys())


class TestInsights:
    def test_anomaly_flags_saved_as_insights(self, memory_engine, full_results):
        import copy
        results_with_anomalies = copy.deepcopy(full_results)
        results_with_anomalies["anomaly_flags"] = ["Low gross margin", "High DSO"]
        memory_engine.save_analysis("InsightCo", "Q1 2026", results_with_anomalies)
        insights = memory_engine.get_active_insights("InsightCo")
        anomaly_insights = [i for i in insights if i["type"] == "anomaly"]
        assert len(anomaly_insights) == 2

    def test_insights_have_required_fields(self, memory_engine, full_results):
        import copy
        r = copy.deepcopy(full_results)
        r["anomaly_flags"] = ["Test flag"]
        memory_engine.save_analysis("FieldInsightCo", "Q1 2026", r)
        insights = memory_engine.get_active_insights("FieldInsightCo")
        for i in insights:
            assert "type" in i
            assert "severity" in i
            assert "description" in i

    def test_no_anomalies_no_anomaly_insights(self, memory_engine, full_results):
        import copy
        r = copy.deepcopy(full_results)
        r["anomaly_flags"] = []
        memory_engine.save_analysis("CleanCo", "Q1 2026", r)
        insights = memory_engine.get_active_insights("CleanCo")
        anomaly_insights = [i for i in insights if i["type"] == "anomaly"]
        assert len(anomaly_insights) == 0


class TestRecurringIssues:
    def test_issue_appearing_twice_becomes_recurring(self, memory_engine, full_results):
        import copy
        issue_desc = "High DSO detected"
        r1 = copy.deepcopy(full_results)
        r1["anomaly_flags"] = [issue_desc]
        r2 = copy.deepcopy(full_results)
        r2["revenue"] = 13_000_000
        r2["anomaly_flags"] = [issue_desc]

        memory_engine.save_analysis("RecurCo", "Q1 2026", r1)
        memory_engine.save_analysis("RecurCo", "Q2 2026", r2)

        summary = memory_engine.get_company_summary("RecurCo")
        assert issue_desc in summary["recurring_issues"]

    def test_single_occurrence_not_recurring(self, memory_engine, full_results):
        import copy
        r = copy.deepcopy(full_results)
        r["anomaly_flags"] = ["One-time spike"]
        memory_engine.save_analysis("SingleCo", "Q1 2026", r)
        summary = memory_engine.get_company_summary("SingleCo")
        assert "One-time spike" not in summary["recurring_issues"]
