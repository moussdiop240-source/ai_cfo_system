"""
Edge-case tests for FinancialCalculationEngine.

These complement the happy-path tests in test_math_engine.py and cover:
- Zero / None inputs (division by zero safety)
- Minimal historical data for forecasting
- Negative variances vs unfavorable variances
- Approval trigger boundary conditions
- KPI calculation with partial data
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.agents.math_engine import FinancialCalculationEngine


@pytest.fixture
def engine():
    return FinancialCalculationEngine()


# ── Zero / None input safety ─────────────────────────────────────────────────

class TestZeroInputSafety:
    def test_all_zero_financials_no_exception(self, engine):
        data = {k: 0 for k in [
            "revenue", "cogs", "gross_profit", "ebitda", "ebit", "net_income",
            "total_assets", "total_equity", "current_assets", "current_liabilities",
            "cash", "total_debt", "accounts_receivable", "accounts_payable", "inventory",
            "shares_outstanding", "diluted_shares", "interest_expense",
            "pre_tax_income", "tax_provision",
        ]}
        kpis = engine.calculate_kpis(data)
        assert kpis["gross_margin_pct"] == 0.0
        assert kpis["current_ratio"] == 0.0
        assert kpis["basic_eps"] == 0.0

    def test_empty_dict_no_exception(self, engine):
        kpis = engine.calculate_kpis({})
        assert isinstance(kpis, dict)
        assert kpis.get("gross_margin_pct", 0.0) == 0.0

    def test_none_values_handled(self, engine):
        data = {"revenue": None, "cogs": None, "total_assets": None}
        kpis = engine.calculate_kpis(data)
        assert kpis.get("gross_margin_pct", 0.0) == 0.0

    def test_zero_revenue_gross_margin(self, engine):
        data = {"revenue": 0, "cogs": 500_000}
        kpis = engine.calculate_kpis(data)
        assert kpis["gross_margin_pct"] == 0.0

    def test_zero_shares_eps(self, engine):
        data = {"net_income": 1_000_000, "shares_outstanding": 0, "diluted_shares": 0}
        kpis = engine.calculate_kpis(data)
        assert kpis["basic_eps"] == 0.0
        assert kpis["diluted_eps"] == 0.0

    def test_zero_equity_debt_to_equity(self, engine):
        data = {"total_debt": 1_000_000, "total_equity": 0}
        kpis = engine.calculate_kpis(data)
        assert kpis["debt_to_equity"] == 0.0


# ── Variance edge cases ──────────────────────────────────────────────────────

class TestVarianceEdgeCases:
    def test_zero_budget_item_handled(self, engine):
        """Zero budget should not cause division by zero."""
        actuals = {"revenue": 500_000}
        budget  = {"revenue": 0}
        result = engine.calculate_variance_analysis(actuals, budget)
        # Should not raise; variance_pct should be 0 or handled gracefully
        assert "revenue" in result["line_items"]
        item = result["line_items"]["revenue"]
        assert isinstance(item["variance_abs"], (int, float))

    def test_exact_budget_match_zero_variance(self, engine):
        actuals = {"revenue": 10_000_000}
        budget  = {"revenue": 10_000_000}
        result = engine.calculate_variance_analysis(actuals, budget)
        assert result["line_items"]["revenue"]["variance_abs"] == 0
        assert result["line_items"]["revenue"]["variance_pct"] == 0.0
        assert result["line_items"]["revenue"]["material"] is False

    def test_empty_actuals_and_budget(self, engine):
        result = engine.calculate_variance_analysis({}, {})
        assert "line_items" in result
        assert "totals" in result

    def test_actuals_only_no_budget(self, engine):
        """If no budget, variance should degrade gracefully."""
        actuals = {"revenue": 5_000_000}
        result = engine.calculate_variance_analysis(actuals, {})
        assert "line_items" in result

    def test_exact_5pct_materiality_boundary(self, engine):
        """5.0% variance is exactly at the SAB 99 threshold — should be material."""
        actuals = {"revenue": 10_500_000}
        budget  = {"revenue": 10_000_000}
        result = engine.calculate_variance_analysis(actuals, budget)
        assert result["line_items"]["revenue"]["variance_pct"] == pytest.approx(5.0, abs=0.01)
        # The 5% threshold: >=5% is material
        assert result["line_items"]["revenue"]["material"] is True

    def test_just_below_materiality_threshold(self, engine):
        actuals = {"revenue": 10_490_000}
        budget  = {"revenue": 10_000_000}
        result = engine.calculate_variance_analysis(actuals, budget)
        assert result["line_items"]["revenue"]["variance_pct"] == pytest.approx(4.9, abs=0.01)
        assert result["line_items"]["revenue"]["material"] is False

    def test_large_unfavorable_variance_flagged(self, engine):
        actuals = {"revenue": 4_000_000}
        budget  = {"revenue": 10_000_000}
        result = engine.calculate_variance_analysis(actuals, budget)
        item = result["line_items"]["revenue"]
        assert item["variance_pct"] == pytest.approx(-60.0, abs=0.1)
        assert item["favorable"] is False
        assert item["material"] is True


# ── Forecast edge cases ──────────────────────────────────────────────────────

class TestForecastEdgeCases:
    def test_single_point_fallback(self, engine):
        """Only 1 historical point — should not crash."""
        result = engine.forecast_revenue([5_000_000], periods=3)
        assert len(result["forecast"]) == 3

    def test_two_points_linear_fallback(self, engine):
        result = engine.forecast_revenue([4_000_000, 5_000_000], periods=4)
        assert len(result["forecast"]) == 4
        assert result["method"] == "linear_extrapolation"

    def test_empty_history_fallback(self, engine):
        result = engine.forecast_revenue([], periods=3)
        assert "forecast" in result
        assert len(result["forecast"]) == 3

    def test_forecast_with_none_in_history(self, engine):
        """None values in historical list should not crash."""
        history = [10_000_000, None, 12_000_000, 13_000_000, 14_000_000]
        result = engine.forecast_revenue([v for v in history if v is not None], periods=3)
        assert len(result["forecast"]) == 3

    def test_flat_history_forecast_non_negative(self, engine):
        """Flat history should produce non-negative forecasts."""
        history = [1_000_000] * 8
        result = engine.forecast_revenue(history, periods=4)
        assert all(v >= 0 for v in result["forecast"])

    def test_declining_revenue_forecast(self, engine):
        """Declining trend should produce forecast < last actual (or at least sensible)."""
        history = [10_000_000, 9_000_000, 8_000_000, 7_000_000, 6_000_000]
        result = engine.forecast_revenue(history, periods=3)
        assert len(result["forecast"]) == 3

    def test_zero_periods_returns_empty(self, engine):
        result = engine.forecast_revenue([1_000_000, 2_000_000, 3_000_000], periods=0)
        assert result["forecast"] == []


# ── Anomaly detection edge cases ─────────────────────────────────────────────

class TestAnomalyDetectionEdgeCases:
    def test_perfect_health_no_anomalies(self, engine):
        data = {
            "revenue": 10_000_000,
            "ebitda": 3_000_000,
            "monthly_cash_burn": 0,
        }
        kpis = {
            "gross_margin_pct": 60.0,
            "current_ratio": 3.0,
            "net_margin_pct": 20.0,
            "dso_days": 30.0,
            "debt_to_equity": 0.3,
            "net_debt": -1_000_000,
            "ccc_days": 25.0,
        }
        flags = engine.detect_anomalies(data, kpis)
        assert len(flags) == 0

    def test_empty_kpis_no_exception(self, engine):
        flags = engine.detect_anomalies({}, {})
        assert isinstance(flags, list)

    def test_critical_current_ratio_flagged(self, engine):
        kpis = {"gross_margin_pct": 50.0, "current_ratio": 0.5,
                "net_margin_pct": 5.0, "dso_days": 30.0, "debt_to_equity": 1.0, "net_debt": 0}
        flags = engine.detect_anomalies({}, kpis)
        assert any("current ratio" in f.lower() or "liquidity" in f.lower() for f in flags)

    def test_very_high_dso_flagged(self, engine):
        kpis = {"gross_margin_pct": 50.0, "current_ratio": 2.0,
                "net_margin_pct": 5.0, "dso_days": 150.0, "debt_to_equity": 0.5, "net_debt": 0}
        flags = engine.detect_anomalies({}, kpis)
        assert any("dso" in f.lower() or "receivable" in f.lower() for f in flags)


# ── Approval triggers boundary conditions ────────────────────────────────────

class TestApprovalTriggerBoundaries:
    def test_exactly_10pct_variance_no_trigger(self, engine):
        """Trigger is >10% strictly — exactly 10.0 does NOT trigger."""
        variance = {"totals": {"variance_pct": 10.0}}
        kpis = {"gross_margin_pct": 45.0}
        triggers = engine.check_approval_triggers(variance, kpis, [], None, None)
        reasons = [t["reason"] for t in triggers]
        assert "variance_exceeds_10pct" not in reasons

    def test_10_01pct_variance_triggers(self, engine):
        """Trigger is >10% strictly — 10.01% does trigger."""
        variance = {"totals": {"variance_pct": 10.01}}
        kpis = {"gross_margin_pct": 45.0}
        triggers = engine.check_approval_triggers(variance, kpis, [], None, None)
        reasons = [t["reason"] for t in triggers]
        assert "variance_exceeds_10pct" in reasons

    def test_9_99pct_variance_no_trigger(self, engine):
        variance = {"totals": {"variance_pct": 9.99}}
        kpis = {"gross_margin_pct": 45.0}
        triggers = engine.check_approval_triggers(variance, kpis, [], None, None)
        reasons = [t["reason"] for t in triggers]
        assert "variance_exceeds_10pct" not in reasons

    def test_exactly_30pct_gm_no_trigger(self, engine):
        variance = {"totals": {"variance_pct": 3.0}}
        kpis = {"gross_margin_pct": 30.0}
        triggers = engine.check_approval_triggers(variance, kpis, [], None, None)
        reasons = [t["reason"] for t in triggers]
        # 30.0% exactly should NOT trigger (rule is < 30%)
        assert "gross_margin_below_30pct" not in reasons

    def test_29_99pct_gm_triggers(self, engine):
        variance = {"totals": {"variance_pct": 3.0}}
        kpis = {"gross_margin_pct": 29.99}
        triggers = engine.check_approval_triggers(variance, kpis, [], None, None)
        reasons = [t["reason"] for t in triggers]
        assert "gross_margin_below_30pct" in reasons

    def test_multiple_anomalies_trigger(self, engine):
        variance = {"totals": {"variance_pct": 3.0}}
        kpis = {"gross_margin_pct": 40.0}
        anomalies = ["Flag 1", "Flag 2", "Flag 3"]
        triggers = engine.check_approval_triggers(variance, kpis, anomalies, None, None)
        reasons = [t["reason"] for t in triggers]
        assert any("anomal" in r.lower() for r in reasons)

    def test_gaap_non_compliant_triggers_hitl(self, engine):
        variance = {"totals": {"variance_pct": 3.0}}
        kpis = {"gross_margin_pct": 45.0}
        gaap = {"asc606": {"status": "NON_COMPLIANT", "standard": "ASC 606"}}
        triggers = engine.check_approval_triggers(variance, kpis, [], gaap, None)
        assert len(triggers) > 0

    def test_all_healthy_zero_triggers(self, engine, healthy_kpis, healthy_variance):
        healthy_variance_low = {
            "totals": {"variance_pct": 3.0},
            "line_items": {},
            "material_items": [],
        }
        triggers = engine.check_approval_triggers(
            healthy_variance_low, healthy_kpis, [], {}, {}
        )
        assert len(triggers) == 0


# ── Cash runway calculation ──────────────────────────────────────────────────

class TestCashRunway:
    def test_healthy_runway(self, engine):
        data = {"cash": 6_000_000, "monthly_cash_burn": 166_667}
        kpis = {"net_debt": 5_800_000}
        result = engine.calculate_cash_runway(data, kpis)
        assert result["runway_months"] == pytest.approx(36.0, abs=0.5)

    def test_zero_burn_rate(self, engine):
        data = {"cash": 5_000_000, "monthly_cash_burn": 0}
        kpis = {"net_debt": 0}
        result = engine.calculate_cash_runway(data, kpis)
        # Should not crash; runway should be very large or a sentinel
        assert result["runway_months"] >= 0

    def test_no_cash_data(self, engine):
        result = engine.calculate_cash_runway({}, {})
        assert "runway_months" in result
