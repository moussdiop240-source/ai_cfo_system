"""
Tests for Pydantic schema validation (FinancialDataSchema).

Covers:
- Period format validation (all valid formats + rejection of bad formats)
- Currency validation
- Positive value enforcement
- compute_derived() logic
- LineItem structure
- Edge cases: empty strings, None values, type coercion
"""
import sys
import os

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.schemas.financial import FinancialDataSchema, LineItem


# ── Minimal valid schema ─────────────────────────────────────────────────────

MINIMAL_VALID = {
    "company_name": "TestCo",
    "period": "Q1 2026",
    "currency": "USD",
}


# ── Period format tests ──────────────────────────────────────────────────────

class TestPeriodValidation:
    @pytest.mark.parametrize("period", [
        "Q1 2026", "Q2 2025", "Q3 2024", "Q4 2023",
        "FY 2026", "FY 2024",
        "H1 2026", "H2 2025",
        "January 2026", "February 2025", "March 2024",
        "April 2026", "May 2025", "June 2024",
        "July 2026", "August 2025", "September 2024",
        "October 2026", "November 2025", "December 2024",
        "2026",
    ])
    def test_valid_period_formats(self, period):
        data = {**MINIMAL_VALID, "period": period}
        schema = FinancialDataSchema(**data)
        assert schema.period == period

    @pytest.mark.parametrize("bad_period", [
        "q1 2026",       # lowercase q
        "Quarter 1 2026",
        "2026 Q1",       # reversed
        "Q5 2026",       # invalid quarter
        "Q1 26",         # 2-digit year
        "H3 2026",       # invalid half
        "FY2026",        # missing space
        "",              # empty
        "2026-01",       # ISO format (not accepted)
        "Jan 2026",      # abbreviated month (not accepted)
    ])
    def test_invalid_period_rejected(self, bad_period):
        with pytest.raises(ValidationError, match="Period"):
            FinancialDataSchema(company_name="X", period=bad_period, currency="USD")


# ── Currency tests ───────────────────────────────────────────────────────────

class TestCurrencyValidation:
    @pytest.mark.parametrize("currency", [
        "USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "CNY", "SGD", "HKD",
    ])
    def test_valid_currencies_accepted(self, currency):
        schema = FinancialDataSchema(**{**MINIMAL_VALID, "currency": currency})
        assert schema.currency == currency

    @pytest.mark.parametrize("bad_currency", [
        "usd", "US", "USDX", "BTC", "XYZ", "", "123",
    ])
    def test_invalid_currencies_rejected(self, bad_currency):
        with pytest.raises(ValidationError, match="[Cc]urrency"):
            FinancialDataSchema(company_name="X", period="Q1 2026", currency=bad_currency)


# ── Positive value validation ────────────────────────────────────────────────

class TestPositiveValueValidation:
    def test_negative_revenue_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            FinancialDataSchema(
                company_name="X", period="Q1 2026", currency="USD",
                revenue=-1_000_000,
            )

    def test_negative_cogs_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            FinancialDataSchema(
                company_name="X", period="Q1 2026", currency="USD",
                cogs=-500_000,
            )

    def test_negative_total_assets_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            FinancialDataSchema(
                company_name="X", period="Q1 2026", currency="USD",
                total_assets=-100,
            )

    def test_zero_revenue_allowed(self):
        schema = FinancialDataSchema(
            company_name="X", period="Q1 2026", currency="USD",
            revenue=0,
        )
        assert schema.revenue == 0

    def test_positive_values_accepted(self):
        schema = FinancialDataSchema(
            company_name="X", period="Q1 2026", currency="USD",
            revenue=5_000_000, cogs=2_000_000, total_assets=10_000_000, total_equity=6_000_000,
        )
        assert schema.revenue == 5_000_000


# ── compute_derived() ────────────────────────────────────────────────────────

class TestComputeDerived:
    def test_gross_profit_computed_from_revenue_minus_cogs(self):
        schema = FinancialDataSchema(
            company_name="X", period="Q1 2026", currency="USD",
            revenue=10_000_000, cogs=4_000_000,
        )
        result = schema.compute_derived()
        assert result.gross_profit == 6_000_000

    def test_gross_profit_not_overwritten_if_provided(self):
        schema = FinancialDataSchema(
            company_name="X", period="Q1 2026", currency="USD",
            revenue=10_000_000, cogs=4_000_000, gross_profit=5_500_000,
        )
        result = schema.compute_derived()
        # Provided gross_profit should NOT be overwritten
        assert result.gross_profit == 5_500_000

    def test_ebit_computed_from_ebitda_minus_da(self):
        schema = FinancialDataSchema(
            company_name="X", period="Q1 2026", currency="USD",
            ebitda=3_000_000, depreciation_amortization=400_000,
        )
        result = schema.compute_derived()
        assert result.ebit == 2_600_000

    def test_ebit_not_overwritten_if_provided(self):
        schema = FinancialDataSchema(
            company_name="X", period="Q1 2026", currency="USD",
            ebitda=3_000_000, depreciation_amortization=400_000, ebit=2_900_000,
        )
        result = schema.compute_derived()
        assert result.ebit == 2_900_000

    def test_no_computation_when_inputs_missing(self):
        schema = FinancialDataSchema(company_name="X", period="Q1 2026", currency="USD")
        result = schema.compute_derived()
        assert result.gross_profit is None
        assert result.ebit is None


# ── LineItem model ───────────────────────────────────────────────────────────

class TestLineItem:
    def test_minimal_line_item(self):
        item = LineItem(name="Revenue", actual=5_000_000)
        assert item.name == "Revenue"
        assert item.actual == 5_000_000
        assert item.budget is None

    def test_full_line_item(self):
        item = LineItem(
            name="COGS", actual=2_000_000, budget=1_900_000,
            prior_year=1_800_000, category="cogs", notes="Includes freight",
        )
        assert item.prior_year == 1_800_000
        assert item.category == "cogs"

    def test_line_item_in_schema(self):
        schema = FinancialDataSchema(
            company_name="X", period="Q1 2026", currency="USD",
            line_items=[
                {"name": "Revenue", "actual": 5_000_000, "budget": 4_500_000},
                {"name": "COGS",    "actual": 2_000_000},
            ],
        )
        assert len(schema.line_items) == 2
        assert schema.line_items[0].name == "Revenue"


# ── Full schema round-trip ───────────────────────────────────────────────────

class TestFullSchemaRoundTrip:
    def test_full_schema_round_trip(self, healthy_financials):
        schema = FinancialDataSchema(**{k: v for k, v in healthy_financials.items()
                                        if k != "actuals" and k != "budget"
                                        and k not in ("historical_revenue",)})
        d = schema.model_dump()
        assert d["company_name"] == healthy_financials["company_name"]
        assert d["period"] == healthy_financials["period"]

    def test_schema_to_dict_serializable(self, healthy_financials):
        import json
        schema = FinancialDataSchema(
            company_name=healthy_financials["company_name"],
            period=healthy_financials["period"],
            currency="USD",
            revenue=healthy_financials["revenue"],
        )
        d = schema.model_dump()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_missing_required_fields_raise(self):
        with pytest.raises(ValidationError):
            FinancialDataSchema(period="Q1 2026", currency="USD")  # missing company_name
