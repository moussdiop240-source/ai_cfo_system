"""
Performance benchmarks for the AI CFO System.

These tests assert that deterministic, non-LLM components meet latency SLOs.
All thresholds are conservative to pass in CI (slow GitHub runners included).

SLO targets:
  - Schema validation:      < 200 ms per call
  - Math engine (KPIs):     < 500 ms per call
  - GAAP engine (12 stds):  < 500 ms per call
  - IFRS engine (12 stds):  < 500 ms per call
  - Variance analysis:      < 300 ms per call
  - 100 × schema validate:  < 5 000 ms total
  - 50  × GAAP check:       < 10 000 ms total
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Shared financial fixture ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fin() -> Dict[str, Any]:
    """Representative mid-size SaaS company financials."""
    return {
        "company_name": "PerfTestCo",
        "period": "Q1 2026",
        "currency": "USD",
        "revenue": 12_500_000,
        "cogs": 5_250_000,
        "gross_profit": 7_250_000,
        "operating_expenses": 4_500_000,
        "rd_expense": 1_800_000,
        "sg_a_expense": 2_100_000,
        "ebitda": 2_800_000,
        "depreciation_amortization": 350_000,
        "ebit": 2_450_000,
        "interest_expense": 120_000,
        "pre_tax_income": 2_330_000,
        "tax_provision": 560_000,
        "net_income": 1_770_000,
        "total_assets": 28_000_000,
        "current_assets": 9_500_000,
        "cash": 4_200_000,
        "accounts_receivable": 3_100_000,
        "inventory": 800_000,
        "total_equity": 14_500_000,
        "current_liabilities": 4_100_000,
        "accounts_payable": 1_200_000,
        "total_debt": 6_000_000,
        "long_term_debt": 4_800_000,
        "cash_from_operations": 2_100_000,
        "capex": 450_000,
        "free_cash_flow": 1_650_000,
        "monthly_cash_burn": 0,
        "shares_outstanding": 10_000_000,
        "diluted_shares": 10_500_000,
        "arr": 48_000_000,
        "nrr_pct": 118.0,
        "churn_rate_pct": 4.2,
        "headcount": 210,
        "jurisdiction": "United States",
        "listing_exchange": "NASDAQ",
        "industry": "Technology",
        "rd_dev_capitalizable_pct": 30,
        "goodwill": 2_000_000,
        "rou_assets": 1_200_000,
        "lease_liability": 1_180_000,
        "operating_lease_expense": 360_000,
    }


@pytest.fixture(scope="module")
def kpis(fin):
    from backend.math_engine.kpi_calculator import KPICalculator
    return KPICalculator().calculate_all(fin)


@pytest.fixture(scope="module")
def variance():
    return {
        "revenue":        {"actual": 12_500_000, "budget": 11_000_000, "variance": 1_500_000,  "pct": 13.6},
        "cogs":           {"actual": 5_250_000,  "budget": 4_800_000,  "variance": -450_000,   "pct": -9.4},
        "gross_profit":   {"actual": 7_250_000,  "budget": 6_200_000,  "variance": 1_050_000,  "pct": 16.9},
        "ebitda":         {"actual": 2_800_000,  "budget": 2_200_000,  "variance": 600_000,    "pct": 27.3},
    }


@pytest.fixture(scope="module")
def runway():
    return {"months": 24.0, "cash": 4_200_000, "burn": 0}


# ── Schema validation performance ─────────────────────────────────────────────

class TestSchemaValidationPerformance:
    SLO_SINGLE_MS = 200
    SLO_BATCH_MS  = 5_000

    def test_single_validation_under_slo(self, fin):
        from backend.schemas.financial import FinancialDataSchema
        t0 = time.perf_counter()
        FinancialDataSchema(**fin)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < self.SLO_SINGLE_MS, f"Schema validation took {elapsed:.1f}ms (SLO={self.SLO_SINGLE_MS}ms)"

    def test_100_validations_under_batch_slo(self, fin):
        from backend.schemas.financial import FinancialDataSchema
        t0 = time.perf_counter()
        for _ in range(100):
            FinancialDataSchema(**fin)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < self.SLO_BATCH_MS, f"100 validations took {elapsed:.1f}ms (SLO={self.SLO_BATCH_MS}ms)"

    def test_validation_throughput_above_floor(self, fin):
        """At least 5 validations per second."""
        from backend.schemas.financial import FinancialDataSchema
        count = 0
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            FinancialDataSchema(**fin)
            count += 1
        assert count >= 5, f"Only {count} validations/sec — expected >= 5"


# ── KPI calculation performance ───────────────────────────────────────────────

class TestKPICalculatorPerformance:
    SLO_SINGLE_MS = 500

    def test_single_kpi_calculation_under_slo(self, fin):
        from backend.math_engine.kpi_calculator import KPICalculator
        calc = KPICalculator()
        t0 = time.perf_counter()
        calc.calculate_all(fin)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < self.SLO_SINGLE_MS, f"KPI calc took {elapsed:.1f}ms (SLO={self.SLO_SINGLE_MS}ms)"

    def test_kpi_result_is_dict(self, fin):
        from backend.math_engine.kpi_calculator import KPICalculator
        result = KPICalculator().calculate_all(fin)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_10_kpi_calcs_total_under_3s(self, fin):
        from backend.math_engine.kpi_calculator import KPICalculator
        calc = KPICalculator()
        t0 = time.perf_counter()
        for _ in range(10):
            calc.calculate_all(fin)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 3_000, f"10 KPI calcs took {elapsed:.1f}ms"


# ── GAAP engine performance ───────────────────────────────────────────────────

class TestGAAPEnginePerformance:
    SLO_SINGLE_MS = 500
    SLO_BATCH_MS  = 10_000

    def test_single_gaap_check_under_slo(self, fin, kpis, variance, runway):
        from backend.compliance.gaap_engine import GAAPEngine
        engine = GAAPEngine()
        t0 = time.perf_counter()
        engine.check_all(fin, kpis, variance, runway)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < self.SLO_SINGLE_MS, f"GAAP check took {elapsed:.1f}ms (SLO={self.SLO_SINGLE_MS}ms)"

    def test_gaap_returns_12_standards(self, fin, kpis, variance, runway):
        from backend.compliance.gaap_engine import GAAPEngine
        results = GAAPEngine().check_all(fin, kpis, variance, runway)
        assert len(results) == 12

    def test_50_gaap_checks_under_batch_slo(self, fin, kpis, variance, runway):
        from backend.compliance.gaap_engine import GAAPEngine
        engine = GAAPEngine()
        t0 = time.perf_counter()
        for _ in range(50):
            engine.check_all(fin, kpis, variance, runway)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < self.SLO_BATCH_MS, f"50 GAAP checks took {elapsed:.1f}ms (SLO={self.SLO_BATCH_MS}ms)"


# ── IFRS engine performance ───────────────────────────────────────────────────

class TestIFRSEnginePerformance:
    SLO_SINGLE_MS = 500

    def test_single_ifrs_check_under_slo(self, fin, kpis, variance, runway):
        from backend.compliance.ifrs_engine import IFRSEngine
        engine = IFRSEngine()
        t0 = time.perf_counter()
        engine.check_all(fin, kpis, variance, runway)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < self.SLO_SINGLE_MS, f"IFRS check took {elapsed:.1f}ms (SLO={self.SLO_SINGLE_MS}ms)"

    def test_ifrs_returns_12_standards(self, fin, kpis, variance, runway):
        from backend.compliance.ifrs_engine import IFRSEngine
        results = IFRSEngine().check_all(fin, kpis, variance, runway)
        assert len(results) == 12

    def test_50_ifrs_checks_under_batch_slo(self, fin, kpis, variance, runway):
        from backend.compliance.ifrs_engine import IFRSEngine
        engine = IFRSEngine()
        t0 = time.perf_counter()
        for _ in range(50):
            engine.check_all(fin, kpis, variance, runway)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 10_000, f"50 IFRS checks took {elapsed:.1f}ms"


# ── Variance analysis performance ─────────────────────────────────────────────

class TestVarianceAnalysisPerformance:
    SLO_SINGLE_MS = 300

    def test_variance_calculation_under_slo(self, fin):
        from backend.math_engine.variance_analyzer import VarianceAnalyzer
        analyzer = VarianceAnalyzer()
        budget = {
            "revenue": 11_000_000,
            "cogs":    4_800_000,
            "ebitda":  2_200_000,
        }
        t0 = time.perf_counter()
        analyzer.analyze(fin, budget)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < self.SLO_SINGLE_MS, f"Variance analysis took {elapsed:.1f}ms (SLO={self.SLO_SINGLE_MS}ms)"


# ── PDF generation performance ────────────────────────────────────────────────

class TestPDFGenerationPerformance:
    SLO_SINGLE_MS = 3_000  # PDF generation can be heavier

    SAMPLE_REPORT = """# PerfTestCo — Q1 2026 Board Report

## 1. EXECUTIVE SUMMARY
Revenue of $12.5M — favorable $1.5M variance to $11.0M budget.
Gross margin 58.2%, EBITDA margin 22.4%.

## 2. REVENUE PERFORMANCE
- ASC 606 five-step model applied
- IFRS 15 performance obligations met

## 3. GAAP COMPLIANCE
ASC 606, ASC 842, ASC 350 — all COMPLIANT.
"""

    def test_pdf_generation_under_slo(self):
        from backend.reporting.pdf_generator import generate_pdf
        t0 = time.perf_counter()
        pdf_bytes = generate_pdf(
            self.SAMPLE_REPORT,
            company_name="PerfTestCo",
            period="Q1 2026",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < self.SLO_SINGLE_MS, f"PDF generation took {elapsed:.1f}ms (SLO={self.SLO_SINGLE_MS}ms)"
        assert len(pdf_bytes) > 1000

    def test_5_pdfs_under_10s(self):
        from backend.reporting.pdf_generator import generate_pdf
        t0 = time.perf_counter()
        for _ in range(5):
            generate_pdf(self.SAMPLE_REPORT, company_name="PerfTestCo", period="Q1 2026")
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 10_000, f"5 PDF generations took {elapsed:.1f}ms"
