"""
Shared pytest fixtures for the AI CFO test suite.
All fixtures are deterministic — no LLM calls, no network.
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Financial data fixtures ──────────────────────────────────────────────────

@pytest.fixture
def healthy_financials():
    """A company in good financial health — should be COMPLIANT on all checks."""
    return {
        "company_name": "HealthyCo Inc",
        "period": "Q1 2026",
        "currency": "USD",
        "revenue": 12_500_000,
        "cogs": 5_225_000,
        "gross_profit": 7_275_000,
        "ebitda": 2_800_000,
        "ebit": 2_460_000,
        "net_income": 1_890_000,
        "total_assets": 45_000_000,
        "total_equity": 28_000_000,
        "current_assets": 18_000_000,
        "current_liabilities": 8_500_000,
        "cash": 6_200_000,
        "total_debt": 12_000_000,
        "accounts_receivable": 4_200_000,
        "accounts_payable": 2_800_000,
        "inventory": 1_800_000,
        "goodwill": 5_000_000,
        "goodwill_impairment_test_date": "2026-01-15",
        "rou_assets": 2_400_000,
        "lease_liability": 2_200_000,
        "shares_outstanding": 4_500_000,
        "diluted_shares": 4_750_000,
        "interest_expense": 420_000,
        "pre_tax_income": 2_040_000,
        "tax_provision": 150_000,
        "revenue_recognition_policy": "5-step model per ASC 606",
        "deferred_revenue": 800_000,
        "interest_cash_flow_classification": "operating",
        "cash_from_operations": 2_100_000,
        "historical_revenue": [9_000_000, 9_800_000, 10_500_000, 11_200_000, 12_000_000, 12_500_000],
        "actuals": {"revenue": 12_500_000, "cogs": 5_225_000},
        "budget": {"revenue": 11_000_000, "cogs": 5_000_000},
    }


@pytest.fixture
def distressed_financials():
    """A company with multiple financial health issues."""
    return {
        "company_name": "DistressedCo LLC",
        "period": "Q2 2026",
        "currency": "USD",
        "revenue": 2_000_000,
        "cogs": 1_800_000,
        "gross_profit": 200_000,
        "ebitda": -150_000,
        "ebit": -180_000,
        "net_income": -250_000,
        "total_assets": 5_000_000,
        "total_equity": 500_000,
        "current_assets": 800_000,
        "current_liabilities": 1_200_000,   # < current_assets → current ratio < 1
        "cash": 300_000,
        "total_debt": 4_000_000,
        "accounts_receivable": 400_000,
        "accounts_payable": 600_000,
        "inventory": 100_000,
        "shares_outstanding": 1_000_000,
        "diluted_shares": 1_000_000,
        "interest_expense": 80_000,
        "pre_tax_income": -260_000,
        "tax_provision": 0,
        "monthly_cash_burn": 100_000,
        "interest_cash_flow_classification": "operating",
        "actuals": {"revenue": 2_000_000, "cogs": 1_800_000},
        "budget": {"revenue": 5_000_000, "cogs": 2_500_000},  # massive miss
    }


@pytest.fixture
def healthy_kpis():
    return {
        "gross_margin_pct":   58.2,
        "net_margin_pct":     15.1,
        "ebitda_margin_pct":  22.4,
        "current_ratio":      2.12,
        "quick_ratio":        1.93,
        "debt_to_equity":     0.43,
        "dso_days":           49.0,
        "dio_days":           63.0,
        "dpo_days":           82.0,
        "ccc_days":           30.0,
        "basic_eps":          0.42,
        "diluted_eps":        0.40,
        "effective_tax_rate": 21.0,
        "net_debt":           5_800_000,
        "roe_pct":            6.75,
        "fcf_to_net_income":  1.11,
    }


@pytest.fixture
def healthy_variance():
    return {
        "totals": {"actual": 17_725_000, "budget": 16_000_000, "variance_pct": 10.8},
        "line_items": {
            "revenue": {"actual": 12_500_000, "budget": 11_000_000, "variance_abs": 1_500_000,
                        "variance_pct": 13.64, "material": True, "favorable": True},
            "cogs":    {"actual":  5_225_000, "budget":  5_000_000, "variance_abs":   225_000,
                        "variance_pct":  4.5, "material": False, "favorable": False},
        },
        "material_items": ["revenue"],
        "method": "pandas_exact_arithmetic",
    }


@pytest.fixture
def healthy_runway():
    return {"runway_months": 36.0, "monthly_burn": 172_222, "status": "ADEQUATE"}


@pytest.fixture
def math_engine():
    from backend.agents.math_engine import FinancialCalculationEngine
    return FinancialCalculationEngine()


@pytest.fixture
def gaap_engine():
    from backend.compliance.gaap_engine import GAAPEngine
    return GAAPEngine()


@pytest.fixture
def ifrs_engine():
    from backend.compliance.ifrs_engine import IFRSEngine
    return IFRSEngine()


@pytest.fixture
def temp_db():
    """Provide a temporary SQLite DB URL for memory engine tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    url = f"sqlite:///{path}"
    yield url
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def memory_engine(temp_db):
    from backend.memory.engine import MemoryEngine
    return MemoryEngine(db_url=temp_db)


@pytest.fixture
def full_results(healthy_financials, healthy_kpis, healthy_variance, healthy_runway):
    """Simulated full pipeline results dict (as saved to memory)."""
    from backend.compliance.gaap_engine import GAAPEngine
    from backend.compliance.ifrs_engine import IFRSEngine
    gaap = GAAPEngine().check_all(healthy_financials, healthy_kpis, healthy_variance, healthy_runway)
    ifrs = IFRSEngine().check_all(healthy_financials, healthy_kpis, healthy_variance, healthy_runway)
    return {
        "revenue":       healthy_financials["revenue"],
        "kpi_metrics":   healthy_kpis,
        "variance_table": healthy_variance,
        "gaap_results":  gaap,
        "ifrs_results":  ifrs,
        "anomaly_flags": [],
        "forecast_outputs": {"forecast": [13_000_000, 13_500_000], "r2": 0.98, "method": "ensemble"},
    }


@pytest.fixture
def temp_log(tmp_path):
    """Temporary security audit log file."""
    return str(tmp_path / "security_audit.jsonl")


@pytest.fixture
def security_logger(temp_log):
    from backend.security.audit_logger import SecurityAuditLogger
    return SecurityAuditLogger(log_path=temp_log)


@pytest.fixture
def sanitizer():
    from backend.security.input_sanitizer import InputSanitizer
    return InputSanitizer()


@pytest.fixture
def minimal_cfo_state():
    """Minimal CFOAgentState-like dict for supervisor routing tests."""
    return {
        "task_id":           "test-001",
        "task_type":         "variance_analysis",
        "task_description":  "Q1 analysis",
        "company_name":      "TestCo",
        "period":            "Q1 2026",
        "report_format":     "board",
        "submitted_by":      "test",
        "submitted_by_role": "analyst",
        "submitted_at":      "2026-05-22T00:00:00",
        "raw_financial_data": {},
        "raw_documents":     None,
        "validated_data":    None,
        "kpi_metrics":       None,
        "rag_chunks":        None,
        "gaap_results":      None,
        "ifrs_results":      None,
        "analysis_narrative": None,
        "final_report":      None,
        "requires_human_approval": False,
        "human_decision":    None,
        "approval_triggers": None,
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
        "variance_table":    None,
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
        "debate_ifrs_advocate": None,
        "debate_gaap_advocate": None,
        "debate_arbiter":    None,
        "debate_complete":   False,
        "human_feedback":    None,
        "approved_by":       None,
        "approved_at":       None,
        "draft_report":      None,
        "report_pdf_path":   None,
        "total_tokens_used": 0,
        "total_cost_usd":    0.0,
        "processing_time_ms": 0,
    }
