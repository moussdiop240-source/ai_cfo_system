"""
Tests for backend.reporting.pdf_generator.

Covers:
- generate_pdf() returns non-empty bytes with valid PDF magic header
- PDF contains multi-page content for long reports
- KPI table renders without crashing on various numeric types
- Compliance summary with all-compliant, all-issues, and empty dicts
- Markdown-heavy report text is rendered without crash
- Missing optional fields degrade gracefully
- Report with no KPIs still generates
- Report with no compliance data still generates
- _strip_markdown() helper removes headings, bold, italic, bullets
- _section_lines() correctly categorizes lines
"""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.reporting.pdf_generator import (
    _section_lines,
    _strip_markdown,
    generate_pdf,
)


# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_REPORT = """# HealthyCo Inc — Q1 2026 Board Report

## 1. EXECUTIVE SUMMARY
Revenue of $12.5M with favorable $1.5M variance to $11.0M budget.
Gross margin 58.2%, EBITDA margin 22.4%.

## 2. REVENUE PERFORMANCE
- ASC 606 five-step model applied
- IFRS 15 performance obligations met
- $0.42 EPS (basic), $0.40 EPS (diluted)

## 3. COST & MARGIN ANALYSIS
COGS of $5.2M. SG&A of $2.1M. R&D of $1.8M.

## 4. US GAAP COMPLIANCE NOTES
ASC 606 — COMPLIANT. ASC 842 — COMPLIANT. ASC 350 — COMPLIANT.

## 5. IFRS COMPLIANCE NOTES
IFRS 15 — COMPLIANT. IFRS 16 — COMPLIANT.

## 6. RISK ASSESSMENT
1. Customer concentration risk (top 3 = 60% of revenue)
2. FX exposure on EMEA revenue

## 7. ACTION PLAN
CFO to present Q2 forecast to board by April 30, 2026.
"""

SAMPLE_KPI = {
    "gross_margin_pct":  58.2,
    "ebitda_margin_pct": 22.4,
    "net_margin_pct":    14.1,
    "current_ratio":     2.3,
    "revenue":           12_500_000.0,
    "free_cash_flow":    2_100_000.0,
}

SAMPLE_GAAP = {
    "asc_606":  {"status": "COMPLIANT",     "standard": "ASC 606"},
    "asc_842":  {"status": "COMPLIANT",     "standard": "ASC 842"},
    "asc_350":  {"status": "REVIEW_NEEDED", "standard": "ASC 350"},
}

SAMPLE_IFRS = {
    "ifrs_15": {"status": "COMPLIANT", "standard": "IFRS 15"},
    "ifrs_16": {"status": "COMPLIANT", "standard": "IFRS 16"},
}


# ── PDF magic-byte constant ───────────────────────────────────────────────────

PDF_HEADER = b"%PDF-"


# ── Helper tests ──────────────────────────────────────────────────────────────

class TestStripMarkdown:
    def test_removes_h1(self):
        assert "# " not in _strip_markdown("# Title")

    def test_removes_bold(self):
        result = _strip_markdown("**bold text**")
        assert "**" not in result
        assert "bold text" in result

    def test_removes_italic(self):
        result = _strip_markdown("*italic*")
        assert "italic" in result
        assert result.count("*") == 0

    def test_converts_dashes_to_bullets(self):
        result = _strip_markdown("- item one")
        assert "item one" in result

    def test_plain_text_unchanged(self):
        result = _strip_markdown("Plain text with no markdown.")
        assert "Plain text with no markdown." in result


class TestSectionLines:
    def test_h1_classified(self):
        parsed = _section_lines("# Big Title")
        assert any(t == "h1" for t, _ in parsed)

    def test_h2_classified(self):
        parsed = _section_lines("## Section")
        assert any(t == "h2" for t, _ in parsed)

    def test_bullet_classified(self):
        parsed = _section_lines("- a bullet")
        assert any(t == "bullet" for t, _ in parsed)

    def test_blank_classified(self):
        parsed = _section_lines("\n")
        assert any(t == "blank" for t, _ in parsed)

    def test_body_classified(self):
        parsed = _section_lines("Normal paragraph text.")
        assert any(t == "body" for t, _ in parsed)

    def test_asterisk_bullet_classified(self):
        parsed = _section_lines("* asterisk bullet")
        assert any(t == "bullet" for t, _ in parsed)


# ── generate_pdf() core ───────────────────────────────────────────────────────

class TestGeneratePDF:
    def test_returns_bytes(self):
        result = generate_pdf(SAMPLE_REPORT, company_name="HealthyCo Inc", period="Q1 2026")
        assert isinstance(result, bytes)

    def test_pdf_magic_header(self):
        result = generate_pdf(SAMPLE_REPORT, company_name="HealthyCo Inc", period="Q1 2026")
        assert result[:5] == PDF_HEADER

    def test_non_empty_output(self):
        result = generate_pdf(SAMPLE_REPORT)
        assert len(result) > 1000

    def test_with_kpi_metrics(self):
        result = generate_pdf(
            SAMPLE_REPORT,
            company_name="HealthyCo Inc",
            period="Q1 2026",
            kpi_metrics=SAMPLE_KPI,
        )
        assert result[:5] == PDF_HEADER
        assert len(result) > 1000

    def test_with_compliance_data(self):
        result = generate_pdf(
            SAMPLE_REPORT,
            company_name="HealthyCo Inc",
            period="Q1 2026",
            gaap_results=SAMPLE_GAAP,
            ifrs_results=SAMPLE_IFRS,
        )
        assert result[:5] == PDF_HEADER

    def test_full_data(self):
        result = generate_pdf(
            SAMPLE_REPORT,
            company_name="HealthyCo Inc",
            period="Q1 2026",
            kpi_metrics=SAMPLE_KPI,
            gaap_results=SAMPLE_GAAP,
            ifrs_results=SAMPLE_IFRS,
        )
        assert result[:5] == PDF_HEADER
        assert len(result) > 2000

    def test_empty_company_name(self):
        result = generate_pdf(SAMPLE_REPORT, company_name="", period="Q1 2026")
        assert result[:5] == PDF_HEADER

    def test_empty_period(self):
        result = generate_pdf(SAMPLE_REPORT, company_name="Acme", period="")
        assert result[:5] == PDF_HEADER

    def test_minimal_report_text(self):
        result = generate_pdf("Hello World")
        assert result[:5] == PDF_HEADER

    def test_empty_report_text(self):
        result = generate_pdf("")
        assert result[:5] == PDF_HEADER

    def test_no_kpi_metrics(self):
        result = generate_pdf(SAMPLE_REPORT, kpi_metrics=None)
        assert result[:5] == PDF_HEADER

    def test_empty_kpi_metrics(self):
        result = generate_pdf(SAMPLE_REPORT, kpi_metrics={})
        assert result[:5] == PDF_HEADER

    def test_no_compliance_data(self):
        result = generate_pdf(SAMPLE_REPORT, gaap_results=None, ifrs_results=None)
        assert result[:5] == PDF_HEADER

    def test_empty_compliance_data(self):
        result = generate_pdf(SAMPLE_REPORT, gaap_results={}, ifrs_results={})
        assert result[:5] == PDF_HEADER

    def test_all_gaap_compliant(self):
        gaap = {f"asc_{i}": {"status": "COMPLIANT"} for i in range(12)}
        result = generate_pdf(SAMPLE_REPORT, gaap_results=gaap)
        assert result[:5] == PDF_HEADER

    def test_some_gaap_issues(self):
        gaap = {
            "asc_606": {"status": "COMPLIANT"},
            "asc_842": {"status": "REVIEW_NEEDED"},
        }
        result = generate_pdf(SAMPLE_REPORT, gaap_results=gaap)
        assert result[:5] == PDF_HEADER

    def test_integer_kpi_values(self):
        kpi = {"gross_margin_pct": 58, "revenue": 12_000_000}
        result = generate_pdf(SAMPLE_REPORT, kpi_metrics=kpi)
        assert result[:5] == PDF_HEADER

    def test_long_report_generates_multiple_pages(self):
        long_report = (SAMPLE_REPORT + "\n\nAdditional analysis.\n") * 10
        result = generate_pdf(long_report, company_name="BigCo", period="FY 2026")
        assert result[:5] == PDF_HEADER
        assert len(result) > 5000

    def test_special_characters_in_report(self):
        text = "Revenue grew 12.5% — above expectations. EBITDA: $2.5M (margin ~22%)."
        result = generate_pdf(text)
        assert result[:5] == PDF_HEADER

    def test_report_larger_than_one_kb(self):
        result = generate_pdf(
            SAMPLE_REPORT,
            company_name="HealthyCo Inc",
            period="Q1 2026",
            kpi_metrics=SAMPLE_KPI,
            gaap_results=SAMPLE_GAAP,
            ifrs_results=SAMPLE_IFRS,
        )
        assert len(result) > 1024

    def test_output_is_readable_as_bytesio(self):
        result = generate_pdf(SAMPLE_REPORT)
        buf = io.BytesIO(result)
        buf.seek(0)
        assert buf.read(5) == PDF_HEADER
