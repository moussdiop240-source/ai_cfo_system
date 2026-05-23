"""
PDF Report Generator for the AI CFO System.

Converts a final_report markdown string (and optional KPI/compliance data)
into a board-ready PDF using fpdf2 (v2.7+).

Usage:
    from backend.reporting.pdf_generator import generate_pdf
    pdf_bytes = generate_pdf(
        report_text="# Acme Corp — Q1 2026 Board Report\n...",
        company_name="Acme Corp",
        period="Q1 2026",
        kpi_metrics={"gross_margin_pct": 58.2, "ebitda_margin_pct": 22.4},
        gaap_results={"asc_606": {"status": "COMPLIANT"}},
        ifrs_results={"ifrs_15": {"status": "COMPLIANT"}},
    )
    with open("report.pdf", "wb") as f:
        f.write(pdf_bytes)
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


_UNICODE_MAP = str.maketrans({
    "—": "--",   # em dash
    "–": "-",    # en dash
    "‘": "'",    # left single quote
    "’": "'",    # right single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "•": "*",    # bullet
    "…": "...",  # ellipsis
    " ": " ",    # non-breaking space
})


def _safe(text: str) -> str:
    """Replace common non-latin-1 characters and encode safely for fpdf core fonts."""
    return text.translate(_UNICODE_MAP).encode("latin-1", errors="replace").decode("latin-1")


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting for plain-text PDF body."""
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)     # headings
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                   # bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)                       # italic
    text = re.sub(r"`(.+?)`", r"\1", text)                         # inline code
    text = re.sub(r"^\s*[-*•]\s+", "  * ", text, flags=re.MULTILINE)  # bullets
    return text


def _section_lines(text: str) -> List[tuple]:
    """
    Split markdown text into (type, content) pairs.
    type: "h1" | "h2" | "h3" | "bullet" | "body" | "blank"
    """
    lines = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            lines.append(("blank", ""))
        elif stripped.startswith("# "):
            lines.append(("h1", stripped[2:]))
        elif stripped.startswith("## "):
            lines.append(("h2", stripped[3:]))
        elif stripped.startswith("### "):
            lines.append(("h3", stripped[4:]))
        elif re.match(r"^[-*•]\s", stripped):
            lines.append(("bullet", re.sub(r"^[-*•]\s+", "", stripped)))
        else:
            lines.append(("body", stripped))
    return lines


def generate_pdf(
    report_text: str,
    company_name: str = "",
    period: str = "",
    kpi_metrics: Optional[Dict[str, Any]] = None,
    gaap_results: Optional[Dict[str, Any]] = None,
    ifrs_results: Optional[Dict[str, Any]] = None,
    logo_path: Optional[str] = None,
) -> bytes:
    """
    Generate a board-ready PDF from the final_report markdown text.

    Returns:
        Raw PDF bytes (write directly to file or HTTP response).

    Raises:
        ImportError: if fpdf2 is not installed.
    """
    try:
        from fpdf import FPDF, XPos, YPos
    except ImportError as e:
        raise ImportError("fpdf2 is required: pip install fpdf2") from e

    class _BoardPDF(FPDF):
        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(140, 140, 140)
            self.cell(
                0, 5,
                f"Page {self.page_no()} | AI CFO System | Confidential",
                align="C",
            )

    pdf = _BoardPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(left=20, top=20, right=20)
    pdf.add_page()

    NL = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}

    # ── Cover header ─────────────────────────────────────────────────────────
    cover_title = _safe(company_name or "AI CFO System")
    subtitle    = _safe(f"Board Report -- {period}" if period else "Board Report")
    ts          = datetime.utcnow().strftime("Generated %Y-%m-%d %H:%M UTC")

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(15, 36, 82)
    pdf.cell(0, 10, cover_title, align="C", **NL)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, subtitle, align="C", **NL)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, ts, align="C", **NL)
    pdf.ln(4)

    # ── Horizontal rule ───────────────────────────────────────────────────────
    pdf.set_draw_color(15, 36, 82)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6)

    # ── KPI summary table ─────────────────────────────────────────────────────
    kpi = kpi_metrics or {}
    kpi_display = [
        ("Gross Margin",   kpi.get("gross_margin_pct"),    "%"),
        ("EBITDA Margin",  kpi.get("ebitda_margin_pct"),   "%"),
        ("Net Margin",     kpi.get("net_margin_pct"),      "%"),
        ("Current Ratio",  kpi.get("current_ratio"),       "x"),
        ("Revenue",        kpi.get("revenue"),             "$"),
        ("Free Cash Flow", kpi.get("free_cash_flow"),      "$"),
    ]
    kpi_rows = [(k, v, u) for k, v, u in kpi_display if v is not None]
    col_w = [90, 50, 30]

    if kpi_rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 36, 82)
        pdf.cell(0, 6, "KEY PERFORMANCE INDICATORS", **NL)
        pdf.set_line_width(0.2)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(230, 235, 245)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(col_w[0], 6, "Metric", border=0, fill=True)
        pdf.cell(col_w[1], 6, "Value",  border=0, fill=True)
        pdf.cell(col_w[2], 6, "Unit",   border=0, fill=True, **NL)

        pdf.set_font("Helvetica", "", 9)
        for i, (label, val, unit) in enumerate(kpi_rows):
            pdf.set_fill_color(248, 249, 252) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            if unit == "$":
                formatted = f"${val:,.0f}" if isinstance(val, float) else f"${val}"
            elif unit == "%":
                formatted = f"{val:.1f}%" if isinstance(val, float) else str(val)
            else:
                formatted = f"{val:.2f}{unit}" if isinstance(val, float) else str(val)
            pdf.cell(col_w[0], 5.5, label,     border=0, fill=True)
            pdf.cell(col_w[1], 5.5, formatted, border=0, fill=True)
            pdf.cell(col_w[2], 5.5, unit,      border=0, fill=True, **NL)
        pdf.ln(5)

    # ── Compliance status summary ──────────────────────────────────────────────
    gaap = gaap_results or {}
    ifrs = ifrs_results or {}
    if gaap or ifrs:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 36, 82)
        pdf.cell(0, 6, "COMPLIANCE SUMMARY", **NL)
        pdf.set_line_width(0.2)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(2)

        def _count_status(results: dict, target: str) -> int:
            return sum(1 for v in results.values() if isinstance(v, dict) and v.get("status") == target)

        def _color_for(ok: int, tot: int):
            if tot == 0:
                return (120, 120, 120)
            return (0, 140, 0) if ok == tot else (200, 80, 0)

        gaap_ok  = _count_status(gaap, "COMPLIANT")
        ifrs_ok  = _count_status(ifrs, "COMPLIANT")

        pdf.set_font("Helvetica", "", 9)
        for label, ok, tot in [("US GAAP", gaap_ok, len(gaap)), ("IFRS", ifrs_ok, len(ifrs))]:
            status_str = f"{ok}/{tot} COMPLIANT" if tot else "N/A"
            r, g, b = _color_for(ok, tot)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(60, 5.5, label + ":", border=0)
            pdf.set_text_color(r, g, b)
            pdf.cell(0, 5.5, status_str, border=0, **NL)
        pdf.ln(5)

    # ── Report body ───────────────────────────────────────────────────────────
    pdf.set_line_width(0.5)
    pdf.set_draw_color(15, 36, 82)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(4)

    safe_text = _safe(report_text)
    for typ, content in _section_lines(safe_text):
        if typ == "blank":
            pdf.ln(2)
        elif typ == "h1":
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(15, 36, 82)
            pdf.ln(3)
            pdf.cell(0, 8, content, **NL)
            pdf.set_draw_color(15, 36, 82)
            pdf.set_line_width(0.3)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(2)
        elif typ == "h2":
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(15, 36, 82)
            pdf.ln(2)
            pdf.cell(0, 7, content, **NL)
        elif typ == "h3":
            pdf.set_font("Helvetica", "BI", 10)
            pdf.set_text_color(40, 60, 100)
            pdf.cell(0, 6, content, **NL)
        elif typ == "bullet":
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            pdf.set_x(pdf.l_margin + 6)
            pdf.multi_cell(0, 5.5, f"* {content}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 5.5, content, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
