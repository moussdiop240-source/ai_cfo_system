"""
LLM Output Validator — post-process and audit analysis agent outputs.

Design principles:
- Never block the pipeline: validation produces warnings, not hard failures.
- Four checks are critical (fail_critical=True): recalculation, confidence bounds,
  hallucinated figures, and prompt injection echoes.
- Remaining checks produce warnings that are logged to the state audit_log.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Pattern that matches raw arithmetic expressions (potential recalculation).
# Requires at least one space around the operator to avoid matching ISO dates
# (e.g. 2026-04-30) and percentage changes (e.g. -15%).
_ARITHMETIC_RE = re.compile(
    r"\b\d[\d,]*\.?\d*\s+[+\-*/÷×]\s+\d[\d,]*\.?\d*\b"
)

# Pattern for prompt injection echo markers
_INJECTION_ECHO_RE = re.compile(
    r"(ignore\s+previous\s+instructions?|you\s+are\s+now\s+|"
    r"disregard\s+all\s+prior|forget\s+everything|from\s+now\s+on|"
    r"system\s*:\s*you\s+are|override\s+(your\s+|all\s+)?(safety|rules?|guidelines?)|"
    r"act\s+as\s+(a|an)\s+\w+\s+(with\s+no|without))",
    re.IGNORECASE,
)

# Valid ASC standard citations
_ASC_CITATION_RE = re.compile(r"\bASC\s*\d{3}[-–]\d+\b|\bSAB\s*99\b", re.IGNORECASE)

# Valid IFRS/IAS citation
_IAS_CITATION_RE = re.compile(r"\b(IAS|IFRS)\s*\d+\b", re.IGNORECASE)

# Financial figure pattern ($X or X%)
_FINANCIAL_FIGURE_RE = re.compile(r"(\$[\d,]+\.?\d*[MKB]?|\d+\.?\d*\s*%)")

# Allowed deviation between stated figure and math engine output (5%)
_FIGURE_TOLERANCE = 0.05


@dataclass
class ValidationResult:
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    score: float = 1.0          # 0.0–1.0 quality score


class AnalysisOutputValidator:
    """
    Validates the output of the analysis_agent_node against known-good math outputs.

    Usage:
        validator = AnalysisOutputValidator(math_results, rag_chunks)
        result = validator.validate(llm_output_dict)
        if not result.passed:
            # handle errors
    """

    def __init__(
        self,
        math_results: Optional[Dict[str, Any]] = None,
        rag_chunks: Optional[List[Dict]] = None,
        strict: bool = False,
    ):
        self._math = math_results or {}
        self._rag  = rag_chunks or []
        self._strict = strict   # strict=True makes warnings into errors

    def validate(self, output: Dict[str, Any]) -> ValidationResult:
        """Run all validation checks. Returns consolidated ValidationResult."""
        result = ValidationResult()
        checks = [
            self._check_confidence_bounds,
            self._check_no_recalculation,
            self._check_prompt_injection_echo,
            self._check_summary_has_figures,
            self._check_action_items_have_owners,
            self._check_rag_sources_cited,
            self._check_gaap_citations,
            self._check_no_hallucinated_figures,
        ]
        penalty = 0.0
        for check in checks:
            w, e = check(output)
            result.warnings.extend(w)
            result.errors.extend(e)
            if e:
                result.passed = False
                penalty += 0.25
            if w and self._strict:
                result.passed = False
                penalty += 0.1

        result.score = max(0.0, round(1.0 - penalty, 2))
        return result

    # ── individual checks ────────────────────────────────────────────────────

    def _check_confidence_bounds(self, output: Dict) -> tuple[list, list]:
        score = output.get("confidence_score")
        if score is None:
            return ["confidence_score missing"], []
        try:
            s = float(score)
        except (TypeError, ValueError):
            return [], [f"confidence_score is not numeric: {score!r}"]
        if not (0.0 <= s <= 1.0):
            return [], [f"confidence_score {s} out of bounds [0.0, 1.0]"]
        return [], []

    def _check_no_recalculation(self, output: Dict) -> tuple[list, list]:
        """Fails if the narrative contains raw arithmetic (e.g. '2500000 / 5000000')."""
        narrative = _extract_text(output)
        matches = _ARITHMETIC_RE.findall(narrative)
        if matches:
            return [], [
                f"Analysis narrative contains {len(matches)} arithmetic expression(s) — "
                f"LLM must not recalculate: {matches[:3]}"
            ]
        return [], []

    def _check_prompt_injection_echo(self, output: Dict) -> tuple[list, list]:
        """Fails if the output echoes prompt injection patterns."""
        text = _extract_text(output)
        if _INJECTION_ECHO_RE.search(text):
            return [], ["Potential prompt injection echo detected in LLM output"]
        return [], []

    def _check_summary_has_figures(self, output: Dict) -> tuple[list, list]:
        summary = output.get("executive_summary", "")
        if not summary:
            return ["executive_summary is empty"], []
        if len(summary) < 50:
            return [f"executive_summary too short ({len(summary)} chars, min 50)"], []
        if not _FINANCIAL_FIGURE_RE.search(summary):
            return ["executive_summary contains no financial figures ($ or %)"], []
        return [], []

    def _check_action_items_have_owners(self, output: Dict) -> tuple[list, list]:
        items = output.get("action_items") or []
        if not items:
            return ["action_items list is empty"], []
        missing_owner = []
        # An owner must be a capitalized name or a role title
        owner_pattern = re.compile(
            r"\b(CFO|CEO|CTO|COO|VP|Controller|Director|Manager|Finance|Analyst|"
            r"Accounting|Auditor|[A-Z][a-z]+\s+[A-Z][a-z]+)\b"
        )
        for item in items:
            if not owner_pattern.search(item):
                missing_owner.append(item[:80])
        if missing_owner:
            return [
                f"{len(missing_owner)} action item(s) may be missing an owner: "
                + "; ".join(missing_owner[:2])
            ], []
        return [], []

    def _check_rag_sources_cited(self, output: Dict) -> tuple[list, list]:
        """If RAG chunks were provided, the output must cite at least one source."""
        if not self._rag:
            return [], []
        cited = output.get("rag_sources_cited") or []
        narrative = _extract_text(output)
        # Accept either explicit rag_sources_cited field or source title appearing in text
        source_titles = {c.get("title", "") for c in self._rag if isinstance(c, dict)}
        in_text = any(title and title in narrative for title in source_titles)
        if not cited and not in_text:
            return [
                f"RAG context was provided ({len(self._rag)} chunks) but no sources cited"
            ], []
        return [], []

    def _check_gaap_citations(self, output: Dict) -> tuple[list, list]:
        """If GAAP findings exist in math results, the narrative should cite ASC standards."""
        gaap = self._math.get("gaap_results") or {}
        non_compliant = [k for k, v in gaap.items() if v.get("status") != "COMPLIANT"]
        if not non_compliant:
            return [], []
        narrative = _extract_text(output)
        if not _ASC_CITATION_RE.search(narrative) and not _IAS_CITATION_RE.search(narrative):
            return [
                f"GAAP/IFRS findings present ({len(non_compliant)} issues) "
                "but no ASC/IAS citations found in narrative"
            ], []
        return [], []

    def _check_no_hallucinated_figures(self, output: Dict) -> tuple[list, list]:
        """
        Extract dollar figures from the narrative and check that they match
        known math engine outputs within _FIGURE_TOLERANCE.
        """
        if not self._math:
            return [], []

        known_values = _extract_known_values(self._math)
        if not known_values:
            return [], []

        narrative = _extract_text(output)
        dollar_matches = re.findall(r"\$([\d,]+(?:\.\d+)?)\s*([MKB]?)", narrative)

        hallucinated = []
        for raw_num, suffix in dollar_matches:
            value = _parse_dollar(raw_num, suffix)
            if value < 1_000:
                continue  # too small to reliably cross-check
            # Check if any known value is within tolerance
            matched = any(
                abs(value - kv) / max(abs(kv), 1) <= _FIGURE_TOLERANCE
                for kv in known_values
            )
            if not matched:
                hallucinated.append(f"${raw_num}{suffix}")

        if hallucinated:
            return [
                f"{len(hallucinated)} financial figure(s) in narrative not traceable to "
                f"math engine outputs: {hallucinated[:3]}"
            ], []
        return [], []


class LLMOutputSanitizer:
    """
    Sanitizes raw LLM output before it is stored or displayed.
    Removes potential injection echoes, normalises whitespace, truncates.
    """

    MAX_SUMMARY_LEN   = 2000
    MAX_LIST_ITEM_LEN = 500
    MAX_LIST_ITEMS    = 10

    def sanitize(self, output: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(output)
        clean["executive_summary"]    = self._clean_text(clean.get("executive_summary", ""), self.MAX_SUMMARY_LEN)
        clean["key_variance_drivers"] = self._clean_list(clean.get("key_variance_drivers"))
        clean["identified_risks"]     = self._clean_list(clean.get("identified_risks"))
        clean["opportunities"]        = self._clean_list(clean.get("opportunities"))
        clean["action_items"]         = self._clean_list(clean.get("action_items"))
        return clean

    def _clean_text(self, text: str, max_len: int) -> str:
        if not isinstance(text, str):
            text = str(text)
        # Remove injection echo patterns
        text = _INJECTION_ECHO_RE.sub("[REDACTED]", text)
        # Normalize whitespace
        text = re.sub(r"\s{3,}", " ", text).strip()
        return text[:max_len]

    def _clean_list(self, items) -> List[str]:
        if not isinstance(items, list):
            return []
        cleaned = []
        for item in items[:self.MAX_LIST_ITEMS]:
            if isinstance(item, str):
                item = _INJECTION_ECHO_RE.sub("[REDACTED]", item)
                item = item.strip()[:self.MAX_LIST_ITEM_LEN]
                if item:
                    cleaned.append(item)
        return cleaned


# ── private helpers ──────────────────────────────────────────────────────────

def _extract_text(output: Dict) -> str:
    """Concatenate all string fields in the output dict for pattern matching."""
    parts = []
    for v in output.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(str(i) for i in v if i)
    return " ".join(parts)


def _extract_known_values(math: Dict) -> List[float]:
    """Pull numeric values from math_results to cross-check against narrative."""
    values = []
    for fname in ("revenue", "net_income", "ebitda", "gross_profit", "total_assets",
                  "total_debt", "cash", "net_debt"):
        v = math.get(fname)
        if isinstance(v, (int, float)) and v > 0:
            values.append(float(v))
    kpis = math.get("kpi_metrics") or {}
    for k in ("net_debt",):
        if k in kpis:
            values.append(float(kpis[k]))
    return values


def _parse_dollar(raw: str, suffix: str) -> float:
    """Parse '$1,234M' style strings to float."""
    num = float(raw.replace(",", ""))
    multiplier = {"M": 1_000_000, "K": 1_000, "B": 1_000_000_000}.get(suffix.upper(), 1)
    return num * multiplier
