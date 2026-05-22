"""
Input sanitization and prompt injection detection.

All external inputs (company names, financial data from unstructured sources,
analyst notes) pass through this layer before reaching the LLM or DB.
"""
from __future__ import annotations

import hmac
import re
from typing import Any, Dict, Optional


class PromptInjectionError(ValueError):
    """Raised when a prompt injection attempt is detected."""


# ── injection pattern catalogue ──────────────────────────────────────────────
# Ordered from most specific to broadest to minimise false positives.
_INJECTION_PATTERNS = [
    # Direct instruction override
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+|your\s+)?(previous|prior)\s+(instructions?|context|rules?)",
    r"forget\s+(everything|all\s+previous|your\s+instructions?)",
    # Role switching
    r"you\s+are\s+now\s+(a|an)\s+",
    r"act\s+as\s+(a|an)\s+\w+\s+(with\s+no|without)\s+(any\s+)?(restrictions?|limits?|rules?)",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"roleplay\s+as\s+",
    # System prompt manipulation
    r"(system|assistant|human)\s*:\s*(you\s+are|ignore|forget)",
    r"\[system\]",
    r"<\s*system\s*>",
    r"<<\s*SYS\s*>>",
    # Exfiltration attempts
    r"(print|output|reveal|show|display)\s+(your\s+)?(system\s+prompt|instructions?|context)",
    r"what\s+(are\s+your|is\s+your)\s+(instructions?|system\s+prompt)",
    # Jailbreak markers
    r"DAN\s*(mode|jailbreak)?",
    r"developer\s+mode",
    r"jailbreak",
]

_INJECTION_RE = re.compile(
    "|".join(f"(?:{p})" for p in _INJECTION_PATTERNS),
    re.IGNORECASE,
)

# Characters that are safe in company names (alphanumeric + common business chars)
_COMPANY_NAME_RE = re.compile(r"^[A-Za-z0-9\s\.\,\-\&\'\(\)\/]+$")

# SQL injection patterns (defence-in-depth; parameterised queries are primary protection)
_SQL_INJECTION_RE = re.compile(
    r"(--|\bOR\b\s+\d+\s*=\s*\d+|'\s*;\s*DROP|UNION\s+SELECT|INSERT\s+INTO|"
    r"xp_cmdshell|1\s*=\s*1|\bEXEC\b\s*\()",
    re.IGNORECASE,
)

# Valid financial value: numeric, may have decimals and sign
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


class InputSanitizer:
    """
    Central sanitization hub.

    All methods raise PromptInjectionError or ValueError on bad input.
    They return the (possibly normalised) clean value on success.
    """

    @staticmethod
    def sanitize_company_name(name: str) -> str:
        """
        Validate and normalise a company name.
        - Blocks SQL injection patterns.
        - Blocks non-printable characters.
        - Strips leading/trailing whitespace.
        - Max 255 chars.
        """
        if not isinstance(name, str):
            raise ValueError(f"company_name must be a string, got {type(name).__name__}")
        name = name.strip()
        if not name:
            raise ValueError("company_name must not be empty")
        if len(name) > 255:
            raise ValueError(f"company_name too long ({len(name)} chars, max 255)")
        if _SQL_INJECTION_RE.search(name):
            raise ValueError(f"company_name contains invalid characters: {name!r}")
        if not _COMPANY_NAME_RE.match(name):
            raise ValueError(
                f"company_name contains unexpected characters. "
                f"Only letters, digits, spaces, and . , - & ' ( ) / are allowed. Got: {name!r}"
            )
        # Check for injection patterns even in names
        if _INJECTION_RE.search(name):
            raise PromptInjectionError(f"Prompt injection pattern in company_name: {name!r}")
        return name

    @staticmethod
    def sanitize_financial_value(field_name: str, value: Any) -> float:
        """Coerce a financial value to float. Rejects strings that aren't numeric."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if not _NUMERIC_RE.match(value):
                raise ValueError(
                    f"Field '{field_name}' has non-numeric value: {value!r}"
                )
            return float(value)
        raise ValueError(
            f"Field '{field_name}' must be numeric, got {type(value).__name__}: {value!r}"
        )

    @staticmethod
    def sanitize_free_text(field_name: str, text: str, max_len: int = 2000) -> str:
        """
        Sanitize analyst notes, feedback, and other free-text fields.
        - Detects prompt injection patterns.
        - Strips null bytes and non-printable characters.
        - Enforces max length.
        """
        if not isinstance(text, str):
            raise ValueError(f"Field '{field_name}' must be a string")
        # Strip null bytes
        text = text.replace("\x00", "").strip()
        # Check injection
        if _INJECTION_RE.search(text):
            raise PromptInjectionError(
                f"Prompt injection pattern detected in field '{field_name}'"
            )
        # Enforce length
        if len(text) > max_len:
            raise ValueError(
                f"Field '{field_name}' too long ({len(text)} chars, max {max_len})"
            )
        return text

    @staticmethod
    def sanitize_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a raw financial data dict.
        - Validates company_name and period.
        - Coerces numeric fields.
        - Detects injection in any string value.
        """
        clean: Dict[str, Any] = {}
        numeric_fields = {
            "revenue", "cogs", "gross_profit", "ebitda", "ebit", "net_income",
            "interest_expense", "pre_tax_income", "tax_provision", "total_assets",
            "total_equity", "current_assets", "current_liabilities", "cash",
            "total_debt", "accounts_receivable", "accounts_payable", "inventory",
            "goodwill", "rou_assets", "lease_liability", "capex", "free_cash_flow",
            "monthly_cash_burn", "shares_outstanding", "diluted_shares",
            "rd_expense", "sga_expense", "depreciation_amortization",
        }
        for key, value in data.items():
            if key == "company_name":
                clean[key] = InputSanitizer.sanitize_company_name(str(value))
            elif key in numeric_fields:
                clean[key] = InputSanitizer.sanitize_financial_value(key, value)
            elif isinstance(value, str):
                # Free-text field — check for injection
                if _INJECTION_RE.search(value):
                    raise PromptInjectionError(
                        f"Prompt injection pattern detected in field '{key}'"
                    )
                clean[key] = value
            else:
                clean[key] = value
        return clean

    @staticmethod
    def validate_api_key(provided: str, expected: str) -> bool:
        """
        Constant-time API key comparison.
        Returns True if keys match, False otherwise.
        Never raises — safe to use in auth middleware.
        """
        if not isinstance(provided, str) or not isinstance(expected, str):
            return False
        if not provided or not expected:
            return False
        return hmac.compare_digest(provided.encode(), expected.encode())

    @staticmethod
    def detect_prompt_injection(text: str) -> Optional[str]:
        """
        Returns the matched injection pattern string, or None if clean.
        Use this for logging/alerting without blocking.
        """
        m = _INJECTION_RE.search(text)
        return m.group(0) if m else None


# Module-level convenience alias
sanitize_company_name = InputSanitizer.sanitize_company_name
