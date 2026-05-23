"""
Startup secrets validation.

Checks environment for weak/missing credentials and emits structured log warnings.
Never raises — validation is advisory; the app still starts.

Checks:
  1. At least one of RBAC_KEYS or ADMIN_API_KEY is set (WARNING if neither)
  2. ADMIN_API_KEY length >= 32 chars (WARNING if shorter)
  3. ADMIN_API_KEY not a known weak value (WARNING if so)
  4. LLM_BACKEND=anthropic without ANTHROPIC_API_KEY (ERROR — will fail at runtime)
"""
import logging
import os

logger = logging.getLogger("ai_cfo.secrets")

_WEAK_KEYS = {"changeme", "secret", "admin", "password", "12345678", "test", "dev"}


def validate() -> dict:
    """Run all checks; return summary dict with counts."""
    warnings = 0
    errors = 0

    rbac_keys = os.environ.get("RBAC_KEYS", "")
    admin_key = os.environ.get("ADMIN_API_KEY", "")
    llm_backend = os.environ.get("LLM_BACKEND", "ollama").lower()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not rbac_keys and not admin_key:
        logger.warning(
            "SECRETS_WARN: Neither RBAC_KEYS nor ADMIN_API_KEY is set — "
            "all authenticated endpoints will reject every request."
        )
        warnings += 1

    if admin_key and len(admin_key) < 32:
        logger.warning(
            f"SECRETS_WARN: ADMIN_API_KEY is only {len(admin_key)} chars — "
            "use at least 32 random characters for production."
        )
        warnings += 1

    if admin_key and admin_key.lower() in _WEAK_KEYS:
        logger.warning(
            "SECRETS_WARN: ADMIN_API_KEY matches a known weak value — "
            "rotate it before deploying."
        )
        warnings += 1

    if llm_backend == "anthropic" and not anthropic_key:
        logger.error(
            "SECRETS_ERROR: LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set — "
            "LLM calls will fail at runtime."
        )
        errors += 1

    return {"warnings": warnings, "errors": errors}
