"""
Role-Based Access Control for the AI CFO System.

Role hierarchy (ascending privilege):
    analyst < manager < vp < cfo < admin

Usage (FastAPI):
    from backend.security.rbac import require_role

    @router.get("/sensitive")
    def endpoint(user=Depends(require_role("cfo"))):
        ...

The API key is passed in the X-API-Key header. Keys are stored in
RBAC_KEYS environment variable as JSON: {"key1": "cfo", "key2": "analyst"}
or defaulted to a single ADMIN_API_KEY that grants "admin" role.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from fastapi import Depends, Header, HTTPException, status

# Role hierarchy — higher index = higher privilege
_ROLE_LEVELS: Dict[str, int] = {
    "analyst": 1,
    "manager": 2,
    "vp":      3,
    "cfo":     4,
    "admin":   5,
}


def _load_key_map() -> Dict[str, str]:
    """
    Load API-key → role mapping.

    Priority:
    1. RBAC_KEYS env var (JSON dict: {"key": "role"})
    2. ADMIN_API_KEY env var → grants "admin"
    3. Fallback: single hardcoded dev key for local testing only
    """
    raw = os.getenv("RBAC_KEYS", "")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    admin_key = os.getenv("ADMIN_API_KEY", "")
    if admin_key:
        return {admin_key: "admin"}

    # Dev-only fallback — no real key exposed in source
    return {}


def _get_role(api_key: str) -> Optional[str]:
    """Return the role for the given key, or None if not found."""
    return _load_key_map().get(api_key)


# ── FastAPI dependencies ──────────────────────────────────────────────────────

class RBACUser:
    """Carries the authenticated user's key and role through the request."""
    def __init__(self, api_key: str, role: str):
        self.api_key = api_key
        self.role    = role
        self.level   = _ROLE_LEVELS.get(role, 0)

    def has_role(self, min_role: str) -> bool:
        required = _ROLE_LEVELS.get(min_role, 999)
        return self.level >= required

    def __repr__(self) -> str:
        return f"RBACUser(role={self.role!r})"


def _authenticate(x_api_key: str = Header(default="")) -> RBACUser:
    """FastAPI dependency — validates X-API-Key header and returns RBACUser."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    role = _get_role(x_api_key)
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return RBACUser(api_key=x_api_key, role=role)


def require_role(min_role: str):
    """
    FastAPI dependency factory.  Raises 403 if the caller's role is below min_role.

    Example:
        @router.post("/approve")
        def approve(user: RBACUser = Depends(require_role("cfo"))):
            ...
    """
    if min_role not in _ROLE_LEVELS:
        raise ValueError(f"Unknown role: {min_role!r}. Valid: {list(_ROLE_LEVELS)}")

    def _dep(user: RBACUser = Depends(_authenticate)) -> RBACUser:
        if not user.has_role(min_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' insufficient — requires '{min_role}' or higher",
            )
        return user

    return _dep


# ── Utility helpers (non-HTTP contexts) ──────────────────────────────────────

def check_role(api_key: str, min_role: str) -> bool:
    """Return True if the key has at least min_role privilege (no HTTP exceptions)."""
    role = _get_role(api_key)
    if role is None:
        return False
    return _ROLE_LEVELS.get(role, 0) >= _ROLE_LEVELS.get(min_role, 999)


def role_for_key(api_key: str) -> Optional[str]:
    """Return the role string for a key, or None if not found."""
    return _get_role(api_key)
