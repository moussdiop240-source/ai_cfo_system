"""
Tests for backend.security.rbac.

Covers:
- Role hierarchy enforcement (analyst < manager < vp < cfo < admin)
- Key lookup from RBAC_KEYS env var
- Key lookup from ADMIN_API_KEY env var
- check_role() utility
- role_for_key() utility
- FastAPI dependency — require_role() raises 401/403 correctly
- RBACUser.has_role()
"""
import json
import os
import sys
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.security.rbac import (
    RBACUser,
    check_role,
    require_role,
    role_for_key,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

KEYS = {
    "key-analyst":  "analyst",
    "key-manager":  "manager",
    "key-vp":       "vp",
    "key-cfo":      "cfo",
    "key-admin":    "admin",
}


def _make_app(min_role: str) -> FastAPI:
    """Small test app with a single protected endpoint."""
    app = FastAPI()

    @app.get("/protected")
    def protected(user=Depends(require_role(min_role))):
        return {"role": user.role}

    return app


# ── RBACUser unit tests ───────────────────────────────────────────────────────

class TestRBACUser:
    def test_analyst_has_analyst(self):
        u = RBACUser("k", "analyst")
        assert u.has_role("analyst") is True

    def test_analyst_lacks_manager(self):
        u = RBACUser("k", "analyst")
        assert u.has_role("manager") is False

    def test_cfo_has_vp(self):
        u = RBACUser("k", "cfo")
        assert u.has_role("vp") is True

    def test_admin_has_all(self):
        u = RBACUser("k", "admin")
        for role in ("analyst", "manager", "vp", "cfo", "admin"):
            assert u.has_role(role) is True

    def test_analyst_lacks_admin(self):
        u = RBACUser("k", "analyst")
        assert u.has_role("admin") is False

    def test_manager_lacks_cfo(self):
        u = RBACUser("k", "manager")
        assert u.has_role("cfo") is False

    def test_repr_contains_role(self):
        assert "cfo" in repr(RBACUser("k", "cfo"))


# ── check_role() utility ──────────────────────────────────────────────────────

class TestCheckRole:
    def _patch(self, keys=KEYS):
        return patch("backend.security.rbac._load_key_map", return_value=keys)

    def test_analyst_key_passes_analyst_check(self):
        with self._patch():
            assert check_role("key-analyst", "analyst") is True

    def test_analyst_key_fails_cfo_check(self):
        with self._patch():
            assert check_role("key-analyst", "cfo") is False

    def test_cfo_key_passes_vp_check(self):
        with self._patch():
            assert check_role("key-cfo", "vp") is True

    def test_admin_key_passes_admin_check(self):
        with self._patch():
            assert check_role("key-admin", "admin") is True

    def test_unknown_key_fails(self):
        with self._patch():
            assert check_role("unknown-key", "analyst") is False

    def test_empty_key_fails(self):
        with self._patch():
            assert check_role("", "analyst") is False


# ── role_for_key() utility ────────────────────────────────────────────────────

class TestRoleForKey:
    def _patch(self, keys=KEYS):
        return patch("backend.security.rbac._load_key_map", return_value=keys)

    def test_returns_correct_role(self):
        with self._patch():
            assert role_for_key("key-cfo") == "cfo"

    def test_unknown_key_returns_none(self):
        with self._patch():
            assert role_for_key("bogus") is None


# ── RBAC_KEYS env var loading ─────────────────────────────────────────────────

class TestKeyLoading:
    def test_rbac_keys_env_var_parsed(self):
        env_keys = {"my-secret": "vp"}
        with patch.dict(os.environ, {"RBAC_KEYS": json.dumps(env_keys)}, clear=False):
            assert role_for_key("my-secret") == "vp"

    def test_invalid_json_falls_through(self):
        with patch.dict(os.environ, {"RBAC_KEYS": "not-json", "ADMIN_API_KEY": "admin-key"}, clear=False):
            assert role_for_key("admin-key") == "admin"

    def test_admin_api_key_env_var(self):
        with patch.dict(os.environ, {"RBAC_KEYS": "", "ADMIN_API_KEY": "super-secret"}, clear=False):
            assert role_for_key("super-secret") == "admin"


# ── FastAPI dependency integration ────────────────────────────────────────────

class TestRequireRoleDependency:
    def _client(self, min_role: str):
        app = _make_app(min_role)
        return TestClient(app, raise_server_exceptions=True)

    def test_missing_header_returns_401(self):
        with patch("backend.security.rbac._load_key_map", return_value=KEYS):
            client = self._client("analyst")
            r = client.get("/protected")
        assert r.status_code == 401

    def test_invalid_key_returns_403(self):
        with patch("backend.security.rbac._load_key_map", return_value=KEYS):
            client = self._client("analyst")
            r = client.get("/protected", headers={"X-API-Key": "wrong"})
        assert r.status_code == 403

    def test_sufficient_role_returns_200(self):
        with patch("backend.security.rbac._load_key_map", return_value=KEYS):
            client = self._client("analyst")
            r = client.get("/protected", headers={"X-API-Key": "key-analyst"})
        assert r.status_code == 200
        assert r.json()["role"] == "analyst"

    def test_insufficient_role_returns_403(self):
        with patch("backend.security.rbac._load_key_map", return_value=KEYS):
            client = self._client("cfo")
            r = client.get("/protected", headers={"X-API-Key": "key-analyst"})
        assert r.status_code == 403

    def test_higher_role_satisfies_lower_requirement(self):
        with patch("backend.security.rbac._load_key_map", return_value=KEYS):
            client = self._client("vp")
            r = client.get("/protected", headers={"X-API-Key": "key-admin"})
        assert r.status_code == 200

    def test_exact_role_match_passes(self):
        with patch("backend.security.rbac._load_key_map", return_value=KEYS):
            client = self._client("cfo")
            r = client.get("/protected", headers={"X-API-Key": "key-cfo"})
        assert r.status_code == 200

    def test_error_message_includes_role_info(self):
        with patch("backend.security.rbac._load_key_map", return_value=KEYS):
            client = self._client("admin")
            r = client.get("/protected", headers={"X-API-Key": "key-analyst"})
        assert r.status_code == 403
        assert "admin" in r.json()["detail"].lower() or "insufficient" in r.json()["detail"].lower()

    def test_invalid_min_role_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown role"):
            require_role("overlord")


# ── Role hierarchy completeness ───────────────────────────────────────────────

class TestRoleHierarchy:
    """Verify the role ladder is strict and transitive."""

    ORDERED = ["analyst", "manager", "vp", "cfo", "admin"]

    def test_each_role_passes_all_lower(self):
        for i, role in enumerate(self.ORDERED):
            u = RBACUser("k", role)
            for lower in self.ORDERED[:i + 1]:
                assert u.has_role(lower), f"{role} should satisfy {lower}"

    def test_each_role_fails_all_higher(self):
        for i, role in enumerate(self.ORDERED):
            u = RBACUser("k", role)
            for higher in self.ORDERED[i + 1:]:
                assert not u.has_role(higher), f"{role} should NOT satisfy {higher}"
