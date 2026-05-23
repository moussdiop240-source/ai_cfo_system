"""
RBAC key expiry tests.

Covers:
- Plain string value (no expiry) — backward-compatible
- Dict value with future expiry — passes
- Dict value with past expiry — returns None (→ 403)
- Dict value expiring today — passes (inclusive)
- Malformed expiry date — treated as expired
- FastAPI 403 response for expired key
"""
import json
import os
import sys
from datetime import date, timedelta
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.security.rbac import (
    RBACUser,
    _get_role,
    check_role,
    require_role,
    role_for_key,
)


# ── _get_role with plain string format ────────────────────────────────────────

class TestPlainStringFormat:
    def test_plain_string_returns_role(self):
        key_map = json.dumps({"mykey": "cfo"})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            role = _get_role("mykey")
        assert role == "cfo"

    def test_unknown_key_returns_none(self):
        key_map = json.dumps({"mykey": "cfo"})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            role = _get_role("unknown")
        assert role is None

    def test_plain_string_all_roles(self):
        for expected_role in ("analyst", "manager", "vp", "cfo", "admin"):
            key_map = json.dumps({"k": expected_role})
            with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
                assert _get_role("k") == expected_role


# ── _get_role with dict/expiry format ─────────────────────────────────────────

class TestDictFormatWithExpiry:
    def test_future_expiry_returns_role(self):
        future = (date.today() + timedelta(days=30)).isoformat()
        key_map = json.dumps({"mykey": {"role": "manager", "expires": future}})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            role = _get_role("mykey")
        assert role == "manager"

    def test_past_expiry_returns_none(self):
        past = (date.today() - timedelta(days=1)).isoformat()
        key_map = json.dumps({"mykey": {"role": "cfo", "expires": past}})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            role = _get_role("mykey")
        assert role is None

    def test_expiry_today_still_valid(self):
        today = date.today().isoformat()
        key_map = json.dumps({"mykey": {"role": "analyst", "expires": today}})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            role = _get_role("mykey")
        assert role == "analyst"

    def test_malformed_expiry_returns_none(self):
        key_map = json.dumps({"mykey": {"role": "cfo", "expires": "not-a-date"}})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            role = _get_role("mykey")
        assert role is None

    def test_missing_expires_field_returns_role(self):
        key_map = json.dumps({"mykey": {"role": "vp"}})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            role = _get_role("mykey")
        assert role == "vp"

    def test_empty_expires_string_returns_role(self):
        key_map = json.dumps({"mykey": {"role": "analyst", "expires": ""}})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            role = _get_role("mykey")
        assert role == "analyst"


# ── Mixed formats in same key map ─────────────────────────────────────────────

class TestMixedFormats:
    def test_plain_and_expiry_coexist(self):
        future = (date.today() + timedelta(days=10)).isoformat()
        key_map = json.dumps({
            "plain_key": "analyst",
            "expiry_key": {"role": "cfo", "expires": future},
        })
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            assert _get_role("plain_key") == "analyst"
            assert _get_role("expiry_key") == "cfo"

    def test_expired_key_alongside_valid_key(self):
        past = (date.today() - timedelta(days=5)).isoformat()
        key_map = json.dumps({
            "good_key": "manager",
            "bad_key": {"role": "admin", "expires": past},
        })
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            assert _get_role("good_key") == "manager"
            assert _get_role("bad_key") is None


# ── FastAPI 403 for expired key ───────────────────────────────────────────────

class TestExpiredKeyReturns403:
    def _make_app(self):
        from fastapi import Depends
        app = FastAPI()

        @app.get("/protected")
        def protected(user: RBACUser = Depends(require_role("analyst"))):
            return {"role": user.role}

        return app

    def test_expired_key_gets_403(self):
        past = (date.today() - timedelta(days=1)).isoformat()
        key_map = json.dumps({"expired": {"role": "cfo", "expires": past}})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            client = TestClient(self._make_app(), raise_server_exceptions=False)
            r = client.get("/protected", headers={"X-API-Key": "expired"})
        assert r.status_code == 403

    def test_future_expiry_key_gets_200(self):
        future = (date.today() + timedelta(days=30)).isoformat()
        key_map = json.dumps({"live_key": {"role": "analyst", "expires": future}})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            client = TestClient(self._make_app(), raise_server_exceptions=False)
            r = client.get("/protected", headers={"X-API-Key": "live_key"})
        assert r.status_code == 200

    def test_plain_string_key_unchanged(self):
        key_map = json.dumps({"plain": "cfo"})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            client = TestClient(self._make_app(), raise_server_exceptions=False)
            r = client.get("/protected", headers={"X-API-Key": "plain"})
        assert r.status_code == 200

    def test_missing_key_still_401(self):
        key_map = json.dumps({"plain": "cfo"})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            client = TestClient(self._make_app(), raise_server_exceptions=False)
            r = client.get("/protected")
        assert r.status_code == 401


# ── Utility helpers respect expiry ────────────────────────────────────────────

class TestUtilityHelpersRespectExpiry:
    def test_check_role_expired_key_returns_false(self):
        past = (date.today() - timedelta(days=1)).isoformat()
        key_map = json.dumps({"k": {"role": "admin", "expires": past}})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            assert check_role("k", "analyst") is False

    def test_role_for_key_expired_returns_none(self):
        past = (date.today() - timedelta(days=1)).isoformat()
        key_map = json.dumps({"k": {"role": "cfo", "expires": past}})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            assert role_for_key("k") is None

    def test_role_for_key_future_returns_role(self):
        future = (date.today() + timedelta(days=365)).isoformat()
        key_map = json.dumps({"k": {"role": "vp", "expires": future}})
        with patch.dict(os.environ, {"RBAC_KEYS": key_map}):
            assert role_for_key("k") == "vp"
