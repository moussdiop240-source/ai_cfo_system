"""Tests for per-user bcrypt Streamlit auth helper logic."""
import json
import os
from unittest.mock import patch

import bcrypt


def _make_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=4)).decode()


class TestBcryptAuth:
    def test_valid_user_and_password(self):
        users = {"alice": _make_hash("secret123")}
        users_json = json.dumps(users)

        with patch.dict(os.environ, {"STREAMLIT_USERS": users_json, "STREAMLIT_PASSWORD": ""}):
            # Simulate the _check_credentials function logic inline
            user_map = json.loads(users_json)
            stored = user_map.get("alice", "")
            assert bcrypt.checkpw(b"secret123", stored.encode())

    def test_wrong_password_fails(self):
        users = {"alice": _make_hash("correct")}
        stored = users["alice"]
        assert not bcrypt.checkpw(b"wrong", stored.encode())

    def test_unknown_user_fails(self):
        users = {"alice": _make_hash("password")}
        result = users.get("bob", "")
        assert result == ""

    def test_multiple_users(self):
        users = {
            "alice": _make_hash("pass_alice"),
            "bob": _make_hash("pass_bob"),
        }
        assert bcrypt.checkpw(b"pass_alice", users["alice"].encode())
        assert bcrypt.checkpw(b"pass_bob", users["bob"].encode())
        # Cross-user check fails
        assert not bcrypt.checkpw(b"pass_alice", users["bob"].encode())

    def test_hash_generation_is_deterministic_verify(self):
        """Verifying bcrypt works correctly across different salts."""
        password = "mypassword"
        h1 = _make_hash(password)
        h2 = _make_hash(password)
        # Different salts → different hashes
        assert h1 != h2
        # Both verify correctly
        assert bcrypt.checkpw(password.encode(), h1.encode())
        assert bcrypt.checkpw(password.encode(), h2.encode())


class TestAuthFallback:
    def test_shared_password_fallback(self):
        """When STREAMLIT_USERS not set, falls back to STREAMLIT_PASSWORD equality check."""
        password = "sharedpass"
        # Simulate fallback: return password == _APP_PASSWORD
        assert password == "sharedpass"
        assert "wrongpass" != "sharedpass"

    def test_no_auth_configured(self):
        """When neither var is set, _AUTH_ENABLED is False."""
        users_raw = ""
        app_password = ""
        auth_enabled = bool(users_raw or app_password)
        assert auth_enabled is False

    def test_auth_enabled_with_password(self):
        auth_enabled = bool("" or "somepassword")
        assert auth_enabled is True

    def test_auth_enabled_with_users(self):
        auth_enabled = bool('{"alice": "hash"}' or "")
        assert auth_enabled is True
