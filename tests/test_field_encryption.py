"""Tests for field-level Fernet encryption TypeDecorator."""
import os
import sys
from unittest.mock import patch

import pytest


def _fresh_module():
    """Import field_encryption with a clean module cache so env patches take effect."""
    mods = [k for k in sys.modules if "field_encryption" in k]
    for m in mods:
        del sys.modules[m]
    import backend.security.field_encryption as fe
    return fe


class TestEncryptDecryptNoKey:
    def test_encrypt_returns_plaintext_when_no_key(self):
        with patch.dict(os.environ, {"FIELD_ENCRYPTION_KEY": ""}, clear=False):
            fe = _fresh_module()
            assert fe.encrypt("hello") == "hello"

    def test_decrypt_returns_plaintext_when_no_key(self):
        with patch.dict(os.environ, {"FIELD_ENCRYPTION_KEY": ""}, clear=False):
            fe = _fresh_module()
            assert fe.decrypt("hello") == "hello"

    def test_encrypt_none_returns_none(self):
        fe = _fresh_module()
        assert fe.encrypt(None) is None

    def test_decrypt_none_returns_none(self):
        fe = _fresh_module()
        assert fe.decrypt(None) is None


class TestEncryptDecryptWithKey:
    @pytest.fixture
    def fernet_key(self):
        from cryptography.fernet import Fernet
        return Fernet.generate_key().decode()

    def test_roundtrip(self, fernet_key):
        with patch.dict(os.environ, {"FIELD_ENCRYPTION_KEY": fernet_key}):
            fe = _fresh_module()
            plaintext = "Top-secret CFO narrative with financials"
            ciphertext = fe.encrypt(plaintext)
            assert ciphertext != plaintext
            assert ciphertext.startswith("gAAAAA")
            assert fe.decrypt(ciphertext) == plaintext

    def test_plaintext_passthrough_on_decrypt(self, fernet_key):
        """Existing unencrypted rows must still be readable after key is set."""
        with patch.dict(os.environ, {"FIELD_ENCRYPTION_KEY": fernet_key}):
            fe = _fresh_module()
            legacy_value = "old plaintext from before encryption was enabled"
            assert fe.decrypt(legacy_value) == legacy_value

    def test_empty_string_roundtrip(self, fernet_key):
        with patch.dict(os.environ, {"FIELD_ENCRYPTION_KEY": fernet_key}):
            fe = _fresh_module()
            assert fe.decrypt(fe.encrypt("")) == ""

    def test_different_ciphertexts_for_same_plaintext(self, fernet_key):
        """Fernet uses a random IV — same plaintext should produce different ciphertexts."""
        with patch.dict(os.environ, {"FIELD_ENCRYPTION_KEY": fernet_key}):
            fe = _fresh_module()
            c1 = fe.encrypt("same text")
            c2 = fe.encrypt("same text")
            assert c1 != c2  # random IV → different ciphertext


class TestEncryptedTextTypeDecorator:
    @pytest.fixture
    def fernet_key(self):
        from cryptography.fernet import Fernet
        return Fernet.generate_key().decode()

    def test_process_bind_encrypts(self, fernet_key):
        with patch.dict(os.environ, {"FIELD_ENCRYPTION_KEY": fernet_key}):
            fe = _fresh_module()
            col = fe.EncryptedText()
            result = col.process_bind_param("sensitive data", None)
            assert result.startswith("gAAAAA")

    def test_process_result_decrypts(self, fernet_key):
        with patch.dict(os.environ, {"FIELD_ENCRYPTION_KEY": fernet_key}):
            fe = _fresh_module()
            col = fe.EncryptedText()
            ciphertext = fe.encrypt("sensitive data")
            assert col.process_result_value(ciphertext, None) == "sensitive data"

    def test_process_result_passthrough_for_plaintext(self, fernet_key):
        with patch.dict(os.environ, {"FIELD_ENCRYPTION_KEY": fernet_key}):
            fe = _fresh_module()
            col = fe.EncryptedText()
            assert col.process_result_value("legacy unencrypted", None) == "legacy unencrypted"

    def test_process_bind_none_returns_none(self):
        fe = _fresh_module()
        col = fe.EncryptedText()
        assert col.process_bind_param(None, None) is None

    def test_process_result_none_returns_none(self):
        fe = _fresh_module()
        col = fe.EncryptedText()
        assert col.process_result_value(None, None) is None
