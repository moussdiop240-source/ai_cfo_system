"""Tests for the FIELD_ENCRYPTION_KEY rotation script."""
import os
import sys
from unittest.mock import patch

from cryptography.fernet import Fernet

sys.path.insert(0, ".")


def _key():
    return Fernet.generate_key().decode()


class TestRotateHelpers:
    def test_already_new_key_true(self):
        from scripts.rotate_encryption_key import _already_new_key, _encrypt
        k = _key()
        f = Fernet(k.encode())
        ct = _encrypt(f, "hello")
        assert _already_new_key(f, ct) is True

    def test_already_new_key_false_wrong_key(self):
        from scripts.rotate_encryption_key import _already_new_key, _encrypt
        k1, k2 = _key(), _key()
        f1, f2 = Fernet(k1.encode()), Fernet(k2.encode())
        ct = _encrypt(f1, "hello")
        assert _already_new_key(f2, ct) is False

    def test_already_new_key_false_for_plaintext(self):
        from scripts.rotate_encryption_key import _already_new_key
        f = Fernet(_key().encode())
        assert _already_new_key(f, "not encrypted") is False

    def test_decrypt_plaintext_passthrough(self):
        from scripts.rotate_encryption_key import _decrypt
        f = Fernet(_key().encode())
        assert _decrypt(f, "legacy plaintext") == "legacy plaintext"

    def test_decrypt_fernet_token(self):
        from scripts.rotate_encryption_key import _decrypt, _encrypt
        f = Fernet(_key().encode())
        ct = _encrypt(f, "secret value")
        assert _decrypt(f, ct) == "secret value"

    def test_decrypt_empty_string(self):
        from scripts.rotate_encryption_key import _decrypt
        f = Fernet(_key().encode())
        assert _decrypt(f, "") == ""

    def test_encrypt_roundtrip(self):
        from scripts.rotate_encryption_key import _decrypt, _encrypt
        f = Fernet(_key().encode())
        assert _decrypt(f, _encrypt(f, "financial data")) == "financial data"


class TestRotateMain:
    def test_fails_when_keys_missing(self):
        from scripts.rotate_encryption_key import rotate
        with patch.dict(os.environ, {"OLD_FIELD_ENCRYPTION_KEY": "", "NEW_FIELD_ENCRYPTION_KEY": ""}, clear=False):
            assert rotate() == 1

    def test_fails_when_keys_identical(self):
        from scripts.rotate_encryption_key import rotate
        k = _key()
        with patch.dict(os.environ, {"OLD_FIELD_ENCRYPTION_KEY": k, "NEW_FIELD_ENCRYPTION_KEY": k}, clear=False):
            assert rotate() == 1

    def test_dry_run_does_not_modify_db(self):
        from scripts.rotate_encryption_key import rotate
        old_k, new_k = _key(), _key()
        old_f = Fernet(old_k.encode())

        # Build a tiny in-memory SQLite DB with one encrypted row
        from sqlalchemy import Column, String, Text, create_engine, text
        from sqlalchemy.orm import declarative_base
        Base = declarative_base()

        class FakeTask(Base):
            __tablename__ = "tasks"
            id = Column(String, primary_key=True)
            final_report = Column(Text)
            analysis_narrative = Column(Text)

        class FakeApproval(Base):
            __tablename__ = "approvals"
            id = Column(String, primary_key=True)
            feedback = Column(Text)

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        original_ct = old_f.encrypt(b"secret report").decode()
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO tasks VALUES ('t1', :r, :n)"),
                         {"r": original_ct, "n": None})
            conn.execute(text("INSERT INTO approvals VALUES ('a1', NULL)"))

        with patch.dict(os.environ, {
            "OLD_FIELD_ENCRYPTION_KEY": old_k,
            "NEW_FIELD_ENCRYPTION_KEY": new_k,
            "DATABASE_URL": "sqlite:///:memory:",
        }):
            # Patch engine creation to return our seeded engine
            with patch("sqlalchemy.create_engine", return_value=engine):
                result = rotate(dry_run=True)

        assert result == 0
        # Verify original ciphertext untouched
        with engine.connect() as conn:
            row = conn.execute(text("SELECT final_report FROM tasks WHERE id='t1'")).fetchone()
        assert row[0] == original_ct

    def test_rotation_re_encrypts_with_new_key(self):
        from scripts.rotate_encryption_key import rotate
        old_k, new_k = _key(), _key()
        old_f = Fernet(old_k.encode())
        new_f = Fernet(new_k.encode())

        from sqlalchemy import Column, String, Text, create_engine, text
        from sqlalchemy.orm import declarative_base
        Base = declarative_base()

        class FakeTask(Base):
            __tablename__ = "tasks"
            id = Column(String, primary_key=True)
            final_report = Column(Text)
            analysis_narrative = Column(Text)

        class FakeApproval(Base):
            __tablename__ = "approvals"
            id = Column(String, primary_key=True)
            feedback = Column(Text)

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        original_ct = old_f.encrypt(b"confidential narrative").decode()
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO tasks VALUES ('t1', :r, :n)"),
                         {"r": original_ct, "n": "plaintext"})
            conn.execute(text("INSERT INTO approvals VALUES ('a1', NULL)"))

        with patch.dict(os.environ, {
            "OLD_FIELD_ENCRYPTION_KEY": old_k,
            "NEW_FIELD_ENCRYPTION_KEY": new_k,
            "DATABASE_URL": "sqlite:///:memory:",
        }):
            with patch("sqlalchemy.create_engine", return_value=engine):
                result = rotate(dry_run=False)

        assert result == 0
        with engine.connect() as conn:
            row = conn.execute(text("SELECT final_report FROM tasks WHERE id='t1'")).fetchone()
        new_ct = row[0]
        assert new_ct != original_ct
        assert new_f.decrypt(new_ct.encode()).decode() == "confidential narrative"
