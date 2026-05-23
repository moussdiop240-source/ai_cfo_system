"""
Field-level Fernet encryption for sensitive database columns.

Usage:
- Set FIELD_ENCRYPTION_KEY to a URL-safe base64-encoded 32-byte key.
  Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
- When FIELD_ENCRYPTION_KEY is absent, columns store and return plaintext (dev mode).
- Existing plaintext rows are read back correctly after key is set (backward-compatible):
  Fernet tokens always start with "gAAAAA"; anything else is returned as-is.

Apply EncryptedText as a TypeDecorator column type in SQLAlchemy models:
    final_report = Column(EncryptedText)
"""
import os
from typing import Optional

from sqlalchemy import String, TypeDecorator

_key_raw = os.environ.get("FIELD_ENCRYPTION_KEY", "")

_fernet = None
if _key_raw:
    try:
        from cryptography.fernet import Fernet
        _fernet = Fernet(_key_raw.encode())
    except Exception as exc:
        import logging
        logging.getLogger("ai_cfo.field_encryption").error(
            "FIELD_ENCRYPTION_KEY is set but invalid — encryption disabled: %s", exc
        )

_FERNET_PREFIX = "gAAAAA"


def encrypt(plaintext: Optional[str]) -> Optional[str]:
    """Encrypt a string if a key is configured; return plaintext otherwise."""
    if plaintext is None or _fernet is None:
        return plaintext
    return _fernet.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: Optional[str]) -> Optional[str]:
    """Decrypt a Fernet token, or return the value as-is if it looks like plaintext."""
    if ciphertext is None or _fernet is None:
        return ciphertext
    if not ciphertext.startswith(_FERNET_PREFIX):
        return ciphertext  # pre-encryption row — pass through
    try:
        from cryptography.fernet import InvalidToken
        return _fernet.decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        return ciphertext  # wrong key or corruption — return raw to avoid data loss


class EncryptedText(TypeDecorator):
    """SQLAlchemy TypeDecorator that transparently encrypts/decrypts Text columns."""
    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return encrypt(value)

    def process_result_value(self, value, dialect):
        return decrypt(value)
