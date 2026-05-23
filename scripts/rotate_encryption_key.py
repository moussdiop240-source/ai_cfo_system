"""
Field encryption key rotation script.

Re-encrypts every sensitive column in the database from an old Fernet key
to a new one in a single atomic transaction. Safe to re-run: rows already
encrypted with the new key are detected and skipped.

Usage:
    OLD_FIELD_ENCRYPTION_KEY=<old-key> \
    NEW_FIELD_ENCRYPTION_KEY=<new-key> \
    DATABASE_URL=postgresql://... \
    python scripts/rotate_encryption_key.py

    # Dry-run (print counts, touch nothing):
    DRY_RUN=1 OLD_FIELD_ENCRYPTION_KEY=... NEW_FIELD_ENCRYPTION_KEY=... python ...

The script exits with code 0 on success, 1 on any error.
"""
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("rotate_key")

_FERNET_PREFIX = "gAAAAA"


def _make_fernet(key: str, label: str):
    try:
        from cryptography.fernet import Fernet
        return Fernet(key.encode())
    except Exception as exc:
        log.error("%s key is invalid: %s", label, exc)
        sys.exit(1)


def _decrypt(fernet, value: str) -> str:
    """Decrypt if it looks like a Fernet token, else return plaintext as-is."""
    if not value or not value.startswith(_FERNET_PREFIX):
        return value
    from cryptography.fernet import InvalidToken
    try:
        return fernet.decrypt(value.encode()).decode()
    except InvalidToken:
        return value  # already re-encrypted or corrupted — leave it


def _encrypt(fernet, value: str) -> str:
    if not value:
        return value
    return fernet.encrypt(value.encode()).decode()


def _already_new_key(new_fernet, value: str) -> bool:
    """Return True if the value decrypts successfully with the NEW key."""
    if not value or not value.startswith(_FERNET_PREFIX):
        return False
    from cryptography.fernet import InvalidToken
    try:
        new_fernet.decrypt(value.encode())
        return True
    except InvalidToken:
        return False


def rotate(dry_run: bool = False) -> int:
    old_key = os.environ.get("OLD_FIELD_ENCRYPTION_KEY", "")
    new_key = os.environ.get("NEW_FIELD_ENCRYPTION_KEY", "")

    if not old_key or not new_key:
        log.error("Both OLD_FIELD_ENCRYPTION_KEY and NEW_FIELD_ENCRYPTION_KEY must be set.")
        return 1

    if old_key == new_key:
        log.error("OLD and NEW keys are identical — nothing to rotate.")
        return 1

    old_fernet = _make_fernet(old_key, "OLD")
    new_fernet = _make_fernet(new_key, "NEW")

    db_url = os.environ.get("DATABASE_URL", "sqlite:///./ai_cfo.db")
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url)

    # Columns to rotate: (table, primary_key_col, [encrypted_cols])
    targets = [
        ("tasks",     "id", ["final_report", "analysis_narrative"]),
        ("approvals", "id", ["feedback"]),
    ]

    total_rotated = 0
    total_skipped = 0

    with engine.begin() as conn:
        for table, pk_col, columns in targets:
            col_list = ", ".join([pk_col] + columns)
            rows = conn.execute(text(f"SELECT {col_list} FROM {table}")).fetchall()  # noqa: S608
            log.info("Table %-12s: %d rows", table, len(rows))

            for row in rows:
                pk = row[0]
                updates = {}

                for i, col in enumerate(columns):
                    val = row[i + 1]
                    if not val:
                        continue
                    if _already_new_key(new_fernet, val):
                        total_skipped += 1
                        continue
                    plaintext = _decrypt(old_fernet, val)
                    updates[col] = _encrypt(new_fernet, plaintext)

                if updates and not dry_run:
                    set_clause = ", ".join(f"{c} = :{c}" for c in updates)
                    conn.execute(
                        text(f"UPDATE {table} SET {set_clause} WHERE {pk_col} = :pk"),  # noqa: S608
                        {**updates, "pk": pk},
                    )
                    total_rotated += len(updates)
                elif updates:
                    total_rotated += len(updates)

    mode = "DRY RUN — " if dry_run else ""
    log.info("%sRotated %d field(s), skipped %d already on new key.", mode, total_rotated, total_skipped)
    return 0


if __name__ == "__main__":
    dry_run = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
    sys.exit(rotate(dry_run=dry_run))
