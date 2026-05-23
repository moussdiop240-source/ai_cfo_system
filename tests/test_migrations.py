"""
Alembic migration tests.

Verifies that:
- upgrade head creates all 7 expected tables on a fresh SQLite DB
- downgrade removes all tables
- upgrade is idempotent (running twice is safe)
- alembic_version table is stamped at revision 0001 after upgrade
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _run_alembic(command: list[str], db_path: str) -> int:
    """Run an alembic CLI command against a temp DB, return exit code."""
    import subprocess
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path}"}
    result = subprocess.run(
        [sys.executable, "-m", "alembic"] + command,
        env=env,
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    return result.returncode


def _table_names(db_path: str) -> set[str]:
    """Return the set of table names in a SQLite DB."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


EXPECTED_TABLES = {
    "tasks", "approvals", "rag_documents",
    "mem_company", "mem_period_snapshot", "mem_kpi_ts", "mem_insights",
    "alembic_version",
}


class TestMigrations:
    def test_upgrade_head_creates_all_tables(self, tmp_path):
        db = str(tmp_path / "test.db")
        rc = _run_alembic(["upgrade", "head"], db)
        assert rc == 0, "alembic upgrade head failed"
        tables = _table_names(db)
        assert EXPECTED_TABLES.issubset(tables), (
            f"Missing tables: {EXPECTED_TABLES - tables}"
        )

    def test_alembic_version_stamped_at_0001(self, tmp_path):
        db = str(tmp_path / "test.db")
        _run_alembic(["upgrade", "head"], db)
        import sqlite3
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT version_num FROM alembic_version").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "0001"

    def test_downgrade_removes_app_tables(self, tmp_path):
        db = str(tmp_path / "test.db")
        _run_alembic(["upgrade", "head"], db)
        rc = _run_alembic(["downgrade", "base"], db)
        assert rc == 0, "alembic downgrade base failed"
        tables = _table_names(db)
        # Only alembic_version should remain after full downgrade
        app_tables = EXPECTED_TABLES - {"alembic_version"}
        assert not app_tables.intersection(tables), (
            f"Tables still present after downgrade: {app_tables.intersection(tables)}"
        )

    def test_upgrade_is_idempotent(self, tmp_path):
        db = str(tmp_path / "test.db")
        rc1 = _run_alembic(["upgrade", "head"], db)
        rc2 = _run_alembic(["upgrade", "head"], db)
        assert rc1 == 0
        assert rc2 == 0  # running twice should not error
        tables = _table_names(db)
        assert EXPECTED_TABLES.issubset(tables)

    def test_tasks_table_has_expected_columns(self, tmp_path):
        db = str(tmp_path / "test.db")
        _run_alembic(["upgrade", "head"], db)
        import sqlite3
        with sqlite3.connect(db) as conn:
            info = conn.execute("PRAGMA table_info(tasks)").fetchall()
        col_names = {row[1] for row in info}
        for col in ("id", "task_type", "company_name", "period", "status", "final_report"):
            assert col in col_names, f"Column '{col}' missing from tasks table"

    def test_mem_company_unique_constraint(self, tmp_path):
        db = str(tmp_path / "test.db")
        _run_alembic(["upgrade", "head"], db)
        from datetime import datetime
        import sqlite3
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO mem_company (company_name, first_seen, last_updated) VALUES (?, ?, ?)",
                ("Acme Corp", datetime.utcnow(), datetime.utcnow()),
            )
            conn.commit()
            with pytest.raises(Exception):  # UNIQUE constraint violation
                conn.execute(
                    "INSERT INTO mem_company (company_name, first_seen, last_updated) VALUES (?, ?, ?)",
                    ("Acme Corp", datetime.utcnow(), datetime.utcnow()),
                )
                conn.commit()
