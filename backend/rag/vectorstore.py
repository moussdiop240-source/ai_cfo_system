"""
SQLite-backed local vector store — drop-in for pgvector in dev/test.

Stores embeddings as JSON blobs; uses numpy cosine similarity for search.
Same interface as pgvector path in pipeline.py so the two are interchangeable.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple


class SQLiteVectorStore:
    """
    Lightweight vector store for local development and CI (no PostgreSQL needed).

    Schema:
        cfo_documents(id TEXT PK, title TEXT, content TEXT,
                      embedding TEXT,   -- JSON list[float]
                      category TEXT, min_role TEXT, metadata TEXT)
    """

    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS cfo_documents (
        id       TEXT PRIMARY KEY,
        title    TEXT,
        content  TEXT,
        embedding TEXT NOT NULL,
        category  TEXT DEFAULT 'general',
        min_role  TEXT DEFAULT 'analyst',
        metadata  TEXT DEFAULT '{}'
    );
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        conn = self._get_conn()
        conn.execute(self._CREATE_SQL)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def upsert(
        self,
        doc_id: str,
        title: str,
        content: str,
        embedding: List[float],
        category: str = "general",
        min_role: str = "analyst",
        metadata: Optional[Dict] = None,
    ) -> str:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT INTO cfo_documents
                    (id, title, content, embedding, category, min_role, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    content=excluded.content,
                    embedding=excluded.embedding,
                    category=excluded.category,
                    metadata=excluded.metadata
                """,
                (
                    doc_id,
                    title,
                    content,
                    json.dumps(embedding),
                    category,
                    min_role,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
        return doc_id

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[str, str, str, str, float]]:
        """
        Return [(id, title, content, category, score)] sorted by cosine similarity descending.
        """
        try:
            import numpy as np
        except ImportError:
            return self._linear_search(query_embedding, top_k)

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, title, content, category, embedding FROM cfo_documents"
        ).fetchall()

        if not rows:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1.0

        scored = []
        for row in rows:
            emb = np.array(json.loads(row["embedding"]), dtype=np.float32)
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(q, emb) / (q_norm * emb_norm))
            scored.append((row["id"], row["title"], row["content"], row["category"], score))

        scored.sort(key=lambda x: x[4], reverse=True)
        return scored[:top_k]

    def _linear_search(
        self, query_embedding: List[float], top_k: int
    ) -> List[Tuple[str, str, str, str, float]]:
        """Pure-Python fallback cosine similarity (no numpy)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, title, content, category, embedding FROM cfo_documents"
        ).fetchall()

        q = query_embedding
        q_norm = sum(x * x for x in q) ** 0.5 or 1.0

        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            dot = sum(a * b for a, b in zip(q, emb))
            emb_norm = sum(x * x for x in emb) ** 0.5 or 1.0
            score = dot / (q_norm * emb_norm)
            scored.append((row["id"], row["title"], row["content"], row["category"], score))

        scored.sort(key=lambda x: x[4], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM cfo_documents").fetchone()[0]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# Module-level singleton (one store per process, keyed by path)
_stores: Dict[str, SQLiteVectorStore] = {}
_store_lock = threading.Lock()


def get_vector_store(db_path: str = ":memory:") -> SQLiteVectorStore:
    with _store_lock:
        if db_path not in _stores:
            _stores[db_path] = SQLiteVectorStore(db_path)
        return _stores[db_path]
