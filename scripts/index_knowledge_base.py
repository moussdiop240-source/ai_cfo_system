"""
Index the 20-document knowledge base into pgvector (production) or
SQLite vector store (local dev / CI).

Usage:
    # pgvector (production)
    PGVECTOR_URL=postgresql://... python scripts/index_knowledge_base.py

    # SQLite (local dev, no PostgreSQL required)
    VECTORSTORE_PATH=./data/vectorstore.db python scripts/index_knowledge_base.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rag.knowledge_base import KNOWLEDGE_BASE


# ── Embedding helper ──────────────────────────────────────────────────────────

def _embed(text: str) -> list:
    """Embed text using sentence-transformers, or deterministic pseudo-embedding."""
    try:
        from sentence_transformers import SentenceTransformer
        if not hasattr(_embed, "_model"):
            print("  Loading sentence-transformers (all-MiniLM-L6-v2)...")
            _embed._model = SentenceTransformer("all-MiniLM-L6-v2")
        return _embed._model.encode(text).tolist()
    except ImportError:
        import random
        digest = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = random.Random(digest)
        return [rng.uniform(-1, 1) for _ in range(384)]


# ── pgvector indexer ──────────────────────────────────────────────────────────

def index_pgvector(db_url: str) -> int:
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
        return 0

    print(f"Connecting to pgvector: {db_url[:40]}...")
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    indexed = 0
    for doc in KNOWLEDGE_BASE:
        doc_id = hashlib.sha256(doc["content"].encode()).hexdigest()[:16]
        embedding = _embed(doc["content"])
        metadata = json.dumps({
            "title": doc["title"],
            "category": doc.get("category", "general"),
            "min_role": doc.get("min_role", "analyst"),
        })
        cur.execute(
            """
            INSERT INTO cfo_documents
                (id, title, content, embedding, category, min_role, metadata)
            VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                content   = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata  = EXCLUDED.metadata
            """,
            (
                doc_id,
                doc["title"],
                doc["content"],
                embedding,
                doc.get("category", "general"),
                doc.get("min_role", "analyst"),
                metadata,
            ),
        )
        indexed += 1
        print(f"  [{indexed:02d}/{len(KNOWLEDGE_BASE)}] {doc['title'][:60]}")

    conn.commit()
    cur.close()
    conn.close()
    return indexed


# ── SQLite vector store indexer ────────────────────────────────────────────────

def index_sqlite(db_path: str) -> int:
    from backend.rag.vectorstore import SQLiteVectorStore
    store = SQLiteVectorStore(db_path)

    indexed = 0
    for doc in KNOWLEDGE_BASE:
        doc_id = hashlib.sha256(doc["content"].encode()).hexdigest()[:16]
        embedding = _embed(doc["content"])
        store.upsert(
            doc_id=doc_id,
            title=doc["title"],
            content=doc["content"],
            embedding=embedding,
            category=doc.get("category", "general"),
            min_role=doc.get("min_role", "analyst"),
            metadata={"title": doc["title"]},
        )
        indexed += 1
        print(f"  [{indexed:02d}/{len(KNOWLEDGE_BASE)}] {doc['title'][:60]}")

    store.close()
    return indexed


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    pgvector_url = os.environ.get("PGVECTOR_URL", "")
    sqlite_path  = os.environ.get("VECTORSTORE_PATH", "./data/vectorstore.db")

    if pgvector_url and not pgvector_url.startswith("sqlite"):
        print(f"Indexing {len(KNOWLEDGE_BASE)} documents into pgvector...")
        n = index_pgvector(pgvector_url)
    else:
        os.makedirs(os.path.dirname(sqlite_path) if os.path.dirname(sqlite_path) else ".", exist_ok=True)
        print(f"Indexing {len(KNOWLEDGE_BASE)} documents into SQLite: {sqlite_path}")
        n = index_sqlite(sqlite_path)

    print(f"\nDone. Indexed {n} documents.")


if __name__ == "__main__":
    main()
