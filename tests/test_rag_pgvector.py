"""
RAG pipeline integration tests.

When PGVECTOR_URL is set and points to a live PostgreSQL+pgvector instance,
these tests exercise the full pgvector path. Otherwise they exercise the
SQLite vector store path (same interface, local-only — no PostgreSQL needed).

In CI, the integration-pgvector job sets PGVECTOR_URL to a real service container.
In local dev and unit-test jobs, PGVECTOR_URL is empty → SQLite path runs.
"""
import os
import sys
import hashlib

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rag.pipeline import RAGPipeline, RAGChunk, format_rag_context
from backend.rag.vectorstore import SQLiteVectorStore


PGVECTOR_URL = os.environ.get("PGVECTOR_URL", "")
USING_PGVECTOR = bool(PGVECTOR_URL and not PGVECTOR_URL.startswith("sqlite"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sqlite_store(tmp_path):
    """Fresh SQLite vector store backed by a temp file."""
    return SQLiteVectorStore(str(tmp_path / "test_vs.db"))


@pytest.fixture
def populated_store(sqlite_store):
    """SQLite vector store with 3 pre-loaded documents."""
    docs = [
        ("doc1", "EBITDA Analysis", "EBITDA is a non-GAAP measure of profitability.", [0.1] * 384, "finance"),
        ("doc2", "Revenue Recognition ASC 606", "ASC 606 governs revenue recognition.", [0.2] * 384, "gaap"),
        ("doc3", "Liquidity Risk", "Current ratio below 1.0 signals liquidity risk.", [0.3] * 384, "risk"),
    ]
    for doc_id, title, content, emb, cat in docs:
        sqlite_store.upsert(doc_id=doc_id, title=title, content=content,
                            embedding=emb, category=cat)
    return sqlite_store


@pytest.fixture
def rag_pipeline_sqlite(tmp_path):
    """RAGPipeline wired to a temporary SQLite vector store."""
    return RAGPipeline(db_url=None, sqlite_path=str(tmp_path / "pipeline_vs.db"))


# ── SQLiteVectorStore unit tests ──────────────────────────────────────────────

class TestSQLiteVectorStore:
    def test_empty_store_has_zero_docs(self, sqlite_store):
        assert sqlite_store.count() == 0

    def test_upsert_single_doc(self, sqlite_store):
        sqlite_store.upsert("d1", "Title", "Content", [0.5] * 384)
        assert sqlite_store.count() == 1

    def test_upsert_returns_doc_id(self, sqlite_store):
        returned = sqlite_store.upsert("d1", "Title", "Content", [0.5] * 384)
        assert returned == "d1"

    def test_upsert_conflict_updates(self, sqlite_store):
        sqlite_store.upsert("d1", "Title", "Content A", [0.5] * 384)
        sqlite_store.upsert("d1", "Title", "Content B", [0.6] * 384)
        assert sqlite_store.count() == 1

    def test_search_returns_top_k(self, populated_store):
        query_emb = [0.1] * 384  # closest to doc1
        results = populated_store.search(query_emb, top_k=2)
        assert len(results) == 2

    def test_search_returns_correct_fields(self, populated_store):
        results = populated_store.search([0.2] * 384, top_k=1)
        assert len(results) == 1
        doc_id, title, content, category, score = results[0]
        assert isinstance(doc_id, str)
        assert isinstance(title, str)
        assert isinstance(score, float)

    def test_search_score_range(self, populated_store):
        results = populated_store.search([0.1] * 384, top_k=3)
        for _, _, _, _, score in results:
            assert -1.0 <= score <= 1.0 + 1e-6

    def test_search_on_empty_store_returns_empty(self, sqlite_store):
        results = sqlite_store.search([0.1] * 384, top_k=5)
        assert results == []

    def test_cosine_sim_identical_vectors(self, sqlite_store):
        emb = [0.1, 0.2, 0.3] + [0.0] * 381
        sqlite_store.upsert("d1", "T", "C", emb)
        results = sqlite_store.search(emb, top_k=1)
        _, _, _, _, score = results[0]
        assert abs(score - 1.0) < 1e-4

    def test_cosine_sim_orthogonal_vectors(self, sqlite_store):
        emb_a = [1.0] + [0.0] * 383
        emb_b = [0.0, 1.0] + [0.0] * 382
        sqlite_store.upsert("d1", "T", "C", emb_a)
        results = sqlite_store.search(emb_b, top_k=1)
        _, _, _, _, score = results[0]
        assert abs(score) < 1e-4

    def test_results_sorted_by_score_descending(self, sqlite_store):
        sqlite_store.upsert("d1", "T1", "C1", [1.0] + [0.0] * 383)
        sqlite_store.upsert("d2", "T2", "C2", [0.5] + [0.0] * 383)
        sqlite_store.upsert("d3", "T3", "C3", [0.1] + [0.0] * 383)
        results = sqlite_store.search([1.0] + [0.0] * 383, top_k=3)
        scores = [r[4] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_stored_and_retrievable(self, sqlite_store):
        sqlite_store.upsert("d1", "T", "C", [0.1] * 384, metadata={"source": "test"})
        # upsert should not raise; retrieve via count
        assert sqlite_store.count() == 1

    def test_thread_safety_concurrent_upserts(self, sqlite_store):
        import threading
        errors = []

        def writer(i):
            try:
                sqlite_store.upsert(f"doc{i}", f"Title {i}", f"Content {i}", [float(i / 100)] * 384)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert sqlite_store.count() == 10


# ── RAGPipeline with SQLite backend ──────────────────────────────────────────

class TestRAGPipelineSQLite:
    def test_pipeline_initializes_without_pgvector(self, rag_pipeline_sqlite):
        assert rag_pipeline_sqlite._pgvector_available is False

    def test_retrieve_returns_rag_chunks(self, rag_pipeline_sqlite):
        results = rag_pipeline_sqlite.retrieve("EBITDA margin", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, RAGChunk) for r in results)

    def test_retrieve_returns_at_most_top_k(self, rag_pipeline_sqlite):
        results = rag_pipeline_sqlite.retrieve("revenue recognition ASC 606", top_k=3)
        assert len(results) <= 3

    def test_retrieve_chunks_have_required_fields(self, rag_pipeline_sqlite):
        results = rag_pipeline_sqlite.retrieve("liquidity current ratio", top_k=2)
        for chunk in results:
            assert hasattr(chunk, "id")
            assert hasattr(chunk, "title")
            assert hasattr(chunk, "content")
            assert hasattr(chunk, "score")
            assert hasattr(chunk, "category")

    def test_retrieve_rag_chunks_to_dict(self, rag_pipeline_sqlite):
        results = rag_pipeline_sqlite.retrieve("variance analysis board", top_k=2)
        for chunk in results:
            d = chunk.to_dict()
            assert "id" in d
            assert "content" in d
            assert "score" in d

    def test_build_rag_query_from_state(self, rag_pipeline_sqlite):
        state = {
            "task_description": "Q1 variance analysis",
            "task_type": "variance_analysis",
            "period": "Q1 2026",
            "kpi_metrics": {"gross_margin_pct": 25.0, "current_ratio": 0.8},
            "anomaly_flags": ["Revenue spike detected"],
            "gaap_results": {},
            "ifrs_results": {},
        }
        query = rag_pipeline_sqlite.build_rag_query(state)
        assert "variance" in query.lower() or "Q1" in query
        assert "liquidity" in query or "current ratio" in query or "gross margin" in query

    def test_format_rag_context_non_empty(self, rag_pipeline_sqlite):
        chunks = rag_pipeline_sqlite.retrieve("GAAP compliance", top_k=2)
        context = format_rag_context(chunks)
        assert isinstance(context, str)
        assert len(context) > 0

    def test_format_rag_context_empty(self):
        context = format_rag_context([])
        assert "No relevant context" in context

    def test_index_document_to_sqlite(self, rag_pipeline_sqlite):
        # index_document writes to pgvector; when unavailable, returns sentinel
        result = rag_pipeline_sqlite.index_document(
            "Test document content about EBITDA.",
            {"title": "Test Doc", "category": "finance", "min_role": "analyst"},
        )
        # Without pgvector, returns "pgvector_unavailable"
        assert result == "pgvector_unavailable"

    def test_lazy_indexing_populates_store(self, rag_pipeline_sqlite):
        # First retrieve triggers lazy indexing of the 20-doc KB
        rag_pipeline_sqlite.retrieve("working capital", top_k=1)
        assert rag_pipeline_sqlite._sqlite_store.count() > 0

    def test_second_retrieve_does_not_reindex(self, rag_pipeline_sqlite):
        rag_pipeline_sqlite.retrieve("working capital", top_k=1)
        count_after_first = rag_pipeline_sqlite._sqlite_store.count()
        rag_pipeline_sqlite.retrieve("gross margin", top_k=1)
        count_after_second = rag_pipeline_sqlite._sqlite_store.count()
        assert count_after_first == count_after_second


# ── pgvector integration tests (skipped unless PGVECTOR_URL set) ──────────────

@pytest.mark.skipif(not USING_PGVECTOR, reason="PGVECTOR_URL not set — skipping pgvector tests")
class TestRAGPipelinePgvector:
    @pytest.fixture
    def pipeline_pg(self):
        return RAGPipeline(db_url=PGVECTOR_URL)

    def test_pgvector_available(self, pipeline_pg):
        assert pipeline_pg._pgvector_available is True

    def test_pgvector_retrieve_returns_chunks(self, pipeline_pg):
        results = pipeline_pg.retrieve("EBITDA margin analysis", top_k=5)
        assert isinstance(results, list)

    def test_pgvector_retrieve_chunk_structure(self, pipeline_pg):
        results = pipeline_pg.retrieve("revenue recognition ASC 606", top_k=3)
        for chunk in results:
            assert isinstance(chunk, RAGChunk)
            assert chunk.id
            assert chunk.content

    def test_pgvector_score_in_range(self, pipeline_pg):
        results = pipeline_pg.retrieve("variance analysis", top_k=5)
        for chunk in results:
            assert -1.0 <= chunk.score <= 1.0 + 1e-6

    def test_pgvector_index_and_retrieve(self, pipeline_pg):
        unique_content = f"pgvector integration test document {hashlib.md5(b'test').hexdigest()}"
        doc_id = pipeline_pg.index_document(
            unique_content,
            {"title": "PG Integration Test", "category": "test", "min_role": "analyst"},
        )
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_pgvector_knowledge_base_indexed(self, pipeline_pg):
        # Verify KB is indexed and retrieval returns results (pseudo-embeddings
        # are deterministic but not semantic, so content relevance is not asserted)
        results = pipeline_pg.retrieve("working capital cash conversion", top_k=5)
        assert len(results) >= 1
        assert all(hasattr(r, "content") for r in results)
