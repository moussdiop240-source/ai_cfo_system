"""
Live LLM integration tests.

These tests call real LLM backends — they are skipped automatically when
the backend is not available.

Ollama tests:  skipped unless Ollama is running at localhost:11434
Anthropic tests: skipped unless ANTHROPIC_API_KEY is set

Run live tests:
    ollama serve &
    python -m pytest tests/test_integration_ollama.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Availability probes ────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    """Return True if Ollama is reachable at localhost:11434."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _anthropic_key_set() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


SKIP_OLLAMA     = pytest.mark.skipif(not _ollama_available(),    reason="Ollama not running at localhost:11434")
SKIP_ANTHROPIC  = pytest.mark.skipif(not _anthropic_key_set(),  reason="ANTHROPIC_API_KEY not set")


# ── Ollama integration ─────────────────────────────────────────────────────────

@SKIP_OLLAMA
class TestOllamaIntegration:
    """
    Calls the real Ollama adapter.  Requires `ollama serve` with at least
    one model pulled (llama3, llama3.2, or mistral recommended).
    """

    @pytest.fixture(autouse=True)
    def _set_backend(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "ollama")

    def _get_adapter(self):
        # Reset singleton so env var is picked up
        import backend.llm.adapter as _mod
        _mod._singleton = None
        from backend.llm.adapter import get_adapter
        return get_adapter()

    def test_adapter_connects_to_ollama(self):
        adapter = self._get_adapter()
        assert adapter.active_backend == "ollama"

    def test_complete_returns_non_empty_string(self):
        adapter = self._get_adapter()
        result = adapter.complete(
            system="You are a helpful assistant.",
            user="Say exactly: HELLO",
            max_tokens=20,
        )
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_complete_with_financial_prompt(self):
        adapter = self._get_adapter()
        result = adapter.complete(
            system="You are a CPA. Answer in one sentence.",
            user="What does ASC 606 govern?",
            max_tokens=80,
        )
        assert isinstance(result, str)
        assert len(result) > 10

    def test_complete_returns_relevant_content(self):
        adapter = self._get_adapter()
        result = adapter.complete(
            system="You are a finance expert.",
            user="Define EBITDA in 10 words or less.",
            max_tokens=50,
        )
        # Should mention earnings or income or similar
        lower = result.lower()
        assert any(kw in lower for kw in ("earnings", "income", "interest", "tax", "depreciation", "amortization", "ebitda"))

    def test_multiple_calls_are_independent(self):
        adapter = self._get_adapter()
        r1 = adapter.complete("Answer briefly.", "What is revenue?", max_tokens=40)
        r2 = adapter.complete("Answer briefly.", "What is COGS?", max_tokens=40)
        assert isinstance(r1, str)
        assert isinstance(r2, str)

    def test_empty_system_prompt_handled(self):
        adapter = self._get_adapter()
        result = adapter.complete("", "Say OK", max_tokens=10)
        assert isinstance(result, str)

    def test_large_token_limit_accepted(self):
        adapter = self._get_adapter()
        result = adapter.complete(
            "You are a financial analyst.",
            "List 3 key financial ratios and what they measure.",
            max_tokens=300,
        )
        assert isinstance(result, str)
        assert len(result) > 20


# ── Anthropic integration ──────────────────────────────────────────────────────

@SKIP_ANTHROPIC
class TestAnthropicIntegration:
    """
    Calls the real Anthropic API.  Requires ANTHROPIC_API_KEY env var.
    Uses claude-haiku-4-5 for cost efficiency.
    """

    @pytest.fixture(autouse=True)
    def _set_backend(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "anthropic")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    def _get_adapter(self):
        import backend.llm.adapter as _mod
        _mod._singleton = None
        from backend.llm.adapter import get_adapter
        return get_adapter()

    def test_adapter_connects_to_anthropic(self):
        adapter = self._get_adapter()
        assert adapter.active_backend == "anthropic"

    def test_complete_returns_non_empty_string(self):
        adapter = self._get_adapter()
        result = adapter.complete(
            system="You are a helpful assistant.",
            user="Say exactly: HELLO",
            max_tokens=20,
        )
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_complete_with_financial_prompt(self):
        adapter = self._get_adapter()
        result = adapter.complete(
            system="You are a CPA. Answer in one sentence.",
            user="What does ASC 606 govern?",
            max_tokens=100,
        )
        assert isinstance(result, str)
        assert len(result) > 10


# ── Adapter unit tests (no live call needed) ──────────────────────────────────

class TestAdapterUnit:
    """Fast unit tests that don't require a live backend."""

    def test_adapter_singleton_returns_same_instance(self, monkeypatch):
        import backend.llm.adapter as _mod
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        _mod._singleton = None
        from backend.llm.adapter import get_adapter
        a1 = get_adapter()
        a2 = get_adapter()
        assert a1 is a2

    def test_adapter_has_active_backend_attribute(self, monkeypatch):
        import backend.llm.adapter as _mod
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        _mod._singleton = None
        from backend.llm.adapter import get_adapter
        adapter = get_adapter()
        assert hasattr(adapter, "active_backend")
        assert adapter.active_backend in ("ollama", "anthropic")

    def test_adapter_has_complete_method(self, monkeypatch):
        import backend.llm.adapter as _mod
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        _mod._singleton = None
        from backend.llm.adapter import get_adapter
        adapter = get_adapter()
        assert callable(adapter.complete)

    def test_adapter_resets_on_env_change(self, monkeypatch):
        import backend.llm.adapter as _mod
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        _mod._singleton = None
        from backend.llm.adapter import get_adapter
        adapter = get_adapter()
        assert adapter is not None
        _mod._singleton = None  # manual reset
        adapter2 = get_adapter()
        assert adapter2 is not None
