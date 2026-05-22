"""
Tests for LLMAdapter — backend selection, JSON extraction, error handling.

No real API calls are made. All LLM interactions are mocked.
"""
import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.llm.adapter import (
    LLMAdapter,
    _extract_json,
    get_adapter,
    reset_adapter,
    trim_for_local,
)


# ── Backend selection ────────────────────────────────────────────────────────

class TestBackendSelection:
    def test_auto_selects_anthropic_when_key_set(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test", "LLM_BACKEND": "auto"}, clear=False):
            adapter = LLMAdapter()
            assert adapter.active_backend == "anthropic"

    def test_auto_selects_ollama_when_no_key(self):
        env = {"LLM_BACKEND": "auto"}
        with patch.dict(os.environ, env, clear=False):
            # Temporarily remove ANTHROPIC_API_KEY if set
            saved = os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                adapter = LLMAdapter()
                assert adapter.active_backend == "ollama"
            finally:
                if saved:
                    os.environ["ANTHROPIC_API_KEY"] = saved

    def test_explicit_anthropic_backend(self):
        adapter = LLMAdapter(backend="anthropic")
        assert adapter.active_backend == "anthropic"

    def test_explicit_ollama_backend(self):
        adapter = LLMAdapter(backend="ollama")
        assert adapter.active_backend == "ollama"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            LLMAdapter(backend="gpt4")

    def test_env_var_backend_override(self):
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama"}, clear=False):
            adapter = LLMAdapter()
            assert adapter.active_backend == "ollama"

    def test_explicit_model_overrides_default(self):
        adapter = LLMAdapter(backend="anthropic", model="claude-opus-4-7")
        assert adapter.active_model == "claude-opus-4-7"

    def test_custom_ollama_host(self):
        adapter = LLMAdapter(backend="ollama", ollama_host="http://192.168.1.100:11434")
        assert "192.168.1.100" in adapter.ollama_host


# ── Singleton behavior ───────────────────────────────────────────────────────

class TestSingleton:
    def test_get_adapter_returns_same_instance(self):
        reset_adapter()
        a1 = get_adapter()
        a2 = get_adapter()
        assert a1 is a2

    def test_reset_adapter_forces_new_instance(self):
        reset_adapter()
        a1 = get_adapter()
        reset_adapter()
        a2 = get_adapter()
        assert a1 is not a2

    def test_get_adapter_with_params_returns_new_instance(self):
        reset_adapter()
        a1 = get_adapter()
        a2 = get_adapter(backend="ollama")
        assert a1 is not a2

    def teardown_method(self):
        reset_adapter()


# ── JSON extraction ──────────────────────────────────────────────────────────

class TestJSONExtraction:
    def test_direct_json_object(self):
        raw = '{"key": "value", "score": 0.9}'
        result = _extract_json(raw)
        assert result["key"] == "value"
        assert result["score"] == 0.9

    def test_json_with_markdown_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = _extract_json(raw)
        assert result["key"] == "value"

    def test_json_with_code_fence_no_lang(self):
        raw = '```\n{"answer": 42}\n```'
        result = _extract_json(raw)
        assert result["answer"] == 42

    def test_json_embedded_in_prose(self):
        raw = 'Here is the result: {"status": "ok", "count": 5} — done.'
        result = _extract_json(raw)
        assert result["status"] == "ok"

    def test_nested_json(self):
        raw = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = _extract_json(raw)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("This is not JSON at all.")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _extract_json("")

    def test_json_with_extra_whitespace(self):
        raw = '   \n  {"key": "val"}  \n  '
        result = _extract_json(raw)
        assert result["key"] == "val"

    def test_large_json_object(self):
        data = {f"key_{i}": i for i in range(50)}
        raw = json.dumps(data)
        result = _extract_json(raw)
        assert len(result) == 50


# ── Anthropic complete() ─────────────────────────────────────────────────────

class TestAnthropicComplete:
    def test_complete_returns_text(self):
        adapter = LLMAdapter(backend="anthropic")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]

        with patch("anthropic.Anthropic") as mock_ant:
            mock_ant.return_value.messages.create.return_value = mock_response
            result = adapter.complete(
                system="You are a CFO analyst.",
                user="Summarize Q1 results.",
            )
        assert result == "Test response"

    def test_complete_passes_correct_args(self):
        adapter = LLMAdapter(backend="anthropic", model="claude-sonnet-4-6")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="OK")]

        with patch("anthropic.Anthropic") as mock_ant:
            mock_client = mock_ant.return_value
            mock_client.messages.create.return_value = mock_response
            adapter.complete("system msg", "user msg", max_tokens=500)

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == "claude-sonnet-4-6"
            assert call_kwargs["max_tokens"] == 500
            assert call_kwargs["system"] == "system msg"
            assert call_kwargs["messages"][0]["content"] == "user msg"

    def test_complete_json_appends_json_instruction(self):
        adapter = LLMAdapter(backend="anthropic")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"key": "val"}')]

        with patch("anthropic.Anthropic") as mock_ant:
            mock_client = mock_ant.return_value
            mock_client.messages.create.return_value = mock_response
            result = adapter.complete_json("system", "user", keys=["key"])

        assert result["key"] == "val"


# ── Ollama complete() ────────────────────────────────────────────────────────

class TestOllamaComplete:
    def test_complete_returns_text(self):
        adapter = LLMAdapter(backend="ollama")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "Ollama response"}}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp):
            result = adapter.complete("system", "user")
        assert result == "Ollama response"

    def test_ollama_connect_error_raises_runtime(self):
        import httpx
        adapter = LLMAdapter(backend="ollama")
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
                adapter.complete("system", "user")

    def test_ollama_json_mode_sets_format(self):
        adapter = LLMAdapter(backend="ollama")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": '{"a": 1}'}}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            adapter.complete_json("system", "user", keys=["a"])
            payload = mock_post.call_args[1]["json"]
            assert payload.get("format") == "json"

    def test_check_ollama_health_returns_bool(self):
        adapter = LLMAdapter(backend="ollama")
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.get", return_value=mock_resp):
            assert adapter.check_ollama_health() is True

    def test_check_ollama_health_offline(self):
        import httpx
        adapter = LLMAdapter(backend="ollama")
        with patch("httpx.get", side_effect=httpx.ConnectError("offline")):
            assert adapter.check_ollama_health() is False


# ── trim_for_local ───────────────────────────────────────────────────────────

class TestTrimForLocal:
    def test_short_text_unchanged(self):
        text = "Short text."
        assert trim_for_local(text, max_chars=100) == text

    def test_long_text_trimmed(self):
        text = "A" * 10_000
        result = trim_for_local(text, max_chars=3000)
        assert len(result) < 10_000
        assert "[...trimmed for local model...]" in result

    def test_trimmed_text_within_limit(self):
        text = "X" * 10_000
        result = trim_for_local(text, max_chars=3000)
        # Should be roughly 3000 chars + the marker
        assert len(result) <= 3_100

    def test_exact_limit_not_trimmed(self):
        text = "B" * 3000
        result = trim_for_local(text, max_chars=3000)
        assert result == text

    def test_keeps_start_and_end(self):
        text = "START" + "M" * 5000 + "END"
        result = trim_for_local(text, max_chars=100)
        assert "START" in result
        assert "END" in result


# ── status_line ──────────────────────────────────────────────────────────────

class TestStatusLine:
    def test_anthropic_status_line_format(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=False):
            adapter = LLMAdapter(backend="anthropic")
            line = adapter.status_line()
            assert "Anthropic" in line
            assert "key=SET" in line

    def test_anthropic_missing_key_status(self):
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            adapter = LLMAdapter(backend="anthropic")
            line = adapter.status_line()
            assert "key=MISSING" in line
        finally:
            if saved:
                os.environ["ANTHROPIC_API_KEY"] = saved

    def test_ollama_status_line_format(self):
        adapter = LLMAdapter(backend="ollama")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            line = adapter.status_line()
            assert "Ollama" in line
            assert "running" in line
