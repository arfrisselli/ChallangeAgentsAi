"""Unit tests for web_search: TavilyClient direct API, answer field, cleanup."""
import pytest
from unittest.mock import patch, MagicMock

from tools.web_search import _run_web_search, _clean_web_content, web_search
from tools.base import WebSearchResult


@patch("tools.web_search.os.getenv")
@patch("tools.web_search.TavilyClient")
def test_web_search_with_tavily_answer(mock_client_cls, mock_getenv):
    """When Tavily returns an answer, use it directly."""
    mock_getenv.return_value = "fake-key"
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "answer": "O prefeito de Londrina é Tiago Amaral, eleito para o mandato 2025-2028.",
        "results": [
            {"url": "https://a.com", "content": "raw content"},
            {"url": "https://b.com", "content": "more content"},
        ],
    }
    mock_client_cls.return_value = mock_client
    result = _run_web_search("prefeito de Londrina")
    assert result.answer is not None
    assert "Tiago Amaral" in result.answer
    assert "Tiago Amaral" in result.summary
    assert "https://a.com" in result.links


@patch("tools.web_search.os.getenv")
@patch("tools.web_search.TavilyClient")
def test_web_search_no_answer_uses_cleaned_content(mock_client_cls, mock_getenv):
    """When Tavily returns no answer, clean and use raw content."""
    mock_getenv.return_value = "fake-key"
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "answer": "",
        "results": [
            {"url": "https://a.com", "content": "Tiago Amaral é o prefeito de Londrina para o mandato 2025-2028."},
        ],
    }
    mock_client_cls.return_value = mock_client
    result = _run_web_search("prefeito")
    assert result.answer is None
    assert "Tiago Amaral" in result.summary


@patch("tools.web_search.os.getenv")
def test_web_search_not_configured(mock_getenv):
    mock_getenv.return_value = None
    result = _run_web_search("query")
    assert "not configured" in result.summary.lower() or "não configurad" in result.summary.lower()
    assert result.links == []


@patch("tools.web_search.os.getenv")
@patch("tools.web_search.TavilyClient")
def test_web_search_rate_limit(mock_client_cls, mock_getenv):
    mock_getenv.return_value = "fake-key"
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("Rate limit exceeded")
    mock_client_cls.return_value = mock_client
    result = _run_web_search("query")
    assert "limite" in result.summary.lower() or "rate" in result.summary.lower()


def test_web_search_tool_invoke():
    with patch("tools.web_search._run_web_search") as m:
        m.return_value = WebSearchResult(summary="Ok", links=["http://x.com"], answer="Ok")
        out = web_search.invoke({"query": "hello"})
        assert "Ok" in out
        assert "http://x.com" in out


class TestCleanWebContent:
    def test_removes_javascript_noise(self):
        text = "Ative o JavaScript no seu browser para que o relógio seja atualizado automaticamente."
        assert _clean_web_content(text) == ""

    def test_removes_app_store(self):
        text = "Download on the App Store para ter acesso ao conteúdo completo do aplicativo."
        assert _clean_web_content(text) == ""

    def test_removes_widget_dimensions(self):
        text = "Escolha o tamanho — menor - 125x125 pequeno - 150x150 médio - 175x175 grande - 200x200"
        assert _clean_web_content(text) == ""

    def test_removes_color_picker(self):
        text = "Escolha a cor — Branco Luz amarelo Amarelo Laranja Coral Rosa Vermelho Luz verde"
        assert _clean_web_content(text) == ""

    def test_keeps_useful_content(self):
        text = "Tiago Amaral, nascido em Londrina em 1987, é o prefeito da cidade para o mandato 2025-2028."
        clean = _clean_web_content(text)
        assert "Tiago Amaral" in clean
        assert "prefeito" in clean

    def test_removes_urls(self):
        text = "Visite nosso site https://example.com/page para mais informações sobre o projeto completo."
        clean = _clean_web_content(text)
        assert "https://" not in clean
