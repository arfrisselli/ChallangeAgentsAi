"""Unit test for web_search: mock API and verify formatted output (summary + links)."""
import pytest
from unittest.mock import patch, MagicMock

from tools.web_search import _run_web_search, web_search
from tools.base import WebSearchResult


@patch("tools.web_search._get_tavily_tool")
def test_web_search_returns_summary_and_links(mock_get_tool):
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = [
        {"content": "O prefeito de Londrina é Tiago Amaral, eleito em 2022 para o mandato.", "url": "https://a.com"},
        {"content": "Tiago Amaral tomou posse como prefeito em janeiro de 2025 na cidade de Londrina.", "url": "https://b.com"},
    ]
    mock_get_tool.return_value = mock_tool
    result = _run_web_search("test query")
    assert "prefeito" in result.summary.lower()
    assert "Tiago Amaral" in result.summary
    assert "https://a.com" in result.links
    assert "https://b.com" in result.links


@patch("tools.web_search._get_tavily_tool")
def test_web_search_not_configured(mock_get_tool):
    mock_get_tool.return_value = None
    result = _run_web_search("query")
    assert "not configured" in result.summary.lower()
    assert result.links == []


@patch("tools.web_search._get_tavily_tool")
def test_web_search_rate_limit(mock_get_tool):
    mock_tool = MagicMock()
    mock_tool.invoke.side_effect = Exception("Rate limit exceeded")
    mock_get_tool.return_value = mock_tool
    result = _run_web_search("query")
    assert "rate limit" in result.summary.lower() or "try again" in result.summary.lower()


def test_web_search_tool_invoke():
    with patch("tools.web_search._run_web_search") as m:
        m.return_value = WebSearchResult(summary="Ok", links=["http://x.com"])
        out = web_search.invoke({"query": "hello"})
        assert "Ok" in out
        assert "http://x.com" in out
