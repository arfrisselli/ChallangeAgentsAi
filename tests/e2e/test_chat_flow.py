"""
E2E: full flow question -> backend -> response. One case RAG/SQL, one case web/weather.
Uses FastAPI TestClient; mocks graph to avoid external services in CI.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@patch("app.main.get_graph")
def test_chat_returns_response(mock_get_graph, client):
    """E2E: POST /chat returns consolidated response (e.g. RAG or SQL path)."""
    mock_graph = MagicMock()
    # stream_mode="values" yields the state dict directly
    mock_graph.stream.return_value = [
        {"messages": [HumanMessage(content="How many products?"), AIMessage(content="There are 3 products.")]}
    ]
    mock_get_graph.return_value = mock_graph
    r = client.post("/chat", json={"message": "How many products are there?"})
    assert r.status_code == 200
    data = r.json()
    assert "response" in data
    assert "conversation_id" in data
    assert "3 products" in data["response"]


@patch("app.main.get_graph")
def test_chat_stream_returns_ndjson(mock_get_graph, client):
    """E2E: POST /chat/stream returns NDJSON with tokens (e.g. web or weather path)."""
    mock_graph = MagicMock()
    # stream_mode="values" yields the state dict directly
    mock_graph.stream.return_value = [
        {"messages": [HumanMessage(content="Weather in London"), AIMessage(content="In London: 15°C, cloudy.")]}
    ]
    mock_get_graph.return_value = mock_graph
    r = client.post("/chat/stream", json={"message": "What's the weather in London?"})
    assert r.status_code == 200
    lines = [ln for ln in r.text.strip().split("\n") if ln]
    assert len(lines) >= 1
    import json as _json
    first = _json.loads(lines[0])
    assert "content" in first


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
