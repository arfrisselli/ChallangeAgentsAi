"""Unit test for vector_search: in-memory or mock Chroma; verify returned documents."""
import pytest
from unittest.mock import patch, MagicMock

from tools.vector_search import search_docs_impl


def test_vector_search_returns_chunks():
    mock_client = MagicMock()
    mock_coll = MagicMock()
    mock_coll.query.return_value = {
        "documents": [["doc1 text", "doc2 text"]],
        "metadatas": [[{"source": "a"}, {"source": "b"}]],
    }
    mock_client.get_or_create_collection.return_value = mock_coll
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [0.1] * 1536
    results = search_docs_impl("query", chroma_client=mock_client, embeddings=mock_emb)
    assert len(results) == 2
    assert results[0].content == "doc1 text"
    assert results[1].metadata["source"] == "b"


def test_vector_search_empty():
    mock_client = MagicMock()
    mock_coll = MagicMock()
    mock_coll.query.return_value = {"documents": [[]], "metadatas": [[]]}
    mock_client.get_or_create_collection.return_value = mock_coll
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [0.1] * 1536
    results = search_docs_impl("q", chroma_client=mock_client, embeddings=mock_emb)
    assert len(results) == 0
