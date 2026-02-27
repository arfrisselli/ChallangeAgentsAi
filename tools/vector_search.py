"""
Vector search (RAG) tool: similarity search on Chroma. Exposed as search_docs.
Use when the user asks about internal docs, FAQs, or knowledge base content.
Returns list of relevant text chunks with metadata.
"""
import os
from typing import Any, Optional

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from tools.base import VectorSearchResult

# Collection name used by ingest and this tool
COLLECTION_NAME = "challenge_docs"


class SearchDocsInput(BaseModel):
    """Input for document similarity search."""
    query: str = Field(description="Natural language question or search query for the knowledge base")


def _get_embeddings():
    """OpenAI or Azure embeddings from env."""
    if os.getenv("AZURE_OPENAI_API_KEY"):
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
        )
    return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


def _get_chroma_client():
    """Chroma HTTP client from env. Import here to avoid loading chromadb at module load (Python 3.14 compat)."""
    import chromadb
    host = os.getenv("CHROMA_HOST", "chroma")
    port = os.getenv("CHROMA_PORT", "8000")
    return chromadb.HttpClient(host=host, port=int(port))


def search_docs_impl(
    query: str,
    *,
    top_k: int = 4,
    chroma_client: Optional[Any] = None,
    embeddings: Optional[Any] = None,
) -> list[VectorSearchResult]:
    """
    Run similarity search. Used by the LangChain tool; can be called with
    injected client/embeddings for tests.
    """
    client = chroma_client or _get_chroma_client()
    emb = embeddings or _get_embeddings()
    try:
        collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"description": "RAG docs"})
    except Exception:
        return []
    try:
        query_embedding = emb.embed_query(query)
    except Exception:
        return []
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas"])
    except Exception:
        return []
    out = []
    docs = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    for i, doc in enumerate(docs):
        meta = metadatas[i] if i < len(metadatas) else {}
        out.append(VectorSearchResult(content=doc or "", metadata=meta))
    return out


@tool(args_schema=SearchDocsInput)
def search_docs(query: str) -> str:
    """
    Search the internal knowledge base (documents/FAQs). Use when the user asks
    about product docs, policies, or pre-loaded content. Do not use for current
    web facts; use web_search for that.
    """
    chunks = search_docs_impl(query)
    if not chunks:
        return "No relevant documents found in the knowledge base."
    parts = [c.content for c in chunks if c.content]
    return "\n\n---\n\n".join(parts)


def get_search_docs_tool():
    return search_docs
