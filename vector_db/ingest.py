"""
Ingest documents into Chroma: load from data/ (or given path), generate embeddings
(OpenAI or Azure from env), and persist to Chroma. Run manually or from entrypoint.
"""
import os
import sys
from pathlib import Path

# Add project root so imports work when run as python -m vector_db.ingest
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv(Path(_project_root) / ".env")

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLLECTION_NAME = "challenge_docs"


def _get_embeddings():
    if os.getenv("AZURE_OPENAI_API_KEY"):
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
        )
    return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


def _get_chroma_client():
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    return chromadb.HttpClient(host=host, port=port)


def load_documents(data_dir: str) -> list:
    """Load and split documents from data_dir."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    loaders = []
    for ext in ["*.txt", "*.md"]:
        for f in data_path.rglob(ext):
            try:
                loaders.append(TextLoader(str(f), encoding="utf-8", autodetect_encoding=True))
            except Exception:
                pass
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception:
            continue
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def run_ingest(data_dir: str | None = None) -> int:
    """Load docs, embed, add to Chroma. Returns number of chunks added."""
    data_dir = data_dir or os.getenv("DATA_DIR") or str(Path(__file__).resolve().parent.parent / "data")
    docs = load_documents(data_dir)
    if not docs:
        return 0
    embeddings = _get_embeddings()
    client = _get_chroma_client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"description": "RAG docs"})
    texts = [d.page_content for d in docs]
    metadatas = [{"source": d.metadata.get("source", "")} for d in docs]
    ids = [f"doc_{i}" for i in range(len(docs))]
    try:
        emb_list = embeddings.embed_documents(texts)
    except Exception as e:
        print(f"Embedding failed: {e}", file=sys.stderr)
        return 0
    try:
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=emb_list)
    except Exception as e:
        print(f"Chroma add failed: {e}", file=sys.stderr)
        return 0
    return len(docs)


if __name__ == "__main__":
    n = run_ingest()
    print(f"Ingested {n} chunks.")
