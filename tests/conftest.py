"""Pytest config: PYTHONPATH and env for tests."""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_DB", "challenge_db")
os.environ.setdefault("CHROMA_HOST", "localhost")
