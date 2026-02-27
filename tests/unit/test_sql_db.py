"""Unit tests for sql_db: whitelist allowed SELECT; reject forbidden queries."""
import pytest
from unittest.mock import MagicMock
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from tools.sql_db import _validate_sql, run_sql_impl
from langchain_community.utilities.sql_database import SQLDatabase


def test_validate_sql_allows_select():
    ok, err = _validate_sql("SELECT 1")
    assert ok is True
    ok, err = _validate_sql("SELECT * FROM users WHERE id = 1")
    assert ok is True


def test_validate_sql_rejects_drop():
    ok, err = _validate_sql("DROP TABLE users")
    assert ok is False
    assert "SELECT" in err or "forbidden" in err.lower()


def test_validate_sql_rejects_delete():
    ok, err = _validate_sql("DELETE FROM users")
    assert ok is False


def test_validate_sql_rejects_update():
    ok, err = _validate_sql("UPDATE users SET x = 1")
    assert ok is False


def test_validate_sql_rejects_truncate():
    ok, err = _validate_sql("TRUNCATE TABLE users")
    assert ok is False


def test_sql_impl_rejects_forbidden():
    engine = create_engine("sqlite:///:memory:")
    db = SQLDatabase(engine)
    out = run_sql_impl("DROP TABLE x", db)
    assert "not allowed" in out or "forbidden" in out.lower()


def test_sql_impl_returns_results(tmp_path):
    path = str(tmp_path / "test.db")
    engine = create_engine(f"sqlite:///{path}")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE t (a INT)"))
        conn.execute(text("INSERT INTO t VALUES (1)"))
        conn.commit()
    db = SQLDatabase(engine)
    out = run_sql_impl("SELECT a FROM t", db)
    assert "1" in out
    assert "a" in out
