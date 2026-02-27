"""
SQL tool: parametrized read-only queries against PostgreSQL. Uses sqlglot to reject
DDL and dangerous DML (UPDATE/DELETE without WHERE). Never concatenate user/LLM text into SQL.
"""
import logging
from typing import Any, Optional

import sqlglot
from sqlglot.expressions import Select
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

FORBIDDEN_KEYWORDS = (
    "DROP", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE", "DELETE",
    "GRANT", "REVOKE", "EXECUTE", "EXEC", "--", "/*", "*/",
)


class SQLQueryInput(BaseModel):
    """Input for SQL query. Query must use placeholders; params are passed separately for binding."""
    query: str = Field(description="SQL SELECT query with placeholders like %(name)s or $1")
    params: Optional[dict[str, Any]] = Field(default=None, description="Parameters for the query (for parameterization)")


def _validate_sql(query: str) -> tuple[bool, str]:
    """
    Validate SQL: only allow SELECT. Reject DDL, DML, and UPDATE/DELETE without safe WHERE.
    Uses sqlglot for parsing. Returns (allowed, error_message).
    """
    query_clean = query.strip()
    upper = query_clean.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if kw in upper:
            return False, f"Query contains forbidden keyword or pattern: {kw}"
    try:
        parsed = sqlglot.parse_one(query_clean, dialect="postgres")
    except Exception as e:
        return False, f"Invalid SQL: {e}"
    if not isinstance(parsed, Select):
        return False, "Only SELECT queries are allowed."
    return True, ""


def run_sql_impl(
    query: str,
    db: SQLDatabase,
    params: Optional[dict[str, Any]] = None,
) -> str:
    """
    Execute a validated SELECT query with parameters. Used by the tool and tests.
    """
    ok, err = _validate_sql(query)
    if not ok:
        logger.warning("SQL validation rejected query: %s", err)
        return f"Query not allowed: {err}"
    try:
        with db._engine.connect() as conn:
            # Use text() with bindparams for safe parameterization
            from sqlalchemy import text
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            rows = result.fetchall()
            if not rows:
                return "No rows returned."
            columns = list(result.keys())
            lines = [" | ".join(str(c) for c in columns)]
            for row in rows:
                # Access by index for compatibility (Row and tuple)
                parts = [str(row[i]) for i in range(len(columns))]
                lines.append(" | ".join(parts))
            return "\n".join(lines)
    except Exception as e:
        logger.warning("SQL execution error: %s", str(e))
        return f"Execution error: {e}"


def get_sql_db_tool(connection_string: str):
    """Build the LangChain SQL tool bound to the given DB. Validation is done inside the tool."""
    db = SQLDatabase.from_uri(connection_string)

    @tool(args_schema=SQLQueryInput)
    def sql_db(query: str, params: Optional[dict[str, Any]] = None) -> str:
        """
        Run a read-only SELECT query on the database. Use for questions about data in tables.
        Only SELECT is allowed. Pass parameters in params to avoid injection.
        """
        return run_sql_impl(query, db, params)

    return sql_db
