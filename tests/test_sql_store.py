import sqlite3

import pytest

from app.data.sql_store import SQLStore, UnsafeQueryError, sanitize_identifier, validate_sql

from .conftest import SAMPLE


@pytest.fixture
def store():
    s = SQLStore(":memory:", row_limit=5)
    s.load_csv(SAMPLE / "tables" / "sales.csv")
    s.load_csv(SAMPLE / "tables" / "support_tickets.csv")
    yield s
    s.close()


def test_csv_types_are_inferred(store):
    schema = store.schema_description()
    assert "TABLE sales (36 rows)" in schema
    assert "units_sold INTEGER" in schema and "revenue_usd INTEGER" in schema
    assert "resolution_hours REAL" in schema


def test_aggregate_query(store):
    res = store.query("SELECT region, SUM(revenue_usd) AS rev FROM sales WHERE quarter='2026-Q3' "
                      "GROUP BY region ORDER BY rev DESC LIMIT 1")
    assert res.rows == [("North America", 3136300)]


def test_row_limit_truncates(store):
    res = store.query("SELECT * FROM support_tickets")
    assert len(res.rows) == 5 and res.truncated


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales",
    "DELETE FROM sales",
    "SELECT 1; DROP TABLE sales",
    "UPDATE sales SET units_sold = 0",
    "ATTACH DATABASE 'x.db' AS x",
    "PRAGMA table_info(sales)",
])
def test_static_guard_rejects_writes(sql):
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql)


def test_semicolon_inside_string_is_allowed():
    assert validate_sql("SELECT 'a;b' AS x;") == "SELECT 'a;b' AS x"


def test_authorizer_blocks_writes_that_slip_past_static_check(store):
    # A CTE that is syntactically a WITH but performs a write must still be denied.
    with pytest.raises((UnsafeQueryError, sqlite3.DatabaseError)):
        store.query("WITH x AS (SELECT 1) DELETE FROM sales")
    assert store.query("SELECT COUNT(*) FROM sales").rows == [(36,)]


def test_runaway_query_is_interrupted():
    s = SQLStore(":memory:", timeout_s=0.2)
    with pytest.raises(sqlite3.OperationalError):
        s.query("WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c) SELECT COUNT(*) FROM c")


def test_identifier_sanitizing():
    assert sanitize_identifier("Revenue (USD)") == "revenue_usd"
    assert sanitize_identifier("2026 sales") == "t_2026_sales"
