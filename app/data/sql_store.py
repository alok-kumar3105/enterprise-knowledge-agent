"""Structured-data store: CSV files loaded into SQLite, queried read-only.

LLM-written SQL is untrusted input, so queries run behind three guards:
1. a static check that allows exactly one SELECT/WITH statement,
2. a SQLite authorizer that denies every operation except reads, and
3. a progress handler that aborts long-running queries.
"""
from __future__ import annotations

import csv
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


class UnsafeQueryError(ValueError):
    pass


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool

    def to_text(self, max_rows: int = 20) -> str:
        if not self.columns:
            return "(no columns)"
        lines = [" | ".join(self.columns)]
        lines += [" | ".join("" if v is None else str(v) for v in row) for row in self.rows[:max_rows]]
        if self.truncated or len(self.rows) > max_rows:
            lines.append("... (truncated)")
        return "\n".join(lines)


_IDENT = re.compile(r"[^a-z0-9_]+")
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex|begin|commit|"
    r"rollback|savepoint|load_extension)\b",
    re.IGNORECASE,
)


def sanitize_identifier(name: str) -> str:
    ident = _IDENT.sub("_", name.strip().lower()).strip("_") or "col"
    return f"t_{ident}" if ident[0].isdigit() else ident


def _infer_type(values: list[str]) -> str:
    present = [v for v in values if v != ""]
    if not present:
        return "TEXT"
    try:
        for v in present:
            int(v)
        return "INTEGER"
    except ValueError:
        pass
    try:
        for v in present:
            float(v)
        return "REAL"
    except ValueError:
        return "TEXT"


def _convert(value: str, sqltype: str):
    if value == "":
        return None
    if sqltype == "INTEGER":
        return int(value)
    if sqltype == "REAL":
        return float(value)
    return value


def validate_sql(sql: str) -> str:
    """Static guard: exactly one read-only statement."""
    stmt = sql.strip().rstrip(";").strip()
    no_strings = re.sub(r"'(?:[^']|'')*'", "''", stmt)
    if ";" in no_strings:
        raise UnsafeQueryError("Only a single statement is allowed")
    if not re.match(r"^(select|with)\b", no_strings, re.IGNORECASE):
        raise UnsafeQueryError("Only SELECT queries are allowed")
    if _FORBIDDEN.search(no_strings):
        raise UnsafeQueryError("Query contains a forbidden keyword")
    return stmt


_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
# Recursive CTEs need this action code (present in Python >= 3.11 builds).
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _ALLOWED_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)


def _authorizer(action, arg1, arg2, dbname, source):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


class SQLStore:
    def __init__(self, path: Path | str = ":memory:", row_limit: int = 50, timeout_s: float = 5.0):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.row_limit = row_limit
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)

    # ---- ingestion (trusted path, no authorizer) --------------------------
    def load_csv(self, path: Path, table: str | None = None) -> tuple[str, int]:
        table = sanitize_identifier(table or Path(path).stem)
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                raise ValueError(f"{Path(path).name} is empty")
            rows = [r for r in reader if any(cell.strip() for cell in r)]
        cols, seen = [], set()
        for h in header:
            c = sanitize_identifier(h)
            while c in seen:
                c += "_"
            seen.add(c)
            cols.append(c)
        width = len(cols)
        rows = [(r + [""] * width)[:width] for r in rows]
        types = [_infer_type([r[i].strip() for r in rows]) for i in range(width)]
        data = [tuple(_convert(r[i].strip(), types[i]) for i in range(width)) for r in rows]
        col_defs = ", ".join(f'"{c}" {t}' for c, t in zip(cols, types, strict=True))
        with self._lock, self._conn:
            self._conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            self._conn.execute(f'CREATE TABLE "{table}" ({col_defs})')
            self._conn.executemany(f'INSERT INTO "{table}" VALUES ({", ".join("?" * width)})', data)
        return table, len(data)

    # ---- introspection ----------------------------------------------------
    def tables(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]

    def schema_description(self, sample_rows: int = 3) -> str:
        """Compact schema + sample rows, formatted for an LLM prompt."""
        parts = []
        with self._lock:
            for (table,) in self._conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
                cols = self._conn.execute(f'PRAGMA table_info("{table}")').fetchall()
                count = self._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                sample = self._conn.execute(f'SELECT * FROM "{table}" LIMIT {sample_rows}').fetchall()
                col_text = ", ".join(f"{c[1]} {c[2]}" for c in cols)
                parts.append(f"TABLE {table} ({count} rows): {col_text}")
                for row in sample:
                    parts.append("  sample: " + ", ".join(repr(v) for v in row))
        return "\n".join(parts)

    # ---- untrusted queries ------------------------------------------------
    def query(self, sql: str) -> QueryResult:
        stmt = validate_sql(sql)
        deadline = time.monotonic() + self.timeout_s
        with self._lock:
            self._conn.set_authorizer(_authorizer)
            self._conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            try:
                cur = self._conn.execute(stmt)
                rows = cur.fetchmany(self.row_limit + 1)
                columns = [d[0] for d in cur.description or []]
            except sqlite3.DatabaseError as exc:
                if "not authorized" in str(exc):
                    raise UnsafeQueryError("Query attempted a non-read operation") from exc
                raise
            finally:
                self._conn.set_authorizer(None)
                self._conn.set_progress_handler(None, 0)
        return QueryResult(columns=columns, rows=rows[: self.row_limit], truncated=len(rows) > self.row_limit)

    def drop_all(self) -> None:
        with self._lock, self._conn:
            names = [r[0] for r in self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for name in names:
                self._conn.execute(f'DROP TABLE IF EXISTS "{name}"')

    def close(self) -> None:
        self._conn.close()
