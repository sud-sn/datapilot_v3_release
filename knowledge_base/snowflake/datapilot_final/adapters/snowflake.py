"""
DataPilot – Snowflake Adapter
Uses snowflake-connector-python (sync) wrapped in asyncio thread executor.
Connects with username/password authentication.
All sessions are set to READ ONLY via session parameter.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Optional

import snowflake.connector
from snowflake.connector import DictCursor

from adapters.base import (
    DatabaseAdapter, SQLDialect, SchemaContext,
    TableInfo, ColumnInfo, QueryResult,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)

# Map Snowflake data type names → normalised types
_SF_TYPE_MAP: dict[str, str] = {
    # Text
    "text": "text", "varchar": "text", "string": "text", "char": "text",
    "character": "text", "nchar": "text", "nvarchar": "text",
    "nvarchar2": "text", "character varying": "text", "variant": "json",
    # Numeric
    "number": "numeric", "decimal": "numeric", "numeric": "numeric",
    "int": "numeric", "integer": "numeric", "bigint": "numeric",
    "smallint": "numeric", "tinyint": "numeric", "byteint": "numeric",
    "float": "numeric", "float4": "numeric", "float8": "numeric",
    "double": "numeric", "double precision": "numeric", "real": "numeric",
    "fixed": "numeric",
    # Timestamp / date
    "date": "date", "time": "time",
    "timestamp": "timestamp", "timestamp_ntz": "timestamp",
    "timestamp_ltz": "timestamp", "timestamp_tz": "timestamp",
    "datetime": "timestamp",
    # Boolean
    "boolean": "boolean",
    # Semi-structured
    "variant": "json", "object": "json", "array": "array",
    # Binary
    "binary": "binary", "varbinary": "binary",
}


def _normalise_sf_type(raw: str) -> str:
    return _SF_TYPE_MAP.get(raw.lower(), raw.lower())


class SnowflakeAdapter(DatabaseAdapter):
    """
    Async-compatible Snowflake adapter.
    All blocking calls run in asyncio thread executor.
    One connection is created per request (Snowflake connector is not
    pool-friendly for async usage; connection creation is fast).
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._conn: Optional[snowflake.connector.SnowflakeConnection] = None

    @property
    def dialect(self) -> SQLDialect:
        return SQLDialect.SNOWFLAKE

    @property
    def source_name(self) -> str:
        s = self._settings
        return (
            f"Snowflake — {s.sf_account}/"
            f"{s.sf_database}/{s.sf_schema}"
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Open the initial connection to verify credentials work.
        Subsequent calls reuse the connection or create a new one if closed.
        """
        await self._run_sync(self._connect_sync)
        logger.info("Snowflake connected: %s", self.source_name)

    def _connect_sync(self) -> None:
        s = self._settings
        self._conn = snowflake.connector.connect(
            account   = s.sf_account,
            user      = s.sf_user,
            password  = s.sf_password,
            database  = s.sf_database,
            schema    = s.sf_schema,
            warehouse = s.sf_warehouse,
            role      = s.sf_role or None,
            # Enforce read-only at session level
            session_parameters={
                "TRANSACTION_DEFAULT_ISOLATION_LEVEL": "READ COMMITTED",
                "AUTOCOMMIT": "TRUE",
            },
        )

    async def disconnect(self) -> None:
        if self._conn:
            await self._run_sync(self._conn.close)
            logger.info("Snowflake connection closed")

    async def health_check(self) -> bool:
        try:
            await self._run_sync(self._health_sync)
            return True
        except Exception as exc:
            logger.warning("Snowflake health check failed: %s", exc)
            return False

    def _health_sync(self) -> None:
        conn = self._get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_conn(self) -> snowflake.connector.SnowflakeConnection:
        """Return existing connection or reconnect if closed."""
        if self._conn is None or self._conn.is_closed():
            self._connect_sync()
        return self._conn

    async def _run_sync(self, fn, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # ── Schema ─────────────────────────────────────────────────────────────

    async def fetch_schema(self, schema_name: Optional[str] = None) -> SchemaContext:
        s      = self._settings
        schema = schema_name or s.sf_schema
        db     = s.sf_database

        tables   = await self._run_sync(self._fetch_tables_sync, db, schema)
        fk_graph = await self._run_sync(self._fetch_fk_graph_sync, db, schema)

        ctx = SchemaContext(
            dialect       = self.dialect,
            database      = db,
            default_schema= schema,
            tables        = tables,
            relationships = fk_graph,
        )
        logger.info(
            "Snowflake schema fetched: %d tables in '%s'.'%s'",
            len(tables), db, schema,
        )
        return ctx

    def _fetch_tables_sync(self, db: str, schema: str) -> list[TableInfo]:
        conn   = self._get_conn()
        cursor = conn.cursor(DictCursor)
        cursor.execute(
            f"""
            SELECT TABLE_NAME, ROW_COUNT
            FROM {db}.INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = %s
              AND TABLE_TYPE   = 'BASE TABLE'
            ORDER BY TABLE_NAME
            """,
            (schema.upper(),),
        )
        tables: list[TableInfo] = []
        for row in cursor.fetchall():
            tname   = row["TABLE_NAME"]
            row_est = row.get("ROW_COUNT")
            cols    = self._fetch_columns_sync(db, schema, tname, conn)
            fks     = self._fetch_table_fks_sync(db, schema, tname, conn)
            tables.append(TableInfo(
                schema            = schema,
                name              = tname,
                columns           = cols,
                row_count_estimate= int(row_est) if row_est else None,
                foreign_keys      = fks,
            ))
        cursor.close()
        return tables

    def _fetch_columns_sync(
        self, db: str, schema: str, table: str,
        conn: snowflake.connector.SnowflakeConnection,
    ) -> list[ColumnInfo]:
        s      = self._settings
        cursor = conn.cursor(DictCursor)
        cursor.execute(
            f"""
            SELECT
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                COMMENT
            FROM {db}.INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME   = %s
            ORDER BY ORDINAL_POSITION
            """,
            (schema.upper(), table),
        )
        rows   = cursor.fetchall()
        cursor.close()

        # Get PK columns
        pk_cols = self._fetch_pk_cols_sync(db, schema, table, conn)

        # Get FK columns
        fk_map  = self._fetch_fk_col_map_sync(db, schema, table, conn)

        columns: list[ColumnInfo] = []
        for row in rows:
            col_name = row["COLUMN_NAME"]
            raw_type = row["DATA_TYPE"]

            sample_vals: list[Any] = []
            if s.include_sample_values:
                sample_vals = self._fetch_sample_values_sync(
                    db, schema, table, col_name, s.sample_value_limit, conn
                )

            columns.append(ColumnInfo(
                name          = col_name,
                data_type     = _normalise_sf_type(raw_type),
                raw_type      = raw_type,
                nullable      = row["IS_NULLABLE"] == "YES",
                is_primary_key= col_name in pk_cols,
                is_foreign_key= col_name in fk_map,
                references    = fk_map.get(col_name),
                sample_values = sample_vals,
                description   = row.get("COMMENT"),
            ))
        return columns

    def _fetch_pk_cols_sync(
        self, db: str, schema: str, table: str,
        conn: snowflake.connector.SnowflakeConnection,
    ) -> set[str]:
        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(
                f"""
                SELECT COLUMN_NAME
                FROM {db}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN {db}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                  ON tc.CONSTRAINT_NAME  = ccu.CONSTRAINT_NAME
                 AND tc.TABLE_SCHEMA     = ccu.TABLE_SCHEMA
                WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                  AND tc.TABLE_SCHEMA    = %s
                  AND tc.TABLE_NAME      = %s
                """,
                (schema.upper(), table),
            )
            return {r["COLUMN_NAME"] for r in cursor.fetchall()}
        except Exception:
            return set()
        finally:
            cursor.close()

    def _fetch_fk_col_map_sync(
        self, db: str, schema: str, table: str,
        conn: snowflake.connector.SnowflakeConnection,
    ) -> dict[str, str]:
        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(
                f"""
                SELECT
                    kcu.COLUMN_NAME,
                    ccu.TABLE_NAME  AS REF_TABLE,
                    ccu.COLUMN_NAME AS REF_COLUMN
                FROM {db}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN {db}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                  ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                 AND tc.TABLE_SCHEMA    = kcu.TABLE_SCHEMA
                JOIN {db}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                  ON ccu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
                 AND ccu.TABLE_SCHEMA    = tc.TABLE_SCHEMA
                WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                  AND tc.TABLE_SCHEMA    = %s
                  AND tc.TABLE_NAME      = %s
                """,
                (schema.upper(), table),
            )
            return {
                r["COLUMN_NAME"]: f"{r['REF_TABLE']}.{r['REF_COLUMN']}"
                for r in cursor.fetchall()
            }
        except Exception:
            return {}
        finally:
            cursor.close()

    def _fetch_sample_values_sync(
        self, db: str, schema: str, table: str,
        column: str, limit: int,
        conn: snowflake.connector.SnowflakeConnection,
    ) -> list[Any]:
        cursor = conn.cursor()
        try:
            cursor.execute(
                f"""
                SELECT DISTINCT "{column}"
                FROM "{db}"."{schema}"."{table}"
                WHERE "{column}" IS NOT NULL
                LIMIT {limit}
                """
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []
        finally:
            cursor.close()

    def _fetch_table_fks_sync(
        self, db: str, schema: str, table: str,
        conn: snowflake.connector.SnowflakeConnection,
    ) -> list[dict]:
        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(
                f"""
                SELECT
                    kcu.COLUMN_NAME,
                    ccu.TABLE_NAME  AS REF_TABLE,
                    ccu.COLUMN_NAME AS REF_COLUMN
                FROM {db}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN {db}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                  ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                 AND tc.TABLE_SCHEMA    = kcu.TABLE_SCHEMA
                JOIN {db}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                  ON ccu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
                 AND ccu.TABLE_SCHEMA    = tc.TABLE_SCHEMA
                WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                  AND tc.TABLE_SCHEMA    = %s
                  AND tc.TABLE_NAME      = %s
                """,
                (schema.upper(), table),
            )
            return [
                {"column": r["COLUMN_NAME"], "ref_table": r["REF_TABLE"], "ref_column": r["REF_COLUMN"]}
                for r in cursor.fetchall()
            ]
        except Exception:
            return []
        finally:
            cursor.close()

    def _fetch_fk_graph_sync(
        self, db: str, schema: str
    ) -> dict[str, list[tuple[str, str, str]]]:
        conn   = self._get_conn()
        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(
                f"""
                SELECT
                    kcu.TABLE_NAME,
                    kcu.COLUMN_NAME,
                    ccu.TABLE_NAME  AS REF_TABLE,
                    ccu.COLUMN_NAME AS REF_COLUMN
                FROM {db}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN {db}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                  ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                 AND tc.TABLE_SCHEMA    = kcu.TABLE_SCHEMA
                JOIN {db}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                  ON ccu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
                 AND ccu.TABLE_SCHEMA    = tc.TABLE_SCHEMA
                WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                  AND tc.TABLE_SCHEMA    = %s
                """,
                (schema.upper(),),
            )
            graph: dict[str, list[tuple[str, str, str]]] = {}
            for r in cursor.fetchall():
                graph.setdefault(r["TABLE_NAME"], []).append(
                    (r["COLUMN_NAME"], r["REF_TABLE"], r["REF_COLUMN"])
                )
            return graph
        except Exception:
            return {}
        finally:
            cursor.close()

    # ── Execution ──────────────────────────────────────────────────────────

    async def execute(self, sql: str, timeout: Optional[int] = None) -> QueryResult:
        s       = self._settings
        timeout = timeout or s.db_query_timeout
        start   = time.monotonic()

        try:
            cols, rows = await self._run_sync(self._exec_sync, sql, timeout)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            truncated  = len(rows) >= s.max_result_rows
            return QueryResult(
                sql        = sql,
                columns    = cols,
                rows       = rows,
                row_count  = len(rows),
                execution_ms = elapsed_ms,
                truncated  = truncated,
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error("Snowflake execution error: %s", exc)
            return QueryResult(
                sql=sql, columns=[], rows=[], row_count=0,
                execution_ms=elapsed_ms, error=str(exc),
            )

    def _exec_sync(
        self, sql: str, timeout: int
    ) -> tuple[list[str], list[dict]]:
        conn   = self._get_conn()
        cursor = conn.cursor(DictCursor)
        cursor.execute_async(sql) if timeout else cursor.execute(sql)
        cursor.execute(sql)
        cols  = [desc[0] for desc in cursor.description] if cursor.description else []
        rows  = []
        limit = self._settings.max_result_rows
        for row in cursor:
            rows.append(dict(row))
            if len(rows) >= limit:
                break
        cursor.close()
        return cols, rows

    async def explain(self, sql: str) -> str:
        try:
            def _explain():
                conn   = self._get_conn()
                cursor = conn.cursor()
                cursor.execute(f"EXPLAIN {sql}")
                rows = cursor.fetchall()
                cursor.close()
                return "\n".join(str(r[0]) for r in rows)
            return await self._run_sync(_explain)
        except Exception as exc:
            raise ValueError(f"EXPLAIN failed: {exc}") from exc
