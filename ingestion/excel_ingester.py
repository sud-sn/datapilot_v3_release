"""
DataPilot – Excel / CSV Ingester
Reads an uploaded Excel or CSV file and converts it into the same
SchemaContext that the DB adapters produce. This lets the interview
engine and SQL pipeline work identically regardless of data source.

Supported formats:
  .xlsx / .xls   — each sheet becomes one TableInfo
  .csv           — single table using the filename as table name
"""
from __future__ import annotations
import io
import logging
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from adapters.base import (
    ColumnInfo, SchemaContext, SQLDialect, TableInfo,
)

logger = logging.getLogger(__name__)

# How many rows to read for type inference and sample values
_SAMPLE_ROWS   = 100
_SAMPLE_VALUES = 8


# ── Type inference ────────────────────────────────────────────────────────────

def _infer_type(series: pd.Series) -> tuple[str, str]:
    """
    Returns (normalised_type, raw_type_label) from a pandas Series.
    normalised_type matches the ColumnInfo.data_type conventions.
    """
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean", "boolean"

    if pd.api.types.is_integer_dtype(dtype):
        return "numeric", "integer"

    if pd.api.types.is_float_dtype(dtype):
        return "numeric", "float"

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "timestamp", "datetime"

    # For object columns try to detect dates and numbers stored as strings
    if pd.api.types.is_object_dtype(dtype):
        non_null = series.dropna()
        if len(non_null) == 0:
            return "text", "text"

        # Try numeric
        try:
            pd.to_numeric(non_null.head(20))
            return "numeric", "numeric"
        except (ValueError, TypeError):
            pass

        # Try datetime
        try:
            pd.to_datetime(non_null.head(20), infer_datetime_format=True)
            return "timestamp", "datetime"
        except (ValueError, TypeError):
            pass

        return "text", "text"

    return "text", str(dtype)


def _sample_values(series: pd.Series, limit: int = _SAMPLE_VALUES) -> list[Any]:
    """Return up to `limit` distinct non-null values from the series."""
    try:
        vals = series.dropna().unique()[:limit]
        # Convert numpy types to Python natives for JSON serialisation
        return [v.item() if hasattr(v, "item") else v for v in vals]
    except Exception:
        return []


def _sanitise_name(name: str) -> str:
    """Turn a sheet/column name into a safe identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(name)).strip("_") or "unnamed"


# ── Sheet → TableInfo ─────────────────────────────────────────────────────────

def _sheet_to_table(
    df: pd.DataFrame,
    table_name: str,
    schema: str = "upload",
) -> TableInfo:
    """Convert one DataFrame (sheet) into a TableInfo object."""
    columns: list[ColumnInfo] = []

    for raw_col in df.columns:
        col_name = _sanitise_name(str(raw_col))
        series   = df[raw_col]
        norm_type, raw_type = _infer_type(series)
        samples  = _sample_values(series)
        nullable = bool(series.isna().any())

        columns.append(
            ColumnInfo(
                name          = col_name,
                data_type     = norm_type,
                raw_type      = raw_type,
                nullable      = nullable,
                is_primary_key= False,
                is_foreign_key= False,
                sample_values = samples,
            )
        )

    return TableInfo(
        schema            = schema,
        name              = _sanitise_name(table_name),
        columns           = columns,
        row_count_estimate= len(df),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_file(
    file_bytes: bytes,
    filename: str,
    schema_name: str = "upload",
) -> tuple[SchemaContext, str]:
    """
    Parse an Excel or CSV file and return:
      - SchemaContext   usable by the pipeline and interview engine
      - source_id       stable identifier for the KB store

    Raises ValueError on unsupported file types or parsing errors.
    """
    suffix = Path(filename).suffix.lower()
    stem   = _sanitise_name(Path(filename).stem)
    source_id = f"upload_{stem}"

    buf = io.BytesIO(file_bytes)

    if suffix == ".csv":
        try:
            df = pd.read_csv(buf, nrows=_SAMPLE_ROWS)
        except Exception as exc:
            raise ValueError(f"Could not parse CSV: {exc}") from exc

        tables = [_sheet_to_table(df, stem, schema=schema_name)]

    elif suffix in (".xlsx", ".xls"):
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        try:
            xl = pd.ExcelFile(buf, engine=engine)
        except Exception as exc:
            raise ValueError(f"Could not open Excel file: {exc}") from exc

        tables: list[TableInfo] = []
        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name, nrows=_SAMPLE_ROWS)
                if df.empty or len(df.columns) == 0:
                    logger.debug("Skipping empty sheet: %s", sheet_name)
                    continue
                tables.append(_sheet_to_table(df, sheet_name, schema=schema_name))
            except Exception as exc:
                logger.warning("Could not parse sheet '%s': %s", sheet_name, exc)

        if not tables:
            raise ValueError("No readable sheets found in the Excel file.")

    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Please upload a .csv, .xlsx, or .xls file."
        )

    ctx = SchemaContext(
        dialect        = SQLDialect.POSTGRESQL,   # default; overridden if DB is connected
        database       = stem,
        default_schema = schema_name,
        tables         = tables,
        relationships  = {},                       # FKs not inferable from flat files
    )

    logger.info(
        "Ingested file '%s': %d table(s), %d total columns",
        filename,
        len(tables),
        sum(len(t.columns) for t in tables),
    )

    return ctx, source_id


def schema_to_prompt_text(schema: SchemaContext) -> str:
    """
    Serialise a SchemaContext into a compact text block suitable for
    injection into an LLM prompt (used by the markdown generator).
    """
    lines: list[str] = []
    for table in schema.tables:
        est = f" ({table.row_count_estimate} rows)" if table.row_count_estimate else ""
        lines.append(f"\nTABLE: {table.name}{est}")
        lines.append(f"{'Column':<30} {'Type':<12} {'Sample values'}")
        lines.append("-" * 70)
        for col in table.columns:
            samples = ", ".join(repr(v) for v in col.sample_values[:5]) if col.sample_values else "—"
            null_flag = " (nullable)" if col.nullable else ""
            lines.append(f"  {col.name:<28} {col.raw_type:<12} {samples}{null_flag}")
    return "\n".join(lines)
