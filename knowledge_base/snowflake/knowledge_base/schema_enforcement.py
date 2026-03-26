"""
DataPilot - Schema Enforcement Utilities

Shared utilities that build the EXACT COLUMN REGISTRY and schema context string
injected into every SQL generation prompt.

The EXACT COLUMN REGISTRY prevents the LLM from hallucinating column names from
training memory. Column notes (user-defined from the KB markdown) are injected
inline so user semantics like "Use this as the doctor name" become authoritative.

All three dialect files (PostgreSQL, Azure SQL, Snowflake) import from here.
"""
from __future__ import annotations


def build_exact_column_registry(
    schema_context,
    column_notes=None,
):
    """
    Build the authoritative column registry block injected at the top of
    every SQL generation prompt.

    column_notes: optional dict {table_name: {column_name: note}}
                  parsed from the KB markdown. User-defined semantics
                  (including resolved NEEDS CONTEXT flags) are injected
                  inline so the LLM treats them as authoritative rules.
    """
    notes = column_notes or {}
    lines = []

    lines.append("=" * 55)
    lines.append("EXACT COLUMN REGISTRY - THE ONLY VALID TABLE AND COLUMN NAMES")
    lines.append("You MUST use these exact names. Any name not listed here does NOT exist.")
    lines.append("=" * 55)
    lines.append("")

    for table in schema_context.tables:
        col_names = [c.name for c in table.columns]
        lines.append(f"TABLE {table.name}:")
        lines.append(f"  COLUMNS: {', '.join(col_names)}")

        table_notes = notes.get(table.name, {})
        if table_notes:
            lines.append("  COLUMN NOTES (follow these exactly when using these columns):")
            for col_name, note in table_notes.items():
                if col_name in col_names:  # Only inject notes for real columns
                    lines.append(f"    {col_name}: {note}")
        lines.append("")

    lines.append("=" * 55)
    lines.append("SCHEMA DETAILS (types, sample values, foreign keys)")
    lines.append("=" * 55)
    lines.append("")

    return "\n".join(lines)


def build_schema_detail_block(schema_context, dialect_name):
    """
    Build the detailed schema block with types, sample values, and FK relationships.
    """
    lines = [
        f"DATABASE: {schema_context.database}",
        f"SCHEMA:   {schema_context.default_schema}",
        f"DIALECT:  {dialect_name}",
        "",
    ]

    for table in schema_context.tables:
        est = f" (~{table.row_count_estimate:,} rows)" if table.row_count_estimate else ""
        lines.append(f"TABLE: {table.schema}.{table.name}{est}")
        for col in table.columns:
            flags = []
            if col.is_primary_key: flags.append("PK")
            if col.is_foreign_key: flags.append(f"FK->{col.references}")
            if not col.nullable:   flags.append("NOT NULL")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            sample = ""
            if col.sample_values:
                vals   = [repr(v) for v in col.sample_values[:6]]
                sample = f"  e.g. {', '.join(vals)}"
            desc = f"  // {col.description}" if col.description else ""
            lines.append(f"  {col.name}  {col.raw_type}{flag_str}{sample}{desc}")

        rels = schema_context.relationships.get(table.name, [])
        if rels:
            lines.append("  JOINS:")
            for fk_col, ref_tbl, ref_col in rels:
                lines.append(f"    {table.name}.{fk_col} -> {ref_tbl}.{ref_col}")
        lines.append("")

    return "\n".join(lines)


def build_full_schema_context(schema_context, dialect_name, column_notes=None):
    """
    Combines the column registry (with optional user notes) and schema detail
    into the full schema context string passed to build_system_prompt.
    """
    registry = build_exact_column_registry(schema_context, column_notes)
    detail   = build_schema_detail_block(schema_context, dialect_name)
    return registry + detail


SQL_GENERATION_RULES = (
    "ABSOLUTE RULES - violating any of these makes your answer wrong:\n"
    "1. Write ONLY a SELECT statement. Never write INSERT, UPDATE, DELETE, DROP, CREATE, ALTER.\n"
    "2. Use ONLY table names listed in the EXACT COLUMN REGISTRY above.\n"
    "3. Use ONLY column names listed under the relevant table in the EXACT COLUMN REGISTRY above.\n"
    "4. NEVER use column names from your training memory or from common naming conventions.\n"
    "   Example: If the registry shows FULL_NAME - use FULL_NAME.\n"
    "            If the registry does NOT show FIRST_NAME - never write FIRST_NAME.\n"
    "            If the registry does NOT show PRESCRIBER_NAME - never write PRESCRIBER_NAME.\n"
    "5. Before writing any column name, verify it appears in the EXACT COLUMN REGISTRY.\n"
    "6. If a column has a COLUMN NOTE in the registry, follow that note exactly.\n"
    "   Example note: 'FULL_NAME: Use this as the prescriber display name'\n"
    "   This means: whenever a query asks for prescriber name, use FULL_NAME.\n"
    "7. Always qualify every column with its table alias when using JOINs.\n"
    "8. Return ONLY the SQL statement - no explanation, no markdown, no ```, no comments.\n"
    "9. End the statement with a semicolon.\n"
    "10. When BUSINESS DEFINITIONS are provided, use their SQL expressions as a guide,\n"
    "    but STILL verify every column name against the EXACT COLUMN REGISTRY first.\n"
    "11. The EXACT COLUMN REGISTRY and COLUMN NOTES are the sole source of truth.\n"
    "    They override everything else including your training knowledge."
)
