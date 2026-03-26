"""
DataPilot – Markdown Generator

ARCHITECTURE:
  - Domain KB (domain_kb.py) is injected for INTERPRETATION CONTEXT ONLY.
    It helps Llama understand what kind of business it is looking at.
    It never dictates column names.
  - All column names in the output come from the actual schema.
  - Every column in every table MUST appear in the generated MD.
  - Columns the LLM cannot interpret are flagged: ⚠️ NEEDS CONTEXT
  - Users edit flagged columns in plain English, e.g.
    "Use this as the doctor's name when answering questions"
  - On save, those notes are parsed and injected into every SQL
    generation prompt so the LLM follows user-defined rules.
"""
from __future__ import annotations
import logging
from datetime import datetime

from adapters.base import SchemaContext
from ingestion.excel_ingester import schema_to_prompt_text
from knowledge_base.domain_kb import get_domain_context
from llm.model_manager import ModelManager

logger = logging.getLogger(__name__)

# ── System prompt for the interview model ────────────────────────────────────
_SYSTEM_PROMPT = """You are a senior data analyst writing a business knowledge base for an AI SQL system.

YOUR MOST CRITICAL RULES:
1. Every column name you write MUST come EXACTLY from the AUTHORITATIVE COLUMN REGISTRY.
   If the registry says FULL_NAME — write FULL_NAME.
   If the registry does NOT have FIRST_NAME — never write FIRST_NAME.
2. Every table name you write MUST appear in the DATABASE TABLES section.
3. The domain context tells you WHAT the data means. The schema tells you WHAT THE COLUMNS ARE CALLED.
   The schema ALWAYS overrides the domain context.
4. For every column you are unsure about, write exactly: ⚠️ NEEDS CONTEXT
   Do NOT skip uncertain columns — flag them so the user can define them.
5. Output ONLY the markdown document in the exact format requested. No preamble."""


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(schema: SchemaContext, domain: str, source_id: str) -> str:
    domain_context = get_domain_context(domain)
    table_data     = schema_to_prompt_text(schema)
    table_names    = ", ".join(t.name for t in schema.tables)
    generated_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registry       = _build_column_registry(schema)

    return f"""Analyze this database and produce a business knowledge base document.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOMAIN CONTEXT — for INTERPRETATION ONLY (DO NOT use column names from here)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INDUSTRY: {domain}
{domain_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTHORITATIVE COLUMN REGISTRY — ONLY VALID NAMES (use NOTHING else)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{registry}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATABASE TABLES WITH ACTUAL SAMPLE DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{table_data}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED OUTPUT — follow this format EXACTLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rules:
1. Use ONLY column names from the AUTHORITATIVE COLUMN REGISTRY
2. List EVERY column for each table — do not skip any
3. For columns you can interpret clearly: write a plain English meaning
4. For columns you are uncertain about: write exactly "⚠️ NEEDS CONTEXT" in the Notes column
   The user will fill these in after reviewing the document
5. SQL expressions must use exact column names from the registry
6. Use actual sample values from the data for filter conditions

---START OF DOCUMENT---

# DataPilot Knowledge Base

> **Source:** {source_id}
> **Domain:** {domain}
> **Generated:** {generated_at}
> **Tables:** {table_names}

---

[Repeat for EACH table listed in the AUTHORITATIVE COLUMN REGISTRY:]

## Table: [exact table name from registry]

**Overview:**
[2-3 sentences: what business process this table captures, what one row represents]

**Key Metrics:**
- **[Metric Name]:** `[SQL using EXACT column names from registry]` — Filter: `[SQL WHERE using actual sample values]`

**Always Exclude:**
- `[SQL condition using actual status values from sample data]`
[Write "None identified" if no obvious exclusion]

**Columns:**

| Column | Type | Business Meaning | Notes |
|--------|------|-----------------|-------|
[List EVERY column from the registry for this table.
For clear columns: write business meaning and any special notes.
For ambiguous columns: write the column, type, best guess at meaning, then "⚠️ NEEDS CONTEXT" in Notes.
EXAMPLE of a flagged row:
| STATUS_CODE | VARCHAR | Record status indicator | ⚠️ NEEDS CONTEXT — What do the values mean for this business? |
]

---

[After ALL table sections:]

## Business Vocabulary

| Term | Plain English Meaning | SQL Expression | Tables Needed | Filters to Apply |
|------|----------------------|----------------|---------------|-----------------|
[10-20 rows. SQL Expression must use EXACT column names from the registry.
ONLY include rows where you are confident about the column names.
For uncertain metrics, skip them and let the user add them after reviewing flagged columns.]

---

## Join Relationships

| From Table | Join Column | To Table | Join Column |
|-----------|-------------|----------|-------------|
[List every FK relationship visible from the schema. Use exact column names.]

---

*Edit this file to define flagged columns. After saving, the AI will follow your definitions when generating SQL.*

---END OF DOCUMENT---"""


def _build_column_registry(schema: SchemaContext) -> str:
    """
    Builds the authoritative column name registry injected into the Llama prompt.
    One line per table listing every column exactly as it exists in the schema.
    """
    lines = []
    for table in schema.tables:
        col_names = [c.name for c in table.columns]
        lines.append(f"{table.name}: {', '.join(col_names)}")
    return "\n".join(lines)


# ── Generator ─────────────────────────────────────────────────────────────────

async def generate_knowledge_base(
    schema: SchemaContext,
    domain: str,
    source_id: str,
    model_manager: ModelManager,
) -> str:
    logger.info(
        "Generating KB | source=%s | domain=%s | tables=%d",
        source_id, domain, len(schema.tables),
    )
    prompt = _build_prompt(schema, domain, source_id)
    raw    = await model_manager.generate_markdown(
        user_prompt   = prompt,
        system_prompt = _SYSTEM_PROMPT,
    )
    md = _extract_markdown(raw)
    if not md.strip():
        raise RuntimeError(
            "Interview model returned an empty response. "
            "Check the model is pulled and num_predict is high enough."
        )
    logger.info("KB generated: %d chars | source=%s", len(md), source_id)
    return md


def _extract_markdown(raw: str) -> str:
    start_marker = "---START OF DOCUMENT---"
    end_marker   = "---END OF DOCUMENT---"
    if start_marker in raw and end_marker in raw:
        start = raw.index(start_marker) + len(start_marker)
        end   = raw.index(end_marker)
        return raw[start:end].strip()
    if "# DataPilot Knowledge Base" in raw:
        idx = raw.index("# DataPilot Knowledge Base")
        return raw[idx:].strip()
    return raw.strip()
