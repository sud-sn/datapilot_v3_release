"""
DataPilot – Markdown Generator
Uses the Llama interview model to analyse a SchemaContext and produce
a structured markdown knowledge base document.

The LLM:
  1. Reads actual data values (not just column names)
  2. Uses the domain knowledge base as context
  3. Generates a full business hypothesis
  4. Writes it in a strictly defined markdown format

The output markdown is the source of truth for the SQL pipeline.
Users edit it to correct any misunderstandings.
"""
from __future__ import annotations
import logging
from datetime import datetime

from adapters.base import SchemaContext
from ingestion.excel_ingester import schema_to_prompt_text
from knowledge_base.domain_kb import get_domain_context
from llm.model_manager import ModelManager

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a senior business data analyst with deep expertise in
translating raw database schemas into clear business knowledge documentation.

Your job is to look at database tables, their columns, and actual sample data values,
then produce a comprehensive business knowledge base document that explains:
- What each table represents in business terms
- What the important columns mean
- How key business metrics should be calculated
- What filters should always be applied for accurate analysis
- A business vocabulary mapping common terms to SQL expressions

You write precise, accurate documentation based on evidence from the data.
You do not make things up — if you are unsure, you say so clearly.
Output ONLY the markdown document in the exact format requested."""


# ── User prompt template ──────────────────────────────────────────────────────

def _build_prompt(
    schema: SchemaContext,
    domain: str,
    source_id: str,
) -> str:
    domain_context = get_domain_context(domain)
    table_data     = schema_to_prompt_text(schema)
    table_names    = ", ".join(t.name for t in schema.tables)
    generated_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""Analyze the following database and generate a business knowledge base document.

DOMAIN: {domain}

DOMAIN CONTEXT (use this to help interpret the tables):
{domain_context}

DATABASE TABLES WITH SAMPLE DATA:
{table_data}

Generate the knowledge base document following EXACTLY this markdown structure.
Replace all [PLACEHOLDER] text with your actual analysis.

---START OF DOCUMENT---

# DataPilot Knowledge Base

> **Source:** {source_id}
> **Domain:** {domain}
> **Generated:** {generated_at}
> **Tables:** {table_names}

---

[For EACH table, add a section like this:]

## Table: [exact_table_name]

**Overview:**
[2-3 sentences describing what business process this table captures, what each row represents, and how it fits into the overall business]

**Key Metrics:**
- **[Metric Name]:** `[SQL expression e.g. SUM(column_name)]` — Filter: `[SQL filter condition]`
[Add one line per important metric you can identify from this table]

**Always Exclude:**
- `[SQL condition for records to always filter out e.g. is_test = TRUE]`
[Add one line per exclusion rule. If none obvious, write: None identified]

**Columns:**

| Column | Type | Meaning | Notes |
|--------|------|---------|-------|
[For each important column add a row:]
| [column_name] | [data_type] | [business meaning in plain English] | [any special rules, value meanings, or caveats] |

---

[After all table sections, add:]

## Business Vocabulary

[This is the most important section. Map every business term a user might ask about to its SQL translation.]

| Term | Plain English Meaning | SQL Expression | Tables Needed | Filters to Apply |
|------|----------------------|----------------|---------------|-----------------|
| [Business term] | [What it means] | `[SQL e.g. SUM(amount)]` | [table names] | `[WHERE conditions]` |

[Add at least 5-10 vocabulary entries covering the main metrics and entities in this data]

---

*Edit this file to correct any misunderstandings. Changes take effect immediately after saving.*

---END OF DOCUMENT---

Rules:
1. Use EXACT column names as they appear in the sample data above
2. For metrics, write complete SQL expressions including aggregate functions
3. For filters, write complete SQL WHERE conditions using actual values you see in the sample data
4. If a column's purpose is unclear, write "Purpose unclear — please update" in the Notes column
5. Output ONLY the markdown document between the START and END markers"""


# ── Generator ─────────────────────────────────────────────────────────────────

async def generate_knowledge_base(
    schema: SchemaContext,
    domain: str,
    source_id: str,
    model_manager: ModelManager,
) -> str:
    """
    Generate the full markdown knowledge base for a schema.

    Returns the raw markdown string. The caller is responsible for
    saving it to the knowledge store.

    Raises RuntimeError if the model call fails.
    """
    logger.info(
        "Generating knowledge base | source=%s | domain=%s | tables=%d",
        source_id, domain, len(schema.tables),
    )

    prompt = _build_prompt(schema, domain, source_id)

    raw = await model_manager.generate_markdown(
        user_prompt=prompt,
        system_prompt=_SYSTEM_PROMPT,
    )

    # Strip the START/END markers if the model included them
    md = _extract_markdown(raw)

    if not md.strip():
        raise RuntimeError(
            "The interview model returned an empty response. "
            "Check that the model is correctly pulled and has enough context length."
        )

    logger.info(
        "Knowledge base generated: %d characters | source=%s",
        len(md), source_id,
    )
    return md


def _extract_markdown(raw: str) -> str:
    """
    Extract the markdown content from between the START/END markers
    if the model included them. Otherwise return the raw output.
    """
    start_marker = "---START OF DOCUMENT---"
    end_marker   = "---END OF DOCUMENT---"

    if start_marker in raw and end_marker in raw:
        start = raw.index(start_marker) + len(start_marker)
        end   = raw.index(end_marker)
        return raw[start:end].strip()

    # Fallback: find the first # heading and return from there
    if "# DataPilot Knowledge Base" in raw:
        idx = raw.index("# DataPilot Knowledge Base")
        return raw[idx:].strip()

    return raw.strip()
