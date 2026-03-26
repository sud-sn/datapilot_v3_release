"""
DataPilot – Semantic Layer
Maps high-level business concepts ("revenue", "active users", "churn")
to concrete SQL fragments so the LLM doesn't have to guess.

HOW TO CONFIGURE:
  Each SemanticConcept describes ONE business metric or entity.
  Fill CONCEPT_REGISTRY with your own definitions, or pass a custom list
  to find_relevant_concepts() / build_semantic_block() for per-tenant use.

HOW IT WORKS:
  1. The pipeline scans the user query for concept keywords.
  2. Matching concepts are serialised and injected into the LLM prompt
     BEFORE the schema section, as a "BUSINESS DEFINITIONS" block.
  3. The LLM is instructed to honour these definitions exactly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SemanticConcept:
    """
    One named business concept with enough SQL information for the LLM to
    use it correctly even across multiple tables and filters.
    """
    name: str                        # e.g. "revenue"
    keywords: list[str]              # words that trigger this concept
    description: str                 # one-line plain-English definition
    primary_table: str               # main table that holds the data
    value_column: str                # the column to SUM/AVG/etc.
    join_path: list[str]             # ordered list of tables to join through
    required_filters: list[str]      # SQL WHERE conditions that MUST apply
    group_by_hints: list[str]        # commonly useful GROUP BY columns
    example_query: Optional[str] = None   # optional worked example

    def to_prompt_block(self) -> str:
        lines = [
            f"CONCEPT: {self.name.upper()}",
            f"  Definition   : {self.description}",
            f"  Primary table: {self.primary_table}",
            f"  Value column : {self.value_column}",
        ]
        if self.join_path:
            lines.append(f"  Join path    : {' -> '.join(self.join_path)}")
        if self.required_filters:
            lines.append("  MUST filter  : " + "  AND  ".join(self.required_filters))
        if self.group_by_hints:
            lines.append(f"  Group-by hints: {', '.join(self.group_by_hints)}")
        if self.example_query:
            lines.append(f"  Example:\n    {self.example_query}")
        return "\n".join(lines)


# ── Default global concept registry ──────────────────────────────────────────
# Edit this to match your database schema.
# For per-tenant concepts, pass a list directly to the functions below.

CONCEPT_REGISTRY: list[SemanticConcept] = [
    SemanticConcept(
        name="revenue",
        keywords=["revenue", "sales", "income", "turnover", "gmv"],
        description="Total value of completed / paid orders, excluding refunded and cancelled.",
        primary_table="orders",
        value_column="total_amount",
        join_path=["orders"],
        required_filters=[
            "status IN ('completed', 'paid', 'shipped')",
            "refunded_at IS NULL",
        ],
        group_by_hints=["DATE_TRUNC('month', created_at)", "customer_id", "product_id"],
        example_query=(
            "SELECT DATE_TRUNC('month', o.created_at) AS month,\n"
            "       SUM(o.total_amount) AS revenue\n"
            "FROM   orders o\n"
            "WHERE  o.status IN ('completed','paid','shipped')\n"
            "  AND  o.refunded_at IS NULL\n"
            "GROUP  BY 1 ORDER BY 1;"
        ),
    ),
    SemanticConcept(
        name="active_users",
        keywords=["active user", "mau", "dau", "engaged user", "active customer"],
        description="Users who performed at least one session or event in the period.",
        primary_table="events",
        value_column="user_id",
        join_path=["events", "users"],
        required_filters=["event_type != 'bot'"],
        group_by_hints=["DATE_TRUNC('month', occurred_at)", "country"],
        example_query=(
            "SELECT DATE_TRUNC('month', e.occurred_at) AS month,\n"
            "       COUNT(DISTINCT e.user_id) AS active_users\n"
            "FROM   events e\n"
            "WHERE  e.event_type != 'bot'\n"
            "GROUP  BY 1 ORDER BY 1;"
        ),
    ),
    SemanticConcept(
        name="gross_profit",
        keywords=["gross profit", "gross margin", "gp"],
        description="Revenue minus cost of goods sold (COGS).",
        primary_table="orders",
        value_column="total_amount - cogs_amount",
        join_path=["orders", "order_items", "products"],
        required_filters=[
            "orders.status IN ('completed', 'paid', 'shipped')",
            "orders.refunded_at IS NULL",
        ],
        group_by_hints=["DATE_TRUNC('month', orders.created_at)", "products.category"],
    ),
    SemanticConcept(
        name="churn",
        keywords=["churn", "churned", "lost customer", "retention"],
        description=(
            "Customers who were active in the previous period but NOT in the current period. "
            "Use a 30-day inactivity window by default."
        ),
        primary_table="orders",
        value_column="customer_id",
        join_path=["orders", "customers"],
        required_filters=[],
        group_by_hints=["DATE_TRUNC('month', last_order_at)"],
    ),
]


# ── Public API ────────────────────────────────────────────────────────────────

def find_relevant_concepts(
    user_query: str,
    registry: Optional[list[SemanticConcept]] = None,
) -> list[SemanticConcept]:
    """
    Returns all concepts whose keywords appear in the user query (case-insensitive).

    registry: optional per-tenant concept list.
              When None, falls back to the global CONCEPT_REGISTRY.
    """
    source = registry if registry is not None else CONCEPT_REGISTRY
    q = user_query.lower()
    return [c for c in source if any(kw in q for kw in c.keywords)]


def build_semantic_block(
    user_query: str,
    registry: Optional[list[SemanticConcept]] = None,
) -> str:
    """
    Returns a formatted prompt block for all concepts relevant to this query.
    Empty string if no concepts matched.

    registry: optional per-tenant concept list.
              When None, falls back to the global CONCEPT_REGISTRY.
    """
    concepts = find_relevant_concepts(user_query, registry=registry)
    if not concepts:
        return ""
    lines = ["=== BUSINESS DEFINITIONS (you MUST follow these exactly) ==="]
    for concept in concepts:
        lines.append("")
        lines.append(concept.to_prompt_block())
    lines.append("")
    lines.append(
        "IMPORTANT: When a BUSINESS DEFINITION exists for a term the user mentions,\n"
        "use EXACTLY the tables, filters, and join path it specifies. Do not deviate."
    )
    return "\n".join(lines)
