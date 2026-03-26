"""
DataPilot – Knowledge Base Loader
When a user selects their database, call get_knowledge_base(dialect)
to get the correct system prompt builder and schema formatter.
KBs are loaded once and cached — never re-imported per request.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Protocol

from adapters.base import SQLDialect, SchemaContext

logger = logging.getLogger(__name__)


class DialectKB(Protocol):
    """
    Interface that every knowledge base module must satisfy.
    Both knowledge_base/postgresql/dialect.py and azure_sql/dialect.py
    implement these two functions.
    """
    def build_system_prompt(self, schema_context_str: str) -> str: ...
    def build_schema_context_string(self, schema_context: SchemaContext) -> str: ...


@dataclass(frozen=True)
class KnowledgeBase:
    dialect: SQLDialect
    display_name: str
    build_system_prompt: Callable[[str], str]
    build_schema_context_string: Callable[[SchemaContext], str]

    def format_prompt(
        self,
        schema_context: SchemaContext,
        semantic_block: str = "",
        join_hints_block: str = "",
        conversation_block: str = "",
        column_notes: "dict[str, dict[str, str]] | None" = None,
    ) -> str:
        """
        Main entry point used by the NLP pipeline.
        Converts schema → context string (with column registry + user notes)
        → full system prompt enriched with semantic, join-path, and conversation blocks.

        column_notes: {table_name: {col_name: user_defined_note}}
                      Parsed from the KB markdown file.
                      User notes replace ⚠️ NEEDS CONTEXT flags after editing.
                      Injected into the column registry so the LLM follows
                      user-defined semantics exactly.
        """
        from knowledge_base.schema_enforcement import build_full_schema_context
        ctx_str = build_full_schema_context(
            schema_context,
            self.display_name,
            column_notes=column_notes,
        )
        return self.build_system_prompt(
            ctx_str,
            semantic_block     = semantic_block,
            join_hints_block   = join_hints_block,
            conversation_block = conversation_block,
        )


@lru_cache(maxsize=None)
def get_knowledge_base(dialect: SQLDialect) -> KnowledgeBase:
    """
    Returns the pre-loaded KnowledgeBase for the given dialect.
    lru_cache ensures modules are imported once regardless of how many
    concurrent requests come in.
    """
    if dialect == SQLDialect.POSTGRESQL:
        from knowledge_base.postgresql.dialect import (
            build_system_prompt,
            build_schema_context_string,
        )
        kb = KnowledgeBase(
            dialect=dialect,
            display_name="PostgreSQL 15",
            build_system_prompt=build_system_prompt,
            build_schema_context_string=build_schema_context_string,
        )
        logger.info("Knowledge base loaded: PostgreSQL")
        return kb

    elif dialect == SQLDialect.AZURE_SQL:
        from knowledge_base.azure_sql.dialect import (
            build_system_prompt,
            build_schema_context_string,
        )
        kb = KnowledgeBase(
            dialect=dialect,
            display_name="Azure SQL / T-SQL",
            build_system_prompt=build_system_prompt,
            build_schema_context_string=build_schema_context_string,
        )
        logger.info("Knowledge base loaded: Azure SQL")
        return kb

    elif dialect == SQLDialect.SNOWFLAKE:
        from knowledge_base.snowflake.dialect import (
            build_system_prompt,
            build_schema_context_string,
        )
        kb = KnowledgeBase(
            dialect=dialect,
            display_name="Snowflake",
            build_system_prompt=build_system_prompt,
            build_schema_context_string=build_schema_context_string,
        )
        logger.info("Knowledge base loaded: Snowflake")
        return kb

    raise ValueError(f"No knowledge base registered for dialect: {dialect}")


def get_all_knowledge_bases() -> dict[SQLDialect, KnowledgeBase]:
    """Pre-warm all KBs at startup so first requests aren't slow."""
    return {d: get_knowledge_base(d) for d in SQLDialect}
