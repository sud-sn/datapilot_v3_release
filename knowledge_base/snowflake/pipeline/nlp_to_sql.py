"""
DataPilot – NLP -> SQL -> Result Pipeline
Orchestrates: KB load -> schema fetch -> semantic enrichment
             -> prompt build -> LLM -> validate -> execute -> record.

Concepts are loaded from the markdown knowledge base (knowledge_store/)
rather than being hardcoded. The pipeline re-reads the KB file on each
instantiation so user edits take effect on the next pipeline refresh.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from adapters.base import DatabaseAdapter, SchemaContext, QueryResult
from knowledge_base.loader import get_knowledge_base
from knowledge_base.semantic_layer import (
    SemanticConcept,
    find_relevant_concepts,
    build_semantic_block,
)
from knowledge_base.join_graph import build_join_hints_block
from knowledge_base.conversation_context import get_session, record_turn
from llm.ollama import OllamaProvider, LLMResponse
from pipeline.validator import validate_sql, ValidationResult

logger = logging.getLogger(__name__)

_schema_cache: dict[str, tuple[SchemaContext, float]] = {}


async def get_schema_cached(
    adapter: DatabaseAdapter, ttl: int = 3600
) -> SchemaContext:
    key   = adapter.source_name
    now   = time.time()
    entry = _schema_cache.get(key)
    if entry and (now - entry[1]) < ttl:
        logger.debug("Schema cache hit: %s", key)
        return entry[0]
    logger.info("Fetching fresh schema: %s", key)
    schema = await adapter.fetch_schema()
    _schema_cache[key] = (schema, now)
    return schema


def invalidate_schema_cache(adapter: DatabaseAdapter) -> None:
    removed = _schema_cache.pop(adapter.source_name, None)
    if removed:
        logger.info("Schema cache invalidated: %s", adapter.source_name)


@dataclass
class PipelineResult:
    user_query:          str
    generated_sql:       str
    canonical_sql:       str
    model_used:          str
    complexity:          str
    validation:          ValidationResult
    query_result:        Optional[QueryResult] = None
    schema_fetch_ms:     int = 0
    llm_ms:              int = 0
    validation_ms:       int = 0
    execution_ms:        int = 0
    total_ms:            int = 0
    fallback_model_used: bool = False
    error:               Optional[str] = None

    @property
    def success(self) -> bool:
        return (
            self.validation.valid
            and self.query_result is not None
            and self.query_result.error is None
        )

    def to_api_response(self) -> dict:
        result_rows: list = []
        result_cols: list = []
        row_count   = 0
        truncated   = False
        exec_error: Optional[str] = None

        if self.query_result:
            result_rows = self.query_result.rows
            result_cols = self.query_result.columns
            row_count   = self.query_result.row_count
            truncated   = self.query_result.truncated
            exec_error  = self.query_result.error

        return {
            "success":       self.success,
            "query":         self.user_query,
            "sql":           self.canonical_sql or self.generated_sql,
            "model_used":    self.model_used,
            "complexity":    self.complexity,
            "fallback_used": self.fallback_model_used,
            "validation": {
                "valid":    self.validation.valid,
                "errors":   self.validation.errors,
                "warnings": self.validation.warnings,
            },
            "data": {
                "columns":   result_cols,
                "rows":      result_rows,
                "row_count": row_count,
                "truncated": truncated,
            },
            "timing": {
                "schema_fetch_ms": self.schema_fetch_ms,
                "llm_ms":          self.llm_ms,
                "validation_ms":   self.validation_ms,
                "execution_ms":    self.execution_ms,
                "total_ms":        self.total_ms,
            },
            "error": self.error or exec_error,
        }


class NLPToSQLPipeline:
    """
    Main orchestrator. One instance per DB source.

    concept_registry:
        List of SemanticConcept objects parsed from the markdown KB file.
    column_notes:
        Dict {table: {column: note}} parsed from the MD Columns tables.
        User-resolved NEEDS CONTEXT entries are included here and injected
        into the column registry for every SQL generation call.
    source_id:
        Identifier for the knowledge store lookup. Used in log messages.
    """

    def __init__(
        self,
        adapter:          DatabaseAdapter,
        llm:              OllamaProvider,
        schema_cache_ttl: int = 3600,
        concept_registry: Optional[list[SemanticConcept]] = None,
        column_notes:     Optional[dict[str, dict[str, str]]] = None,
        source_id:        str = "",
    ) -> None:
        self._adapter          = adapter
        self._llm              = llm
        self._cache_ttl        = schema_cache_ttl
        self._concept_registry = concept_registry
        self._column_notes     = column_notes or {}
        self._source_id        = source_id
        self._kb               = get_knowledge_base(adapter.dialect)
        logger.info(
            "Pipeline ready: source=%s | kb=%s | concepts=%s | col_notes=%d tables",
            source_id or adapter.source_name,
            self._kb.display_name,
            len(concept_registry) if concept_registry is not None else "global",
            len(self._column_notes),
        )

    def refresh_concepts(self, concepts: list[SemanticConcept]) -> None:
        """Hot-reload concepts without recreating the pipeline."""
        self._concept_registry = concepts
        logger.info(
            "Concepts refreshed: source=%s | count=%d",
            self._source_id, len(concepts),
        )

    def refresh_column_notes(self, column_notes: dict[str, dict[str, str]]) -> None:
        """
        Hot-reload column notes after user edits the KB markdown.
        Called by main.py whenever the user saves the KB.
        """
        self._column_notes = column_notes
        total = sum(len(v) for v in column_notes.values())
        logger.info(
            "Column notes refreshed: source=%s | %d tables | %d columns annotated",
            self._source_id, len(column_notes), total,
        )

    async def run(
        self,
        user_query: str,
        session_id: str = "default",
    ) -> PipelineResult:
        """
        Full pipeline:
          1  Fetch schema (TTL-cached)
          2a Match semantic concepts from KB
          2b Resolve join paths for matched concepts
          2c Load conversation history
          3  Build enriched system prompt
          4  Generate SQL (with model routing + fallback)
          5  Validate SQL (blocklist / parse / schema whitelist)
          6  Execute SQL
          7  Record turn for follow-up context
          8  Return PipelineResult
        """
        total_start = time.monotonic()

        # 1. Schema
        t0        = time.monotonic()
        schema    = await get_schema_cached(self._adapter, self._cache_ttl)
        schema_ms = int((time.monotonic() - t0) * 1000)

        # 2a. Semantic concepts
        concepts       = find_relevant_concepts(user_query, registry=self._concept_registry)
        semantic_block = build_semantic_block(user_query, registry=self._concept_registry)

        # 2b. Join path resolution
        concept_tables: list[str] = []
        seen: set[str] = set()
        for c in concepts:
            for t in c.join_path:
                if t not in seen:
                    concept_tables.append(t)
                    seen.add(t)
        join_hints_block = build_join_hints_block(user_query, schema, concept_tables)

        # 2c. Conversation history
        session_ctx        = get_session(session_id)
        conversation_block = session_ctx.format_history_block()

        # 3. Build prompt — pass column notes so user-defined semantics are enforced
        system_prompt = self._kb.format_prompt(
            schema,
            semantic_block     = semantic_block,
            join_hints_block   = join_hints_block,
            conversation_block = conversation_block,
            column_notes       = self._column_notes if self._column_notes else None,
        )

        # 4. LLM
        t0     = time.monotonic()
        llm_resp: LLMResponse = await self._llm.generate_sql(user_query, system_prompt)
        llm_ms = int((time.monotonic() - t0) * 1000)

        # 5. Validate
        t0         = time.monotonic()
        validation = validate_sql(llm_resp.sql, self._adapter.dialect, schema)
        val_ms     = int((time.monotonic() - t0) * 1000)

        result = PipelineResult(
            user_query          = user_query,
            generated_sql       = llm_resp.sql,
            canonical_sql       = validation.canonical_sql,
            model_used          = llm_resp.model_used,
            complexity          = llm_resp.complexity.value,
            validation          = validation,
            schema_fetch_ms     = schema_ms,
            llm_ms              = llm_ms,
            validation_ms       = val_ms,
            fallback_model_used = llm_resp.fallback_used,
        )

        if not validation.valid:
            logger.warning("Validation failed '%s': %s", user_query, validation.errors)
            result.total_ms = int((time.monotonic() - total_start) * 1000)
            result.error    = "; ".join(validation.errors)
            return result

        # 6. Execute
        t0           = time.monotonic()
        query_result = await self._adapter.execute(validation.canonical_sql)
        exec_ms      = int((time.monotonic() - t0) * 1000)
        result.query_result = query_result
        result.execution_ms = exec_ms
        result.total_ms     = int((time.monotonic() - total_start) * 1000)

        # 7. Record turn
        if query_result and not query_result.error:
            record_turn(
                session_id     = session_id,
                user_query     = user_query,
                generated_sql  = validation.canonical_sql,
                row_count      = query_result.row_count,
                tables_touched = concept_tables or [],
            )

        logger.info(
            "Pipeline OK | source=%s | model=%s | complexity=%s | concepts=%s | rows=%d | %dms",
            self._source_id,
            llm_resp.model_used,
            llm_resp.complexity.value,
            [c.name for c in concepts] if concepts else "none",
            query_result.row_count,
            result.total_ms,
        )
        return result
