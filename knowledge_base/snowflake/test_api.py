"""
DataPilot – API test suite
Tests every endpoint and validates that the semantic layer, session handling,
and concept registry all function correctly without a live database.

Run with:   python test_api.py [--url http://localhost:8000]

Exit 0 = all passed. Exit 1 = at least one failure.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import uuid
import requests

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
SKIP = "\033[93m  SKIP\033[0m"

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"{PASS}  {name}")
    else:
        msg = f"{name}" + (f" — {detail}" if detail else "")
        print(f"{FAIL}  {msg}")
        _failures.append(msg)
    return condition


def section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 55 - len(title))}")


# ── Unit tests (no live server required) ─────────────────────────────────────

def test_semantic_layer() -> None:
    section("Semantic layer (unit)")
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from knowledge_base.semantic_layer import (
        SemanticConcept, CONCEPT_REGISTRY,
        find_relevant_concepts, build_semantic_block,
    )

    # Global registry
    hits = find_relevant_concepts("what is the revenue this month")
    check("global registry: finds revenue", any(c.name == "revenue" for c in hits))

    # No match
    hits2 = find_relevant_concepts("how many rows are in the table")
    check("global registry: no false match", len(hits2) == 0)

    # Per-tenant override
    custom = [SemanticConcept(
        name="mrr", keywords=["mrr", "monthly recurring"],
        description="Monthly recurring revenue.", primary_table="subscriptions",
        value_column="amount", join_path=["subscriptions"],
        required_filters=["status='active'"], group_by_hints=[],
    )]
    hits3 = find_relevant_concepts("show mrr trend", registry=custom)
    check("custom registry: finds mrr", any(c.name == "mrr" for c in hits3))

    hits4 = find_relevant_concepts("show revenue", registry=custom)
    check("custom registry: no revenue bleed", len(hits4) == 0,
          f"expected 0, got {len(hits4)}")

    # Empty override
    hits5 = find_relevant_concepts("revenue churn active users", registry=[])
    check("empty registry: no matches", len(hits5) == 0)

    # Prompt block
    block = build_semantic_block("revenue by month")
    check("prompt block: has header", "BUSINESS DEFINITIONS" in block)
    check("prompt block: has concept", "CONCEPT: REVENUE" in block)
    check("prompt block: has filters", "refunded_at IS NULL" in block)

    empty_block = build_semantic_block("what time is it", registry=[])
    check("empty registry: empty block", empty_block == "")


def test_join_graph() -> None:
    section("Join graph (unit)")
    from knowledge_base.join_graph import JoinGraph, JoinEdge, build_join_hints_block
    from adapters.base import SchemaContext, SQLDialect

    g = JoinGraph()
    g.add_edge(JoinEdge("orders", "customer_id", "customers", "id"))
    g.add_edge(JoinEdge("orders", "product_id",  "products",  "id"))
    g.add_edge(JoinEdge("products", "category_id", "categories", "id"))

    # Direct neighbour
    path = g.find_join_path("orders", "customers")
    check("path: orders->customers (1 hop)", path is not None and len(path) == 1)

    # 2 hops
    path2 = g.find_join_path("customers", "products")
    check("path: customers->products (2 hops)", path2 is not None and len(path2) == 2)

    # 3 hops
    path3 = g.find_join_path("customers", "categories")
    check("path: customers->categories (3 hops)", path3 is not None and len(path3) == 3)

    # Same table
    same = g.find_join_path("orders", "orders")
    check("path: same table returns []", same == [])

    # Unknown table
    none_path = g.find_join_path("orders", "nonexistent")
    check("path: unknown table returns None", none_path is None)

    # build_join_hints_block: 1 table → no hint
    schema = SchemaContext(
        dialect=SQLDialect.POSTGRESQL, database="db", default_schema="public",
        relationships={"orders": [("customer_id", "customers", "id")]},
    )
    hint = build_join_hints_block("q", schema, ["orders"])
    check("hint: 1 table = empty string", hint == "")

    hint2 = build_join_hints_block("q", schema, ["orders", "customers"])
    check("hint: 2 tables = JOIN PATH block", "JOIN PATH" in hint2)
    check("hint: contains JOIN keyword", "JOIN" in hint2)


def test_conversation_context() -> None:
    section("Conversation context (unit)")
    from knowledge_base.conversation_context import get_session, record_turn

    sid_a = f"pg:test-{uuid.uuid4().hex[:8]}"
    sid_b = f"pg:test-{uuid.uuid4().hex[:8]}"

    # Fresh session is empty
    ctx_a = get_session(sid_a)
    check("new session: empty history", ctx_a.format_history_block() == "")

    # Record a turn
    record_turn(sid_a, "show revenue", "SELECT SUM(total_amount) FROM orders;", 12, ["orders"])
    ctx_a2 = get_session(sid_a)
    block = ctx_a2.format_history_block()
    check("history: contains header", "CONVERSATION HISTORY" in block)
    check("history: contains user query", "show revenue" in block)
    check("history: contains SQL", "SELECT SUM" in block)
    check("history: contains row count", "12 rows" in block)

    # Different session must be isolated
    ctx_b = get_session(sid_b)
    check("isolation: different session is empty", ctx_b.format_history_block() == "")

    # Rolling window: add 6 turns to a session with max_turns=5
    sid_c = f"pg:test-{uuid.uuid4().hex[:8]}"
    for i in range(6):
        record_turn(sid_c, f"query {i}", f"SELECT {i};", i, [])
    ctx_c = get_session(sid_c)
    check("rolling window: max 5 turns kept", len(ctx_c.turns) == 5)
    check("rolling window: oldest turn dropped", ctx_c.turns[0].user_query == "query 1")


def test_pipeline_constructor() -> None:
    section("Pipeline constructor (unit)")
    from unittest.mock import MagicMock
    from adapters.base import SQLDialect
    from knowledge_base.semantic_layer import SemanticConcept
    from pipeline.nlp_to_sql import NLPToSQLPipeline

    mock_adapter = MagicMock()
    mock_adapter.dialect = SQLDialect.POSTGRESQL
    mock_adapter.source_name = "mock-pg"
    mock_llm = MagicMock()

    p1 = NLPToSQLPipeline(mock_adapter, mock_llm)
    check("no registry param: _concept_registry is None", p1._concept_registry is None)

    p2 = NLPToSQLPipeline(mock_adapter, mock_llm, concept_registry=[])
    check("empty registry: _concept_registry is []", p2._concept_registry == [])

    custom = [SemanticConcept(
        name="mrr", keywords=["mrr"], description="MRR", primary_table="subs",
        value_column="amount", join_path=["subs"], required_filters=[], group_by_hints=[],
    )]
    p3 = NLPToSQLPipeline(mock_adapter, mock_llm, concept_registry=custom)
    check("custom registry: stored correctly", p3._concept_registry == custom)

    # None registry uses global defaults, empty registry uses nothing
    from knowledge_base.semantic_layer import find_relevant_concepts
    hits_global = find_relevant_concepts("revenue", registry=p1._concept_registry)
    hits_empty  = find_relevant_concepts("revenue", registry=p2._concept_registry)
    check("None registry uses global defaults", len(hits_global) > 0)
    check("empty registry matches nothing",     len(hits_empty) == 0)


def test_dialect_prompt_signatures() -> None:
    section("Dialect prompt signatures (unit)")
    from knowledge_base.postgresql.dialect import build_system_prompt as pg_prompt
    from knowledge_base.azure_sql.dialect   import build_system_prompt as az_prompt

    for name, fn in [("postgresql", pg_prompt), ("azure_sql", az_prompt)]:
        # Must accept all four positional/keyword args without error
        try:
            result = fn(
                "SCHEMA: orders",
                semantic_block="=== BUSINESS DEFINITIONS ===",
                join_hints_block="=== JOIN PATH ===",
                conversation_block="=== CONVERSATION HISTORY ===",
            )
            check(f"{name}: accepts 4-arg signature", True)
            check(f"{name}: injects semantic block",     "BUSINESS DEFINITIONS" in result)
            check(f"{name}: injects join hint",          "JOIN PATH" in result)
            check(f"{name}: injects conversation block", "CONVERSATION HISTORY" in result)
            check(f"{name}: injects schema",             "SCHEMA" in result)
        except TypeError as e:
            check(f"{name}: signature OK", False, str(e))

        # Must work with only the required arg (backward compat)
        try:
            result_min = fn("SCHEMA: orders")
            check(f"{name}: backward-compat single arg", "SCHEMA" in result_min)
        except TypeError as e:
            check(f"{name}: backward-compat single arg", False, str(e))


def test_loader_format_prompt() -> None:
    section("KnowledgeBase.format_prompt (unit)")
    from knowledge_base.loader import get_knowledge_base
    from adapters.base import SchemaContext, SQLDialect, TableInfo, ColumnInfo

    for dialect, name in [(SQLDialect.POSTGRESQL, "PostgreSQL"), (SQLDialect.AZURE_SQL, "Azure SQL")]:
        kb = get_knowledge_base(dialect)
        schema = SchemaContext(
            dialect=dialect, database="testdb", default_schema="public",
            tables=[TableInfo(schema="public", name="orders", columns=[
                ColumnInfo(name="id", data_type="numeric", raw_type="int4"),
                ColumnInfo(name="total_amount", data_type="numeric", raw_type="numeric"),
            ])],
            relationships={},
        )
        prompt = kb.format_prompt(
            schema,
            semantic_block="=== BUSINESS DEFINITIONS ===\nCONCEPT: REVENUE",
            join_hints_block="=== JOIN PATH ===\nFROM orders",
            conversation_block="=== CONVERSATION HISTORY ===\nTurn 1: test",
        )
        check(f"{name}: all blocks injected",
              all(x in prompt for x in ["BUSINESS DEFINITIONS", "JOIN PATH", "CONVERSATION HISTORY", "SCHEMA"]))

        # Backward-compat: no extra blocks
        prompt_min = kb.format_prompt(schema)
        check(f"{name}: no-block call still works", "SCHEMA" in prompt_min)
        check(f"{name}: no spurious block header", "=== BUSINESS DEFINITIONS ===" not in prompt_min)


# ── Live server tests (skipped if server not reachable) ───────────────────────

def _get(url: str, **kwargs):
    return requests.get(url, timeout=10, **kwargs)

def _post(url: str, payload: dict, **kwargs):
    return requests.post(url, json=payload, timeout=30, **kwargs)


def test_live_health(base: str) -> bool:
    section("Live: /health")
    try:
        r = _get(f"{base}/health")
        check("status 200", r.status_code == 200)
        data = r.json()
        check("has 'status' key", "status" in data)
        check("has 'databases' key", "databases" in data)
        return True
    except requests.ConnectionError:
        print(f"{SKIP}  Server not reachable at {base}")
        return False


def test_live_sources(base: str) -> None:
    section("Live: /sources")
    r = _get(f"{base}/sources")
    check("status 200", r.status_code == 200)
    data = r.json()
    check("has 'sources' key", "sources" in data)
    if data.get("sources"):
        src = data["sources"][0]
        check("source has 'id'",      "id"      in src)
        check("source has 'dialect'", "dialect" in src)
        check("source has 'name'",    "name"    in src)


def test_live_api_health(base: str) -> None:
    section("Live: /api/health")
    r = _get(f"{base}/api/health")
    check("status 200", r.status_code == 200)
    data = r.json()
    check("has 'status'", "status" in data)
    check("has 'schema'", "schema" in data)


def test_live_concepts(base: str) -> None:
    section("Live: /api/concepts")
    r = _get(f"{base}/api/concepts")
    check("status 200", r.status_code == 200)
    data = r.json()
    check("has 'concepts' key", "concepts" in data)
    for src, concepts in data.get("concepts", {}).items():
        check(f"{src}: is list", isinstance(concepts, list))
        if concepts:
            c = concepts[0]
            check(f"{src}: concept has name",        "name"        in c)
            check(f"{src}: concept has keywords",    "keywords"    in c)
            check(f"{src}: concept has description", "description" in c)


def test_live_ask_single(base: str) -> None:
    section("Live: /api/ask — single turn")
    sid = f"test-{uuid.uuid4().hex[:8]}"
    r = _post(f"{base}/api/ask", {
        "question":             "how many tables exist in the database",
        "session_id":           sid,
        "conversation_history": [],
    })
    check("status 200", r.status_code == 200)
    data = r.json()
    check("has 'answer'",         "answer"         in data)
    check("has 'sql'",            "sql"            in data)
    check("has 'data'",           "data"           in data)
    check("has 'execution_time'", "execution_time_ms" in data)
    check("sql is non-empty",     bool(data.get("sql", "").strip()))


def test_live_ask_followup(base: str) -> None:
    section("Live: /api/ask — follow-up session")
    sid = f"test-{uuid.uuid4().hex[:8]}"

    # Turn 1
    r1 = _post(f"{base}/api/ask", {
        "question":   "show me the most recent 3 records",
        "session_id": sid,
        "conversation_history": [],
    })
    check("turn 1: status 200", r1.status_code == 200)

    time.sleep(0.5)  # small gap so session is written

    # Turn 2 — follow-up: should context-carry without re-specifying the table
    r2 = _post(f"{base}/api/ask", {
        "question":   "now show me only the ones from last month",
        "session_id": sid,
        "conversation_history": [],
    })
    check("turn 2: status 200", r2.status_code == 200)
    data2 = r2.json()
    check("turn 2: has answer", "answer" in data2)
    check("turn 2: has sql",    bool(data2.get("sql", "").strip()))


def test_live_schema_refresh(base: str) -> None:
    section("Live: /schema/refresh")
    # Peek at available sources first
    r_src = _get(f"{base}/sources")
    sources = r_src.json().get("sources", [])
    if not sources:
        print(f"{SKIP}  No sources available")
        return
    src_id = sources[0]["id"]
    r = _post(f"{base}/schema/refresh", {"db_source": src_id})
    check("status 200", r.status_code == 200)
    data = r.json()
    check("refreshed=True", data.get("refreshed") is True)
    check("has 'tables'",   "tables" in data)


def test_live_bad_request(base: str) -> None:
    section("Live: error handling")
    # Missing required field
    r = _post(f"{base}/api/ask", {})
    check("missing question: 422", r.status_code == 422)

    # Unknown source on /schema/refresh
    r2 = _post(f"{base}/schema/refresh", {"db_source": "nonexistent_source"})
    check("unknown source: 4xx", r2.status_code >= 400)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DataPilot API test suite")
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="Base URL of the running DataPilot server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--unit-only", action="store_true",
        help="Run only unit tests (no live server required)"
    )
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print("=" * 60)
    print("DataPilot test suite")
    print("=" * 60)

    # Unit tests — always run
    test_semantic_layer()
    test_join_graph()
    test_conversation_context()
    test_pipeline_constructor()
    test_dialect_prompt_signatures()
    test_loader_format_prompt()

    if not args.unit_only:
        # Live server tests
        server_up = test_live_health(base)
        if server_up:
            test_live_sources(base)
            test_live_api_health(base)
            test_live_concepts(base)
            test_live_ask_single(base)
            test_live_ask_followup(base)
            test_live_schema_refresh(base)
            test_live_bad_request(base)
        else:
            print(f"\n{SKIP}  All live tests skipped — server not running")

    print()
    print("=" * 60)
    if _failures:
        print(f"\033[91mFAILED: {len(_failures)} test(s)\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\033[92mAll tests passed.\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
