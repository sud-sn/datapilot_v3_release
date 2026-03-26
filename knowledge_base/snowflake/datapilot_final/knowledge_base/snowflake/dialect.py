"""
DataPilot – Snowflake SQL Knowledge Base
Injected into the LLM system prompt when the selected DB is Snowflake.
"""
from __future__ import annotations

DATE_TIME_FUNCTIONS = """
=== DATE & TIME (Snowflake) ===
CURRENT TIMESTAMP  : CURRENT_TIMESTAMP()   or   SYSDATE()
TODAY              : CURRENT_DATE()
DATE ADD           : DATEADD(day,   -30, CURRENT_DATE())
                     DATEADD(month, -1,  CURRENT_DATE())
                     DATEADD(year,  -1,  CURRENT_DATE())
DATE DIFF          : DATEDIFF(day,   start_col, end_col)
                     DATEDIFF(month, start_col, end_col)
DATE PART          : DATE_PART('year',  col)
                     DATE_PART('month', col)
                     DATE_PART('dayofweek', col)
DATE TRUNC         : DATE_TRUNC('month', col)
                     DATE_TRUNC('year',  col)
                     DATE_TRUNC('week',  col)
TO_DATE            : TO_DATE('2024-01-01')
                     TO_DATE(col, 'YYYY-MM-DD')
CONVERT TIMEZONE   : CONVERT_TIMEZONE('UTC', 'US/Eastern', col)
COMMON RANGES:
  Last 30 days     : WHERE col >= DATEADD(day, -30, CURRENT_DATE())
  This month       : WHERE DATE_TRUNC('month', col) = DATE_TRUNC('month', CURRENT_DATE())
  Last month       : WHERE DATE_TRUNC('month', col) = DATE_TRUNC('month', DATEADD(month, -1, CURRENT_DATE()))
  This year        : WHERE DATE_PART('year', col) = DATE_PART('year', CURRENT_DATE())
NEVER USE: GETDATE(), NOW() with parens, INTERVAL syntax, EOMONTH() — these are T-SQL/PostgreSQL only.
"""

STRING_FUNCTIONS = """
=== STRINGS (Snowflake) ===
CONCAT          : col1 || col2   or   CONCAT(col1, col2)
CASE-INSENSITIVE: ILIKE is supported: col ILIKE '%pattern%'
                  Or use: LOWER(col) LIKE LOWER('%pattern%')
LOWER / UPPER   : LOWER(col)   UPPER(col)
TRIM            : TRIM(col)   LTRIM(col)   RTRIM(col)
SUBSTRING       : SUBSTR(col, 1, 5)   or   SUBSTRING(col, 1, 5)
POSITION        : POSITION('str' IN col)   or   CHARINDEX('str', col)
LENGTH          : LENGTH(col)   or   LEN(col)
REPLACE         : REPLACE(col, 'old', 'new')
SPLIT           : SPLIT_PART(col, ',', 1)   -- 1-indexed
ARRAY AGG       : ARRAY_AGG(col)   or   LISTAGG(col, ', ')
COALESCE        : COALESCE(col, 'default')
NULLIF          : NULLIF(col, '')
REGEXP          : REGEXP_LIKE(col, 'pattern')
                  REGEXP_REPLACE(col, 'pattern', 'replacement')
                  REGEXP_SUBSTR(col, 'pattern')
PARSE JSON      : PARSE_JSON(col):key::string
"""

AGGREGATION_FUNCTIONS = """
=== AGGREGATION (Snowflake) ===
SUM / AVG / MIN / MAX / COUNT(*) / COUNT(DISTINCT col)
LISTAGG         : LISTAGG(col, ',') WITHIN GROUP (ORDER BY col)
ARRAY_AGG       : ARRAY_AGG(col)
OBJECT_AGG      : OBJECT_AGG(key_col, value_col)
PERCENTILE      : PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)
APPROX COUNT    : APPROX_COUNT_DISTINCT(col)   -- faster for large tables

=== WINDOW FUNCTIONS (Snowflake) ===
ROW_NUMBER()    OVER (PARTITION BY x ORDER BY y)
RANK()          OVER (PARTITION BY x ORDER BY y)
DENSE_RANK()    OVER (...)
LAG(col, 1)     OVER (ORDER BY date_col)
LEAD(col, 1)    OVER (ORDER BY date_col)
SUM(col)        OVER (PARTITION BY x ORDER BY y ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
NTILE(4)        OVER (ORDER BY col)
RATIO_TO_REPORT(col) OVER (PARTITION BY x)

QUALIFY clause  : Use QUALIFY to filter window function results directly:
  SELECT *, ROW_NUMBER() OVER (PARTITION BY x ORDER BY y) AS rn
  FROM t
  QUALIFY rn = 1;
  This is SNOWFLAKE-SPECIFIC and much cleaner than a subquery.
"""

PAGINATION = """
=== PAGINATION (Snowflake) ===
TOP N (simple)  : SELECT TOP 100 * FROM t;
LIMIT / OFFSET  : SELECT * FROM t ORDER BY id LIMIT 100 OFFSET 0;
SAMPLE          : SELECT * FROM t SAMPLE (10 ROWS);     -- random 10 rows
                  SELECT * FROM t SAMPLE (1 PERCENT);   -- random 1%
NOTE: LIMIT requires no ORDER BY (unlike T-SQL OFFSET/FETCH).
NEVER USE: FETCH NEXT N ROWS ONLY — that is T-SQL syntax.
"""

SEMI_STRUCTURED = """
=== SEMI-STRUCTURED DATA (Snowflake) ===
VARIANT access  : col:key::string          -- dot notation
                  col['key']::integer       -- bracket notation
                  col:nested.key::float     -- nested path
FLATTEN         : SELECT f.value::string AS item
                  FROM table, LATERAL FLATTEN(INPUT => json_col) f
PARSE JSON      : PARSE_JSON('{"key": "value"}')
CHECK VALID     : TRY_PARSE_JSON(col) IS NOT NULL
ARRAY CONTAINS  : ARRAY_CONTAINS('value'::variant, array_col)
OBJECT KEYS     : OBJECT_KEYS(col)
"""

CTE_RULES = """
=== CTEs (Snowflake) ===
STANDARD CTE   : WITH cte AS ( SELECT ... ) SELECT * FROM cte;
MULTIPLE CTEs  : WITH a AS (...), b AS (...) SELECT ...
RECURSIVE CTE  : WITH RECURSIVE cte AS (
                     SELECT ...            -- anchor
                     UNION ALL
                     SELECT ...            -- recursive member
                 )
NOTE: Use RECURSIVE keyword explicitly (unlike T-SQL which omits it).
"""

IDENTIFIER_RULES = """
=== IDENTIFIERS (Snowflake) ===
- Snowflake is CASE-INSENSITIVE by default for unquoted identifiers.
- Unquoted identifiers are stored and compared in UPPERCASE internally.
- To preserve mixed case, wrap in double quotes: "MyColumn"
- Best practice: use UPPER_CASE for all identifiers and avoid quotes.
- Schema qualify when needed: DATABASE.SCHEMA.TABLE
- Never use square brackets [] — that is T-SQL syntax.
"""

FORBIDDEN_PATTERNS = """
=== NEVER USE IN SNOWFLAKE ===
T-SQL patterns       : GETDATE(), DATEPART(), EOMONTH(), TOP WITH TIES
T-SQL brackets       : [tablename]  (use "tablename" or just tablename)
T-SQL pagination     : OFFSET x ROWS FETCH NEXT y ROWS ONLY
PostgreSQL patterns  : NOW(), INTERVAL '1 day', EXTRACT() with PostgreSQL syntax
PostgreSQL casting   : col::date is supported in Snowflake — this is OK
PostgreSQL ILIKE     : ILIKE IS supported in Snowflake — this is OK too
MySQL syntax         : LIMIT without ORDER BY is fine in Snowflake
"""

SNOWFLAKE_SPECIFIC = """
=== SNOWFLAKE-SPECIFIC FEATURES ===
QUALIFY clause   : Filter window function results without subquery (preferred):
  SELECT *, RANK() OVER (ORDER BY revenue DESC) AS rnk FROM t QUALIFY rnk <= 10;
SAMPLE clause    : Random sampling without ORDER BY overhead:
  SELECT * FROM large_table SAMPLE (1000 ROWS);
TIME TRAVEL      : Query historical data:
  SELECT * FROM t AT (OFFSET => -60*5);         -- 5 minutes ago
  SELECT * FROM t AT (TIMESTAMP => '2024-01-01 00:00:00'::timestamp);
CLONE            : Zero-copy clone (metadata only): CREATE TABLE t2 CLONE t1;
"""


def build_system_prompt(
    schema_context_str: str,
    semantic_block: str = "",
    join_hints_block: str = "",
    conversation_block: str = "",
) -> str:
    """
    Builds the complete Snowflake SQL generation prompt.
    The EXACT COLUMN REGISTRY is injected first as sole source of truth.
    """
    from knowledge_base.schema_enforcement import SQL_GENERATION_RULES
    extra = ""
    if conversation_block:
        extra += f"\n{conversation_block}\n"
    if semantic_block:
        extra += f"\n{semantic_block}\n"
    if join_hints_block:
        extra += f"\n{join_hints_block}\n"

    return f"""You are a Snowflake SQL expert.
Your ONLY job is to write a single, correct, read-only SELECT statement that answers the user's question.

{schema_context_str}

{SQL_GENERATION_RULES}
Rule 11: Write ONLY valid Snowflake SQL syntax. Never use T-SQL or PostgreSQL-exclusive constructs.
Rule 12: Use UPPER_CASE for all table and column names (Snowflake default).
Rule 13: Use LIMIT for row caps, not TOP or FETCH NEXT.
Rule 14: When filtering window functions, use the QUALIFY clause instead of subqueries.

{DATE_TIME_FUNCTIONS}
{STRING_FUNCTIONS}
{AGGREGATION_FUNCTIONS}
{PAGINATION}
{CTE_RULES}
{SEMI_STRUCTURED}
{IDENTIFIER_RULES}
{SNOWFLAKE_SPECIFIC}
{FORBIDDEN_PATTERNS}
{extra}
Now write the SQL:"""


def build_schema_context_string(schema_context) -> str:
    """
    Returns the EXACT COLUMN REGISTRY + detailed schema block for Snowflake.
    """
    from knowledge_base.schema_enforcement import build_full_schema_context
    return build_full_schema_context(schema_context, "Snowflake")
