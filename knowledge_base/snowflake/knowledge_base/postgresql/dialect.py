"""
DataPilot – PostgreSQL Knowledge Base
This is the authoritative dialect reference injected into the LLM system prompt.
Keep it dense but structured — every line burns tokens.
"""
from __future__ import annotations

# ── Date / Time Functions ─────────────────────────────────────────────────────
DATE_TIME_FUNCTIONS = """
=== DATE & TIME (PostgreSQL) ===
CURRENT TIMESTAMP : NOW()  or  CURRENT_TIMESTAMP
TODAY             : CURRENT_DATE
INTERVALS         : NOW() - INTERVAL '30 days'   (use single quotes, space before unit)
                    NOW() - INTERVAL '1 month'
                    NOW() - INTERVAL '1 year'
DATE TRUNC        : DATE_TRUNC('month', col)  → first day of month
                    DATE_TRUNC('week',  col)  → Monday of week
                    DATE_TRUNC('day',   col)  → midnight
EXTRACT           : EXTRACT(YEAR  FROM col)
                    EXTRACT(MONTH FROM col)
                    EXTRACT(DOW   FROM col)   → 0=Sunday .. 6=Saturday
AGE               : AGE(timestamp1, timestamp2)  → interval
TO_CHAR           : TO_CHAR(col, 'YYYY-MM-DD')
                    TO_CHAR(col, 'Mon YYYY')
TO_DATE           : TO_DATE('2024-01-31', 'YYYY-MM-DD')
CAST              : col::date   col::timestamp   col::time
AT TIME ZONE      : col AT TIME ZONE 'UTC'
COMMON RANGES:
  Last 30 days    : WHERE col >= NOW() - INTERVAL '30 days'
  This month      : WHERE DATE_TRUNC('month', col) = DATE_TRUNC('month', NOW())
  Last month      : WHERE col >= DATE_TRUNC('month', NOW() - INTERVAL '1 month')
                      AND col <  DATE_TRUNC('month', NOW())
  This year       : WHERE EXTRACT(YEAR FROM col) = EXTRACT(YEAR FROM NOW())
NEVER USE: DATEADD(), DATEDIFF(), GETDATE(), EOMONTH() — these are T-SQL only.
"""

# ── String Functions ──────────────────────────────────────────────────────────
STRING_FUNCTIONS = """
=== STRINGS (PostgreSQL) ===
CONCAT        : col1 || col2   or   CONCAT(col1, col2)
CASE-INSENSITIVE LIKE : col ILIKE '%pattern%'   (preferred over LOWER+LIKE)
LOWER / UPPER : LOWER(col)   UPPER(col)
TRIM          : TRIM(col)   LTRIM(col)   RTRIM(col)
SUBSTRING     : SUBSTRING(col FROM 1 FOR 5)   or   col[1:5]
POSITION      : POSITION('str' IN col)
LENGTH        : LENGTH(col)   or   CHAR_LENGTH(col)
REPLACE       : REPLACE(col, 'old', 'new')
REGEXP        : col ~ 'pattern'      (case-sensitive match)
               col ~* 'pattern'     (case-insensitive)
               REGEXP_MATCHES(col, 'pattern')
               REGEXP_REPLACE(col, 'pattern', 'replacement')
SPLIT         : SPLIT_PART(col, ',', 1)
FORMAT        : FORMAT('%s - %s', col1, col2)
COALESCE      : COALESCE(col, 'default')
NULLIF        : NULLIF(col, '')   — returns NULL if col = ''
NEVER USE: CHARINDEX(), PATINDEX(), LEN() — these are T-SQL only.
"""

# ── Aggregation & Window Functions ────────────────────────────────────────────
AGGREGATION_FUNCTIONS = """
=== AGGREGATION (PostgreSQL) ===
SUM / AVG / MIN / MAX / COUNT(*) / COUNT(DISTINCT col)
ARRAY_AGG     : ARRAY_AGG(col ORDER BY col)
STRING_AGG    : STRING_AGG(col, ', ' ORDER BY col)
PERCENTILE    : PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)
MODE          : MODE() WITHIN GROUP (ORDER BY col)
JSON_AGG      : JSON_AGG(row_to_json(t))
FILTER        : SUM(amount) FILTER (WHERE status = 'completed')

=== WINDOW FUNCTIONS (PostgreSQL) ===
ROW_NUMBER()  OVER (PARTITION BY x ORDER BY y)
RANK()        OVER (PARTITION BY x ORDER BY y)
DENSE_RANK()  OVER (...)
LAG(col, 1)   OVER (ORDER BY date_col)   — previous row value
LEAD(col, 1)  OVER (ORDER BY date_col)   — next row value
SUM(col)      OVER (PARTITION BY x ORDER BY y ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
NTILE(4)      OVER (ORDER BY col)        — quartile bucketing
"""

# ── Pagination ────────────────────────────────────────────────────────────────
PAGINATION = """
=== PAGINATION (PostgreSQL) ===
LIMIT / OFFSET: SELECT ... FROM t ORDER BY id LIMIT 100 OFFSET 0;
CURSOR-BASED  : WHERE id > :last_id ORDER BY id LIMIT 100   ← preferred for large tables
NEVER USE: TOP, FETCH FIRST (without OFFSET) as standalone — use LIMIT.
"""

# ── JSON Functions ────────────────────────────────────────────────────────────
JSON_FUNCTIONS = """
=== JSON / JSONB (PostgreSQL) ===
EXTRACT VALUE : col->'key'         → JSON value (keeps JSON type)
               col->>'key'        → TEXT value  (extracts as string)
               col#>>'{a,b}'      → nested path as text
CONTAINS      : col @> '{"key":"value"}'::jsonb
EXISTS        : col ? 'key'
JSON_BUILD    : jsonb_build_object('key', val)
JSONB_EACH    : SELECT * FROM jsonb_each(col)
ARRAY ACCESS  : col->0   (first element of JSON array)
"""

# ── CTEs and Subqueries ───────────────────────────────────────────────────────
CTE_RULES = """
=== CTEs (PostgreSQL) ===
STANDARD CTE  : WITH cte_name AS ( SELECT ... ) SELECT * FROM cte_name;
RECURSIVE CTE : WITH RECURSIVE cte AS (
                    SELECT ...   -- anchor
                    UNION ALL
                    SELECT ...   -- recursive member
                )
MULTIPLE CTEs : WITH a AS (...), b AS (...) SELECT ...
MATERIALIZED  : WITH m AS MATERIALIZED (...) — force materialisation
"""

# ── Identifier Quoting ────────────────────────────────────────────────────────
IDENTIFIER_RULES = """
=== IDENTIFIERS (PostgreSQL) ===
- Unquoted identifiers are case-FOLDED to lowercase.
- Quote identifiers with double quotes ONLY if they contain spaces or capitals.
- NEVER use square brackets [like_this] — that is T-SQL syntax.
- Qualify columns when joining: alias.column_name, not just column_name.
"""

# ── Common Mistakes to Avoid ─────────────────────────────────────────────────
FORBIDDEN_PATTERNS = """
=== NEVER USE IN POSTGRESQL ===
T-SQL date functions : DATEADD, DATEDIFF, GETDATE, EOMONTH, DATEFROMPARTS
T-SQL string fns     : LEN(), CHARINDEX(), PATINDEX(), STUFF()
T-SQL pagination     : TOP N  (use LIMIT instead)
T-SQL brackets       : [tablename]  (use "tablename" or no quotes)
T-SQL boolean        : 1=1 for TRUE  (use TRUE/FALSE)
T-SQL ISNULL()       : use COALESCE() instead
T-SQL CONVERT()      : use CAST() or :: casting  e.g. col::date
"""

# ── Full System Prompt ────────────────────────────────────────────────────────
def build_system_prompt(
    schema_context_str: str,
    semantic_block: str = "",
    join_hints_block: str = "",
    conversation_block: str = "",
) -> str:
    """
    Builds the complete PostgreSQL SQL generation prompt.
    The EXACT COLUMN REGISTRY is injected first — it is the sole source
    of truth for all table and column names. The LLM must not use any
    identifier not listed there.
    """
    from knowledge_base.schema_enforcement import SQL_GENERATION_RULES
    extra = ""
    if conversation_block:
        extra += f"\n{conversation_block}\n"
    if semantic_block:
        extra += f"\n{semantic_block}\n"
    if join_hints_block:
        extra += f"\n{join_hints_block}\n"

    return f"""You are a PostgreSQL 15 SQL expert. Your ONLY job is to write a single,
correct, read-only SELECT statement that answers the user's question.

{schema_context_str}

{SQL_GENERATION_RULES}
Rule 11: Write ONLY valid PostgreSQL 15 syntax. Never use T-SQL or Snowflake constructs.

{DATE_TIME_FUNCTIONS}
{STRING_FUNCTIONS}
{AGGREGATION_FUNCTIONS}
{PAGINATION}
{CTE_RULES}
{IDENTIFIER_RULES}
{FORBIDDEN_PATTERNS}
{extra}
Now write the SQL:"""


def build_schema_context_string(schema_context) -> str:
    """
    Returns the EXACT COLUMN REGISTRY + detailed schema block.
    The registry enumerates every valid column name so the LLM
    cannot substitute names from its training memory.
    """
    from knowledge_base.schema_enforcement import build_full_schema_context
    return build_full_schema_context(schema_context, "PostgreSQL 15")
