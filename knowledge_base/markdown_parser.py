"""
DataPilot – Markdown Parser & Knowledge Store Manager
Reads the generated (and user-edited) markdown knowledge base file
and converts it into SemanticConcept objects that the SQL pipeline
understands.

Also manages reading/writing KB files to the knowledge_store directory.
"""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from knowledge_base.semantic_layer import SemanticConcept

logger = logging.getLogger(__name__)


# ── Knowledge Store Manager ───────────────────────────────────────────────────

@dataclass
class KBMetadata:
    source_id:    str
    domain:       str
    generated_at: str
    table_count:  int
    version:      int = 1


class KnowledgeStore:
    """
    Manages the directory of markdown knowledge base files.
    One .md file per source, one .meta.json file per source.
    """

    def __init__(self, store_path: str) -> None:
        self._path = Path(store_path)
        self._path.mkdir(parents=True, exist_ok=True)

    def _md_path(self, source_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", source_id)
        return self._path / f"{safe}.md"

    def _meta_path(self, source_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", source_id)
        return self._path / f"{safe}.meta.json"

    def exists(self, source_id: str) -> bool:
        return self._md_path(source_id).exists()

    def save(self, source_id: str, markdown: str, metadata: KBMetadata) -> None:
        self._md_path(source_id).write_text(markdown, encoding="utf-8")
        meta_dict = {
            "source_id":    metadata.source_id,
            "domain":       metadata.domain,
            "generated_at": metadata.generated_at,
            "table_count":  metadata.table_count,
            "version":      metadata.version,
        }
        self._meta_path(source_id).write_text(
            json.dumps(meta_dict, indent=2), encoding="utf-8"
        )
        logger.info("KB saved: %s (%d chars)", source_id, len(markdown))

    def load_markdown(self, source_id: str) -> Optional[str]:
        p = self._md_path(source_id)
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8")

    def load_metadata(self, source_id: str) -> Optional[KBMetadata]:
        p = self._meta_path(source_id)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return KBMetadata(**data)

    def update_markdown(self, source_id: str, markdown: str) -> None:
        """Save user-edited markdown without changing metadata."""
        if not self._md_path(source_id).exists():
            raise FileNotFoundError(f"No KB found for source: {source_id}")
        self._md_path(source_id).write_text(markdown, encoding="utf-8")
        logger.info("KB updated by user: %s", source_id)

    def delete(self, source_id: str) -> None:
        self._md_path(source_id).unlink(missing_ok=True)
        self._meta_path(source_id).unlink(missing_ok=True)

    def list_sources(self) -> list[str]:
        return [
            p.stem
            for p in self._path.glob("*.md")
            if not p.stem.startswith(".")
        ]

    def parse_concepts(self, source_id: str) -> list[SemanticConcept]:
        """
        Parse the KB markdown for a source and return SemanticConcept objects
        ready for injection into the SQL pipeline's concept_registry.
        """
        md = self.load_markdown(source_id)
        if not md:
            return []
        return parse_markdown_to_concepts(md)


# ── Markdown Parser ────────────────────────────────────────────────────────────

def parse_markdown_to_concepts(markdown: str) -> list[SemanticConcept]:
    """
    Parse the Business Vocabulary section of a knowledge base markdown
    into a list of SemanticConcept objects.

    Expected table format in the markdown:
    | Term | Plain English Meaning | SQL Expression | Tables Needed | Filters to Apply |
    |------|-----------------------|----------------|---------------|-----------------|
    | Revenue | Total sales value | `SUM(total_amount)` | orders | `status IN ('paid','shipped')` |
    """
    concepts: list[SemanticConcept] = []

    # Find the Business Vocabulary section
    vocab_match = re.search(
        r"## Business Vocabulary\s*\n(.*?)(?=\n## |\Z)",
        markdown,
        flags=re.DOTALL,
    )
    if not vocab_match:
        logger.debug("No Business Vocabulary section found in markdown")
        return []

    vocab_section = vocab_match.group(1)

    # Parse the markdown table rows (skip header and separator lines)
    for line in vocab_section.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[-| ]+\|$", line):
            continue  # separator row

        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 4:
            continue

        term    = _clean_cell(cols[0])
        meaning = _clean_cell(cols[1]) if len(cols) > 1 else ""
        sql_exp = _clean_cell(cols[2]) if len(cols) > 2 else ""
        tables  = _clean_cell(cols[3]) if len(cols) > 3 else ""
        filters = _clean_cell(cols[4]) if len(cols) > 4 else ""

        # Skip header row and empty/placeholder rows
        if not term or term.lower() in ("term", "[business term]", "---"):
            continue
        if "placeholder" in term.lower() or term.startswith("["):
            continue
        if not sql_exp or sql_exp.startswith("["):
            continue

        # Build keywords from the term itself (lowercase words)
        keywords = _build_keywords(term)

        # Parse table list
        table_list = [
            t.strip()
            for t in re.split(r"[,\s]+", tables)
            if t.strip() and not t.startswith("[")
        ]

        # Parse filters into a list
        filter_list = _parse_filters(filters)

        concept = SemanticConcept(
            name             = term.lower().replace(" ", "_"),
            keywords         = keywords,
            description      = meaning or term,
            primary_table    = table_list[0] if table_list else "",
            value_column     = _extract_value_column(sql_exp),
            join_path        = table_list,
            required_filters = filter_list,
            group_by_hints   = [],
            example_query    = None,
        )
        concepts.append(concept)
        logger.debug("Parsed concept: %s (keywords: %s)", concept.name, concept.keywords)

    logger.info("Parsed %d concepts from vocabulary section", len(concepts))
    return concepts


def extract_table_overview(markdown: str) -> dict[str, str]:
    """
    Extract table overview descriptions from the markdown.
    Returns {table_name: overview_text}
    """
    overviews: dict[str, str] = {}
    for match in re.finditer(
        r"## Table:\s*(\w+)\s*\n.*?\*\*Overview:\*\*\s*\n(.*?)(?=\n\*\*|\Z)",
        markdown,
        flags=re.DOTALL,
    ):
        table_name = match.group(1).strip()
        overview   = match.group(2).strip()
        overviews[table_name] = overview
    return overviews


def extract_always_exclude_filters(markdown: str) -> dict[str, list[str]]:
    """
    Extract Always Exclude filters per table from the markdown.
    Returns {table_name: [filter_condition, ...]}
    """
    result: dict[str, list[str]] = {}
    for match in re.finditer(
        r"## Table:\s*(\w+).*?\*\*Always Exclude:\*\*\s*\n(.*?)(?=\n\*\*|\n##|\Z)",
        markdown,
        flags=re.DOTALL,
    ):
        table_name = match.group(1).strip()
        block      = match.group(2).strip()
        filters    = []
        for line in block.split("\n"):
            line = line.strip().lstrip("- ").strip()
            if line and not line.lower().startswith("none"):
                # Remove surrounding backticks
                line = line.strip("`")
                if line:
                    filters.append(line)
        if filters:
            result[table_name] = filters
    return result


# ── Helper functions ──────────────────────────────────────────────────────────

def _clean_cell(text: str) -> str:
    """Remove backticks and extra whitespace from a table cell."""
    return text.strip().strip("`").strip()


def _build_keywords(term: str) -> list[str]:
    """
    Build a keyword list from a business term.
    Includes the full term plus common abbreviations/synonyms.
    """
    term_lower = term.lower()
    keywords   = [term_lower]

    # Add individual words if multi-word term
    words = term_lower.split()
    if len(words) > 1:
        keywords.extend(words)

    # Common synonym mappings
    synonyms = {
        "revenue":        ["sales", "income", "turnover", "gmv"],
        "mrr":            ["monthly recurring", "subscription revenue"],
        "arr":            ["annual recurring"],
        "active users":   ["mau", "dau", "engaged users"],
        "churn":          ["churned", "lost customer", "attrition"],
        "gross profit":   ["gp", "gross margin"],
        "net profit":     ["net income", "bottom line"],
        "customer count": ["total customers", "number of customers"],
        "order count":    ["total orders", "number of orders"],
        "aov":            ["average order value", "average basket"],
    }
    for key, syns in synonyms.items():
        if key in term_lower or term_lower in syns:
            keywords.extend(syns)
            keywords.append(key)

    return list(dict.fromkeys(keywords))  # deduplicate preserving order


def _extract_value_column(sql_expr: str) -> str:
    """
    Extract the column name from a SQL expression like SUM(total_amount).
    Falls back to the full expression if parsing fails.
    """
    match = re.search(r"\w+\((\w+)\)", sql_expr)
    if match:
        return match.group(1)
    # If it's a plain column reference
    match2 = re.match(r"^\w+$", sql_expr.strip())
    if match2:
        return sql_expr.strip()
    return sql_expr


def _parse_filters(filters_text: str) -> list[str]:
    """
    Parse a filters cell into individual SQL conditions.
    Handles comma-separated conditions and AND-separated conditions.
    """
    if not filters_text or filters_text.startswith("["):
        return []
    # Split on AND (case-insensitive) or comma, but only at top level
    parts = re.split(r"\s+AND\s+", filters_text, flags=re.IGNORECASE)
    result = []
    for part in parts:
        part = part.strip().strip("`").strip()
        if part and not part.lower().startswith("none"):
            result.append(part)
    return result


# ── Module-level store instance (initialised by main.py) ─────────────────────

_store: Optional[KnowledgeStore] = None


def init_store(store_path: str) -> KnowledgeStore:
    """Initialise the global knowledge store. Called once at startup."""
    global _store
    _store = KnowledgeStore(store_path)
    return _store


def get_store() -> KnowledgeStore:
    """Return the global knowledge store instance."""
    if _store is None:
        raise RuntimeError(
            "Knowledge store not initialised. Call init_store() at startup."
        )
    return _store
