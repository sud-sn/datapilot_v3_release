"""
DataPilot – Join Graph
Builds a navigable graph from the FK relationships in SchemaContext
so the pipeline can automatically inject the optimal JOIN path when
the user's query spans multiple tables.

HOW IT WORKS:
  1. build_join_graph() reads schema.relationships and constructs a
     weighted undirected graph (each FK = one edge, bidirectional).
  2. find_join_path(start, end) runs Dijkstra's to find the shortest
     hop-count path between any two tables.
  3. format_join_path_hint() serialises the path into a SQL JOIN
     snippet that the LLM is instructed to use verbatim.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import heapq

from adapters.base import SchemaContext


@dataclass
class JoinEdge:
    from_table: str
    from_col: str
    to_table: str
    to_col: str

    def reverse(self) -> "JoinEdge":
        return JoinEdge(self.to_table, self.to_col, self.from_table, self.from_col)


@dataclass
class JoinGraph:
    """
    Undirected FK graph. adjacency[table] = list of directly connected edges.
    """
    adjacency: dict[str, list[JoinEdge]] = field(default_factory=dict)

    def add_edge(self, edge: JoinEdge) -> None:
        self.adjacency.setdefault(edge.from_table, []).append(edge)
        self.adjacency.setdefault(edge.to_table, []).append(edge.reverse())

    def find_join_path(
        self, start: str, end: str
    ) -> Optional[list[JoinEdge]]:
        """
        Dijkstra's shortest path by hop count.
        Returns ordered list of JoinEdge objects, or None if unreachable.
        """
        if start == end:
            return []
        if start not in self.adjacency or end not in self.adjacency:
            return None

        # (cost, table, path_so_far)
        heap: list[tuple[int, str, list[JoinEdge]]] = [(0, start, [])]
        visited: set[str] = set()

        while heap:
            cost, current, path = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            for edge in self.adjacency.get(current, []):
                if edge.to_table in visited:
                    continue
                new_path = path + [edge]
                if edge.to_table == end:
                    return new_path
                heapq.heappush(heap, (cost + 1, edge.to_table, new_path))

        return None

    def find_tables_for_concepts(
        self,
        concept_tables: list[str],
    ) -> Optional[str]:
        """
        Given a list of tables a query must touch, finds the minimal spanning
        join path and returns it as a SQL snippet.
        Works by iteratively joining each new table to the existing set.
        """
        if not concept_tables:
            return None
        if len(concept_tables) == 1:
            return f"FROM {concept_tables[0]}"

        anchor = concept_tables[0]
        accumulated_hint = f"FROM {anchor}"
        visited: set[str] = {anchor}

        for target in concept_tables[1:]:
            if target in visited:
                continue
            # Find the nearest already-visited table to target
            best_path: Optional[tuple[str, list[JoinEdge]]] = None
            best_len = 999
            for src in list(visited):
                p = self.find_join_path(src, target)
                if p is not None and len(p) < best_len:
                    best_len = len(p)
                    best_path = (src, p)

            if best_path:
                _, path = best_path
                for edge in path:
                    if edge.to_table not in visited:
                        visited.add(edge.to_table)
                        a_from = edge.from_table[0].lower()
                        a_to   = edge.to_table[0].lower()
                        accumulated_hint += (
                            f"\nJOIN {edge.to_table} {a_to}  "
                            f"ON {a_from}.{edge.from_col} = {a_to}.{edge.to_col}"
                        )

        return accumulated_hint


# ── Factory ───────────────────────────────────────────────────────────────────

def build_join_graph(schema: SchemaContext) -> JoinGraph:
    """
    Builds a JoinGraph from all FK relationships discovered in the schema.
    SchemaContext.relationships is: {table -> [(fk_col, ref_table, ref_col)]}
    """
    graph = JoinGraph()
    for from_table, rels in schema.relationships.items():
        for fk_col, ref_table, ref_col in rels:
            graph.add_edge(JoinEdge(
                from_table=from_table,
                from_col=fk_col,
                to_table=ref_table,
                to_col=ref_col,
            ))
    return graph


def build_join_hints_block(
    user_query: str,
    schema: SchemaContext,
    concept_tables: list[str],
) -> str:
    """
    If the query spans multiple tables (declared via semantic concepts),
    resolves the join path and returns a prompt block.
    Returns empty string if fewer than 2 tables or no path found.
    """
    if len(concept_tables) < 2:
        return ""

    graph = build_join_graph(schema)
    hint = graph.find_tables_for_concepts(concept_tables)
    if not hint:
        return ""

    return (
        "=== JOIN PATH (use this exact join structure) ===\n"
        f"{hint}\n\n"
        "Always alias tables as shown above and qualify all column references.\n"
    )
