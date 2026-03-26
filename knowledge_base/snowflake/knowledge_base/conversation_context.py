"""
DataPilot – Conversation Context
Maintains a lightweight per-session history so the LLM can handle
follow-up questions ("now filter by last month", "break that down by region").

HOW IT WORKS:
  1. Each API session gets a ConversationContext object (keyed by session_id).
  2. After every successful pipeline run, the turn is recorded.
  3. On the next request, the last N turns are serialised and injected
     into the system prompt BEFORE the user query.

The session_id should be prefixed with the tenant/source ID so tenants
never see each other's histories.

  e.g. session_id = "postgresql:user-abc-session-1"
       session_id = "azure_sql:user-xyz-session-1"
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationTurn:
    user_query: str
    generated_sql: str
    result_summary: str       # e.g. "returned 12 rows: revenue by month"
    tables_touched: list[str] # which tables the SQL touched
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationContext:
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    max_turns: int = 5        # rolling window size

    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def format_history_block(self) -> str:
        """
        Returns a prompt block summarising recent turns so the LLM understands
        what has already been asked and which tables are in play.
        """
        if not self.turns:
            return ""
        lines = ["=== CONVERSATION HISTORY (most recent last) ==="]
        for i, turn in enumerate(self.turns, 1):
            lines.append(f"\nTurn {i}:")
            lines.append(f"  User asked : {turn.user_query}")
            lines.append(f"  SQL used   : {turn.generated_sql.strip()}")
            lines.append(f"  Result     : {turn.result_summary}")
        lines.append(
            "\nIf the new question is a follow-up, build on the SQL above. "
            "Re-use the same aliases and WHERE filters unless the user changes them explicitly."
        )
        return "\n".join(lines)

    def get_active_tables(self) -> list[str]:
        """Tables touched in the most recent turn — useful for join-path seeding."""
        if not self.turns:
            return []
        return self.turns[-1].tables_touched


# ── Session store ─────────────────────────────────────────────────────────────

_sessions: dict[str, ConversationContext] = {}
_SESSION_TTL = 3600   # 1 hour of inactivity → session dropped
_session_last_used: dict[str, float] = {}


def get_session(session_id: str) -> ConversationContext:
    """Return (or create) the conversation context for a session ID."""
    _evict_stale()
    if session_id not in _sessions:
        _sessions[session_id] = ConversationContext(session_id=session_id)
    _session_last_used[session_id] = time.time()
    return _sessions[session_id]


def record_turn(
    session_id: str,
    user_query: str,
    generated_sql: str,
    row_count: int,
    tables_touched: Optional[list[str]] = None,
) -> None:
    """Append a completed turn to the session history."""
    ctx = get_session(session_id)
    summary = f"returned {row_count} row{'s' if row_count != 1 else ''}"
    if tables_touched:
        summary += f" (tables: {', '.join(tables_touched)})"

    ctx.add_turn(ConversationTurn(
        user_query=user_query,
        generated_sql=generated_sql,
        result_summary=summary,
        tables_touched=tables_touched or [],
    ))


def _evict_stale() -> None:
    """Remove sessions that haven't been used within SESSION_TTL seconds."""
    now = time.time()
    stale = [sid for sid, t in _session_last_used.items() if now - t > _SESSION_TTL]
    for sid in stale:
        _sessions.pop(sid, None)
        _session_last_used.pop(sid, None)
