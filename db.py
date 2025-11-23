from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "chat_history.db"


def _get_connection() -> sqlite3.Connection:
    """Return a cached SQLite connection."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = getattr(_get_connection, "_conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        setattr(_get_connection, "_conn", conn)
    return conn


def init_db() -> None:
    """Create the SQLite schema if it does not exist."""
    conn = _get_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rate_limits (
                session_id TEXT PRIMARY KEY,
                count INTEGER NOT NULL,
                reset_at TEXT NOT NULL
            )
            """
        )


def save_message(session_id: str, role: str, content: str, model: str, timestamp: str) -> None:
    """Persist a single chat message."""
    conn = _get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, model, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, model, timestamp),
        )
        # Ensure conversation metadata exists
        conn.execute(
            """
            INSERT INTO conversations (session_id, title, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET updated_at=excluded.updated_at
            """,
            (session_id, f"Session {session_id[-6:]}", timestamp),
        )


def load_messages(session_id: str) -> List[Dict[str, str]]:
    """Return ordered chat messages for a session."""
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT id, role, content, model, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def clear_messages(session_id: str) -> None:
    """Remove all messages for a session."""
    conn = _get_connection()
    with conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute(
            """
            INSERT INTO conversations (session_id, title, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at
            """,
            (session_id, "Untitled conversation", datetime.now(timezone.utc).isoformat()),
        )


def export_messages(session_id: str) -> List[Dict[str, str]]:
    """Return messages for export."""
    return load_messages(session_id)


def get_conversation_title(session_id: str) -> str:
    """Return the stored conversation title or a fallback."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT title FROM conversations WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row:
        return row["title"]
    default_title = f"Session {session_id[-6:]}"
    conn.execute(
        "INSERT OR IGNORE INTO conversations (session_id, title, updated_at) VALUES (?, ?, ?)",
        (session_id, default_title, datetime.now(timezone.utc).isoformat()),
    )
    return default_title


def set_conversation_title(session_id: str, title: str) -> None:
    """Persist a custom title for the conversation."""
    conn = _get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO conversations (session_id, title, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE
            SET title=excluded.title, updated_at=excluded.updated_at
            """,
            (session_id, title.strip() or f"Session {session_id[-6:]}", datetime.now(timezone.utc).isoformat()),
        )


def increment_rate_counter(
    session_id: str, *, limit: int, window_minutes: int
) -> Tuple[bool, int, int]:
    """
    Increment a persistent rate-limit counter.

    Returns:
        allowed (bool): Whether the call is permitted.
        remaining (int): Requests remaining until the limit.
        reset_seconds (int): Seconds until the counter resets.
    """
    now = datetime.now(timezone.utc)
    reset_at = now + timedelta(minutes=window_minutes)
    conn = _get_connection()
    row = conn.execute(
        "SELECT count, reset_at FROM rate_limits WHERE session_id = ?", (session_id,)
    ).fetchone()

    if row:
        stored_reset = datetime.fromisoformat(row["reset_at"])
        if stored_reset <= now:
            count = 1
            new_reset = reset_at
        else:
            count = row["count"] + 1
            new_reset = stored_reset
    else:
        count = 1
        new_reset = reset_at

    if row and row["count"] >= limit and datetime.fromisoformat(row["reset_at"]) > now:
        remaining = 0
        reset_seconds = int((datetime.fromisoformat(row["reset_at"]) - now).total_seconds())
        return False, remaining, max(reset_seconds, 0)

    with conn:
        conn.execute(
            """
            INSERT INTO rate_limits (session_id, count, reset_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE
            SET count=?, reset_at=?
            """,
            (session_id, count, new_reset.isoformat(), count, new_reset.isoformat()),
        )

    remaining = max(limit - count, 0)
    reset_seconds = int((new_reset - now).total_seconds())
    if count > limit:
        return False, 0, reset_seconds
    return True, remaining, reset_seconds

