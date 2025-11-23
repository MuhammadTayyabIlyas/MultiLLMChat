from __future__ import annotations

import json
import secrets
import string
from datetime import datetime, timezone
from typing import Iterable


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, friendly assistant that writes clear, structured answers."
)


def utc_now_iso() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def format_human_timestamp(timestamp: str) -> str:
    """Return a concise human-readable timestamp."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return timestamp
    return dt.strftime("%b %d, %Y • %H:%M UTC")


def secure_session_id(length: int = 16) -> str:
    """Generate a URL-safe session identifier."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def emit_chunks(text: str, on_token, chunk_size: int = 20) -> None:
    """
    Best-effort streaming helper for providers without native streaming.
    """
    if not on_token:
        return
    for idx in range(0, len(text), chunk_size):
        on_token(text[idx : idx + chunk_size])


def serialize_messages(messages: Iterable[dict]) -> str:
    """Return JSON string for export."""
    return json.dumps(
        list(messages),
        indent=2,
        ensure_ascii=False,
    )

