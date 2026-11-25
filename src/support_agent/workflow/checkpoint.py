"""SQLite-backed LangGraph checkpoints.

Workflow state is plain JSON-compatible data (see ``conversation.py`` and
``review.py``), so the serializer runs in strict mode: it refuses to revive
arbitrary Python objects from the checkpoint file, and pickle is disabled.
Credentials (signed sessions, PINs) are never part of workflow state.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


def strict_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=None)


def connect_checkpointer(path: str | Path) -> SqliteSaver:
    """Open (and create if needed) a checkpoint database. Close it with ``saver.conn.close()``."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    saver = SqliteSaver(conn, serde=strict_serializer())
    saver.setup()
    return saver


@contextmanager
def open_checkpointer(path: str | Path) -> Iterator[SqliteSaver]:
    saver = connect_checkpointer(path)
    try:
        yield saver
    finally:
        saver.conn.close()


def list_thread_ids(saver: BaseCheckpointSaver, *, prefix: str | None = None) -> list[str]:
    """Distinct thread ids, most recently updated first."""
    seen: dict[str, None] = {}
    for item in saver.list(None):
        thread_id = item.config["configurable"]["thread_id"]
        if prefix is None or thread_id.startswith(prefix):
            seen.setdefault(thread_id, None)
    return list(seen)
