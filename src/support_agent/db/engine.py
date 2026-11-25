"""Engine, session factory, schema creation, and the transaction helper."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from support_agent.db.models import Base

logger = logging.getLogger(__name__)


def create_db_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create an engine. SQLite connections get foreign keys, WAL, and proper BEGIN handling."""
    url = make_url(database_url)
    connect_args: dict[str, Any] = {}
    is_sqlite = url.get_backend_name() == "sqlite"
    file_based = is_sqlite and url.database not in (None, "", ":memory:")

    if is_sqlite:
        # Allow use from worker threads (FastAPI/Streamlit in later phases).
        connect_args["check_same_thread"] = False
        if file_based:
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)  # type: ignore[arg-type]

    engine = create_engine(url, echo=echo, connect_args=connect_args)
    if is_sqlite:
        _configure_sqlite(engine, file_based=file_based)
    return engine


def _configure_sqlite(engine: Engine, *, file_based: bool) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        # Hand transaction control to SQLAlchemy so BEGIN/SAVEPOINT behave correctly.
        # See "Serializable isolation / Savepoints" in the SQLAlchemy SQLite docs.
        dbapi_connection.isolation_level = None
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA busy_timeout = 5000")
        if file_based:
            cursor.execute("PRAGMA journal_mode = WAL")
        cursor.close()

    @event.listens_for(engine, "begin")
    def _on_begin(connection: Any) -> None:
        # IMMEDIATE takes the write lock up front. Almost every CRM transaction writes
        # (at least an audit row), and a deferred BEGIN would make two concurrent
        # requests fail with "database is locked" when both try to upgrade from
        # reading to writing. With IMMEDIATE the second one waits (busy_timeout) and
        # then sees the first one's committed result, so decisions stay atomic.
        connection.exec_driver_sql("BEGIN IMMEDIATE")


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False lets callers read returned objects after commit.
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def transaction(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Unit of work: commit on success, roll back on any exception, always close."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine, *, reset: bool = False) -> None:
    """Create all tables. ``reset=True`` drops existing tables first (local dev only)."""
    if reset:
        logger.warning("Dropping all tables", extra={"db": engine.url.render_as_string()})
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
