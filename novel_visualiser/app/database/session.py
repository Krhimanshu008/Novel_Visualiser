"""
Novel Visualiser — Database Session Manager

Provides engine initialization, scoped session management, and a
sessionmaker factory for use with FastAPI dependency injection.

Usage::

    from app.database.session import init_db, get_session

    init_db()                          # creates tables on first run

    with get_session() as session:     # context-managed session
        session.add(Project(...))
        session.commit()
"""

from __future__ import annotations

import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.database.models import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db(database_path: str | None = None) -> Engine:
    """Create the SQLAlchemy engine, enable WAL mode, and run ``create_all``.

    Args:
        database_path: Explicit path to the SQLite file.  Falls back to
            ``Settings.db.database_path`` when *None*.

    Returns:
        The newly created (or cached) :class:`~sqlalchemy.engine.Engine`.
    """
    global _engine, _SessionFactory

    if _engine is not None:
        logger.debug("Database engine already initialised — returning cached instance.")
        return _engine

    settings = get_settings()
    db_path = database_path or settings.db.database_path

    # Ensure the parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    url = f"sqlite:///{db_path}"
    _engine = create_engine(
        url,
        echo=settings.db.echo_sql,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )

    # Enable WAL journal mode for better concurrent-read performance.
    @event.listens_for(_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

    Base.metadata.create_all(bind=_engine)
    logger.info("Database tables created/verified at %s", db_path)

    # Lightweight SQLite auto-migration for newly added columns
    try:
        with _engine.connect() as conn:
            cursor = conn.connection.cursor()
            cursor.execute("PRAGMA table_info(arcs);")
            existing_cols = {row[1] for row in cursor.fetchall()}
            new_cols = [
                ("parent_arc_id", "VARCHAR(36)"),
                ("arc_type", "VARCHAR(16) DEFAULT 'major'"),
                ("parallel_thread_id", "VARCHAR(64)"),
                ("timeline_day_start", "INTEGER"),
                ("timeline_day_end", "INTEGER"),
                ("in_universe_weather", "VARCHAR(128)"),
            ]
            for col_name, col_def in new_cols:
                if col_name not in existing_cols:
                    try:
                        cursor.execute(f"ALTER TABLE arcs ADD COLUMN {col_name} {col_def};")
                    except Exception as exc:
                        logger.debug("Column %s add note: %s", col_name, exc)
            cursor.close()
            conn.commit()
    except Exception as exc:
        logger.debug("Auto-migration note: %s", exc)

    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)

    return _engine


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional database session (context-manager pattern).

    The session is committed on clean exit and rolled back on exception.

    Example::

        with get_session() as session:
            project = session.get(Project, project_id)
    """
    if _SessionFactory is None:
        raise RuntimeError(
            "Database not initialised. Call init_db() before requesting a session."
        )

    session: Session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database sessions."""
    with get_session() as session:
        yield session


def get_session_factory() -> sessionmaker[Session]:
    """Return the :class:`sessionmaker` for FastAPI dependency injection.

    Typical FastAPI usage::

        from fastapi import Depends
        from sqlalchemy.orm import Session

        def get_db() -> Generator[Session, None, None]:
            factory = get_session_factory()
            db = factory()
            try:
                yield db
            finally:
                db.close()

        @app.get("/projects")
        def list_projects(db: Session = Depends(get_db)):
            return db.query(Project).all()
    """
    if _SessionFactory is None:
        raise RuntimeError(
            "Database not initialised. Call init_db() before requesting the session factory."
        )
    return _SessionFactory


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_engine() -> Engine:
    """Return the current engine, raising if :func:`init_db` hasn't been called."""
    if _engine is None:
        raise RuntimeError(
            "Database not initialised. Call init_db() before requesting the engine."
        )
    return _engine


def backup_to_drive(db_path: str, remote_path: str) -> None:
    """Create a backup of the SQLite database file.

    This is a stub that performs a local file copy and logs the intent to
    upload to Google Drive.  A full implementation should use the Google
    Drive API or ``rclone`` to sync the backup.

    Args:
        db_path: Absolute path to the local SQLite database file.
        remote_path: Target path on Google Drive (or other remote).
    """
    backup_file = Path(db_path).with_suffix(".db.bak")
    try:
        shutil.copy2(db_path, backup_file)
        logger.info(
            "Local backup created: %s  (remote upload to '%s' not yet implemented)",
            backup_file,
            remote_path,
        )
    except FileNotFoundError:
        logger.error("Cannot backup — database file not found: %s", db_path)
    except OSError as exc:
        logger.error("Backup failed: %s", exc)
