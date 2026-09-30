"""Database engine configuration and FastAPI session dependency."""

import os
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.core.config import settings
from backend.db.models import Base


def _prepare_sqlite_dir(database_url: str) -> None:
    if database_url.startswith("sqlite:///"):
        raw_path = database_url[len("sqlite:///") :]
        if raw_path and raw_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(raw_path))
            if parent:
                os.makedirs(parent, exist_ok=True)


def create_db_engine(database_url: str) -> Engine:
    _prepare_sqlite_dir(database_url)
    is_sqlite = database_url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    eng = create_engine(database_url, connect_args=connect_args, future=True)

    if is_sqlite:
        @event.listens_for(eng, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return eng


engine: Engine = create_db_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

_SCHEMA_INITIALIZED = False


def init_db(target_engine: Engine | None = None) -> None:
    """Create all tables if they do not exist yet."""
    global _SCHEMA_INITIALIZED
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)
    if eng is engine:
        _SCHEMA_INITIALIZED = True


def reset_schema_flag() -> None:
    global _SCHEMA_INITIALIZED
    _SCHEMA_INITIALIZED = False


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a transactional SQLAlchemy Session."""
    if not _SCHEMA_INITIALIZED:
        init_db(engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
