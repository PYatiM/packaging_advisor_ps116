"""Pytest session isolation for SQLAlchemy SQLite database."""

import os
import sys
import tempfile

import pytest
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import backend.db.session as db_session
from backend.db.models import Base
from backend.db.seed import seed_initial_data
from backend.main import app


@pytest.fixture(scope="session", autouse=True)
def isolated_test_database():
    """Bind FastAPI get_db and startup seeding to an isolated temporary SQLite file."""
    tmp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmp_dir.name, "test_packaging_advisor.db")
    test_url = f"sqlite:///{db_path}"

    test_engine = db_session.create_db_engine(test_url)
    TestSessionLocal = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )

    Base.metadata.create_all(bind=test_engine)
    db = TestSessionLocal()
    try:
        seed_initial_data(db)
    finally:
        db.close()

    def _override_get_db():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[db_session.get_db] = _override_get_db
    yield TestSessionLocal
    app.dependency_overrides.clear()
    test_engine.dispose()
    tmp_dir.cleanup()
