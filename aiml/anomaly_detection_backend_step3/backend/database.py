"""
database.py
===========
SQLAlchemy 2.0 engine/session wiring for PostgreSQL, plus the
`get_db()` FastAPI dependency used by every router.

Table definitions live in `models.py` (Step 1); this module only owns
the *connection*, not the schema.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import settings

# `pool_pre_ping` avoids the classic "server closed the connection
# unexpectedly" error after DB idle timeouts / restarts, which matters
# for a long-lived FastAPI process sitting behind a background worker.
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
