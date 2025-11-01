"""Database utilities and session management for the FastAPI app."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Generator
from urllib.parse import quote_plus

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from .models import db_models


def build_database_url() -> str:
    """Construct a SQLAlchemy database URL from environment variables."""
    if url := os.getenv("DATABASE_URL"):
        return url

    user = os.getenv("POSTGRES_USER", "postgres")
    password = quote_plus(os.getenv("POSTGRES_PASSWORD", "postgres"))
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "postgres")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def create_engine_with_retry(url: str, retries: int = 5, delay: float = 2.0) -> Engine:
    """Create a SQLAlchemy engine, retrying until the database becomes available."""
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)

    for attempt in range(retries):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            break
        except OperationalError:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))

    return engine


def ensure_result_columns(engine: Engine) -> None:
    """Ensure optional columns exist when running against an old SQLite file."""
    with engine.connect() as connection:
        inspector = inspect(connection)
        if "results" not in inspector.get_table_names():
            return

        existing = {column["name"] for column in inspector.get_columns("results")}

        migrations = {
            "label": "ALTER TABLE results ADD COLUMN label VARCHAR",
            "description": "ALTER TABLE results ADD COLUMN description TEXT",
            "treatment": "ALTER TABLE results ADD COLUMN treatment TEXT",
            "prevention": "ALTER TABLE results ADD COLUMN prevention TEXT",
            "pathogen": "ALTER TABLE results ADD COLUMN pathogen TEXT",
            "created_at": "ALTER TABLE results ADD COLUMN created_at TIMESTAMP",
        }

        for column_name, ddl in migrations.items():
            if column_name not in existing:
                connection.execute(text(ddl))


DATABASE_URL = build_database_url()
engine = create_engine_with_retry(DATABASE_URL)
db_models.Base.metadata.create_all(bind=engine)
ensure_result_columns(engine)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


__all__ = [
    "DATABASE_URL",
    "engine",
    "get_session",
]
