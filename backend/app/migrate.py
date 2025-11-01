"""Command-line entry point for running database migrations."""
from __future__ import annotations

import logging

from .db import engine, run_role_migrations, run_schema_migrations


def main() -> None:
    """Execute schema and role migrations for the application database."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logging.info("Running schema migrations")
    run_schema_migrations(engine)

    logging.info("Ensuring role privileges")
    run_role_migrations(engine)

    logging.info("Database migration completed")


if __name__ == "__main__":
    main()
