"""pgvector test environment provider.

Uses testcontainers to spin up a PostgreSQL container with pgvector for testing.
Registered as an entry point: datus.storage.vector.testing:postgresql
"""

import logging
from typing import Any, Dict, Optional

from datus.storage.testing import TestEnvConfig, VectorTestEnv

logger = logging.getLogger(__name__)


class PgvectorTestEnv(VectorTestEnv):
    """VectorTestEnv implementation using a testcontainers PostgreSQL + pgvector instance."""

    def __init__(self):
        self._container = None
        self._config: Optional[Dict[str, Any]] = None

    def setup(self) -> None:
        from testcontainers.postgres import PostgresContainer

        self._container = PostgresContainer(
            image="pgvector/pgvector:pg17",
            username="datus_test",
            password="datus_test",
            dbname="datus_test",
        )
        self._container.start()
        self._config = {
            "host": self._container.get_container_host_ip(),
            "port": int(self._container.get_exposed_port(5432)),
            "user": "datus_test",
            "password": "datus_test",
            "dbname": "datus_test",
        }
        logger.info("Started PostgreSQL test container for vector backend")

    def teardown(self) -> None:
        if self._container is not None:
            try:
                self._container.stop()
                logger.info("Stopped PostgreSQL test container for vector backend")
            except Exception:
                pass
            self._container = None
            self._config = None

    def clear_data(self, namespace: str) -> None:
        if self._config is None:
            return

        import psycopg

        conninfo = (
            f"host={self._config['host']} port={self._config['port']} "
            f"user={self._config['user']} password={self._config['password']} "
            f"dbname={self._config['dbname']}"
        )
        with psycopg.connect(conninfo, autocommit=True) as conn:
            if namespace:
                conn.execute(f"DROP SCHEMA IF EXISTS {namespace} CASCADE")
            else:
                # Drop all tables in public schema
                rows = conn.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                ).fetchall()
                for row in rows:
                    table_name = row[0] if not isinstance(row, dict) else row["tablename"]
                    conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")

    def get_config(self) -> TestEnvConfig:
        return TestEnvConfig(backend_type="postgresql", params=dict(self._config or {}))


def create_test_env() -> PgvectorTestEnv:
    """Factory function for entry point registration."""
    return PgvectorTestEnv()
