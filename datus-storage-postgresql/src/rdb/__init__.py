"""PostgreSQL RDB backend adapter for datus-agent."""

from rdb.backend import PostgresRdbBackend


def register():
    """Register the PostgreSQL RDB backend with the datus registry."""
    from datus.storage.rdb.registry import RdbRegistry

    RdbRegistry.register("postgresql", PostgresRdbBackend)


__all__ = ["PostgresRdbBackend", "register"]
