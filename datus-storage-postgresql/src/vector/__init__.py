"""pgvector Vector backend adapter for datus-agent."""

from vector.backend import PgvectorBackend


def register():
    """Register the pgvector backend with the datus registry."""
    from datus.storage.vector.registry import VectorRegistry

    VectorRegistry.register("postgresql", PgvectorBackend)


__all__ = ["PgvectorBackend", "register"]
