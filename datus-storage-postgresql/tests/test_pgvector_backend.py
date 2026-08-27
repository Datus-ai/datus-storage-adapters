"""Tests for PgvectorBackend (three-layer architecture)."""

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pytest
from conftest import MockEmbeddingFunction
from psycopg.errors import UndefinedColumn
from psycopg_pool import PoolClosed

from datus_storage_base.backend_config import LOGICAL_NAMESPACE_COLUMN
from datus_storage_base.conditions import and_, eq, like, not_, or_
from datus_storage_base.vector.fts import FtsField, FtsIndexStatus, FtsSpec
from datus_storage_postgresql.vector.backend import (
    PgvectorBackend,
    PgVectorDb,
    PgVectorTable,
    _ngram_terms,
    _physical_schema_name,
)


@pytest.fixture
def backend(pg_config):
    """Create a PgvectorBackend instance with connection config."""
    b = PgvectorBackend()
    b.initialize(pg_config)
    yield b
    b.close()


@pytest.fixture
def db(backend):
    """Connect to the test database and return a VectorDatabase handle."""
    return backend.connect("")


@pytest.fixture
def test_schema():
    """A simple PyArrow schema for testing."""
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("description", pa.string()),
            pa.field("category", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=4)),
        ]
    )


@pytest.fixture
def embedding_function():
    """Mock embedding function."""
    return MockEmbeddingFunction()


@pytest.fixture
def table(db, test_schema, embedding_function):
    """Create a test table and return the VectorTable handle."""
    db.drop_table("test_vectors", ignore_missing=True)
    tbl = db.create_table(
        "test_vectors",
        schema=test_schema,
        embedding_function=embedding_function,
        vector_column="vector",
        source_column="description",
    )
    with db.pool.connection() as conn:
        conn.execute(f"ALTER TABLE {tbl.table_name} ADD CONSTRAINT uq_test_vectors_id UNIQUE (id)")
        conn.commit()
    return tbl


@pytest.fixture
def fts_table(db):
    """Create a text-only table for native PostgreSQL FTS tests."""

    db.drop_table("test_fts", ignore_missing=True)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("search_text", pa.string()),
            pa.field("category", pa.string()),
        ]
    )
    return db.create_table("test_fts", schema=schema, unique_columns=["id"])


def _sample_df(ids, descriptions=None, categories=None):
    """Helper to build a DataFrame for tests."""
    n = len(ids)
    return pd.DataFrame(
        {
            "id": ids,
            "description": descriptions or [f"desc {i}" for i in range(n)],
            "category": categories or ["cat"] * n,
        }
    )


# ==============================================================================
# Backend lifecycle tests
# ==============================================================================


class TestBackendLifecycle:
    def test_initialize_stores_config(self, pg_config):
        b = PgvectorBackend()
        b.initialize(pg_config)
        assert b._config == pg_config

    def test_connect_missing_config(self):
        b = PgvectorBackend()
        b.initialize({})
        with pytest.raises(ValueError, match="Missing required"):
            b.connect("test")

    def test_connect_does_not_eagerly_install_vector_extension(self, db):
        with db.pool.connection() as conn:
            row = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
        assert row is None

    def test_connect_returns_pg_vector_db(self, db):
        assert isinstance(db, PgVectorDb)
        assert db.pool is not None

    def test_connect_multiple(self, backend):
        db1 = backend.connect("multi_ns_1")
        db2 = backend.connect("multi_ns_2")
        assert db1 is not db2

    def test_close(self, pg_config):
        b = PgvectorBackend()
        b.initialize(pg_config)
        db = b.connect("")
        b.close()
        with pytest.raises(PoolClosed):
            with db.pool.connection():
                pass


# ==============================================================================
# Database-level tests (PgVectorDb)
# ==============================================================================


class TestPgVectorDb:
    def test_table_exists(self, db, table):
        assert db.table_exists("test_vectors")
        assert not db.table_exists("nonexistent_table")

    def test_table_names(self, db, table):
        names = db.table_names()
        assert "test_vectors" in names

    def test_table_names_limit(self, db, test_schema, embedding_function):
        for i in range(3):
            db.drop_table(f"tn_limit_{i}", ignore_missing=True)
            db.create_table(f"tn_limit_{i}", schema=test_schema, embedding_function=embedding_function)
        names = db.table_names(limit=2)
        assert len(names) <= 2

    def test_create_table(self, db, test_schema, embedding_function):
        db.drop_table("ct_test", ignore_missing=True)
        tbl = db.create_table("ct_test", schema=test_schema, embedding_function=embedding_function)
        assert isinstance(tbl, PgVectorTable)
        assert tbl.vector_dim == 4
        assert db.table_exists("ct_test")

    def test_create_table_no_schema_no_exist_ok_raises(self, db):
        with pytest.raises(ValueError, match="Schema is required"):
            db.create_table("fail_tbl", schema=None, exist_ok=False)

    def test_create_table_unsupported_schema_raises(self, db):
        with pytest.raises(TypeError, match="Unsupported schema type"):
            db.create_table("fail_tbl2", schema={"bad": "schema"})

    def test_open_table_missing_raises_value_error(self, db):
        """open_table keeps a stable missing-table error before backend-specific DDL runs."""
        with pytest.raises(ValueError, match="Table 'missing_vec' not found"):
            db.open_table("missing_vec")

    def test_open_table_cached(self, db, table, embedding_function):
        """open_table returns the cached handle if available."""
        opened = db.open_table("test_vectors", embedding_function=embedding_function)
        assert opened is table

    def test_open_table_uncached(self, db, test_schema, embedding_function):
        """open_table for uncached table reads columns from information_schema."""
        db.drop_table("open_uc", ignore_missing=True)
        db.create_table("open_uc", schema=test_schema, embedding_function=embedding_function)
        # Clear cache to force re-read
        db._table_cache.pop("open_uc", None)
        opened = db.open_table("open_uc")
        assert isinstance(opened, PgVectorTable)
        assert "id" in opened.column_names

    def test_drop_table(self, db, test_schema, embedding_function):
        db.drop_table("drop_me", ignore_missing=True)
        db.create_table("drop_me", schema=test_schema, embedding_function=embedding_function)
        assert db.table_exists("drop_me")
        db.drop_table("drop_me")
        assert not db.table_exists("drop_me")

    def test_drop_table_ignore_missing(self, db):
        db.drop_table("no_such_table_xyz", ignore_missing=True)

    def test_drop_table_missing_raises(self, db):
        from psycopg.errors import UndefinedTable

        with pytest.raises(UndefinedTable):
            db.drop_table("no_such_table_xyz", ignore_missing=False)

    def test_refresh_table(self, db, table):
        """refresh_table re-opens the table."""
        refreshed = db.refresh_table("test_vectors")
        assert isinstance(refreshed, PgVectorTable)


# ==============================================================================
# Write operations (PgVectorTable)
# ==============================================================================


class TestVectorTableWrite:
    def test_add(self, table):
        table.add(_sample_df(["1", "2", "3"]))
        assert table.count_rows() == 3

    def test_add_empty(self, table):
        table.add(pd.DataFrame({"id": [], "description": [], "category": []}))
        assert table.count_rows() == 0

    def test_add_with_precomputed_vectors(self, table):
        """When vector column is already filled, skip embedding computation."""
        df = pd.DataFrame(
            {
                "id": ["pv1"],
                "description": ["precomp"],
                "category": ["c"],
                "vector": [[0.5, 0.5, 0.5, 0.5]],
            }
        )
        table.add(df)
        assert table.count_rows() == 1

    def test_merge_insert(self, table):
        table.add(_sample_df(["u1", "u2"]))
        update_df = pd.DataFrame(
            {
                "id": ["u1", "u3"],
                "description": ["updated_u1", "new_u3"],
                "category": ["updated", "new"],
            }
        )
        table.merge_insert(update_df, "id")
        assert table.count_rows() == 3
        result = table.search_all(where=eq("id", "u1"))
        assert result.column("category")[0].as_py() == "updated"

    @pytest.mark.parametrize("write_method", ["add", "merge_insert"])
    def test_write_converts_dataframe_nan_to_null(self, db, write_method):
        table_name = f"nullable_boolean_{write_method}"
        db.drop_table(table_name, ignore_missing=True)
        nullable_boolean_table = db.create_table(
            table_name,
            schema=pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("kind", pa.string()),
                    pa.field("is_primary_key", pa.bool_()),
                ]
            ),
            unique_columns=["id"],
        )
        rows = pd.DataFrame(
            [
                {"id": "dataset:orders", "kind": "dataset"},
                {"id": "field:orders.id", "kind": "field", "is_primary_key": True},
            ]
        )
        assert pd.isna(rows.loc[0, "is_primary_key"])

        if write_method == "add":
            nullable_boolean_table.add(rows)
        else:
            nullable_boolean_table.merge_insert(rows, "id")

        stored = {row["id"]: row for row in nullable_boolean_table.search_all().to_pylist()}
        assert stored["dataset:orders"]["is_primary_key"] is None
        assert stored["field:orders.id"]["is_primary_key"] is True

    def test_write_operations_convert_dataframe_scalar_wrappers(self, db):
        db.drop_table("scalar_values", ignore_missing=True)
        scalar_table = db.create_table(
            "scalar_values",
            schema=pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("updated_at", pa.timestamp("us", tz="UTC")),
                    pa.field("row_count", pa.int64()),
                ]
            ),
            unique_columns=["id"],
        )
        first_timestamp = datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc)
        second_timestamp = datetime(2026, 7, 13, 9, 0, tzinfo=timezone.utc)
        third_timestamp = datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc)

        scalar_table.add(
            pd.DataFrame(
                {
                    "id": ["scalar-1"],
                    "updated_at": [pa.scalar(first_timestamp, type=pa.timestamp("us", tz="UTC"))],
                    "row_count": [pa.scalar(1, type=pa.int64())],
                }
            )
        )
        scalar_table.merge_insert(
            pd.DataFrame(
                {
                    "id": ["scalar-1"],
                    "updated_at": [pa.scalar(second_timestamp, type=pa.timestamp("us", tz="UTC"))],
                    "row_count": [pa.scalar(2, type=pa.int64())],
                }
            ),
            "id",
        )
        scalar_table.update(
            eq("id", "scalar-1"),
            {
                "updated_at": pa.scalar(third_timestamp, type=pa.timestamp("us", tz="UTC")),
                "row_count": pa.scalar(3, type=pa.int64()),
            },
        )

        result = scalar_table.search_all()
        assert result.column("updated_at")[0].as_py() == third_timestamp.replace(tzinfo=None)
        assert result.column("row_count")[0].as_py() == 3

    def test_delete_rejects_raw_where_string(self, table):
        table.add(_sample_df(["d1", "d2", "d3"], categories=["rm", "keep", "rm"]))
        with pytest.raises(TypeError, match="Raw string"):
            table.delete("category = 'rm'")
        assert table.count_rows() == 3

    def test_delete_where_expr(self, table):
        table.add(_sample_df(["de1", "de2", "de3"], categories=["rm", "keep", "rm"]))
        table.delete(eq("category", "rm"))
        assert table.count_rows() == 1

    def test_update(self, table):
        table.add(_sample_df(["up1", "up2"], categories=["old", "old"]))
        table.update(eq("id", "up1"), {"category": "new"})
        result = table.search_all(where=eq("id", "up1"))
        assert result.column("category")[0].as_py() == "new"

    def test_update_no_where(self, table):
        table.add(_sample_df(["uw1", "uw2"], categories=["old", "old"]))
        table.update(None, {"category": "all_new"})
        result = table.search_all()
        assert all(v.as_py() == "all_new" for v in result.column("category"))


# ==============================================================================
# Search operations (PgVectorTable)
# ==============================================================================


class TestVectorTableSearch:
    def test_search_vector(self, table):
        table.add(_sample_df(["s1", "s2", "s3"]))
        results = table.search_vector(query_text="test", vector_column="vector", top_n=2)
        assert results.num_rows == 2

    def test_search_vector_rejects_raw_where_string(self, table):
        # Contract: raw SQL where strings are rejected (SQL injection, plus
        # psycopg re-parses inlined text for placeholders); callers must build
        # conditions with the AST helpers (eq/like/and_/...).
        table.add(_sample_df(["w1", "w2", "w3"], categories=["alpha", "beta", "alpha"]))
        with pytest.raises(TypeError, match="Raw string"):
            table.search_vector(
                query_text="test",
                vector_column="vector",
                top_n=10,
                where="category = 'alpha'",
            )

    def test_search_vector_with_where_expr(self, table):
        table.add(_sample_df(["we1", "we2", "we3"], categories=["alpha", "beta", "alpha"]))
        results = table.search_vector(
            query_text="test",
            vector_column="vector",
            top_n=10,
            where=eq("category", "alpha"),
        )
        assert results.num_rows == 2

    def test_search_vector_with_like_filter(self, table):
        # Regression: search_vector always executes with parameters (embedding
        # + LIMIT), so an inlined LIKE pattern's literal '%' used to be
        # re-parsed by psycopg as placeholder syntax and rejected with
        # "only '%s', '%b', '%t' are allowed as placeholders".
        table.add(_sample_df(["lk1", "lk2", "lk3"], categories=["metric", "meta", "doc"]))
        results = table.search_vector(
            query_text="test",
            vector_column="vector",
            top_n=10,
            where=like("category", "met*"),
        )
        assert results.num_rows == 2

    def test_search_all_with_eq_value_containing_percent(self, table):
        # A literal '%' inside a plain value must round-trip exactly — it may
        # neither crash the placeholder parser nor be altered by escaping.
        table.add(_sample_df(["pct1", "pct2"], categories=["50% off", "full price"]))
        results = table.search_all(where=eq("category", "50% off"))
        assert results.num_rows == 1
        assert results.column("id")[0].as_py() == "pct1"

    def test_search_vector_with_select_fields(self, table):
        table.add(_sample_df(["sel1"]))
        results = table.search_vector(
            query_text="test",
            vector_column="vector",
            top_n=1,
            select_fields=["id", "category"],
        )
        assert results.num_rows == 1
        assert "id" in results.column_names
        assert "description" not in results.column_names

    def test_search_vector_no_embedding_fn_raises(self, db, test_schema):
        """Table without embedding_function cannot do vector search."""
        db.drop_table("no_emb", ignore_missing=True)
        tbl = db.create_table("no_emb", schema=test_schema)
        with pytest.raises(RuntimeError, match="No embedding function"):
            tbl.search_vector(query_text="test", vector_column="vector", top_n=1)

    def test_search_hybrid_fallback(self, table):
        table.add(_sample_df(["h1"]))
        results = table.search_hybrid(
            query_text="test",
            vector_source_column="description",
            top_n=1,
        )
        assert results.num_rows == 1

    def test_search_all(self, table):
        table.add(_sample_df(["a1", "a2"]))
        results = table.search_all()
        assert results.num_rows == 2

    def test_search_all_rejects_raw_where_string(self, table):
        table.add(_sample_df(["f1", "f2", "f3"], categories=["keep", "drop", "keep"]))
        with pytest.raises(TypeError, match="Raw string"):
            table.search_all(where="category = 'keep'")

    def test_search_all_with_where_expr(self, table):
        table.add(_sample_df(["fe1", "fe2", "fe3"], categories=["keep", "drop", "keep"]))
        results = table.search_all(where=eq("category", "keep"))
        assert results.num_rows == 2

    def test_search_all_with_and(self, table):
        table.add(_sample_df(["ae1", "ae2", "ae3"], categories=["keep", "keep", "drop"]))
        results = table.search_all(where=and_(eq("category", "keep"), eq("id", "ae1")))
        assert results.num_rows == 1

    def test_search_all_with_or(self, table):
        table.add(_sample_df(["or1", "or2", "or3"], categories=["a", "b", "c"]))
        results = table.search_all(where=or_(eq("category", "a"), eq("category", "c")))
        assert results.num_rows == 2

    def test_search_all_with_not(self, table):
        table.add(_sample_df(["nt1", "nt2", "nt3"], categories=["keep", "drop", "keep"]))
        results = table.search_all(where=not_(eq("category", "drop")))
        assert results.num_rows == 2

    def test_search_all_with_limit(self, table):
        table.add(_sample_df([f"lim{i}" for i in range(5)]))
        results = table.search_all(limit=3)
        assert results.num_rows == 3

    def test_search_all_no_limit(self, table):
        table.add(_sample_df([f"nl{i}" for i in range(5)]))
        results = table.search_all(limit=None)
        assert results.num_rows == 5

    def test_search_all_with_select_fields(self, table):
        table.add(_sample_df(["sf1"]))
        results = table.search_all(select_fields=["id", "category"])
        assert results.num_rows == 1
        assert "id" in results.column_names
        assert "description" not in results.column_names

    def test_empty_table(self, table):
        assert table.count_rows() == 0
        results = table.search_all()
        assert results.num_rows == 0


# ==============================================================================
# count_rows() tests
# ==============================================================================


class TestCountRows:
    def test_count_no_filter(self, table):
        table.add(_sample_df(["c1", "c2", "c3"]))
        assert table.count_rows() == 3

    def test_count_rejects_raw_where_string(self, table):
        table.add(_sample_df(["cs1", "cs2", "cs3"], categories=["p", "q", "p"]))
        with pytest.raises(TypeError, match="Raw string"):
            table.count_rows(where="category = 'p'")

    def test_count_with_where_expr(self, table):
        table.add(_sample_df(["ce1", "ce2", "ce3"], categories=["p", "q", "p"]))
        assert table.count_rows(where=eq("category", "p")) == 2

    def test_count_empty(self, table):
        assert table.count_rows() == 0


# ==============================================================================
# Index operations tests
# ==============================================================================


class TestIndexOperations:
    def test_vector_index_cosine(self, table):
        table.add(_sample_df([f"vi{i}" for i in range(10)]))
        table.create_vector_index("vector", metric="cosine")

    def test_vector_index_l2(self, table):
        table.add(_sample_df([f"vl{i}" for i in range(10)]))
        table.create_vector_index("vector", metric="l2")

    def test_vector_index_ip(self, table):
        table.add(_sample_df([f"vp{i}" for i in range(10)]))
        table.create_vector_index("vector", metric="ip")

    def test_scalar_index(self, table):
        table.create_scalar_index("category")

    def test_fts_index_single_field(self, table):
        table.create_fts_index("description")

    def test_fts_index_multiple_fields(self, table):
        table.create_fts_index(["description", "category"])


class TestConcurrentIndexBuilds:
    """Runtime index builds run CONCURRENTLY so they never lock writers out,
    which costs them the ability to fail cleanly."""

    @staticmethod
    def _index_state(table, index_name):
        schema, _ = table._table_parts()
        with table._pool.connection() as conn:
            return conn.execute(
                "SELECT i.indisvalid, i.indisready FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s",
                (schema, index_name),
            ).fetchone()

    @staticmethod
    def _scalar_index_name(table, column):
        _, bare = table._table_parts()
        return f"idx_{bare}_{column}_btree"

    @staticmethod
    def _assert_still_transactional(table):
        """A connection handed back to the pool must not carry autocommit with it."""
        with table._pool.connection() as conn:
            assert conn.autocommit is False
            conn.execute(f"INSERT INTO {table.table_name} (id) VALUES ('rollback-probe')")
            conn.rollback()
        assert table.count_rows(where=eq("id", "rollback-probe")) == 0

    def test_scalar_index_ends_up_valid(self, table):
        table.add(_sample_df([f"ci{i}" for i in range(5)]))
        table.create_scalar_index("category")

        state = self._index_state(table, self._scalar_index_name(table, "category"))
        assert state is not None
        assert state["indisvalid"] and state["indisready"]

    def test_vector_index_ends_up_valid(self, table):
        table.add(_sample_df([f"cv{i}" for i in range(5)]))
        table.create_vector_index("vector", metric="cosine")

        _, bare = table._table_parts()
        state = self._index_state(table, f"idx_{bare}_vector_hnsw")
        assert state is not None
        assert state["indisvalid"] and state["indisready"]

    def test_build_leaves_pooled_connections_transactional(self, table):
        table.create_scalar_index("category")

        self._assert_still_transactional(table)

    def test_failed_build_leaves_pooled_connections_transactional(self, table):
        with pytest.raises(UndefinedColumn):
            table.create_scalar_index("no_such_column")

        self._assert_still_transactional(table)

    def test_index_left_invalid_by_an_interrupted_build_is_replaced(self, table):
        """An interrupted concurrent build leaves the index in place but invalid.
        IF NOT EXISTS sees it as present, so without cleanup the table would
        never get a working index again."""
        table.add(_sample_df([f"cb{i}" for i in range(5)]))
        table.create_scalar_index("category")
        schema, _ = table._table_parts()
        index_name = self._scalar_index_name(table, "category")

        with table._pool.connection() as conn:
            conn.execute(
                "UPDATE pg_index SET indisvalid = false WHERE indexrelid = %s::regclass",
                (f"{schema}.{index_name}",),
            )
            conn.commit()
        assert self._index_state(table, index_name)["indisvalid"] is False

        table.create_scalar_index("category")

        assert self._index_state(table, index_name)["indisvalid"] is True

    def test_same_index_builds_are_serialized(self, table, monkeypatch):
        """A second caller must not inspect or drop the first caller's in-flight index."""
        first_in_cleanup = threading.Event()
        release_first = threading.Event()
        second_attempted_lock = threading.Event()
        second_in_cleanup = threading.Event()
        call_count = 0
        call_count_lock = threading.Lock()

        original_lock = table._index_build_lock
        original_drop = table._drop_invalid_index

        def observed_lock(conn, index_name):
            nonlocal call_count
            with call_count_lock:
                call_count += 1
                if call_count == 2:
                    second_attempted_lock.set()
            return original_lock(conn, index_name)

        def controlled_drop(conn, index_name):
            if not first_in_cleanup.is_set():
                first_in_cleanup.set()
                if not release_first.wait(timeout=10):
                    raise TimeoutError("timed out waiting to release the first index build")
            else:
                second_in_cleanup.set()
            return original_drop(conn, index_name)

        monkeypatch.setattr(table, "_index_build_lock", observed_lock)
        monkeypatch.setattr(table, "_drop_invalid_index", controlled_drop)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(table.create_scalar_index, "category")
            assert first_in_cleanup.wait(timeout=10)

            second = executor.submit(table.create_scalar_index, "category")
            try:
                assert second_attempted_lock.wait(timeout=10)
                assert not second_in_cleanup.is_set()
            finally:
                release_first.set()

            first.result(timeout=10)
            second.result(timeout=10)

        assert second_in_cleanup.is_set()
        state = self._index_state(table, self._scalar_index_name(table, "category"))
        assert state is not None
        assert state["indisvalid"] and state["indisready"]

    def test_conflict_index_ends_up_valid(self, table):
        table._ensure_conflict_index(["category"])

        _, bare = table._table_parts()
        state = self._index_state(table, f"idx_{bare}_category_uq")
        assert state is not None
        assert state["indisvalid"] and state["indisready"]


class TestNativeFts:
    NGRAM_SPEC = FtsSpec((FtsField("search_text", tokenizer="ngram"),), version=1)

    def test_ngram_terms_split_identifier_punctuation(self):
        assert _ngram_terms("Sales_Order-Fact", 2, 2) == [
            "sa",
            "al",
            "le",
            "es",
            "or",
            "rd",
            "de",
            "er",
            "fa",
            "ac",
            "ct",
        ]

    def test_database_and_table_report_fts_support(self, db, fts_table):
        assert db.supports_fts() is True
        assert fts_table.supports_fts() is True

    def test_search_fts_with_where_filter(self, fts_table):
        # search_fts interleaves score/match parameters with the compiled
        # WHERE's parameters in one statement; a LIKE pattern must ride along
        # as a bind parameter in placeholder order, not as inlined text that
        # psycopg would re-parse as a broken placeholder.
        fts_table.add(
            pd.DataFrame(
                {
                    "id": ["orders", "orders_old", "customers"],
                    "title": ["Sales orders", "Sales orders (old)", "Customers"],
                    "search_text": ["sales_order fact current", "sales_order fact archived", "customer dimension"],
                    "category": ["fact_current", "fact_archived", "dimension"],
                }
            )
        )
        fts_table.create_fts_index(self.NGRAM_SPEC)

        filtered = fts_table.search_fts("sales order", self.NGRAM_SPEC, top_n=5, where=like("category", "fact*"))
        assert sorted(filtered.column("id").to_pylist()) == ["orders", "orders_old"]

        exact = fts_table.search_fts("sales order", self.NGRAM_SPEC, top_n=5, where=eq("category", "fact_current"))
        assert exact.column("id").to_pylist() == ["orders"]

    def test_internal_fts_metadata_table_is_hidden(self, db, fts_table):
        fts_table.create_fts_index(self.NGRAM_SPEC)
        assert "_datus_fts_specs" not in db.table_names()

    def test_create_and_query_chinese_ngram_index(self, db, fts_table):
        fts_table.add(
            pd.DataFrame(
                {
                    "id": ["orders", "customers"],
                    "title": ["Sales orders", "Customers"],
                    "search_text": ["table sales_order definition 销售订单明细", "table customer definition 客户资料"],
                    "category": ["fact", "dimension"],
                }
            )
        )

        assert fts_table.fts_index_status(self.NGRAM_SPEC) == FtsIndexStatus.MISSING
        fts_table.create_fts_index(self.NGRAM_SPEC)
        assert fts_table.fts_index_status(self.NGRAM_SPEC) == FtsIndexStatus.READY

        result = fts_table.search_fts("销售订单", self.NGRAM_SPEC, top_n=5)

        assert result.column("id").to_pylist() == ["orders"]
        assert result.column("_score")[0].as_py() > 0
        generated_column = fts_table._fts_vector_column_name("search_text")
        with fts_table._pool.connection() as conn:
            index_definition = conn.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' AND indexname = %s",
                (fts_table._fts_index_name("search_text"),),
            ).fetchone()["indexdef"]
            generation_expression = conn.execute(
                "SELECT generation_expression FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'test_fts' AND column_name = %s",
                (generated_column,),
            ).fetchone()["generation_expression"]
            pg_trgm = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'").fetchone()
        assert generated_column in index_definition
        assert "_datus_fts_ngrams" in generation_expression
        assert "to_tsvector" in generation_expression
        assert pg_trgm is None

        reopened = db.refresh_table("test_fts")
        assert all(not column.startswith("_datus_fts_vector_") for column in reopened.search_all().column_names)

    def test_raw_substring_search_installs_pg_trgm_lazily(self, fts_table):
        spec = FtsSpec((FtsField("search_text", tokenizer="raw"),))
        fts_table.add(
            pd.DataFrame(
                {
                    "id": ["orders"],
                    "title": ["Orders"],
                    "search_text": ["sales_order_fact"],
                    "category": ["fact"],
                }
            )
        )
        fts_table.create_fts_index(spec)

        result = fts_table.search_fts("order", spec, top_n=5)

        assert result.column("id").to_pylist() == ["orders"]
        with fts_table._pool.connection() as conn:
            index_definition = conn.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' AND indexname = %s",
                (fts_table._fts_index_name("search_text"),),
            ).fetchone()["indexdef"]
            pg_trgm = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'").fetchone()
        assert "gin_trgm_ops" in index_definition
        assert pg_trgm is not None

    def test_zero_match_does_not_fall_back(self, fts_table):
        fts_table.add(
            pd.DataFrame(
                {
                    "id": ["orders"],
                    "title": ["Sales orders"],
                    "search_text": ["销售订单明细"],
                    "category": ["fact"],
                }
            )
        )
        fts_table.create_fts_index(self.NGRAM_SPEC)

        result = fts_table.search_fts("完全无关词", self.NGRAM_SPEC, top_n=5)

        assert result.num_rows == 0

    def test_multifield_boost_controls_ranking(self, fts_table):
        spec = FtsSpec((FtsField("title", boost=3.0), FtsField("search_text")), version=2)
        fts_table.add(
            pd.DataFrame(
                {
                    "id": ["title_hit", "body_hit"],
                    "title": ["revenue", "other"],
                    "search_text": ["other", "revenue"],
                    "category": ["metric", "metric"],
                }
            )
        )
        fts_table.create_fts_index(spec)

        result = fts_table.search_fts("revenue", spec, top_n=5)

        assert result.column("id").to_pylist() == ["title_hit", "body_hit"]
        assert result.column("_score")[0].as_py() > result.column("_score")[1].as_py()

    def test_spec_change_reports_version_mismatch(self, fts_table):
        fts_table.create_fts_index(self.NGRAM_SPEC)
        changed = FtsSpec((FtsField("search_text", tokenizer="ngram"),), version=2)

        assert fts_table.fts_index_status(changed) == FtsIndexStatus.VERSION_MISMATCH

    def test_legacy_fixed_tsv_index_is_detected_and_removed(self, fts_table):
        with fts_table._pool.connection() as conn:
            conn.execute(
                "ALTER TABLE test_fts ADD COLUMN tsv tsvector "
                "GENERATED ALWAYS AS (to_tsvector('english', COALESCE(search_text, ''))) STORED"
            )
            conn.execute("CREATE INDEX idx_test_fts_fts ON test_fts USING gin (tsv)")
            conn.commit()

        assert fts_table.fts_index_status(self.NGRAM_SPEC) == FtsIndexStatus.LEGACY
        assert fts_table.remove_legacy_fts_index() is True
        assert fts_table.fts_index_status(self.NGRAM_SPEC) == FtsIndexStatus.MISSING

    def test_upsert_and_delete_update_index_without_rebuild(self, fts_table):
        fts_table.create_fts_index(self.NGRAM_SPEC)
        fts_table.merge_insert(
            pd.DataFrame(
                {
                    "id": ["orders"],
                    "title": ["Orders"],
                    "search_text": ["销售订单"],
                    "category": ["fact"],
                }
            ),
            "id",
        )
        assert fts_table.search_fts("销售订单", self.NGRAM_SPEC, top_n=5).num_rows == 1

        fts_table.merge_insert(
            pd.DataFrame(
                {
                    "id": ["orders"],
                    "title": ["Orders"],
                    "search_text": ["退款记录"],
                    "category": ["fact"],
                }
            ),
            "id",
        )
        fts_table.optimize()
        assert fts_table.search_fts("销售订单", self.NGRAM_SPEC, top_n=5).num_rows == 0
        assert fts_table.search_fts("退款记录", self.NGRAM_SPEC, top_n=5).num_rows == 1

        fts_table.delete(eq("id", "orders"))
        assert fts_table.search_fts("退款记录", self.NGRAM_SPEC, top_n=5).num_rows == 0


# ==============================================================================
# Namespace (schema) isolation tests
# ==============================================================================


class TestVectorNamespace:
    def test_physical_schema_name_preserves_safe_names(self):
        assert _physical_schema_name("vec_ns_test") == "vec_ns_test"

    def test_physical_schema_name_maps_opaque_names_stably(self):
        namespace = "Users-kangxue-work-datus-datus-benchmark"
        physical_name = _physical_schema_name(namespace)

        assert physical_name == _physical_schema_name(namespace)
        assert len(physical_name) <= 63
        assert physical_name.startswith("users_kangxue_work_datus_datus_benchmark_")
        assert physical_name != _physical_schema_name("Users_kangxue_work_datus_datus_benchmark")
        assert _physical_schema_name("Project") != _physical_schema_name("project")

    def test_namespace_creates_schema(self, backend):
        db = backend.connect("vec_ns_test")
        assert db.namespace == "vec_ns_test"

    def test_opaque_namespace_supports_native_fts(self, backend):
        namespace = "project-with-hyphens"
        db = backend.connect(namespace)
        db.drop_table("opaque_fts", ignore_missing=True)
        table = db.create_table(
            "opaque_fts",
            schema=pa.schema([pa.field("id", pa.string()), pa.field("search_text", pa.string())]),
            unique_columns=["id"],
        )
        spec = FtsSpec(fields=(FtsField("search_text", tokenizer="ngram"),))
        table.add(pd.DataFrame({"id": ["orders"], "search_text": ["销售订单"]}))
        table.create_fts_index(spec)

        assert db.namespace == namespace
        assert table.table_name == f"{_physical_schema_name(namespace)}.opaque_fts"
        assert table.search_fts("销售订单", spec, top_n=5).column("id").to_pylist() == ["orders"]

    def test_namespace_qualified_table_name(self, backend, test_schema, embedding_function):
        db = backend.connect("qn_ns")
        db.drop_table("ns_tbl", ignore_missing=True)
        tbl = db.create_table("ns_tbl", schema=test_schema, embedding_function=embedding_function)
        assert tbl.table_name == "qn_ns.ns_tbl"

    def test_namespace_isolation(self, backend, test_schema, embedding_function):
        """Tables in different namespaces are independent."""
        db_a = backend.connect("iso_a")
        db_b = backend.connect("iso_b")

        db_a.drop_table("shared", ignore_missing=True)
        db_b.drop_table("shared", ignore_missing=True)

        tbl_a = db_a.create_table("shared", schema=test_schema, embedding_function=embedding_function)
        tbl_b = db_b.create_table("shared", schema=test_schema, embedding_function=embedding_function)

        tbl_a.add(_sample_df(["a1"]))
        tbl_b.add(_sample_df(["b1", "b2"]))

        assert tbl_a.count_rows() == 1
        assert tbl_b.count_rows() == 2

    def test_public_namespace(self, db):
        """Empty namespace uses 'public' — no schema prefix."""
        assert db.namespace == ""


# ==============================================================================
# Arrow conversion tests
# ==============================================================================


class TestArrowConversion:
    def test_rows_to_arrow_types(self, table):
        """Verify returned PyArrow table has correct types."""
        table.add(_sample_df(["ar1", "ar2"]))
        result = table.search_all()
        assert isinstance(result, pa.Table)
        assert result.num_rows == 2
        assert result.column("id").type == pa.string()

    def test_empty_result_with_select_fields(self, table):
        result = table.search_all(select_fields=["id", "category"])
        assert result.num_rows == 0
        assert "id" in result.column_names
        assert "category" in result.column_names


# ==============================================================================
# Vector logical isolation tests
# ==============================================================================


@pytest.fixture
def logical_backend(pg_config):
    """Create a PgvectorBackend with logical isolation."""
    config = {**pg_config, "isolation": "logical"}
    b = PgvectorBackend()
    b.initialize(config)
    yield b
    b.close()


@pytest.fixture
def logical_db(logical_backend):
    """Connect with a namespace under logical isolation."""
    return logical_backend.connect("tenant_a")


def _drop_table_raw(pool, table_name):
    """Drop a table directly via pool, bypassing logical isolation guard (test-only)."""
    with pool.connection() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()


@pytest.fixture
def logical_table(logical_db, test_schema, embedding_function):
    """Create a test table under logical isolation."""
    _drop_table_raw(logical_db.pool, "logical_vectors")
    tbl = logical_db.create_table(
        "logical_vectors",
        schema=test_schema,
        embedding_function=embedding_function,
        vector_column="vector",
        source_column="description",
    )
    return tbl


class TestVectorLogicalIsolation:
    def test_table_in_public_schema(self, logical_table):
        """Logical isolation uses public schema."""
        assert "." not in logical_table.table_name  # no schema prefix

    def test_logical_namespace_column_created(self, logical_db, logical_table):
        """create_table auto-adds the internal logical namespace column."""
        with logical_db.pool.connection() as conn:
            rows = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s AND column_name = %s",
                ("logical_vectors", LOGICAL_NAMESPACE_COLUMN),
            ).fetchall()
            assert len(rows) == 1

    def test_add_injects_logical_namespace(self, logical_db, logical_table):
        """add() auto-injects the internal logical namespace."""
        logical_table.add(_sample_df(["la1"]))
        with logical_db.pool.connection() as conn:
            rows = conn.execute(f"SELECT {LOGICAL_NAMESPACE_COLUMN} FROM logical_vectors WHERE id = 'la1'").fetchall()
            val = rows[0][LOGICAL_NAMESPACE_COLUMN] if isinstance(rows[0], dict) else rows[0][0]
            assert val == "tenant_a"

    def test_preserves_application_datasource_id(self, logical_db, embedding_function):
        """Application datasource_id remains queryable under logical isolation."""
        _drop_table_raw(logical_db.pool, "logical_scoped_vectors")
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("datasource_id", pa.string()),
                pa.field("description", pa.string()),
                pa.field("category", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), list_size=4)),
            ]
        )
        table = logical_db.create_table(
            "logical_scoped_vectors",
            schema=schema,
            embedding_function=embedding_function,
            vector_column="vector",
            source_column="description",
            unique_columns=["id"],
        )
        table.add(
            pd.DataFrame(
                {
                    "id": ["sv1"],
                    "datasource_id": ["jeff_shop"],
                    "description": ["scoped vector"],
                    "category": ["metric"],
                }
            )
        )

        with logical_db.pool.connection() as conn:
            row = conn.execute(
                f"SELECT datasource_id, {LOGICAL_NAMESPACE_COLUMN} FROM logical_scoped_vectors WHERE id = 'sv1'"
            ).fetchone()
        assert row["datasource_id"] == "jeff_shop"
        assert row[LOGICAL_NAMESPACE_COLUMN] == "tenant_a"

        result = table.search_all(where=eq("datasource_id", "jeff_shop"))
        assert result.num_rows == 1
        assert result.column("datasource_id")[0].as_py() == "jeff_shop"
        assert LOGICAL_NAMESPACE_COLUMN not in result.column_names

    def test_open_table_missing_raises_value_error_in_logical_mode(self, logical_backend):
        """Logical open_table checks existence before running namespace migration DDL."""
        db = logical_backend.connect("tenant_a")
        with pytest.raises(ValueError, match="Table 'missing_logical_vec' not found"):
            db.open_table("missing_logical_vec")

    def test_like_filter_under_logical_namespace(self, logical_table):
        """Regression for the production crash: the logical-namespace scope
        adds real %s parameters to every statement, so a LIKE filter whose
        pattern used to be inlined made psycopg reject the literal '%' as an
        invalid placeholder. The filter must both execute and scope rows."""
        logical_table.add(_sample_df(["ll1", "ll2", "ll3"], categories=["metric", "meta", "doc"]))
        assert logical_table.count_rows(where=like("category", "met*")) == 2
        results = logical_table.search_all(where=like("category", "met*"))
        assert results.num_rows == 2

    def test_delete_with_like_filter_under_logical_namespace(self, logical_table):
        logical_table.add(_sample_df(["ld1", "ld2"], categories=["tmp_a", "keep"]))
        logical_table.delete(where=like("category", "tmp*"))
        assert logical_table.count_rows() == 1

    def test_unique_columns_scoped_to_logical_namespace(self, logical_backend, test_schema, embedding_function):
        """Fresh logical tables scope unique_columns by backend namespace."""
        db_a = logical_backend.connect("tenant_a")
        db_b = logical_backend.connect("tenant_b")

        _drop_table_raw(db_a.pool, "logical_unique_vectors")
        tbl_a = db_a.create_table(
            "logical_unique_vectors",
            schema=test_schema,
            embedding_function=embedding_function,
            unique_columns=["id"],
        )
        tbl_b = db_b.create_table(
            "logical_unique_vectors",
            schema=test_schema,
            embedding_function=embedding_function,
            unique_columns=["id"],
        )

        tbl_a.add(_sample_df(["same_id"]))
        tbl_b.add(_sample_df(["same_id"]))

        assert tbl_a.count_rows() == 1
        assert tbl_b.count_rows() == 1

    def test_migrates_legacy_unique_column_to_logical_namespace(self, logical_backend, test_schema, embedding_function):
        """Existing global unique_columns are replaced with namespace-scoped indexes."""
        db_a = logical_backend.connect("tenant_a")
        db_b = logical_backend.connect("tenant_b")

        _drop_table_raw(db_a.pool, "legacy_unique_vec")
        with db_a.pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE legacy_unique_vec (
                    id TEXT UNIQUE,
                    description TEXT,
                    category TEXT,
                    vector vector(4)
                )
                """
            )
            conn.commit()

        tbl_a = db_a.create_table(
            "legacy_unique_vec",
            schema=test_schema,
            embedding_function=embedding_function,
            unique_columns=["id"],
        )
        tbl_b = db_b.create_table(
            "legacy_unique_vec",
            schema=test_schema,
            embedding_function=embedding_function,
            unique_columns=["id"],
        )

        tbl_a.add(_sample_df(["same_id"]))
        tbl_b.add(_sample_df(["same_id"]))

        assert tbl_a.count_rows() == 1
        assert tbl_b.count_rows() == 1

    def test_open_table_migrates_missing_logical_namespace(self, logical_backend, test_schema, embedding_function):
        """open_table() self-heals legacy logical tables missing the namespace column."""
        db_a = logical_backend.connect("tenant_a")
        db_b = logical_backend.connect("tenant_b")

        _drop_table_raw(db_a.pool, "legacy_open_vec")
        with db_a.pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE legacy_open_vec (
                    id TEXT UNIQUE,
                    description TEXT,
                    category TEXT,
                    vector vector(4)
                )
                """
            )
            conn.execute(
                "INSERT INTO legacy_open_vec (id, description, category, vector) "
                "VALUES ('same_id', 'legacy row', 'legacy', '[0.1,0.2,0.3,0.4]')"
            )
            conn.commit()

        tbl_a = db_a.open_table("legacy_open_vec", embedding_function=embedding_function)
        assert tbl_a.count_rows() == 1
        with db_a.pool.connection() as conn:
            row = conn.execute(
                f"SELECT {LOGICAL_NAMESPACE_COLUMN} FROM legacy_open_vec WHERE id = 'same_id'"
            ).fetchone()
        assert row[LOGICAL_NAMESPACE_COLUMN] == "tenant_a"

        tbl_b = db_b.open_table("legacy_open_vec", embedding_function=embedding_function)
        tbl_b.add(_sample_df(["same_id"]))

        assert tbl_a.count_rows() == 1
        assert tbl_b.count_rows() == 1

    def test_open_table_migrates_legacy_unique_index(self, logical_backend, embedding_function):
        """open_table() also migrates pre-existing unscoped unique indexes."""
        db_a = logical_backend.connect("tenant_a")
        db_b = logical_backend.connect("tenant_b")

        _drop_table_raw(db_a.pool, "legacy_open_index_vec")
        with db_a.pool.connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE legacy_open_index_vec (
                    id TEXT,
                    description TEXT,
                    category TEXT,
                    vector vector(4),
                    {LOGICAL_NAMESPACE_COLUMN} TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute("CREATE UNIQUE INDEX legacy_open_index_vec_id_uq ON legacy_open_index_vec(id)")
            conn.execute(
                "INSERT INTO legacy_open_index_vec (id, description, category, vector) "
                "VALUES ('same_id', 'legacy row', 'legacy', '[0.1,0.2,0.3,0.4]')"
            )
            conn.commit()

        tbl_a = db_a.open_table("legacy_open_index_vec", embedding_function=embedding_function)
        assert tbl_a.count_rows() == 1
        with db_a.pool.connection() as conn:
            row = conn.execute(
                f"SELECT {LOGICAL_NAMESPACE_COLUMN} FROM legacy_open_index_vec WHERE id = 'same_id'"
            ).fetchone()
        assert row[LOGICAL_NAMESPACE_COLUMN] == "tenant_a"

        tbl_b = db_b.open_table("legacy_open_index_vec", embedding_function=embedding_function)
        tbl_b.add(_sample_df(["same_id"]))

        assert tbl_a.count_rows() == 1
        assert tbl_b.count_rows() == 1

    def test_search_all_filters_by_datasource(self, logical_backend, test_schema, embedding_function):
        """search_all only returns rows for the connected namespace."""
        db_a = logical_backend.connect("tenant_a")
        db_b = logical_backend.connect("tenant_b")

        _drop_table_raw(db_a.pool, "shared_vec")
        tbl_a = db_a.create_table("shared_vec", schema=test_schema, embedding_function=embedding_function)
        tbl_b = db_b.open_table("shared_vec", embedding_function=embedding_function)

        tbl_a.add(_sample_df(["a1"]))
        tbl_b.add(_sample_df(["b1", "b2"]))

        assert tbl_a.count_rows() == 1
        assert tbl_b.count_rows() == 2

    def test_delete_scoped_to_logical_namespace(self, logical_backend, test_schema, embedding_function):
        """delete() only affects rows for the connected namespace."""
        db_a = logical_backend.connect("tenant_a")
        db_b = logical_backend.connect("tenant_b")

        _drop_table_raw(db_a.pool, "del_vec")
        tbl_a = db_a.create_table("del_vec", schema=test_schema, embedding_function=embedding_function)
        tbl_b = db_b.open_table("del_vec", embedding_function=embedding_function)

        tbl_a.add(_sample_df(["da1"]))
        tbl_b.add(_sample_df(["db1"]))

        tbl_a.delete(eq("id", "da1"))
        assert tbl_a.count_rows() == 0
        assert tbl_b.count_rows() == 1

    def test_update_scoped_to_logical_namespace(self, logical_backend, test_schema, embedding_function):
        """update() only affects rows for the connected namespace."""
        db_a = logical_backend.connect("tenant_a")
        db_b = logical_backend.connect("tenant_b")

        _drop_table_raw(db_a.pool, "upd_vec")
        tbl_a = db_a.create_table("upd_vec", schema=test_schema, embedding_function=embedding_function)
        tbl_b = db_b.open_table("upd_vec", embedding_function=embedding_function)

        tbl_a.add(_sample_df(["ua1"], categories=["old"]))
        tbl_b.add(_sample_df(["ub1"], categories=["old"]))

        tbl_a.update(eq("id", "ua1"), {"category": "new"})
        result_a = tbl_a.search_all()
        result_b = tbl_b.search_all()
        assert result_a.column("category")[0].as_py() == "new"
        assert result_b.column("category")[0].as_py() == "old"

    def test_search_all_excludes_logical_namespace_from_results(self, logical_table):
        """Default SELECT should not include the internal namespace column."""
        logical_table.add(_sample_df(["ex1"]))
        result = logical_table.search_all()
        assert LOGICAL_NAMESPACE_COLUMN not in result.column_names

    def test_fts_search_is_scoped_to_logical_namespace(self, logical_backend):
        db_a = logical_backend.connect("tenant_a")
        db_b = logical_backend.connect("tenant_b")
        _drop_table_raw(db_a.pool, "logical_fts")
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("search_text", pa.string()),
            ]
        )
        table_a = db_a.create_table("logical_fts", schema=schema, unique_columns=["id"])
        table_b = db_b.open_table("logical_fts")
        spec = FtsSpec((FtsField("search_text", tokenizer="ngram"),))
        table_a.create_fts_index(spec)
        table_a.merge_insert(pd.DataFrame({"id": ["a"], "search_text": ["销售订单"]}), "id")
        table_b.merge_insert(pd.DataFrame({"id": ["b"], "search_text": ["销售订单"]}), "id")

        assert table_a.search_fts("销售订单", spec, top_n=5).column("id").to_pylist() == ["a"]
        assert table_b.search_fts("销售订单", spec, top_n=5).column("id").to_pylist() == ["b"]


# ==============================================================================
# Schema-evolution (ensure_columns) tests
# ==============================================================================


def _column_names(pool, table_name):
    """Return the live column names of a table."""
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (table_name,),
        ).fetchall()
        return {r["column_name"] if isinstance(r, dict) else r[0] for r in rows}


class TestVectorEnsureColumns:
    """Pre-existing tables get scope columns added and backfilled on demand."""

    def test_adds_and_backfills_missing_columns(self, db, test_schema, embedding_function):
        # test_schema has no datasource_id / storage_key — it models an old table.
        db.drop_table("ec_vectors", ignore_missing=True)
        tbl = db.create_table("ec_vectors", schema=test_schema, embedding_function=embedding_function)
        tbl.add(_sample_df(["e1", "e2"]))

        tbl.ensure_columns({"datasource_id": "''", "storage_key": "'legacy:' || id"})

        assert {"datasource_id", "storage_key"} <= _column_names(db.pool, "ec_vectors")
        with db.pool.connection() as conn:
            rows = conn.execute("SELECT id, datasource_id, storage_key FROM ec_vectors ORDER BY id").fetchall()
        assert [r["storage_key"] for r in rows] == ["legacy:e1", "legacy:e2"]
        assert [r["datasource_id"] for r in rows] == ["", ""]

    def test_is_idempotent_and_preserves_values(self, db, test_schema, embedding_function):
        db.drop_table("ec_idem", ignore_missing=True)
        tbl = db.create_table("ec_idem", schema=test_schema, embedding_function=embedding_function)
        tbl.add(_sample_df(["i1"]))

        tbl.ensure_columns({"storage_key": "'legacy:' || id"})
        # A second call must not re-run the backfill or fail.
        tbl.ensure_columns({"storage_key": "'legacy:' || id"})

        with db.pool.connection() as conn:
            row = conn.execute("SELECT storage_key FROM ec_idem WHERE id = 'i1'").fetchone()
        assert (row["storage_key"] if isinstance(row, dict) else row[0]) == "legacy:i1"

    def test_empty_expressions_is_noop(self, db, test_schema, embedding_function):
        db.drop_table("ec_noop", ignore_missing=True)
        tbl = db.create_table("ec_noop", schema=test_schema, embedding_function=embedding_function)
        before = _column_names(db.pool, "ec_noop")
        tbl.ensure_columns({})
        assert _column_names(db.pool, "ec_noop") == before

    def test_upsert_on_migrated_column_self_heals_unique_index(self, db, test_schema, embedding_function):
        # The migrated storage_key column has no UNIQUE index, which ON CONFLICT
        # needs; merge_insert must create it on demand and succeed.
        db.drop_table("ec_upsert", ignore_missing=True)
        tbl = db.create_table("ec_upsert", schema=test_schema, embedding_function=embedding_function)
        tbl.add(_sample_df(["m1"]))
        tbl.ensure_columns({"storage_key": "'legacy:' || id"})

        update_df = pd.DataFrame(
            {
                "id": ["m1", "m2"],
                "storage_key": ["legacy:m1", "legacy:m2"],
                "description": ["updated_m1", "new_m2"],
                "category": ["updated", "new"],
            }
        )
        tbl.merge_insert(update_df, "storage_key")

        assert tbl.count_rows() == 2
        result = tbl.search_all(where=eq("id", "m1"))
        assert result.column("category")[0].as_py() == "updated"
