"""A role that may only SELECT must still be able to open pre-existing stores.

PostgreSQL checks schema-level ``CREATE`` *before* ``IF NOT EXISTS`` short-circuits,
so emitting DDL that would be a no-op still fails for a read-only role. These tests
pin the rule that the adapters only emit DDL when it would actually change something.
"""

from dataclasses import dataclass
from typing import Optional

import psycopg
import pyarrow as pa
import pytest

from datus_storage_base.rdb.base import ColumnDef, IndexDef, TableDefinition
from datus_storage_postgresql.rdb.backend import PostgresRdbBackend
from datus_storage_postgresql.vector.backend import PgvectorBackend

RO_USER = "datus_readonly_probe"
RO_PASSWORD = "readonlypass"


@dataclass
class Item:
    id: Optional[int] = None
    name: Optional[str] = None
    datasource_id: Optional[str] = None


def _table_def(name: str) -> TableDefinition:
    return TableDefinition(
        table_name=name,
        columns=[
            ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True),
            ColumnDef(name="name", col_type="TEXT", nullable=False),
            ColumnDef(name="datasource_id", col_type="TEXT", default=""),
        ],
        indices=[IndexDef(name=f"idx_{name}_name", columns=["name", "datasource_id"], unique=True)],
        constraints=["UNIQUE(name, datasource_id)"],
    )


@pytest.fixture
def admin_conn(pg_config):
    """Autocommit connection as the container superuser."""
    conn = psycopg.connect(
        host=pg_config["host"],
        port=pg_config["port"],
        user=pg_config["user"],
        password=pg_config["password"],
        dbname=pg_config["dbname"],
        autocommit=True,
    )
    yield conn
    conn.close()


@pytest.fixture
def readonly_config(pg_config, admin_conn):
    """A config for a role with USAGE but no CREATE, plus a `grant_reads` callback."""
    dbname = pg_config["dbname"]
    admin_conn.execute(f"DROP OWNED BY {RO_USER} CASCADE" if _role_exists(admin_conn) else "SELECT 1")
    if _role_exists(admin_conn):
        admin_conn.execute(f"DROP ROLE {RO_USER}")
    admin_conn.execute(f"CREATE ROLE {RO_USER} LOGIN PASSWORD '{RO_PASSWORD}'")
    admin_conn.execute(f"GRANT CONNECT ON DATABASE {dbname} TO {RO_USER}")
    admin_conn.execute(f"GRANT USAGE ON SCHEMA public TO {RO_USER}")
    admin_conn.execute(f"REVOKE CREATE ON SCHEMA public FROM {RO_USER}")
    admin_conn.execute(f"REVOKE CREATE ON DATABASE {dbname} FROM {RO_USER}")

    yield {**pg_config, "user": RO_USER, "password": RO_PASSWORD}

    admin_conn.execute(f"REASSIGN OWNED BY {RO_USER} TO {pg_config['user']}")
    admin_conn.execute(f"DROP OWNED BY {RO_USER} CASCADE")
    admin_conn.execute(f"DROP ROLE IF EXISTS {RO_USER}")


def _role_exists(conn) -> bool:
    return conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (RO_USER,)).fetchone() is not None


def _grant_reads(conn, *schemas: str) -> None:
    for schema in schemas or ("public",):
        conn.execute(f"GRANT USAGE ON SCHEMA {schema} TO {RO_USER}")
        conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO {RO_USER}")


def _drop_table(conn, name: str) -> None:
    # Unquoted, so it folds exactly like the DDL the RDB backend emits.
    conn.execute(f"DROP TABLE IF EXISTS {name} CASCADE")


def _relkind(conn, relname: str, schema: str = "public") -> str:
    """pg_class.relkind of a relation in `schema`, by its *stored* name."""
    row = conn.execute(
        "SELECT c.relkind FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = %s AND c.relname = %s",
        (schema, relname),
    ).fetchone()
    return row[0] if row else ""


def _assert_no_create_privilege(config) -> None:
    """The premise: the role really cannot create anything."""
    with psycopg.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        dbname=config["dbname"],
    ) as conn:
        row = conn.execute(
            "SELECT has_schema_privilege(current_user, 'public', 'USAGE') AS usage, "
            "has_schema_privilege(current_user, 'public', 'CREATE') AS create_"
        ).fetchone()
        assert row[0] is True
        assert row[1] is False


class TestReadOnlyRdbStore:
    def test_ensure_table_on_existing_table_emits_no_ddl(self, pg_config, admin_conn, readonly_config):
        """The whole point: an existing table opens fine for a role that cannot CREATE."""
        table_def = _table_def("ro_items")
        _drop_table(admin_conn, "ro_items")

        owner = PostgresRdbBackend()
        owner.initialize(pg_config)
        try:
            owner_db = owner.connect(namespace="", store_db_name="test")
            owner_db.ensure_table(table_def).insert(Item(name="row-one", datasource_id=""))
        finally:
            owner.close()
        _grant_reads(admin_conn)
        _assert_no_create_privilege(readonly_config)

        reader = PostgresRdbBackend()
        reader.initialize(readonly_config)
        try:
            reader_db = reader.connect(namespace="", store_db_name="test")
            table = reader_db.ensure_table(table_def)
            assert [item.name for item in table.query(Item)] == ["row-one"]
        finally:
            reader.close()
            _drop_table(admin_conn, "ro_items")

    def test_ensure_table_still_fails_when_the_table_is_really_missing(self, pg_config, admin_conn, readonly_config):
        """Skipping no-op DDL must not paper over a genuinely absent table."""
        table_def = _table_def("ro_absent")
        _drop_table(admin_conn, "ro_absent")
        _grant_reads(admin_conn)

        reader = PostgresRdbBackend()
        reader.initialize(readonly_config)
        try:
            reader_db = reader.connect(namespace="", store_db_name="test")
            with pytest.raises(RuntimeError, match="table 'ro_absent' does not exist"):
                reader_db.ensure_table(table_def)
        finally:
            reader.close()

    def test_ensure_table_creates_a_missing_index_on_an_existing_table(self, pg_config, admin_conn):
        """An owner still gets index DDL when the table exists but the index does not."""
        table_def = _table_def("ro_partial")
        _drop_table(admin_conn, "ro_partial")
        admin_conn.execute(
            "CREATE TABLE ro_partial (id SERIAL PRIMARY KEY, name TEXT NOT NULL, datasource_id TEXT DEFAULT '')"
        )

        owner = PostgresRdbBackend()
        owner.initialize(pg_config)
        try:
            owner_db = owner.connect(namespace="", store_db_name="test")
            owner_db.ensure_table(table_def)
        finally:
            owner.close()

        indexes = admin_conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = %s AND indexname = %s",
            ("ro_partial", "idx_ro_partial_name"),
        ).fetchall()
        assert len(indexes) == 1
        _drop_table(admin_conn, "ro_partial")

    def test_ensure_table_matches_a_folded_mixed_case_name(self, pg_config, admin_conn, readonly_config):
        """Unquoted DDL folds `RoCaseItems` to `rocaseitems`; the probe has to resolve it the same way."""
        table_def = _table_def("RoCaseItems")
        _drop_table(admin_conn, "RoCaseItems")

        owner = PostgresRdbBackend()
        owner.initialize(pg_config)
        try:
            owner_db = owner.connect(namespace="", store_db_name="test")
            owner_db.ensure_table(table_def).insert(Item(name="folded", datasource_id=""))
        finally:
            owner.close()
        assert _relkind(admin_conn, "rocaseitems") == "r"
        _grant_reads(admin_conn)

        reader = PostgresRdbBackend()
        reader.initialize(readonly_config)
        try:
            reader_db = reader.connect(namespace="", store_db_name="test")
            assert [i.name for i in reader_db.ensure_table(table_def).query(Item)] == ["folded"]
        finally:
            reader.close()
            _drop_table(admin_conn, "RoCaseItems")

    def test_ensure_table_matches_an_index_name_past_namedatalen(self, pg_config, admin_conn, readonly_config):
        """PostgreSQL truncates identifiers at 63 bytes, so the stored name is not the requested one.

        Not hypothetical: `_scoped_unique_index_name` builds
        `idx_pub_tb_subject_nodes_parent_id_name_datasource_id__datus_namespace_uq`
        — 73 characters — for the very deployment this PR is about.
        """
        long_index = "idx_ro_overlength_" + "n" * 55  # 73 chars, stored as 63
        assert len(long_index) > 63
        table_def = TableDefinition(
            table_name="ro_overlength",
            columns=[
                ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True),
                ColumnDef(name="name", col_type="TEXT", nullable=False),
                ColumnDef(name="datasource_id", col_type="TEXT", default=""),
            ],
            indices=[IndexDef(name=long_index, columns=["name"])],
        )
        _drop_table(admin_conn, "ro_overlength")

        owner = PostgresRdbBackend()
        owner.initialize(pg_config)
        try:
            owner_db = owner.connect(namespace="", store_db_name="test")
            owner_db.ensure_table(table_def).insert(Item(name="truncated", datasource_id=""))
        finally:
            owner.close()
        assert _relkind(admin_conn, long_index[:63]) == "i"
        _grant_reads(admin_conn)

        reader = PostgresRdbBackend()
        reader.initialize(readonly_config)
        try:
            reader_db = reader.connect(namespace="", store_db_name="test")
            assert [i.name for i in reader_db.ensure_table(table_def).query(Item)] == ["truncated"]
        finally:
            reader.close()
            _drop_table(admin_conn, "ro_overlength")

    def test_ensure_table_matches_a_partitioned_parent_index(self, pg_config, admin_conn, readonly_config):
        """A partitioned table's parent index is relkind 'I', not 'i'."""
        table_def = TableDefinition(
            table_name="ro_part",
            columns=[
                ColumnDef(name="id", col_type="INTEGER"),
                ColumnDef(name="name", col_type="TEXT", nullable=False),
                ColumnDef(name="datasource_id", col_type="TEXT", default=""),
            ],
            indices=[IndexDef(name="idx_ro_part_name", columns=["name", "datasource_id"])],
        )
        _drop_table(admin_conn, "ro_part")
        admin_conn.execute(
            "CREATE TABLE ro_part (id INTEGER, name TEXT NOT NULL, datasource_id TEXT DEFAULT '') "
            "PARTITION BY RANGE (id)"
        )
        admin_conn.execute("CREATE TABLE ro_part_p1 PARTITION OF ro_part FOR VALUES FROM (0) TO (100)")
        admin_conn.execute("CREATE INDEX idx_ro_part_name ON ro_part (name, datasource_id)")
        admin_conn.execute("INSERT INTO ro_part (id, name) VALUES (1, 'partitioned')")
        assert _relkind(admin_conn, "ro_part") == "p"
        assert _relkind(admin_conn, "idx_ro_part_name") == "I"
        _grant_reads(admin_conn)

        reader = PostgresRdbBackend()
        reader.initialize(readonly_config)
        try:
            reader_db = reader.connect(namespace="", store_db_name="test")
            assert [i.name for i in reader_db.ensure_table(table_def).query(Item)] == ["partitioned"]
        finally:
            reader.close()
            _drop_table(admin_conn, "ro_part")

    def test_index_probe_is_pinned_to_the_tables_schema_not_the_search_path(self, pg_config, admin_conn):
        """`CREATE INDEX name ON t` creates and tests `name` in t's schema.

        A same-named relation in a different `search_path` entry is a different object;
        counting it would silently skip a required unique index and surface much later
        as a uniqueness or upsert failure.

        The target table sits in `ro_shadow`, which is also where an unqualified
        `CREATE TABLE` would land under this search path, so the only thing this test
        varies is where the *index name* is looked up.
        """
        table_def = _table_def("ro_probe_target")
        admin_conn.execute("DROP SCHEMA IF EXISTS ro_shadow CASCADE")
        _drop_table(admin_conn, "idx_ro_probe_target_name")
        admin_conn.execute("CREATE SCHEMA ro_shadow")
        admin_conn.execute(
            "CREATE TABLE ro_shadow.ro_probe_target "
            "(id SERIAL PRIMARY KEY, name TEXT NOT NULL, datasource_id TEXT DEFAULT '')"
        )
        # A decoy carrying the index's name, in the other schema on the search path.
        admin_conn.execute("CREATE TABLE public.idx_ro_probe_target_name (x INTEGER)")
        admin_conn.execute(f"ALTER ROLE {pg_config['user']} SET search_path = ro_shadow, public")
        try:
            owner = PostgresRdbBackend()
            owner.initialize(pg_config)
            try:
                owner_db = owner.connect(namespace="", store_db_name="test")
                # Unqualified, so both the DDL and the probe go through search_path.
                assert owner_db._qualified("ro_probe_target") == "ro_probe_target"
                owner_db.ensure_table(table_def)
            finally:
                owner.close()
            assert _relkind(admin_conn, "idx_ro_probe_target_name", "ro_shadow") == "i"
        finally:
            admin_conn.execute(f"ALTER ROLE {pg_config['user']} RESET search_path")
            admin_conn.execute("DROP SCHEMA IF EXISTS ro_shadow CASCADE")
            _drop_table(admin_conn, "idx_ro_probe_target_name")

    def test_connect_to_an_existing_namespace_emits_no_create_schema(self, pg_config, admin_conn, readonly_config):
        """`CREATE SCHEMA IF NOT EXISTS` is checked before it short-circuits, too."""
        admin_conn.execute("DROP SCHEMA IF EXISTS ro_ns CASCADE")
        admin_conn.execute("CREATE SCHEMA ro_ns")
        table_def = _table_def("ro_ns_items")

        owner = PostgresRdbBackend()
        owner.initialize(pg_config)
        try:
            owner_db = owner.connect(namespace="ro_ns", store_db_name="test")
            owner_db.ensure_table(table_def).insert(Item(name="scoped", datasource_id=""))
        finally:
            owner.close()
        _grant_reads(admin_conn, "ro_ns")

        reader = PostgresRdbBackend()
        reader.initialize(readonly_config)
        try:
            reader_db = reader.connect(namespace="ro_ns", store_db_name="test")
            table = reader_db.ensure_table(table_def)
            assert [item.name for item in table.query(Item)] == ["scoped"]
        finally:
            reader.close()
            admin_conn.execute("DROP SCHEMA IF EXISTS ro_ns CASCADE")


class TestReadOnlyLogicalIsolation:
    def test_ensure_table_on_an_already_scoped_table(self, pg_config, admin_conn, readonly_config):
        """Under logical isolation the namespace column and scoped indexes already exist."""
        table_def = _table_def("ro_logical_items")
        _drop_table(admin_conn, "ro_logical_items")
        logical = {**pg_config, "isolation": "logical"}

        owner = PostgresRdbBackend()
        owner.initialize(logical)
        try:
            owner_db = owner.connect(namespace="tenant_ro", store_db_name="test")
            owner_db.ensure_table(table_def).insert(Item(name="tenant-row", datasource_id=""))
        finally:
            owner.close()
        _grant_reads(admin_conn)

        reader = PostgresRdbBackend()
        reader.initialize({**readonly_config, "isolation": "logical"})
        try:
            reader_db = reader.connect(namespace="tenant_ro", store_db_name="test")
            table = reader_db.ensure_table(table_def)
            assert [item.name for item in table.query(Item)] == ["tenant-row"]
        finally:
            reader.close()
            _drop_table(admin_conn, "ro_logical_items")


class TestReadOnlyVectorStore:
    def test_connect_to_an_existing_namespace_emits_no_create_schema(self, pg_config, admin_conn, readonly_config):
        """`PgVectorDb.__init__` creates the physical namespace schema on every connect."""
        schema = pa.schema([pa.field("id", pa.string()), pa.field("description", pa.string())])
        admin_conn.execute("DROP SCHEMA IF EXISTS ro_vec_ns CASCADE")
        admin_conn.execute("CREATE SCHEMA ro_vec_ns")

        owner = PgvectorBackend()
        owner.initialize(pg_config)
        try:
            owner.connect("ro_vec_ns").create_table("ro_ns_vectors", schema=schema, source_column="description")
        finally:
            owner.close()
        admin_conn.execute("INSERT INTO ro_vec_ns.ro_ns_vectors (id, description) VALUES ('v1', 'hello')")
        _grant_reads(admin_conn, "ro_vec_ns")

        reader = PgvectorBackend()
        reader.initialize(readonly_config)
        try:
            reader_db = reader.connect("ro_vec_ns")
            assert reader_db.open_table("ro_ns_vectors", source_column="description").count_rows() == 1
        finally:
            reader.close()
            admin_conn.execute("DROP SCHEMA IF EXISTS ro_vec_ns CASCADE")

    def test_open_table_on_an_already_scoped_table(self, pg_config, admin_conn, readonly_config):
        """`open_table` runs the logical-scope migration on every open; it must no-op.

        The schema deliberately carries no vector column: this exercises the scope
        migration, and a float-list column would install the `vector` extension into
        the shared container and leak into other tests.
        """
        schema = pa.schema([pa.field("id", pa.string()), pa.field("description", pa.string())])
        logical = {**pg_config, "isolation": "logical"}
        _drop_table(admin_conn, "ro_vectors")

        owner = PgvectorBackend()
        owner.initialize(logical)
        try:
            owner_db = owner.connect("tenant_ro")
            owner_db.create_table("ro_vectors", schema=schema, source_column="description")
        finally:
            owner.close()
        admin_conn.execute(
            "INSERT INTO ro_vectors (id, description, _datus_namespace) VALUES ('v1', 'hello', 'tenant_ro')"
        )
        _grant_reads(admin_conn)

        reader = PgvectorBackend()
        reader.initialize({**readonly_config, "isolation": "logical"})
        try:
            reader_db = reader.connect("tenant_ro")
            assert reader_db.table_exists("ro_vectors") is True
            table = reader_db.open_table("ro_vectors", source_column="description")
            assert table.count_rows() == 1
        finally:
            reader.close()
            _drop_table(admin_conn, "ro_vectors")
