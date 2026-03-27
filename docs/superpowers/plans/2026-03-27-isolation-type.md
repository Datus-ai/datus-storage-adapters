# Isolation Type Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `IsolationType` enum (PHYSICAL / LOGICAL) to `StorageBackendConfig` that controls multi-tenant data isolation across RDB and Vector backends.

**Architecture:** Physical isolation (default) keeps current schema-per-namespace behavior. Logical isolation puts all namespaces into a single configurable schema (`default_schema`, defaults to `public`) and auto-injects a `datasource_id` TEXT column for filtering. The isolation config lives in `StorageBackendConfig` and propagates through `config` dicts to backends.

**Tech Stack:** Python 3.12, psycopg v3, pgvector, PyArrow, pytest, testcontainers

---

### Task 1: Add IsolationType enum and update StorageBackendConfig

**Files:**
- Modify: `datus-storage-base/datus_storage_base/backend_config.py`

- [ ] **Step 1: Add IsolationType enum and update dataclasses**

```python
# At top of file, add:
from enum import Enum

class IsolationType(str, Enum):
    """Controls how multi-tenant data isolation is implemented."""
    PHYSICAL = "physical"
    LOGICAL = "logical"
```

Update `StorageBackendConfig`:

```python
@dataclass
class StorageBackendConfig:
    """Unified configuration for all storage backends."""

    isolation: IsolationType = IsolationType.PHYSICAL
    default_schema: str = "public"
    rdb: RdbBackendConfig = field(default_factory=RdbBackendConfig)
    vector: VectorBackendConfig = field(default_factory=VectorBackendConfig)
```

Update `from_dict()` to parse the new fields:

```python
    @staticmethod
    def from_dict(storage_config: Dict[str, Any]) -> "StorageBackendConfig":
        rdb_section = dict(storage_config.get("rdb", {})) if isinstance(storage_config.get("rdb", {}), dict) else {}
        vector_section = (
            dict(storage_config.get("vector", {})) if isinstance(storage_config.get("vector", {}), dict) else {}
        )

        rdb_default = RdbBackendConfig().type
        vector_default = VectorBackendConfig().type

        rdb_type = rdb_section.pop("type", rdb_default)
        vector_type = vector_section.pop("type", vector_default)

        isolation_str = storage_config.get("isolation", IsolationType.PHYSICAL.value)
        isolation = IsolationType(isolation_str) if isinstance(isolation_str, str) else isolation_str
        default_schema = storage_config.get("default_schema", "public")

        return StorageBackendConfig(
            isolation=isolation,
            default_schema=default_schema,
            rdb=RdbBackendConfig(type=rdb_type, params=rdb_section),
            vector=VectorBackendConfig(type=vector_type, params=vector_section),
        )
```

- [ ] **Step 2: Update `__init__.py` exports**

In `datus-storage-base/datus_storage_base/__init__.py`, ensure `IsolationType` is importable:

```python
from datus_storage_base.backend_config import IsolationType, StorageBackendConfig
```

- [ ] **Step 3: Verify import works**

Run: `cd /Users/lyf/GitHub/datus-storage-adapters && .venv/bin/python -c "from datus_storage_base.backend_config import IsolationType; print(IsolationType.PHYSICAL, IsolationType.LOGICAL)"`

Expected: `physical logical`

- [ ] **Step 4: Commit**

```bash
git add datus-storage-base/datus_storage_base/backend_config.py datus-storage-base/datus_storage_base/__init__.py
git commit -m "feat(base): add IsolationType enum and update StorageBackendConfig"
```

---

### Task 2: Update PgRdbDatabase and PgRdbTable for logical isolation

**Files:**
- Modify: `datus-storage-postgresql/datus_storage_postgresql/rdb/backend.py`

The key constants used throughout:

```python
DATASOURCE_ID_COLUMN = "datasource_id"
```

- [ ] **Step 1: Add DATASOURCE_ID_COLUMN constant and update PostgresRdbBackend.initialize()**

Add at module level after `logger`:

```python
from datus_storage_base.backend_config import IsolationType

DATASOURCE_ID_COLUMN = "datasource_id"
```

Update `PostgresRdbBackend.initialize()` to store isolation config:

```python
    def initialize(self, config: Dict[str, Any]) -> None:
        _REQUIRED_KEYS = ("host", "port", "user", "password", "dbname")
        missing = [k for k in _REQUIRED_KEYS if k not in config]
        if missing:
            raise ValueError(f"Missing required PostgreSQL config keys: {', '.join(missing)}")
        self._config = config
        self._isolation = IsolationType(config.get("isolation", IsolationType.PHYSICAL.value))
        self._default_schema = config.get("default_schema", "public")
```

- [ ] **Step 2: Update PostgresRdbBackend.connect() to pass isolation info**

```python
    def connect(self, namespace: str, store_db_name: str) -> PgRdbDatabase:
        pool = self._get_or_create_pool()
        db = PgRdbDatabase(
            pool=pool,
            namespace=namespace,
            store_db_name=store_db_name,
            isolation=self._isolation,
            default_schema=self._default_schema,
        )
        self._databases.append(db)
        return db
```

- [ ] **Step 3: Update PgRdbDatabase constructor for isolation-aware schema selection**

```python
class PgRdbDatabase(RdbDatabase):
    def __init__(
        self,
        pool: ConnectionPool,
        namespace: str = "",
        store_db_name: str = "",
        isolation: IsolationType = IsolationType.PHYSICAL,
        default_schema: str = "public",
    ):
        self._pool = pool
        self._namespace = namespace
        self._store_db_name = store_db_name
        self._isolation = isolation
        self._local = threading.local()

        if isolation == IsolationType.LOGICAL:
            self._schema = _validate_identifier(default_schema) if default_schema != "public" else "public"
            self._datasource_id = namespace
        else:
            self._schema = _validate_identifier(namespace) if namespace else "public"
            self._datasource_id = None

        # Ensure schema exists for non-public namespaces
        if self._schema != "public":
            with self._pool.connection() as conn:
                conn.execute(
                    psql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        psql.Identifier(self._schema)
                    )
                )
                conn.commit()
```

- [ ] **Step 4: Update PgRdbDatabase.ensure_table() to auto-inject datasource_id column and index**

```python
    def ensure_table(self, table_def: TableDefinition) -> PgRdbTable:
        # For logical isolation, inject datasource_id column if not present
        if self._isolation == IsolationType.LOGICAL:
            has_datasource_col = any(c.name == DATASOURCE_ID_COLUMN for c in table_def.columns)
            if not has_datasource_col:
                extra_col = ColumnDef(
                    name=DATASOURCE_ID_COLUMN,
                    col_type="TEXT",
                    nullable=False,
                )
                table_def = TableDefinition(
                    table_name=table_def.table_name,
                    columns=list(table_def.columns) + [extra_col],
                    indices=list(table_def.indices) + [
                        IndexDef(
                            name=f"idx_{table_def.table_name}_{DATASOURCE_ID_COLUMN}",
                            columns=[DATASOURCE_ID_COLUMN],
                        )
                    ],
                    constraints=list(table_def.constraints),
                )

        qualified = self._qualified(table_def.table_name)
        ddl_statements = self._generate_ddl(qualified, table_def)
        try:
            with self._pool.connection() as conn:
                for stmt in ddl_statements:
                    conn.execute(stmt)
                conn.commit()
        except Exception as e:
            ddl_text = "\n".join(ddl_statements)
            logger.exception("Auto-create table '%s' failed", table_def.table_name)
            raise RuntimeError(
                f"Failed to create table '{table_def.table_name}'. "
                f"Please create it manually:\n\n{ddl_text}"
            ) from e

        pk_column = "id"
        for col in table_def.columns:
            if col.primary_key:
                pk_column = col.name
                break

        return PgRdbTable(
            self._pool,
            qualified,
            self._local,
            pk_column=pk_column,
            isolation=self._isolation,
            datasource_id=self._datasource_id,
        )
```

- [ ] **Step 5: Update PgRdbTable constructor and add datasource_id injection helpers**

```python
class PgRdbTable(RdbTable):
    def __init__(
        self,
        pool: ConnectionPool,
        qualified_name: str,
        local: threading.local,
        pk_column: str = "id",
        isolation: IsolationType = IsolationType.PHYSICAL,
        datasource_id: Optional[str] = None,
    ):
        self._pool = pool
        self._qualified_name = qualified_name
        self._local = local
        self._pk_column = pk_column
        self._isolation = isolation
        self._datasource_id = datasource_id

    def _inject_datasource_where(self, where: Optional[WhereClause]) -> Optional[WhereClause]:
        """Prepend datasource_id condition for logical isolation."""
        if self._isolation != IsolationType.LOGICAL or self._datasource_id is None:
            return where
        ds_condition = (DATASOURCE_ID_COLUMN, WhereOp.EQ, self._datasource_id)
        conditions = _normalize_where(where)
        return [ds_condition] + conditions

    def _inject_datasource_into_record(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add datasource_id to a record dict for logical isolation."""
        if self._isolation != IsolationType.LOGICAL or self._datasource_id is None:
            return data
        return {**data, DATASOURCE_ID_COLUMN: self._datasource_id}
```

- [ ] **Step 6: Update insert() to inject datasource_id**

Replace the existing `insert()` method. The key change is injecting `datasource_id` into the record data dict before building the SQL:

```python
    def insert(self, record: Any) -> int:
        data = {k: v for k, v in dataclasses.asdict(record).items() if v is not None}
        data = self._inject_datasource_into_record(data)
        if not data:
            sql = f"INSERT INTO {self._qualified_name} DEFAULT VALUES RETURNING {self._pk_column}"
        else:
            columns = [_validate_identifier(k) for k in data.keys()]
            placeholders = ", ".join(["%s"] * len(columns))
            col_names = ", ".join(columns)
            sql = f"INSERT INTO {self._qualified_name} ({col_names}) VALUES ({placeholders}) RETURNING {self._pk_column}"
        try:
            with self._auto_conn() as conn:
                cursor = conn.execute(sql, tuple(data.values()))
                row = cursor.fetchone()
                if row:
                    if isinstance(row, dict):
                        return next(iter(row.values()))
                    return row[0]
                return 0
        except Exception as e:
            exc_type_name = type(e).__name__
            if "UniqueViolation" in exc_type_name:
                raise UniqueViolationError(str(e)) from e
            if "IntegrityError" in exc_type_name:
                raise IntegrityError(str(e)) from e
            raise
```

- [ ] **Step 7: Update query() to inject datasource_id WHERE condition**

```python
    def query(
        self,
        model: Type[T],
        where: Optional[WhereClause] = None,
        columns: Optional[List[str]] = None,
        order_by: Optional[List[str]] = None,
    ) -> List[T]:
        where = self._inject_datasource_where(where)
        if columns:
            for c in columns:
                _validate_identifier(c)
        col_str = ", ".join(columns) if columns else "*"
        where_sql, params = self._build_where(where)
        order_sql = self._build_order_by(order_by)
        sql = f"SELECT {col_str} FROM {self._qualified_name}{where_sql}{order_sql}"
        with self._auto_conn() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            # Filter out datasource_id from model kwargs in logical isolation
            if self._isolation == IsolationType.LOGICAL:
                results = []
                for row in rows:
                    row_dict = dict(row)
                    row_dict.pop(DATASOURCE_ID_COLUMN, None)
                    results.append(model(**row_dict))
                return results
            return [model(**dict(row)) for row in rows]
```

- [ ] **Step 8: Update update() and delete() to inject datasource_id WHERE condition**

```python
    def update(self, data: Dict[str, Any], where: Optional[WhereClause] = None) -> int:
        if not data:
            return 0
        where = self._inject_datasource_where(where)
        for col in data.keys():
            _validate_identifier(col)
        set_parts = [f"{col} = %s" for col in data.keys()]
        set_sql = ", ".join(set_parts)
        where_sql, where_params = self._build_where(where)
        sql = f"UPDATE {self._qualified_name} SET {set_sql}{where_sql}"
        params = list(data.values()) + where_params
        try:
            with self._auto_conn() as conn:
                cursor = conn.execute(sql, params)
                return cursor.rowcount
        except Exception as e:
            exc_type_name = type(e).__name__
            if "UniqueViolation" in exc_type_name:
                raise UniqueViolationError(str(e)) from e
            if "IntegrityError" in exc_type_name:
                raise IntegrityError(str(e)) from e
            raise

    def delete(self, where: Optional[WhereClause] = None) -> int:
        where = self._inject_datasource_where(where)
        where_sql, params = self._build_where(where)
        sql = f"DELETE FROM {self._qualified_name}{where_sql}"
        with self._auto_conn() as conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount
```

- [ ] **Step 9: Update upsert() to inject datasource_id**

```python
    def upsert(self, record: Any, conflict_columns: List[str]) -> None:
        data = {k: v for k, v in dataclasses.asdict(record).items() if v is not None}
        data = self._inject_datasource_into_record(data)
        if not data:
            raise ValueError("Cannot upsert a record with no non-None fields")
        columns = [_validate_identifier(k) for k in data.keys()]
        placeholders = ", ".join(["%s"] * len(columns))
        col_names = ", ".join(columns)
        conflict_cols = ", ".join(_validate_identifier(c) for c in conflict_columns)
        update_cols = [c for c in columns if c not in conflict_columns]
        update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

        if update_set:
            sql = (
                f"INSERT INTO {self._qualified_name} ({col_names}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_set}"
            )
        else:
            sql = (
                f"INSERT INTO {self._qualified_name} ({col_names}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_cols}) DO NOTHING"
            )
        try:
            with self._auto_conn() as conn:
                conn.execute(sql, tuple(data.values()))
        except Exception as e:
            exc_type_name = type(e).__name__
            if "UniqueViolation" in exc_type_name:
                raise UniqueViolationError(str(e)) from e
            if "IntegrityError" in exc_type_name:
                raise IntegrityError(str(e)) from e
            raise
```

- [ ] **Step 10: Verify existing physical isolation tests still pass**

Run: `.venv/bin/pytest datus-storage-postgresql/tests/test_pg_rdb_backend.py -v`

Expected: All existing tests PASS (no behavioral change for physical isolation).

- [ ] **Step 11: Commit**

```bash
git add datus-storage-postgresql/datus_storage_postgresql/rdb/backend.py
git commit -m "feat(pg-rdb): support logical isolation with datasource_id"
```

---

### Task 3: Update PgVectorDb and PgVectorTable for logical isolation

**Files:**
- Modify: `datus-storage-postgresql/datus_storage_postgresql/vector/backend.py`

- [ ] **Step 1: Add imports and constant**

```python
from datus_storage_base.backend_config import IsolationType

DATASOURCE_ID_COLUMN = "datasource_id"
```

- [ ] **Step 2: Update PgvectorBackend.initialize() and connect()**

```python
    def initialize(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._isolation = IsolationType(config.get("isolation", IsolationType.PHYSICAL.value))
        self._default_schema = config.get("default_schema", "public")

    def connect(self, namespace: str) -> PgVectorDb:
        pool = self._get_or_create_pool()
        db = PgVectorDb(
            pool=pool,
            config=self._config,
            namespace=namespace,
            isolation=self._isolation,
            default_schema=self._default_schema,
        )
        self._connections.append(db)
        return db
```

- [ ] **Step 3: Update PgVectorDb constructor for isolation-aware schema**

```python
class PgVectorDb(VectorDatabase):
    def __init__(
        self,
        pool: ConnectionPool,
        config: Dict[str, Any],
        namespace: str = "",
        isolation: IsolationType = IsolationType.PHYSICAL,
        default_schema: str = "public",
    ):
        self._pool = pool
        self._config = config
        self._namespace = namespace
        self._isolation = isolation
        self._table_cache: Dict[tuple, PgVectorTable] = {}

        if isolation == IsolationType.LOGICAL:
            self._schema = _validate_identifier(default_schema) if default_schema != "public" else "public"
            self._datasource_id = namespace
        else:
            self._schema = _validate_identifier(namespace) if namespace else "public"
            self._datasource_id = None

        if self._schema != "public":
            with self._pool.connection() as conn:
                conn.execute(
                    psql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        psql.Identifier(self._schema)
                    )
                )
                conn.commit()
```

- [ ] **Step 4: Update PgVectorDb.create_table() to inject datasource_id into schema**

In `create_table()`, before generating DDL, append `datasource_id` to the PyArrow schema if logical isolation and the column is not already present:

```python
    def create_table(
        self,
        table_name: str,
        schema: Optional[pa.Schema] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
        vector_column: str = "",
        source_column: str = "",
        exist_ok: bool = True,
        unique_columns: Optional[List[str]] = None,
    ) -> PgVectorTable:
        qualified = self._qualified(table_name)
        vector_column = vector_column or "vector"
        source_column = source_column or "description"
        vector_dim = embedding_function.ndims() if embedding_function else 384

        column_names = []
        if schema is not None:
            if isinstance(schema, pa.Schema):
                # Inject datasource_id column for logical isolation
                if self._isolation == IsolationType.LOGICAL:
                    if DATASOURCE_ID_COLUMN not in schema.names:
                        schema = schema.append(pa.field(DATASOURCE_ID_COLUMN, pa.string()))
                ddl = schema_to_create_table_sql(qualified, schema, unique_columns=unique_columns)
                column_names = [f.name for f in schema]
            else:
                raise TypeError(f"Unsupported schema type: {type(schema)}")

            with self._pool.connection() as conn:
                conn.execute(ddl)
                # Create B-tree index on datasource_id for logical isolation
                if self._isolation == IsolationType.LOGICAL:
                    table_token = table_name
                    idx_name = f"idx_{table_token}_{DATASOURCE_ID_COLUMN}"
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS {idx_name} "
                        f"ON {qualified} ({DATASOURCE_ID_COLUMN})"
                    )
                conn.commit()
        elif not exist_ok:
            raise ValueError(f"Schema is required to create table '{table_name}'")
        else:
            if not self.table_exists(table_name):
                raise ValueError(
                    f"Table '{table_name}' does not exist and no schema was provided to create it."
                )

        table = PgVectorTable(
            table_name=qualified,
            pool=self._pool,
            embedding_fn=embedding_function,
            vector_column=vector_column,
            source_column=source_column,
            vector_dim=vector_dim,
            column_names=column_names,
            isolation=self._isolation,
            datasource_id=self._datasource_id,
        )
        cache_key = (table_name, id(embedding_function), vector_dim, vector_column, source_column)
        self._table_cache[cache_key] = table
        return table
```

- [ ] **Step 5: Update PgVectorDb.open_table() to pass isolation params**

In `open_table()`, pass `isolation` and `datasource_id` when constructing `PgVectorTable`:

```python
        table = PgVectorTable(
            table_name=qualified,
            pool=self._pool,
            embedding_fn=embedding_function,
            vector_column=vector_column,
            source_column=source_column,
            vector_dim=vector_dim,
            column_names=column_names,
            isolation=self._isolation,
            datasource_id=self._datasource_id,
        )
```

- [ ] **Step 6: Update PgVectorTable constructor and add injection helpers**

```python
class PgVectorTable(VectorTable):
    def __init__(
        self,
        table_name: str,
        pool: ConnectionPool,
        embedding_fn: Any = None,
        vector_column: str = "vector",
        source_column: str = "description",
        vector_dim: int = 384,
        column_names: Optional[List[str]] = None,
        isolation: IsolationType = IsolationType.PHYSICAL,
        datasource_id: Optional[str] = None,
    ):
        self._table_name = table_name
        self._pool = pool
        self._embedding_fn = embedding_fn
        self._vector_column = vector_column
        self._source_column = source_column
        self._vector_dim = vector_dim
        self._column_names = column_names or []
        self._isolation = isolation
        self._datasource_id = datasource_id

    def _ds_where_clause(self, existing_compiled: Optional[str] = None) -> str:
        """Build WHERE clause with datasource_id for logical isolation.

        Args:
            existing_compiled: Already-compiled WHERE string (without 'WHERE' keyword).

        Returns:
            Combined WHERE string (without 'WHERE' keyword), or empty string.
        """
        if self._isolation != IsolationType.LOGICAL or self._datasource_id is None:
            return existing_compiled or ""
        ds_cond = f"{DATASOURCE_ID_COLUMN} = '{self._datasource_id}'"
        if existing_compiled:
            return f"{ds_cond} AND {existing_compiled}"
        return ds_cond

    def _inject_datasource_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add datasource_id column to DataFrame for logical isolation."""
        if self._isolation != IsolationType.LOGICAL or self._datasource_id is None:
            return df
        if DATASOURCE_ID_COLUMN not in df.columns:
            df = df.copy()
            df[DATASOURCE_ID_COLUMN] = self._datasource_id
        return df
```

- [ ] **Step 7: Update add() and merge_insert() to inject datasource_id**

```python
    def add(self, data: pd.DataFrame) -> None:
        df = self._inject_datasource_df(data)
        df = self._compute_embeddings_for_insert(df)
        self._insert_dataframe(df)

    def merge_insert(self, data: pd.DataFrame, on_column: str) -> None:
        df = self._inject_datasource_df(data)
        df = self._compute_embeddings_for_insert(df)
        self._upsert_dataframe(df, on_column)
```

- [ ] **Step 8: Update delete() and update() to inject datasource_id**

```python
    def delete(self, where: WhereExpr) -> None:
        compiled = build_where(where) if not isinstance(where, str) else where
        combined = self._ds_where_clause(compiled)
        if combined:
            sql = f"DELETE FROM {self._table_name} WHERE {combined}"
            with self._pool.connection() as conn:
                conn.execute(sql)
                conn.commit()

    def update(self, where: WhereExpr, values: Dict[str, Any]) -> None:
        compiled = build_where(where) if not isinstance(where, str) else where
        set_parts = []
        params = []
        for col, val in values.items():
            _validate_identifier(col)
            set_parts.append(f"{col} = %s")
            params.append(val)
        set_clause = ", ".join(set_parts)
        combined = self._ds_where_clause(compiled)
        where_clause = f" WHERE {combined}" if combined else ""
        sql = f"UPDATE {self._table_name} SET {set_clause}{where_clause}"
        with self._pool.connection() as conn:
            conn.execute(sql, params)
            conn.commit()
```

- [ ] **Step 9: Update search_vector(), search_all(), count_rows() to inject datasource_id**

For `search_vector()`:
```python
    def search_vector(self, query_text, vector_column, top_n, where=None, select_fields=None):
        compiled = build_where(where) if not isinstance(where, str) else where
        combined = self._ds_where_clause(compiled)
        query_embedding = self._compute_query_embedding(query_text)

        columns = self._validate_select_fields(select_fields) if select_fields else self._select_columns()
        _validate_identifier(vector_column)
        where_clause = f"WHERE {combined}" if combined else ""
        sql = (
            f"SELECT {columns} FROM {self._table_name} "
            f"{where_clause} "
            f"ORDER BY {vector_column} <=> %s::vector "
            f"LIMIT %s"
        )
        with self._pool.connection() as conn:
            rows = conn.execute(sql, (str(query_embedding), top_n)).fetchall()
        return self._rows_to_arrow(rows, select_fields)
```

For `search_all()`:
```python
    def search_all(self, where=None, select_fields=None, limit=None):
        compiled = build_where(where) if not isinstance(where, str) else where
        combined = self._ds_where_clause(compiled)
        columns = self._validate_select_fields(select_fields) if select_fields else self._select_columns()
        where_clause = f"WHERE {combined}" if combined else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        sql = f"SELECT {columns} FROM {self._table_name} {where_clause} {limit_clause}"
        with self._pool.connection() as conn:
            rows = conn.execute(sql).fetchall()
        return self._rows_to_arrow(rows, select_fields)
```

For `count_rows()`:
```python
    def count_rows(self, where=None):
        compiled = build_where(where) if not isinstance(where, str) else where
        combined = self._ds_where_clause(compiled)
        where_clause = f"WHERE {combined}" if combined else ""
        sql = f"SELECT COUNT(*) AS cnt FROM {self._table_name} {where_clause}"
        with self._pool.connection() as conn:
            row = conn.execute(sql).fetchone()
            if isinstance(row, dict):
                return row["cnt"]
            return row[0] if row else 0
```

- [ ] **Step 10: Update _select_columns() to exclude datasource_id from default SELECT**

```python
    def _select_columns(self) -> str:
        """Build the default SELECT column list, excluding datasource_id in logical mode."""
        if self._column_names:
            cols = self._column_names
            if self._isolation == IsolationType.LOGICAL:
                cols = [c for c in cols if c != DATASOURCE_ID_COLUMN]
            return ", ".join(cols)
        return "*"
```

- [ ] **Step 11: Verify existing physical isolation tests still pass**

Run: `.venv/bin/pytest datus-storage-postgresql/tests/test_pgvector_backend.py -v`

Expected: All existing tests PASS.

- [ ] **Step 12: Commit**

```bash
git add datus-storage-postgresql/datus_storage_postgresql/vector/backend.py
git commit -m "feat(pg-vector): support logical isolation with datasource_id"
```

---

### Task 4: Update testing helpers for logical isolation clear_data

**Files:**
- Modify: `datus-storage-postgresql/datus_storage_postgresql/rdb/testing.py`
- Modify: `datus-storage-postgresql/datus_storage_postgresql/vector/testing.py`

- [ ] **Step 1: Update PostgresRdbTestEnv.clear_data() for logical isolation**

Add `isolation` parameter to `clear_data()`. Since the ABC signature uses `(self, namespace)`, we store the isolation mode and pass it at setup time:

In `PostgresRdbTestEnv.__init__()`, add `self._isolation = IsolationType.PHYSICAL`.

Add a `set_isolation(isolation)` method:

```python
from datus_storage_base.backend_config import IsolationType

class PostgresRdbTestEnv(RdbTestEnv):
    def __init__(self):
        self._container = None
        self._config: Optional[Dict[str, Any]] = None
        self._isolation = IsolationType.PHYSICAL
        self._default_schema = "public"

    def set_isolation(self, isolation: IsolationType, default_schema: str = "public") -> None:
        self._isolation = isolation
        self._default_schema = default_schema
```

Update `clear_data()`:

```python
    def clear_data(self, namespace: str) -> None:
        if self._config is None:
            return

        import psycopg
        from psycopg import sql

        conninfo = (
            f"host={self._config['host']} port={self._config['port']} "
            f"user={self._config['user']} password={self._config['password']} "
            f"dbname={self._config['dbname']}"
        )
        with psycopg.connect(conninfo, autocommit=True) as conn:
            if self._isolation == IsolationType.LOGICAL:
                # Delete rows by datasource_id in all tables within the schema
                schema = self._default_schema
                rows = conn.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = %s",
                    (schema,),
                ).fetchall()
                for row in rows:
                    tbl = row[0] if not isinstance(row, dict) else row["tablename"]
                    qualified = f"{schema}.{tbl}" if schema != "public" else tbl
                    conn.execute(
                        f"DELETE FROM {qualified} WHERE datasource_id = %s",
                        (namespace,),
                    )
            else:
                if namespace:
                    conn.execute(
                        sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                            sql.Identifier(namespace)
                        )
                    )
                else:
                    rows = conn.execute(
                        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                    ).fetchall()
                    for row in rows:
                        tbl = row[0] if not isinstance(row, dict) else row["tablename"]
                        conn.execute(
                            sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                                sql.Identifier(tbl)
                            )
                        )
```

- [ ] **Step 2: Apply same pattern to PgvectorTestEnv.clear_data()**

In `datus-storage-postgresql/datus_storage_postgresql/vector/testing.py`, apply the same changes to `PgvectorTestEnv`:

```python
from datus_storage_base.backend_config import IsolationType

class PgvectorTestEnv(VectorTestEnv):
    def __init__(self):
        self._dbname: Optional[str] = None
        self._config: Optional[Dict[str, Any]] = None
        self._isolation = IsolationType.PHYSICAL
        self._default_schema = "public"

    def set_isolation(self, isolation: IsolationType, default_schema: str = "public") -> None:
        self._isolation = isolation
        self._default_schema = default_schema
```

Update `clear_data()` with the same logical isolation branch (DELETE by datasource_id instead of DROP).

- [ ] **Step 3: Commit**

```bash
git add datus-storage-postgresql/datus_storage_postgresql/rdb/testing.py datus-storage-postgresql/datus_storage_postgresql/vector/testing.py
git commit -m "feat(pg-testing): support logical isolation in clear_data"
```

---

### Task 5: Add RDB logical isolation tests

**Files:**
- Modify: `datus-storage-postgresql/tests/test_pg_rdb_backend.py`

- [ ] **Step 1: Add logical isolation fixture**

```python
@pytest.fixture
def logical_backend(pg_config):
    """Create a PostgresRdbBackend with logical isolation."""
    config = {**pg_config, "isolation": "logical"}
    b = PostgresRdbBackend()
    b.initialize(config)
    yield b
    b.close()


@pytest.fixture
def logical_db(logical_backend):
    """Connect with a namespace under logical isolation."""
    return logical_backend.connect(namespace="tenant_a", store_db_name="test")
```

- [ ] **Step 2: Write TestLogicalIsolation test class**

```python
class TestLogicalIsolation:

    def test_table_in_public_schema(self, logical_db, test_table_def):
        """Logical isolation uses public schema, not namespace as schema."""
        table = logical_db.ensure_table(test_table_def)
        assert table.table_name == "test_items"  # no schema prefix

    def test_datasource_id_column_created(self, logical_db, test_table_def):
        """ensure_table auto-adds datasource_id column."""
        logical_db.ensure_table(test_table_def)
        with logical_db.get_connection() as conn:
            rows = logical_db.execute_query(
                conn,
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = %s",
                ("test_items", "datasource_id"),
            )
            assert len(rows) == 1

    def test_datasource_id_index_created(self, logical_db, test_table_def):
        """ensure_table auto-creates B-tree index on datasource_id."""
        logical_db.ensure_table(test_table_def)
        with logical_db.get_connection() as conn:
            rows = logical_db.execute_query(
                conn,
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = %s AND indexname LIKE %s",
                ("test_items", "%datasource_id%"),
            )
            assert len(rows) >= 1

    def test_insert_injects_datasource_id(self, logical_db, test_table_def):
        """Insert auto-injects datasource_id = namespace."""
        table = logical_db.ensure_table(test_table_def)
        table.insert(TestItem(name="li1", value="v1"))
        with logical_db.get_connection() as conn:
            rows = logical_db.execute_query(
                conn,
                "SELECT datasource_id FROM test_items WHERE name = %s",
                ("li1",),
            )
            assert rows[0]["datasource_id"] == "tenant_a"

    def test_query_filters_by_datasource_id(self, logical_backend, test_table_def):
        """Query only returns rows for the connected namespace."""
        db_a = logical_backend.connect(namespace="tenant_a", store_db_name="test")
        db_b = logical_backend.connect(namespace="tenant_b", store_db_name="test")

        tbl_a = db_a.ensure_table(test_table_def)
        tbl_b = db_b.ensure_table(test_table_def)

        tbl_a.insert(TestItem(name="a1", value="from_a"))
        tbl_b.insert(TestItem(name="b1", value="from_b"))
        tbl_b.insert(TestItem(name="b2", value="from_b"))

        assert len(tbl_a.query(TestItem)) == 1
        assert len(tbl_b.query(TestItem)) == 2

    def test_update_scoped_to_datasource(self, logical_backend, test_table_def):
        """Update only affects rows for the connected namespace."""
        db_a = logical_backend.connect(namespace="tenant_a", store_db_name="test")
        db_b = logical_backend.connect(namespace="tenant_b", store_db_name="test")

        tbl_a = db_a.ensure_table(test_table_def)
        tbl_b = db_b.ensure_table(test_table_def)

        tbl_a.insert(TestItem(name="ua1", value="old"))
        tbl_b.insert(TestItem(name="ub1", value="old"))

        count = tbl_a.update({"value": "new"})
        assert count == 1
        assert tbl_a.query(TestItem)[0].value == "new"
        assert tbl_b.query(TestItem)[0].value == "old"

    def test_delete_scoped_to_datasource(self, logical_backend, test_table_def):
        """Delete only affects rows for the connected namespace."""
        db_a = logical_backend.connect(namespace="tenant_a", store_db_name="test")
        db_b = logical_backend.connect(namespace="tenant_b", store_db_name="test")

        tbl_a = db_a.ensure_table(test_table_def)
        tbl_b = db_b.ensure_table(test_table_def)

        tbl_a.insert(TestItem(name="da1", value="v"))
        tbl_b.insert(TestItem(name="db1", value="v"))

        tbl_a.delete()
        assert len(tbl_a.query(TestItem)) == 0
        assert len(tbl_b.query(TestItem)) == 1

    def test_upsert_injects_datasource_id(self, logical_db):
        """Upsert auto-injects datasource_id."""
        table_def = TableDefinition(
            table_name="logical_upsert",
            columns=[
                ColumnDef(name="key_col", col_type="TEXT", primary_key=True),
                ColumnDef(name="data", col_type="TEXT"),
            ],
        )
        table = logical_db.ensure_table(table_def)
        table.upsert(UpsertRecord(key_col="k1", data="d1"), ["key_col"])
        results = table.query(UpsertRecord)
        assert len(results) == 1
        assert results[0].data == "d1"

    def test_query_result_excludes_datasource_id(self, logical_db, test_table_def):
        """Query results should not include datasource_id in model fields."""
        table = logical_db.ensure_table(test_table_def)
        table.insert(TestItem(name="excl1", value="v1"))
        results = table.query(TestItem)
        assert len(results) == 1
        assert not hasattr(results[0], "datasource_id") or results[0].__class__.__dataclass_fields__.keys() == TestItem.__dataclass_fields__.keys()
```

- [ ] **Step 3: Run all RDB tests**

Run: `.venv/bin/pytest datus-storage-postgresql/tests/test_pg_rdb_backend.py -v`

Expected: All existing tests PASS + all new logical isolation tests PASS.

- [ ] **Step 4: Commit**

```bash
git add datus-storage-postgresql/tests/test_pg_rdb_backend.py
git commit -m "test(pg-rdb): add logical isolation tests"
```

---

### Task 6: Add Vector logical isolation tests

**Files:**
- Modify: `datus-storage-postgresql/tests/test_pgvector_backend.py`

- [ ] **Step 1: Add logical isolation fixtures**

```python
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


@pytest.fixture
def logical_table(logical_db, test_schema, embedding_function):
    """Create a test table under logical isolation."""
    logical_db.drop_table("logical_vectors", ignore_missing=True)
    tbl = logical_db.create_table(
        "logical_vectors",
        schema=test_schema,
        embedding_function=embedding_function,
        vector_column="vector",
        source_column="description",
    )
    return tbl
```

- [ ] **Step 2: Write TestVectorLogicalIsolation test class**

```python
class TestVectorLogicalIsolation:

    def test_table_in_public_schema(self, logical_table):
        """Logical isolation uses public schema."""
        assert "." not in logical_table.table_name  # no schema prefix

    def test_datasource_id_column_created(self, logical_db, logical_table):
        """create_table auto-adds datasource_id column."""
        with logical_db.pool.connection() as conn:
            rows = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = %s",
                ("logical_vectors", "datasource_id"),
            ).fetchall()
            assert len(rows) == 1

    def test_add_injects_datasource_id(self, logical_db, logical_table):
        """add() auto-injects datasource_id."""
        logical_table.add(_sample_df(["la1"]))
        with logical_db.pool.connection() as conn:
            rows = conn.execute(
                "SELECT datasource_id FROM logical_vectors WHERE id = 'la1'"
            ).fetchall()
            val = rows[0]["datasource_id"] if isinstance(rows[0], dict) else rows[0][0]
            assert val == "tenant_a"

    def test_search_all_filters_by_datasource(self, logical_backend, test_schema, embedding_function):
        """search_all only returns rows for the connected namespace."""
        db_a = logical_backend.connect("tenant_a")
        db_b = logical_backend.connect("tenant_b")

        db_a.drop_table("shared_vec", ignore_missing=True)
        tbl_a = db_a.create_table("shared_vec", schema=test_schema, embedding_function=embedding_function)
        tbl_b = db_b.open_table("shared_vec", embedding_function=embedding_function)

        tbl_a.add(_sample_df(["a1"]))
        tbl_b.add(_sample_df(["b1", "b2"]))

        assert tbl_a.count_rows() == 1
        assert tbl_b.count_rows() == 2

    def test_delete_scoped_to_datasource(self, logical_backend, test_schema, embedding_function):
        """delete() only affects rows for the connected namespace."""
        db_a = logical_backend.connect("tenant_a")
        db_b = logical_backend.connect("tenant_b")

        db_a.drop_table("del_vec", ignore_missing=True)
        tbl_a = db_a.create_table("del_vec", schema=test_schema, embedding_function=embedding_function)
        tbl_b = db_b.open_table("del_vec", embedding_function=embedding_function)

        tbl_a.add(_sample_df(["da1"]))
        tbl_b.add(_sample_df(["db1"]))

        tbl_a.delete(eq("id", "da1"))
        assert tbl_a.count_rows() == 0
        assert tbl_b.count_rows() == 1

    def test_update_scoped_to_datasource(self, logical_backend, test_schema, embedding_function):
        """update() only affects rows for the connected namespace."""
        db_a = logical_backend.connect("tenant_a")
        db_b = logical_backend.connect("tenant_b")

        db_a.drop_table("upd_vec", ignore_missing=True)
        tbl_a = db_a.create_table("upd_vec", schema=test_schema, embedding_function=embedding_function)
        tbl_b = db_b.open_table("upd_vec", embedding_function=embedding_function)

        tbl_a.add(_sample_df(["ua1"], categories=["old"]))
        tbl_b.add(_sample_df(["ub1"], categories=["old"]))

        tbl_a.update(eq("id", "ua1"), {"category": "new"})
        result_a = tbl_a.search_all()
        result_b = tbl_b.search_all()
        assert result_a.column("category")[0].as_py() == "new"
        assert result_b.column("category")[0].as_py() == "old"

    def test_search_all_excludes_datasource_id_from_results(self, logical_table):
        """Default SELECT should not include datasource_id column."""
        logical_table.add(_sample_df(["ex1"]))
        result = logical_table.search_all()
        assert "datasource_id" not in result.column_names
```

- [ ] **Step 3: Run all vector tests**

Run: `.venv/bin/pytest datus-storage-postgresql/tests/test_pgvector_backend.py -v`

Expected: All existing tests PASS + all new logical isolation tests PASS.

- [ ] **Step 4: Commit**

```bash
git add datus-storage-postgresql/tests/test_pgvector_backend.py
git commit -m "test(pg-vector): add logical isolation tests"
```

---

### Task 7: Run full test suite and verify

- [ ] **Step 1: Run all tests**

Run: `.venv/bin/pytest datus-storage-postgresql/tests/ -v`

Expected: All tests PASS.

- [ ] **Step 2: Final commit (if any fixups needed)**
