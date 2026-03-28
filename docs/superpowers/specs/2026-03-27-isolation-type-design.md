# Isolation Type Configuration Design

## Summary

Add an `IsolationType` enum to `StorageBackendConfig` that controls how multi-tenant data isolation is implemented across all storage backends. Two modes: **physical** (current behavior, namespace maps to schema/database/path) and **logical** (shared table with `datasource_id` column filtering).

## Motivation

Physical isolation (one schema per namespace) is clean but creates many schemas in multi-tenant deployments. Logical isolation shares tables across tenants, reducing operational overhead at the cost of requiring per-query filtering.

## Configuration

### Enum

```python
class IsolationType(str, Enum):
    PHYSICAL = "physical"
    LOGICAL = "logical"
```

Location: `datus_storage_base/backend_config.py`

### StorageBackendConfig changes

```python
@dataclass
class StorageBackendConfig:
    isolation: IsolationType = IsolationType.PHYSICAL  # default = backward compatible
    rdb: RdbBackendConfig = field(default_factory=RdbBackendConfig)
    vector: VectorBackendConfig = field(default_factory=VectorBackendConfig)
```

`from_dict()` parses the top-level `isolation` key.

### Config propagation

`isolation` is passed into the backend `config` dict (alongside host/port/etc). Backends read it in `initialize()` and store it. `connect()` uses it to decide behavior. Logical isolation always uses the `public` schema.

## Behavior by Mode

### Physical Isolation (default, current behavior)

- `connect(namespace)` creates schema = namespace (empty = `public`)
- No `datasource_id` column
- Truncate/clear: `DROP TABLE` or `DROP SCHEMA CASCADE`
- No changes to existing code paths

### Logical Isolation

- `connect(namespace)` uses `public` schema, ignores namespace for schema creation
- `datasource_id` column (TEXT, NOT NULL) auto-injected into every table
- B-tree index auto-created on `datasource_id`
- `datasource_id` value = namespace passed to `connect()`
- Truncate/clear: `DELETE FROM table WHERE datasource_id = ?`

## RDB Backend Changes

### PgRdbDatabase (logical mode)

- Constructor: schema = `public` instead of namespace
- `ensure_table(table_def)`: appends `datasource_id TEXT NOT NULL` to column list, adds B-tree index `idx_{table}_{datasource_id}`
- Stores namespace as `self._datasource_id` for passing to table handles

### PgRdbTable (logical mode)

All operations automatically scope to the current datasource_id:

| Operation | Change |
|-----------|--------|
| `insert(record)` | Inject `datasource_id = self._datasource_id` into column values |
| `query(model, where)` | Append `AND datasource_id = ?` to WHERE clause |
| `update(data, where)` | Append `AND datasource_id = ?` to WHERE clause |
| `delete(where)` | Append `AND datasource_id = ?` to WHERE clause |
| `upsert(record, conflict_columns)` | Inject `datasource_id` into record values |

Implementation approach: the table receives an `isolation_type` and `datasource_id` parameter. When `LOGICAL`, a private helper prepends the datasource_id condition to all WHERE clauses and injects it into all write payloads.

## Vector Backend Changes

### PgVectorDb (logical mode)

- Constructor: schema = `public` instead of namespace
- `create_table()`: appends `datasource_id` (string, not null) to PyArrow schema before generating DDL, adds B-tree index
- Stores namespace as datasource_id for passing to table handles

### PgVectorTable (logical mode)

| Operation | Change |
|-----------|--------|
| `add(df)` | Inject `datasource_id` column into DataFrame |
| `merge_insert(df, on_column)` | Inject `datasource_id` column into DataFrame |
| `search_vector()` | Append `AND datasource_id = ?` to WHERE |
| `search_hybrid()` | Append `AND datasource_id = ?` to WHERE |
| `search_all()` | Append `AND datasource_id = ?` to WHERE |
| `count_rows()` | Append `AND datasource_id = ?` to WHERE |
| `delete(where)` | Append `AND datasource_id = ?` to WHERE |
| `update(where, values)` | Append `AND datasource_id = ?` to WHERE |

Implementation approach: same pattern as RDB. Table receives `isolation_type` and `datasource_id`. A helper injects the condition.

## Base Layer

The abstract base classes (`BaseRdbBackend`, `BaseVectorBackend`, `RdbTable`, `VectorTable`) do NOT enforce isolation logic. The `IsolationType` enum and config fields are defined in `backend_config.py`. Each adapter implements the isolation behavior internally based on config values.

## Unchanged Components

- `conditions.py` (condition DSL): no changes
- `RdbRegistry` / `VectorRegistry`: no changes
- Entry point registration: no changes
- `testing.py`: test environment ABCs remain the same

## Clear/Truncate Semantics

Both `RdbTestEnv.clear_data()` and `VectorTestEnv.clear_data()` already exist. The PostgreSQL testing implementations will branch on isolation type:

- **Physical**: `DROP SCHEMA {namespace} CASCADE` (current behavior)
- **Logical**: `DELETE FROM {table} WHERE datasource_id = '{namespace}'` for each table in the schema

## Constants

- Column name: `datasource_id` (hardcoded, not configurable)
- Column type: `TEXT NOT NULL`
- Index: B-tree, named `idx_{table}_datasource_id`
