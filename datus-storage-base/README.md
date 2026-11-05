# datus-storage-base

Abstract interfaces for Datus storage backends (vector & RDB).

This package provides the base classes and registries that all storage adapter implementations depend on:

- `BaseRdbBackend` / `RdbDatabase` / `RdbTable` — relational database abstractions
- `BaseVectorBackend` / `VectorDatabase` / `VectorTable` — vector database abstractions
- `FtsField` / `FtsSpec` / `FtsIndexStatus` — backend-independent full-text search contracts
- `RdbRegistry` / `VectorRegistry` — entry-point-based backend discovery
- `BackendConfig` — unified backend configuration
- Condition AST (`eq`, `ne`, `gt`, `and_`, `or_`, …) for portable query filters

## Installation

```bash
pip install datus-storage-base
```

## Usage

This package is not used directly. Install a concrete adapter (e.g. `datus-storage-postgresql`) which depends on this package automatically.

`build_where()` renders membership as native `IN`, keeping large retained-ID
lists shallow enough for LanceDB deletion. An empty list is always false, and
`None` in a membership list is an explicit `IS NULL` alternative. Negating the
condition preserves these semantics.

## Tests

From the repository root:

```bash
pip install -e './datus-storage-base[dev,lancedb-test]'
pytest datus-storage-base/tests/unit datus-storage-base/tests/integration
```

The LanceDB regression runs deletion in a subprocess and checks retained IDs,
stale records, model/datasource scope, and NULL handling. A native crash fails
that test without terminating the pytest process.

## License

Apache-2.0
