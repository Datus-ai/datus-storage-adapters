"""pgvector implementation of BaseVectorBackend using psycopg v3.

Three-layer architecture:
  PgvectorBackend(BaseVectorBackend)  - lifecycle & embedding config
      |
      +-- connect(namespace) -> PgVectorDb(VectorDatabase)
                                    |
                                    +-- open_table(name) -> PgVectorTable(VectorTable)
"""

import hashlib
import json
import logging
import re
import threading
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
from psycopg import sql as psql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from datus_storage_base.backend_config import LOGICAL_NAMESPACE_COLUMN, IsolationType
from datus_storage_base.conditions import WhereExpr, build_where
from datus_storage_base.vector.base import BaseVectorBackend, EmbeddingFunction, VectorDatabase, VectorTable
from datus_storage_base.vector.fts import FtsField, FtsIndexStatus, FtsSpec, FtsSpecInput, normalize_fts_spec
from datus_storage_postgresql.vector.schema_converter import schema_to_create_table_sql

logger = logging.getLogger(__name__)

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_SCHEMA_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
_FTS_METADATA_TABLE = "_datus_fts_specs"
_FTS_NGRAM_FUNCTION = "_datus_fts_ngrams"
_FTS_GENERATED_COLUMN_PREFIX = "_datus_fts_vector_"
_FTS_BACKEND_VERSION = 1
_FTS_SCORE_COLUMNS = {"_score", "_relevance_score", "_distance"}


def _validate_identifier(name: str) -> str:
    """Validate that a name is a safe SQL identifier."""
    if not _SAFE_IDENTIFIER.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def _physical_schema_name(namespace: str) -> str:
    """Map an opaque Datus namespace to a stable PostgreSQL schema name."""

    if not namespace:
        return "public"
    if _SAFE_SCHEMA_IDENTIFIER.fullmatch(namespace) and len(namespace) <= 63:
        return namespace
    normalized = re.sub(r"[^a-z0-9_]", "_", namespace.lower()).strip("_") or "namespace"
    if normalized[0].isdigit():
        normalized = f"n_{normalized}"
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:10]
    return f"{normalized[:52]}_{digest}"


def _ensure_postgres_extension(conn: Any, extension: str) -> None:
    installed = conn.execute("SELECT 1 FROM pg_extension WHERE extname = %s", (extension,)).fetchone()
    if installed:
        return
    available = conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = %s", (extension,)).fetchone()
    if not available:
        raise RuntimeError(f"PostgreSQL extension '{extension}' is not available on this server")
    try:
        conn.execute(psql.SQL("CREATE EXTENSION IF NOT EXISTS {}").format(psql.Identifier(extension)))
    except Exception as exc:
        raise RuntimeError(
            f"PostgreSQL extension '{extension}' is required; ask a database superuser to run "
            f"CREATE EXTENSION {extension};"
        ) from exc


def _fts_spec_payload(spec: FtsSpec) -> Dict[str, Any]:
    return {
        "backend_version": _FTS_BACKEND_VERSION,
        "version": spec.version,
        "fields": [asdict(field) for field in spec.fields],
    }


def _fts_spec_json(spec: FtsSpec) -> str:
    return json.dumps(_fts_spec_payload(spec), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fts_spec_hash(spec: FtsSpec) -> str:
    return hashlib.sha256(_fts_spec_json(spec).encode("utf-8")).hexdigest()


def _ngram_terms(text: str, min_length: int, max_length: int) -> List[str]:
    """Build deterministic Unicode ngrams while keeping the SQL query bounded."""

    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    terms: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < min_length:
            candidates = [token] if token else []
        else:
            candidates = [
                token[start : start + size]
                for size in range(min_length, min(max_length, len(token)) + 1)
                for start in range(0, len(token) - size + 1)
            ]
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                terms.append(candidate)
                if len(terms) >= 64:
                    return terms
    return terms


def _schema_uses_vector(schema: Optional[pa.Schema]) -> bool:
    if schema is None:
        return False
    return any(
        isinstance(field.type, (pa.FixedSizeListType, pa.ListType)) and pa.types.is_floating(field.type.value_type)
        for field in schema
    )


# ---------------------------------------------------------------------------
# Table-level implementation
# ---------------------------------------------------------------------------


class PgVectorTable(VectorTable):
    """pgvector implementation of VectorTable."""

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
        logical_namespace: Optional[str] = None,
    ):
        self._table_name = table_name
        self._pool = pool
        self._embedding_fn = embedding_fn
        self._vector_column = vector_column
        self._source_column = source_column
        self._vector_dim = vector_dim
        self._column_names = column_names or []
        self._isolation = isolation
        self._logical_namespace = logical_namespace
        # Conflict-target columns whose UNIQUE index we have already ensured on
        # this handle, so upsert self-healing runs its DDL at most once.
        self._ensured_conflict_indexes: set = set()

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def embedding_fn(self) -> Any:
        return self._embedding_fn

    @property
    def vector_column(self) -> str:
        return self._vector_column

    @property
    def source_column(self) -> str:
        return self._source_column

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    @property
    def column_names(self) -> List[str]:
        return self._column_names

    # -- Write operations --

    def add(self, data: pd.DataFrame) -> None:
        df = self._inject_namespace_df(data)
        df = self._compute_embeddings_for_insert(df)
        self._insert_dataframe(df)

    def merge_insert(self, data: pd.DataFrame, on_column: str) -> None:
        df = self._inject_namespace_df(data)
        df = self._compute_embeddings_for_insert(df)
        self._upsert_dataframe(df, on_column)

    def delete(self, where: WhereExpr) -> None:
        if isinstance(where, str):
            compiled = where
        else:
            compiled = build_where(where)
        combined, ds_params = self._namespace_where_fragment(compiled)
        if combined:
            sql = f"DELETE FROM {self._table_name} WHERE {combined}"
            with self._pool.connection() as conn:
                conn.execute(sql, ds_params or None)
                conn.commit()

    def update(self, where: WhereExpr, values: Dict[str, Any]) -> None:
        if self._isolation == IsolationType.LOGICAL and LOGICAL_NAMESPACE_COLUMN in values:
            raise ValueError(f"{LOGICAL_NAMESPACE_COLUMN} is managed internally and cannot be updated")
        if isinstance(where, str):
            compiled = where
        else:
            compiled = build_where(where)
        set_parts = []
        params = []
        for col, val in values.items():
            _validate_identifier(col)
            set_parts.append(f"{col} = %s")
            params.append(val)
        set_clause = ", ".join(set_parts)
        combined, ds_params = self._namespace_where_fragment(compiled)
        where_clause = f" WHERE {combined}" if combined else ""
        sql = f"UPDATE {self._table_name} SET {set_clause}{where_clause}"
        with self._pool.connection() as conn:
            conn.execute(sql, params + ds_params)
            conn.commit()

    # -- Schema evolution --

    def ensure_columns(self, expressions: Dict[str, str]) -> None:
        """Add missing columns (TEXT) and backfill them from SQL expressions.

        Migration hook the storage layer calls after opening a pre-existing
        table, used to add datasource-scoping columns (``datasource_id`` /
        ``storage_key``) that newer code expects. ``expressions`` maps a column
        name to the SQL used to populate existing rows; these are trusted
        backend-internal expressions (e.g. ``'legacy:' || id``), not user input.
        """
        if not expressions:
            return
        existing = set(self._fetch_column_names())
        missing = {name: expr for name, expr in expressions.items() if name not in existing}
        if not missing:
            return
        with self._pool.connection() as conn:
            for name, expr in missing.items():
                _validate_identifier(name)
                conn.execute(f"ALTER TABLE {self._table_name} ADD COLUMN IF NOT EXISTS {name} TEXT")
                # Backfill rows that predate the column. expr is trusted internal SQL.
                conn.execute(f"UPDATE {self._table_name} SET {name} = {expr} WHERE {name} IS NULL")
            conn.commit()
        # Keep the cached column list in sync for subsequent reads on this handle.
        for name in missing:
            if name not in self._column_names:
                self._column_names.append(name)

    # -- Search operations --

    @staticmethod
    def _validate_select_fields(fields: List[str]) -> str:
        """Validate and join select field names."""
        for f in fields:
            _validate_identifier(f)
        return ", ".join(fields)

    def search_vector(
        self,
        query_text: str,
        vector_column: str,
        top_n: int,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        if isinstance(where, str):
            compiled = where
        else:
            compiled = build_where(where)
        combined, ds_params = self._namespace_where_fragment(compiled)
        query_embedding = self._compute_query_embedding(query_text)

        columns = self._validate_select_fields(select_fields) if select_fields else self._select_columns()
        _validate_identifier(vector_column)
        where_clause = f"WHERE {combined}" if combined else ""
        sql = (
            f"SELECT {columns} FROM {self._table_name} {where_clause} ORDER BY {vector_column} <=> %s::vector LIMIT %s"
        )
        with self._pool.connection() as conn:
            rows = conn.execute(sql, ds_params + [str(query_embedding), top_n]).fetchall()

        return self._rows_to_arrow(rows, select_fields)

    def search_hybrid(
        self,
        query_text: str,
        vector_source_column: str,
        top_n: int,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        # Fallback to vector search since full hybrid requires tsvector setup
        return self.search_vector(
            query_text,
            self._vector_column,
            top_n,
            where=where,
            select_fields=select_fields,
        )

    def search_fts(
        self,
        query_text: str,
        fts_spec: FtsSpec,
        top_n: int,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """Search PostgreSQL native FTS indexes without vector fallback."""

        spec = normalize_fts_spec(fts_spec)
        status = self.fts_index_status(spec)
        if status != FtsIndexStatus.READY:
            raise RuntimeError(f"FTS index for '{self._table_name}' is {status.value}")
        if top_n <= 0:
            return self._empty_fts_result(select_fields)

        if isinstance(where, str):
            compiled = where
        else:
            compiled = build_where(where)
        combined, namespace_params = self._namespace_where_fragment(compiled)

        score_parts: List[Any] = []
        score_params: List[Any] = []
        match_parts: List[Any] = []
        match_params: List[Any] = []
        for field in spec.fields:
            field_score, field_score_params, field_match, field_match_params = self._fts_field_query(field, query_text)
            if field_match is None:
                continue
            score_parts.append(field_score)
            score_params.extend(field_score_params)
            match_parts.append(field_match)
            match_params.extend(field_match_params)

        if not match_parts:
            return self._empty_fts_result(select_fields)

        result_fields = [field for field in (select_fields or self._default_columns) if field not in _FTS_SCORE_COLUMNS]
        score_sql = psql.SQL(" + ").join(score_parts)
        select_items = [psql.Identifier(field) for field in result_fields]
        select_items.append(psql.SQL("({}) AS _score").format(score_sql))
        match_sql = psql.SQL(" OR ").join(match_parts)
        where_parts = [psql.SQL("({})").format(match_sql)]
        if combined:
            where_parts.insert(0, psql.SQL("({})").format(psql.SQL(combined)))
        where_sql = psql.SQL(" AND ").join(where_parts)
        query = psql.SQL("SELECT {columns} FROM {table} WHERE {where_clause} ORDER BY _score DESC LIMIT %s").format(
            columns=psql.SQL(", ").join(select_items),
            table=self._qualified_table_identifier(),
            where_clause=where_sql,
        )
        params = score_params + namespace_params + match_params + [top_n]
        with self._pool.connection() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return self._empty_fts_result(select_fields)
        return self._rows_to_arrow(rows)

    def search_all(
        self,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pa.Table:
        if isinstance(where, str):
            compiled = where
        else:
            compiled = build_where(where)
        combined, ds_params = self._namespace_where_fragment(compiled)
        columns = self._validate_select_fields(select_fields) if select_fields else self._select_columns()
        where_clause = f"WHERE {combined}" if combined else ""

        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        sql = f"SELECT {columns} FROM {self._table_name} {where_clause} {limit_clause}"

        with self._pool.connection() as conn:
            rows = conn.execute(sql, ds_params or None).fetchall()

        return self._rows_to_arrow(rows, select_fields)

    def count_rows(self, where: WhereExpr = None) -> int:
        if isinstance(where, str):
            compiled = where
        else:
            compiled = build_where(where)
        combined, ds_params = self._namespace_where_fragment(compiled)
        where_clause = f"WHERE {combined}" if combined else ""
        sql = f"SELECT COUNT(*) AS cnt FROM {self._table_name} {where_clause}"
        with self._pool.connection() as conn:
            row = conn.execute(sql, ds_params or None).fetchone()
            if isinstance(row, dict):
                return row["cnt"]
            return row[0] if row else 0

    # -- Index operations --

    def create_vector_index(self, column: str, metric: str = "cosine", **kwargs) -> None:
        _validate_identifier(column)
        table_token = self._table_name.rsplit(".", 1)[-1]
        index_name = f"idx_{table_token}_{column}_hnsw"
        ops_map = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "ip": "vector_ip_ops",
        }
        ops = ops_map.get(metric, "vector_cosine_ops")
        sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {self._table_name} USING hnsw ({column} {ops})"
        with self._pool.connection() as conn:
            conn.execute(sql)
            conn.commit()

    def create_fts_index(self, spec: FtsSpecInput) -> None:
        """Create backend-native indexes for the complete FTS specification."""

        normalized = normalize_fts_spec(spec)
        existing_columns = set(self._fetch_column_names())
        for field in normalized.fields:
            _validate_identifier(field.name)
            if field.name not in existing_columns:
                raise ValueError(f"FTS field '{field.name}' does not exist on table '{self._table_name}'")

        with self._pool.connection() as conn:
            self._ensure_fts_metadata_table(conn)
            if self._fts_index_status(conn, normalized) == FtsIndexStatus.READY:
                return

            previous = self._read_fts_spec(conn)
            fields_to_drop = {field.name for field in normalized.fields}
            if previous is not None:
                fields_to_drop.update(field.name for field in previous.fields)
            for field_name in fields_to_drop:
                self._drop_fts_index(conn, field_name)

            if any(field.tokenizer == "raw" for field in normalized.fields):
                self._ensure_extension(conn, "pg_trgm")
            if any(field.tokenizer == "ngram" for field in normalized.fields):
                self._ensure_fts_ngram_function(conn)
            for field in normalized.fields:
                self._create_fts_field_index(conn, field)
            self._write_fts_spec(conn, normalized)
            conn.commit()

    def supports_fts(self) -> bool:
        return True

    def fts_index_status(self, spec: FtsSpec) -> FtsIndexStatus:
        normalized = normalize_fts_spec(spec)
        with self._pool.connection() as conn:
            return self._fts_index_status(conn, normalized)

    def remove_legacy_fts_index(self) -> bool:
        """Remove the fixed ``tsv`` index created by adapter versions before the FTS contract."""

        with self._pool.connection() as conn:
            if not self._has_legacy_fts_index(conn):
                return False
            schema, table = self._table_parts()
            legacy_index = f"idx_{table}_fts"
            conn.execute(psql.SQL("DROP INDEX IF EXISTS {}").format(self._qualified_index_identifier(legacy_index)))
            generated = conn.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_schema = %s AND table_name = %s "
                "AND column_name = 'tsv' AND is_generated = 'ALWAYS'",
                (schema, table),
            ).fetchone()
            if generated:
                conn.execute(
                    psql.SQL("ALTER TABLE {} DROP COLUMN {}").format(
                        self._qualified_table_identifier(),
                        psql.Identifier("tsv"),
                    )
                )
            conn.commit()
            return True

    def create_scalar_index(self, column: str) -> None:
        _validate_identifier(column)
        table_token = self._table_name.rsplit(".", 1)[-1]
        index_name = f"idx_{table_token}_{column}_btree"
        sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {self._table_name} ({column})"
        with self._pool.connection() as conn:
            conn.execute(sql)
            conn.commit()

    def optimize(self) -> None:
        """PostgreSQL maintains GIN indexes transactionally during DML."""

    # -- Private helpers --

    def _table_parts(self) -> tuple[str, str]:
        if "." in self._table_name:
            return tuple(self._table_name.split(".", 1))
        return "public", self._table_name

    def _qualified_table_identifier(self) -> Any:
        schema, table = self._table_parts()
        return psql.Identifier(schema, table) if schema != "public" else psql.Identifier(table)

    def _qualified_metadata_identifier(self) -> Any:
        schema, _ = self._table_parts()
        return (
            psql.Identifier(schema, _FTS_METADATA_TABLE) if schema != "public" else psql.Identifier(_FTS_METADATA_TABLE)
        )

    def _qualified_index_identifier(self, index_name: str) -> Any:
        schema, _ = self._table_parts()
        return psql.Identifier(schema, index_name) if schema != "public" else psql.Identifier(index_name)

    def _qualified_ngram_function_identifier(self) -> Any:
        schema, _ = self._table_parts()
        return (
            psql.Identifier(schema, _FTS_NGRAM_FUNCTION) if schema != "public" else psql.Identifier(_FTS_NGRAM_FUNCTION)
        )

    def _fts_index_name(self, field_name: str) -> str:
        _, table = self._table_parts()
        digest = hashlib.sha256(f"{table}:{field_name}".encode("utf-8")).hexdigest()[:10]
        readable = re.sub(r"[^A-Za-z0-9_]", "_", f"idx_{table}_{field_name}_datus_fts")
        return f"{readable[:52]}_{digest}"

    def _fts_vector_column_name(self, field_name: str) -> str:
        _, table = self._table_parts()
        digest = hashlib.sha256(f"{table}:{field_name}".encode("utf-8")).hexdigest()[:16]
        return f"{_FTS_GENERATED_COLUMN_PREFIX}{digest}"

    def _ensure_extension(self, conn: Any, extension: str) -> None:
        _ensure_postgres_extension(conn, extension)

    def _ensure_fts_metadata_table(self, conn: Any) -> None:
        conn.execute(
            psql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "table_name TEXT PRIMARY KEY, spec_hash TEXT NOT NULL, spec_json JSONB NOT NULL, "
                "version INTEGER NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            ).format(self._qualified_metadata_identifier())
        )

    def _ensure_fts_ngram_function(self, conn: Any) -> None:
        conn.execute(
            psql.SQL(
                "CREATE OR REPLACE FUNCTION {}(input_text TEXT, min_len INTEGER, max_len INTEGER) "
                "RETURNS TEXT LANGUAGE SQL IMMUTABLE PARALLEL SAFE AS $datus_fts$ "
                "WITH tokens AS ("
                "SELECT token FROM regexp_split_to_table(LOWER(COALESCE(input_text, '')), "
                "'[[:space:][:punct:]]+') AS token WHERE token <> ''"
                "), grams AS ("
                "SELECT token AS gram FROM tokens WHERE char_length(token) < min_len "
                "UNION ALL "
                "SELECT substr(token, gram_position, gram_size) FROM tokens "
                "CROSS JOIN LATERAL generate_series(min_len, LEAST(max_len, char_length(token))) AS gram_size "
                "CROSS JOIN LATERAL generate_series(1, char_length(token) - gram_size + 1) AS gram_position "
                "WHERE char_length(token) >= min_len"
                ") SELECT COALESCE(string_agg(gram, ' ' ORDER BY gram), '') FROM grams $datus_fts$"
            ).format(self._qualified_ngram_function_identifier())
        )

    def _fts_metadata_table_exists(self, conn: Any) -> bool:
        schema, _ = self._table_parts()
        qualified = f"{schema}.{_FTS_METADATA_TABLE}"
        row = conn.execute("SELECT to_regclass(%s) IS NOT NULL AS exists", (qualified,)).fetchone()
        return bool(row["exists"] if isinstance(row, dict) else row[0])

    def _read_fts_spec(self, conn: Any) -> Optional[FtsSpec]:
        if not self._fts_metadata_table_exists(conn):
            return None
        _, table = self._table_parts()
        row = conn.execute(
            psql.SQL("SELECT spec_json FROM {} WHERE table_name = %s").format(self._qualified_metadata_identifier()),
            (table,),
        ).fetchone()
        if not row:
            return None
        payload = row["spec_json"] if isinstance(row, dict) else row[0]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FtsSpec(
            tuple(FtsField(**field) for field in payload["fields"]),
            version=int(payload["version"]),
        )

    def _write_fts_spec(self, conn: Any, spec: FtsSpec) -> None:
        _, table = self._table_parts()
        conn.execute(
            psql.SQL(
                "INSERT INTO {} (table_name, spec_hash, spec_json, version) VALUES (%s, %s, %s::jsonb, %s) "
                "ON CONFLICT (table_name) DO UPDATE SET spec_hash = EXCLUDED.spec_hash, "
                "spec_json = EXCLUDED.spec_json, version = EXCLUDED.version, updated_at = CURRENT_TIMESTAMP"
            ).format(self._qualified_metadata_identifier()),
            (table, _fts_spec_hash(spec), _fts_spec_json(spec), spec.version),
        )

    def _create_fts_field_index(self, conn: Any, field: FtsField) -> None:
        index_name = self._fts_index_name(field.name)
        if field.tokenizer == "raw":
            statement = psql.SQL("CREATE INDEX {} ON {} USING gin ({} gin_trgm_ops)").format(
                psql.Identifier(index_name),
                self._qualified_table_identifier(),
                psql.Identifier(field.name),
            )
        elif field.tokenizer == "ngram":
            generated_column = self._fts_vector_column_name(field.name)
            conn.execute(
                psql.SQL(
                    "ALTER TABLE {} ADD COLUMN {} TSVECTOR GENERATED ALWAYS AS ("
                    "to_tsvector('simple', {}(COALESCE({}, ''), {}, {}))) STORED"
                ).format(
                    self._qualified_table_identifier(),
                    psql.Identifier(generated_column),
                    self._qualified_ngram_function_identifier(),
                    psql.Identifier(field.name),
                    psql.Literal(field.ngram_min_length),
                    psql.Literal(field.ngram_max_length),
                )
            )
            statement = psql.SQL("CREATE INDEX {} ON {} USING gin ({})").format(
                psql.Identifier(index_name),
                self._qualified_table_identifier(),
                psql.Identifier(generated_column),
            )
        else:
            statement = psql.SQL("CREATE INDEX {} ON {} USING gin (to_tsvector('simple', COALESCE({}, '')))").format(
                psql.Identifier(index_name),
                self._qualified_table_identifier(),
                psql.Identifier(field.name),
            )
        conn.execute(statement)

    def _drop_fts_index(self, conn: Any, field_name: str) -> None:
        conn.execute(
            psql.SQL("DROP INDEX IF EXISTS {}").format(
                self._qualified_index_identifier(self._fts_index_name(field_name))
            )
        )
        conn.execute(
            psql.SQL("ALTER TABLE {} DROP COLUMN IF EXISTS {}").format(
                self._qualified_table_identifier(),
                psql.Identifier(self._fts_vector_column_name(field_name)),
            )
        )

    def _fts_index_exists(self, conn: Any, field_name: str) -> bool:
        schema, _ = self._table_parts()
        row = conn.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s",
            (schema, self._fts_index_name(field_name)),
        ).fetchone()
        return bool(row)

    def _fts_vector_column_exists(self, conn: Any, field_name: str) -> bool:
        schema, table = self._table_parts()
        row = conn.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_schema = %s AND table_name = %s AND column_name = %s",
            (schema, table, self._fts_vector_column_name(field_name)),
        ).fetchone()
        return bool(row)

    def _has_legacy_fts_index(self, conn: Any) -> bool:
        schema, table = self._table_parts()
        legacy_index = f"idx_{table}_fts"
        index_row = conn.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s",
            (schema, legacy_index),
        ).fetchone()
        column_row = conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s AND column_name = 'tsv'",
            (schema, table),
        ).fetchone()
        return bool(index_row or column_row)

    def _fts_index_status(self, conn: Any, spec: FtsSpec) -> FtsIndexStatus:
        if not self._fts_metadata_table_exists(conn):
            return FtsIndexStatus.LEGACY if self._has_legacy_fts_index(conn) else FtsIndexStatus.MISSING
        _, table = self._table_parts()
        row = conn.execute(
            psql.SQL("SELECT spec_hash FROM {} WHERE table_name = %s").format(self._qualified_metadata_identifier()),
            (table,),
        ).fetchone()
        if not row:
            return FtsIndexStatus.LEGACY if self._has_legacy_fts_index(conn) else FtsIndexStatus.MISSING
        stored_hash = row["spec_hash"] if isinstance(row, dict) else row[0]
        if stored_hash != _fts_spec_hash(spec):
            return FtsIndexStatus.VERSION_MISMATCH
        if not all(self._fts_index_exists(conn, field.name) for field in spec.fields):
            return FtsIndexStatus.MISSING
        if any(
            field.tokenizer == "ngram" and not self._fts_vector_column_exists(conn, field.name) for field in spec.fields
        ):
            return FtsIndexStatus.MISSING
        return FtsIndexStatus.READY

    def _fts_field_query(self, field: FtsField, query_text: str) -> tuple[Any, List[Any], Optional[Any], List[Any]]:
        column = psql.Identifier(field.name)
        if field.tokenizer in {"simple", "whitespace"}:
            vector = psql.SQL("to_tsvector('simple', COALESCE({}, ''))").format(column)
            query = psql.SQL("plainto_tsquery('simple', %s)")
            score = psql.SQL("ts_rank_cd({}, {}) * %s").format(vector, query)
            match = psql.SQL("{} @@ {}").format(vector, query)
            return score, [query_text, field.boost], match, [query_text]

        if field.tokenizer == "raw":
            pattern = f"%{query_text}%"
            score = psql.SQL("CASE WHEN COALESCE({}, '') ILIKE %s THEN %s ELSE 0 END").format(column)
            match = psql.SQL("COALESCE({}, '') ILIKE %s").format(column)
            return score, [pattern, field.boost], match, [pattern]

        terms = _ngram_terms(query_text, field.ngram_min_length, field.ngram_max_length)
        if not terms:
            return psql.SQL("0"), [], None, []
        vector = psql.Identifier(self._fts_vector_column_name(field.name))
        query = psql.SQL("to_tsquery('simple', %s)")
        query_value = " | ".join(terms)
        score = psql.SQL("ts_rank({}, {}) * %s").format(vector, query)
        match = psql.SQL("{} @@ {}").format(vector, query)
        return score, [query_value, field.boost], match, [query_value]

    def _empty_fts_result(self, select_fields: Optional[List[str]]) -> pa.Table:
        fields = [field for field in (select_fields or self._default_columns) if field not in _FTS_SCORE_COLUMNS]
        arrays: Dict[str, pa.Array] = {}
        for field in fields:
            if field == self._vector_column:
                arrays[field] = pa.array([], type=pa.list_(pa.float32(), list_size=self._vector_dim))
            else:
                arrays[field] = pa.array([], type=pa.string())
        arrays["_score"] = pa.array([], type=pa.float64())
        return pa.table(arrays)

    def _namespace_where_fragment(self, existing_compiled: Optional[str] = None) -> tuple:
        """Build WHERE clause fragment with backend namespace for logical isolation.

        Returns:
            (clause_str, params_list) where clause_str may be empty and params
            is a list of bind values for %s placeholders.
        """
        if self._isolation != IsolationType.LOGICAL or self._logical_namespace is None:
            return (existing_compiled or "", [])
        namespace_cond = f"{LOGICAL_NAMESPACE_COLUMN} = %s"
        if existing_compiled:
            return (f"{namespace_cond} AND ({existing_compiled})", [self._logical_namespace])
        return (namespace_cond, [self._logical_namespace])

    def _inject_namespace_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Force backend namespace column on DataFrame for logical isolation."""
        if self._isolation != IsolationType.LOGICAL or self._logical_namespace is None:
            return df
        df = df.copy()
        df[LOGICAL_NAMESPACE_COLUMN] = self._logical_namespace
        return df

    def _compute_embeddings_for_insert(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute source embeddings and fill the vector column in the DataFrame."""
        if self._embedding_fn is None:
            return df

        if self._vector_column not in df.columns:
            df = df.copy()
            df[self._vector_column] = self._embedding_fn.generate_embeddings(df[self._source_column].tolist())
            return df

        missing = df[self._vector_column].isna()
        if missing.any():
            df = df.copy()
            df.loc[missing, self._vector_column] = self._embedding_fn.generate_embeddings(
                df.loc[missing, self._source_column].tolist()
            )

        return df

    def _compute_query_embedding(self, query_text: str) -> List[float]:
        """Compute embedding for a query text."""
        if self._embedding_fn is None:
            raise RuntimeError(
                f"No embedding function available for table '{self._table_name}'. "
                "Ensure the table was created with an embedding_function."
            )
        embeddings = self._embedding_fn.generate_embeddings([query_text])
        return embeddings[0]

    def _insert_dataframe(self, df: pd.DataFrame) -> None:
        """Insert all rows from a DataFrame into the table."""
        if df.empty:
            return

        columns = list(df.columns)
        for c in columns:
            _validate_identifier(c)
        col_names = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        sql = f"INSERT INTO {self._table_name} ({col_names}) VALUES ({placeholders})"

        rows = []
        for _, row in df.iterrows():
            values = []
            for col in columns:
                val = row[col]
                if col == self._vector_column and val is not None:
                    val = str(list(val)) if not isinstance(val, str) else val
                values.append(val)
            rows.append(tuple(values))

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()

    def _upsert_dataframe(self, df: pd.DataFrame, on_column: str) -> None:
        """Upsert all rows from a DataFrame into the table."""
        if df.empty:
            return

        columns = list(df.columns)
        for c in columns:
            _validate_identifier(c)
        _validate_identifier(on_column)

        # In logical mode, scope conflict target to backend namespace
        if self._isolation == IsolationType.LOGICAL and self._logical_namespace is not None:
            conflict_cols = [on_column, LOGICAL_NAMESPACE_COLUMN]
        else:
            conflict_cols = [on_column]
        conflict_target = ", ".join(conflict_cols)
        # A table migrated to a new conflict key (e.g. storage_key) has no
        # matching UNIQUE index, which ON CONFLICT requires; ensure it exists.
        self._ensure_conflict_index(conflict_cols)

        col_names = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        skip_cols = {on_column}
        if self._isolation == IsolationType.LOGICAL:
            skip_cols.add(LOGICAL_NAMESPACE_COLUMN)
        update_cols = [c for c in columns if c not in skip_cols]
        update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

        if update_set:
            sql = (
                f"INSERT INTO {self._table_name} ({col_names}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_target}) DO UPDATE SET {update_set}"
            )
        else:
            sql = (
                f"INSERT INTO {self._table_name} ({col_names}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_target}) DO NOTHING"
            )

        rows = []
        for _, row in df.iterrows():
            values = []
            for col in columns:
                val = row[col]
                if col == self._vector_column and val is not None:
                    val = str(list(val)) if not isinstance(val, str) else val
                values.append(val)
            rows.append(tuple(values))

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()

    def _fetch_column_names(self) -> List[str]:
        """Return the live column names of this table from the catalog."""
        if "." in self._table_name:
            schema, bare = self._table_name.split(".", 1)
        else:
            schema, bare = "public", self._table_name
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s",
                (schema, bare),
            ).fetchall()
        return [r["column_name"] if isinstance(r, dict) else r[0] for r in rows]

    def _ensure_conflict_index(self, conflict_cols: List[str]) -> None:
        """Ensure a UNIQUE index exists for an upsert conflict target.

        ``create_table`` builds these for fresh tables, but a pre-existing table
        migrated to a new conflict key lacks one, so ON CONFLICT would fail.
        Idempotent (``IF NOT EXISTS``) and cached per handle. The index name
        mirrors the one ``create_table`` uses for the logical composite key.
        """
        key = tuple(conflict_cols)
        if key in self._ensured_conflict_indexes:
            return
        for col in conflict_cols:
            _validate_identifier(col)
        table_token = self._table_name.rsplit(".", 1)[-1]
        index_name = f"idx_{table_token}_{'_'.join(conflict_cols)}_uq"
        cols_sql = ", ".join(conflict_cols)
        with self._pool.connection() as conn:
            conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {self._table_name} ({cols_sql})")
            conn.commit()
        self._ensured_conflict_indexes.add(key)

    @property
    def _default_columns(self) -> List[str]:
        """Return column names filtered for the current isolation mode."""
        cols = [column for column in self._column_names if not column.startswith(_FTS_GENERATED_COLUMN_PREFIX)]
        if self._isolation == IsolationType.LOGICAL:
            cols = [c for c in cols if c != LOGICAL_NAMESPACE_COLUMN]
        return cols

    def _select_columns(self) -> str:
        """Build the default SELECT column list, excluding the backend namespace in logical mode."""
        cols = self._default_columns
        return ", ".join(cols) if cols else "*"

    def _rows_to_arrow(
        self,
        rows: List[Any],
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """Convert fetched rows (list of dicts) to a PyArrow Table."""
        default_cols = self._default_columns
        if not rows:
            if select_fields:
                arrays = {
                    f: pa.array(
                        [],
                        type=pa.list_(pa.float32(), list_size=self._vector_dim)
                        if f == self._vector_column
                        else pa.string(),
                    )
                    for f in select_fields
                }
            elif default_cols:
                arrays = {
                    c: pa.array(
                        [],
                        type=pa.list_(pa.float32(), list_size=self._vector_dim)
                        if c == self._vector_column
                        else pa.string(),
                    )
                    for c in default_cols
                }
            else:
                return pa.table({})
            return pa.table(arrays)

        if isinstance(rows[0], dict):
            col_names = select_fields or list(rows[0].keys())
        else:
            col_names = select_fields or default_cols

        arrays = {}
        for idx, col in enumerate(col_names):
            values = [r[col] if isinstance(r, dict) else r[idx] for r in rows]
            if col == self._vector_column:
                parsed = []
                for v in values:
                    if isinstance(v, str):
                        parsed.append([float(x) for x in v.strip("[]").split(",")])
                    elif isinstance(v, list):
                        parsed.append(v)
                    else:
                        parsed.append(list(v) if v is not None else [0.0] * self._vector_dim)
                arrays[col] = pa.array(parsed, type=pa.list_(pa.float32(), list_size=self._vector_dim))
            else:
                arrays[col] = pa.array(values)

        return pa.table(arrays)


# ---------------------------------------------------------------------------
# Database-level implementation
# ---------------------------------------------------------------------------


class PgVectorDb(VectorDatabase):
    """pgvector implementation of VectorDatabase.

    Uses PostgreSQL schemas to implement namespace-based data isolation.
    """

    def __init__(
        self,
        pool: ConnectionPool,
        config: Dict[str, Any],
        namespace: str = "",
        isolation: IsolationType = IsolationType.PHYSICAL,
    ):
        self._pool = pool
        self._config = config
        self._namespace = namespace
        self._isolation = isolation
        self._table_cache: Dict[tuple, PgVectorTable] = {}

        if isolation == IsolationType.LOGICAL:
            self._schema = "public"
            self._logical_namespace = namespace
        else:
            self._schema = _physical_schema_name(namespace)
            self._logical_namespace = None

        # Ensure schema exists for non-public namespaces
        if self._schema != "public":
            with self._pool.connection() as conn:
                conn.execute(psql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(psql.Identifier(self._schema)))
                conn.commit()

    @property
    def pool(self) -> ConnectionPool:
        return self._pool

    @property
    def namespace(self) -> str:
        return self._namespace

    def supports_fts(self) -> bool:
        return True

    def _qualified(self, table_name: str) -> str:
        """Return schema-qualified table name."""
        _validate_identifier(table_name)
        if self._schema == "public":
            return table_name
        return f"{self._schema}.{table_name}"

    def table_exists(self, table_name: str) -> bool:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s)",
                (self._schema, table_name),
            ).fetchone()
            if isinstance(rows, dict):
                return next(iter(rows.values()))
            return rows[0] if rows else False

    def table_names(self, limit: int = 100) -> List[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name <> %s ORDER BY table_name LIMIT %s",
                (self._schema, _FTS_METADATA_TABLE, limit),
            ).fetchall()
            return [r["table_name"] if isinstance(r, dict) else r[0] for r in rows]

    def _find_legacy_unique_constraints(self, conn: Any, table_name: str, columns: List[str]) -> List[str]:
        rows = conn.execute(
            """
            SELECT c.conname, array_agg(a.attname ORDER BY keys.ordinality) AS columns
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN unnest(c.conkey) WITH ORDINALITY AS keys(attnum, ordinality) ON TRUE
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = keys.attnum
            WHERE n.nspname = %s AND t.relname = %s AND c.contype = 'u'
            GROUP BY c.conname
            """,
            (self._schema, table_name),
        ).fetchall()
        return [row["conname"] for row in rows if list(row["columns"]) == columns]

    def _find_legacy_unique_indexes(self, conn: Any, table_name: str, columns: List[str]) -> List[str]:
        rows = conn.execute(
            """
            SELECT i.relname AS indexname, array_agg(a.attname ORDER BY keys.ordinality) AS columns
            FROM pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN unnest(ix.indkey) WITH ORDINALITY AS keys(attnum, ordinality) ON TRUE
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = keys.attnum
            LEFT JOIN pg_constraint c ON c.conindid = ix.indexrelid
            WHERE n.nspname = %s
              AND t.relname = %s
              AND ix.indisunique
              AND NOT ix.indisprimary
              AND ix.indpred IS NULL
              AND ix.indexprs IS NULL
              AND c.oid IS NULL
            GROUP BY i.relname
            """,
            (self._schema, table_name),
        ).fetchall()
        return [row["indexname"] for row in rows if list(row["columns"]) == columns]

    def _qualified_sql_identifier(self, table_name: str) -> Any:
        return psql.Identifier(self._schema, table_name) if self._schema != "public" else psql.Identifier(table_name)

    def _index_sql_identifier(self, index_name: str) -> Any:
        return psql.Identifier(self._schema, index_name) if self._schema != "public" else psql.Identifier(index_name)

    def _scoped_unique_index_name(self, table_name: str, columns: List[str]) -> str:
        return f"idx_{table_name}_{'_'.join(columns)}_uq"

    def _migrate_unique_scope_specs(
        self,
        conn: Any,
        table_name: str,
        specs: List[tuple[str, str, List[str]]],
    ) -> None:
        """Replace unscoped unique constraints/indexes with namespace-scoped indexes."""
        if self._isolation != IsolationType.LOGICAL:
            return
        qualified = self._qualified_sql_identifier(table_name)
        for kind, name, old_columns in specs:
            if LOGICAL_NAMESPACE_COLUMN in old_columns:
                continue
            scoped_columns = old_columns + [LOGICAL_NAMESPACE_COLUMN]
            if kind == "constraint":
                conn.execute(
                    psql.SQL("ALTER TABLE {} DROP CONSTRAINT IF EXISTS {}").format(
                        qualified,
                        psql.Identifier(name),
                    )
                )
            elif kind == "index":
                conn.execute(psql.SQL("DROP INDEX IF EXISTS {}").format(self._index_sql_identifier(name)))
            else:
                raise ValueError(f"Unsupported unique scope spec kind: {kind}")
            conn.execute(
                psql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS {} ON {} ({})").format(
                    psql.Identifier(self._scoped_unique_index_name(table_name, scoped_columns)),
                    qualified,
                    psql.SQL(", ").join(psql.Identifier(col) for col in scoped_columns),
                )
            )

    def _migrate_legacy_unique_scopes(self, conn: Any, table_name: str, unique_columns: Optional[List[str]]) -> None:
        """Replace known unscoped unique columns with namespace-scoped unique indexes."""
        if self._isolation != IsolationType.LOGICAL or not unique_columns:
            return
        specs: List[tuple[str, str, List[str]]] = []
        for ucol in unique_columns:
            _validate_identifier(ucol)
            old_columns = [ucol]
            specs.extend(
                ("constraint", name, old_columns)
                for name in self._find_legacy_unique_constraints(conn, table_name, old_columns)
            )
            specs.extend(
                ("index", name, old_columns) for name in self._find_legacy_unique_indexes(conn, table_name, old_columns)
            )
        self._migrate_unique_scope_specs(conn, table_name, specs)

    def _find_unscoped_unique_specs(self, conn: Any, table_name: str) -> List[tuple[str, str, List[str]]]:
        """Return all non-primary unique constraints/indexes missing the namespace column."""
        constraint_rows = conn.execute(
            """
            SELECT c.conname, array_agg(a.attname ORDER BY keys.ordinality) AS columns
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN unnest(c.conkey) WITH ORDINALITY AS keys(attnum, ordinality) ON TRUE
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = keys.attnum
            WHERE n.nspname = %s AND t.relname = %s AND c.contype = 'u'
            GROUP BY c.conname
            """,
            (self._schema, table_name),
        ).fetchall()
        index_rows = conn.execute(
            """
            SELECT i.relname AS indexname, array_agg(a.attname ORDER BY keys.ordinality) AS columns
            FROM pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN unnest(ix.indkey) WITH ORDINALITY AS keys(attnum, ordinality) ON TRUE
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = keys.attnum
            LEFT JOIN pg_constraint c ON c.conindid = ix.indexrelid
            WHERE n.nspname = %s
              AND t.relname = %s
              AND ix.indisunique
              AND NOT ix.indisprimary
              AND ix.indpred IS NULL
              AND ix.indexprs IS NULL
              AND c.oid IS NULL
            GROUP BY i.relname
            """,
            (self._schema, table_name),
        ).fetchall()
        specs: List[tuple[str, str, List[str]]] = []
        for row in constraint_rows:
            columns = list(row["columns"])
            if LOGICAL_NAMESPACE_COLUMN not in columns:
                specs.append(("constraint", row["conname"], columns))
        for row in index_rows:
            columns = list(row["columns"])
            if LOGICAL_NAMESPACE_COLUMN not in columns:
                specs.append(("index", row["indexname"], columns))
        return specs

    def _ensure_logical_namespace_scope(
        self,
        conn: Any,
        table_name: str,
        unique_columns: Optional[List[str]] = None,
        migrate_all_unique: bool = False,
    ) -> None:
        """Ensure an existing logical table has the internal namespace scope."""
        if self._isolation != IsolationType.LOGICAL:
            return

        qualified = self._qualified_sql_identifier(table_name)
        conn.execute(
            psql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} TEXT NOT NULL DEFAULT ''").format(
                qualified,
                psql.Identifier(LOGICAL_NAMESPACE_COLUMN),
            )
        )
        if self._logical_namespace is not None:
            conn.execute(
                psql.SQL("UPDATE {} SET {} = %s WHERE {} = ''").format(
                    qualified,
                    psql.Identifier(LOGICAL_NAMESPACE_COLUMN),
                    psql.Identifier(LOGICAL_NAMESPACE_COLUMN),
                ),
                (self._logical_namespace,),
            )

        if migrate_all_unique:
            self._migrate_unique_scope_specs(conn, table_name, self._find_unscoped_unique_specs(conn, table_name))
        else:
            self._migrate_legacy_unique_scopes(conn, table_name, unique_columns)

        idx_name = f"idx_{table_name}_{LOGICAL_NAMESPACE_COLUMN}"
        conn.execute(
            psql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} ({})").format(
                psql.Identifier(idx_name),
                qualified,
                psql.Identifier(LOGICAL_NAMESPACE_COLUMN),
            )
        )

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

        # Apply defaults when not specified
        vector_column = vector_column or "vector"
        source_column = source_column or "description"
        vector_dim = embedding_function.ndims() if embedding_function else 384

        # Build column list from schema
        column_names = []
        if schema is not None:
            if isinstance(schema, pa.Schema):
                # Inject internal namespace column for logical isolation.
                if self._isolation == IsolationType.LOGICAL:
                    if LOGICAL_NAMESPACE_COLUMN not in schema.names:
                        schema = schema.append(pa.field(LOGICAL_NAMESPACE_COLUMN, pa.string()))
                ddl_unique_columns = None if self._isolation == IsolationType.LOGICAL else unique_columns
                ddl = schema_to_create_table_sql(qualified, schema, unique_columns=ddl_unique_columns)
                column_names = [f.name for f in schema]
            else:
                raise TypeError(f"Unsupported schema type: {type(schema)}")

            with self._pool.connection() as conn:
                if _schema_uses_vector(schema):
                    _ensure_postgres_extension(conn, "vector")
                conn.execute(ddl)
                # Create indexes for logical isolation
                if self._isolation == IsolationType.LOGICAL:
                    table_token = table_name
                    self._ensure_logical_namespace_scope(conn, table_name, unique_columns)
                    # Create composite unique indexes for upsert conflict targets
                    if unique_columns:
                        for ucol in unique_columns:
                            _validate_identifier(ucol)
                            comp_idx = f"idx_{table_token}_{ucol}_{LOGICAL_NAMESPACE_COLUMN}_uq"
                            conn.execute(
                                f"CREATE UNIQUE INDEX IF NOT EXISTS {comp_idx} "
                                f"ON {qualified} ({ucol}, {LOGICAL_NAMESPACE_COLUMN})"
                            )
                conn.commit()
        elif not exist_ok:
            raise ValueError(f"Schema is required to create table '{table_name}'")
        else:
            if not self.table_exists(table_name):
                raise ValueError(f"Table '{table_name}' does not exist and no schema was provided to create it.")

        table = PgVectorTable(
            table_name=qualified,
            pool=self._pool,
            embedding_fn=embedding_function,
            vector_column=vector_column,
            source_column=source_column,
            vector_dim=vector_dim,
            column_names=column_names,
            isolation=self._isolation,
            logical_namespace=self._logical_namespace,
        )
        cache_key = (table_name, id(embedding_function), vector_dim, vector_column, source_column)
        self._table_cache[cache_key] = table
        return table

    def open_table(
        self,
        table_name: str,
        embedding_function: Optional[EmbeddingFunction] = None,
        vector_column: str = "",
        source_column: str = "",
    ) -> PgVectorTable:
        vector_column = vector_column or "vector"
        source_column = source_column or "description"
        vector_dim = embedding_function.ndims() if embedding_function else 384

        # Build a cache key that includes runtime options so changed options
        # don't return a stale handle.
        cache_key = (table_name, id(embedding_function), vector_dim, vector_column, source_column)
        if cache_key in self._table_cache:
            return self._table_cache[cache_key]

        qualified = self._qualified(table_name)

        column_query = (
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s "
            "ORDER BY ordinal_position"
        )
        with self._pool.connection() as conn:
            rows = conn.execute(column_query, (self._schema, table_name)).fetchall()
            column_names = [r["column_name"] if isinstance(r, dict) else r[0] for r in rows]
            if not column_names:
                raise ValueError(
                    f"Table '{table_name}' not found in schema '{self._schema}'. Use create_table() first."
                )
            if self._isolation == IsolationType.LOGICAL:
                self._ensure_logical_namespace_scope(conn, table_name, migrate_all_unique=True)
                conn.commit()
                rows = conn.execute(column_query, (self._schema, table_name)).fetchall()
                column_names = [r["column_name"] if isinstance(r, dict) else r[0] for r in rows]

        table = PgVectorTable(
            table_name=qualified,
            pool=self._pool,
            embedding_fn=embedding_function,
            vector_column=vector_column,
            source_column=source_column,
            vector_dim=vector_dim,
            column_names=column_names,
            isolation=self._isolation,
            logical_namespace=self._logical_namespace,
        )
        self._table_cache[cache_key] = table
        return table

    def _invalidate_cache(self, table_name: str) -> None:
        """Remove all cache entries for the given table name."""
        keys_to_remove = [k for k in self._table_cache if k[0] == table_name]
        for k in keys_to_remove:
            del self._table_cache[k]

    def refresh_table(
        self,
        table_name: str,
        embedding_function: Optional[EmbeddingFunction] = None,
        vector_column: str = "",
        source_column: str = "",
    ) -> PgVectorTable:
        """Invalidate cache and re-open the table."""
        self._invalidate_cache(table_name)
        return self.open_table(table_name, embedding_function, vector_column, source_column)

    def drop_table(self, table_name: str, ignore_missing: bool = False) -> None:
        if self._isolation == IsolationType.LOGICAL:
            raise RuntimeError(
                f"drop_table('{table_name}') is not allowed in logical isolation mode "
                "because the table is shared across all tenants. "
                "Use delete() with scoped filters to remove tenant data."
            )
        qualified = self._qualified(table_name)
        if_exists = "IF EXISTS " if ignore_missing else ""
        sql = f"DROP TABLE {if_exists}{qualified}"
        with self._pool.connection() as conn:
            conn.execute(sql)
            metadata_name = f"{self._schema}.{_FTS_METADATA_TABLE}"
            metadata_exists = conn.execute("SELECT to_regclass(%s) IS NOT NULL AS exists", (metadata_name,)).fetchone()
            if metadata_exists and (
                metadata_exists["exists"] if isinstance(metadata_exists, dict) else metadata_exists[0]
            ):
                metadata_table = (
                    psql.Identifier(self._schema, _FTS_METADATA_TABLE)
                    if self._schema != "public"
                    else psql.Identifier(_FTS_METADATA_TABLE)
                )
                conn.execute(
                    psql.SQL("DELETE FROM {} WHERE table_name = %s").format(metadata_table),
                    (table_name,),
                )
            conn.commit()
        self._invalidate_cache(table_name)


# ---------------------------------------------------------------------------
# Backend-level implementation (lifecycle only)
# ---------------------------------------------------------------------------


class PgvectorBackend(BaseVectorBackend):
    """pgvector implementation of the vector backend.

    Responsible only for lifecycle management and embedding configuration.
    """

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._connections: List[PgVectorDb] = []
        self._pool: Optional[ConnectionPool] = None
        self._pool_lock = threading.Lock()
        self._isolation: IsolationType = IsolationType.PHYSICAL

    def initialize(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._isolation = IsolationType(config.get("isolation", IsolationType.PHYSICAL.value))

    def _get_or_create_pool(self) -> ConnectionPool:
        """Return the shared connection pool, creating it on first use."""
        if self._pool is not None:
            return self._pool

        with self._pool_lock:
            if self._pool is not None:
                return self._pool

            config = self._config

            _REQUIRED_KEYS = ("host", "port", "user", "password", "dbname")
            missing = [k for k in _REQUIRED_KEYS if k not in config]
            if missing:
                raise ValueError(f"Missing required PostgreSQL config keys: {', '.join(missing)}")

            host = config["host"]
            port = config["port"]
            user = config["user"]
            password = config["password"]
            dbname = config["dbname"]
            min_size = config.get("pool_min_size", 1)
            max_size = config.get("pool_max_size", 10)

            conninfo = f"host={host} port={port} user={user} password={password} dbname={dbname}"
            pool = ConnectionPool(
                conninfo=conninfo,
                min_size=min_size,
                max_size=max_size,
                kwargs={"row_factory": dict_row},
            )

            self._pool = pool
        return self._pool

    def connect(self, namespace: str) -> PgVectorDb:
        """Connect to PostgreSQL and return a VectorDatabase handle.

        Args:
            namespace: Logical namespace for data isolation.
        """
        pool = self._get_or_create_pool()
        db = PgVectorDb(
            pool=pool,
            config=self._config,
            namespace=namespace,
            isolation=self._isolation,
        )
        self._connections.append(db)
        return db

    def close(self) -> None:
        self._connections.clear()
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception as e:
                logger.warning("Error closing vector database connection pool: %s", e)
            self._pool = None
