"""Unit tests for PgVectorTable WHERE compilation — no database required.

Guards the failure mode that hit production: a WHERE fragment with inlined
literals containing ``%`` (LIKE patterns) spliced into a statement executed
with parameters made psycopg's client-side placeholder parser raise
``ProgrammingError: only '%s', '%b', '%t' are allowed as placeholders``.
The psycopg smoke tests below run the real placeholder parser
(``PostgresQuery.convert``) without a server connection.
"""

import pytest
from psycopg._queries import PostgresQuery
from psycopg.adapt import Transformer

from datus_storage_base.backend_config import IsolationType
from datus_storage_base.conditions import and_, eq, like
from datus_storage_postgresql.vector.backend import PgVectorTable


def _table(isolation=IsolationType.LOGICAL, namespace="tenant_a") -> PgVectorTable:
    """Bare table handle with only the attributes _compiled_where reads."""
    table = PgVectorTable.__new__(PgVectorTable)
    table._isolation = isolation
    table._logical_namespace = namespace
    return table


def _assert_psycopg_accepts(sql: str, params: list) -> None:
    """Run psycopg's client-side placeholder parser on the statement."""
    PostgresQuery(Transformer()).convert(sql.encode(), params)


class TestCompiledWhere:
    def test_namespace_param_precedes_condition_params(self):
        clause, params = _table()._compiled_where(like("source_table", "orders*"))
        assert clause == "_datus_namespace = %s AND (source_table LIKE %s ESCAPE '\\')"
        assert params == ["tenant_a", "orders%"]

    def test_physical_isolation_passes_condition_through(self):
        clause, params = _table(isolation=IsolationType.PHYSICAL)._compiled_where(eq("category", "alpha"))
        assert clause == "category = %s"
        assert params == ["alpha"]

    def test_no_where_under_logical_isolation_scopes_namespace_only(self):
        clause, params = _table()._compiled_where(None)
        assert clause == "_datus_namespace = %s"
        assert params == ["tenant_a"]

    def test_no_where_physical_isolation_is_empty(self):
        assert _table(isolation=IsolationType.PHYSICAL)._compiled_where(None) == ("", [])

    def test_raw_string_where_rejected(self):
        with pytest.raises(TypeError, match="SQL injection"):
            _table()._compiled_where("category = 'alpha'")


class TestPsycopgAcceptsComposedStatements:
    def test_like_filter_with_namespace_params(self):
        # The production crash shape: LIKE pattern + logical-namespace param.
        clause, params = _table()._compiled_where(and_(like("source_table", "orders*"), eq("schema_name", "public")))
        sql = f"SELECT id FROM t WHERE {clause} ORDER BY vec <=> %s::vector LIMIT %s"
        _assert_psycopg_accepts(sql, params + ["[0.1]", 5])

    def test_value_containing_percent_with_params(self):
        clause, params = _table()._compiled_where(eq("category", "50% off"))
        _assert_psycopg_accepts(f"SELECT id FROM t WHERE {clause}", params)

    def test_like_filter_without_namespace_still_parameterized(self):
        # search_vector always executes with params (embedding + LIMIT), so a
        # LIKE filter must be parameterized even under physical isolation.
        clause, params = _table(isolation=IsolationType.PHYSICAL)._compiled_where(like("name", "*Bob*"))
        sql = f"SELECT id FROM t WHERE {clause} ORDER BY vec <=> %s::vector LIMIT %s"
        _assert_psycopg_accepts(sql, params + ["[0.1]", 5])
