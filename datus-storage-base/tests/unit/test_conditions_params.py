"""Unit tests for compile_where_params — the parameterized WHERE compiler."""

import pytest

from datus_storage_base.conditions import (
    and_,
    build_where,
    compile_where_params,
    eq,
    gt,
    in_,
    like,
    lte,
    ne,
    not_,
    or_,
)


class TestScalarConditions:
    def test_eq_compiles_to_placeholder(self):
        assert compile_where_params(eq("status", "active")) == ("status = %s", ["active"])

    def test_comparison_operators(self):
        assert compile_where_params(gt("score", 80)) == ("score > %s", [80])
        assert compile_where_params(lte("age", 30)) == ("age <= %s", [30])
        assert compile_where_params(ne("kind", "x")) == ("kind != %s", ["x"])

    def test_eq_none_compiles_to_is_null_without_params(self):
        assert compile_where_params(eq("deleted_at", None)) == ("deleted_at IS NULL", [])

    def test_ne_none_compiles_to_is_not_null(self):
        assert compile_where_params(ne("deleted_at", None)) == ("deleted_at IS NOT NULL", [])

    def test_gt_none_raises(self):
        with pytest.raises(ValueError):
            compile_where_params(gt("score", None))

    def test_nan_rejected_as_parameter(self):
        with pytest.raises(ValueError):
            compile_where_params(eq("score", float("nan")))


class TestValuesNeverEnterSql:
    """The compiler's core contract: values travel as params, not as SQL text."""

    def test_percent_in_value_stays_out_of_sql(self):
        fragment, params = compile_where_params(eq("category", "50% off"))
        assert "%" not in fragment.replace("%s", "")
        assert params == ["50% off"]

    def test_quote_in_value_stays_out_of_sql(self):
        fragment, params = compile_where_params(eq("name", "x' OR '1'='1"))
        assert "'" not in fragment
        assert params == ["x' OR '1'='1"]

    def test_like_pattern_is_a_parameter(self):
        fragment, params = compile_where_params(like("source_table", "orders*"))
        assert fragment == "source_table LIKE %s ESCAPE '\\'"
        assert params == ["orders%"]

    def test_like_escapes_literal_wildcards_in_pattern(self):
        _, params = compile_where_params(like("name", "100%*"))
        assert params == ["100\\%%"]


class TestInCompilation:
    def test_in_compiles_to_native_in(self):
        assert compile_where_params(in_("t", ["a", "b"])) == ("(t IN (%s, %s))", ["a", "b"])

    def test_in_with_null_adds_is_null_alternative(self):
        assert compile_where_params(in_("t", ["a", None])) == ("(t IN (%s) OR t IS NULL)", ["a"])

    def test_empty_in_is_always_false(self):
        assert compile_where_params(in_("t", [])) == ("1 = 0", [])

    def test_in_rejects_string_value(self):
        # The in_() factory list()s its argument, so the guard is only
        # reachable through a directly-built Condition — same as build_where.
        from datus_storage_base.conditions import Condition, Op

        with pytest.raises(TypeError):
            compile_where_params(Condition("t", Op.IN, "abc"))


class TestNestedComposition:
    def test_nested_three_levels_orders_params_left_to_right(self):
        expr = and_(
            not_(eq("is_blocked", True)),
            or_(like("name", "Alice*"), and_(eq("country", "US"), gt("age", 18))),
        )
        fragment, params = compile_where_params(expr)
        assert fragment == ("((NOT is_blocked = %s) AND (name LIKE %s ESCAPE '\\' OR (country = %s AND age > %s)))")
        assert params == [True, "Alice%", "US", 18]

    def test_empty_and_or_keep_build_where_semantics(self):
        assert compile_where_params(and_()) == ("1 = 1", [])
        assert compile_where_params(or_()) == ("1 = 0", [])

    def test_parenthesization_matches_build_where(self):
        expr = and_(eq("a", 1), or_(eq("b", 2), eq("c", 3)))
        fragment, _ = compile_where_params(expr)
        # Same structure as the inline renderer, with values swapped for %s.
        assert build_where(expr) == "(a = 1 AND (b = 2 OR c = 3))"
        assert fragment == "(a = %s AND (b = %s OR c = %s))"


class TestInputContract:
    def test_none_compiles_to_none(self):
        assert compile_where_params(None) == (None, [])

    def test_raw_string_rejected(self):
        with pytest.raises(TypeError, match="SQL injection"):
            compile_where_params("category = 'alpha'")
