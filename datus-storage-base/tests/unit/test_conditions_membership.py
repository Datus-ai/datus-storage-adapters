"""Inline membership rendering and SQL three-valued semantics."""

import sqlite3
from datetime import date

import pytest

from datus_storage_base.conditions import build_where, in_, not_


def test_membership_uses_native_in_and_escapes_literals():
    """Quoted identifiers and values remain safe inside native membership."""
    assert build_where(in_("object id", ["a", "O'Hara"])) == "(\"object id\" IN ('a', 'O''Hara'))"


def test_large_membership_does_not_create_a_linear_boolean_tree():
    """Retained-ID lists must not grow the deletion predicate's boolean depth."""
    where = build_where(not_(in_("id", [f"object_{i}" for i in range(2048)])))
    assert "id IN (" in where
    assert " OR " not in where


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], "1 = 0"),
        ([None], "(id IS NULL)"),
        (["a", None], "(id IN ('a') OR id IS NULL)"),
        ([True, False], "(id IN (TRUE, FALSE))"),
        ([1, 2.5], "(id IN (1, 2.5))"),
        ([date(2026, 11, 6)], "(id IN ('2026-11-06'))"),
    ],
)
def test_membership_literals(values, expected):
    """Render empty, nullable, and typed lists without changing their meaning."""
    assert build_where(in_("id", values)) == expected


@pytest.mark.parametrize(
    ("values", "positive", "negative"),
    [
        ([], [0, 0, 0, 0], [1, 1, 1, 1]),
        ([None], [1, 0, 0, 0], [0, 1, 1, 1]),
        (["a"], [None, 1, 0, 0], [None, 0, 1, 1]),
        (["a", None], [1, 1, 0, 0], [0, 0, 1, 1]),
        ([None, "a", "a", None], [1, 1, 0, 0], [0, 0, 1, 1]),
        (["O'Hara"], [None, 0, 0, 1], [None, 1, 1, 0]),
    ],
)
def test_membership_preserves_null_and_negation_semantics(values, positive, negative):
    """Execute both predicates to verify SQL true, false, and unknown results."""
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE objects (id TEXT)")
        conn.executemany("INSERT INTO objects VALUES (?)", [(None,), ("a",), ("b",), ("O'Hara",)])
        predicate = in_("id", values)
        actual = conn.execute(
            f"SELECT {build_where(predicate)}, {build_where(not_(predicate))} FROM objects ORDER BY rowid"
        ).fetchall()
    assert actual == list(zip(positive, negative, strict=True))
