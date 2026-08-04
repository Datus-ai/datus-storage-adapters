from datus_storage_postgresql._testcontainers import CI_RUN_ID_LABEL
from datus_storage_postgresql._testcontainers import testcontainer_labels as _testcontainer_labels


def test_testcontainer_labels_is_empty_without_run_id(monkeypatch):
    monkeypatch.delenv("DATUS_TEST_RUN_ID", raising=False)

    assert _testcontainer_labels() == {}


def test_testcontainer_labels_uses_run_id(monkeypatch):
    monkeypatch.setenv("DATUS_TEST_RUN_ID", "nightly-123-1")

    assert _testcontainer_labels() == {CI_RUN_ID_LABEL: "nightly-123-1"}
