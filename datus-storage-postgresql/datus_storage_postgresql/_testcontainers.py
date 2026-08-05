"""Shared Testcontainers configuration for PostgreSQL test environments."""

import os

CI_RUN_ID_ENV = "DATUS_TEST_RUN_ID"
CI_RUN_ID_LABEL = "com.datus.ci.run-id"


def testcontainer_labels() -> dict[str, str]:
    """Return the current CI run label when one was provided."""
    run_id = os.environ.get(CI_RUN_ID_ENV, "").strip()
    return {CI_RUN_ID_LABEL: run_id} if run_id else {}
