"""Run native deletion in a child so a LanceDB crash becomes a test failure.

Install datus-storage-base[lancedb-test] before running this suite.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import datus_storage_base


@pytest.mark.parametrize(
    ("keep_count", "expected_extra"),
    [
        (0, ["other-model", "other-datasource"]),
        (645, ["other-model", "other-datasource", "null-id"]),
        (2048, ["other-model", "other-datasource", "null-id"]),
    ],
)
def test_delete_stale_objects_preserves_retained_ids_and_scope(tmp_path, keep_count, expected_extra):
    script = textwrap.dedent(
        """
        import json
        import sys

        import lancedb
        from datus_storage_base.conditions import and_, build_where, eq, in_, not_

        path, count, extra = sys.argv[1:]
        keep_ids = [f"object_{i:04d}" for i in range(int(count))]
        rows = [
            {"label": item, "id": item, "yaml_path": "model.yml", "datasource_id": "games"}
            for item in keep_ids
        ]
        rows.extend([
            {"label": "stale", "id": "stale", "yaml_path": "model.yml", "datasource_id": "games"},
            {"label": "other-model", "id": "outside", "yaml_path": "other.yml", "datasource_id": "games"},
            {"label": "other-datasource", "id": "outside", "yaml_path": "model.yml", "datasource_id": "other"},
            {"label": "null-id", "id": None, "yaml_path": "model.yml", "datasource_id": "games"},
        ])
        table = lancedb.connect(path).create_table("objects", data=rows)
        condition = and_(eq("yaml_path", "model.yml"), not_(in_("id", keep_ids)), eq("datasource_id", "games"))
        table.delete(build_where(condition))
        actual = sorted(table.to_arrow()["label"].to_pylist())
        expected = sorted(keep_ids + json.loads(extra))
        assert actual == expected, (actual, expected)
        print(json.dumps({"remaining": len(actual)}))
        """
    )
    # Ensure the subprocess exercises the same source package as this test run.
    package_root = str(Path(datus_storage_base.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": package_root, "PYTHONFAULTHANDLER": "1"}
    result = subprocess.run(
        [sys.executable, "-u", "-c", script, str(tmp_path / "db"), str(keep_count), json.dumps(expected_extra)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"LanceDB child exited {result.returncode}\n{result.stdout}\n{result.stderr}"
    assert json.loads(result.stdout) == {"remaining": keep_count + len(expected_extra)}
