"""DAG structure tests. They need Airflow, so they are skipped in a plain virtualenv.

Run them inside the Airflow image (see the README or the CI workflow):
    docker compose run --rm -v ./test:/opt/airflow/test airflow \
        bash -c "pip install -q pytest==8.3.5 && python -m pytest /opt/airflow/test/test_dag.py"
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("airflow")
from airflow.dag_processing.dagbag import DagBag  # noqa: E402

DAGS_DIR = Path("/opt/airflow/dags")


@pytest.fixture(scope="module")
def dagbag():
    # Airflow puts the dags folder on sys.path when it runs; a bare DagBag doesn't
    sys.path.insert(0, str(DAGS_DIR))
    # Example DAGs are already off (AIRFLOW__CORE__LOAD_EXAMPLES=false in docker-compose.yaml)
    return DagBag(dag_folder=str(DAGS_DIR))


@pytest.fixture(scope="module")
def dag(dagbag):
    # Read the parsed DAG directly; dagbag.get_dag() would need the metadata database
    return dagbag.dags["lab5_digit_clustering"]


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}


def test_tasks_and_order(dag):
    assert set(dag.task_ids) == {
        "extract_data", "validate_data", "candidate_ks", "score_k",
        "choose_k", "train_model", "evaluate_model", "write_report", "cleanup_old_runs",
    }
    downstream = {t.task_id: set(t.downstream_task_ids) for t in dag.tasks}
    assert downstream["extract_data"] == {"validate_data"}
    assert {"candidate_ks", "score_k", "train_model"} <= downstream["validate_data"]
    assert downstream["score_k"] == {"choose_k"}
    assert downstream["evaluate_model"] == {"write_report"}
    assert downstream["write_report"] == {"cleanup_old_runs"}
    assert dag.get_task("cleanup_old_runs").task_type == "BashOperator"


def test_schedule_and_params(dag):
    assert dag.schedule == "0 6 * * 1"
    assert type(dag.timetable).__name__ == "CronTriggerTimetable"
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    assert dag.params["select_by"] == "silhouette"
    assert dag.params["k_min"] < dag.params["k_max"]
