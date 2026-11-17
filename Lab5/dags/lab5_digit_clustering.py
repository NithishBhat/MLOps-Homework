"""
### Lab 5: digit clustering pipeline

Clusters the scikit-learn handwritten digits (8x8 images) with PCA + KMeans.

1. **extract_data**: load the images and save a stratified train / hold-out split
2. **validate_data**: check shapes, NaNs, pixel range and that every digit is present
3. **candidate_ks**: turn the `k_min`/`k_max` params into a list of k values
4. **score_k**: one mapped task per k, run in parallel (inertia + silhouette)
5. **choose_k**: pick k by silhouette (default) or by the elbow of the inertia curve
6. **train_model**: fit the final model at the chosen k
7. **evaluate_model**: hold-out ARI / NMI / silhouette, and a quality gate on mapped accuracy
8. **write_report**: Markdown report in the run's folder
9. **cleanup_old_runs**: delete run folders older than 28 days (BashOperator)

Outputs go to `working_data/<run_id>/`.
"""
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param, dag, task

from clustering import pipeline

WORKING_DIR = Path(os.environ.get("LAB5_WORKING_DIR", "/opt/airflow/working_data"))
KEEP_RUNS_DAYS = 28


def run_folder(run_id):
    # run ids look like "manual__2026-11-17T10:00:00+00:00"; keep them safe as folder names
    return str(WORKING_DIR / re.sub(r"[^A-Za-z0-9_.-]", "_", run_id))


@dag(
    dag_id="lab5_digit_clustering",
    description="PCA + KMeans clustering of handwritten digits, with k chosen by parallel search",
    schedule="0 6 * * 1",  # cron: Mondays at 06:00 UTC
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "NithishBhat",
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
    },
    params={
        "k_min": Param(2, type="integer", minimum=2, maximum=30, description="Smallest k to try"),
        "k_max": Param(20, type="integer", minimum=3, maximum=40, description="Largest k to try"),
        "select_by": Param("silhouette", type="string", enum=["silhouette", "elbow"],
                           description="How to choose k"),
        "pca_variance": Param(0.8, type="number", exclusiveMinimum=0, maximum=1,
                              description="Fraction of pixel variance PCA keeps"),
        "min_accuracy": Param(0.8, type="number", minimum=0, maximum=1,
                              description="Quality gate: minimum hold-out mapped accuracy"),
        "seed": Param(42, type="integer"),
    },
    tags=["lab5", "clustering", "mlops"],
    doc_md=__doc__,
)
def lab5_digit_clustering():

    @task
    def extract_data(params=None, run_id=None):
        return pipeline.extract(run_folder(run_id), seed=params["seed"])

    @task
    def validate_data(run):
        pipeline.validate(run["run_dir"])
        return run

    @task
    def candidate_ks(run, params=None):
        if params["k_max"] <= params["k_min"]:
            raise ValueError("k_max must be larger than k_min")
        return list(range(params["k_min"], params["k_max"] + 1))

    # No retries: a failure here is a bug, not a flaky dependency
    @task(retries=0)
    def score_k(k, run, params=None):
        return pipeline.score_k(run["run_dir"], k, params["pca_variance"], params["seed"])

    @task
    def choose_k(scores, params=None):
        choice = pipeline.choose_k(list(scores), params["select_by"])
        print(f"Chosen k={choice['chosen_k']} (elbow={choice['elbow_k']}, "
              f"best silhouette={choice['best_silhouette_k']})")
        return choice

    @task
    def train_model(run, choice, params=None):
        info = pipeline.train(run["run_dir"], choice["chosen_k"], params["pca_variance"], params["seed"])
        return {**run, **info}

    # A failed quality gate should fail the run straight away, so no retries
    @task(retries=0)
    def evaluate_model(trained, params=None):
        return pipeline.evaluate(trained["run_dir"], params["min_accuracy"])

    @task
    def write_report(trained, choice, metrics):
        path = pipeline.report(trained["run_dir"], choice, metrics)
        print(Path(path).read_text())
        return path

    # Housekeeping: each run leaves ~1 MB in working_data/, so drop folders older than 4 weeks
    cleanup_old_runs = BashOperator(
        task_id="cleanup_old_runs",
        bash_command=(
            f'find "{WORKING_DIR}" -mindepth 1 -maxdepth 1 -type d -mtime +{KEEP_RUNS_DAYS} -print -exec rm -rf {{}} + ; '
            f'echo "Run folders kept: $(ls -1 "{WORKING_DIR}" | wc -l)"'
        ),
    )

    run = validate_data(extract_data())
    scores = score_k.partial(run=run).expand(k=candidate_ks(run))
    choice = choose_k(scores)
    trained = train_model(run, choice)
    metrics = evaluate_model(trained)
    write_report(trained, choice, metrics) >> cleanup_old_runs


lab5_digit_clustering()
