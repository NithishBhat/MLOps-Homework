# Lab 5: An Airflow Pipeline for Clustering Handwritten Digits

Based on [Airflow_Labs/Lab_1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs/Lab_1)
from the course repo. The original lab runs Airflow in Docker with a DAG of
four `PythonOperator` tasks that cluster credit-card customers with KMeans and
report the elbow k. I kept the idea (an Airflow DAG that trains and checks a
KMeans model) but changed the dataset, the model, the DAG and the Airflow
setup. The full list is in [What I changed](#what-i-changed).

### Summary of changes

**Data and model** (`dags/clustering/pipeline.py`)
- Dataset changed from the credit-card CSV to scikit-learn's **handwritten digits** (1,797 8x8 images, built in, so nothing to download).
- PCA (keeps 80% of the pixel variance, 64 → 13 features) + KMeans, saved as one scikit-learn `Pipeline`.
- k is chosen by **silhouette score** (default) or by the elbow, and the final model is trained at that k.
- Evaluated on a hold-out set the model never saw: adjusted Rand index, normalized mutual information, silhouette, and **mapped accuracy** (each cluster is named after the digit most of its training images show).
- Quality gate: the run fails if mapped accuracy is below `min_accuracy`.

**DAG** (`dags/lab5_digit_clustering.py`)
- 9 tasks: 8 written with the TaskFlow API (`@task`), including data validation and a Markdown report, plus a `BashOperator` that deletes old run folders.
- **Dynamic task mapping**: one `score_k` task per candidate k (2–20), run in parallel.
- Tasks pass small JSON dicts through XCom. Data and models are files in a per-run folder, instead of base64-encoded pickles in XCom.
- Cron schedule (Mondays 06:00 UTC), run-time **params** (k range, selection method, PCA variance, gate threshold, seed) and retries.

**Setup, tests and CI**
- One-container Airflow 3.3 (`airflow standalone`) instead of the official multi-container compose file.
- 7 pipeline tests + 3 DAG structure tests (the original had none).
- GitHub Actions workflow that runs the tests and then runs the whole DAG inside Airflow.

---

## 1. What you need installed

| Tool | Version | Notes |
|---|---|---|
| Docker | Docker Desktop 4.x or Docker Engine 24+ with Compose v2 | <https://docs.docker.com/get-docker/>. Give Docker at least 4 GB of memory. The running container uses about 1.8 GB. |
| Git | any recent (2.30+) | <https://git-scm.com/downloads> |
| Python | 3.12 (optional) | Only to run the pipeline tests without Docker |

The image is built from `apache/airflow:3.3.2-python3.12`. It already contains
numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1 and joblib 1.6.0. The only extra
package is `kneed` 0.9.0 (for the elbow), pinned in
[`requirements.txt`](requirements.txt).

## 2. Start Airflow

```bash
git clone https://github.com/NithishBhat/MLOps-Homework.git
cd MLOps-Homework/Lab5

# Linux only: make files Airflow writes belong to your user (step 2c of the original lab)
echo "AIRFLOW_UID=$(id -u)" > .env

docker compose up -d --build
```

**Run every command below from inside the `Lab5/` folder.**

The first build downloads the Airflow image (about 3 GB), so it takes a few
minutes. After that, startup takes about 20 seconds. Check that it's ready:

```bash
docker compose ps        # STATUS shows "(healthy)"
```

Open <http://localhost:8080> and log in as **`admin`**. There is no default
password: Airflow generates a random one the first time the container starts.
Print it with:

```bash
docker compose exec airflow cat simple_auth_manager_passwords.json.generated
```

The output is `{"admin": "<password>"}`. To use another user name, put
`AIRFLOW_ADMIN_USER=yourname` in `.env` before `docker compose up`. A new
container (after `docker compose down`) gets a new password.

To stop Airflow: `docker compose down`. Your runs' files in `working_data/` are
kept.

## 3. Run the DAG

In the UI, open **Dags → lab5_digit_clustering**:

1. Switch the toggle **on** (DAGs start paused). Because the DAG is scheduled
   `0 6 * * 1` (Mondays 06:00 UTC) with `catchup=False`, Airflow immediately starts one run for the
   most recent week.
2. To run it with your own settings, click **Trigger**. The form shows the
   params:

| Param | Default | Meaning |
|---|---|---|
| `k_min`, `k_max` | 2, 20 | Range of k values to try (one parallel `score_k` task each) |
| `select_by` | `silhouette` | `silhouette` (highest score) or `elbow` (knee of the inertia curve) |
| `pca_variance` | 0.8 | Fraction of pixel variance PCA keeps |
| `min_accuracy` | 0.8 | Quality gate on hold-out mapped accuracy |
| `seed` | 42 | Seed for the split, PCA and KMeans |

The same from the command line:

```bash
docker compose exec airflow airflow dags unpause lab5_digit_clustering
docker compose exec airflow airflow dags trigger lab5_digit_clustering --conf '{"k_max": 25}'
docker compose exec airflow airflow dags list-runs lab5_digit_clustering
```

(In PowerShell, write the JSON as `--conf '{\"k_max\": 25}'`.)

### The tasks

```
extract_data → validate_data → candidate_ks → score_k ×19 (parallel) → choose_k → train_model → evaluate_model → write_report → cleanup_old_runs
```

| Task | What it does |
|---|---|
| `extract_data` | Loads the digits and saves a stratified 80/20 train / hold-out split |
| `validate_data` | Fails the run if the data has the wrong shape, NaNs, pixels outside 0–16, or a missing digit |
| `candidate_ks` | Turns `k_min`/`k_max` into a list |
| `score_k` | One mapped task per k: fits PCA + KMeans on the training images and returns inertia and silhouette |
| `choose_k` | Picks k by silhouette or elbow, and logs both |
| `train_model` | Trains the final model at the chosen k and names each cluster after its majority digit |
| `evaluate_model` | Scores the hold-out images and applies the quality gate |
| `write_report` | Writes `report.md` and prints it in the task log |
| `cleanup_old_runs` | `BashOperator`: deletes run folders in `working_data/` older than 28 days |

### Results

Each run writes to `working_data/<run_id>/`: `train.npz`, `holdout.npz`,
`model.joblib`, `metrics.json` and `report.md`. With the default params:

```
Chosen k=17 (elbow=9, best silhouette=17)
```

| Hold-out metric | Value |
|---|---|
| mapped accuracy | 0.906 |
| adjusted Rand index | 0.670 |
| normalized mutual info | 0.802 |
| silhouette | 0.234 |

KMeans never sees the labels, but 17 clusters map back to the right digit 91%
of the time. Several digits get more than one cluster, one for each common way
of writing them (digit 1 gets three). The report has a table of every cluster
with its size and purity.

**The quality gate in action**: trigger the DAG with
`{"select_by": "elbow"}`. The elbow is at k=9, which is too few clusters for
10 digits. Mapped accuracy drops to 0.706, `evaluate_model` fails with
`Quality gate failed: mapped accuracy 0.7056 < 0.8`, and the run is marked
failed without writing a report.

## 4. Run the tests

**Pipeline tests** (no Airflow needed; Python 3.12 recommended):

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

```
.......s                                                                 [100%]
7 passed, 1 skipped
```

They cover the stratified split, validation (passes on real data, and catches
NaNs, out-of-range pixels and missing digits), scoring one k, choosing k by
silhouette and by elbow, training + evaluation + the report, and the quality
gate. The skipped file is `test_dag.py`, which needs Airflow.

**DAG structure tests** (inside the Airflow image):

```bash
docker compose run --rm -v "$PWD/test:/opt/airflow/test" airflow \
  bash -c "pip install -q pytest==8.3.5 && python -m pytest /opt/airflow/test/test_dag.py"
```

(PowerShell: `-v "${PWD}/test:/opt/airflow/test"`. Git Bash: `-v "$(pwd -W)/test:/opt/airflow/test"`.)
These load the `dags/` folder with Airflow's `DagBag` and check: no import
errors, the 9 task ids and their order, the cron schedule, `catchup=False`, `max_active_runs=1`
and the default params.

**The whole DAG without the scheduler**:

```bash
docker compose run --rm airflow bash -c "airflow db migrate && airflow dags test lab5_digit_clustering"
```

## 5. Running it in GitHub Actions

The workflow is [`.github/workflows/lab5_airflow.yml`](../.github/workflows/lab5_airflow.yml).
It runs on pushes and pull requests that change `Lab5/`, and can be started
from **Actions → "Lab5 - Airflow Clustering DAG" → Run workflow**.

A successful run:

- has green **Pipeline tests** and **Run the DAG in Airflow** jobs;
- in the second job, builds the image, runs the DAG structure tests inside it,
  then runs the whole DAG with `airflow dags test` and checks that the report says the gate passed;
- shows `report.md` on the run's **Summary** page;
- has a downloadable artifact `lab5-run-<number>` with the report, metrics and model.

---

## What I changed

### Data and model

| Change | Why |
|---|---|
| Credit-card CSV → **handwritten digits** (`sklearn.datasets.load_digits`) | Built into scikit-learn, so nothing to download or commit. The true labels let me check how good the clusters are, which the original couldn't do. |
| **Train / hold-out split** | The original chose k on the same data it trained on and never measured the clusters. Now the metrics come from images the model never saw. |
| **PCA** before KMeans, in one `Pipeline` | Reduces 64 noisy pixel features to 13 components. Because it's one saved pipeline, the hold-out images get exactly the same transform as the training images. |
| **Final model trained at the chosen k** | The original's loop saved the last model it fitted (k=49), whatever the elbow said. |
| **Prediction on transformed data** | The original trained on Min-Max-scaled data but called `predict` on raw `test.csv`. |
| **Silhouette** as the default way to choose k, elbow kept as an option | The digits' inertia curve has no sharp elbow. Silhouette gives a better k here (accuracy 0.91 vs 0.71). |
| **Evaluation + quality gate** | ARI, NMI, silhouette and mapped accuracy on the hold-out set, and a failed run when accuracy is too low. The original only printed the elbow k. |
| **Validation task** | Catches bad data (shape, NaNs, pixel range, missing digits) before any training. |

### DAG

| Change | Why |
|---|---|
| **TaskFlow API** (`@dag`, `@task`) instead of `PythonOperator` + `op_args` | Dependencies come from passing return values, so the code reads like normal Python. |
| **Dynamic task mapping** (`score_k.partial(...).expand(k=...)`) | Scoring each k is its own task and they run in parallel. Each k gets its own log and its own retry, and the k range is set when the DAG is triggered. |
| **File paths in XCom, not data** | The original passed base64-encoded pickled DataFrames through XCom, which stores them in the database. Now XCom holds small JSON dicts and the data is in `working_data/<run_id>/`, one folder per run. |
| **Params** with types and limits | `k_min`, `k_max`, `select_by`, `pca_variance`, `min_accuracy` and `seed` show up as a form in the Trigger dialog, and bad values are rejected. |
| Cron schedule `0 6 * * 1`, `catchup=False`, `max_active_runs=1`, retries | The original was manual only. Gate and scoring tasks have `retries=0`, because retrying a failed gate gives the same answer. |
| `BashOperator` cleanup task | Each run leaves its data and model in `working_data/`. Old run folders are deleted after 28 days so the folder doesn't grow forever. |
| `doc_md` | The task list shows up on the DAG's page in the UI. |
| Pipeline code in its own module (`dags/clustering/pipeline.py`) | No Airflow imports, so every step can be tested with pytest. |

### Airflow setup

| Change | Why |
|---|---|
| **Airflow 3.3.2** | The original README fetches the 2.5.1 compose file, while its DAG uses Airflow 3 imports. Here the DAG, the image and the docs all use the same version. |
| **One container** (`airflow standalone`) | The official compose file runs Postgres, Redis, Celery workers and more, and asks for 4–8 GB. This uses about 1.8 GB and one `docker compose up`. |
| **Custom image** with `kneed` installed at build time | `_PIP_ADDITIONAL_REQUIREMENTS` (used by the original) reinstalls packages every time the container starts. |
| Generated admin password instead of fixed credentials | Step 2d of the original sets the login to `airflow2` / `airflow2`. Here Airflow makes a random password, so there is no known default to guess. |
| Health check, auto-restart | `docker compose ps` shows when Airflow is ready. |

### Tests and CI

| Change | Why |
|---|---|
| `test/test_pipeline.py` (7 tests) and `test/test_dag.py` (3 tests) | The original had no tests. The DAG tests fail if any DAG file has an import error. |
| `.github/workflows/lab5_airflow.yml` | Runs the tests, then the whole DAG in the real Airflow image, on every change to `Lab5/`. |

## Folder layout

```
Lab5/
├── dags/
│   ├── lab5_digit_clustering.py   # the DAG (TaskFlow, dynamic mapping, params)
│   └── clustering/
│       └── pipeline.py            # the code each task runs (no Airflow imports)
├── test/
│   ├── test_pipeline.py           # pipeline tests (plain pytest)
│   └── test_dag.py                # DAG structure tests (inside the Airflow image)
├── working_data/                  # created by runs: one folder per run (gitignored)
├── logs/                          # Airflow task logs (gitignored)
├── Dockerfile                     # Airflow 3.3.2 + kneed
├── docker-compose.yaml            # single-container Airflow
├── requirements.txt               # extra packages for the image
├── requirements-dev.txt           # for running the pipeline tests without Docker
├── pytest.ini
└── README.md
```
