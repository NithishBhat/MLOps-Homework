# Lab 3: Training and Serving a Model in Docker

Based on [Docker_Labs/Lab1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Docker_Labs/Lab1)
from the course repo. The original lab builds an image that trains a random
forest on the Iris dataset and saves it inside the container, then exports the
image with `docker save`. I kept the build → run → save flow but changed the
dataset, the models, the image and how results get out of the container. The
full list of changes is in [What I changed](#what-i-changed).

### Summary of changes

**Model** (`src/main.py`)
- Dataset changed from Iris to **breast cancer** (569 tumours, 30 features, malignant vs benign).
- Trains and compares **three models** (logistic regression, random forest, gradient boosting) on the same stratified hold-out split, and picks the best one by F1.
- Reports 5-fold CV accuracy, hold-out accuracy, F1 and ROC-AUC, and writes them to `metrics.json`.
- Quality gate: the container exits with code 1 if the best model's accuracy is below `MIN_ACCURACY`.
- Configurable with environment variables (`docker run -e MODEL=... -e SEED=...`) or command-line flags.

**New `src/predict.py`**: loads a saved model in a second container and predicts on sample rows, which shows the model can be reused after the training container is gone.

**Image** (`Dockerfile`)
- `python:3.11-slim` instead of the full `python:3.10` image.
- Dependencies are installed before the code is copied, so code changes don't reinstall packages.
- Runs as a non-root user, and writes to a `/app/output` volume instead of inside the container.
- `.dockerignore` keeps the build context small.

**CI** (`.github/workflows/lab3_docker.yml`): runs the tests, builds the image, trains and predicts in containers, posts a results table, and uploads the saved image as a download.

---

## 1. What you need installed

| Tool | Version | Notes |
|---|---|---|
| Docker | Docker Desktop 4.x or Docker Engine 24+ | <https://docs.docker.com/get-docker/>. On Windows, Docker Desktop must be **running** (whale icon in the tray) before any `docker` command works. |
| Git | any recent (2.30+) | <https://git-scm.com/downloads> |
| Python | 3.10 – 3.13 | Only needed to run the tests outside Docker |

To check:

```bash
docker version      # must show both a Client and a Server section
git --version
```

Python packages (pinned in [`src/requirements.txt`](src/requirements.txt), installed inside the image):

| Package | Version | Used for |
|---|---|---|
| numpy | 2.2.6 | arrays |
| scipy | 1.15.3 | needed by scikit-learn (pinned so its version matches scikit-learn 1.6) |
| scikit-learn | 1.6.1 | dataset, models, metrics |
| joblib | 1.4.2 | saving models |
| pytest | 8.3.5 | tests (only in `requirements-dev.txt`, not in the image) |

## 2. Build the image

```bash
git clone https://github.com/NithishBhat/MLOps-Homework.git
cd MLOps-Homework/Lab3

docker build -t lab3:v1 .
docker image ls lab3:v1
```

**Run every command below from inside the `Lab3/` folder.**

## 3. Train the models

Mount a local `output/` folder at `/app/output` so the models and metrics are
kept after the container exits:

```bash
# macOS / Linux / Git Bash
docker run --rm -v "$PWD/output:/app/output" lab3:v1

# Windows PowerShell
docker run --rm -v "${PWD}/output:/app/output" lab3:v1
```

Output:

```
Training on 455 rows, testing on 114 rows (seed=42)

model                   cv_acc     acc      f1     auc
logistic_regression     0.9802  0.9825  0.9861  0.9954
random_forest           0.9604  0.9561  0.9655  0.9931
gradient_boosting       0.9560  0.9561  0.9660  0.9907

Best model: logistic_regression (f1=0.9861, accuracy=0.9825)
Saved models and metrics.json to /app/output
```

After the run, `output/` contains:

- `models/logistic_regression.joblib`, `models/random_forest.joblib`, `models/gradient_boosting.joblib`
- `metrics.json`: the scores of every model, the best model, the seed, the scikit-learn version and whether the quality gate passed

Logistic regression (with feature scaling) wins on this dataset. The data is
small and close to linearly separable, so the simpler model beats the tree
ensembles.

On Linux, the container's user (uid 1000) may not be able to write to a folder
your user created. If you get `Permission denied`, run
`mkdir -p output && chmod 777 output` first.

### Options

Each option can be set with `-e NAME=value` on `docker run` (or as a flag if you
run `python main.py` yourself):

| Env variable | Flag | Default | Meaning |
|---|---|---|---|
| `MODEL` | `--model` | `all` | `all`, `logistic_regression`, `random_forest` or `gradient_boosting` |
| `TEST_SIZE` | `--test-size` | `0.2` | Fraction of rows held out for testing |
| `SEED` | `--seed` | `42` | Seed for the split and the models |
| `MIN_ACCURACY` | `--min-accuracy` | `0.90` | Quality gate on the best model's hold-out accuracy |
| `OUTPUT_DIR` | `--output-dir` | `/app/output` | Where models and `metrics.json` are written |

For example, to train only the random forest with a different seed:

```bash
docker run --rm -v "$PWD/output:/app/output" -e MODEL=random_forest -e SEED=7 lab3:v1
```

To see the quality gate fail, pass `-e MIN_ACCURACY=0.99`. The container prints
`Quality gate failed…` and exits with code 1 (check with `echo $?`, or
`$LASTEXITCODE` in PowerShell).

## 4. Predict with a saved model

The same image can run a different command. This starts a new container that
loads the best model from `output/` and predicts on random rows:

```bash
docker run --rm -v "$PWD/output:/app/output" lab3:v1 python predict.py -n 8
```

```
Predictions from logistic_regression:

  row  predicted  confidence  actual
  478  benign          0.999  benign
   42  malignant       1.000  malignant
    9  malignant       1.000  malignant
  ...
```

Use `--model random_forest` to pick a specific model. Wrong predictions are
marked with `<- wrong`.

## 5. Save and load the image

```bash
docker save lab3:v1 -o lab3_v1.tar
```

This writes a file of about 126 MB. The original lab used
`docker save lab3:v1 > my_image.tar`. In Windows PowerShell 5.1, `>` re-encodes
output as text and corrupts the tar file, so `-o` is safer on every OS.

To load it on another machine (or after deleting the image) and run it:

```bash
docker load -i lab3_v1.tar
docker run --rm lab3:v1
```

`output/` and `*.tar` are gitignored.

## 6. Run the tests

The tests run the training and prediction code directly with Python (no Docker):

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

```
.....                                                                    [100%]
5 passed
```

They check that one model or all models can be trained and saved, that the
best model is the one with the highest F1, that the quality gate exits with 1,
that the split can be reproduced from its seed, and that `predict.py` loads the
best model. They write to a temp folder, not to `output/`.

## 7. Running it in GitHub Actions

The workflow is [`.github/workflows/lab3_docker.yml`](../.github/workflows/lab3_docker.yml).
It runs on pushes and pull requests that change `Lab3/`, and can be started
manually from **Actions → "Lab3 - Build, Run & Save Docker Image" → Run workflow**,
where you can choose the model and the accuracy threshold.

A successful run:

- has green **Unit tests** and **Build, run and save image** jobs (the image is only built if the tests pass);
- shows the image size, the training table and the predictions in the job log;
- shows a metrics table for every model on the run's **Summary** page;
- has a downloadable artifact `lab3-run-<number>` with `output/` (models and
  `metrics.json`) and `lab3_v1.tar.gz`. Load it with
  `gunzip -c lab3_v1.tar.gz | docker load` (or `docker load -i lab3_v1.tar.gz`,
  which reads gzip directly).

---

## What I changed

### Model and scripts (`src/`)

| Change | Why |
|---|---|
| Iris → **breast cancer** dataset | A binary medical task with 30 features is closer to a real use case than 3 flower species. It also makes F1 and ROC-AUC meaningful. |
| **Three models compared** on one stratified split | The original trained one model and printed only "Model trained successfully" with no score. Now every model is scored on the same unseen rows and the best is chosen by F1. |
| **Metrics**: 5-fold CV accuracy, accuracy, F1, ROC-AUC | CV accuracy on the training set shows how stable a model is. The hold-out metrics show how well it does on unseen data. |
| `metrics.json` report | Records the scores together with the seed, the row counts and the scikit-learn version, so a run can be compared and reproduced later. |
| **Quality gate** (`MIN_ACCURACY`) | The container exits with code 1 if the best model is too weak, so a script or CI job can tell that the run failed. |
| Config by **environment variables** | `docker run -e ...` is the normal way to configure a container without rebuilding the image. Flags still work for local runs. |
| New **`predict.py`** | Loads a saved model in a separate container to show that the trained model is usable outside the training run. |
| Pinned `requirements.txt` (with scipy) | The original had no versions. An unpinned newer SciPy caused warnings with scikit-learn 1.6, so every package is pinned. |

### Image (`Dockerfile`, `.dockerignore`)

| Change | Why |
|---|---|
| `python:3.11-slim` base | The full `python:3.10` image includes compilers and tools the lab doesn't need. The slim image makes the saved tar about 126 MB. |
| Copy `requirements.txt` first, then the code | Docker caches the `pip install` layer, so changing only the code rebuilds in seconds. |
| `PYTHONUNBUFFERED`, `PYTHONDONTWRITEBYTECODE`, `PIP_NO_CACHE_DIR` | Logs appear right away in `docker run`/`docker logs`, and no `.pyc` files or pip cache end up in the image. |
| **Non-root user** (`appuser`) | Containers shouldn't run as root unless they have to. |
| **`/app/output` volume** | In the original, `iris_model.pkl` was saved inside the container and lost when it was removed. Now models and metrics are written to a mounted folder on the host. |
| `.dockerignore` | Keeps `output/`, tests, virtualenvs and saved tars out of the build context. |
| `Dockerfile` (capital D) | The standard name, so `docker build .` finds it without `-f` on case-sensitive file systems. |

### CI (`.github/workflows/lab3_docker.yml`)

| Change | Why |
|---|---|
| Tests before the Docker build | The image is only built if the code passes its tests. |
| Train and predict in containers | Checks that the image works, not just that it builds. A failed quality gate fails the job. |
| Results table on the Summary page | Shows the scores without opening the logs. |
| `docker save` uploaded as an artifact | Each run's image (and its models) can be downloaded and loaded with `docker load`. |
| `paths:` filter, `workflow_dispatch` inputs | Only Lab 3 changes trigger it. Manual runs can pick the model and the threshold. |

## Folder layout

```
Lab3/
├── src/
│   ├── main.py            # train + compare models, write models/ and metrics.json
│   ├── predict.py         # load a saved model and predict on sample rows
│   └── requirements.txt   # installed in the image
├── test/test_main.py      # pytest suite (run in CI before the image is built)
├── Dockerfile
├── .dockerignore
├── pytest.ini
├── requirements-dev.txt   # image requirements + pytest
└── README.md
```
