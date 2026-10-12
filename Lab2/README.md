# Lab 2: Model Training, Calibration and Versioning with GitHub Actions

Based on [Github_Labs/Lab2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2)
from the course repo. The lab's main idea is a GitHub Actions pipeline that
trains a model, evaluates it, versions it with a timestamp and commits the
results back to the repo. I rewrote the pipeline (the workflow and the scripts
it runs) as a **gated train → calibrate → evaluate → promote** pipeline. The
full list of changes is in [What I changed](#what-i-changed).

### Summary of changes

**Pipeline** (`.github/workflows/lab2_model_pipeline.yml`)
- Merged the original two workflows into one.
- Tests run on Python 3.11 and 3.13 before any training starts.
- Quality gate: the run fails, and commits nothing, if F1 is below the threshold.
- Champion/challenger: a new model replaces the saved one only if it scores at least as well.
- Can be started manually (choose sample size, threshold and calibration method). It also runs weekly, and pull requests only run the tests.
- Each run posts a results table on its GitHub page, saves its files for download, and commits safely as `github-actions[bot]`.

**Scripts** (`src/`)
- New `data.py`: the original scored the model on its own training data, with a random sample size that could be 0. Now every run is scored on the same fixed set of 2,000 unseen rows.
- New `calibrate_model.py`: the original calibration workflow never calibrated anything. Now it does, which lowers the Brier score (for example 0.100 → 0.095; closer to 0 means more accurate probabilities).
- Rewritten `evaluate_model.py`: computes the metrics, applies the quality gate and decides whether to promote the new model.
- Updated `train_model.py`: cleaner MLflow logging and compressed model files.
- New `test/test_pipeline.py`: 8 tests.
- `requirements.txt`: versions pinned.

---

## 1. What you need installed

| Tool | Version | Notes |
|---|---|---|
| Git | any recent (2.30+) | <https://git-scm.com/downloads> |
| Python | **3.10 – 3.13** (CI uses 3.11 and 3.13) | <https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"** in the installer. |
| GitHub account | n/a | Only needed to run the pipeline in GitHub Actions |

To check:

```bash
git --version
python --version      # on macOS/Linux this may be python3
```

Python packages (pinned in [`requirements.txt`](requirements.txt), installed in step 2):

| Package | Version | Used for |
|---|---|---|
| numpy | 2.2.6 | arrays, sampling |
| scikit-learn | 1.6.1 | random forest, calibration, metrics (1.6+ is needed for `FrozenEstimator`) |
| joblib | 1.4.2 | saving models |
| mlflow-skinny | 2.22.0 | logging training params and metrics to `mlruns/` |
| pytest | 8.3.5 | tests |

## 2. Setup

```bash
git clone https://github.com/NithishBhat/MLOps-Homework.git
cd MLOps-Homework/Lab2

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use cmd.

**Run every command below from inside the `Lab2/` folder** with the virtual
environment active. The scripts are run as modules (`python -m src.…`) so
that `src` imports work.

## 3. Running it locally

### Run the tests

```bash
pytest
```

A successful run looks like this:

```
........                                                                 [100%]
8 passed in 6.10s
```

The tests cover: the hold-out set is fixed, training samples can be reproduced
from their seed, invalid sample sizes are rejected, the gate and promotion
rules, a full two-run pipeline (the champion is kept and only one model file
is left in `models/`), and a gate failure that exits non-zero and promotes
nothing. They write to a temp folder, not to the repo.

### Run the pipeline by hand (the same 3 steps CI runs)

```bash
python -m src.train_model     --timestamp local1 --n-samples 2000 --seed 1
python -m src.calibrate_model --timestamp local1 --method sigmoid
python -m src.evaluate_model  --timestamp local1 --min-f1 0.80
```

A successful run ends by printing a report like this one (the numbers depend
on the seed):

```json
{
    "timestamp": "local1",
    "holdout_size": 2000,
    "min_f1": 0.8,
    "raw":        { "f1": 0.8685, "accuracy": 0.866,  "roc_auc": 0.9362, "brier": 0.1003 },
    "calibrated": { "f1": 0.8688, "accuracy": 0.8665, "roc_auc": 0.9362, "brier": 0.0954 },
    "passed_gate": true,
    "previous_champion": null,
    "promoted": true
}
```

A lower `brier` score for `calibrated` means its probabilities are closer to
the real outcomes. After the run you will have:

- `artifacts/`: raw model, calibrated model and the calibration split for this run (gitignored)
- `metrics/local1_metrics.json`: the report above
- `metrics/champion.json`: the current best model and its scores
- `models/model_local1.joblib`: the promoted (calibrated) model
- `mlruns/`: MLflow tracking data (gitignored)

Run it again with `--timestamp local2 --seed 2` to see champion/challenger at
work: `local2` is promoted only if its F1 is at least the champion's. If it
is promoted, `model_local1.joblib` is deleted. To see the quality gate fail,
pass `--min-f1 0.99`. The script prints `Quality gate failed…` and exits with
code 1.

To see the MLflow runs, `pip install mlflow==2.22.0` (the full package
includes the UI) and run `mlflow ui --backend-store-uri ./mlruns`.

To start over locally, delete `artifacts/`, `models/`, `metrics/` and
`mlruns/`.

## 4. Running it in GitHub Actions

The workflow is
[`.github/workflows/lab2_model_pipeline.yml`](../.github/workflows/lab2_model_pipeline.yml).

1. Fork or push this repo to your GitHub account.
2. **Settings → Actions → General → Workflow permissions**: the workflow asks
   for `contents: write` itself. If your account or org forces read-only
   tokens, choose "Read and write permissions" here.
3. Start a run in one of these ways:
   - push a change under `Lab2/src`, `Lab2/test` or `Lab2/requirements.txt` to `main`
   - **Actions → "Lab2 - Train, Calibrate & Promote Model" → Run workflow**, where you can set
     `n_samples`, `min_f1` and `calibration_method`
   - wait for the weekly schedule (Mondays 06:00 UTC)
   - open a pull request (this runs only the tests, no training)

A successful run:

- has green **Unit tests (Python 3.11)**, **Unit tests (Python 3.13)** and
  **Train, calibrate, evaluate** jobs;
- shows a results table (raw vs calibrated metrics, gate result, previous
  champion, promoted yes/no) on the run's **Summary** page;
- has downloadable artifacts `lab2-run-<timestamp>` (models, metrics, mlruns)
  and `lab2-test-report-py*` (JUnit XML);
- adds a commit by `github-actions[bot]`, titled
  `Lab2: model <timestamp> (promoted=true|false) [skip ci]`, containing
  `Lab2/metrics/<timestamp>_metrics.json` and, if the model was promoted, the
  new `Lab2/models/model_<timestamp>.joblib` and `champion.json`.

---

## What I changed

### Pipeline (`.github/workflows/lab2_model_pipeline.yml`)

The original lab had two workflows: one that retrains on push, and a
"calibration" workflow that retrains on a cron schedule but never calibrates
anything. I replaced them with one workflow:

| Change | What it does |
|---|---|
| **`test` job before training, Python 3.11 + 3.13 matrix** | Training only starts (`needs: test`) once the tests pass on both Python versions. The JUnit reports are uploaded as artifacts. |
| **Separate train → calibrate → evaluate steps** | Each stage is its own script and CI step, so you can see which stage failed. |
| **Quality gate** (`min_f1`) | The evaluate step exits non-zero if hold-out F1 is below the threshold. The job fails and nothing is committed. |
| **Champion/challenger promotion** | A new model goes into `models/` only if it beats or ties `metrics/champion.json`. The old champion's file is deleted, so the repo keeps one model instead of growing on every push. Metrics are committed for every run. |
| **Job outputs + step summary** | `evaluate_model.py` writes `promoted=true/false` to `$GITHUB_OUTPUT`, and the commit step uses it to decide whether to commit `models/`. A metrics table is written to `$GITHUB_STEP_SUMMARY`. |
| **Triggers** | `push` and `pull_request` with `paths:` filters (only Lab 2 changes trigger it), a weekly `schedule` that retrains on a fresh sample, and `workflow_dispatch` with typed inputs (`n_samples`, `min_f1`, `calibration_method` as a dropdown). PRs run tests only. |
| **Safe commits back to the repo** | Explicit `permissions: contents: write`. Commits as `github-actions[bot]` instead of the original author's hard-coded identity. Adds `[skip ci]`. Runs `git pull --rebase` before pushing. A `concurrency` group stops two runs from racing on `champion.json`. |
| **Run artifacts** | Each run's models, metrics and MLflow data are uploaded with `actions/upload-artifact` (kept for 14 days). |
| **Faster, current actions** | `actions/checkout@v4`, `setup-python@v5` with pip caching, instead of `@v2`. |

### Scripts (`src/`)

| Change | Why |
|---|---|
| `data.py`: one fixed data pool, with a **fixed 2,000-row hold-out set** | The original evaluated the model on the data it was trained on, using `random.randint(0, 2000)` rows (which can be 0). Now every run is scored on the same unseen data, so F1 can be compared across runs, which champion/challenger needs. |
| Training sample chosen by seed (`github.run_number`) | Each CI run trains on a different sample, as if new data had arrived, and any run can be reproduced from its seed. |
| `calibrate_model.py`: **real calibration** | Wraps the trained forest in `CalibratedClassifierCV(FrozenEstimator(model))` and fits a sigmoid (Platt) or isotonic calibrator on a held-back 25% split. The forest itself is not refit. Brier score drops (for example 0.100 → 0.095). |
| `evaluate_model.py` | Computes F1, accuracy, ROC-AUC and Brier for the raw and calibrated models, applies the gate, handles promotion and writes the GitHub outputs/summary. |
| `train_model.py` | Logs to one MLflow experiment (`github-lab2`) instead of creating a new experiment per run. Saves compressed models to `artifacts/`. |
| `test/test_pipeline.py` | New pytest suite (8 tests) that runs in CI before training. |
| `requirements.txt` | Versions pinned so local runs and CI behave the same. `mlflow-skinny` replaces full `mlflow` for a faster install. Unused `pandas`/`ipykernel` removed. |

## Folder layout

```
Lab2/
├── src/
│   ├── data.py              # dataset pool, fixed hold-out, seeded training sample
│   ├── train_model.py       # stage 1: train random forest, log to MLflow
│   ├── calibrate_model.py   # stage 2: sigmoid/isotonic probability calibration
│   └── evaluate_model.py    # stage 3: metrics, quality gate, champion/challenger
├── test/test_pipeline.py    # pytest suite run in CI
├── metrics/                 # committed by CI: <timestamp>_metrics.json, champion.json
├── models/                  # committed by CI: current champion model only
├── pytest.ini
├── requirements.txt
└── README.md
```
