# Lab 4: Serving a Model as an API with FastAPI

Based on [API_Labs/FastAPI_Labs](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs)
from the course repo. The original lab trains a decision tree on the Iris
dataset and serves it with FastAPI and uvicorn through a single `/predict`
endpoint that returns a class number. I kept FastAPI + uvicorn but changed the
dataset, the model, the API and how it is run and tested. The full list of
changes is in [What I changed](#what-i-changed).

### Summary of changes

**Model** (`src/data.py`, `src/train.py`)
- Dataset changed from Iris to **Wine** (178 wines, 13 chemical measurements, 3 cultivars).
- Random forest instead of a depth-3 decision tree, trained on a stratified split.
- Training saves a `metadata.json` next to the model with a version, metrics (5-fold CV accuracy, test accuracy, macro F1) and feature importances.

**API** (`src/main.py`, `src/schemas.py`)
- The model is loaded **once at startup** instead of from disk on every request.
- `/predict` returns the cultivar name, the confidence, the probability of every class and the model version, instead of a bare class number.
- New endpoints: `/predict/batch` (1–100 wines per request), `/model/info` (metrics and feature importances) and `/health`, which shows whether a model is loaded.
- Stricter validation: every value must be a positive number, and unknown fields (typos) are rejected with 422.
- Returns 503 with a clear message if no model has been trained, and adds an `X-Process-Time-Ms` header to every response.

**Tests, Docker and CI**
- 13 pytest tests using FastAPI's `TestClient` (the original had none).
- `Dockerfile` that trains the model at build time and serves it with a health check.
- GitHub Actions workflow that runs the tests, builds the image, starts the API and calls every endpoint.

---

## 1. What you need installed

| Tool | Version | Notes |
|---|---|---|
| Python | **3.10 – 3.13** (CI uses 3.11) | <https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"** in the installer. |
| Git | any recent (2.30+) | <https://git-scm.com/downloads> |
| Docker | optional | Only for [section 5](#5-run-it-in-docker) |

Python packages (pinned in [`requirements.txt`](requirements.txt) and [`requirements-dev.txt`](requirements-dev.txt)):

| Package | Version | Used for |
|---|---|---|
| fastapi | 0.162.0 | the API, request validation, `/docs` |
| uvicorn | 0.54.0 | ASGI server that runs the app |
| numpy | 2.2.6 | arrays |
| scipy | 1.15.3 | needed by scikit-learn (pinned so it matches scikit-learn 1.6) |
| scikit-learn | 1.6.1 | dataset, random forest, metrics |
| joblib | 1.4.2 | saving the model |
| pytest | 8.3.5 | tests (dev only) |
| httpx2 | 2.16.0 | HTTP client used by FastAPI's `TestClient` (dev only) |

## 2. Setup

```bash
git clone https://github.com/NithishBhat/MLOps-Homework.git
cd MLOps-Homework/Lab4

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements-dev.txt
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use cmd.

**Run every command below from inside the `Lab4/` folder** with the virtual
environment active. The scripts are run as modules (`python -m src.…`,
`src.main:app`) so that `src` imports work from any working directory.

## 3. Train and serve

### Train the model

```bash
python -m src.train
```

```
Trained RandomForestClassifier version 20261105042314 on 133 rows
5-fold CV accuracy: 0.9698
Test set (45 rows): accuracy=1.0, f1_macro=1.0
Saved to .../Lab4/model
```

This writes `model/wine_model.joblib` and `model/metadata.json` (both
gitignored). The Wine dataset is easy to separate, so the 45-row test set is
often classified perfectly. The CV accuracy (about 0.97) is the more realistic
estimate. Options: `--n-estimators 300`, `--seed 7`, `--model-dir <path>`.

### Start the server

```bash
uvicorn src.main:app --reload
```

`--reload` restarts the server when you edit the code (for development only).
Open <http://127.0.0.1:8000/docs> for the interactive Swagger page:
expand an endpoint → **Try it out** → **Execute**. The example request body is
already filled in. `http://127.0.0.1:8000/` redirects there too.

If you start the server before training, `/health` reports
`"model_loaded": false` and the prediction endpoints return 503 until you train
and restart.

## 4. Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | `{"status": "ok", "model_loaded": true, "model_version": "..."}` |
| GET | `/model/info` | Model type, version, training date, metrics, feature list and feature importances |
| POST | `/predict` | Predicts the cultivar of one wine |
| POST | `/predict/batch` | Predicts 1–100 wines in one request: `{"wines": [ {...}, {...} ]}` |
| GET | `/docs`, `/redoc` | Generated API documentation |

### Example: predict one wine

macOS / Linux / Git Bash:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
       "magnesium": 127, "total_phenols": 2.8, "flavanoids": 3.06, "nonflavanoid_phenols": 0.28,
       "proanthocyanins": 2.29, "color_intensity": 5.64, "hue": 1.04, "od280_od315": 3.92,
       "proline": 1065}'
```

Windows PowerShell:

```powershell
$wine = @{ alcohol=13.4; malic_acid=3.91; ash=2.48; alcalinity_of_ash=23; magnesium=102;
           total_phenols=1.8; flavanoids=0.75; nonflavanoid_phenols=0.43; proanthocyanins=1.41;
           color_intensity=7.3; hue=0.7; od280_od315=1.56; proline=750 }
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType 'application/json' -Body ($wine | ConvertTo-Json)
```

Response (for the curl example):

```json
{
  "cultivar": "cultivar_1",
  "class_id": 0,
  "confidence": 1.0,
  "probabilities": { "cultivar_1": 1.0, "cultivar_2": 0.0, "cultivar_3": 0.0 },
  "model_version": "20261105042314"
}
```

The PowerShell example is a `cultivar_3` wine. Every response also has an
`X-Process-Time-Ms` header with the time the server took.

### Validation errors

Bad input is rejected with **422 Unprocessable Entity** and a message saying
which field is wrong. For example:

| Request body | Why it's rejected |
|---|---|
| `proline` missing | every one of the 13 fields is required |
| `"alcohol": -1` | all values must be greater than 0 |
| `"hue": "dark"` | values must be numbers |
| `"alchol": 13.0` (typo) | unknown fields are not allowed |
| `{"wines": []}` on `/predict/batch` | a batch needs 1–100 wines |

## 5. Run it in Docker

The image trains the model during the build, so it can serve as soon as it
starts:

```bash
docker build -t lab4:v1 .
docker run -d --name lab4-api -p 8000:8000 lab4:v1
```

Open <http://localhost:8000/docs>, or call the endpoints as above. After about
10 seconds, `docker ps` shows the container as `(healthy)`: the image's
`HEALTHCHECK` calls `/health` every 30 seconds. To stop it:

```bash
docker rm -f lab4-api
```

## 6. Run the tests

```bash
pytest
```

```
.............                                                            [100%]
13 passed in 2.44s
```

The tests train a small model into a temp folder (not `model/`) and call the
API through FastAPI's `TestClient`. They cover: `/health` and the `/` redirect;
one wine of each cultivar is predicted correctly; probabilities add up to 1;
missing, negative, non-numeric and unknown fields are rejected with 422; batch
predictions and the 1–100 batch limit; `/model/info` matches the saved
metadata; and 503 responses when no model has been trained.

## 7. Running it in GitHub Actions

The workflow is [`.github/workflows/lab4_fastapi.yml`](../.github/workflows/lab4_fastapi.yml).
It runs on pushes and pull requests that change `Lab4/`, and can be started
manually from **Actions → "Lab4 - Test & Serve FastAPI Model" → Run workflow**.

A successful run:

- has green **API tests** and **Build image and smoke-test the API** jobs (the image is only built if the tests pass);
- in the second job, starts the container and checks with `curl` that `/health`
  answers, `/predict` returns `cultivar_1` for the example wine, `/predict/batch`
  returns 2 predictions, an invalid body gets 422 and `/docs` loads;
- shows the model's metrics and top features on the run's **Summary** page;
- prints the container's request log at the end.

---

## What I changed

### Model (`src/data.py`, `src/train.py`)

| Change | Why |
|---|---|
| Iris → **Wine** dataset | 13 features and 3 classes instead of 4 features, so the request body and validation are more realistic. |
| Depth-3 decision tree → **random forest** (200 trees) | Gives better-calibrated probabilities, which the API now returns. |
| **Stratified** split | The original split was not stratified. With only 178 rows, stratifying keeps all three cultivars in the test set in the same proportions. |
| `metadata.json` saved next to the model | The original saved only the model. Now the version, training date, metrics, feature names and importances travel with it, and `/model/info` serves them. |
| Paths built from the file's location | The original used `"../model/iris_model.pkl"`, which only works when you start from inside `src/`. |

### API (`src/main.py`, `src/schemas.py`)

| Change | Why |
|---|---|
| **Model loaded once at startup** (`lifespan`) | The original called `joblib.load` inside `predict_data`, so it read the model from disk on every request. |
| `def` instead of `async def` for prediction routes | `predict_proba` is CPU-bound. In an `async def` it would block the event loop. FastAPI runs plain `def` routes in a thread pool. |
| Richer response | The cultivar name, the confidence, the probability of every class and the model version, instead of just `{"response": 0}`. |
| `/predict/batch` | Predicting many wines in one request is faster than one request per wine. Limited to 100 per request. |
| `/model/info`, `/health` | Callers can check which model version is live and how good it is. `/health` reports whether a model is loaded and is used by Docker's health check. |
| Validation: `gt=0`, `extra="forbid"` | The original accepted any float, including negatives, and silently ignored misspelled fields. |
| 503 when no model is trained | The original caught every error and returned 500 with the raw exception text. |
| Example request in the schema | `/docs` opens with a real wine already filled in. |
| `X-Process-Time-Ms` header (middleware) | Shows how long each request took, without opening the server logs. |
| `/` redirects to `/docs` | The docs are the easiest way to try the API. |

The original README also said to run `uvicorn app:main --reload`, which fails
(the module is `main` and the app object is `app`). Here it is
`uvicorn src.main:app --reload`, run from `Lab4/`.

### Tests, Docker and CI

| Change | Why |
|---|---|
| `test/test_api.py` (13 tests) | The original had no tests. These call the API through `TestClient`, the same way a real client would. |
| `Dockerfile` | Trains at build time, runs as a non-root user, has a `HEALTHCHECK`, and runs uvicorn without `--reload`. |
| `.github/workflows/lab4_fastapi.yml` | Runs the tests, then builds the image, starts it and checks the endpoints with `curl`. |
| Pinned requirements, dev split | `fastapi[all]` installed many optional extras. Now only what's used is installed, and `pytest`/`httpx2` are kept out of the image. |

## Folder layout

```
Lab4/
├── src/
│   ├── data.py          # Wine dataset, feature names, stratified split
│   ├── train.py         # train the random forest, save model + metadata.json
│   ├── schemas.py       # Pydantic request/response models and validation
│   └── main.py          # FastAPI app: /health, /model/info, /predict, /predict/batch
├── test/
│   ├── conftest.py      # trains a model in a temp folder, starts TestClient
│   └── test_api.py      # 13 API tests
├── model/               # created by training (gitignored)
├── Dockerfile
├── .dockerignore
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── README.md
```
