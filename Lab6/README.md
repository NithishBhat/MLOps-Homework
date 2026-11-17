# Lab 6: A Streamlit Dashboard for the Wine Classifier API

Based on [API_Labs/Streamlit_Labs](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/Streamlit_Labs)
from the course repo. The original lab builds a one-page Streamlit dashboard
for the FastAPI lab's Iris model: four sliders (or an uploaded JSON file), a
Predict button, and a line saying which flower was predicted. I built the
dashboard for **my own FastAPI service from [Lab 4](../Lab4/)** instead (a
wine classifier with 13 measurements) and added more pages, batch prediction,
charts, error handling, tests and Docker. The full list is in
[What I changed](#what-i-changed).

Lab 4 is used as-is: `docker-compose.yaml` builds it from `../Lab4`, and nothing
in that folder was changed for this lab.

### Summary of changes

- **Different model and data**: the Lab4 wine API (3 cultivars, 13 chemical measurements) instead of Iris.
- **Three pages**:
  - **Dashboard**: 13 sliders with presets, or a JSON upload. Shows the cultivar, confidence, runner-up and a probability chart.
  - **Batch Predictions**: upload a CSV/JSON of many wines. Shows a results table, a summary chart, low-confidence warnings and a CSV download.
  - **Model Info**: metrics and a sorted feature-importance chart from the API's `/model/info`.
- **Backend status on every page** using the API's `/health` (online, and which model version is loaded).
- **Readable errors**: validation errors from the API are listed by field, and connection problems and timeouts get clear messages.
- **API URL from an environment variable** instead of a hard-coded `localhost`.
- **Tests**: 22 unit/UI tests with a fake backend (Streamlit's `AppTest`), plus 5 end-to-end tests against the real API.
- **Docker Compose** starts the API and the dashboard together. CI runs everything.

---

## 1. What you need installed

| Tool | Version | Notes |
|---|---|---|
| Python | **3.10 – 3.13** (CI uses 3.12) | <https://www.python.org/downloads/> |
| Git | any recent (2.30+) | <https://git-scm.com/downloads> |
| Docker | Docker Desktop 4.x or Docker Engine 24+ | For [option A](#option-a-docker-compose-api--dashboard-together) and the end-to-end tests |

Python packages (pinned in [`requirements.txt`](requirements.txt)):

| Package | Version | Used for |
|---|---|---|
| streamlit | 1.67.0 | the dashboard |
| requests | 2.34.2 | calling the API |
| pandas | 3.1.1 | tables for batch predictions |
| altair | 6.3.0 | the sorted feature-importance chart |
| pytest | 8.3.5 | tests (`requirements-dev.txt` only) |

## 2. Run it

### Option A: Docker Compose (API + dashboard together)

```bash
git clone https://github.com/NithishBhat/MLOps-Homework.git
cd MLOps-Homework/Lab6

docker compose up -d --build --wait
```

`--wait` returns once both containers are healthy. The dashboard waits for the
API's health check before it starts. Then open:

- Dashboard: <http://localhost:8501>
- API docs (Lab 4): <http://localhost:8000/docs>

Stop both with `docker compose down`.

### Option B: without Docker

Start the Lab 4 API first (see [Lab 4's README](../Lab4/README.md), sections 2–3):

```bash
cd MLOps-Homework/Lab4
# (with Lab 4's virtual environment active)
python -m src.train
uvicorn src.main:app
```

Then, in a second terminal:

```bash
cd MLOps-Homework/Lab6
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements-dev.txt

cd dashboard
streamlit run Dashboard.py
```

Streamlit opens <http://localhost:8501>. The dashboard looks for the API at
`http://localhost:8000`. To use another address, set `API_URL` first, for
example `export API_URL=http://my-server:8000` (PowerShell:
`$env:API_URL = "http://my-server:8000"`). `API_TIMEOUT` (seconds, default 10)
sets how long it waits for an answer.

## 3. Using the dashboard

**Sidebar (every page)**: shows the API address and a green *Backend online,
model version …* box, a yellow box if the API is up but has no model, or a red
*Backend offline* box with the reason.

**Dashboard** (single wine):
1. Pick **Sliders** or **Upload JSON**.
2. With sliders, choose a starting point under **Start from**: the dataset
   median, or a real wine of each cultivar. Each slider covers the dataset's
   range, and its **?** shows the range and median.
3. Click **Predict**. The sliders are in a form, so moving them doesn't call
   the API until you click.
4. The result shows the predicted cultivar, the confidence, the runner-up and
   a chart of all three probabilities. **Request sent to the API** shows the
   exact JSON that was sent.

JSON upload accepts the 13 measurements as a plain object, or the original
lab's `{"input_test": {...}}` format. Try [`dashboard/sample_wine.json`](dashboard/sample_wine.json)
(a cultivar 3 wine). Fields the model doesn't use are ignored, with a warning.

**Batch Predictions**: upload a CSV or JSON (a list of wines, or
`{"wines": [...]}`). **Download a sample CSV** gives a 5-wine file with a
`wine_id` column. Extra columns like that are kept, and are not sent to the
model. Click **Predict all** to get:
- a chart of how many wines went to each cultivar;
- a warning if any prediction has confidence below 60%;
- a table with your columns, the prediction, the confidence and every class
  probability, then the measurements;
- **Download results CSV**.

Files with more than 100 wines are sent in chunks of 100, the API's batch limit.
Missing columns, or empty or non-numeric cells, are reported by row before
anything is sent.

**Model Info**: CV accuracy, test accuracy and macro F1. Also the model type,
version, training date and train/test sizes, a feature-importance chart sorted
from most to least important, and the raw `/model/info` response.

## 4. Run the tests

```bash
pytest
```

```
......................sssss                                              [100%]
22 passed, 5 skipped
```

- `test_api_client.py` (7): the client sends the right requests with a
  timeout, turns 422 bodies into `field: message` lines, and reports 503,
  non-JSON errors, connection errors and timeouts clearly.
- `test_inputs.py` (8): presets cover all 13 features; JSON (both formats)
  and CSV parsing; missing and extra columns; empty or text cells; chunking
  into 100s; the results table's column order.
- `test_dashboard.py` (7): runs the real pages with Streamlit's
  [`AppTest`](https://docs.streamlit.io/develop/api-reference/app-testing)
  against a fake backend. Covers online and offline sidebar states, sliders
  starting at the median, a preset moving the sliders and Predict sending
  exactly those values, no request when the backend is down, and the Model Info
  and Batch pages loading.
- `test_live.py` (5, skipped by default): the same flows against the **real**
  Lab 4 API. One wine of each cultivar is predicted correctly through the
  dashboard, batch predictions work, and validation errors are readable. Run
  them with the containers up:

```bash
docker compose up -d --build --wait
LAB6_LIVE_API=1 pytest test/test_live.py          # PowerShell: $env:LAB6_LIVE_API=1; pytest test/test_live.py
```

## 5. Running it in GitHub Actions

The workflow is [`.github/workflows/lab6_streamlit.yml`](../.github/workflows/lab6_streamlit.yml).
It runs on pushes and pull requests that change `Lab6/`, and can be started
from **Actions → "Lab6 - Streamlit Dashboard" → Run workflow**.

A successful run:

- has a green **Dashboard tests (fake backend)** job: all unit and AppTest tests;
- has a green **Dashboard + Lab4 API in Docker** job: builds both images,
  waits for both health checks, then runs the 5 end-to-end tests against the
  real API;
- prints both containers' logs at the end.

---

## What I changed

### Model and backend

| Change | Why |
|---|---|
| Iris → **Wine** (my Lab 4 API) | 13 measurements and probabilities to show, instead of 4 sliders and a class number. |
| Health from **`/health`** | The original called `/` and also checked that the FastAPI lab's model **file** existed on the dashboard's own disk, which breaks as soon as the API runs anywhere else. Now the API itself reports whether a model is loaded, and which version. |
| **`API_URL`** / **`API_TIMEOUT`** environment variables | The original hard-coded `http://localhost:8000`, which doesn't work inside Docker. |

### Dashboard

| Change | Why |
|---|---|
| **Three pages** (`pages/1_Batch_Predictions.py`, `pages/2_Model_Info.py`) | The original README describes multi-page apps but stops at one page. |
| **Presets** + sliders in a **form** | Starting from a real wine of each cultivar makes it easy to try the model. With a form, the API is only called when you click Predict, not on every slider move. |
| Slider ranges from the dataset, with help text | Every slider covers the real range of that measurement. |
| **Probabilities chart, confidence, runner-up** | The original showed only the predicted name. |
| **Batch predictions** with CSV/JSON upload, chunking, a results download and low-confidence warnings | The original's uploader took one wine at a time. |
| **Model Info page** | Shows how good the served model is and which measurements it relies on. |
| JSON upload accepts the original `{"input_test": ...}` format | Old test files still work. Unknown fields are reported instead of causing errors. |
| **Field-level error messages** | A 422 from the API lists each bad field. Offline and timeout errors say what to check. |
| Fixed the status message | The original's `st.success("Backend online ✅" if ok else st.warning(...))` shows a yellow warning *and* a green success box at the same time when the API answers with an error. |
| `.streamlit/config.toml` | Turns off Streamlit's usage statistics, limits upload size and sets the theme colour. |

### Tests, Docker and CI

| Change | Why |
|---|---|
| 22 tests + 5 live tests | The original had none. `AppTest` runs the real page scripts, so the UI logic is tested too, not just helper functions. |
| `Dockerfile` (non-root, health check) + `docker-compose.yaml` | One command starts the API and the dashboard, in the right order. |
| `.github/workflows/lab6_streamlit.yml` | Runs the tests, then both containers and the end-to-end tests, on every change to `Lab6/`. |
| Pinned requirements | The original installed `fastapi[all]` and scikit-learn in the dashboard environment, although the dashboard only needs `streamlit` and `requests`. |

## Folder layout

```
Lab6/
├── dashboard/
│   ├── Dashboard.py               # main page: one wine, sliders/presets or JSON upload
│   ├── pages/
│   │   ├── 1_Batch_Predictions.py # CSV/JSON upload, results table and download
│   │   └── 2_Model_Info.py        # metrics and feature importance from /model/info
│   ├── api.py                     # client for the Lab4 API (errors -> ApiError)
│   ├── inputs.py                  # file parsing, validation, chunking, results table
│   ├── sidebar.py                 # backend status box shared by all pages
│   ├── wine_features.py           # the 13 features, slider ranges and presets
│   ├── sample_wines.csv           # example batch file (also downloadable in the app)
│   ├── sample_wine.json           # example single wine in the original lab's format
│   └── .streamlit/config.toml
├── test/
│   ├── conftest.py                # fake backend
│   ├── test_api_client.py
│   ├── test_inputs.py
│   ├── test_dashboard.py          # AppTest UI tests
│   └── test_live.py               # end-to-end, needs the API running
├── Dockerfile
├── docker-compose.yaml            # Lab4 API (built from ../Lab4) + this dashboard
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```
