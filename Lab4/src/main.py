"""FastAPI app that serves the wine classifier.

Run (from the Lab4 folder, after `python -m src.train`):
    uvicorn src.main:app --reload
Then open http://127.0.0.1:8000/docs
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from src.data import CLASS_NAMES, FEATURES
from src.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictionResponse,
    WineFeatures,
)
from src.train import METADATA_FILE, MODEL_DIR, MODEL_FILE

# Filled in once at startup, so requests don't reload the model from disk
state = {"model": None, "metadata": None}


def load_model(model_dir):
    if not (model_dir / MODEL_FILE).exists():
        return None, None
    return joblib.load(model_dir / MODEL_FILE), json.loads((model_dir / METADATA_FILE).read_text())


@asynccontextmanager
async def lifespan(app):
    model_dir = Path(os.environ.get("MODEL_DIR", MODEL_DIR))
    state["model"], state["metadata"] = load_model(model_dir)
    yield
    state.update(model=None, metadata=None)


app = FastAPI(
    title="Wine Cultivar Classifier",
    description="Predicts which of three cultivars a wine comes from, using 13 chemical measurements.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


def require_model():
    if state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Train it with `python -m src.train` and restart the server.",
        )
    return state["model"]


def predict_rows(wines):
    model = require_model()
    X = np.array([[getattr(w, f) for f in FEATURES] for w in wines])
    predictions = []
    for probs in model.predict_proba(X):
        i = int(probs.argmax())
        predictions.append(Prediction(
            cultivar=CLASS_NAMES[i],
            class_id=i,
            confidence=round(float(probs[i]), 4),
            probabilities={name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
        ))
    return predictions


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = state["model"] is not None
    return HealthResponse(
        status="ok" if loaded else "model_missing",
        model_loaded=loaded,
        model_version=state["metadata"]["version"] if loaded else None,
    )


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    require_model()
    return state["metadata"]


# Plain `def` (not `async def`): predict_proba is CPU-bound, so FastAPI runs it in a
# thread pool instead of blocking the event loop for other requests.
@app.post("/predict", response_model=PredictionResponse)
def predict(wine: WineFeatures):
    prediction = predict_rows([wine])[0]
    return PredictionResponse(**prediction.model_dump(), model_version=state["metadata"]["version"])


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    predictions = predict_rows(batch.wines)
    return BatchResponse(
        model_version=state["metadata"]["version"],
        count=len(predictions),
        predictions=predictions,
    )
