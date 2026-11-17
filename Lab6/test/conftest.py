from pathlib import Path

import pytest

import api

DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard"

HEALTH = {"status": "ok", "model_loaded": True, "model_version": "20261117120000"}
MODEL_INFO = {
    "version": "20261117120000",
    "trained_at": "2026-11-17T12:00:00+00:00",
    "model": "RandomForestClassifier",
    "n_estimators": 200,
    "seed": 42,
    "n_train": 133,
    "n_test": 45,
    "metrics": {"cv_accuracy": 0.9698, "accuracy": 1.0, "f1_macro": 1.0},
    "features": ["alcohol"],
    "classes": ["cultivar_1", "cultivar_2", "cultivar_3"],
    "feature_importance": {"proline": 0.2, "flavanoids": 0.18, "alcohol": 0.1},
    "sklearn_version": "1.6.1",
}


def fake_prediction(wine):
    # Stand-in for the model: high proline -> cultivar 1, low flavanoids -> cultivar 3, else cultivar 2
    if wine["proline"] > 900:
        cultivar = "cultivar_1"
    elif wine["flavanoids"] < 1:
        cultivar = "cultivar_3"
    else:
        cultivar = "cultivar_2"
    probs = {c: (0.9 if c == cultivar else 0.05) for c in ("cultivar_1", "cultivar_2", "cultivar_3")}
    return {"cultivar": cultivar, "class_id": int(cultivar[-1]) - 1, "confidence": 0.9, "probabilities": probs}


@pytest.fixture
def fake_backend(monkeypatch):
    """Replace the API client's calls with an in-memory fake. Records every wine sent."""
    sent = []

    def predict(wine):
        sent.append(wine)
        return {**fake_prediction(wine), "model_version": HEALTH["model_version"]}

    def predict_batch(wines):
        sent.extend(wines)
        return {"model_version": HEALTH["model_version"], "count": len(wines),
                "predictions": [fake_prediction(w) for w in wines]}

    monkeypatch.setattr(api, "health", lambda: HEALTH)
    monkeypatch.setattr(api, "model_info", lambda: MODEL_INFO)
    monkeypatch.setattr(api, "predict", predict)
    monkeypatch.setattr(api, "predict_batch", predict_batch)
    return sent


@pytest.fixture
def offline_backend(monkeypatch):
    def fail(*args, **kwargs):
        raise api.ApiError("Can't reach the API at http://localhost:8000. Is it running?")

    for name in ("health", "model_info", "predict", "predict_batch"):
        monkeypatch.setattr(api, name, fail)
