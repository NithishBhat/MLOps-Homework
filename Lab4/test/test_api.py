import json

from sklearn.datasets import load_wine

from src.data import CLASS_NAMES, FEATURES
from src.schemas import EXAMPLE_WINE


def wine_row(index):
    wine = load_wine()
    return dict(zip(FEATURES, wine.data[index].tolist())), CLASS_NAMES[wine.target[index]]


def test_features_match_dataset_columns():
    assert len(FEATURES) == load_wine().data.shape[1]
    assert set(EXAMPLE_WINE) == set(FEATURES)


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_predict_example_wine(client):
    response = client.post("/predict", json=EXAMPLE_WINE)
    assert response.status_code == 200
    body = response.json()
    assert body["cultivar"] == "cultivar_1"
    assert body["class_id"] == 0
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["confidence"] == max(body["probabilities"].values())
    assert "x-process-time-ms" in response.headers


def test_predict_one_wine_of_each_cultivar(client):
    # Rows 0, 100 and 170 belong to cultivars 1, 2 and 3
    for index in (0, 100, 170):
        payload, expected = wine_row(index)
        assert client.post("/predict", json=payload).json()["cultivar"] == expected


def test_missing_field_is_rejected(client):
    payload = {k: v for k, v in EXAMPLE_WINE.items() if k != "proline"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "proline"]


def test_negative_value_is_rejected(client):
    response = client.post("/predict", json={**EXAMPLE_WINE, "alcohol": -1})
    assert response.status_code == 422


def test_unknown_field_is_rejected(client):
    response = client.post("/predict", json={**EXAMPLE_WINE, "alchol": 13.0})
    assert response.status_code == 422


def test_non_numeric_value_is_rejected(client):
    response = client.post("/predict", json={**EXAMPLE_WINE, "hue": "dark"})
    assert response.status_code == 422


def test_batch_predict(client):
    wines = [wine_row(i)[0] for i in (0, 100, 170)]
    body = client.post("/predict/batch", json={"wines": wines}).json()
    assert body["count"] == 3
    assert [p["cultivar"] for p in body["predictions"]] == CLASS_NAMES


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"wines": []}).status_code == 422
    assert client.post("/predict/batch", json={"wines": [EXAMPLE_WINE] * 101}).status_code == 422


def test_model_info_matches_metadata_file(client, model_dir):
    body = client.get("/model/info").json()
    saved = json.loads((model_dir / "metadata.json").read_text())
    assert body["version"] == saved["version"]
    assert body["features"] == FEATURES
    assert body["metrics"]["accuracy"] >= 0.9
    assert list(body["feature_importance"].values()) == sorted(body["feature_importance"].values(), reverse=True)


def test_without_model_returns_503(client_without_model):
    assert client_without_model.get("/health").json()["model_loaded"] is False
    assert client_without_model.post("/predict", json=EXAMPLE_WINE).status_code == 503
    assert client_without_model.get("/model/info").status_code == 503
