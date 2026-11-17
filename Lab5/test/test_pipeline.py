import json

import numpy as np
import pytest

from clustering import pipeline


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("run")
    pipeline.extract(path)
    return path


def test_extract_makes_stratified_split(run_dir):
    _, y_train = pipeline._load(run_dir, pipeline.TRAIN_FILE)
    _, y_hold = pipeline._load(run_dir, pipeline.HOLDOUT_FILE)
    assert len(y_train) + len(y_hold) == 1797
    # Every digit is about 10% of both splits
    assert np.allclose(np.bincount(y_train) / len(y_train), np.bincount(y_hold) / len(y_hold), atol=0.01)


def test_validate_passes_on_real_data(run_dir):
    assert pipeline.validate(run_dir)["checks_passed"] == 5


def test_validate_rejects_bad_data(tmp_path):
    X = np.full((20, 64), 20.0)  # pixels above 16
    X[0, 0] = np.nan
    y = np.zeros(20, dtype=int)  # only digit 0
    for name in (pipeline.TRAIN_FILE, pipeline.HOLDOUT_FILE):
        np.savez(tmp_path / name, X=X, y=y)
    with pytest.raises(ValueError) as err:
        pipeline.validate(tmp_path)
    message = str(err.value)
    assert "NaN" in message and "outside 0-16" in message and "missing" in message


def test_score_k_returns_inertia_and_silhouette(run_dir):
    score = pipeline.score_k(run_dir, 5)
    assert score["k"] == 5
    assert score["inertia"] > 0
    assert -1 <= score["silhouette"] <= 1


def test_choose_k_by_silhouette_and_by_elbow():
    # Inertia drops fast until k=4, then slowly: the elbow is at 4. Silhouette peaks at 6.
    scores = [
        {"k": 2, "inertia": 1000, "silhouette": 0.10},
        {"k": 3, "inertia": 500, "silhouette": 0.15},
        {"k": 4, "inertia": 200, "silhouette": 0.20},
        {"k": 5, "inertia": 180, "silhouette": 0.22},
        {"k": 6, "inertia": 165, "silhouette": 0.25},
        {"k": 7, "inertia": 155, "silhouette": 0.21},
    ]
    assert pipeline.choose_k(scores)["chosen_k"] == 6
    by_elbow = pipeline.choose_k(scores, select_by="elbow")
    assert by_elbow["chosen_k"] == by_elbow["elbow_k"] == 4


def test_train_evaluate_report(run_dir):
    info = pipeline.train(run_dir, 12)
    assert info["k"] == 12
    assert 0 < info["pca_components"] < 64
    assert set(info["cluster_digit"]) == set(range(12))

    metrics = pipeline.evaluate(run_dir, min_accuracy=0.5)
    assert metrics["passed_gate"] is True
    assert metrics["mapped_accuracy"] >= 0.5
    assert json.loads((run_dir / pipeline.METRICS_FILE).read_text()) == metrics

    choice = {"chosen_k": 12, "select_by": "silhouette", "elbow_k": 9, "best_silhouette_k": 12,
              "scores": [{"k": 12, "inertia": 1.0, "silhouette": 0.2}]}
    report = (run_dir / "report.md")
    assert pipeline.report(run_dir, choice, metrics) == str(report)
    text = report.read_text()
    assert "Chosen k = **12**" in text
    assert text.count("\n| ") >= 12  # one table row per cluster at least


def test_quality_gate_raises(run_dir):
    pipeline.train(run_dir, 3)
    with pytest.raises(ValueError, match="Quality gate failed"):
        pipeline.evaluate(run_dir, min_accuracy=0.99)
    assert json.loads((run_dir / pipeline.METRICS_FILE).read_text())["passed_gate"] is False
