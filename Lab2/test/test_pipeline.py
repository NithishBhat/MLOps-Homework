import json

import numpy as np
import pytest

from src import calibrate_model, data, evaluate_model, train_model


def test_holdout_is_fixed_and_disjoint_size():
    X1, y1 = data.load_holdout()
    X2, y2 = data.load_holdout()
    assert X1.shape == (data.HOLDOUT_SIZE, data.N_FEATURES)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_training_data_is_reproducible_per_seed():
    a = data.load_training_data(500, seed=1)
    b = data.load_training_data(500, seed=1)
    c = data.load_training_data(500, seed=2)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])
    X_train, X_cal, _, _ = a
    assert len(X_train) + len(X_cal) == 500


@pytest.mark.parametrize("n", [0, data.MIN_SAMPLES - 1, data.POOL_SIZE])
def test_training_data_rejects_bad_sizes(n):
    with pytest.raises(ValueError):
        data.load_training_data(n, seed=0)


def test_gate_and_promotion_rules():
    assert evaluate_model.passes_gate({"f1": 0.85}, 0.80)
    assert not evaluate_model.passes_gate({"f1": 0.79}, 0.80)
    assert evaluate_model.should_promote({"f1": 0.80}, None)
    assert evaluate_model.should_promote({"f1": 0.85}, {"f1": 0.85})
    assert not evaluate_model.should_promote({"f1": 0.84}, {"f1": 0.85})


@pytest.fixture
def lab_dirs(tmp_path, monkeypatch):
    """Point every stage at a temp folder so tests never touch the real repo."""
    dirs = {name: tmp_path / name for name in ("artifacts", "models", "metrics")}
    for module in (train_model, calibrate_model, evaluate_model):
        for attr, name in (("ARTIFACTS_DIR", "artifacts"), ("MODELS_DIR", "models"), ("METRICS_DIR", "metrics")):
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, dirs[name])
    monkeypatch.setattr(train_model, "LAB_DIR", tmp_path)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    return dirs


def test_full_pipeline_promotes_then_keeps_champion(lab_dirs):
    for ts, seed in (("001", 1), ("002", 2)):
        train_model.main(["--timestamp", ts, "--n-samples", "800", "--seed", str(seed)])
        calibrate_model.main(["--timestamp", ts])
        evaluate_model.main(["--timestamp", ts, "--min-f1", "0.5"])

    first = json.loads((lab_dirs["metrics"] / "001_metrics.json").read_text())
    second = json.loads((lab_dirs["metrics"] / "002_metrics.json").read_text())
    champion = json.loads((lab_dirs["metrics"] / "champion.json").read_text())

    assert first["promoted"] is True
    assert second["previous_champion"] == "001"
    # Exactly one model is kept in models/: whichever run is champion.
    assert [p.name for p in lab_dirs["models"].iterdir()] == [f"model_{champion['timestamp']}.joblib"]
    assert champion["f1"] == max(first["calibrated"]["f1"], second["calibrated"]["f1"])


def test_quality_gate_failure_exits_and_promotes_nothing(lab_dirs):
    train_model.main(["--timestamp", "003", "--n-samples", "400", "--seed", "3"])
    calibrate_model.main(["--timestamp", "003"])
    with pytest.raises(SystemExit):
        evaluate_model.main(["--timestamp", "003", "--min-f1", "0.999"])
    assert not (lab_dirs["metrics"] / "champion.json").exists()
    assert not lab_dirs["models"].exists()
