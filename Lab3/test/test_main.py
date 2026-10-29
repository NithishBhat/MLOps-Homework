import json

import main
import predict


def test_train_one_model_writes_model_and_metrics(tmp_path):
    assert main.main(["--model", "logistic_regression", "--output-dir", str(tmp_path)]) == 0

    report = json.loads((tmp_path / "metrics.json").read_text())
    assert report["best_model"] == "logistic_regression"
    assert report["n_train"] + report["n_test"] == 569
    assert set(report["results"]["logistic_regression"]) == {"cv_accuracy", "accuracy", "f1", "roc_auc"}
    assert (tmp_path / "models" / "logistic_regression.joblib").exists()


def test_train_all_picks_best_by_f1(tmp_path):
    assert main.main(["--output-dir", str(tmp_path)]) == 0

    report = json.loads((tmp_path / "metrics.json").read_text())
    assert set(report["results"]) == set(main.MODEL_NAMES)
    best_f1 = max(r["f1"] for r in report["results"].values())
    assert report["results"][report["best_model"]]["f1"] == best_f1
    assert report["passed_gate"] is True


def test_quality_gate_fails_with_exit_code_1(tmp_path):
    code = main.main(["--model", "random_forest", "--min-accuracy", "1.01", "--output-dir", str(tmp_path)])

    assert code == 1
    assert json.loads((tmp_path / "metrics.json").read_text())["passed_gate"] is False


def test_split_is_reproducible():
    a = main.load_split(0.2, seed=3)
    b = main.load_split(0.2, seed=3)
    assert all((x == y).all() for x, y in zip(a, b))


def test_predict_uses_best_model(tmp_path):
    main.main(["--model", "gradient_boosting", "--output-dir", str(tmp_path)])

    model_name, preds = predict.predict(tmp_path, n=4)
    assert model_name == "gradient_boosting"
    assert len(preds) == 4
    assert all(p["predicted"] in {"malignant", "benign"} for p in preds)
