"""Train and compare classifiers on the breast cancer dataset.

Every model is scored on the same stratified hold-out split. The trained models
and a metrics.json report are written to OUTPUT_DIR, which is meant to be a
volume mounted from the host so the results are still there after the
container exits.

Usage (in the container, or from the Lab3/src folder):
    python main.py                                  # train and compare all models
    python main.py --model random_forest --seed 7   # train one model

Every option can also be set with an environment variable, e.g.
    docker run -e MODEL=random_forest -e SEED=7 lab3:v1
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["logistic_regression", "random_forest", "gradient_boosting"]


def build_model(name, seed):
    if name == "logistic_regression":
        # Scaling matters for logistic regression; the tree models don't need it.
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def load_split(test_size, seed):
    data = load_breast_cancer()
    return train_test_split(
        data.data, data.target, test_size=test_size, random_state=seed, stratify=data.target
    )


def evaluate(model, X_train, X_test, y_train, y_test):
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "cv_accuracy": round(float(cv.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }


def parse_args(argv):
    env = os.environ.get
    parser = argparse.ArgumentParser(description="Train and compare breast cancer classifiers.")
    parser.add_argument("--model", choices=["all", *MODEL_NAMES], default=env("MODEL", "all"))
    parser.add_argument("--test-size", type=float, default=float(env("TEST_SIZE", "0.2")))
    parser.add_argument("--seed", type=int, default=int(env("SEED", "42")))
    parser.add_argument("--min-accuracy", type=float, default=float(env("MIN_ACCURACY", "0.90")),
                        help="Quality gate: exit with code 1 if the best model scores below this")
    parser.add_argument("--output-dir", type=Path, default=Path(env("OUTPUT_DIR", "output")))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    names = MODEL_NAMES if args.model == "all" else [args.model]
    X_train, X_test, y_train, y_test = load_split(args.test_size, args.seed)

    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training on {len(y_train)} rows, testing on {len(y_test)} rows (seed={args.seed})\n")
    print(f"{'model':<22}{'cv_acc':>8}{'acc':>8}{'f1':>8}{'auc':>8}")
    results = {}
    for name in names:
        model = build_model(name, args.seed)
        r = results[name] = evaluate(model, X_train, X_test, y_train, y_test)
        joblib.dump(model, models_dir / f"{name}.joblib", compress=3)
        print(f"{name:<22}{r['cv_accuracy']:>8.4f}{r['accuracy']:>8.4f}{r['f1']:>8.4f}{r['roc_auc']:>8.4f}")

    best = max(results, key=lambda n: (results[n]["f1"], results[n]["roc_auc"]))
    passed = results[best]["accuracy"] >= args.min_accuracy
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "sklearn breast_cancer (569 rows, 30 features)",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "seed": args.seed,
        "sklearn_version": sklearn.__version__,
        "results": results,
        "best_model": best,
        "min_accuracy": args.min_accuracy,
        "passed_gate": passed,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    print(f"\nBest model: {best} (f1={results[best]['f1']}, accuracy={results[best]['accuracy']})")
    print(f"Saved models and metrics.json to {args.output_dir.resolve()}")
    if not passed:
        print(f"Quality gate failed: best accuracy {results[best]['accuracy']} < {args.min_accuracy}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
