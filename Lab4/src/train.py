"""Train the wine classifier and save it together with its metadata.

Usage (from the Lab4 folder):
    python -m src.train
    python -m src.train --n-estimators 300 --seed 7
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

from src.data import CLASS_NAMES, FEATURES, load_data, split_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_FILE = "wine_model.joblib"
METADATA_FILE = "metadata.json"


def train(n_estimators=200, seed=42):
    X_train, X_test, y_train, y_test = split_data(*load_data(), seed=seed)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    # The test set is small (45 rows), so 5-fold CV on the training rows gives a steadier estimate
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "cv_accuracy": round(float(cv.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
    }
    return model, metrics, len(y_train), len(y_test)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the wine classifier.")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args(argv)

    model, metrics, n_train, n_test = train(args.n_estimators, args.seed)
    trained_at = datetime.now(timezone.utc)
    importance = sorted(zip(FEATURES, model.feature_importances_), key=lambda kv: kv[1], reverse=True)
    metadata = {
        "version": trained_at.strftime("%Y%m%d%H%M%S"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "model": type(model).__name__,
        "n_estimators": args.n_estimators,
        "seed": args.seed,
        "n_train": n_train,
        "n_test": n_test,
        "metrics": metrics,
        "features": FEATURES,
        "classes": CLASS_NAMES,
        "feature_importance": {name: round(float(v), 4) for name, v in importance},
        "sklearn_version": sklearn.__version__,
    }

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_dir / MODEL_FILE, compress=3)
    (args.model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))

    print(f"Trained {metadata['model']} version {metadata['version']} on {n_train} rows")
    print(f"5-fold CV accuracy: {metrics['cv_accuracy']}")
    print(f"Test set ({n_test} rows): accuracy={metrics['accuracy']}, f1_macro={metrics['f1_macro']}")
    print(f"Saved to {args.model_dir}")


if __name__ == "__main__":
    main()
