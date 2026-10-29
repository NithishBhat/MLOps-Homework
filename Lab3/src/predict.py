"""Load a model saved by main.py and predict on a few sample rows.

Usage (in the container, or from the Lab3/src folder):
    python predict.py                          # best model from metrics.json
    python predict.py --model random_forest -n 10
"""
import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer


def predict(output_dir, model_name=None, n=5, seed=0):
    if model_name is None:
        model_name = json.loads((output_dir / "metrics.json").read_text())["best_model"]
    model = joblib.load(output_dir / "models" / f"{model_name}.joblib")

    data = load_breast_cancer()
    rows = np.random.default_rng(seed).choice(len(data.target), size=n, replace=False)
    probs = model.predict_proba(data.data[rows])
    return model_name, [
        {
            "row": int(i),
            "predicted": str(data.target_names[p.argmax()]),
            "confidence": round(float(p.max()), 3),
            "actual": str(data.target_names[data.target[i]]),
        }
        for i, p in zip(rows, probs)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict with a saved model.")
    parser.add_argument("--model", default=None, help="Model name (default: best model in metrics.json)")
    parser.add_argument("-n", type=int, default=5, help="Number of sample rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "output")))
    args = parser.parse_args(argv)

    if not (args.output_dir / "metrics.json").exists():
        print(f"No metrics.json in {args.output_dir}. Run main.py first.", file=sys.stderr)
        return 1

    model_name, preds = predict(args.output_dir, args.model, args.n, args.seed)
    print(f"Predictions from {model_name}:\n")
    print(f"{'row':>5}  {'predicted':<10}{'confidence':>11}  {'actual':<10}")
    for p in preds:
        mark = "" if p["predicted"] == p["actual"] else "  <- wrong"
        print(f"{p['row']:>5}  {p['predicted']:<10}{p['confidence']:>11.3f}  {p['actual']:<10}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
