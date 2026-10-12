"""Stage 2: calibrate the trained model's predicted probabilities.

The original lab's "calibration" workflow only retrained the model; nothing was
actually calibrated. This stage wraps the frozen forest in
CalibratedClassifierCV and fits only the calibrator (Platt/sigmoid or
isotonic) on the held-back calibration split.

Usage (from the Lab2 folder):
    python -m src.calibrate_model --timestamp 20261012120000 --method sigmoid
"""
import argparse

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.data import ARTIFACTS_DIR


def calibrate(model, X_cal, y_cal, method="sigmoid"):
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated.fit(X_cal, y_cal)
    return calibrated


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--method", choices=["sigmoid", "isotonic"], default="sigmoid")
    args = parser.parse_args(argv)

    raw_path = ARTIFACTS_DIR / f"model_{args.timestamp}_raw.joblib"
    split_path = ARTIFACTS_DIR / f"calibration_split_{args.timestamp}.npz"
    if not raw_path.exists() or not split_path.exists():
        raise SystemExit(f"Missing {raw_path.name} or {split_path.name}; run src.train_model first")

    split = np.load(split_path)
    calibrated = calibrate(joblib.load(raw_path), split["X"], split["y"], args.method)

    out_path = ARTIFACTS_DIR / f"model_{args.timestamp}_calibrated.joblib"
    joblib.dump(calibrated, out_path, compress=3)
    print(f"Calibrated with method={args.method} on {len(split['y'])} rows -> {out_path}")


if __name__ == "__main__":
    main()
