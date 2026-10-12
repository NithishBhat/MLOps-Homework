"""Synthetic dataset shared by every stage of the Lab 2 pipeline.

The original lab regenerated a *different* random dataset (with a random size,
possibly 0 rows) in both the training and evaluation scripts, and scored the
model on the data it was trained on. Here one fixed pool of rows defines the
problem:

* the last HOLDOUT_SIZE rows are a fixed hold-out set, never trained on, so
  F1 scores from different pipeline runs are directly comparable;
* each run samples its training rows from the rest of the pool using a seed
  (the CI run number), simulating "new data arrived" on every run.
"""
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

LAB_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_DIR / "artifacts"   # scratch output of one run (gitignored)
MODELS_DIR = LAB_DIR / "models"         # promoted (champion) models, committed by CI
METRICS_DIR = LAB_DIR / "metrics"       # per-run metrics + champion.json, committed by CI

POOL_SIZE = 10_000
HOLDOUT_SIZE = 2_000
N_FEATURES = 6
MIN_SAMPLES = 200


def _pool():
    return make_classification(
        n_samples=POOL_SIZE,
        n_features=N_FEATURES,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        flip_y=0.05,
        class_sep=0.8,
        random_state=0,
        shuffle=True,
    )


def load_holdout():
    """Fixed evaluation set, identical on every run."""
    X, y = _pool()
    return X[-HOLDOUT_SIZE:], y[-HOLDOUT_SIZE:]


def load_training_data(n_samples, seed):
    """Sample n_samples training rows and split them 75/25 into
    (X_train, X_cal, y_train, y_cal). The calibration split is used only to
    fit the probability calibrator, never to fit the forest itself."""
    max_samples = POOL_SIZE - HOLDOUT_SIZE
    if not MIN_SAMPLES <= n_samples <= max_samples:
        raise ValueError(f"n_samples must be between {MIN_SAMPLES} and {max_samples}, got {n_samples}")

    X, y = _pool()
    X, y = X[:-HOLDOUT_SIZE], y[:-HOLDOUT_SIZE]
    idx = np.random.default_rng(seed).choice(len(X), size=n_samples, replace=False)
    return train_test_split(X[idx], y[idx], test_size=0.25, stratify=y[idx], random_state=seed)
