"""Clustering pipeline for handwritten digits: the code each Airflow task runs.

The functions here are plain Python (no Airflow imports), so they can be tested
with pytest and run outside Airflow. Tasks pass small JSON-safe dicts to each
other through XCom; the data and the model live as files in a per-run folder.

Steps: extract -> validate -> score one k (run once per candidate k, in parallel)
-> choose k -> train -> evaluate -> report.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

TRAIN_FILE = "train.npz"
HOLDOUT_FILE = "holdout.npz"
MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"
REPORT_FILE = "report.md"


def _load(run_dir, name):
    data = np.load(Path(run_dir) / name)
    return data["X"], data["y"]


def build_pipeline(k, pca_variance, seed):
    # PCA keeps enough components to explain `pca_variance` of the pixel variance (13 of 64 at 0.8)
    return Pipeline([
        ("pca", PCA(n_components=pca_variance, random_state=seed)),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
    ])


def extract(run_dir, test_size=0.2, seed=42):
    """Load the 8x8 digit images and save a stratified train / hold-out split."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_digits(return_X_y=True)
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    np.savez(run_dir / TRAIN_FILE, X=X_train, y=y_train)
    np.savez(run_dir / HOLDOUT_FILE, X=X_hold, y=y_hold)
    return {"run_dir": str(run_dir), "n_train": len(y_train), "n_holdout": len(y_hold), "n_features": X.shape[1]}


def validate(run_dir):
    """Fail the run early if the data isn't what the model expects."""
    problems = []
    for name in (TRAIN_FILE, HOLDOUT_FILE):
        X, y = _load(run_dir, name)
        if X.ndim != 2 or X.shape[1] != 64:
            problems.append(f"{name}: expected 64 pixel columns, got shape {X.shape}")
        if len(X) != len(y):
            problems.append(f"{name}: {len(X)} images but {len(y)} labels")
        if np.isnan(X).any():
            problems.append(f"{name}: contains NaN pixels")
        # nanmin/nanmax: with plain min/max a single NaN would make this check pass silently
        if X.size and (np.nanmin(X) < 0 or np.nanmax(X) > 16):
            problems.append(f"{name}: pixel values outside 0-16")
        counts = np.bincount(y.astype(int), minlength=10)
        if (counts == 0).any():
            problems.append(f"{name}: digits {np.flatnonzero(counts == 0).tolist()} are missing")
    if problems:
        raise ValueError("Data validation failed:\n" + "\n".join(problems))
    return {"checks_passed": 5}


def score_k(run_dir, k, pca_variance=0.8, seed=42):
    """Fit one candidate k on the training images and return its inertia and silhouette score."""
    X, _ = _load(run_dir, TRAIN_FILE)
    pipe = build_pipeline(k, pca_variance, seed).fit(X)
    Z = pipe.named_steps["pca"].transform(X)
    labels = pipe.named_steps["kmeans"].labels_
    return {
        "k": k,
        "inertia": round(float(pipe.named_steps["kmeans"].inertia_), 2),
        "silhouette": round(float(silhouette_score(Z, labels)), 4),
    }


def choose_k(scores, select_by="silhouette"):
    """Pick k from the per-k scores: highest silhouette, or the elbow of the inertia curve."""
    scores = sorted(scores, key=lambda s: s["k"])
    ks = [s["k"] for s in scores]
    elbow = KneeLocator(ks, [s["inertia"] for s in scores], curve="convex", direction="decreasing").elbow
    best_silhouette = max(scores, key=lambda s: s["silhouette"])["k"]
    if select_by == "elbow" and elbow is not None:
        chosen = int(elbow)
    else:
        chosen = best_silhouette
    return {
        "chosen_k": chosen,
        "select_by": select_by,
        "elbow_k": int(elbow) if elbow is not None else None,
        "best_silhouette_k": best_silhouette,
        "scores": scores,
    }


def train(run_dir, k, pca_variance=0.8, seed=42):
    """Train the final model at the chosen k and save it.

    The digit labels are not used for training. They are only used afterwards to
    name each cluster after the digit most of its training images show.
    """
    X, y = _load(run_dir, TRAIN_FILE)
    pipe = build_pipeline(k, pca_variance, seed).fit(X)
    clusters = pipe.named_steps["kmeans"].labels_
    cluster_digit = {int(c): int(np.bincount(y[clusters == c], minlength=10).argmax()) for c in range(k)}
    joblib.dump({"pipeline": pipe, "cluster_digit": cluster_digit}, Path(run_dir) / MODEL_FILE, compress=3)
    return {
        "k": k,
        "pca_components": int(pipe.named_steps["pca"].n_components_),
        "cluster_digit": cluster_digit,
    }


def evaluate(run_dir, min_accuracy=0.8):
    """Score the saved model on the hold-out images and apply the quality gate."""
    X, y = _load(run_dir, HOLDOUT_FILE)
    saved = joblib.load(Path(run_dir) / MODEL_FILE)
    pipe, cluster_digit = saved["pipeline"], saved["cluster_digit"]
    clusters = pipe.predict(X)
    predicted_digit = np.array([cluster_digit[int(c)] for c in clusters])

    metrics = {
        "n_holdout": len(y),
        "k": int(pipe.named_steps["kmeans"].n_clusters),
        "adjusted_rand_index": round(float(adjusted_rand_score(y, clusters)), 4),
        "normalized_mutual_info": round(float(normalized_mutual_info_score(y, clusters)), 4),
        "silhouette": round(float(silhouette_score(pipe.named_steps["pca"].transform(X), clusters)), 4),
        "mapped_accuracy": round(float((predicted_digit == y).mean()), 4),
        "min_accuracy": min_accuracy,
    }
    metrics["passed_gate"] = metrics["mapped_accuracy"] >= min_accuracy
    (Path(run_dir) / METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    if not metrics["passed_gate"]:
        raise ValueError(f"Quality gate failed: mapped accuracy {metrics['mapped_accuracy']} < {min_accuracy}")
    return metrics


def report(run_dir, choice, metrics):
    """Write a Markdown report: how k was chosen, hold-out metrics and a per-cluster table."""
    X, y = _load(run_dir, HOLDOUT_FILE)
    saved = joblib.load(Path(run_dir) / MODEL_FILE)
    clusters = saved["pipeline"].predict(X)

    lines = [
        "# Digit clustering report",
        "",
        f"Chosen k = **{choice['chosen_k']}** (by {choice['select_by']}; "
        f"elbow k = {choice['elbow_k']}, best silhouette k = {choice['best_silhouette_k']})",
        "",
        "## Hold-out metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value} |" for name, value in metrics.items()]
    lines += ["", "## Candidate k", "", "| k | inertia | silhouette |", "|---|---|---|"]
    lines += [f"| {s['k']} | {s['inertia']} | {s['silhouette']} |" for s in choice["scores"]]
    lines += ["", "## Clusters (hold-out images)", "", "| Cluster | Named after digit | Images | Purity |", "|---|---|---|---|"]
    for c, digit in sorted(saved["cluster_digit"].items(), key=lambda kv: (kv[1], kv[0])):
        members = y[clusters == c]
        purity = (members == digit).mean() if len(members) else 0.0
        lines.append(f"| {c} | {digit} | {len(members)} | {purity:.2f} |")

    path = Path(run_dir) / REPORT_FILE
    path.write_text("\n".join(lines) + "\n")
    return str(path)
