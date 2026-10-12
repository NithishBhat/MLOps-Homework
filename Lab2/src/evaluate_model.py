"""Stage 3: evaluate on the fixed hold-out set, enforce a quality gate, and
promote the model only if it beats the current champion.

* Quality gate: if the calibrated model's hold-out F1 is below --min-f1 the
  script exits with code 1, which fails the GitHub Actions job so nothing is
  committed.
* Champion/challenger: metrics/champion.json records the best model so far.
  The new (challenger) model is copied into models/ and becomes champion only
  if its F1 is at least the champion's. Otherwise only its metrics are kept.

Writes `promoted=true|false` to $GITHUB_OUTPUT and a results table to
$GITHUB_STEP_SUMMARY when run inside GitHub Actions.

Usage (from the Lab2 folder):
    python -m src.evaluate_model --timestamp 20261012120000 --min-f1 0.80
"""
import argparse
import json
import os
import shutil

import joblib
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

from src.data import ARTIFACTS_DIR, METRICS_DIR, MODELS_DIR, load_holdout


def score(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    return {
        "f1": round(float(f1_score(y, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
    }


def passes_gate(metrics, min_f1):
    return metrics["f1"] >= min_f1


def should_promote(metrics, champion):
    return champion is None or metrics["f1"] >= champion["f1"]


def load_champion():
    path = METRICS_DIR / "champion.json"
    return json.loads(path.read_text()) if path.exists() else None


def _write_github(key, value=None, summary=None):
    if value is not None and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"{key}={value}\n")
    if summary and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--min-f1", type=float, default=0.80)
    args = parser.parse_args(argv)

    X, y = load_holdout()
    raw = score(joblib.load(ARTIFACTS_DIR / f"model_{args.timestamp}_raw.joblib"), X, y)
    calibrated_path = ARTIFACTS_DIR / f"model_{args.timestamp}_calibrated.joblib"
    calibrated = score(joblib.load(calibrated_path), X, y)

    champion = load_champion()
    gate_ok = passes_gate(calibrated, args.min_f1)
    promoted = gate_ok and should_promote(calibrated, champion)

    report = {
        "timestamp": args.timestamp,
        "holdout_size": len(y),
        "min_f1": args.min_f1,
        "raw": raw,
        "calibrated": calibrated,
        "passed_gate": gate_ok,
        "previous_champion": champion["timestamp"] if champion else None,
        "promoted": promoted,
    }
    METRICS_DIR.mkdir(exist_ok=True)
    (METRICS_DIR / f"{args.timestamp}_metrics.json").write_text(json.dumps(report, indent=4) + "\n")

    if promoted:
        MODELS_DIR.mkdir(exist_ok=True)
        shutil.copy(calibrated_path, MODELS_DIR / f"model_{args.timestamp}.joblib")
        if champion:
            # Keep only the current champion's model in git to avoid repo bloat.
            (MODELS_DIR / f"model_{champion['timestamp']}.joblib").unlink(missing_ok=True)
        champion_record = {"timestamp": args.timestamp, "model": f"models/model_{args.timestamp}.joblib", **calibrated}
        (METRICS_DIR / "champion.json").write_text(json.dumps(champion_record, indent=4) + "\n")

    print(json.dumps(report, indent=4))

    champ_f1 = f"{champion['f1']:.4f} ({champion['timestamp']})" if champion else "none yet"
    _write_github("promoted", str(promoted).lower(), summary=(
        f"## Lab 2 model `{args.timestamp}`\n\n"
        "| metric | raw forest | calibrated |\n|---|---|---|\n"
        + "".join(f"| {k} | {raw[k]} | {calibrated[k]} |\n" for k in raw)
        + f"\n- Quality gate (F1 >= {args.min_f1}): **{'passed' if gate_ok else 'FAILED'}**\n"
        f"- Previous champion F1: {champ_f1}\n"
        f"- Promoted to champion: **{promoted}**\n"
    ))

    if not gate_ok:
        raise SystemExit(f"Quality gate failed: F1 {calibrated['f1']} < {args.min_f1}")


if __name__ == "__main__":
    main()
