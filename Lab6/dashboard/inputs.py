"""Reading uploaded files and shaping them into API requests and result tables."""
import io
import json

import pandas as pd

from wine_features import FEATURE_NAMES

BATCH_LIMIT = 100  # the API accepts at most 100 wines per /predict/batch request


def parse_single(data):
    """Read one wine from uploaded JSON.

    Accepts the original lab's format ({"input_test": {...}}) or a plain object with
    the 13 measurements. Returns (wine, ignored_keys).
    """
    if isinstance(data, dict) and "input_test" in data:
        data = data["input_test"]
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object with the 13 measurements.")
    missing = [name for name in FEATURE_NAMES if name not in data]
    if missing:
        raise ValueError(f"Missing measurements: {', '.join(missing)}")
    ignored = sorted(set(data) - set(FEATURE_NAMES))
    return {name: data[name] for name in FEATURE_NAMES}, ignored


def read_batch(filename, content):
    """Read a CSV, or JSON (a list of wines or {"wines": [...]}), into a DataFrame."""
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    data = json.loads(content)
    if isinstance(data, dict) and "wines" in data:
        data = data["wines"]
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError('JSON must be a list of wines or {"wines": [...]}.')
    return pd.DataFrame(data)


def check_columns(df):
    """Return (missing, extra) columns. Extra columns (an id, say) are kept in the results."""
    missing = [name for name in FEATURE_NAMES if name not in df.columns]
    extra = [col for col in df.columns if col not in FEATURE_NAMES]
    return missing, extra


def to_wines(df):
    """Convert the measurement columns to a list of request dicts, refusing empty cells."""
    measurements = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    bad_rows = measurements.index[measurements.isna().any(axis=1)].tolist()
    if bad_rows:
        rows = ", ".join(str(i + 1) for i in bad_rows[:10])
        raise ValueError(f"Rows with empty or non-numeric values: {rows}")
    return measurements.astype(float).to_dict(orient="records")


def chunks(items, size=BATCH_LIMIT):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def results_table(df, predictions):
    """The uploaded rows plus the predicted cultivar, its confidence and every class probability.

    Column order: the user's own columns (an id, say), then the predictions, then the measurements.
    """
    df = df.reset_index(drop=True)
    _, extra = check_columns(df)
    out = df[extra].copy()
    out["predicted_cultivar"] = [p["cultivar"] for p in predictions]
    out["confidence"] = [p["confidence"] for p in predictions]
    for name in predictions[0]["probabilities"]:
        out[f"p_{name}"] = [p["probabilities"][name] for p in predictions]
    return pd.concat([out, df[FEATURE_NAMES]], axis=1)
