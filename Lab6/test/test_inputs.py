import json

import pandas as pd
import pytest

from inputs import check_columns, chunks, parse_single, read_batch, results_table, to_wines
from wine_features import FEATURE_NAMES, PRESETS

WINE = PRESETS["Cultivar 1 wine (dataset row 0)"]


def test_presets_have_every_feature():
    for values in PRESETS.values():
        assert list(values) == FEATURE_NAMES


def test_parse_single_accepts_original_lab_format_and_plain_object():
    assert parse_single({"input_test": WINE}) == (WINE, [])
    wine, ignored = parse_single({**WINE, "note": "from the cellar"})
    assert wine == WINE
    assert ignored == ["note"]


def test_parse_single_reports_missing_fields():
    with pytest.raises(ValueError, match="proline"):
        parse_single({k: v for k, v in WINE.items() if k != "proline"})
    with pytest.raises(ValueError):
        parse_single([WINE])


def test_read_batch_csv_and_json(tmp_path):
    df = pd.DataFrame([WINE, WINE])
    from_csv = read_batch("wines.csv", df.to_csv(index=False).encode())
    from_list = read_batch("wines.json", json.dumps([WINE, WINE]).encode())
    from_obj = read_batch("wines.json", json.dumps({"wines": [WINE]}).encode())
    assert len(from_csv) == len(from_list) == 2 and len(from_obj) == 1
    with pytest.raises(ValueError):
        read_batch("wines.json", b'{"something": 1}')


def test_check_columns_keeps_extra_columns():
    df = pd.DataFrame([{**WINE, "wine_id": "A-1"}]).drop(columns=["hue"])
    assert check_columns(df) == (["hue"], ["wine_id"])


def test_to_wines_rejects_empty_and_text_cells():
    df = pd.DataFrame([WINE, {**WINE, "ash": None}, {**WINE, "hue": "dark"}])
    with pytest.raises(ValueError, match="Rows with empty or non-numeric values: 2, 3"):
        to_wines(df)
    assert to_wines(pd.DataFrame([WINE])) == [WINE]


def test_chunks_respects_the_api_limit():
    sizes = [len(c) for c in chunks(list(range(250)))]
    assert sizes == [100, 100, 50]


def test_results_table_adds_prediction_columns():
    df = pd.DataFrame([{**WINE, "wine_id": "A-1"}])
    preds = [{"cultivar": "cultivar_1", "confidence": 0.97,
              "probabilities": {"cultivar_1": 0.97, "cultivar_2": 0.02, "cultivar_3": 0.01}}]
    out = results_table(df, preds)
    assert out.loc[0, "predicted_cultivar"] == "cultivar_1"
    # id first, then the predictions, then the 13 measurements
    assert list(out.columns[:6]) == [
        "wine_id", "predicted_cultivar", "confidence", "p_cultivar_1", "p_cultivar_2", "p_cultivar_3",
    ]
    assert list(out.columns[6:]) == FEATURE_NAMES
