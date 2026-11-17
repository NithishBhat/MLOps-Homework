"""End-to-end tests against a running Lab4 API (no fakes).

Skipped unless LAB6_LIVE_API is set. With `docker compose up -d` running:
    LAB6_LIVE_API=1 pytest test/test_live.py          (macOS / Linux / Git Bash)
    $env:LAB6_LIVE_API=1; pytest test/test_live.py    (PowerShell)
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

import api
from conftest import DASHBOARD
from wine_features import PRESETS

pytestmark = pytest.mark.skipif(not os.environ.get("LAB6_LIVE_API"), reason="set LAB6_LIVE_API=1 to run")


def test_api_client_against_real_api():
    assert api.health()["model_loaded"] is True
    result = api.predict(PRESETS["Cultivar 1 wine (dataset row 0)"])
    assert result["cultivar"] == "cultivar_1"
    batch = api.predict_batch([PRESETS[name] for name in list(PRESETS)[1:]])
    assert [p["cultivar"] for p in batch["predictions"]] == ["cultivar_1", "cultivar_2", "cultivar_3"]


def test_real_api_validation_errors_are_readable():
    with pytest.raises(api.ApiError) as err:
        api.predict({**PRESETS["Dataset median"], "alcohol": -1})
    assert err.value.status == 422
    assert err.value.details[0].startswith("alcohol:")


@pytest.mark.parametrize("preset, expected", [
    ("Cultivar 1 wine (dataset row 0)", "Cultivar 1"),
    ("Cultivar 2 wine (dataset row 100)", "Cultivar 2"),
    ("Cultivar 3 wine (dataset row 170)", "Cultivar 3"),
])
def test_dashboard_predicts_with_real_api(preset, expected):
    at = AppTest.from_file(str(DASHBOARD / "Dashboard.py"), default_timeout=30).run()
    assert "Backend online" in at.sidebar.success[0].value
    at.selectbox(key="preset").set_value(preset).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value == expected
