"""Run the real Streamlit pages with AppTest, against a fake backend (see conftest.py)."""
from streamlit.testing.v1 import AppTest

from conftest import DASHBOARD, MODEL_INFO
from wine_features import PRESETS


def run_page(path):
    at = AppTest.from_file(str(DASHBOARD / path), default_timeout=30)
    return at.run()


def test_sidebar_shows_backend_online(fake_backend):
    at = run_page("Dashboard.py")
    assert not at.exception
    assert "Backend online" in at.sidebar.success[0].value
    assert len(at.slider) == 13


def test_sidebar_shows_backend_offline(offline_backend):
    at = run_page("Dashboard.py")
    assert not at.exception
    assert "Backend offline" in at.sidebar.error[0].value


def test_sliders_start_at_dataset_median(fake_backend):
    at = run_page("Dashboard.py")
    assert at.slider(key="proline").value == 673.5
    assert at.slider(key="alcohol").value == 13.05


def test_preset_moves_sliders_and_predict_sends_them(fake_backend):
    at = run_page("Dashboard.py")
    preset = "Cultivar 3 wine (dataset row 170)"
    at.selectbox(key="preset").set_value(preset).run()
    assert at.slider(key="flavanoids").value == PRESETS[preset]["flavanoids"]

    at.button[0].click().run()  # the form's Predict button
    assert not at.exception
    assert fake_backend[-1] == {k: float(v) for k, v in PRESETS[preset].items()}
    assert at.metric[0].value == "Cultivar 3"
    assert at.metric[1].value == "90%"
    assert at.metric[2].value == "Cultivar 2 (5%)"


def test_predict_is_blocked_when_backend_offline(offline_backend):
    at = run_page("Dashboard.py")
    at.button[0].click().run()
    assert any("backend is offline" in e.value for e in at.main.error)


def test_model_info_page(fake_backend):
    at = run_page("pages/2_Model_Info.py")
    assert not at.exception
    values = [m.value for m in at.metric]
    assert values == [f"{v:.3f}" for v in MODEL_INFO["metrics"].values()]
    assert "RandomForestClassifier" in at.main.caption[0].value


def test_batch_page_loads(fake_backend):
    at = run_page("pages/1_Batch_Predictions.py")
    assert not at.exception
    assert at.title[0].value == "Batch predictions"
