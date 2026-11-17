"""Lab 6: Streamlit dashboard for the Lab4 wine cultivar API.

Run (from the Lab6/dashboard folder, with the Lab4 API running on port 8000):
    streamlit run Dashboard.py
"""
import json

import pandas as pd
import streamlit as st

import api
from inputs import parse_single
from sidebar import backend_status, show_api_error
from wine_features import CULTIVAR_LABELS, FEATURES, PRESETS

st.set_page_config(page_title="Wine Cultivar Classifier", page_icon="🍷", layout="wide")


def apply_preset():
    # Runs before the rerun, so the sliders below pick up the new values
    for name, value in PRESETS[st.session_state["preset"]].items():
        st.session_state[name] = float(value)


def show_prediction(result):
    label = CULTIVAR_LABELS.get(result["cultivar"], result["cultivar"])
    runner_up, runner_up_prob = sorted(result["probabilities"].items(), key=lambda kv: kv[1])[-2]
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted cultivar", label)
    col2.metric("Confidence", f"{result['confidence']:.0%}")
    col3.metric("Runner-up", f"{CULTIVAR_LABELS.get(runner_up, runner_up)} ({runner_up_prob:.0%})")
    st.caption(f"Model version {result['model_version']}")

    probs = pd.DataFrame({
        "cultivar": [CULTIVAR_LABELS.get(k, k) for k in result["probabilities"]],
        "probability": list(result["probabilities"].values()),
    }).set_index("cultivar")
    st.bar_chart(probs, y="probability", horizontal=True, height=180)


def main():
    health = backend_status()

    st.title("🍷 Wine Cultivar Classifier")
    st.write(
        "Predicts which of three cultivars a wine comes from, using 13 chemical measurements. "
        "The prediction comes from the Lab4 FastAPI service."
    )

    source = st.radio("Input", ["Sliders", "Upload JSON"], horizontal=True, label_visibility="collapsed")

    wine = None
    if source == "Sliders":
        # Defaults on first load; afterwards the sliders keep their own values
        for f in FEATURES:
            st.session_state.setdefault(f.name, float(f.median))
        st.selectbox("Start from", list(PRESETS), key="preset", on_change=apply_preset)

        with st.form("wine_form"):
            columns = st.columns(3)
            for i, f in enumerate(FEATURES):
                columns[i % 3].slider(
                    f"{f.label} ({f.unit})" if f.unit else f.label,
                    min_value=f.min, max_value=f.max, step=f.step, key=f.name,
                    help=f"Dataset range {f.min:g} – {f.max:g}, median {f.median:g}",
                )
            submitted = st.form_submit_button("Predict", type="primary")
        if submitted:
            wine = {f.name: st.session_state[f.name] for f in FEATURES}
    else:
        uploaded = st.file_uploader(
            "JSON with one wine: the 13 measurements, or the original lab's {\"input_test\": {...}} format",
            type=["json"],
        )
        if uploaded:
            try:
                parsed, ignored = parse_single(json.load(uploaded))
            except (ValueError, json.JSONDecodeError) as err:
                st.error(f"Can't use this file: {err}")
            else:
                if ignored:
                    st.warning(f"Ignoring fields the model doesn't use: {', '.join(ignored)}")
                with st.expander("File contents", expanded=True):
                    st.json(parsed)
                if st.button("Predict", type="primary"):
                    wine = parsed

    if wine is not None:
        if health is None:
            st.error("The backend is offline, so there is nothing to predict with.")
            return
        # Placeholder that is filled once the API answers (st.empty, as in the original lab)
        result_area = st.empty()
        try:
            with st.spinner("Predicting..."):
                result = api.predict(wine)
        except api.ApiError as err:
            with result_area.container():
                show_api_error(err)
        else:
            st.toast(f"Predicted {CULTIVAR_LABELS.get(result['cultivar'], result['cultivar'])}", icon="🍷")
            with result_area.container():
                show_prediction(result)
                with st.expander("Request sent to the API"):
                    st.json(wine)


main()
