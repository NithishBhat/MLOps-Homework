"""Upload a CSV or JSON of wines and predict them all."""
from pathlib import Path

import streamlit as st

import api
from inputs import BATCH_LIMIT, check_columns, chunks, read_batch, results_table, to_wines
from sidebar import backend_status, show_api_error
from wine_features import CULTIVAR_LABELS

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "sample_wines.csv"

st.set_page_config(page_title="Batch predictions", page_icon="🍷", layout="wide")
health = backend_status()

st.title("Batch predictions")
st.write(
    "Upload a CSV or JSON file with one wine per row. It needs the 13 measurement columns. "
    "Other columns (an ID, say) are kept and shown next to the results. "
    f"Files with more than {BATCH_LIMIT} rows are sent in batches of {BATCH_LIMIT}."
)
st.download_button("Download a sample CSV", SAMPLE_CSV.read_bytes(), file_name="sample_wines.csv", mime="text/csv")

uploaded = st.file_uploader("Wines to predict", type=["csv", "json"])
if uploaded:
    try:
        df = read_batch(uploaded.name, uploaded.getvalue())
    except ValueError as err:
        st.error(f"Can't read this file: {err}")
        st.stop()

    missing, extra = check_columns(df)
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()
    if extra:
        st.info(f"Kept but not sent to the model: {', '.join(extra)}")
    st.write(f"{len(df)} wines")
    st.dataframe(df, use_container_width=True, height=220)

    if st.button("Predict all", type="primary", disabled=health is None):
        try:
            wines = to_wines(df)
        except ValueError as err:
            st.error(str(err))
            st.stop()

        predictions = []
        progress = st.progress(0.0, text="Predicting...")
        try:
            for batch in chunks(wines):
                predictions += api.predict_batch(batch)["predictions"]
                progress.progress(len(predictions) / len(wines), text=f"{len(predictions)} / {len(wines)}")
        except api.ApiError as err:
            progress.empty()
            show_api_error(err)
            st.stop()
        progress.empty()

        results = results_table(df, predictions)
        st.subheader("Results")
        counts = results["predicted_cultivar"].map(lambda c: CULTIVAR_LABELS.get(c, c)).value_counts().sort_index()
        st.bar_chart(counts, horizontal=True, height=160)
        low = results[results["confidence"] < 0.6]
        if len(low):
            what = "1 prediction has" if len(low) == 1 else f"{len(low)} predictions have"
            st.warning(f"{what} confidence below 60%. Check those wines.")
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Download results CSV", results.to_csv(index=False).encode(), file_name="predictions.csv", mime="text/csv"
        )
