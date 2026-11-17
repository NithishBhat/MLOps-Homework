"""What model the API is serving and how good it is."""
import altair as alt
import pandas as pd
import streamlit as st

import api
from sidebar import backend_status, show_api_error

st.set_page_config(page_title="Model info", page_icon="🍷", layout="wide")
health = backend_status()

st.title("Model info")
if health is None:
    st.stop()

try:
    info = api.model_info()
except api.ApiError as err:
    show_api_error(err)
    st.stop()

st.caption(f"{info['model']} with {info['n_estimators']} trees, version {info['version']}, trained {info['trained_at']}")

metric_names = {"cv_accuracy": "5-fold CV accuracy", "accuracy": "Test accuracy", "f1_macro": "Test macro F1"}
columns = st.columns(len(info["metrics"]))
for column, (name, value) in zip(columns, info["metrics"].items()):
    column.metric(metric_names.get(name, name), f"{value:.3f}")
st.write(
    f"Trained on {info['n_train']} wines and tested on {info['n_test']}. With so few test wines, "
    "the CV accuracy is the steadier estimate."
)

st.subheader("Which measurements matter most")
importance = pd.DataFrame(
    {"feature": list(info["feature_importance"]), "importance": list(info["feature_importance"].values())}
)
# Altair instead of st.bar_chart, which would sort the bars alphabetically
chart = alt.Chart(importance).mark_bar().encode(
    x=alt.X("importance", title="Random forest importance"),
    y=alt.Y("feature", sort="-x", title=None),
    tooltip=["feature", "importance"],
)
st.altair_chart(chart, height=420)

with st.expander("Full /model/info response"):
    st.json(info)
