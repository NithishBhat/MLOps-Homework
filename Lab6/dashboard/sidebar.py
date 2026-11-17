"""Backend status box shown in the sidebar of every page."""
import streamlit as st

import api


def backend_status():
    """Show whether the API is up and which model it serves. Returns the /health body, or None."""
    with st.sidebar:
        st.caption(f"API: `{api.API_URL}`")
        try:
            health = api.health()
        except api.ApiError as err:
            st.error(f"Backend offline. {err}")
            return None
        if health.get("model_loaded"):
            st.success(f"Backend online. Model version {health['model_version']}")
        else:
            st.warning("Backend online, but no model is loaded. Train it in Lab4 and restart the API.")
        return health


def show_api_error(err):
    st.error(str(err))
    for line in err.details:
        st.markdown(f"- `{line}`")
