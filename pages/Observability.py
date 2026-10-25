import streamlit as st

from engimind.observability.dashboard import render_dashboard

st.set_page_config(page_title="EngiMind Observability", layout="wide")
render_dashboard()
