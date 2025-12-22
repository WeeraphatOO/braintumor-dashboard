import streamlit as st

st.set_page_config(page_title="Settings", layout="wide")

st.title("⚙️ Settings")

theme = st.selectbox("Theme", ["Light", "Dark"])
refresh = st.slider("Auto-refresh (seconds)", 5, 300, 60)

st.success("Settings saved (demo).")
