import streamlit as st
from data import load_data

st.set_page_config(page_title="Data Table", layout="wide")

df = load_data()

st.title("📋 Data Table")

st.dataframe(df, width='stretch')

st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode("utf-8"),
    "data.csv",
    "text/csv"
)
