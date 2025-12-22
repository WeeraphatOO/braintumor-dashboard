import streamlit as st
from data import load_data

st.set_page_config(page_title="Overview", layout="wide")

df = load_data()

st.title("📊 Overview")

k1, k2, k3 = st.columns(3)
k1.metric("Total Revenue", f"{df.revenue.sum():,.0f}")
k2.metric("Average Revenue", f"{df.revenue.mean():,.0f}")
k3.metric("Records", len(df))

st.dataframe(df.head(), width='stretch')