import streamlit as st
import plotly.express as px
from data import load_data

st.set_page_config(page_title="Analytics", layout="wide")

df = load_data()

st.title("📈 Analytics")

monthly = df.groupby("month", as_index=False).revenue.sum()
fig = px.line(monthly, x="month", y="revenue", title="Revenue Trend")

st.plotly_chart(fig, width='stretch')