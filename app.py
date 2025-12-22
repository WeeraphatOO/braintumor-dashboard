import streamlit as st

st.set_page_config(
    page_title="Dashboard Home",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Dashboard Home")

st.markdown("""
Welcome to the **Streamlit multi-page dashboard**.

Use the sidebar to navigate:
- 📊 Overview
- 📈 Analytics
- 📋 Data Table
- ⚙️ Settings
""")

st.info("Each page is split into its own Python file.")
