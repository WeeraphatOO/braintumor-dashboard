import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data
def load_data(n=100):
    rng = np.random.default_rng(42)
    start = datetime.today() - timedelta(days=180)

    df = pd.DataFrame({
        "date": [start + timedelta(days=int(x)) for x in rng.integers(0, 180, n)],
        "category": rng.choice(["A", "B", "C"], n),
        "region": rng.choice(["Bangkok", "Chiang Mai", "Phuket"], n),
        "revenue": rng.normal(1500, 500, n).clip(100)
    })

    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df