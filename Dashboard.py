# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(layout="wide", page_title="Test Report Dashboard")

@st.cache_data
def load_data(path):
    return pd.read_excel(path, sheet_name=None)

uploaded = st.file_uploader("Upload summary.xlsx", type=["xlsx"])
if uploaded:
    sheets = load_data(uploaded)
    df = sheets["All Steps"]
    flaky = sheets["Flaky Scenarios"]
    fail_rates = sheets["All Scenario Fail Rates"]

    st.header("Overview")
    st.dataframe(df.head(20))

    st.subheader("Failure Rate by Scenario")
    st.bar_chart(fail_rates.set_index("scenario"))

    st.subheader("Flaky Scenarios (>30% fail)")
    st.dataframe(flaky)

    st.subheader("Pass/Fail Distribution")
    pf = df["status"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pf, labels=pf.index, autopct="%1.1f%%", startangle=90)
    st.pyplot(fig)

    st.subheader("Failures Over Time")
    trend = df.groupby(["date", "status"]).size().unstack().fillna(0)
    st.line_chart(trend)
