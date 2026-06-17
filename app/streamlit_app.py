"""Small Streamlit shell for inspecting ranking outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Redrob Ranker", layout="wide")
st.title("Redrob Ranker")

submission_path = st.text_input("Submission CSV", "outputs/submission.csv")
path = Path(submission_path)

if path.exists():
    st.dataframe(pd.read_csv(path), use_container_width=True)
else:
    st.info("Run the ranking CLI to create a submission CSV.")
