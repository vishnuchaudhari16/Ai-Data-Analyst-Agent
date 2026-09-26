"""
Explore Data UI component.
Allows interactive filtering, sorting, and manual data exploration.
"""

import streamlit as st
import pandas as pd
from tools.pandas_tool import filter_rows, top_n


def render_explore_page(df: pd.DataFrame, meta: dict):
    """Render Explore Data page."""
    st.markdown("## 🔍 Explore & Filter Data")
    st.markdown("Interactively inspect, filter, and sort your dataset without writing code.")

    if df is None or df.empty:
        st.warning("⚠️ Please upload a dataset to explore.")
        return

    st.markdown("### 🎛️ Data Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_col = st.selectbox("Select Column to Filter:", options=df.columns)
    with col2:
        operator = st.selectbox("Operator:", options=["==", "!=", ">", ">=", "<", "<=", "contains"])
    with col3:
        filter_val = st.text_input("Filter Value:", value="")

    if st.button("Apply Filter", use_container_width=True) and filter_val:
        res = filter_rows(df, column=selected_col, operator=operator, value=filter_val)
        if res.get("status") == "success":
            filtered_df = res.get("data")
            st.success(f"Matched {len(filtered_df)} records (showing up to 200).")
            st.dataframe(filtered_df, use_container_width=True)
            with st.expander("🔧 Show Generated Filter Code", expanded=False):
                st.code(res.get("code_display"), language="python")
        else:
            st.error(res.get("message"))
    else:
        st.dataframe(df.head(100), use_container_width=True)
