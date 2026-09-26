"""
Statistical Analysis UI component.
Interface for hypothesis tests (T-Test, Chi-Square, ANOVA, Correlation, Descriptive stats).
"""

import streamlit as st
import pandas as pd
from tools.statistics_tool import run_statistical_test


def render_statistics_page(df: pd.DataFrame, meta: dict):
    """Render Statistics page."""
    st.markdown("## 🧪 Statistical Analysis")
    st.markdown("Run hypothesis tests and correlation analysis with strict data type applicability guards.")

    if df is None or df.empty:
        st.warning("⚠️ Please upload a dataset to perform statistical tests.")
        return

    test_type = st.selectbox(
        "Select Statistical Test:",
        options=["descriptive", "correlation", "t_test", "chi_square", "anova"],
        format_func=lambda x: {
            "descriptive": "📊 Descriptive Statistics",
            "correlation": "📈 Correlation Analysis (Pearson & Spearman)",
            "t_test": "⚖️ Independent 2-Sample T-Test",
            "chi_square": "🧩 Chi-Square Test of Independence",
            "anova": "📊 One-Way ANOVA Test"
        }.get(x, x)
    )

    col1, col2 = st.columns(2)
    with col1:
        c1 = st.selectbox("Primary Column (col1):", options=df.columns)
    with col2:
        c2 = st.selectbox("Secondary Column (col2 - optional):", options=[None] + list(df.columns))

    if st.button("Run Statistical Test", use_container_width=True):
        res = run_statistical_test(df, test_type=test_type, col1=c1, col2=c2)
        if res.get("status") == "success":
            st.success("Test completed successfully.")
            st.dataframe(res.get("data"), use_container_width=True)
            with st.expander("🔧 Show Code", expanded=False):
                st.code(res.get("code_display"), language="python")
        else:
            st.error(res.get("message"))
