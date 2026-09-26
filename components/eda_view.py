"""
Automated EDA UI component.
Generates an end-to-end Exploratory Data Analysis report by sequentially invoking registered tools.
"""

import streamlit as st
import pandas as pd
from tools.data_profiler import profile_dataset
from tools.pandas_tool import correlation_matrix
from tools.anomaly_tool import detect_anomalies
from tools.visualization_tool import build_plotly_chart
from components.cards import render_quality_score_card, render_metric_card
from utils.formatters import format_number


def render_eda_page(df: pd.DataFrame, meta: dict):
    """Render Automated EDA page."""
    st.markdown("## ⚡ Automated Exploratory Data Analysis (AI EDA)")
    st.markdown("Generate a comprehensive EDA report evaluating dataset health, correlations, distributions, and outliers.")

    if df is None or df.empty:
        st.warning("⚠️ Please upload a dataset to run Automated EDA.")
        return

    if st.button("🚀 Generate Full AI EDA Report", use_container_width=True):
        with st.spinner("🔍 Executing automated profiling & statistical checks..."):
            profile = profile_dataset(df)
            quality = profile.get("quality_info", {})

            st.markdown("### 1. 📊 Executive Dataset Summary")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Total Rows", format_number(profile.get("total_rows")), icon="🔢")
            with c2:
                render_metric_card("Total Columns", format_number(profile.get("total_cols")), icon="📊")
            with c3:
                render_metric_card("Missing Cells", f"{profile.get('missing_pct')}%", icon="⚠️")
            with c4:
                render_metric_card("Quality Score", f"{quality.get('score')}/100", quality.get("label"), icon="🛡️")

            st.markdown("---")

            st.markdown("### 2. 🛡️ Data Quality Score & Health")
            render_quality_score_card(quality.get("score", 0), quality.get("label", "N/A"), quality.get("breakdown", {}))

            st.markdown("---")

            # Numerical Analysis
            num_cols = profile.get("numeric_cols", [])
            if num_cols:
                st.markdown("### 3. 📈 Numerical Distributions & Summary")
                st.dataframe(profile.get("numeric_summary"), use_container_width=True)

                if len(num_cols) >= 2:
                    st.markdown("### 4. 🔗 Correlation Analysis")
                    corr_res = correlation_matrix(df, columns=num_cols)
                    if corr_res.get("status") == "success":
                        fig_corr = build_plotly_chart(corr_res.get("data"), result_type="correlation", title="Pearson Correlation Matrix")
                        if fig_corr:
                            st.plotly_chart(fig_corr, use_container_width=True)

                st.markdown("### 5. 🚨 Outlier Check (IQR Method)")
                target_num = num_cols[0]
                anom_res = detect_anomalies(df, column=target_num, method="iqr")
                if anom_res.get("status") == "success":
                    anom_stats = anom_res.get("summary_stats", {})
                    st.write(f"- Analyzed Feature: `{target_num}`")
                    st.write(f"- Anomalies Detected: `{anom_stats.get('anomalies_detected')} ({anom_stats.get('anomaly_rate')}%)`")
            else:
                st.info("No numerical columns found for correlation or anomaly checks.")
