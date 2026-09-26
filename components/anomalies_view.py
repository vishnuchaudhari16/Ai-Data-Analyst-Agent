"""
Anomalies UI component.
Interactive outlier detection page using IQR, Z-Score, and Isolation Forest.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from tools.anomaly_tool import detect_anomalies
from components.cards import render_metric_card
from utils.formatters import format_number


def render_anomalies_page(df: pd.DataFrame, meta: dict):
    """Render Anomalies page."""
    st.markdown("## 🚨 Anomaly & Outlier Detection")
    st.markdown("Identify abnormal observations in numerical features using statistical and machine learning algorithms.")

    if df is None or df.empty:
        st.warning("⚠️ Please upload a dataset to detect anomalies.")
        return

    num_cols = list(df.select_dtypes(include=["number"]).columns)
    if not num_cols:
        st.error("No numerical columns found in the uploaded dataset.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        target_col = st.selectbox("Select Numeric Column:", options=num_cols)
    with col2:
        method = st.selectbox("Detection Algorithm:", options=["iqr", "zscore", "isolation_forest"], format_func=lambda x: x.upper().replace("_", " "))
    with col3:
        threshold = st.number_input("Threshold (e.g. 1.5 for IQR, 3.0 for Z-Score):", min_value=0.5, max_value=10.0, value=1.5, step=0.1)

    if st.button("Detect Anomalies", use_container_width=True):
        res = detect_anomalies(df, column=target_col, method=method, threshold=threshold)
        if res.get("status") == "success":
            stats = res.get("summary_stats", {})
            anomalous_df = res.get("data")

            # Metric Summary Cards
            c1, c2, c3 = st.columns(3)
            with c1:
                render_metric_card("Total Records", format_number(stats.get("total_records")), icon="📊")
            with c2:
                render_metric_card("Anomalies Detected", format_number(stats.get("anomalies_detected")), icon="🚨")
            with c3:
                render_metric_card("Anomaly Rate", f"{stats.get('anomaly_rate')}%", stats.get("method"), icon="📈")

            st.markdown("---")

            # Scatter Plot with Anomalies Highlighted
            st.markdown("### 📈 Visualizing Anomalies")
            df_plot = df.copy()
            df_plot["Is_Anomaly"] = df_plot.index.isin(anomalous_df.index)
            df_plot["Status"] = df_plot["Is_Anomaly"].map({True: "Anomaly 🚨", False: "Normal Point"})

            fig = px.scatter(
                df_plot,
                x=df_plot.index,
                y=target_col,
                color="Status",
                color_discrete_map={"Normal Point": "#3B82F6", "Anomaly 🚨": "#EF4444"},
                title=f"Outlier Detection on {target_col} ({stats.get('method')})"
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)")
            st.plotly_chart(fig, use_container_width=True)

            # Display Anomalous Records Table
            st.markdown("### 📋 Anomalous Records Table")
            if not anomalous_df.empty:
                st.dataframe(anomalous_df, use_container_width=True)
            else:
                st.info("No anomalies detected at the specified threshold.")

            with st.expander("🔧 Show Analysis Code", expanded=False):
                st.code(res.get("code_display"), language="python")
        else:
            st.error(res.get("message"))
