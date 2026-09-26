"""
Dashboard UI component for displaying dataset overview, metrics, data preview, and column statistics.
"""

import streamlit as st
import pandas as pd
from tools.data_profiler import profile_dataset
from components.cards import render_metric_card, render_quality_score_card
from utils.formatters import format_number, format_bytes


def render_dashboard(df: pd.DataFrame, meta: dict):
    """Render the main dashboard overview page."""
    if df is None or df.empty:
        st.info("👈 Upload a dataset in the sidebar or pick a sample dataset to get started!")
        return

    st.markdown("## 📊 Dataset Dashboard")
    st.markdown(f"Overview for **{meta.get('filename', 'Uploaded Dataset')}**")

    if meta.get("is_sampled"):
        st.warning(f"⚠️ {meta.get('sample_info')}")

    profile = profile_dataset(df)
    quality = profile.get("quality_info", {})

    # Top Metric Grid - 4 columns x 2 rows
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Rows", format_number(profile.get("total_rows")), icon="🔢")
    with c2:
        render_metric_card("Total Columns", format_number(profile.get("total_cols")), icon="📊")
    with c3:
        render_metric_card("Missing Values", f"{profile.get('missing_pct')}%", f"{format_number(profile.get('missing_cells'))} cells", icon="⚠️")
    with c4:
        render_metric_card("Duplicate Rows", format_number(profile.get("duplicate_rows")), icon="📑")

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_metric_card("Numeric Cols", str(len(profile.get("numeric_cols", []))), icon="📈")
    with c6:
        render_metric_card("Categorical Cols", str(len(profile.get("categorical_cols", []))), icon="🏷️")
    with c7:
        render_metric_card("Date Cols", str(len(profile.get("date_cols", []))), icon="📅")
    with c8:
        render_metric_card("Quality Score", f"{quality.get('score', 0)}/100", quality.get("label", "N/A"), icon="🛡️")

    st.markdown("---")

    # Quality Score Card & Main Visual Breakdown
    col_q, col_summary = st.columns([1, 2])
    with col_q:
        render_quality_score_card(
            quality.get("score", 0),
            quality.get("label", "N/A"),
            quality.get("breakdown", {})
        )

    with col_summary:
        st.markdown("### 📋 Quick Dataset Profile")
        st.write(f"- **File Size**: `{meta.get('size_mb', 0)} MB`")
        st.write(f"- **Numeric Attributes**: `{', '.join(profile.get('numeric_cols', [])[:5]) or 'None'}`")
        st.write(f"- **Categorical Attributes**: `{', '.join(profile.get('categorical_cols', [])[:5]) or 'None'}`")
        if profile.get("date_cols"):
            st.write(f"- **Detected Date Columns**: `{', '.join(profile.get('date_cols', []))}`")

    st.markdown("---")

    # Tabs for Data Preview & Column Information
    tab_preview, tab_cols, tab_numeric = st.tabs(["👁️ Data Preview", "📑 Column Information", "📈 Numeric Summary"])

    with tab_preview:
        st.markdown("#### First 100 Rows")
        st.dataframe(df.head(100), use_container_width=True)

    with tab_cols:
        st.markdown("#### Column Data Types & Missingness")
        col_table = profile.get("column_info_table", pd.DataFrame())
        st.dataframe(col_table, use_container_width=True)

    with tab_numeric:
        st.markdown("#### Descriptive Statistics for Numeric Columns")
        num_summary = profile.get("numeric_summary", pd.DataFrame())
        if not num_summary.empty:
            st.dataframe(num_summary, use_container_width=True)
        else:
            st.info("No numerical columns found in this dataset.")
