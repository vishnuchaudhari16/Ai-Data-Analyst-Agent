"""
Sidebar UI component.
SaaS navigation sidebar, dataset uploader, sample dataset selector, and API status indicator.
"""

import os
from pathlib import Path
import streamlit as st
from config import PROJECT_NAME, SAMPLE_DATA_DIR, LLM_API_KEY
from utils.file_handler import load_dataset_from_bytes


def render_sidebar():
    """Render sidebar navigation and dataset loader."""
    with st.sidebar:
        st.markdown(f"# 🤖 {PROJECT_NAME}")
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

        # Main Navigation Menu
        page = st.radio(
            "Navigation:",
            options=[
                "📊 Dashboard",
                "💬 AI Analyst",
                "🔍 Explore Data",
                "⚡ SQL Analyst",
                "🧪 Statistics",
                "🚨 Anomalies",
                "🤖 Predictive Analytics",
                "⚡ AI EDA",
                "📑 AI Report",
                "⚙️ Settings",
                "ℹ️ About",
            ],
            key="nav_radio"
        )

        st.markdown("---")
        st.markdown("### 📁 Dataset Source")

        # Sample Datasets Selector
        sample_files = {
            "Select Sample Dataset...": None,
            "Sales Data (sales.csv)": SAMPLE_DATA_DIR / "sales.csv",
            "Employee Productivity (employee_productivity.csv)": SAMPLE_DATA_DIR / "employee_productivity.csv",
            "Customer Churn (customer_churn.csv)": SAMPLE_DATA_DIR / "customer_churn.csv",
        }

        selected_sample_label = st.selectbox("Try with Sample Dataset:", options=list(sample_files.keys()))

        if selected_sample_label and sample_files[selected_sample_label]:
            sample_path = sample_files[selected_sample_label]
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    file_bytes = f.read()
                df, meta, err = load_dataset_from_bytes(file_bytes, sample_path.name)
                if df is not None:
                    st.session_state.dataset = df
                    st.session_state.dataset_meta = meta
                    st.toast(f"Loaded sample: {sample_path.name}", icon="✅")

        # File Upload Widget
        uploaded_file = st.file_uploader("Upload CSV / Excel File:", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            df, meta, err = load_dataset_from_bytes(file_bytes, uploaded_file.name)
            if err:
                st.error(f"⚠️ {err}")
            elif df is not None:
                st.session_state.dataset = df
                st.session_state.dataset_meta = meta
                st.toast(f"Dataset loaded successfully ({len(df):,} rows)", icon="✅")

        st.markdown("---")

        # API Key Status Badge
        if LLM_API_KEY:
            st.markdown("<span style='color:#10B981; font-weight:600; font-size:0.85rem;'>🟢 LLM Connected</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#F59E0B; font-weight:600; font-size:0.85rem;'>🟡 Zero-LLM Mode (Rule Engine Active)</span>", unsafe_allow_html=True)

        return page
