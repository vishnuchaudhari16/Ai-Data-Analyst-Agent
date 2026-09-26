"""
AI Data Analyst Agent.
Main Streamlit application entrypoint.
"""

import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import PROJECT_NAME, SAMPLE_DATA_DIR
from utils.session_manager import init_session_state
from utils.file_handler import load_dataset_from_bytes
from components.cards import apply_custom_css
from components.sidebar import render_sidebar
from components.dashboard import render_dashboard
from components.chat import render_chat_interface
from components.explore import render_explore_page
from components.sql_analyst import render_sql_analyst
from components.statistics_view import render_statistics_page
from components.anomalies_view import render_anomalies_page
from components.predictive_view import render_predictive_page
from components.eda_view import render_eda_page
from components.report_view import render_report_page
from components.settings_view import render_settings_page
from components.about_view import render_about_page


def main():
    """Main application launcher."""
    # Apply custom CSS aesthetic
    apply_custom_css()

    # Initialize session state
    init_session_state()

    # Auto-load default sample dataset (sales.csv) if none loaded yet
    if st.session_state.dataset is None:
        default_sample = SAMPLE_DATA_DIR / "sales.csv"
        if default_sample.exists():
            with open(default_sample, "rb") as f:
                bytes_data = f.read()
            df, meta, err = load_dataset_from_bytes(bytes_data, "sales.csv")
            if df is not None:
                st.session_state.dataset = df
                st.session_state.dataset_meta = meta

    # Render Sidebar Navigation
    active_page = render_sidebar()

    df = st.session_state.get("dataset")
    meta = st.session_state.get("dataset_meta", {})

    # Route to selected page component
    if active_page == "📊 Dashboard":
        render_dashboard(df, meta)
    elif active_page == "💬 AI Analyst":
        render_chat_interface(df, meta)
    elif active_page == "🔍 Explore Data":
        render_explore_page(df, meta)
    elif active_page == "⚡ SQL Analyst":
        render_sql_analyst(df, meta)
    elif active_page == "🧪 Statistics":
        render_statistics_page(df, meta)
    elif active_page == "🚨 Anomalies":
        render_anomalies_page(df, meta)
    elif active_page == "🤖 Predictive Analytics":
        render_predictive_page(df, meta)
    elif active_page == "⚡ AI EDA":
        render_eda_page(df, meta)
    elif active_page == "📑 AI Report":
        render_report_page(df, meta)
    elif active_page == "⚙️ Settings":
        render_settings_page()
    elif active_page == "ℹ️ About":
        render_about_page()


if __name__ == "__main__":
    main()
