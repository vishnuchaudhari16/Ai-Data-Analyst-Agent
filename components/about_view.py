"""
About UI component.
App metadata, system architecture summary, and technology stack overview.
"""

import streamlit as st
from config import PROJECT_NAME, SUBTITLE, APP_VERSION, DEVELOPER_NAME


def render_about_page():
    """Render About page."""
    st.markdown(f"## ℹ️ About {PROJECT_NAME}")
    st.markdown(f"*{SUBTITLE}*")

    st.markdown("---")

    st.markdown("### 🏛️ Core Design Principle")
    st.info(
        "**Zero Code Execution Security Model**: The LLM never generates or executes free-form Python or SQL strings against your dataset. "
        "Instead, the LLM acts strictly as an intent planner that selects from a catalog of typed, unit-tested Python functions in a tool registry. "
        "Arguments are validated using Pydantic schemas, and DuckDB queries are parsed via `sqlglot` AST allowlist verification before execution."
    )

    st.markdown("### 🛠️ Technology Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- **Core**: Python 3.10+, Streamlit")
        st.markdown("- **Data Processing**: Pandas, NumPy")
        st.markdown("- **SQL Engine**: DuckDB, Sqlglot")
        st.markdown("- **Visualizations**: Plotly Express")
    with col2:
        st.markdown("- **Machine Learning**: Scikit-Learn")
        st.markdown("- **Statistical & Forecasting**: Scipy, Statsmodels")
        st.markdown("- **PDF / Report Generation**: FPDF2, HTML")
        st.markdown("- **Validation**: Pydantic v2")

    st.markdown("---")

    st.markdown("### 👩‍💻 App Metadata")
    st.write(f"- **Version**: `{APP_VERSION}`")
    st.write(f"- **Developer**: `{DEVELOPER_NAME}`")
    st.write("- **License**: MIT")
