"""
AI Report UI component.
Generates interactive executive reports with downloadable HTML, CSV, and PDF formats.
"""

import streamlit as st
import pandas as pd
from tools.report_tool import generate_html_report, generate_pdf_report
from agents.report_agent import synthesize_executive_summary


def render_report_page(df: pd.DataFrame, meta: dict):
    """Render AI Executive Report Page."""
    st.markdown("## 📑 AI Executive Report")
    st.markdown("Generate and export full analytical reports for stakeholders in HTML, CSV, or PDF format.")

    if df is None or df.empty:
        st.warning("⚠️ Please upload a dataset to generate a report.")
        return

    with st.spinner("🧠 Synthesizing executive report..."):
        exec_summary = synthesize_executive_summary(df, meta)

    st.markdown("### 📌 Executive Summary")
    st.info(exec_summary)

    st.markdown("---")
    st.markdown("### 📥 Download Options")

    col_h, col_c, col_p = st.columns(3)

    # 1. HTML Download
    with col_h:
        html_content = generate_html_report(df, meta, executive_summary=exec_summary)
        st.download_button(
            label="📄 Download HTML Report",
            data=html_content,
            file_name=f"Executive_Report_{meta.get('filename', 'dataset')}.html",
            mime="text/html",
            use_container_width=True,
        )

    # 2. CSV Download
    with col_c:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📊 Download Processed CSV",
            data=csv_data,
            file_name=f"Processed_{meta.get('filename', 'dataset')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 3. PDF Download (fpdf2)
    with col_p:
        try:
            pdf_bytes = generate_pdf_report(df, meta, executive_summary=exec_summary)
            st.download_button(
                label="📕 Download PDF Report",
                data=pdf_bytes,
                file_name=f"Executive_Report_{meta.get('filename', 'dataset')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF generation error: {str(e)}")
