"""
UI Cards and Custom CSS components for Streamlit.
Provides metric cards, data quality score meters, custom badges, and responsive CSS styling.
"""

import streamlit as st


def apply_custom_css():
    """Inject custom modern CSS styles for Streamlit."""
    custom_css = """
    <style>
    /* Global Styling & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background & Padding */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
    }
    
    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .metric-card .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    
    .metric-card .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    
    .metric-card .metric-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }
    
    /* Quality Score Badge */
    .quality-container {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    .quality-score-circle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100px;
        height: 100px;
        border-radius: 50%;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    
    .quality-excellent {
        background: radial-gradient(circle, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.05) 100%);
        color: #10B981;
        border: 3px solid #10B981;
    }
    
    .quality-good {
        background: radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.05) 100%);
        color: #3B82F6;
        border: 3px solid #3B82F6;
    }
    
    .quality-fair {
        background: radial-gradient(circle, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.05) 100%);
        color: #F59E0B;
        border: 3px solid #F59E0B;
    }
    
    .quality-poor {
        background: radial-gradient(circle, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.05) 100%);
        color: #EF4444;
        border: 3px solid #EF4444;
    }

    /* Execution Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(79, 70, 229, 0.15);
        color: #818CF8;
        border: 1px solid rgba(129, 140, 248, 0.3);
    }
    
    /* Code Viewer Block */
    .code-viewer-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 8px;
    }

    /* Buttons & Interactive Elements */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* Hide Streamlit Default Footers */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", icon: str = ""):
    """Render a single modern glassmorphic metric card."""
    icon_html = f"<span style='float:right; font-size:1.2rem;'>{icon}</span>" if icon else ""
    subtext_html = f"<div class='metric-subtext'>{subtext}</div>" if subtext else ""
    
    html = f"""
    <div class="metric-card">
        {icon_html}
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {subtext_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_quality_score_card(score: int, label: str, breakdown: dict):
    """Render the Data Quality Score visual card with breakdown progress bars."""
    badge_class = f"quality-{label.lower()}"
    
    st.markdown(f"""
    <div class="quality-container">
        <div class="quality-score-circle {badge_class}">
            {score}
        </div>
        <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 4px;">Data Quality: {label}</div>
        <div style="font-size: 0.85rem; color: #94A3B8;">Calculated across 5 quality dimensions</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 View Data Quality Score Breakdown", expanded=False):
        st.markdown(f"**Completeness (Missing Values)**: {breakdown.get('completeness', 0)}/100")
        st.progress(min(1.0, max(0.0, breakdown.get('completeness', 0) / 100.0)))

        st.markdown(f"**Uniqueness (Duplicates)**: {breakdown.get('uniqueness', 0)}/100")
        st.progress(min(1.0, max(0.0, breakdown.get('uniqueness', 0) / 100.0)))

        st.markdown(f"**Column Variability**: {breakdown.get('column_variability', 0)}/100")
        st.progress(min(1.0, max(0.0, breakdown.get('column_variability', 0) / 100.0)))

        st.markdown(f"**Cardinality Health**: {breakdown.get('cardinality_health', 0)}/100")
        st.progress(min(1.0, max(0.0, breakdown.get('cardinality_health', 0) / 100.0)))

        st.markdown(f"**Value Regularity (Outlier Rate)**: {breakdown.get('value_regularity', 0)}/100")
        st.progress(min(1.0, max(0.0, breakdown.get('value_regularity', 0) / 100.0)))
