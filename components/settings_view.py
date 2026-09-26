"""
Settings UI component.
Configures LLM credentials, model parameters, masked API key status, and session limits.
"""

import streamlit as st
from config import LLM_API_KEY, LLM_MODEL, MAX_SESSION_LLM_CALLS


def render_settings_page():
    """Render Settings page."""
    st.markdown("## ⚙️ Application Settings")
    st.markdown("Manage AI LLM provider options, API credentials, model parameters, and call budgets.")

    st.markdown("### 🤖 LLM Provider Configuration")

    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("LLM Provider:", options=["OpenAI / OpenAI-Compatible", "Google Gemini", "Custom Gateway"])
        model = st.selectbox("Model:", options=["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash", "claude-3-5-sonnet"])
    with col2:
        temp = st.slider("Temperature (Creativity):", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        max_tokens = st.number_input("Max Response Tokens:", min_value=100, max_value=2000, value=500)

    st.markdown("---")

    st.markdown("### 🔑 API Key Status")
    if LLM_API_KEY:
        masked_key = f"{LLM_API_KEY[:4]}...{LLM_API_KEY[-4:]}" if len(LLM_API_KEY) > 8 else "****"
        st.success(f"✓ API Key Configured (`{masked_key}`)")
    else:
        st.warning("⚠️ No API Key configured in `.env` or `secrets.toml`.")
        st.info("💡 Zero-LLM Fallback Mode is active: Data exploration, profiling, SQL query builder, statistical tests, anomaly detection, and ML modeling work seamlessly without an API key.")

    api_key_input = st.text_input("Override API Key for this Session:", type="password", help="Entered key stays strictly in session state.")
    if api_key_input:
        import config
        config.LLM_API_KEY = api_key_input
        st.success("API Key updated for active session!")

    st.markdown("---")

    st.markdown("### 📊 Session Usage & Resource Limits")
    cnt = st.session_state.get("llm_call_count", 0)
    st.write(f"- **Session LLM Calls**: `{cnt} / {MAX_SESSION_LLM_CALLS}`")
    st.progress(min(1.0, cnt / float(MAX_SESSION_LLM_CALLS)))
