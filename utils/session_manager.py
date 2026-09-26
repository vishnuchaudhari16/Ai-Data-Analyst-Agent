"""
Session Manager.
Manages Streamlit session state, conversation history, loaded dataset, and LLM call counters.
"""

import streamlit as st
from config import MAX_SESSION_LLM_CALLS


def init_session_state():
    """Initialize default Streamlit session state variables."""
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "dataset_meta" not in st.session_state:
        st.session_state.dataset_meta = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_call_count" not in st.session_state:
        st.session_state.llm_call_count = 0
    if "active_page" not in st.session_state:
        st.session_state.active_page = "📊 Dashboard"


def increment_llm_counter():
    """Increment session LLM counter."""
    if "llm_call_count" in st.session_state:
        st.session_state.llm_call_count += 1


def is_llm_budget_exceeded() -> bool:
    """Check if session LLM call budget is exceeded."""
    cnt = st.session_state.get("llm_call_count", 0)
    return cnt >= MAX_SESSION_LLM_CALLS
