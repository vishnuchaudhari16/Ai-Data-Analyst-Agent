"""
Chat UI component for the AI Analyst tab.
Interactive natural language interface with execution status tracking, interactive charts, and code rendering.
"""

import streamlit as st
import pandas as pd
from agents.analyst_agent import AnalystAgent

EXAMPLE_QUESTIONS = [
    "Which department has the highest average salary?",
    "Show top 10 products by sales.",
    "Find correlations between numerical variables.",
    "What are the biggest outliers in sales?",
    "Filter orders from the North region.",
    "Show total sales by category.",
]


def render_chat_interface(df: pd.DataFrame, meta: dict):
    """Render interactive AI Analyst Chat tab."""
    st.markdown("## 💬 AI Data Analyst Chat")
    st.markdown("Ask natural language questions about your dataset. The agent executes pure deterministic tools and provides grounded insights.")

    if df is None or df.empty:
        st.warning("⚠️ Please upload a dataset or select a sample dataset to chat with the AI Analyst.")
        return

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Example question chips
    st.markdown("##### 💡 Example Questions (Click to ask):")
    cols = st.columns(3)
    clicked_question = None

    for idx, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[idx % 3]:
            if st.button(q, key=f"btn_q_{idx}", use_container_width=True):
                clicked_question = q

    # Chat Input Form
    with st.form(key="chat_input_form", clear_on_submit=True):
        user_input = st.text_input("💬 Ask anything about your data...", value=clicked_question or "")
        submit_button = st.form_submit_button("🤖 Analyze", use_container_width=True)

    query_to_process = clicked_question or (user_input if submit_button else None)

    if query_to_process:
        with st.spinner("🤖 Planning analysis & executing tool..."):
            agent = AnalystAgent()
            res = agent.analyze_question(query_to_process, df, history=st.session_state.chat_history)
            st.session_state.chat_history.append(res)

    # Render Chat History (most recent first)
    if not st.session_state.chat_history:
        st.info("👋 Ask a question above or click an example prompt to begin analysis.")
        return

    st.markdown("---")
    st.markdown("### 📜 Analysis History")

    for idx, turn in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"#### ❓ **Question**: {turn.get('question')}")

            # Execution Pipeline Status Pills
            steps = turn.get("status_steps", [])
            if steps:
                steps_html = " ".join([f"<span class='status-pill'>{s}</span>" for s in steps])
                st.markdown(f"<div style='margin-bottom:12px;'>{steps_html}</div>", unsafe_allow_html=True)

            if turn.get("status") == "error":
                st.error(f"⚠️ {turn.get('message')}")
            else:
                # Insights Text
                st.markdown(turn.get("insights", ""))

                # Visualization Chart
                fig = turn.get("figure")
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_turn_{idx}")

                # Evidence Dataframe
                tool_res = turn.get("tool_result", {})
                data = tool_res.get("data")
                if isinstance(data, pd.DataFrame) and not data.empty:
                    with st.expander("📊 View Computed Evidence Data", expanded=False):
                        st.dataframe(data, use_container_width=True)

                # Show Analysis Code
                code_snippet = turn.get("code_display")
                if code_snippet:
                    with st.expander("🔧 Show Analysis Code", expanded=False):
                        st.code(code_snippet, language="python")

            st.markdown("---")
