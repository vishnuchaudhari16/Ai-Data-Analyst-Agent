"""
Predictive Analytics UI component.
Interactive machine learning model builder and time-series forecaster.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from tools.ml_tool import train_predictive_model, generate_time_series_forecast
from components.cards import render_metric_card


def render_predictive_page(df: pd.DataFrame, meta: dict):
    """Render Predictive Analytics page."""
    st.markdown("## 🤖 Predictive Analytics & Forecasting")
    st.markdown("Build machine learning models to classify outcomes, predict numeric metrics, or forecast future trends.")

    if df is None or df.empty:
        st.warning("⚠️ Please upload a dataset to build predictive models.")
        return

    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    date_cols = list(df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns)

    tab_ml, tab_ts = st.tabs(["🤖 Machine Learning Model", "📈 Time-Series Forecasting"])

    # -----------------------------------------------------------------------
    # TAB 1: Machine Learning Model (Regression / Classification)
    # -----------------------------------------------------------------------
    with tab_ml:
        col1, col2, col3 = st.columns(3)
        with col1:
            task_type = st.selectbox("Select ML Task:", options=["regression", "classification"])
        with col2:
            target_options = num_cols if task_type == "regression" else (cat_cols if cat_cols else df.columns)
            target_col = st.selectbox("Target Variable (y):", options=target_options)
        with col3:
            model_options = ["random_forest", "gradient_boosting", "linear" if task_type == "regression" else "logistic", "decision_tree"]
            model_name = st.selectbox("Model Algorithm:", options=model_options, format_func=lambda x: x.replace("_", " ").title())

        feature_options = [c for c in df.columns if c != target_col]
        selected_features = st.multiselect("Select Feature Columns (X):", options=feature_options, default=feature_options[:6])

        if st.button("🚀 Train & Evaluate Model", use_container_width=True):
            if not selected_features:
                st.error("Please select at least one feature column.")
            else:
                with st.spinner("🤖 Training machine learning model..."):
                    res = train_predictive_model(
                        df,
                        task_type=task_type,
                        target_col=target_col,
                        feature_cols=selected_features,
                        model_name=model_name
                    )

                    if res.get("status") == "success":
                        st.success("Model trained and evaluated successfully!")
                        metrics_df = res.get("data")
                        st.markdown("### 📊 Model Evaluation Metrics")
                        st.dataframe(metrics_df, use_container_width=True)

                        # Feature Importance Chart
                        fi_df = res.get("feature_importance")
                        if isinstance(fi_df, pd.DataFrame) and not fi_df.empty:
                            st.markdown("### 🔍 Feature Importance")
                            fig = px.bar(
                                fi_df,
                                x="Importance Score",
                                y="Feature",
                                orientation="h",
                                color="Importance Score",
                                color_continuous_scale="Viridis",
                                title="Top Feature Importance"
                            )
                            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig, use_container_width=True)

                        with st.expander("🔧 Show Model Training Code", expanded=False):
                            st.code(res.get("code_display"), language="python")
                    else:
                        st.error(res.get("message"))

    # -----------------------------------------------------------------------
    # TAB 2: Time-Series Forecasting
    # -----------------------------------------------------------------------
    with tab_ts:
        if not date_cols:
            st.warning("⚠️ Time-series forecasting requires a date column in the dataset.")
        else:
            col_d, col_v, col_p = st.columns(3)
            with col_d:
                date_col = st.selectbox("Date Column:", options=date_cols)
            with col_v:
                value_col = st.selectbox("Target Numeric Metric to Forecast:", options=num_cols if num_cols else df.columns)
            with col_p:
                periods = st.slider("Forecast Periods Ahead:", min_value=1, max_value=24, value=6)

            if st.button("📈 Generate Time-Series Forecast", use_container_width=True):
                with st.spinner("Calculating Holt-Winters forecast..."):
                    ts_res = generate_time_series_forecast(df, date_col=date_col, value_col=value_col, periods=periods)

                    if ts_res.get("status") == "success":
                        hist_df = ts_res.get("historical_data")
                        fc_df = ts_res.get("forecast_data")

                        st.success("Forecast generated!")

                        # Plot Historical vs Forecast
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=hist_df[date_col], y=hist_df[value_col], mode="lines+markers", name="Historical Data", line=dict(color="#3B82F6")))
                        fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df[f"Forecasted_{value_col}"], mode="lines+markers", name="Forecast Projection", line=dict(color="#10B981", dash="dash")))

                        fig.update_layout(
                            title=f"Time-Series Forecast for {value_col}",
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15, 23, 42, 0.6)"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("### 📋 Forecasted Data Table")
                        st.dataframe(fc_df, use_container_width=True)
                    else:
                        st.error(ts_res.get("message"))
