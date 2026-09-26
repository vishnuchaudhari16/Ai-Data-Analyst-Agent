"""
Visualization Tool.
Deterministically maps query result shapes to interactive Plotly charts.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Palette: Indigo, Cyan, Emerald, Amber, Rose, Purple, Teal, Blue
COLOR_PALETTE = ["#6366F1", "#06B6D4", "#10B981", "#F59E0B", "#F43F5E", "#8B5CF6", "#14B8A6", "#3B82F6"]


def build_plotly_chart(
    data: pd.DataFrame,
    result_type: str = "auto",
    title: str = "",
    x_col: Optional[str] = None,
    y_col: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Deterministically map result shape to the appropriate Plotly chart.
    """
    if data is None or data.empty:
        return None

    df = data.copy()
    cols = list(df.columns)

    # Dark Theme Layout defaults
    theme_layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(bgcolor="#1E293B", font_size=13, font_family="Inter"),
    )

    # 1. Correlation Matrix Heatmap
    if result_type == "correlation" or (len(cols) > 2 and cols[0] == "Variable"):
        feature_cols = [c for c in cols if c != "Variable"]
        fig = px.imshow(
            df[feature_cols].values,
            x=feature_cols,
            y=df["Variable"] if "Variable" in cols else feature_cols,
            color_continuous_scale="Viridis",
            aspect="auto",
            title=title or "Correlation Heatmap"
        )
        fig.update_layout(**theme_layout)
        return fig

    # If x_col and y_col are not explicitly passed, infer them from data shape
    if not x_col or x_col not in cols:
        x_col = cols[0]
    if not y_col or y_col not in cols:
        y_col = cols[1] if len(cols) > 1 else cols[0]

    # Check data types for smart chart selection
    is_x_date = pd.api.types.is_datetime64_any_dtype(df[x_col]) or "date" in str(x_col).lower() or "month" in str(x_col).lower()
    is_x_numeric = pd.api.types.is_numeric_dtype(df[x_col])
    is_y_numeric = pd.api.types.is_numeric_dtype(df[y_col])

    # 2. Time-series Line Chart
    if is_x_date and is_y_numeric:
        df = df.sort_values(by=x_col)
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            markers=True,
            color_discrete_sequence=[COLOR_PALETTE[0]],
            title=title or f"{y_col} over Time"
        )
        fig.update_layout(**theme_layout)
        return fig

    # 3. Categorical vs Aggregate Numeric -> Bar Chart or Pie Chart
    if not is_x_numeric and is_y_numeric:
        # Pie chart if categories <= 8 and all values non-negative
        if len(df) <= 8 and (df[y_col] >= 0).all() and result_type in ["pie", "proportion"]:
            fig = px.pie(
                df,
                names=x_col,
                values=y_col,
                color_discrete_sequence=COLOR_PALETTE,
                title=title or f"{y_col} Distribution by {x_col}"
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(**theme_layout)
            return fig
        else:
            # Bar Chart
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                color=x_col,
                color_discrete_sequence=COLOR_PALETTE,
                title=title or f"{y_col} by {x_col}"
            )
            fig.update_layout(**theme_layout, showlegend=False)
            return fig

    # 4. Two Row-Level Numeric Columns -> Scatter Plot (+ Trendline)
    if is_x_numeric and is_y_numeric and x_col != y_col:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            trendline="ols" if len(df) > 5 else None,
            color_discrete_sequence=[COLOR_PALETTE[1]],
            title=title or f"{y_col} vs {x_col}"
        )
        fig.update_layout(**theme_layout)
        return fig

    # 5. Single Numeric Distribution -> Histogram
    if len(cols) == 1 or (is_x_numeric and x_col == y_col):
        fig = px.histogram(
            df,
            x=x_col,
            nbins=20,
            color_discrete_sequence=[COLOR_PALETTE[2]],
            title=title or f"Distribution of {x_col}"
        )
        fig.update_layout(**theme_layout)
        return fig

    # Default Bar Chart Fallback
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color_discrete_sequence=[COLOR_PALETTE[0]],
        title=title or f"{y_col} by {x_col}"
    )
    fig.update_layout(**theme_layout)
    return fig
