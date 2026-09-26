"""
Pandas Analyzer Tools.
Pure, deterministic Pandas functions registered with Pydantic schemas.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from tools.registry import (
    register_tool,
    GroupbyAggArgs,
    FilterRowsArgs,
    TopNArgs,
    PivotTableArgs,
    CorrelationArgs,
)


@register_tool(
    name="groupby_agg",
    description="Groups dataset by a categorical column and computes aggregated metric (mean, sum, count, min, max, std, median) on a numeric column.",
    schema_model=GroupbyAggArgs,
)
def groupby_agg(df: pd.DataFrame, group_col: str, value_col: str, agg_func: str = "mean", top_n: Optional[int] = 10, sort_ascending: bool = False) -> Dict[str, Any]:
    """Group by column and aggregate metric."""
    if group_col not in df.columns:
        return {"status": "error", "message": f"Column '{group_col}' not found in dataset."}
    if value_col not in df.columns:
        return {"status": "error", "message": f"Column '{value_col}' not found in dataset."}

    valid_aggs = ["mean", "sum", "count", "min", "max", "std", "median"]
    agg_func = agg_func.lower()
    if agg_func not in valid_aggs:
        agg_func = "mean"

    try:
        grouped = df.groupby(group_col)[value_col].agg(agg_func).reset_index()
        grouped.columns = [group_col, f"{value_col}_{agg_func}"]
        grouped = grouped.sort_values(by=f"{value_col}_{agg_func}", ascending=sort_ascending)

        if top_n and top_n > 0:
            grouped = grouped.head(top_n)

        code_snippet = f"""# Groupby Aggregation Analysis
result = (
    df.groupby("{group_col}")["{value_col}"]
      .{agg_func}()
      .reset_index()
      .sort_values(by="{value_col}_{agg_func}", ascending={sort_ascending})
)"""
        if top_n:
            code_snippet += f"\nresult = result.head({top_n})"

        summary_stats = {
            "group_count": len(grouped),
            "max_value": float(grouped[f"{value_col}_{agg_func}"].max()) if not grouped.empty else 0,
            "min_value": float(grouped[f"{value_col}_{agg_func}"].min()) if not grouped.empty else 0,
            "top_category": str(grouped.iloc[0][group_col]) if not grouped.empty else "N/A",
        }

        return {
            "status": "success",
            "data": grouped,
            "summary_stats": summary_stats,
            "code_display": code_snippet,
            "result_type": "groupby",
        }
    except Exception as e:
        return {"status": "error", "message": f"Error performing groupby aggregation: {str(e)}"}


@register_tool(
    name="filter_rows",
    description="Filters dataset rows based on a column, comparison operator (==, !=, >, >=, <, <=, contains), and target value.",
    schema_model=FilterRowsArgs,
)
def filter_rows(df: pd.DataFrame, column: str, operator: str, value: Any) -> Dict[str, Any]:
    """Filter rows matching condition."""
    if column not in df.columns:
        return {"status": "error", "message": f"Column '{column}' not found in dataset."}

    try:
        op = str(operator).strip()
        if op == "==":
            filtered = df[df[column] == value]
        elif op == "!=":
            filtered = df[df[column] != value]
        elif op in [">", ">="]:
            num_val = float(value)
            filtered = df[df[column] > num_val] if op == ">" else df[df[column] >= num_val]
        elif op in ["<", "<="]:
            num_val = float(value)
            filtered = df[df[column] < num_val] if op == "<" else df[df[column] <= num_val]
        elif op == "contains":
            filtered = df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
        else:
            filtered = df[df[column] == value]

        code_snippet = f"""# Filter Rows Condition
filtered_df = df[df["{column}"] {op} {repr(value)}]"""

        return {
            "status": "success",
            "data": filtered.head(200),  # Safety cap for returned rows
            "summary_stats": {"total_matched": len(filtered), "total_rows": len(df)},
            "code_display": code_snippet,
            "result_type": "filter",
        }
    except Exception as e:
        return {"status": "error", "message": f"Error filtering rows: {str(e)}"}


@register_tool(
    name="top_n",
    description="Returns top N rows sorted by a specified numeric column.",
    schema_model=TopNArgs,
)
def top_n(df: pd.DataFrame, column: str, n: int = 10, ascending: bool = False) -> Dict[str, Any]:
    """Get top N rows by column."""
    if column not in df.columns:
        return {"status": "error", "message": f"Column '{column}' not found in dataset."}

    try:
        sorted_df = df.sort_values(by=column, ascending=ascending).head(n)
        code_snippet = f"""# Top {n} Analysis
top_df = df.sort_values(by="{column}", ascending={ascending}).head({n})"""

        return {
            "status": "success",
            "data": sorted_df,
            "summary_stats": {"top_n_count": len(sorted_df), "sorted_column": column},
            "code_display": code_snippet,
            "result_type": "top_n",
        }
    except Exception as e:
        return {"status": "error", "message": f"Error sorting top N: {str(e)}"}


@register_tool(
    name="pivot_table",
    description="Creates a 2D pivot table summarizing a metric across row and column categories.",
    schema_model=PivotTableArgs,
)
def pivot_table(df: pd.DataFrame, index_col: str, columns_col: str, values_col: str, agg_func: str = "mean") -> Dict[str, Any]:
    """Create a pivot table."""
    for col in [index_col, columns_col, values_col]:
        if col not in df.columns:
            return {"status": "error", "message": f"Column '{col}' not found in dataset."}

    try:
        pt = pd.pivot_table(
            df,
            index=index_col,
            columns=columns_col,
            values=values_col,
            aggfunc=agg_func,
            fill_value=0
        ).reset_index()

        code_snippet = f"""# Pivot Table Analysis
pivot_df = pd.pivot_table(
    df,
    index="{index_col}",
    columns="{columns_col}",
    values="{values_col}",
    aggfunc="{agg_func}",
    fill_value=0
).reset_index()"""

        return {
            "status": "success",
            "data": pt,
            "summary_stats": {"index_col": index_col, "columns_col": columns_col, "values_col": values_col},
            "code_display": code_snippet,
            "result_type": "pivot",
        }
    except Exception as e:
        return {"status": "error", "message": f"Error creating pivot table: {str(e)}"}


@register_tool(
    name="correlation_matrix",
    description="Calculates Pearson correlation matrix for numeric columns.",
    schema_model=CorrelationArgs,
)
def correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate correlation matrix for numeric columns."""
    try:
        if columns:
            valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        else:
            valid_cols = list(df.select_dtypes(include=[np.number]).columns)

        if len(valid_cols) < 2:
            return {"status": "error", "message": "At least 2 numeric columns are required to calculate correlation."}

        corr_df = df[valid_cols].corr().round(3).reset_index()
        corr_df.rename(columns={"index": "Variable"}, inplace=True)

        code_snippet = f"""# Correlation Analysis
numeric_cols = {valid_cols}
corr_matrix = df[numeric_cols].corr().round(3)"""

        return {
            "status": "success",
            "data": corr_df,
            "summary_stats": {"analyzed_columns": valid_cols},
            "code_display": code_snippet,
            "result_type": "correlation",
        }
    except Exception as e:
        return {"status": "error", "message": f"Error calculating correlation matrix: {str(e)}"}
