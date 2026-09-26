"""
Data profiler tool.
Computes dataset statistics, data type breakdowns, missingness, cardinality, and Data Quality Score.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def compute_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute an intelligent 0-100 Data Quality Score with detailed factor weight breakdown.
    
    Factors:
    - Missing Values Penalty (weight 35%)
    - Duplicate Rows Penalty (weight 25%)
    - Constant/Zero-Variance Columns Penalty (weight 15%)
    - High-Cardinality Categorical Penalty (weight 15%)
    - Data Type / Outlier Penalty (weight 10%)
    """
    if df.empty:
        return {"score": 0, "label": "Poor", "breakdown": {}}

    total_cells = df.size
    total_rows = len(df)
    total_cols = len(df.columns)

    # 1. Missingness factor (0-100)
    missing_cells = df.isna().sum().sum()
    missing_pct = (missing_cells / total_cells) if total_cells > 0 else 0
    missing_score = max(0, 100 - (missing_pct * 250))  # 10% missing = 75 score, 40% missing = 0

    # 2. Duplicate factor (0-100)
    dup_rows = df.duplicated().sum()
    dup_pct = (dup_rows / total_rows) if total_rows > 0 else 0
    dup_score = max(0, 100 - (dup_pct * 300))  # 10% dups = 70 score

    # 3. Constant columns factor (0-100)
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    const_pct = (len(constant_cols) / total_cols) if total_cols > 0 else 0
    constant_score = max(0, 100 - (const_pct * 200))

    # 4. High-cardinality non-numeric columns factor
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    high_card_cols = [col for col in cat_cols if df[col].nunique() / total_rows > 0.8 and df[col].nunique() > 20]
    high_card_pct = (len(high_card_cols) / total_cols) if total_cols > 0 else 0
    cardinality_score = max(0, 100 - (high_card_pct * 150))

    # 5. Outlier/Type factor (IQR check on numeric)
    num_cols = df.select_dtypes(include=[np.number]).columns
    outlier_count = 0
    for col in num_cols:
        s = df[col].dropna()
        if len(s) > 10:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = s[(s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)]
                outlier_count += len(outliers)
    outlier_pct = (outlier_count / total_cells) if total_cells > 0 else 0
    outlier_score = max(0, 100 - (outlier_pct * 500))

    # Weighted final score
    final_score = int(round(
        (missing_score * 0.35) +
        (dup_score * 0.25) +
        (constant_score * 0.15) +
        (cardinality_score * 0.15) +
        (outlier_score * 0.10)
    ))
    final_score = max(0, min(100, final_score))

    if final_score >= 85:
        label = "Excellent"
    elif final_score >= 70:
        label = "Good"
    elif final_score >= 50:
        label = "Fair"
    else:
        label = "Poor"

    breakdown = {
        "completeness": round(missing_score, 1),
        "uniqueness": round(dup_score, 1),
        "column_variability": round(constant_score, 1),
        "cardinality_health": round(cardinality_score, 1),
        "value_regularity": round(outlier_score, 1),
        "missing_pct": round(missing_pct * 100, 2),
        "duplicate_rows": dup_rows,
        "constant_columns_count": len(constant_cols),
        "high_cardinality_cols_count": len(high_card_cols),
    }

    return {
        "score": final_score,
        "label": label,
        "breakdown": breakdown,
    }


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data profiling statistics for a dataframe.
    """
    if df is None or df.empty:
        return {}

    total_rows = len(df)
    total_cols = len(df.columns)
    missing_cells = int(df.isna().sum().sum())
    total_cells = total_rows * total_cols
    missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())

    # Column type classifications
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    date_cols = list(df.select_dtypes(include=["datetime", "datetime64[ns]", "datetimetz"]).columns)
    categorical_cols = list(df.select_dtypes(include=["object", "string", "category", "bool"]).columns)

    # Column info table
    col_summary = []
    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        missing_cnt = int(s.isna().sum())
        missing_p = round((missing_cnt / total_rows) * 100, 2) if total_rows > 0 else 0.0
        unique_cnt = int(s.nunique(dropna=True))

        info = {
            "Column": col,
            "Data Type": dtype_str,
            "Missing": missing_cnt,
            "Missing %": f"{missing_p}%",
            "Unique": unique_cnt,
            "Mean": None,
            "Median": None,
            "Min": None,
            "Max": None,
            "Std": None,
        }

        if col in numeric_cols and s.dropna().shape[0] > 0:
            info["Mean"] = round(float(s.mean()), 2)
            info["Median"] = round(float(s.median()), 2)
            info["Min"] = round(float(s.min()), 2)
            info["Max"] = round(float(s.max()), 2)
            info["Std"] = round(float(s.std()), 2) if len(s.dropna()) > 1 else 0.0

        col_summary.append(info)

    col_summary_df = pd.DataFrame(col_summary)

    # Descriptive statistics dataframe for numeric columns
    numeric_summary = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()

    # Data Quality Score
    quality_info = compute_data_quality_score(df)

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_cols": date_cols,
        "column_info_table": col_summary_df,
        "numeric_summary": numeric_summary,
        "quality_info": quality_info,
    }
