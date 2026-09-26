"""
Anomaly Detection Tool.
Detects outliers using IQR, Z-Score, and Isolation Forest algorithms.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from tools.registry import register_tool, AnomalyArgs


@register_tool(
    name="detect_anomalies",
    description="Detects anomalies/outliers in a numeric column using IQR, Z-Score, or Isolation Forest.",
    schema_model=AnomalyArgs,
)
def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5
) -> Dict[str, Any]:
    """Detect anomalies in dataset."""
    if column not in df.columns:
        return {"status": "error", "message": f"Column '{column}' not found in dataset."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"status": "error", "message": f"Column '{column}' must be numerical for anomaly detection."}

    df_clean = df.copy()
    s = df_clean[column]
    total_records = len(s.dropna())

    if total_records < 5:
        return {"status": "error", "message": "At least 5 valid numeric rows are required for anomaly detection."}

    method = method.lower().strip()
    is_anomaly = pd.Series(False, index=df_clean.index)

    try:
        # 1. IQR Method
        if method == "iqr":
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
            is_anomaly = (s < lower_bound) | (s > upper_bound)
            method_desc = f"IQR (Factor {threshold}, Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f})"

        # 2. Z-Score Method
        elif method == "zscore":
            mean_val = s.mean()
            std_val = s.std()
            if std_val == 0:
                is_anomaly = pd.Series(False, index=df_clean.index)
            else:
                z_scores = np.abs((s - mean_val) / std_val)
                is_anomaly = z_scores > threshold
            method_desc = f"Z-Score (|Z| > {threshold})"

        # 3. Isolation Forest Method
        elif method == "isolation_forest":
            iso = IsolationForest(contamination="auto", random_state=42)
            valid_mask = s.notna()
            preds = iso.fit_predict(s.dropna().values.reshape(-1, 1))
            is_anomaly.loc[valid_mask] = preds == -1
            method_desc = "Isolation Forest (scikit-learn)"

        else:
            return {"status": "error", "message": f"Unknown anomaly detection method '{method}'."}

        anomalous_df = df_clean[is_anomaly].copy()
        anomalies_cnt = len(anomalous_df)
        anomaly_rate = round((anomalies_cnt / total_records) * 100, 2) if total_records > 0 else 0.0

        summary_stats = {
            "total_records": total_records,
            "anomalies_detected": anomalies_cnt,
            "anomaly_rate": anomaly_rate,
            "method": method_desc,
            "target_column": column,
        }

        code_snippet = f"""# Anomaly Detection ({method_desc})
# Detected {anomalies_cnt} anomalies ({anomaly_rate}%) in column '{column}'"""

        return {
            "status": "success",
            "data": anomalous_df.head(100),  # Limit displayed anomalous rows
            "summary_stats": summary_stats,
            "code_display": code_snippet,
            "result_type": "anomaly",
        }

    except Exception as e:
        return {"status": "error", "message": f"Anomaly detection failed: {str(e)}"}
