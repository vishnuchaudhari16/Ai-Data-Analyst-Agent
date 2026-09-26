"""
Machine Learning and Time-Series Forecasting Tool.
Supports Regression, Classification, Feature Importance, and Holt-Winters / ARIMA Forecasting.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tools.registry import register_tool, MLArgs


@register_tool(
    name="build_predictive_model",
    description="Builds machine learning models (Regression/Classification) or time-series forecast on the dataset.",
    schema_model=MLArgs,
)
def train_predictive_model(
    df: pd.DataFrame,
    task_type: str,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    model_name: str = "random_forest"
) -> Dict[str, Any]:
    """Train ML model and evaluate metrics."""
    if target_col not in df.columns:
        return {"status": "error", "message": f"Target column '{target_col}' not found in dataset."}

    df_clean = df.copy()

    # Pre-process features
    if not feature_cols:
        feature_cols = [c for c in df_clean.columns if c != target_col]
    else:
        feature_cols = [c for c in feature_cols if c in df_clean.columns and c != target_col]

    if not feature_cols:
        return {"status": "error", "message": "No valid feature columns provided for model training."}

    # Drop missing target rows
    df_clean = df_clean.dropna(subset=[target_col])
    if len(df_clean) < 20:
        return {"status": "error", "message": "At least 20 valid non-null rows are required to train a machine learning model."}

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    # Encode categorical features
    label_encoders = {}
    for c in X.columns:
        if X[c].dtype == "object" or isinstance(X[c].dtype, pd.CategoricalDtype):
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            label_encoders[c] = le

    # Impute missing feature values with median/mode
    for c in X.columns:
        if X[c].isna().any():
            if pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].fillna(X[c].median())
            else:
                X[c] = X[c].fillna(X[c].mode()[0] if not X[c].mode().empty else 0)

    task_type = task_type.lower().strip()

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        feature_importances = []

        # -------------------------------------------------------------------
        # 1. Regression Task
        # -------------------------------------------------------------------
        if task_type == "regression":
            if model_name == "linear":
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                coeffs = np.abs(model.coef_)
                feature_importances = sorted(zip(feature_cols, coeffs), key=lambda x: x[1], reverse=True)
            elif model_name == "gradient_boosting":
                model = GradientBoostingRegressor(random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                feature_importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
            else:
                model = RandomForestRegressor(random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                feature_importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)

            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = float(r2_score(y_test, preds))

            metrics_df = pd.DataFrame([{
                "Task": "Regression",
                "Model": model_name.replace("_", " ").title(),
                "Target Variable": target_col,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R² Score": round(r2, 4),
                "Test Set Rows": len(y_test),
            }])

            fi_df = pd.DataFrame(feature_importances, columns=["Feature", "Importance Score"]).head(10)

            return {
                "status": "success",
                "data": metrics_df,
                "feature_importance": fi_df,
                "summary_stats": {"r2": r2, "mae": mae, "rmse": rmse, "task_type": "regression"},
                "code_display": f"# ML Regression ({model_name})\nmodel = RandomForestRegressor().fit(X_train, y_train)\npreds = model.predict(X_test)",
                "result_type": "ml",
            }

        # -------------------------------------------------------------------
        # 2. Classification Task
        # -------------------------------------------------------------------
        elif task_type == "classification":
            # Target encoding if object
            if y.dtype == "object":
                target_le = LabelEncoder()
                y_train = target_le.fit_transform(y_train.astype(str))
                y_test = target_le.transform(y_test.astype(str))

            if model_name == "logistic":
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
            elif model_name == "decision_tree":
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                feature_importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
            elif model_name == "gradient_boosting":
                model = GradientBoostingClassifier(random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                feature_importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
            else:
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                feature_importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)

            acc = float(accuracy_score(y_test, preds))
            prec = float(precision_score(y_test, preds, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, preds, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))

            metrics_df = pd.DataFrame([{
                "Task": "Classification",
                "Model": model_name.replace("_", " ").title(),
                "Target Variable": target_col,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4),
                "Test Set Rows": len(y_test),
            }])

            fi_df = pd.DataFrame(feature_importances, columns=["Feature", "Importance Score"]).head(10) if feature_importances else pd.DataFrame()

            return {
                "status": "success",
                "data": metrics_df,
                "feature_importance": fi_df,
                "summary_stats": {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "task_type": "classification"},
                "code_display": f"# ML Classification ({model_name})\nmodel = RandomForestClassifier().fit(X_train, y_train)\nacc = accuracy_score(y_test, preds)",
                "result_type": "ml",
            }

        else:
            return {"status": "error", "message": f"Unsupported ML task type '{task_type}'."}

    except Exception as e:
        return {"status": "error", "message": f"ML model training error: {str(e)}"}


def generate_time_series_forecast(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int = 6
) -> Dict[str, Any]:
    """Generate time-series forecast using statsmodels Holt-Winters Exponential Smoothing."""
    if date_col not in df.columns or value_col not in df.columns:
        return {"status": "error", "message": f"Columns '{date_col}' or '{value_col}' not found."}

    try:
        ts_df = df[[date_col, value_col]].dropna().copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
        ts_df = ts_df.dropna(subset=[date_col]).sort_values(by=date_col)

        if len(ts_df) < 5:
            return {"status": "error", "message": "At least 5 historical time points are required for time-series forecasting."}

        # Aggregate values by date
        aggregated_ts = ts_df.groupby(date_col)[value_col].sum().reset_index()

        # Fit Exponential Smoothing
        model = ExponentialSmoothing(aggregated_ts[value_col].values, trend="add", seasonal=None, initialization_method="estimated")
        fit_model = model.fit()

        forecast_vals = fit_model.forecast(periods)

        # Generate future dates
        last_date = aggregated_ts[date_col].max()
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            f"Forecasted_{value_col}": np.round(forecast_vals, 2)
        })

        return {
            "status": "success",
            "historical_data": aggregated_ts,
            "forecast_data": forecast_df,
            "summary_stats": {"periods": periods, "last_historical_date": str(last_date.date())},
            "code_display": f"# Holt-Winters Forecasting\nmodel = ExponentialSmoothing(data, trend='add').fit()\nforecast = model.forecast(periods={periods})",
        }
    except Exception as e:
        return {"status": "error", "message": f"Time-series forecasting failed: {str(e)}"}
