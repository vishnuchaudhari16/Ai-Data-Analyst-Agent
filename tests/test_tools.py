"""
Unit tests for Pandas, Statistics, Anomaly, and ML tools.
"""

import pytest
import pandas as pd
import numpy as np

from tools.pandas_tool import groupby_agg, filter_rows, top_n, pivot_table, correlation_matrix
from tools.statistics_tool import run_statistical_test
from tools.anomaly_tool import detect_anomalies
from tools.ml_tool import train_predictive_model


@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        "Department": ["Sales", "Sales", "Engineering", "Engineering", "HR", "HR", "Sales", "Engineering"],
        "Salary": [70000, 80000, 120000, 130000, 60000, 65000, 75000, 125000],
        "Experience": [2, 4, 8, 10, 3, 5, 3, 9],
        "Performance_Score": [75.0, 80.0, 95.0, 98.0, 70.0, 72.0, 78.0, 92.0]
    })


def test_groupby_agg(sample_df):
    res = groupby_agg(sample_df, group_col="Department", value_col="Salary", agg_func="mean")
    assert res["status"] == "success"
    data = res["data"]
    assert len(data) == 3
    assert "Salary_mean" in data.columns
    # Engineering mean should be (120k + 130k + 125k) / 3 = 125000
    eng_mean = data[data["Department"] == "Engineering"]["Salary_mean"].values[0]
    assert eng_mean == 125000.0


def test_filter_rows(sample_df):
    res = filter_rows(sample_df, column="Salary", operator=">", value=100000)
    assert res["status"] == "success"
    data = res["data"]
    assert len(data) == 3
    assert (data["Salary"] > 100000).all()


def test_top_n(sample_df):
    res = top_n(sample_df, column="Salary", n=3, ascending=False)
    assert res["status"] == "success"
    data = res["data"]
    assert len(data) == 3
    assert data.iloc[0]["Salary"] == 130000


def test_correlation_matrix(sample_df):
    res = correlation_matrix(sample_df)
    assert res["status"] == "success"
    data = res["data"]
    assert "Salary" in data.columns


def test_statistical_test(sample_df):
    res = run_statistical_test(sample_df, test_type="descriptive", col1="Salary")
    assert res["status"] == "success"
    data = res["data"]
    assert data.iloc[0]["Mean"] == 90625.0


def test_detect_anomalies(sample_df):
    # Add an outlier
    outlier_df = sample_df.copy()
    outlier_df.loc[0, "Salary"] = 5000000  # Extreme outlier
    res = detect_anomalies(outlier_df, column="Salary", method="iqr")
    assert res["status"] == "success"
    stats = res["summary_stats"]
    assert stats["anomalies_detected"] >= 1


def test_train_predictive_model(sample_df):
    # Expand dataset to have >= 20 rows for ML training
    expanded_df = pd.concat([sample_df] * 4, ignore_index=True)
    res = train_predictive_model(
        expanded_df,
        task_type="regression",
        target_col="Salary",
        feature_cols=["Experience", "Performance_Score"]
    )
    assert res["status"] == "success"
    metrics = res["data"]
    assert "MAE" in metrics.columns

