"""
Unit tests for data profiler and data quality score engine.
"""

import pytest
import pandas as pd
from tools.data_profiler import profile_dataset, compute_data_quality_score


def test_compute_data_quality_score_perfect():
    df = pd.DataFrame({
        "ID": [1, 2, 3, 4, 5],
        "Name": ["A", "B", "C", "D", "E"],
        "Value": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    res = compute_data_quality_score(df)
    assert res["score"] >= 85
    assert res["label"] in ["Excellent", "Good"]


def test_compute_data_quality_score_poor():
    # DataFrame with missing values and duplicates
    df = pd.DataFrame({
        "A": [None, None, None, 1, 1],
        "B": [None, None, None, 1, 1],
        "C": [1, 1, 1, 1, 1]  # Constant column
    })
    res = compute_data_quality_score(df)
    assert res["score"] < 70
    assert "breakdown" in res


def test_profile_dataset():
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40],
        "City": ["NY", "LA", "SF", "CHI"],
        "Join_Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    })
    profile = profile_dataset(df)
    assert profile["total_rows"] == 4
    assert profile["total_cols"] == 3
    assert len(profile["numeric_cols"]) == 1
    assert len(profile["date_cols"]) == 1
