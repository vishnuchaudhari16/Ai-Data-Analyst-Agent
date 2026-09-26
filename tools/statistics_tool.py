"""
Statistics Tool.
Performs statistical hypothesis tests (T-Test, Chi-Square, ANOVA, Correlation) with applicability checks.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from tools.registry import register_tool, StatisticsArgs


@register_tool(
    name="statistical_test",
    description="Performs statistical tests (t_test, chi_square, anova, correlation, descriptive) with applicability guards.",
    schema_model=StatisticsArgs,
)
def run_statistical_test(
    df: pd.DataFrame,
    test_type: str,
    col1: str,
    col2: Optional[str] = None
) -> Dict[str, Any]:
    """Execute statistical hypothesis tests safely."""
    if col1 not in df.columns:
        return {"status": "error", "message": f"Column '{col1}' not found in dataset."}
    if col2 and col2 not in df.columns:
        return {"status": "error", "message": f"Column '{col2}' not found in dataset."}

    test_type = test_type.lower().strip()

    try:
        # 1. Descriptive Statistics
        if test_type == "descriptive":
            s = df[col1].dropna()
            if not pd.api.types.is_numeric_dtype(s):
                return {"status": "error", "message": f"Descriptive stats require numeric column, got {s.dtype}."}
            
            res_df = pd.DataFrame([{
                "Metric": col1,
                "Mean": float(s.mean()),
                "Std Dev": float(s.std()),
                "Variance": float(s.var()),
                "Median": float(s.median()),
                "25th %": float(s.quantile(0.25)),
                "75th %": float(s.quantile(0.75)),
                "IQR": float(s.quantile(0.75) - s.quantile(0.25)),
                "Skewness": float(s.skew()),
                "Kurtosis": float(s.kurtosis()),
            }])

            return {
                "status": "success",
                "data": res_df,
                "summary_stats": {"test_type": "descriptive", "column": col1},
                "code_display": f"# Descriptive Stats\nstats_df = df['{col1}'].describe()",
                "result_type": "stats",
            }

        # 2. Correlation Test (Pearson & Spearman)
        elif test_type == "correlation":
            if not col2:
                return {"status": "error", "message": "Correlation test requires a second column (col2)."}
            s1, s2 = df[col1], df[col2]
            if not (pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2)):
                return {"status": "error", "message": f"Both columns must be numerical for correlation test."}

            clean_df = df[[col1, col2]].dropna()
            if len(clean_df) < 5:
                return {"status": "error", "message": "At least 5 paired non-null observations are required for correlation test."}

            pearson_r, pearson_p = stats.pearsonr(clean_df[col1], clean_df[col2])
            spearman_r, spearman_p = stats.spearmanr(clean_df[col1], clean_df[col2])

            res_df = pd.DataFrame([
                {"Method": "Pearson (Linear)", "Correlation Coefficient": round(float(pearson_r), 4), "p-value": float(pearson_p), "Significant (p < 0.05)": pearson_p < 0.05},
                {"Method": "Spearman (Rank)", "Correlation Coefficient": round(float(spearman_r), 4), "p-value": float(spearman_p), "Significant (p < 0.05)": spearman_p < 0.05},
            ])

            return {
                "status": "success",
                "data": res_df,
                "summary_stats": {"col1": col1, "col2": col2, "pearson_r": pearson_r, "pearson_p": pearson_p},
                "code_display": f"# Correlation Test\nr, p_val = scipy.stats.pearsonr(df['{col1}'], df['{col2}'])",
                "result_type": "stats",
            }

        # 3. Two-Sample Independent T-Test
        elif test_type == "t_test":
            if not col2:
                return {"status": "error", "message": "T-Test requires a numeric metric (col1) and a binary group column (col2)."}
            
            numeric_col = col1 if pd.api.types.is_numeric_dtype(df[col1]) else (col2 if pd.api.types.is_numeric_dtype(df[col2]) else None)
            group_col = col2 if numeric_col == col1 else col1

            if not numeric_col or pd.api.types.is_numeric_dtype(df[group_col]):
                return {"status": "error", "message": "T-Test requires exactly one numeric metric and one categorical grouping column."}

            groups = df[group_col].dropna().unique()
            if len(groups) != 2:
                return {"status": "error", "message": f"Independent T-Test requires exactly 2 groups in '{group_col}'. Found {len(groups)} groups: {groups[:5]}."}

            g1_vals = df[df[group_col] == groups[0]][numeric_col].dropna()
            g2_vals = df[df[group_col] == groups[1]][numeric_col].dropna()

            if len(g1_vals) < 3 or len(g2_vals) < 3:
                return {"status": "error", "message": "Each group must have at least 3 non-null observations for T-Test."}

            t_stat, p_val = stats.ttest_ind(g1_vals, g2_vals, equal_var=False)

            res_df = pd.DataFrame([{
                "Test": "Independent Welch's T-Test",
                "Group Column": group_col,
                "Group 1": str(groups[0]),
                "Group 1 Mean": round(float(g1_vals.mean()), 2),
                "Group 2": str(groups[1]),
                "Group 2 Mean": round(float(g2_vals.mean()), 2),
                "T-Statistic": round(float(t_stat), 4),
                "p-value": float(p_val),
                "Statistically Significant (p < 0.05)": p_val < 0.05,
            }])

            return {
                "status": "success",
                "data": res_df,
                "summary_stats": {"t_stat": t_stat, "p_val": p_val, "significant": p_val < 0.05},
                "code_display": f"# Independent T-Test\nt_stat, p_val = scipy.stats.ttest_ind(group1_data, group2_data, equal_var=False)",
                "result_type": "stats",
            }

        # 4. Chi-Square Test of Independence
        elif test_type == "chi_square":
            if not col2:
                return {"status": "error", "message": "Chi-Square test requires two categorical columns."}
            
            contingency_table = pd.crosstab(df[col1], df[col2])
            if contingency_table.size == 0 or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                return {"status": "error", "message": "Chi-Square test requires contingency table of at least 2x2 dimensions."}

            chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)

            res_df = pd.DataFrame([{
                "Test": "Chi-Square Independence Test",
                "Variable 1": col1,
                "Variable 2": col2,
                "Chi-Square Stat": round(float(chi2), 4),
                "Degrees of Freedom": dof,
                "p-value": float(p_val),
                "Significant (p < 0.05)": p_val < 0.05,
            }])

            return {
                "status": "success",
                "data": res_df,
                "summary_stats": {"chi2": chi2, "p_val": p_val, "significant": p_val < 0.05},
                "code_display": f"# Chi-Square Test\nchi2, p, dof, ex = scipy.stats.chi2_contingency(pd.crosstab(df['{col1}'], df['{col2}']))",
                "result_type": "stats",
            }

        # 5. One-Way ANOVA
        elif test_type == "anova":
            if not col2:
                return {"status": "error", "message": "ANOVA requires a numeric metric (col1) and a categorical grouping column (col2)."}
            
            numeric_col = col1 if pd.api.types.is_numeric_dtype(df[col1]) else (col2 if pd.api.types.is_numeric_dtype(df[col2]) else None)
            group_col = col2 if numeric_col == col1 else col1

            if not numeric_col or pd.api.types.is_numeric_dtype(df[group_col]):
                return {"status": "error", "message": "ANOVA requires exactly one numeric metric and one categorical grouping column."}

            groups = [group[numeric_col].dropna() for _, group in df.groupby(group_col) if len(group[numeric_col].dropna()) >= 2]

            if len(groups) < 2:
                return {"status": "error", "message": "ANOVA requires at least 2 distinct groups with valid data."}

            f_stat, p_val = stats.f_oneway(*groups)

            res_df = pd.DataFrame([{
                "Test": "One-Way ANOVA",
                "Metric Column": numeric_col,
                "Group Column": group_col,
                "Group Count": len(groups),
                "F-Statistic": round(float(f_stat), 4),
                "p-value": float(p_val),
                "Statistically Significant (p < 0.05)": p_val < 0.05,
            }])

            return {
                "status": "success",
                "data": res_df,
                "summary_stats": {"f_stat": f_stat, "p_val": p_val, "significant": p_val < 0.05},
                "code_display": f"# One-Way ANOVA\nf_stat, p_val = scipy.stats.f_oneway(*group_list)",
                "result_type": "stats",
            }

        else:
            return {"status": "error", "message": f"Unsupported statistical test: {test_type}"}

    except Exception as e:
        return {"status": "error", "message": f"Statistical test execution failed: {str(e)}"}
