"""
SQL Analyst UI component.
Interactive SQL interface powered by DuckDB and secured via sqlglot AST allowlist verification.
"""

import streamlit as st
import pandas as pd
from tools.sql_tool import execute_sql_query, validate_sql_query_ast


def render_sql_analyst(df: pd.DataFrame, meta: dict):
    """Render SQL Analyst page."""
    st.markdown("## ⚡ SQL Analyst (DuckDB)")
    st.markdown("Query your uploaded dataset using structured parameters or custom SELECT queries against the read-only `dataset` view.")

    if df is None or df.empty:
        st.warning("⚠️ Please upload a dataset to run SQL queries.")
        return

    st.info("🔒 **Security Note**: All queries are restricted to single `SELECT`/`WITH` statements against the `dataset` view. Administrative operations and filesystem access are strictly blocked.")

    tab_builder, tab_custom = st.tabs(["🎛️ SQL Query Builder", "💻 Custom SQL Editor"])

    with tab_builder:
        col1, col2 = st.columns(2)
        with col1:
            group_cols = st.multiselect("Group By Columns:", options=df.columns)
            select_cols = st.multiselect("Select Columns:", options=df.columns, default=list(df.columns[:3]))
        with col2:
            num_cols = list(df.select_dtypes(include=["number"]).columns)
            agg_col = st.selectbox("Aggregate Column:", options=num_cols if num_cols else df.columns)
            agg_func = st.selectbox("Aggregate Function:", options=["AVG", "SUM", "COUNT", "MAX", "MIN"])
            limit_val = st.number_input("Limit Rows:", min_value=1, max_value=500, value=50)

        agg_exprs = [f"{agg_func}({agg_col}) AS {agg_func.lower()}_{agg_col}"] if agg_col else None

        if st.button("Run Query Builder", use_container_width=True):
            res = execute_sql_query(
                df,
                select_cols=select_cols or ["*"],
                group_by_cols=group_cols,
                agg_exprs=agg_exprs if group_cols else None,
                limit=limit_val
            )
            if res.get("status") == "success":
                st.success(f"Query returned {len(res.get('data'))} rows.")
                st.dataframe(res.get("data"), use_container_width=True)
                with st.expander("🔧 Generated SQL Query", expanded=True):
                    st.code(res.get("code_display"), language="sql")
            else:
                st.error(res.get("message"))

    with tab_custom:
        default_query = f"SELECT * FROM dataset LIMIT 10;"
        custom_sql = st.text_area("Enter SQL Query (against 'dataset' view):", value=default_query, height=120)

        if st.button("Execute Custom SQL", use_container_width=True):
            is_valid, err_msg = validate_sql_query_ast(custom_sql)
            if not is_valid:
                st.error(f"⚠️ SQL Security Policy Violation: {err_msg}")
            else:
                try:
                    import duckdb
                    con = duckdb.connect(database=":memory:")
                    con.register("dataset", df)
                    res_df = con.execute(custom_sql.strip().rstrip(";")).df()
                    con.close()
                    st.success(f"Query executed successfully ({len(res_df)} rows).")
                    st.dataframe(res_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
