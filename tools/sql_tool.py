"""
SQL Tool for AI Data Analyst Agent.
Executes DuckDB queries against a registered read-only 'dataset' view.
Uses sqlglot AST validation to strictly enforce a single SELECT/WITH allowlist policy.
"""

from typing import Dict, Any, List, Optional, Tuple
import duckdb
import pandas as pd
import sqlglot
import sqlglot.expressions as exp

from tools.registry import register_tool, SQLQueryArgs

# Dangerous SQL AST nodes and functions to reject
DISALLOWED_AST_NODES = (
    exp.Drop, exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Alter,
    exp.Command, exp.Pragma, exp.Copy
)

DISALLOWED_FUNCTIONS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_text",
    "scan_location", "sqlite_scan", "postgres_scan", "attach", "detach"
}


def validate_sql_query_ast(query: str) -> Tuple[bool, str]:
    """
    Parse SQL query using sqlglot and verify AST satisfies security allowlist rules:
    1. Must be a single statement.
    2. Must be SELECT or WITH ... SELECT.
    3. Must only reference the 'dataset' view.
    4. Must not contain prohibited statements or filesystem functions.
    """
    cleaned_query = query.strip().rstrip(";")
    if ";" in cleaned_query:
        return False, "Multiple SQL statements are not permitted."

    # Direct keyword check for dangerous statements
    query_upper = cleaned_query.upper()
    for kw in ["ATTACH ", "DETACH ", "DROP ", "ALTER ", "COPY ", "PRAGMA "]:
        if kw in query_upper or query_upper.startswith(kw.strip()):
            return False, f"Disallowed SQL statement detected: {kw.strip()}."

    try:
        parsed_statements = sqlglot.parse(cleaned_query, read="duckdb")
        if len(parsed_statements) != 1:
            return False, "Only a single SELECT statement is permitted."

        stmt = parsed_statements[0]
        if stmt is None:
            return False, "Failed to parse SQL query."

        # Verify query type is SELECT or WITH
        if not isinstance(stmt, (exp.Select, exp.Expression)):
            return False, f"Only SELECT queries are allowed. Got {type(stmt).__name__}."

        # Check for disallowed AST nodes recursively
        for node in stmt.walk():
            if isinstance(node, DISALLOWED_AST_NODES):
                return False, f"Disallowed SQL statement detected: {type(node).__name__}."
            
            # Check function calls for filesystem functions
            fn_name = ""
            if isinstance(node, exp.Anonymous):
                fn_name = str(node.this).lower() if hasattr(node, "this") else ""
            elif isinstance(node, exp.Func):
                fn_name = str(node.key).lower() if hasattr(node, "key") else ""

            if fn_name and fn_name in DISALLOWED_FUNCTIONS:
                return False, f"Disallowed function call detected: {fn_name}."

            # Check table references
            if isinstance(node, exp.Table):
                table_name = str(node.this.this).lower() if hasattr(node.this, "this") else str(node.this).lower()
                if table_name != "dataset":
                    return False, f"Access denied to table/view '{table_name}'. Only 'dataset' is permitted."

        return True, "Valid SQL query"

    except Exception as e:
        return False, f"SQL syntax parsing error: {str(e)}"



def build_sql_from_args(
    select_cols: List[str],
    group_by_cols: Optional[List[str]] = None,
    agg_exprs: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
    order_by: Optional[str] = None,
    limit: int = 50
) -> str:
    """Build a safe SQL query string from validated structured arguments."""
    select_parts = []
    
    if group_by_cols:
        select_parts.extend(group_by_cols)
    
    if agg_exprs:
        select_parts.extend(agg_exprs)
    elif not select_parts:
        select_parts = select_cols if select_cols else ["*"]

    sql = f"SELECT {', '.join(select_parts)} FROM dataset"

    if where_clause:
        sql += f" WHERE {where_clause}"

    if group_by_cols:
        sql += f" GROUP BY {', '.join(group_by_cols)}"

    if order_by:
        sql += f" ORDER BY {order_by}"

    sql += f" LIMIT {min(max(1, limit), 500)}"
    return sql


@register_tool(
    name="sql_query",
    description="Constructs and executes an allowlisted DuckDB SQL query against the 'dataset' view using structured parameters.",
    schema_model=SQLQueryArgs,
)
def execute_sql_query(
    df: pd.DataFrame,
    select_cols: List[str] = ["*"],
    group_by_cols: Optional[List[str]] = None,
    agg_exprs: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
    order_by: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Execute allowlisted DuckDB query on dataset."""
    if df is None or df.empty:
        return {"status": "error", "message": "Dataset is empty."}

    sql_str = build_sql_from_args(select_cols, group_by_cols, agg_exprs, where_clause, order_by, limit)
    is_valid, err_msg = validate_sql_query_ast(sql_str)

    if not is_valid:
        return {"status": "error", "message": f"SQL Security Violation: {err_msg}"}

    try:
        con = duckdb.connect(database=":memory:")
        con.register("dataset", df)
        result_df = con.execute(sql_str).df()
        con.close()

        return {
            "status": "success",
            "data": result_df,
            "summary_stats": {"rows_returned": len(result_df), "query": sql_str},
            "code_display": f"-- Executed SQL Query\n{sql_str}",
            "result_type": "sql",
        }
    except Exception as e:
        return {"status": "error", "message": f"DuckDB execution error: {str(e)}"}
