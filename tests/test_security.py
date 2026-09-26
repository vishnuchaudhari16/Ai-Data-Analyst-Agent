"""
Security unit tests for SQL allowlist AST parser and Pydantic argument validation.
"""

import pytest
from tools.sql_tool import validate_sql_query_ast
from utils.security import validate_tool_args
from tools.registry import GroupbyAggArgs


def test_sql_ast_valid_query():
    valid, msg = validate_sql_query_ast("SELECT Department, AVG(Salary) FROM dataset GROUP BY Department")
    assert valid is True


def test_sql_ast_reject_drop_table():
    valid, msg = validate_sql_query_ast("DROP TABLE dataset;")
    assert valid is False
    assert "Disallowed" in msg or "Only SELECT" in msg


def test_sql_ast_reject_multiple_statements():
    valid, msg = validate_sql_query_ast("SELECT * FROM dataset; SELECT * FROM dataset;")
    assert valid is False
    assert "Multiple SQL statements" in msg or "Only a single SELECT" in msg


def test_sql_ast_reject_filesystem_functions():
    valid, msg = validate_sql_query_ast("SELECT * FROM read_csv('/etc/passwd');")
    assert valid is False
    assert "Disallowed function" in msg or "Access denied" in msg


test_sql_ast_reject_attach_cases = [
    "ATTACH 'database.db' AS db;",
    "SELECT * FROM dataset; ATTACH 'db.db';",
]


@pytest.mark.parametrize("query", test_sql_ast_reject_attach_cases)
def test_sql_ast_reject_attach(query):
    valid, msg = validate_sql_query_ast(query)
    assert valid is False


def test_pydantic_arg_validation_success():
    args = {"group_col": "Department", "value_col": "Salary", "agg_func": "mean"}
    valid, obj, err = validate_tool_args(GroupbyAggArgs, args)
    assert valid is True
    assert obj.group_col == "Department"


def test_pydantic_arg_validation_missing_required():
    args = {"group_col": "Department"}  # missing value_col
    valid, obj, err = validate_tool_args(GroupbyAggArgs, args)
    assert valid is False
    assert "value_col" in err
