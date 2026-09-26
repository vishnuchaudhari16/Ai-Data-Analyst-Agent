"""
Tool Registry module.
Defines Pydantic schemas for all tools callable by the AI Agent and registers them in a centralized catalog.
"""

from typing import Dict, Any, List, Optional, Type, Callable
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schemas for Tool Inputs
# ---------------------------------------------------------------------------

class GroupbyAggArgs(BaseModel):
    group_col: str = Field(..., description="Column name to group by")
    value_col: str = Field(..., description="Numeric column name to aggregate")
    agg_func: str = Field("mean", description="Aggregation function: 'mean', 'sum', 'count', 'min', 'max', 'std', 'median'")
    top_n: Optional[int] = Field(10, description="Limit output to top N rows")
    sort_ascending: bool = Field(False, description="Sort order: False for descending (highest first)")


class FilterRowsArgs(BaseModel):
    column: str = Field(..., description="Column to filter on")
    operator: str = Field("==", description="Comparison operator: '==', '!=', '>', '>=', '<', '<=', 'contains', 'in'")
    value: Any = Field(..., description="Value to compare against")


class TopNArgs(BaseModel):
    column: str = Field(..., description="Numeric column to sort by")
    n: int = Field(10, description="Number of top rows to return")
    ascending: bool = Field(False, description="Sort order: False for highest first")


class PivotTableArgs(BaseModel):
    index_col: str = Field(..., description="Column to use as pivot table index (rows)")
    columns_col: str = Field(..., description="Column to use as pivot table columns")
    values_col: str = Field(..., description="Numeric column to aggregate")
    agg_func: str = Field("mean", description="Aggregation function: 'mean', 'sum', 'count'")


class CorrelationArgs(BaseModel):
    columns: Optional[List[str]] = Field(None, description="Optional list of numeric columns. If None, uses all numeric columns.")


class SQLQueryArgs(BaseModel):
    select_cols: List[str] = Field(default_factory=lambda: ["*"], description="Columns to select")
    group_by_cols: Optional[List[str]] = Field(None, description="Columns to GROUP BY")
    agg_exprs: Optional[List[str]] = Field(None, description="Aggregations like 'AVG(Salary) AS avg_sal'")
    where_clause: Optional[str] = Field(None, description="Safe WHERE filter condition e.g. \"Region = 'North'\"")
    order_by: Optional[str] = Field(None, description="ORDER BY clause e.g. 'avg_sal DESC'")
    limit: int = Field(50, description="LIMIT clause")


class StatisticsArgs(BaseModel):
    test_type: str = Field(..., description="Type of statistical test: 'descriptive', 'correlation', 't_test', 'chi_square', 'anova'")
    col1: str = Field(..., description="Primary column name")
    col2: Optional[str] = Field(None, description="Secondary column name (for bivariate tests)")


class AnomalyArgs(BaseModel):
    column: str = Field(..., description="Numeric column to check for anomalies")
    method: str = Field("iqr", description="Detection method: 'iqr', 'zscore', 'isolation_forest'")
    threshold: float = Field(1.5, description="Threshold multiplier (e.g. 1.5 for IQR, 3.0 for Z-Score)")


class MLArgs(BaseModel):
    task_type: str = Field(..., description="Task type: 'regression' or 'classification'")
    target_col: str = Field(..., description="Target variable column name")
    feature_cols: Optional[List[str]] = Field(None, description="Feature columns list. If None, uses all remaining features.")
    model_name: str = Field("random_forest", description="Model algorithm: 'random_forest', 'linear', 'gradient_boosting', 'logistic'")


# ---------------------------------------------------------------------------
# Tool Registry Structure
# ---------------------------------------------------------------------------

class ToolEntry:
    def __init__(self, name: str, description: str, schema_model: Type[BaseModel], func: Callable):
        self.name = name
        self.description = description
        self.schema_model = schema_model
        self.func = func


TOOL_REGISTRY: Dict[str, ToolEntry] = {}


def register_tool(name: str, description: str, schema_model: Type[BaseModel]):
    """Decorator to register a tool function in the TOOL_REGISTRY catalog."""
    def decorator(func: Callable):
        TOOL_REGISTRY[name] = ToolEntry(
            name=name,
            description=description,
            schema_model=schema_model,
            func=func
        )
        return func
    return decorator


def get_tool_catalog_summary() -> List[Dict[str, str]]:
    """Return a list of tool names and descriptions for the LLM Planner prompt."""
    catalog = []
    for name, tool in TOOL_REGISTRY.items():
        # Represent pydantic fields
        fields_info = []
        for fname, ffield in tool.schema_model.model_fields.items():
            fields_info.append(f"{fname}: {ffield.annotation.__name__ if hasattr(ffield.annotation, '__name__') else str(ffield.annotation)} ({ffield.description})")
        
        catalog.append({
            "name": name,
            "description": tool.description,
            "args_schema": ", ".join(fields_info)
        })
    return catalog
