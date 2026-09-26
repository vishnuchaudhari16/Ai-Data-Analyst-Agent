"""
Planner Agent.
Parses user intent, maps column references, selects appropriate tool from TOOL_REGISTRY,
and outputs structured JSON arguments.
Includes deterministic rule-based fallback when no LLM API key is configured.
"""

import json
import re
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd

from config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL
from tools.registry import get_tool_catalog_summary, TOOL_REGISTRY
from utils.security import validate_tool_args
from utils.validators import fuzzy_match_columns


def rule_based_planner(question: str, df: pd.DataFrame, resolved_cols: Dict[str, str]) -> Dict[str, Any]:
    """
    Deterministic rule-based planner used when LLM API key is absent or on API failure.
    """
    q_lower = question.lower()
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    date_cols = list(df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns)

    # 1. Correlation query
    if "correlation" in q_lower or "relationship" in q_lower or "correlated" in q_lower:
        return {"tool": "correlation_matrix", "args": {"columns": num_cols}}

    # 2. Anomaly query
    if "anomaly" in q_lower or "outlier" in q_lower or "abnormal" in q_lower:
        target_num = num_cols[0] if num_cols else df.columns[0]
        for col in num_cols:
            if col.lower() in q_lower:
                target_num = col
                break
        return {"tool": "detect_anomalies", "args": {"column": target_num, "method": "iqr", "threshold": 1.5}}

    # 3. Top N query
    if "top" in q_lower or "highest" in q_lower or "best" in q_lower or "largest" in q_lower:
        # Check if there is a group column mentioned
        group_candidate = None
        for col in cat_cols:
            if col.lower() in q_lower or col in resolved_cols.values():
                group_candidate = col
                break
        
        val_candidate = num_cols[0] if num_cols else df.columns[0]
        for col in num_cols:
            if col.lower() in q_lower or col in resolved_cols.values():
                val_candidate = col
                break

        if group_candidate:
            return {
                "tool": "groupby_agg",
                "args": {
                    "group_col": group_candidate,
                    "value_col": val_candidate,
                    "agg_func": "mean" if "average" in q_lower or "avg" in q_lower else "sum",
                    "top_n": 10,
                    "sort_ascending": False
                }
            }
        else:
            return {
                "tool": "top_n",
                "args": {"column": val_candidate, "n": 10, "ascending": False}
            }

    # 4. Groupby Aggregation query (e.g. "average salary by department")
    if any(k in q_lower for k in ["by", "average", "avg", "total", "sum", "count", "mean"]):
        group_candidate = cat_cols[0] if cat_cols else df.columns[0]
        for col in cat_cols:
            if col.lower() in q_lower or col in resolved_cols.values():
                group_candidate = col
                break
        
        val_candidate = num_cols[0] if num_cols else df.columns[0]
        for col in num_cols:
            if col.lower() in q_lower or col in resolved_cols.values():
                val_candidate = col
                break

        agg_func = "mean"
        if "sum" in q_lower or "total" in q_lower:
            agg_func = "sum"
        elif "count" in q_lower or "number of" in q_lower:
            agg_func = "count"
        elif "max" in q_lower or "highest" in q_lower:
            agg_func = "max"
        elif "min" in q_lower or "lowest" in q_lower:
            agg_func = "min"

        return {
            "tool": "groupby_agg",
            "args": {
                "group_col": group_candidate,
                "value_col": val_candidate,
                "agg_func": agg_func,
                "top_n": 10,
                "sort_ascending": False
            }
        }

    # Default fallback: Groupby or Top N
    default_group = cat_cols[0] if cat_cols else df.columns[0]
    default_val = num_cols[0] if num_cols else df.columns[-1]
    return {
        "tool": "groupby_agg",
        "args": {
            "group_col": default_group,
            "value_col": default_val,
            "agg_func": "mean",
            "top_n": 10,
            "sort_ascending": False
        }
    }


def plan_tool_call(
    question: str,
    df: pd.DataFrame,
    history: Optional[List[dict]] = None
) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    """
    Selects tool name and arguments for user question.
    Returns (tool_name, validated_args_dict, error_msg).
    """
    resolved_cols = fuzzy_match_columns(question, list(df.columns))

    if not LLM_API_KEY:
        # Fallback to rule-based planner
        plan = rule_based_planner(question, df, resolved_cols)
        tool_name = plan["tool"]
        raw_args = plan["args"]
        
        if tool_name in TOOL_REGISTRY:
            entry = TOOL_REGISTRY[tool_name]
            valid, validated_obj, err = validate_tool_args(entry.schema_model, raw_args)
            if valid:
                return tool_name, validated_obj.model_dump(), ""
            else:
                return None, None, f"Rule planner generated invalid args: {err}"
        return None, None, f"Unknown tool: {tool_name}"

    # LLM-based planning call
    try:
        from openai import OpenAI
        client_kwargs = {"api_key": LLM_API_KEY}
        if LLM_BASE_URL:
            client_kwargs["base_url"] = LLM_BASE_URL
        client = OpenAI(**client_kwargs)

        catalog = get_tool_catalog_summary()
        columns_info = {col: str(df[col].dtype) for col in df.columns}

        prompt = f"""
You are an expert Data Analyst Agent Planner.
Your job is to select the BEST tool from the available catalog to answer the user's question, and extract parameter arguments.

User Question: "{question}"

Resolved Column Mappings from Question:
{json.dumps(resolved_cols, indent=2)}

Available Dataset Columns & Types:
{json.dumps(columns_info, indent=2)}

Available Tool Catalog:
{json.dumps(catalog, indent=2)}

CRITICAL INSTRUCTIONS:
1. Output ONLY a valid JSON object with keys "tool" and "args".
2. "tool" MUST be one of the registered tool names in the catalog.
3. "args" MUST strictly match the schema parameters for that tool.
4. Always use EXACT column names as listed in Dataset Columns.
5. Do NOT output markdown code fences (```json), commentary, or extra text.

JSON Response:
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )

        content = response.choices[0].message.content.strip()
        # Clean potential markdown fences
        content = re.sub(r"^```json\s*", "", content, flags=re.MULTILINE)
        content = re.sub(r"^```\s*", "", content, flags=re.MULTILINE)
        content = content.strip()

        parsed = json.loads(content)
        tool_name = parsed.get("tool")
        raw_args = parsed.get("args", {})

        if tool_name not in TOOL_REGISTRY:
            # Fallback to rule planner
            plan = rule_based_planner(question, df, resolved_cols)
            tool_name = plan["tool"]
            raw_args = plan["args"]

        entry = TOOL_REGISTRY[tool_name]
        valid, validated_obj, err = validate_tool_args(entry.schema_model, raw_args)

        if not valid:
            # Single retry using rule-based planner
            plan = rule_based_planner(question, df, resolved_cols)
            tool_name = plan["tool"]
            entry = TOOL_REGISTRY[tool_name]
            valid, validated_obj, err = validate_tool_args(entry.schema_model, plan["args"])
            if valid:
                return tool_name, validated_obj.model_dump(), ""
            return None, None, f"Tool argument validation failed: {err}"

        return tool_name, validated_obj.model_dump(), ""

    except Exception as e:
        # Fail gracefully to rule-based planner
        plan = rule_based_planner(question, df, resolved_cols)
        tool_name = plan["tool"]
        if tool_name in TOOL_REGISTRY:
            entry = TOOL_REGISTRY[tool_name]
            valid, validated_obj, _ = validate_tool_args(entry.schema_model, plan["args"])
            if valid:
                return tool_name, validated_obj.model_dump(), ""
        return None, None, f"Planner error: {str(e)}"
