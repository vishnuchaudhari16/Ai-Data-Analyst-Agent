"""
Analyst Agent.
Master orchestrator connecting intent planning, tool execution, chart selection, and insight generation.
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
from tools.registry import TOOL_REGISTRY
from tools.visualization_tool import build_plotly_chart
from agents.planner import plan_tool_call
from agents.insight_agent import generate_insights
from utils.security import execute_with_timeout


class AnalystAgent:
    """Master agent coordinating natural language query resolution."""

    def analyze_question(
        self,
        question: str,
        df: pd.DataFrame,
        history: Optional[List[dict]] = None
    ) -> Dict[str, Any]:
        """
        Main execution pipeline:
        Query -> Planner -> Security Check -> Execute Tool -> Build Chart -> Generate Insights.
        """
        status_steps = []

        if df is None or df.empty:
            return {
                "status": "error",
                "message": "No dataset uploaded or dataset is empty.",
                "status_steps": status_steps,
            }

        # Step 1: Dataset inspected
        status_steps.append("✓ Dataset inspected")

        # Step 2 & 3: Plan tool call
        tool_name, tool_args, plan_err = plan_tool_call(question, df, history)
        if plan_err or not tool_name or tool_name not in TOOL_REGISTRY:
            return {
                "status": "error",
                "message": f"I can't answer that from the current dataset because {plan_err or 'no matching analysis tool was found'}.",
                "status_steps": status_steps,
            }

        status_steps.append("✓ Question classified")
        status_steps.append(f"✓ Analysis tool selected ({tool_name})")

        # Step 4: Execute tool deterministically
        entry = TOOL_REGISTRY[tool_name]
        try:
            tool_result = execute_with_timeout(entry.func, args=(df,), kwargs=tool_args)
            if tool_result.get("status") == "error":
                return {
                    "status": "error",
                    "message": tool_result.get("message", "Tool execution failed."),
                    "status_steps": status_steps,
                }
            status_steps.append("✓ Analysis completed")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Execution error while running tool '{tool_name}': {str(e)}",
                "status_steps": status_steps,
            }

        # Step 5: Visualization generation
        data_res = tool_result.get("data")
        result_type = tool_result.get("result_type", "auto")
        figure = None
        if isinstance(data_res, pd.DataFrame) and not data_res.empty:
            figure = build_plotly_chart(data_res, result_type=result_type, title=f"Analysis: {question[:50]}")
            if figure:
                status_steps.append("✓ Visualization generated")

        # Step 6: Generate Insights
        insights_text = generate_insights(tool_name, tool_result, question)
        status_steps.append("✓ Insights prepared")

        return {
            "status": "success",
            "question": question,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "status_steps": status_steps,
            "tool_result": tool_result,
            "figure": figure,
            "insights": insights_text,
            "code_display": tool_result.get("code_display", ""),
        }
