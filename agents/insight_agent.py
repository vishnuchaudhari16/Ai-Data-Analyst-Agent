"""
Insight Agent.
Generates business insights grounded strictly in computed tool output JSON.
Includes rule-based fallback when LLM API key is not configured.
"""

import json
from typing import Dict, Any
from config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL
from utils.formatters import format_number, format_currency


def generate_rule_based_insight(tool_name: str, tool_result: Dict[str, Any], question: str) -> str:
    """Generate structured rule-based insight when LLM is unavailable."""
    summary = tool_result.get("summary_stats", {})
    data = tool_result.get("data")

    key_finding = "Analysis completed successfully."
    evidence = "Data computed from dataset."
    insight = "Key metrics evaluated."
    recommendation = "Review results for operational decisions."

    if tool_name == "groupby_agg":
        top_cat = summary.get("top_category", "N/A")
        max_val = summary.get("max_value", 0)
        min_val = summary.get("min_value", 0)
        cnt = summary.get("group_count", 0)

        key_finding = f"**{top_cat}** ranks highest with an aggregate value of **{format_number(max_val)}**."
        evidence = f"Across **{cnt}** groups, maximum is **{format_number(max_val)}** and minimum is **{format_number(min_val)}**."
        insight = f"The dataset shows significant variance across categories, with top category leading."
        recommendation = f"Focus resources on high-performing segment '{top_cat}' while evaluating lower metrics."

    elif tool_name == "top_n":
        cnt = summary.get("top_n_count", 0)
        col = summary.get("sorted_column", "metric")
        key_finding = f"Top **{cnt}** records retrieved sorted by **{col}**."
        evidence = f"Analyzed top {cnt} entries from dataset."
        insight = f"Top entries represent key drivers for {col}."
        recommendation = "Investigate drivers behind the leading entries."

    elif tool_name == "correlation_matrix":
        cols = summary.get("analyzed_columns", [])
        key_finding = f"Pearson correlation calculated across **{len(cols)}** numerical columns."
        evidence = f"Columns analyzed: {', '.join(cols)}."
        insight = "Strong positive or negative correlations indicate key linear relationships."
        recommendation = "Consider strongly correlated feature pairs for predictive modeling."

    elif tool_name == "detect_anomalies":
        total_anom = summary.get("anomalies_detected", 0)
        anom_rate = summary.get("anomaly_rate", 0)
        method = summary.get("method", "IQR")
        key_finding = f"Detected **{total_anom}** anomalies (**{anom_rate}%** rate) using **{method}**."
        evidence = f"Threshold method: {method}."
        insight = "Outlier values deviate significantly from normal distribution."
        recommendation = "Audit anomalous records to verify data quality or investigate operational exceptions."

    return f"""### 🔎 Key Finding
{key_finding}

### 📊 Evidence
{evidence}

### 💡 Insight
{insight}

### 📌 Recommendation
{recommendation}"""


def generate_insights(
    tool_name: str,
    tool_result: Dict[str, Any],
    question: str
) -> str:
    """
    Generates natural language business explanation grounded strictly in tool_result payload.
    """
    if tool_result.get("status") == "error":
        return f"⚠️ {tool_result.get('message', 'Unable to complete analysis.')}"

    if not LLM_API_KEY:
        return generate_rule_based_insight(tool_name, tool_result, question)

    try:
        from openai import OpenAI
        client_kwargs = {"api_key": LLM_API_KEY}
        if LLM_BASE_URL:
            client_kwargs["base_url"] = LLM_BASE_URL
        client = OpenAI(**client_kwargs)

        # Prepare serializable payload (convert dataframe head to dict)
        data_sample = []
        data_obj = tool_result.get("data")
        if hasattr(data_obj, "head"):
            data_sample = data_obj.head(15).to_dict(orient="records")

        payload = {
            "tool_name": tool_name,
            "summary_stats": tool_result.get("summary_stats", {}),
            "data_sample": data_sample,
        }

        prompt = f"""
You are an expert AI Data Analyst.
Explain the analysis results to a business user based ONLY on the computed JSON payload below.

User Question: "{question}"
Executed Tool Payload:
{json.dumps(payload, indent=2, default=str)}

STRICT GROUNDING RULES:
1. Use ONLY numbers, values, and names present in the Executed Tool Payload above. Do NOT invent, assume, or hallucinate numbers.
2. Structure your response using EXACTLY these four markdown headings:
### 🔎 Key Finding
### 📊 Evidence
### 💡 Insight
### 📌 Recommendation

3. Keep it concise, practical, and business-focused.
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return generate_rule_based_insight(tool_name, tool_result, question)
