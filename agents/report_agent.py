"""
Report Agent.
Synthesizes executive summary content for reports using dataset profile and analysis history.
"""

from typing import Dict, Any
from tools.data_profiler import profile_dataset
from config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL


def synthesize_executive_summary(df: pd.DataFrame if 'pd' in locals() else Any, meta: dict) -> str:
    """Generate executive summary text for report."""
    profile = profile_dataset(df)
    quality = profile.get("quality_info", {})

    default_summary = (
        f"The dataset '{meta.get('filename', 'Dataset')}' comprises {profile.get('total_rows', 0):,} records "
        f"across {profile.get('total_cols', 0)} attributes. Overall Data Quality is rated as '{quality.get('label', 'N/A')}' "
        f"with a score of {quality.get('score', 0)}/100. Missing value rate is {profile.get('missing_pct', 0)}% and duplicate rows count is {profile.get('duplicate_rows', 0)}."
    )

    if not LLM_API_KEY:
        return default_summary

    try:
        from openai import OpenAI
        client_kwargs = {"api_key": LLM_API_KEY}
        if LLM_BASE_URL:
            client_kwargs["base_url"] = LLM_BASE_URL
        client = OpenAI(**client_kwargs)

        prompt = f"""
Synthesize a concise 3-sentence executive summary for a business intelligence report based on these dataset stats:
- Filename: {meta.get('filename')}
- Rows: {profile.get('total_rows')}
- Columns: {profile.get('total_cols')}
- Missingness: {profile.get('missing_pct')}%
- Data Quality Score: {quality.get('score')}/100 ({quality.get('label')})
- Numeric Attributes: {', '.join(profile.get('numeric_cols', [])[:5])}

Summary:
"""
        res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return default_summary
