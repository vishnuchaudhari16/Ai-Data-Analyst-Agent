from typing import Tuple, List, Optional, Any, Dict
import pandas as pd


ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def validate_file_extension(filename: str) -> Tuple[bool, str]:
    """Validate file extension against allowed types."""
    if not filename:
        return False, "No file provided."
    ext = filename.lower()[filename.rfind("."):]
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Please upload a CSV (.csv) or Excel (.xlsx, .xls) file."
    return True, "Valid extension"


def validate_dataframe(df: Any) -> Tuple[bool, str]:
    """Validate loaded dataframe structure and contents."""
    if df is None:
        return False, "DataFrame is None."
    if not isinstance(df, pd.DataFrame):
        return False, f"Expected pandas DataFrame, got {type(df).__name__}."
    if df.empty:
        return False, "Uploaded dataset is empty (0 rows or 0 columns)."
    if len(df.columns) == 0:
        return False, "Uploaded dataset contains no columns."
    return True, "Valid DataFrame"


def sanitize_column_names(columns: List[str]) -> List[str]:
    """Clean column names to remove leading/trailing spaces and illegal characters."""
    sanitized = []
    seen = {}
    for col in columns:
        clean_name = str(col).strip()
        clean_name = clean_name.replace("\n", " ").replace("\r", " ")
        if not clean_name:
            clean_name = "Unnamed_Column"
        
        # Handle duplicates
        if clean_name in seen:
            seen[clean_name] += 1
            clean_name = f"{clean_name}_{seen[clean_name]}"
        else:
            seen[clean_name] = 0
            
        sanitized.append(clean_name)
    return sanitized


def fuzzy_match_columns(question: str, df_columns: List[str]) -> Dict[str, str]:
    """
    Fuzzy match terms in user's question to actual dataset column names.
    Returns dictionary mapping {matched_term_in_question: actual_column_name}.
    """
    resolved = {}
    question_lower = question.lower()
    
    # Common synonyms mapping
    synonyms = {
        "revenue": ["sales", "income", "amount", "profit"],
        "cost": ["expense", "price", "discount"],
        "dept": ["department"],
        "emp": ["employee", "staff"],
        "user": ["customer", "client"],
        "date": ["time", "day", "month", "year"],
    }
    
    for col in df_columns:
        col_lower = col.lower().replace("_", " ")
        # Direct exact match or word match
        if col_lower in question_lower:
            resolved[col_lower] = col
            continue
        
        # Split tokens
        tokens = col_lower.split()
        for token in tokens:
            if len(token) >= 3 and token in question_lower:
                resolved[token] = col
                break
            
            # Check synonyms
            for syn_key, syn_vals in synonyms.items():
                if token == syn_key and any(v in question_lower for v in syn_vals):
                    resolved[syn_key] = col
                elif any(v == token for v in syn_vals) and syn_key in question_lower:
                    resolved[syn_key] = col

    return resolved

