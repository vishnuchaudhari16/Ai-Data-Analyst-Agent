"""
File loading and pre-processing module.
Handles CSV and Excel parsing, encoding fallback, memory management, and auto-sampling.
"""

import io
import os
import chardet
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from config import MAX_UPLOAD_MB, MAX_ROWS_BEFORE_SAMPLING, DEFAULT_SAMPLE_SIZE
from utils.validators import validate_file_extension, validate_dataframe, sanitize_column_names


def detect_file_encoding(file_bytes: bytes) -> str:
    """Detect file encoding using chardet with utf-8 fallback."""
    try:
        # Try UTF-8 first
        file_bytes[:10000].decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        result = chardet.detect(file_bytes[:50000])
        return result.get("encoding") or "latin1"


def auto_detect_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert string columns with date patterns into datetime objects."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            # Check a small non-null sample
            sample = df_copy[col].dropna().astype(str).head(50)
            if sample.empty:
                continue
            
            # Check if majority of sample looks like dates
            has_date_keywords = any(kw in col.lower() for kw in ["date", "time", "day", "month", "year", "created", "updated"])
            
            try:
                # Try parsing if sample contains common date separators or date keywords in column name
                if has_date_keywords or sample.str.contains(r"[-/:]").mean() > 0.7:
                    converted = pd.to_datetime(df_copy[col], errors="coerce")
                    # Only convert if at least 70% of non-null sample parses successfully
                    if converted.notna().mean() >= 0.7:
                        df_copy[col] = converted
            except Exception:
                pass
    return df_copy


def load_dataset_from_bytes(
    file_bytes: bytes,
    filename: str
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], Optional[str]]:
    """
    Load dataset from raw bytes (UploadedFile or file read).
    Returns (dataframe, metadata_info, error_message).
    """
    valid, ext_msg = validate_file_extension(filename)
    if not valid:
        return None, {}, ext_msg

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return None, {}, f"File size ({size_mb:.1f} MB) exceeds maximum allowed size of {MAX_UPLOAD_MB} MB."

    try:
        filename_lower = filename.lower()
        if filename_lower.endswith(".csv"):
            encoding = detect_file_encoding(file_bytes)
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            except Exception:
                # Fallback to latin1 / python engine if C engine fails
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1", engine="python")
        elif filename_lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None, {}, f"Unsupported file extension: {filename}"

        valid_df, df_msg = validate_dataframe(df)
        if not valid_df:
            return None, {}, df_msg

        # Clean column names
        df.columns = sanitize_column_names(list(df.columns))

        # Auto-detect dates
        df = auto_detect_date_columns(df)

        original_row_count = len(df)
        is_sampled = False
        sample_info = ""

        # Row sampling for large datasets
        if original_row_count > MAX_ROWS_BEFORE_SAMPLING:
            df = df.sample(n=DEFAULT_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
            is_sampled = True
            sample_info = f"Sampled {DEFAULT_SAMPLE_SIZE:,} rows out of {original_row_count:,} total records for cloud performance."

        meta = {
            "filename": filename,
            "size_mb": round(size_mb, 2),
            "original_rows": original_row_count,
            "current_rows": len(df),
            "columns": len(df.columns),
            "is_sampled": is_sampled,
            "sample_info": sample_info,
        }

        return df, meta, None

    except Exception as e:
        return None, {}, f"Error reading file '{filename}': {str(e)}"
