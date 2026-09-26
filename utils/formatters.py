"""
Formatting utilities for numbers, currencies, percentages, and byte sizes.
Locale-agnostic and robust to NaN/None values.
"""

import math
from typing import Union, Optional
from config import DEFAULT_CURRENCY_SYMBOL, DEFAULT_DECIMAL_PLACES


def format_number(val: Optional[Union[int, float]], decimal_places: int = DEFAULT_DECIMAL_PLACES) -> str:
    """Format a numeric value with thousands separators and optional decimals."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "N/A"
    
    try:
        if isinstance(val, int) or val.is_integer():
            return f"{int(val):,}"
        return f"{val:,.{decimal_places}f}"
    except Exception:
        return str(val)


def format_currency(
    val: Optional[Union[int, float]],
    symbol: str = DEFAULT_CURRENCY_SYMBOL,
    decimal_places: int = DEFAULT_DECIMAL_PLACES
) -> str:
    """Format a value as currency."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "N/A"
    
    formatted_num = format_number(val, decimal_places=decimal_places)
    if formatted_num.startswith("-"):
        return f"-{symbol}{formatted_num[1:]}"
    return f"{symbol}{formatted_num}"


def format_percentage(val: Optional[Union[int, float]], decimal_places: int = 1) -> str:
    """Format a float/ratio as percentage (e.g. 0.156 -> '15.6%')."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "N/A"
    
    try:
        # If val is already > 1 (e.g. 15.6), format directly
        num = val * 100 if abs(val) <= 1.0 and val != 0 else val
        return f"{num:.{decimal_places}f}%"
    except Exception:
        return str(val)


def format_bytes(bytes_count: int) -> str:
    """Format byte sizes in human-readable KB/MB/GB."""
    if not bytes_count:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(bytes_count)
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {units[i]}"
