"""
Security module for validating LLM output, Pydantic schema checking, and SQL AST allowlisting.
"""

from typing import Dict, Any, Type, Tuple, Callable
import functools
import concurrent.futures
from pydantic import BaseModel, ValidationError
from config import LLM_CALL_TIMEOUT_SECONDS


def validate_tool_args(schema_model: Type[BaseModel], raw_args: Dict[str, Any]) -> Tuple[bool, Any, str]:
    """
    Validate raw dictionary arguments against a Pydantic model.
    Returns (is_valid, validated_model_instance, error_message).
    """
    if not isinstance(raw_args, dict):
        return False, None, f"Expected dictionary arguments, got {type(raw_args).__name__}"

    try:
        validated_instance = schema_model(**raw_args)
        return True, validated_instance, ""
    except ValidationError as ve:
        error_details = "; ".join([f"{err['loc']}: {err['msg']}" for err in ve.errors()])
        return False, None, f"Argument validation failed: {error_details}"
    except Exception as e:
        return False, None, f"Invalid arguments format: {str(e)}"


def execute_with_timeout(func: Callable, args: tuple = (), kwargs: dict = None, timeout_seconds: int = LLM_CALL_TIMEOUT_SECONDS) -> Any:
    """
    Execute a function with a timeout using concurrent.futures for cross-platform portability.
    """
    if kwargs is None:
        kwargs = {}
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Execution timed out after {timeout_seconds} seconds.")
