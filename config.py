"""
Configuration file for AI Data Analyst Agent.
Handles environment variables, default settings, and app metadata.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if available
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# App Metadata
PROJECT_NAME = "AI Data Analyst Agent"
SUBTITLE = "Turn your data into answers, insights, and decisions."
APP_VERSION = "1.0.0"
DEVELOPER_NAME = "AI Data Engineering Team"

# Resource & Limit Settings
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_ROWS_BEFORE_SAMPLING = int(os.getenv("MAX_ROWS_BEFORE_SAMPLING", "200000"))
DEFAULT_SAMPLE_SIZE = int(os.getenv("DEFAULT_SAMPLE_SIZE", "50000"))
LLM_CALL_TIMEOUT_SECONDS = int(os.getenv("LLM_CALL_TIMEOUT_SECONDS", "30"))
MAX_SESSION_LLM_CALLS = int(os.getenv("MAX_SESSION_LLM_CALLS", "50"))

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # openai, gemini, custom
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", None)  # Optional for OpenAI-compatible endpoints
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Paths
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample_data"
ASSETS_DIR = BASE_DIR / "assets"

# Locale / Formatting Defaults
DEFAULT_CURRENCY_SYMBOL = "$"
DEFAULT_DECIMAL_PLACES = 2
