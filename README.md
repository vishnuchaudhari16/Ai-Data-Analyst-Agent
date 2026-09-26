# 🤖 AI Data Analyst Agent

> **"Turn your data into answers, insights, and decisions."**

A production-grade, secure AI Data Analyst application built with **Python + Streamlit**, **Pandas**, **Plotly**, **DuckDB**, **Scikit-Learn**, **Statsmodels**, and **Pydantic**.

---

## 🏛️ Key Design Architecture: Zero Code Execution Security Model

Unlike traditional AI coding assistants that generate untrusted raw Python or SQL strings and execute them directly against datasets, the **AI Data Analyst Agent** operates on a **Deterministic Tool-Registry Architecture**:

1. **Structured Tool Selection**: The LLM acts exclusively as an intent planner that selects from a fixed catalog of pre-written, typed, unit-tested Python functions (e.g., `groupby_agg`, `top_n`, `correlation_matrix`, `detect_anomalies`, `build_predictive_model`).
2. **Pydantic Argument Validation**: Arguments outputted by the LLM are parsed as JSON and strictly validated against Pydantic schema models before execution.
3. **Deterministic Execution**: Analysis runs via pure Python/Pandas or an allowlisted DuckDB query builder.
4. **AST-Based SQL Allowlist**: Custom SQL queries are parsed using `sqlglot` AST analysis to enforce single `SELECT`/`WITH` statements against a read-only `dataset` view. Administrative commands (`DROP`, `ATTACH`) and filesystem functions are blocked.
5. **Zero Hallucination Insights**: Final natural language business explanations are generated strictly from the computed tool JSON payload.
6. **Zero-LLM Fallback Mode**: The entire application (profiling, SQL, statistics, anomalies, ML, report exports) functions seamlessly in deterministic mode even when no API key is provided.

---

## 🌟 Key Features

- **📁 Multi-Format Upload & Smart Profiling**: Upload `.csv`, `.xlsx`, `.xls` with encoding detection (`utf-8`, `chardet`), memory caps (50MB), and automatic sampling for datasets >200k rows.
- **🛡️ Data Quality Score (0–100)**: Multi-factor score evaluating completeness, uniqueness, variability, cardinality health, and outlier regularity.
- **💬 Natural Language Chat**: ChatGPT-style interface with fuzzy column name resolution, execution status pipeline pills (`✓ Dataset inspected`, `✓ Question classified`, etc.), Plotly charts, grounded insights, and collapsible templated code views.
- **⚡ SQL Analyst (DuckDB)**: Query builder and AST-secured SQL editor.
- **🧪 Statistical Analysis**: Descriptive stats, Pearson/Spearman correlations, Independent 2-Sample T-Test, Chi-Square Test of Independence, and One-Way ANOVA with validity guards.
- **🚨 Anomaly Detection**: Outlier detection using IQR, Z-Score, and Isolation Forest with visual scatter highlight charts.
- **🤖 Predictive Analytics & Forecasting**: Machine Learning models (Random Forest, Gradient Boosting, Linear/Logistic, Decision Trees) with MAE/RMSE/R² or Accuracy/Precision/Recall/F1 metrics and Feature Importance. Time-series forecasting using Holt-Winters Exponential Smoothing.
- **⚡ Automated EDA**: One-click comprehensive exploratory data analysis report generation.
- **📑 Executive Report Generation**: Downloadable reports in HTML, CSV, and pure-Python PDF (via `fpdf2`).

---

## 🚀 Quickstart & Installation

```bash
# 1. Clone Repository
git clone https://github.com/your-repo/ai-data-analyst-agent.git
cd ai-data-analyst-agent

# 2. Create and Activate Virtual Environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Configure Environment (Optional for LLM features)
cp .env.example .env
# Edit .env and add your LLM_API_KEY if desired

# 5. Launch Application
streamlit run app.py
```

---

## 🧪 Running Unit Tests

Run the complete test suite across tool logic, security AST allowlists, and profiler algorithms:

```bash
pytest tests/ -v
```

---

## 📂 Project Architecture

```text
ai-data-analyst-agent/
│
├── app.py                      # Main Streamlit application entrypoint
├── config.py                   # Global configuration & environment settings
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── .gitignore
├── .env.example
│
├── agents/                     # Agent orchestrators & planners
│   ├── analyst_agent.py        # Master pipeline orchestrator
│   ├── planner.py              # LLM / Rule-based tool planner
│   ├── insight_agent.py        # Grounded prose generator
│   └── report_agent.py         # Executive report synthesizer
│
├── tools/                      # Deterministic Tool Registry
│   ├── registry.py             # Pydantic schemas & registration catalog
│   ├── data_profiler.py        # Profiling stats & 0-100 Quality Score
│   ├── pandas_tool.py          # Groupbys, filters, top_n, pivot, correlation
│   ├── sql_tool.py             # DuckDB SQL executor & sqlglot AST parser
│   ├── visualization_tool.py   # Shape-to-chart decision table & Plotly engine
│   ├── statistics_tool.py     # Hypothesis testing (T-Test, Chi2, ANOVA)
│   ├── anomaly_tool.py         # IQR, Z-Score, Isolation Forest
│   ├── ml_tool.py              # Scikit-learn ML & Statsmodels forecasting
│   └── report_tool.py          # HTML, CSV, and fpdf2 PDF report generator
│
├── utils/                      # Utilities & helpers
│   ├── file_handler.py         # Dataset loading, encoding, and sampling
│   ├── validators.py           # File & fuzzy column name resolution
│   ├── formatters.py           # Number, currency, and byte formatters
│   ├── security.py             # Pydantic arg validation & execution timeouts
│   └── session_manager.py      # Streamlit session state management
│
├── components/                 # Streamlit UI page components
│   ├── sidebar.py              # SaaS navigation & file loader
│   ├── dashboard.py            # Overview metrics & preview
│   ├── chat.py                 # Interactive AI chat interface
│   ├── explore.py              # Data exploration & filtering
│   ├── sql_analyst.py          # DuckDB SQL builder & query runner
│   ├── statistics_view.py      # Statistical testing view
│   ├── anomalies_view.py       # Outlier detection page
│   ├── predictive_view.py      # Machine Learning & Forecasting
│   ├── eda_view.py             # Automated EDA page
│   ├── report_view.py          # Executive report export
│   ├── settings_view.py        # Settings & masked API key status
│   ├── about_view.py           # Tech stack & metadata
│   └── cards.py                # Modern glassmorphism CSS & cards
│
├── data/sample_data/           # Sample datasets
│   ├── sales.csv
│   ├── employee_productivity.csv
│   └── customer_churn.csv
│
└── tests/                      # Pytest test suite
    ├── test_tools.py
    ├── test_security.py
    └── test_profiler.py
```

---

## 🔒 Security & Deployment (Streamlit Community Cloud)

- **Streamlit Cloud Ready**: Uses lightweight dependencies (`fpdf2` for PDF export, `statsmodels` for time-series) avoiding native binary compilation failures.
- **No API Key Logging**: API credentials are stored securely in `.streamlit/secrets.toml` or `.env` and are masked in the Settings UI.
- **Execution Timeouts**: Tool invocations are wrapped in execution timeouts to prevent resource exhaustion.
