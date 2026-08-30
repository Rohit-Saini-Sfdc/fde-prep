# FDE Preparation & Reusable Google Colab Code (`fde-prep`)

[![GitHub Repo](https://img.shields.io/badge/GitHub-fde--prep-blue?logo=github)](https://github.com/Rohit-Saini-Sfdc/fde-prep)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A curated repository of production-ready, reusable **Google Colab notebooks**, **data engineering utilities**, and **async API templates** designed for Forward Deployed Engineer (FDE) technical preparation and interview coding practice.

---

## 🚀 Interactive Google Colab Notebooks

Click any badge below to immediately launch and execute the notebook in Google Colab:

| # | Notebook | Description | Open in Colab |
|---|---|---|---|
| 01 | **Async API Ingestion & Concurrency** | Concurrency limits (`Semaphore`), retries, `httpx`, `nest_asyncio` in Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/01_async_api_ingestion.ipynb) |
| 02 | **Data Validation & Pydantic v2** | Runtime data validation, nested payload flattening, custom type validators | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/02_data_validation_pydantic.ipynb) |
| 03 | **PySpark & Pandas ETL Pipeline** | Local PySpark setup in Colab, aggregations, joins, window functions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/03_etl_pandas_pyspark.ipynb) |
| 04 | **High-Performance DuckDB SQL** | Execute SQL directly on Pandas DataFrames, CSV, & Parquet in Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/04_duckdb_sql_helpers.ipynb) |
| 05 | **Google Drive & Secrets API** | Google Drive mounting, API secrets management via `google.colab.userdata` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/05_colab_drive_and_secrets.ipynb) |

---

## 📂 Repository Layout

```
fde-prep/
├── README.md                           # Master documentation and Colab badges
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Standard Python & Jupyter ignore rules
├── notebooks/
│   ├── 01_async_api_ingestion.ipynb     # Async fetching & rate limiting
│   ├── 02_data_validation_pydantic.ipynb# Pydantic schema enforcement
│   ├── 03_etl_pandas_pyspark.ipynb      # PySpark ETL transformations
│   ├── 04_duckdb_sql_helpers.ipynb      # Embedded SQL execution with DuckDB
│   └── 05_colab_drive_and_secrets.ipynb # Secrets management & Drive mounting
└── src/
    ├── __init__.py
    ├── async_client.py                 # Reusable async API client class
    ├── models.py                       # Reusable Pydantic data models
    └── main.py                         # Local entrypoint script
```

---

## 🛠️ Quickstart (Local Execution)

Clone the repository and run the standalone example locally:

```bash
git clone https://github.com/Rohit-Saini-Sfdc/fde-prep.git
cd fde-prep

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run main script
python -m src.main
```

---

## 🔑 License & Author
Created by **Rohit Kumar Saini** for Forward Deployed Engineering preparation.
Licensed under the [MIT License](LICENSE).
