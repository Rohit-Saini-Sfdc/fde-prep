# FDE Preparation & Reusable Google Colab Code (`fde-prep`)

[![GitHub Repo](https://img.shields.io/badge/GitHub-fde--prep-blue?logo=github)](https://github.com/Rohit-Saini-Sfdc/fde-prep)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A curated repository of production-ready, reusable **Google Colab notebooks**, **weekly course assignments**, **data engineering utilities**, **SupportDesk RAG modules**, and **GenAI agent templates** designed for Forward Deployed Engineer (FDE) technical preparation and interview coding practice.

---

## 🚀 Interactive Google Colab Notebooks & Weekly Assignments

Click any badge below to launch and execute notebooks directly in Google Colab or inspect weekly module code:

### 📚 Weekly Assignments (`notebooks/assignments/`)

| Week | Assignment / Module | Description | Link / Colab |
|---|---|---|---|
| **Week 0** | **Hello World Week 0** | Week 0 introductory assignment notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/assignments/week_0/Hello_World_Week_0_Rohit_Saini.ipynb) |
| **Week 0** | **Hello World Week 0 (Ver 2)** | Week 0 assignment notebook (Version 2) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/assignments/week_0/Hello_World_Week_0_Rohit_Saini_ipynb_Ver_2.ipynb) |
| **Week 1** | **IT Monitoring Agent** | Automated IT infrastructure monitoring & log analysis agent | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/assignments/week_1/Week1_IT_Monitoring_Agent_Assignment.ipynb) |
| **Week 2** | **SupportDesk RAG Foundations** | **Module 1**: OpenAI Embeddings & Similarity Search<br>**Module 2**: Text Chunking & Chroma Vector Store<br>**Module 3**: LlamaIndex Strategies (Vector, Keyword, Tree) | [`notebooks/assignments/week_2/`](https://github.com/Rohit-Saini-Sfdc/fde-prep/tree/main/notebooks/assignments/week_2) |
| **Week 3** | **Advanced RAG, Eval & Agents** | **Module 4**: LangChain LCEL RAG Pipeline & Multi-Turn<br>**Module 5**: Groundedness, Completeness & LLM-as-Judge<br>**Module 6**: ReAct Agentic RAG with Dynamic Tool Calling | [`notebooks/assignments/week_3/`](https://github.com/Rohit-Saini-Sfdc/fde-prep/tree/main/notebooks/assignments/week_3) |

---

## 🏗️ SupportDesk RAG System (Week 2 & Week 3)

The **SupportDesk RAG System** is a production-grade troubleshooting assistant built to retrieve historical support ticket context and answer technical support queries using OpenAI and LangChain/LlamaIndex.

### 🧩 Module Breakdown

* **[Week 2 / Module 1: Embeddings & Similarity Search](https://github.com/Rohit-Saini-Sfdc/fde-prep/tree/main/notebooks/assignments/week_2/1_embeddings)**  
  Generates 1,536-dimensional embeddings with OpenAI (`text-embedding-3-small`), measures semantic similarity using Cosine Similarity, applies similarity score thresholding, and visualizes similarity matrices with heatmaps.
  
* **[Week 2 / Module 2: Chunking & Vector Stores](https://github.com/Rohit-Saini-Sfdc/fde-prep/tree/main/notebooks/assignments/week_2/2_chunking)**  
  Explores fixed-size, recursive, and semantic text chunking strategies. Builds persistent **Chroma DB** collections and executes multi-condition metadata filtering (`$and` operators).

* **[Week 2 / Module 3: Indexing Strategies](https://github.com/Rohit-Saini-Sfdc/fde-prep/tree/main/notebooks/assignments/week_2/3_indexing)**  
  Uses **LlamaIndex** to benchmark index types (`VectorStoreIndex` for semantic search, `KeywordTableIndex` for exact IDs, `TreeIndex` for hierarchical summaries) and implements hybrid retrieval.

* **[Week 3 / Module 4: RAG Pipeline](https://github.com/Rohit-Saini-Sfdc/fde-prep/tree/main/notebooks/assignments/week_3/4_rag_pipeline)**  
  Assembles LangChain Expression Language (LCEL) chains, configures Maximal Marginal Relevance (MMR) for diverse retrieval, implements anti-hallucination safeguards, inline ticket citations `[TICK-XXX]`, and query reformulation for multi-turn conversations.

* **[Week 3 / Module 5: Evaluation & Metrics](https://github.com/Rohit-Saini-Sfdc/fde-prep/tree/main/notebooks/assignments/week_3/5_evaluation)**  
  Calculates retrieval metrics (Precision@k, Recall@k, F1@k) across evaluation datasets, uses LLM-as-Judge for Groundedness and Completeness scores, performs failure analysis, and establishes release deployment gates (`PASS`/`REVIEW`/`BLOCK`).

* **[Week 3 / Module 6: Agentic RAG](https://github.com/Rohit-Saini-Sfdc/fde-prep/tree/main/notebooks/assignments/week_3/6_agentic_rag)**  
  Constructs an autonomous ReAct agent equipped with dynamic tools (`SearchSimilarTickets`, `GetTicketByID`, `SearchByCategory`, `GetTicketStatistics`). Executes multi-step reasoning loops to answer complex support requests.

---

### 🤖 Generative AI & LLM Agents (`notebooks/genai_and_llms/`)

| Notebook | Description | Open in Colab |
|---|---|---|
| **Calling LLMs Programmatically** | LLM API integration, prompt orchestration, and structured generation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/genai_and_llms/Calling_LLMs_programmatically.ipynb) |
| **CRM Lead Qualifier Agent** | AI agent workflow for automated CRM lead scoring, enrichment, & qualification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/genai_and_llms/crm_lead_qualifier_agent.ipynb) |
| **Gemini Integration** | Google Gemini API integration and response handling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/genai_and_llms/Gemini.ipynb) |
| **OpenAI Integration** | OpenAI API integration, chat completions, & structured output | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/genai_and_llms/OpenAi.ipynb) |
| **Streamlit UI App** | Interactive Streamlit prototyping for AI applications | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/genai_and_llms/Streamlit.ipynb) |

---

### 🛠️ Core Utilities & Data Engineering (`notebooks/core_utilities/`)

| Notebook | Description | Open in Colab |
|---|---|---|
| **01. Async API Ingestion** | Concurrency limits (`Semaphore`), retries, `httpx`, `nest_asyncio` in Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/core_utilities/01_async_api_ingestion.ipynb) |
| **02. Data Validation (Pydantic)** | Runtime data validation, nested payload flattening, custom type validators | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/core_utilities/02_data_validation_pydantic.ipynb) |
| **03. PySpark & Pandas ETL** | Local PySpark setup in Colab, aggregations, joins, window functions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/core_utilities/03_etl_pandas_pyspark.ipynb) |
| **04. High-Performance DuckDB** | Execute SQL directly on Pandas DataFrames, CSV, & Parquet in Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/core_utilities/04_duckdb_sql_helpers.ipynb) |
| **05. Google Drive & Secrets** | Google Drive mounting, API secrets management via `google.colab.userdata` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/core_utilities/05_colab_drive_and_secrets.ipynb) |
| **Read Files from Drive** | Reading and processing files mounted from Google Drive | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/core_utilities/Read_files_from_Drive.ipynb) |
| **Python Basics for GenAI** | Core Python techniques & foundational data processing for GenAI | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohit-Saini-Sfdc/fde-prep/blob/main/notebooks/core_utilities/python_basics_for_Gen_Ai_IK.ipynb) |

---

## 📂 Repository Layout

```
fde-prep/
├── README.md                           # Master documentation and Colab badges
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Standard Python & Jupyter ignore rules
├── notebooks/
│   ├── assignments/
│   │   ├── week_0/                     # Week 0: Intro assignments
│   │   │   ├── Hello_World_Week_0_Rohit_Saini.ipynb
│   │   │   └── Hello_World_Week_0_Rohit_Saini_ipynb_Ver_2.ipynb
│   │   ├── week_1/                     # Week 1: IT Monitoring Agent
│   │   │   └── Week1_IT_Monitoring_Agent_Assignment.ipynb
│   │   ├── week_2/                     # Week 2: SupportDesk RAG Foundations
│   │   │   ├── 1_embeddings/           # Module 1: OpenAI Embeddings & Similarity Search
│   │   │   ├── 2_chunking/             # Module 2: Text Chunking & Chroma Vector Store
│   │   │   ├── 3_indexing/             # Module 3: LlamaIndex Strategies
│   │   │   └── data/
│   │   ├── week_3/                     # Week 3: Advanced RAG, Evaluation & Agents
│   │   │   ├── 4_rag_pipeline/         # Module 4: LangChain RAG Pipeline & Multi-Turn
│   │   │   ├── 5_evaluation/           # Module 5: Evaluation & Metrics (LLM-as-Judge)
│   │   │   ├── 6_agentic_rag/          # Module 6: ReAct Agentic RAG & Tool Calling
│   │   │   └── data/
│   │   └── SupportDesk-RAG-Workshop/   # Full SupportDesk RAG Workshop Reference
│   ├── core_utilities/                 # Reusable data engineering & Colab helpers
│   │   ├── 01_async_api_ingestion.ipynb
│   │   ├── 02_data_validation_pydantic.ipynb
│   │   ├── 03_etl_pandas_pyspark.ipynb
│   │   ├── 04_duckdb_sql_helpers.ipynb
│   │   ├── 05_colab_drive_and_secrets.ipynb
│   │   ├── Read_files_from_Drive.ipynb
│   │   └── python_basics_for_Gen_Ai_IK.ipynb
│   └── genai_and_llms/                 # GenAI, LLM APIs, and AI agent workflows
│       ├── Calling_LLMs_programmatically.ipynb
│       ├── Gemini.ipynb
│       ├── OpenAi.ipynb
│       ├── Streamlit.ipynb
│       └── crm_lead_qualifier_agent.ipynb
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
