# 📦 Mexican Customs & Trade SME Chatbot

A **RAG-powered conversational assistant** developed to assist importers, exporters, and customs brokers navigating the **US-Mexico** trade corridor. 

This system operates as a specialized **technical advisor**: it queries official Mexican and US trade legislation (*LIGIE/HTS, T-MEC/USMCA, Ley Aduanera, CFF*) using a semantic vector search engine (FAISS), retrieves precise legal foundations, and generates structured compliance reports and technical dictamens.

FAISS stands for Facebook AI Similarity Search. It is an open-source library created by Meta designed to search through millions of vector embeddings rapidly.Similarity Search: When a user submits a query, FAISS converts the query into a vector and calculates the distance (e.g., Cosine Similarity or Euclidean Distance) between the query vector and all document vectors in your dataset.High Efficiency: Instead of scanning every document line-by-line, FAISS uses optimized indexing algorithms (like IndexFlatL2 or IVF) to locate the top $k$ most relevant text chunks in milliseconds, even across vast databases.
---

## 🌟 Key Features

* **Semantic Legal Grounding (RAG)**: Ingests official trade laws, tariff schedules, and regulatory manuals (*Ley Aduanera, CFF, Anexo 22, USMCA/T-MEC*) from `./pdf_files_comercio_exterior` and indexes them with **FAISS** and **NVIDIA Embeddings** (`NV-Embed-QA`) for accurate retrieval.
* **LLM Expert Core**: Powered by **NVIDIA NIMs** (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) via `langchain-nvidia-ai-endpoints` to synthesize expert trade opinions and legal citations.
* **Risk & Compliance Reporting**: Generates technical trade opinions finalized with a **RAID (Risks, Actions, Issues, Decisions)** matrix table and explicit legal disclaimers.
* **PDF Export Utility**: Compiles generated technical dictamens into downloadable PDF reports instantly using **ReportLab**.
* **Interactive UI**: Streamlined **Streamlit** interface featuring document indexing, query handling, and regulatory excerpt inspection.

---

## 🛠️ Architecture & Tech Stack

* **Framework & UI**: [Streamlit](https://streamlit.io/)
* **LLM Core & Embeddings**: NVIDIA NIMs (`nvidia/llama-3.3-nemotron-super-49b-v1.5`, `NV-Embed-QA`) via `langchain-nvidia-ai-endpoints`
* **Vector Search & RAG**: `FAISS`, `PyPDFDirectoryLoader`, `RecursiveCharacterTextSplitter`, `tiktoken`
* **Report Generation**: [ReportLab](https://www.reportlab.com/)

---

## 📂 Project Structure

```text
mexican-customs-trade-sme-chatbot/
│
├── app.py                         # Main Streamlit application and RAG chatbot logic
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .env                           # Template for environment variables
└── pdf_files_comercio_exterior/   # Local directory for reference PDFs (Ley Aduanera, T-MEC, CFF, HTS)