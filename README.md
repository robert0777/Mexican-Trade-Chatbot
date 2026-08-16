# 📦 Mexican Customs & Trade SME Chatbot

A **RAG-powered conversational assistant** developed to assist importers, exporters, and customs brokers navigating the **US-Mexico** trade corridor. 

This system operates as a specialized **technical advisor**: it queries official Mexican and US trade legislation (*LIGIE/HTS, T-MEC/USMCA, Ley Aduanera, CFF*) using a semantic vector search engine (FAISS), retrieves precise legal foundations, and generates structured compliance reports and technical dictamens.

---

## 💡 How Semantic Retrieval Works (FAISS)

To deliver precise legal grounding, the system utilizes **FAISS** (Facebook AI Similarity Search), an open-source vector search library developed by Meta:

* **Vector Search Engine**: Documents are converted into high-dimensional vector embeddings using NVIDIA Embeddings (`NV-Embed-QA`). FAISS calculates the mathematical distance (e.g., Cosine Similarity / Euclidean Distance) between the user's query vector and all legal document vectors.
* **High Efficiency Retrieval**: Rather than scanning dense PDF files line-by-line, FAISS uses optimized indexing algorithms to retrieve the top $k$ most relevant legal excerpts in milliseconds.

---

## 🌟 Key Features

* **Semantic Legal Grounding (RAG)**: Ingests official trade laws, tariff schedules, and regulatory manuals (*Ley Aduanera, CFF, Anexo 22, USMCA/T-MEC*) from `./pdf_files_comercio_exterior` and indexes them with **FAISS** and **NVIDIA Embeddings** for context-aware retrieval.
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