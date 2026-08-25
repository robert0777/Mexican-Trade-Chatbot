# 📦 Mexican Customs & Trade SME Chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/) 

A **RAG-powered conversational assistant** built with Streamlit and NVIDIA NIMs to assist importers, exporters, and customs brokers navigating the **US-Mexico (USMCA / T-MEC)** trade corridor and Mexican customs legislation.

👉 **[Try the Live App: Asesor AI de Comercio Exterior y Aduanas](https://mexican-trade-chatbot.streamlit.app/)**

---

## 🌟 Key Features

* **Dynamic In-Memory RAG Pipeline**: Ingests trade documentation from `./pdf_files_comercio_exterior` using `PyPDFDirectoryLoader`, normalizes whitespace, and adaptively splits text into chunks via `RecursiveCharacterTextSplitter` based on target token limits calculated from the total file count.
* **Smart Context Selection**: Ranks document chunks by keyword overlap and length scaling metrics to dynamically select diverse source chunks within token boundaries (`tiktoken` model encoding).
* **LLM Technical Core**: Connects directly to **NVIDIA NIMs** (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) via the `openai` Python SDK with system-level reasoning (`/think`) enabled.
* **Risk & Compliance Focus**: Generates technical compliance reports with specific legal citations (*Ley Aduanera, Anexo 22, CFF, T-MEC*), tax breakdowns (IGE, IVA, DTA), disclaimers, and a required **RAID (Risks, Actions, Issues, Decisions)** matrix table.
* **Contextual Greeting Handler**: Detects greetings in Spanish without needlessly triggering full LLM queries and returns time-aware responses (*Buenos días / Tardes / Noches*).
* **Interactive UI & Source Verification**: Streamlit wide-layout interface featuring custom sidebar branding, author credentials, execution timing, and expandable panels for source verification.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend / UI**: Streamlit (`>=1.30.0`)
* **LLM Engine**: NVIDIA NIMs via standard `openai` SDK (`openai>=1.0.0`)
* **Document Loaders & Text Splitting**: `langchain-community`, `langchain-text-splitters`, `pypdf`
* **Token Encoding & Utilities**: `tiktoken`, `python-dotenv`

---

## 📂 Project Structure

```text
├── app.py                            # Main Streamlit application
├── pdf_files_comercio_exterior/      # Target folder for trade & legal PDF files
├── ai-advisor-icon.svg               # Application icon
├── requirements.txt                  # Python package dependencies
├── .env                              # Environment variable configuration
└── README.md                         # Project documentation