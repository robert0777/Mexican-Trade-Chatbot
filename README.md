# 📦 Mexican Customs & Trade SME Chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/) 

A **RAG-powered conversational assistant** built with Streamlit and OpenRouter to assist importers, exporters, and customs brokers navigating the **US-Mexico (USMCA / T-MEC)** trade corridor and Mexican customs legislation.

👉 **[Try the Live App: Asesor AI de Comercio Exterior y Aduanas](https://mexican-trade-chatbot.streamlit.app/)**

---

## 🌟 Key Features

* **Dynamic In-Memory RAG Pipeline**: Ingests trade documentation from `./pdf_files_comercio_exterior` using `PyPDFDirectoryLoader`, applies domain-specific legal text normalization (expanding abbreviations like *Art.*, *Fracc.*, *L.A.*, *CFF*), and adaptively splits text into chunks via `RecursiveCharacterTextSplitter` calculated from total document volume.
* **Smart Context Selection**: Ranks document chunks by keyword overlap and length scaling metrics to dynamically select diverse source chunks within a 5,000-token context boundary using `tiktoken`.
* **Multi-Model LLM Engine (OpenRouter)**: Integrates with **OpenRouter API** featuring automated fallback across high-throughput non-Llama models (`minimax/minimax-m3:free`, `google/gemma-4-31b:free`, `cohere/north-mini-code:free`, and `openrouter/free-models-router`) to prevent rate limits or service drops.
* **Risk & Compliance Focus**: Generates technical compliance reports with specific legal citations (*Ley Aduanera, Anexo 22, CFF, T-MEC*), tax breakdowns (IGE, IVA, DTA, IEPS), disclaimers, and a required **RAID (Risks, Actions, Issues, Decisions)** matrix table.
* **Contextual Greeting Handler**: Detects greetings in Spanish without needlessly triggering full LLM queries and returns time-aware responses (*Buenos días / Tardes / Noches*).
* **Interactive UI & Source Verification**: Streamlit wide-layout interface featuring custom sidebar branding, author credentials, execution timing, and expandable panels for source verification.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend / UI**: Streamlit (`>=1.30.0`)
* **LLM Engine**: OpenRouter API via standard `openai` SDK (`openai>=1.0.0`)
* **Document Processing**: `langchain-community`, `langchain-text-splitters`, `pypdf`
* **Token Encoding & Utilities**: `tiktoken`, `python-dotenv`

---

## 📂 Project Structure

```text
.
├── app.py                            # Main Streamlit application
├── pdf_files_comercio_exterior/      # Directory containing legal & trade PDF files
├── ai-advisor-icon.svg               # Application branding icon
├── requirements.txt                  # Python package dependencies
├── .env                              # Environment variable configuration (OPENROUTER_API_KEY)
└── README.md                         # Project documentation