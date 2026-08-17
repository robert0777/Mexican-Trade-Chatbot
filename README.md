# 📦 Mexican Customs & Trade SME Chatbot

A **RAG-powered conversational assistant** built with Streamlit and NVIDIA NIMs to assist importers, exporters, and customs brokers navigating the **US-Mexico** trade corridor.

This system operates as a specialized **technical advisor**: it ingests official trade legislation (*Ley Aduanera, T-MEC/USMCA, CFF*) from local PDF files, retrieves the most relevant legal context based on dynamic token optimization and keyword relevance, and generates structured compliance reports with RAID matrices using NVIDIA's `llama-3.3-nemotron-super-49b-v1.5` model.

---

## 🌟 Key Features

* **Dynamic In-Memory RAG Pipeline**: Ingests and normalizes trade documentation from `./pdf_files_comercio_exterior` using `PyPDFDirectoryLoader`, splitting text adaptively with `RecursiveCharacterTextSplitter` based on target token counts.
* **Smart Chunk Selection**: Scores document relevance dynamically using keyword overlap and length-scaling metrics, fitting context seamlessly within model token constraints via `tiktoken`.
* **LLM Expert Core**: Integrates directly with **NVIDIA NIMs** (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) via the `openai` SDK to deliver technical trade opinions, citations, and compliance reports.
* **Risk & Compliance Focus**: Outputs detailed legal analyses complete with tax metric breakdowns (IGE, IVA, DTA), normative citations, and a mandatory **RAID (Risks, Actions, Issues, Decisions)** matrix table.
* **Contextual Greeting Handler**: Normalizes user input and handles time-based greetings in Spanish without needlessly burning model context.
* **Interactive UI**: Streamlined **Streamlit** wide-layout application with an interactive sidebar, custom link navigation, document indexing trigger, and expandable source verification panels.

---

## 🛠️ Architecture & Tech Stack

* **Framework & UI**: [Streamlit](https://streamlit.io/)
* **LLM Core**: **NVIDIA NIMs** (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) accessed through the standard `openai` Python SDK (`OpenAI` client)
* **Document Processing**: `PyPDFDirectoryLoader`, `RecursiveCharacterTextSplitter` (LangChain Community & Text Splitters)
* **Tokenizer & Utilities**: `tiktoken` (for token length counting), `python-dotenv`

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have Python 3.10+ installed and an active NVIDIA NIM API Key.

### 2. Environment Setup

Create a `.env` file in the project root directory:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here