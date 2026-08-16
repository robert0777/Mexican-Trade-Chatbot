# 🛃 Agentic AI Mexican Customs & Trade Assistant

An **Agentic AI System (SME)** developed to assist importers, exporters, and customs brokers navigating the **US-Mexico (USMCA/T-MEC)** and **EU-Mexico (TLCUEM)** trade corridors.

Operating as an autonomous specialist in official Mexican and international trade legislation, this agent analyzes legal frameworks, performs precision cost and contribution calculations, and generates compliance reports complete with **(RAID) Risk, Actions, Issues, and Decisions** evaluations.

---

## 🌟 Key Features

* **Agentic Trade SME Persona**: Acts as an expert advisor on Mexican tariff classification (HS/TIGIE), USMCA/T-MEC Rules of Origin, and EU-Mexico trade agreements.
* **Legal Ingestion & Grounding (RAG)**: Ingests documents from `./pdf_files_comercio_exterior` (*Ley Aduanera, LIGIE, RGCE, CFF, HTS, USMCA*) using `PyPDFDirectoryLoader` and custom chunking logic.
* **Deterministic Financial Calculations**: Computes base values and import taxes (*IGIE, DTA, IVA, ISAN, IEPS*) formatted accurately in respective currencies ($USD,$ MXN, € EUR) and percentages.
* **RAID Matrix Compliance Standard**: Every formal consultation concludes with a structured **(RAID) Risk, Actions, Issues, Decisions** markdown table for clear risk assessment.
* **Legal References & Disclaimers**: Cites specific articles, laws, and annexes for further consultation while supplying mandatory regulatory disclaimers.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend & UX**: [Streamlit](https://streamlit.io/)
* **LLM Engine**: NVIDIA NIM (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) via OpenAI API endpoints
* **Agentic Orchestration**: [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph)
* **Document Processing & RAG**: `pypdf`, `RecursiveCharacterTextSplitter`, `tiktoken`
* **PDF Report Exports**: `reportlab`

---

## 📂 Project Structure

```text
customs-agent-ai/
│
├── app.py                         # Streamlit application containing UI & Trade SME Agent
├── requirements.txt               # Dependencies (Streamlit, OpenAI, LangChain, LangGraph, etc.)
├── README.md                      # Updated project documentation
├── .env                           # Environment variables (NVIDIA_API_KEY)
└── pdf_files_comercio_exterior/   # Reference directory for trade laws, gazettes, and T-MEC texts