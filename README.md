# 📦 Agentic AI Mexican Customs & Trade Assistant

An **Agentic AI System** developed to assist importers, exporters, and customs brokers navigating the **US-Mexico** trade corridor. 

Unlike conventional chatbots, this system operates as an **autonomous consultor**: it plans multi-step tasks, queries official Mexican and US trade legislation (*LIGIE/HTS, T-MEC/USMCA, Ley Aduanera, CFF*) using a local RAG engine, performs deterministic landed-cost calculations (*CIF, IGIE, DTA, IVA*), and generates structured compliance reports.

---

## 🌟 Key Features

* **Autonomous ReAct Agent Loop**: Utilizes **LangGraph** and **NVIDIA NIMs** (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) to plan, execute, and observe multi-step trade analysis workflows.
* **Legal Grounding (RAG)**: Ingests official trade laws, tariff schedules, and regulatory manuals (*Ley Aduanera, CFF, Anexo 22, USMCA/T-MEC*) from `./pdf_files_comercio_exterior` to cite exact legal foundations.
* **Deterministic Tax & Duty Calculator**: A dedicated calculation framework that computes exact base values (*CIF/FOB*), import duties (*IGIE*), customs handling fees (*DTA*), and value-added tax (*IVA*) with clear currency formatting ($USD,$ MXN).
* **Risk & Compliance Reporting**: Generates technical trade opinions finalized with a **RAID (Risks, Actions, Issues, Decisions)** matrix table and explicit legal disclaimers.
* **Interactive UI**: Displays real-time reasoning loops, document source extracts, and query handling via a streamlined Streamlit interface.

---

## 🛠️ Architecture & Tech Stack

* **Framework & UI**: [Streamlit](https://streamlit.io/)
* **Agentic Orchestration**: [LangGraph](https://www.langchain.com/langgraph) & [LangChain](https://www.langchain.com/)
* **LLM Core**: NVIDIA NIMs (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) via `openai` / `langchain-nvidia-ai-endpoints`
* **PDF Document Ingestion & RAG**: `PyPDFDirectoryLoader`, `RecursiveCharacterTextSplitter`, `tiktoken`
* **Report Generation**: [ReportLab](https://www.reportlab.com/)

---

## 📂 Project Structure

```text
customs-agent-ai/
│
├── app.py                         # Main Streamlit application and Agent logic
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .env                           # Template for environment variables
└── pdf_files_comercio_exterior/   # Local directory for reference PDFs (Ley Aduanera, T-MEC, CFF, HTS)