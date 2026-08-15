# 📦 Agentic AI Mexican Customs & Trade Assistant

An **Agentic AI System** developed to assist importers, exporters, and customs brokers navigating the **US-Mexico** and **EU-Mexico** trade corridors. 

Unlike conventional chatbots, this system operates as an **autonomous consultor**: it plans multi-step tasks, queries official Mexican trade legislation (*TIGIE, T-MEC, NOMs*) using a local RAG engine, performs deterministic landed-cost calculations (*CIF, IGE, DTA, IVA*), and generates downloadable official compliance reports in PDF.

---

## 🌟 Key Features

* **Autonomous ReAct Agent Loop**: Utilizes **LangGraph** and **NVIDIA NIMs** (`llama-3.3-nemotron-super-49b-v1.5`) to plan, execute, and observe multi-step trade analysis workflows.
* **Legal Grounding (RAG)**: Ingests local trade manuals, tariff schedules, and regulatory gazettes (*TIGIE, SAT/VUCEM, NOMs*) to cite official legal foundations.
* **Deterministic Tax & Duty Calculator**: A dedicated code-execution tool that computes exact base values (*CIF*), import duties (*IGE*), customs handling fees (*DTA*), and value-added tax (*IVA*).
* **Automated PDF Export**: Generates downloadable, styled PDF reports containing executive summaries, itemized cost breakdowns, and document clearance checklists.
* **Transparent Execution**: Displays real-time reasoning loops, tool invocations, and parameter selections directly in the Streamlit UI.

---

## 🛠️ Architecture & Tech Stack

* **Framework & UI**: [Streamlit](https://streamlit.io/)
* **Agentic Orchestration**: [LangGraph](https://www.langchain.com/langgraph) & [LangChain](https://www.langchain.com/)
* **LLM Core**: NVIDIA NIMs (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) via `langchain-nvidia-ai-endpoints`
* **PDF Document Ingestion & RAG**: `PyPDFDirectoryLoader`, `RecursiveCharacterTextSplitter`, `tiktoken`
* **PDF Report Generation**: [ReportLab](https://www.reportlab.com/)

---

## 📂 Project Structure

```text
customs-agent-ai/
│
├── app.py                         # Main Streamlit application and Agent logic
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .env.example                   # Template for environment variables
└── pdf_files_comercio_exterior/   # Local directory for reference PDFs (TIGIE, NOMs, FTAs)