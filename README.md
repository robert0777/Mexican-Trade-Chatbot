# 📦 Agentic AI Mexican Customs & Trade Assistant

An **Agentic AI System** developed to assist importers, exporters, and customs brokers navigating the **US-Mexico** trade corridor[cite: 6]. 

Unlike conventional chatbots, this system operates as an **autonomous consultor**: it plans multi-step tasks, queries official Mexican and US trade legislation (*LIGIE/HTS, T-MEC/USMCA, Ley Aduanera, CFF*) using a local RAG engine, performs deterministic landed-cost calculations (*CIF, IGIE, DTA, IVA*), and generates structured compliance reports[cite: 6].

---

## 🌟 Key Features

* **Autonomous ReAct Agent Loop**: Utilizes **LangGraph** and **NVIDIA NIMs** (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) to plan, execute, and observe multi-step trade analysis workflows[cite: 6].
* **Legal Grounding (RAG)**: Ingests official trade laws, tariff schedules, and regulatory manuals (*Ley Aduanera, CFF, Anexo 22, USMCA/T-MEC*) from `./pdf_files_comercio_exterior` to cite exact legal foundations[cite: 6].
* **Deterministic Tax & Duty Calculator**: A dedicated calculation framework that computes exact base values (*CIF/FOB*), import duties (*IGIE*), customs handling fees (*DTA*), and value-added tax (*IVA*) with clear currency formatting ($USD,$MXN)[cite: 6].
* **Risk & Compliance Reporting**: Generates technical trade opinions finalized with a **RAID (Risks, Actions, Issues, Decisions)** matrix table and explicit legal disclaimers[cite: 6].
* **Interactive UI**: Displays real-time reasoning loops, document source extracts, and query handling via a streamlined Streamlit interface[cite: 6].

---

## 🛠️ Architecture & Tech Stack

* **Framework & UI**: <a href="https://streamlit.io/" target="_blank" rel="noopener noreferrer">Streamlit</a>[cite: 6]
* **Agentic Orchestration**: <a href="https://www.langchain.com/langgraph" target="_blank" rel="noopener noreferrer">LangGraph</a> & <a href="https://www.langchain.com/" target="_blank" rel="noopener noreferrer">LangChain</a>[cite: 6]
* **LLM Core**: NVIDIA NIMs (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) via `openai` / `langchain-nvidia-ai-endpoints`[cite: 6]
* **PDF Document Ingestion & RAG**: `PyPDFDirectoryLoader`, `RecursiveCharacterTextSplitter`, `tiktoken`[cite: 6]
* **Report Generation**: <a href="https://www.reportlab.com/" target="_blank" rel="noopener noreferrer">ReportLab</a>[cite: 6]

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