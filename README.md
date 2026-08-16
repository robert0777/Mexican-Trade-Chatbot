# 📦 Agentic AI Mexican Customs & Trade Assistant

An **Agentic AI System** developed to assist importers, exporters, and customs brokers navigating the **US-Mexico** trade corridor[cite: 3]. 

Unlike conventional chatbots, this system operates as an **autonomous consultor**: it plans multi-step tasks, queries official Mexican and US trade legislation (*LIGIE/HTS, T-MEC/USMCA, Ley Aduanera, CFF*) using a local RAG engine, performs deterministic landed-cost calculations (*CIF, IGIE, DTA, IVA*), and generates structured compliance reports[cite: 3].

---

## 🌟 Key Features

* **Autonomous ReAct Agent Loop**: Utilizes **LangGraph** and **NVIDIA NIMs** (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) to plan, execute, and observe multi-step trade analysis workflows[cite: 3].
* **Legal Grounding (RAG)**: Ingests official trade laws, tariff schedules, and regulatory manuals (*Ley Aduanera, CFF, Anexo 22, USMCA/T-MEC*) from `./pdf_files_comercio_exterior` to cite exact legal foundations[cite: 3].
* **Deterministic Tax & Duty Calculator**: A dedicated calculation framework that computes exact base values (*CIF/FOB*), import duties (*IGIE*), customs handling fees (*DTA*), and value-added tax (*IVA*) with clear currency formatting ($USD,$ MXN)[cite: 3].
* **Risk & Compliance Reporting**: Generates technical trade opinions finalized with a **RAID (Risks, Actions, Issues, Decisions)** matrix table and explicit legal disclaimers[cite: 3].
* **Interactive UI**: Displays real-time reasoning loops, document source extracts, and query handling via a streamlined Streamlit interface[cite: 3].

---

## 🛠️ Architecture & Tech Stack

* **Framework & UI**: [Streamlit](https://streamlit.io/)[cite: 3]
* **Agentic Orchestration**: [LangGraph](https://www.langchain.com/langgraph) & [LangChain](https://www.langchain.com/)[cite: 3]
* **LLM Core**: NVIDIA NIMs (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) via `openai` / `langchain-nvidia-ai-endpoints`[cite: 3]
* **PDF Document Ingestion & RAG**: `PyPDFDirectoryLoader`, `RecursiveCharacterTextSplitter`, `tiktoken`[cite: 3]
* **Report Generation**: [ReportLab](https://www.reportlab.com/)[cite: 2, 3]

---

## 📂 Project Structure

```text
customs-agent-ai/
│
├── app.py                         # Main Streamlit application and Agent logic
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .env.example                   # Template for environment variables
└── pdf_files_comercio_exterior/   # Local directory for reference PDFs (Ley Aduanera, T-MEC, CFF, HTS)