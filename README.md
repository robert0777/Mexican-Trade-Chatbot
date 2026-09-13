# 📦 Mexican Customs & Trade SME Chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/) 

A **RAG-powered conversational assistant** built with Streamlit and OpenRouter to assist importers, exporters, and customs brokers navigating the **US-Mexico (USMCA / T-MEC)** trade corridor and Mexican customs legislation.

👉 **[Try the Live App: Asesor AI de Comercio Exterior y Aduanas](https://mexican-trade-chatbot.streamlit.app/)**

---

## 🌟 Key Features

* **Dynamic In-Memory RAG Pipeline**: Ingests legal & trade documentation from `./pdf_files_comercio_exterior` using `PyPDFDirectoryLoader`, applies domain-specific legal text normalization (expanding abbreviations like *Art.*, *Fracc.*, *L.A.*, *CFF*), and adaptively splits text into chunks via `RecursiveCharacterTextSplitter` calculated from total document volume.
* **Smart Context Selection**: Ranks document chunks by keyword overlap and length scaling metrics (`calculate_chunk_relevance`) to dynamically select diverse source chunks within a 5,000-token context boundary using `tiktoken`.
* **Streaming Multi-Model LLM Engine with Fallback**: Integrates with **OpenRouter API** via a streaming fallback architecture (`generate_completion_with_fallback`). Automatically cycles through active free endpoints (`nex-agi/nex-n2.5-pro:free`, `thinking-machines/inkling-small:free`, `cohere/north-mini-code:free`, `nex-agi/nex-n2.5-mini:free`, `google/gemma-4-31b:free`, `google/gemma-4-26b-a4b:free`, `nvidia/nemotron-3-nano-omni:free`, `minimax/minimax-m3:free`, `openrouter/free-models-router`) and displays the active model executing the query: `📝 Respuesta (Modelo activo: nex-agi/nex-n2.5-pro:free):`.
* **Risk & Compliance Focus**: Generates structured technical compliance reports with specific legal citations (*Ley Aduanera, Anexo 22, CFF, T-MEC*), tax breakdowns (IGE, IVA, DTA), disclaimers, and a mandatory **RAID (Risks, Actions, Issues, Decisions)** matrix table.
* **Contextual Greeting & Intent Handler**: Detects Spanish conversational greetings without triggering unnecessary LLM token consumption. Returns time-aware responses (*Buenos días / Tardes / Noches*) and displays domain guidance when no technical query is appended.
* **Secure Key Management**: Uses native Streamlit Secrets (`st.secrets["OPENROUTER_API_KEY"]`) as the primary credential source with fallback to environment variables.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend / UI**: Streamlit (`>=1.30.0`)
* **LLM Engine**: OpenRouter API via standard `openai` SDK (`openai>=1.0.0`)
* **Document Processing**: `langchain-community`, `langchain-text-splitters`, `pypdf`
* **Token Encoding & Utilities**: `tiktoken`

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/robert0777/mexican-trade-chatbot.git
cd mexican-trade-chatbot
```

### 2. Streamlit Cloud Secrets Setup
In your deployed app dashboard on Streamlit Cloud, navigate to **Settings** -> **Secrets**, and store your OpenRouter API key:
```toml
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
```

### 3. Add Target PDF Documents
Place official Mexican Customs & Trade Compliance documentation PDF files into:
```text
./pdf_files_comercio_exterior/
```

### 4. Application Execution Entry
Point Streamlit Cloud deployment to:
```text
app.py
```

---

## 📦 Dependencias (`requirements.txt`)

```txt
streamlit>=1.30.0
langchain-core>=0.1.0
langchain-community>=0.0.20
langchain-nvidia-ai-endpoints>=0.1.0
langgraph>=0.0.20
openai>=1.0.0
tiktoken>=0.5.0
pypdf>=3.0.0
reportlab>=4.0.0
faiss-cpu>=1.7.4
```

---

## 📂 Project Structure

```text
.
├── app.py                            # Main Streamlit application file
├── pdf_files_comercio_exterior/      # Directory containing legal & trade PDF files
├── ai-advisor-icon.svg               # Application branding icon
├── requirements.txt                  # Python package dependencies
└── README.md                         # Project documentation
```

---

## 👤 Author

**Dr. Robert Hernández Martínez**  
*Consultant in Actuarial Science, Finance, Risk Modeling, and Applied AI*

* 📝 [Articles on Medium](https://chomchom216.medium.com/)
* 🎓 [Academic Publications](https://unam1.academia.edu/Robert_Hernandez_Martinez)
* 🏆 [Credentials on Credly](https://www.credly.com/users/robert-hernandez.89bffe7b)
* 🐙 [GitHub Profile](https://github.com/robert0777)
* 📧 Email: [robert@actuariayfinanzas.net](mailto:robert@actuariayfinanzas.net)
