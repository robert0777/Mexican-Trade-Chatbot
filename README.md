# 📦 Mexican Customs & Trade SME Chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/) 

A **RAG-powered conversational assistant** built with Streamlit and OpenRouter to assist importers, exporters, and customs brokers navigating the **US-Mexico (USMCA / T-MEC)** trade corridor and Mexican customs legislation.

👉 **[Try the Live App: Asesor AI de Comercio Exterior y Aduanas](https://mexican-trade-chatbot.streamlit.app/)**

---

## 🌟 Key Features

* **Dynamic In-Memory RAG Pipeline**: Ingests legal & trade documentation from `./pdf_files_comercio_exterior` using `PyPDFDirectoryLoader`, applies domain-specific legal text normalization (expanding abbreviations like *Art.*, *Fracc.*, *L.A.*, *CFF*), and adaptively splits text into chunks via `RecursiveCharacterTextSplitter` calculated from total document volume.
* **Smart Context Selection**: Ranks document chunks by keyword overlap and length scaling metrics to dynamically select diverse source chunks within a 5,000-token context boundary using `tiktoken`.
* **Streaming Multi-Model LLM Engine with Active Model Display**: Integrates with **OpenRouter API** via a streaming fallback architecture (`generate_completion_with_fallback`). Automatically cycles through active free endpoints (`minimax/minimax-m3:free`, `google/gemma-4-31b:free`, `cohere/north-mini-code:free`, `openrouter/free-models-router`) and displays the exact active model executing the query: `📝 Respuesta (Modelo activo: minimax/minimax-m3:free):`.
* **Risk & Compliance Focus**: Generates structured technical compliance reports with specific legal citations (*Ley Aduanera, Anexo 22, CFF, T-MEC*), tax breakdowns (IGE, IVA, DTA), disclaimers, and a mandatory **RAID (Risks, Actions, Issues, Decisions)** matrix table.
* **Contextual Greeting Handler**: Detects conversational greetings in Spanish without triggering unnecessary LLM token calls and provides time-aware responses (*Buenos días / Tardes / Noches*).
* **Interactive UI & Source Verification**: Streamlit wide-layout interface featuring custom sidebar branding, author academic credentials, execution timing metrics, and expandable panels displaying source chunks per document.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend / UI**: Streamlit (`>=1.30.0`)
* **LLM Engine**: OpenRouter API via standard `openai` SDK (`openai>=1.0.0`)
* **Document Processing**: `langchain-community`, `langchain-text-splitters`, `pypdf`
* **Token Encoding & Utilities**: `tiktoken`, `python-dotenv`

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/robert0777/mexican-trade-chatbot.git
cd mexican-trade-chatbot
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 5. Add Document PDFs
Place your target PDF files into the data directory:
```text
./pdf_files_comercio_exterior/
```

### 6. Run the Application
```bash
streamlit run app_3.py
```

---

## 📂 Project Structure

```text
.
├── app_3.py                          # Main Streamlit application with active model streaming
├── pdf_files_comercio_exterior/      # Directory containing legal & trade PDF files
├── ai-advisor-icon.svg               # Application branding icon
├── requirements.txt                  # Python package dependencies
├── .env                              # Environment variable configuration (OPENROUTER_API_KEY)
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
