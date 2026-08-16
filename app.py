import streamlit as st
import os
import re
import time
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

# PDF Generation Imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# LangChain, LangGraph & Model Imports
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

load_dotenv()

# ==============================================================================
# 1. RAG & Helper Utilities
# ==============================================================================
@lru_cache(maxsize=None)
def get_tokenizer():
    return tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text))

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_documents():
    if "documents" not in st.session_state:
        pdf_dir = Path("./pdf_files_comercio_exterior")
        if not pdf_dir.exists():
            pdf_dir.mkdir(parents=True, exist_ok=True)
            
        loader = PyPDFDirectoryLoader(path="./pdf_files_comercio_exterior", silent_errors=True)
        docs = loader.load()
        if docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=350, 
                chunk_overlap=50, 
                length_function=count_tokens
            )
            for doc in docs:
                doc.page_content = normalize_text(doc.page_content)
            st.session_state.documents = text_splitter.split_documents(docs)
        else:
            st.session_state.documents = []

def select_relevant_chunks(question: str, chunks: list, max_total_tokens: int = 2000) -> list:
    question_words = set(question.lower().split())
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk.page_content.lower().split())
        overlap = len(question_words.intersection(chunk_words))
        scored_chunks.append((chunk, overlap))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    selected = []
    current_tokens = 0
    for chunk, score in scored_chunks:
        tokens = count_tokens(chunk.page_content)
        if current_tokens + tokens <= max_total_tokens:
            selected.append(chunk)
            current_tokens += tokens
    return selected

# Direct PDF Builder Function (Bypasses LLM Tool Execution Delays)
def build_pdf_report(title: str, body_text: str, filename: str = "USMCA_Compliance_Report.pdf") -> str:
    pdf_path = Path(filename)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1E3A8A'), spaceAfter=12)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#1D4ED8'), spaceBefore=10, spaceAfter=6)
    body_style = styles['BodyText']

    story = [
        Paragraph(f"<b>{title}</b>", title_style),
        Paragraph("<b>USMCA / T-MEC Official Trade Assessment</b>", styles['Normal']),
        Spacer(1, 12),
        Paragraph("Executive Assessment & Action Plan", heading_style),
        Paragraph(body_text.replace('\n', '<br/>'), body_style)
    ]
    doc.build(story)
    return str(pdf_path)

# ==============================================================================
# 2. Autonomous Agentic Tools
# ==============================================================================
@tool
def search_usmca_trade_regulations(query: str) -> str:
    """Searches loaded PDF documents for USMCA/T-MEC agreement rules, tariffs, and regulations."""
    if "documents" not in st.session_state or not st.session_state.documents:
        return "No local PDF documents loaded in memory."
    
    chunks = select_relevant_chunks(query, st.session_state.documents)
    if not chunks:
        return "No relevant text chunks found in local docs."
        
    context = "\n\n".join([f"[Doc: {Path(c.metadata['source']).name}]\n{c.page_content}" for c in chunks])
    return context

@tool
def calculate_usmca_import_duties(invoice_value: float, freight: float, insurance: float, ige_duty_rate: float, has_usmca_certificate: bool) -> dict:
    """Calculates CIF base value, ad-valorem duty (IGE), customs handling fee (DTA), Value Added Tax (IVA 16%), and total landed cost."""
    cif = invoice_value + freight + insurance
    effective_ige = 0.0 if has_usmca_certificate else ige_duty_rate
    ige_amount = cif * effective_ige
    dta_amount = 410.0 if has_usmca_certificate else (cif * 0.008)
    iva_amount = (cif + ige_amount + dta_amount) * 0.16
    total_landed = cif + ige_amount + dta_amount + iva_amount
    
    return {
        "cif_base_value": round(cif, 2),
        "ige_duty_amount": round(ige_amount, 2),
        "dta_customs_fee": round(dta_amount, 2),
        "iva_vat_16_percent": round(iva_amount, 2),
        "total_landed_cost": round(total_landed, 2)
    }

tools = [search_usmca_trade_regulations, calculate_usmca_import_duties]

# ==============================================================================
# 3. Agent Executor Initialization
# ==============================================================================
@st.cache_resource
def get_agent_executor():
    api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
    if not api_key:
        st.error("NVIDIA_API_KEY not configured. Add it to .env or st.secrets.")
        st.stop()
        
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.1,
        api_key=api_key,
        timeout=180,  
        max_retries=3  
    )
    
    system_prompt = (
        "You are an expert Autonomous Trade Agent focusing strictly on USMCA / T-MEC trade between Mexico, the USA, and Canada.\n"
        "CRITICAL LANGUAGE RULE:\n"
        "- If the user writes in English, respond ENTIRELY in English.\n"
        "- If the user writes in Spanish, respond ENTIRELY in Spanish.\n"
        "REASONING & CALCULATIONS:\n"
        "- Always calculate duties using `calculate_usmca_import_duties` when values are provided.\n"
        "- Detail phytosanitary (USDA/APHIS, SENASICA) and customs clearance requirements."
    )
    
    return create_react_agent(llm, tools, prompt=system_prompt)

# ==============================================================================
# 4. Streamlit User Interface
# ==============================================================================
st.set_page_config(layout="wide", page_title="USMCA Trade Agent", page_icon="📦")

# Sidebar
with st.sidebar:
    st.markdown("### 🌐 Language / Idioma")
    selected_lang = st.radio("Interface Language / Idioma", ["English", "Español"], index=0, key="lang_radio")
    st.session_state.lang = selected_lang
    
    st.markdown("---")
    st.markdown("### ⚙️ " + ("Settings" if st.session_state.lang == "English" else "Configuración"))
    st.markdown("**Author / Autor:** Dr. Robert Hernández Martínez")
    st.markdown("---")
    
    btn_label = "🔄 Load / Refresh Local PDFs" if st.session_state.lang == "English" else "🔄 Cargar / Actualizar PDFs Locales"
    if st.button(btn_label):
        with st.spinner("Processing ./pdf_files_comercio_exterior..."):
            load_documents()
            num_chunks = len(st.session_state.get('documents', []))
            if num_chunks > 0:
                msg = f"Loaded {num_chunks} document chunks." if st.session_state.lang == "English" else f"Cargados {num_chunks} fragmentos."
                st.success(msg)
            else:
                msg = "No PDFs found in ./pdf_files_comercio_exterior" if st.session_state.lang == "English" else "No se encontraron PDFs."
                st.warning(msg)

# Localization
if st.session_state.lang == "English":
    title_text = "📦 USMCA / T-MEC Autonomous Trade Agent"
    desc_text = "> **Specialized USMCA (US-Mexico-Canada) Compliance Assistant**"
    input_placeholder = "Describe your USMCA shipment..."
    status_label = "🧠 Agent is analyzing query and invoking USMCA tools..."
    tool_exec_label = "🛠️ **Executing Tool:**"
    tool_obs_label = "👁️ **Tool Observation:**"
    result_header = "### 📝 Compliance Assessment & Action Plan:"
    pdf_btn_text = "📄 Generate & Download Official PDF Report"
else:
    title_text = "📦 Agente Autónomo de Comercio Exterior T-MEC"
    desc_text = "> **Asistente Especializado en Cumplimiento T-MEC (México-EUA-Canadá)**"
    input_placeholder = "Describa su embarque T-MEC..."
    status_label = "🧠 El Agente está evaluando la consulta y ejecutando herramientas..."
    tool_exec_label = "🛠️ **Ejecutando Herramienta:**"
    tool_obs_label = "👁️ **Observación de Herramienta:**"
    result_header = "### 📝 Dictamen y Plan de Acción:"
    pdf_btn_text = "📄 Generar y Descargar Dictamen Oficial en PDF"

st.title(title_text)
st.markdown(desc_text)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Loop
if prompt := st.chat_input(input_placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            agent_executor = get_agent_executor()
            
            with st.status(status_label, expanded=True) as status:
                start_time = time.time()
                response = agent_executor.invoke({"messages": [("user", prompt)]})
                
                for message in response["messages"]:
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            st.write(f"{tool_exec_label} `{tool_call['name']}`")
                            st.json(tool_call['args'])
                    elif message.type == "tool":
                        st.write(tool_obs_label)
                        st.caption(str(message.content)[:300] + "...")
                        
                elapsed = time.time() - start_time
                status.update(label=f"Done in {elapsed:.2f}s", state="complete", expanded=False)

            # Safely parse text response
            ai_texts = []
            for msg in response["messages"]:
                if msg.type == "ai" and hasattr(msg, 'content'):
                    if isinstance(msg.content, str) and msg.content.strip():
                        ai_texts.append(msg.content.strip())

            final_answer = "\n\n".join(ai_texts) if ai_texts else "USMCA compliance analysis complete."

            st.markdown(result_header)
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

            # Render Direct PDF Generation Download Button
            pdf_path = build_pdf_report(
                title="USMCA Trade Compliance Report", 
                body_text=final_answer
            )
            
            with open(pdf_path, "rb") as file:
                st.download_button(
                    label=pdf_btn_text,
                    data=file,
                    file_name="USMCA_Compliance_Report.pdf",
                    mime="application/pdf",
                    key=f"dl_btn_{int(time.time())}"
                )

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")