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

# Load environment variables
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
    """Loads and splits PDFs from the ./pdf_files_comercio_exterior directory."""
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
    """Selects top matching text chunks based on word overlap relevance."""
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

# ==============================================================================
# 2. Autonomous Agentic Tools (USMCA / T-MEC Focused)
# ==============================================================================
@tool
def search_usmca_trade_regulations(query: str) -> str:
    """
    Searches loaded PDF documents for USMCA/T-MEC agreement rules, Mexican customs regulations, TIGIE tariffs, and NOM safety standards.
    """
    if "documents" not in st.session_state or not st.session_state.documents:
        return "No local PDF documents loaded in memory."
    
    chunks = select_relevant_chunks(query, st.session_state.documents)
    if not chunks:
        return "No relevant text chunks found in local docs."
        
    context = "\n\n".join([f"[Doc: {Path(c.metadata['source']).name}]\n{c.page_content}" for c in chunks])
    return context

@tool
def calculate_usmca_import_duties(invoice_value: float, freight: float, insurance: float, ige_duty_rate: float, has_usmca_certificate: bool) -> dict:
    """
    Calculates CIF base value, ad-valorem duty (IGE), customs processing fee (DTA), Value Added Tax (IVA 16%), and total landed cost for USMCA imports into Mexico.
    """
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

@tool
def generate_usmca_pdf_report(title: str, executive_summary: str, cost_breakdown: str, document_checklist: str) -> str:
    """
    Generates a downloadable official USMCA compliance PDF report and document checklist in English or Spanish.
    Call this tool whenever the user requests a PDF report, official dictamen, or downloadable summary.
    """
    pdf_filename = "USMCA_Customs_Compliance_Report.pdf"
    pdf_path = Path(pdf_filename)
    
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1D4ED8'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = styles['BodyText']
    body_style.spaceAfter = 8

    story = []
    lang = st.session_state.get('lang', 'English')
    
    sub_title = "Agente de Comercio Exterior T-MEC" if lang == "Español" else "USMCA / T-MEC Trade Compliance Agent"
    sec1 = "1. Resumen Ejecutivo y Dictamen" if lang == "Español" else "1. Executive Summary & Assessment"
    sec2 = "2. Desglose Estimado de Impuestos (CIF/IVA)" if lang == "Español" else "2. Estimated Duty & Tax Breakdown (CIF/VAT)"
    sec3 = "3. Lista de Verificación Documental T-MEC" if lang == "Español" else "3. USMCA Clearance Document Checklist"

    story.append(Paragraph(f"<b>{title}</b>", title_style))
    story.append(Paragraph(f"<b>{sub_title}</b>", styles['Normal']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(sec1, heading_style))
    story.append(Paragraph(executive_summary.replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(sec2, heading_style))
    story.append(Paragraph(cost_breakdown.replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(sec3, heading_style))
    story.append(Paragraph(document_checklist.replace('\n', '<br/>'), body_style))
    
    doc.build(story)
    
    st.session_state['latest_pdf_generated'] = str(pdf_path)
    return f"PDF report generated: {pdf_filename}"

tools = [search_usmca_trade_regulations, calculate_usmca_import_duties, generate_usmca_pdf_report]

# ==============================================================================
# 3. Agent Executor Initialization (Strict Language & PDF Instruction)
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
        "- If the user writes in English, you MUST respond ENTIRELY in English.\n"
        "- If the user writes in Spanish, you MUST respond ENTIRELY in Spanish.\n"
        "- NEVER respond in Spanish if the user asked their question in English, even if the retrieved legal PDF context is in Spanish. Translate the context into English.\n"
        "PDF GENERATION RULE:\n"
        "- Whenever the user asks for a PDF, report, dictamen, checklist, or downloadable summary, YOU MUST CALL the `generate_usmca_pdf_report` tool.\n"
        "- Provide well-formatted string arguments for title, executive_summary, cost_breakdown, and document_checklist in the user's language.\n"
        "REASONING & TOOLS:\n"
        "- Use tools autonomously to search USMCA regulations, compute duties, or export PDF reports."
    )
    
    return create_react_agent(llm, tools, prompt=system_prompt)

# ==============================================================================
# 4. Streamlit User Interface
# ==============================================================================
st.set_page_config(layout="wide", page_title="USMCA Trade Agent", page_icon="📦")

# Sidebar - Language Selection & Settings
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

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.8rem; color: #6B7280; text-align: center;">
            © 2026 USMCA Agent AI
        </div>
    """, unsafe_allow_html=True)

# UI Localization Setup
if st.session_state.lang == "English":
    title_text = "📦 USMCA / T-MEC Autonomous Trade Agent"
    desc_text = """
    > **Specialized USMCA (US-Mexico-Canada) Compliance Assistant**  
    > 1. **Strict Language Matching**: Ask questions in English or Spanish, and receive full responses in your selected language.  
    > 2. **USMCA Trade Tools**: Queries USMCA tariff schedules, calculates landed duties (*CIF, IGE, DTA, VAT*), and **generates PDF compliance reports**.  
    """
    input_placeholder = "Describe your USMCA shipment or request a PDF compliance report..."
    status_label = "🧠 Agent is analyzing query and invoking USMCA tools..."
    tool_exec_label = "🛠️ **Executing Tool:**"
    tool_obs_label = "👁️ **Tool Observation:**"
    result_header = "### 📝 Compliance Assessment & Action Plan:"
    download_btn_label = "📄 Download USMCA PDF Report"
else:
    title_text = "📦 Agente Autónomo de Comercio Exterior T-MEC"
    desc_text = """
    > **Asistente Especializado en Cumplimiento T-MEC (México-EUA-Canadá)**  
    > 1. **Correspondencia de Idioma**: Formule preguntas en Inglés o Español y reciba respuestas completas en ese idioma.  
    > 2. **Herramientas T-MEC**: Consulta la TIGIE, calcula impuestos (*CIF, IGE, DTA, IVA*) y **genera reportes PDF**.  
    """
    input_placeholder = "Describa su embarque T-MEC o solicite un dictamen en PDF..."
    status_label = "🧠 El Agente está evaluando la consulta y ejecutando herramientas..."
    tool_exec_label = "🛠️ **Ejecutando Herramienta:**"
    tool_obs_label = "👁️ **Observación de Herramienta:**"
    result_header = "### 📝 Dictamen y Plan de Acción:"
    download_btn_label = "📄 Descargar Dictamen Oficial en PDF"

st.title(title_text)
st.markdown(desc_text)

# Initialize Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Persistent PDF Download Button ---
if 'latest_pdf_generated' in st.session_state and os.path.exists(st.session_state['latest_pdf_generated']):
    pdf_path = st.session_state['latest_pdf_generated']
    with open(pdf_path, "rb") as file:
        st.download_button(
            label=download_btn_label,
            data=file,
            file_name="USMCA_Compliance_Report.pdf",
            mime="application/pdf",
            key="persistent_pdf_download"
        )

# User Chat Loop
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
                
                # Render intermediate tool execution steps
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

            # Safely extract final assistant response text
            final_answer = ""
            for msg in reversed(response["messages"]):
                if msg.type == "ai" and isinstance(msg.content, str) and msg.content.strip():
                    final_answer = msg.content
                    break
            
            if not final_answer:
                final_answer = "Analysis complete. Please review the tool outputs above."

            # Render Final Response
            st.markdown(result_header)
            st.markdown(final_answer)
            
            # Trigger immediate UI refresh if a PDF was just generated
            if 'latest_pdf_generated' in st.session_state and os.path.exists(st.session_state['latest_pdf_generated']):
                st.rerun()

            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")