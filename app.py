import streamlit as st
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

# PDF Generation Imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# LangChain, LangGraph & NVIDIA NIM Imports
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# ==============================================================================
# 1. FAISS Vector Database Initialization (Trade Regulations & Ley Aduanera)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def initialize_trade_vector_db():
    """Builds or retrieves the FAISS vector database for Mexican Trade & USMCA laws."""
    pdf_dir = Path("./pdf_files_comercio_exterior")
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        return None

    loader = PyPDFDirectoryLoader(path=str(pdf_dir), silent_errors=True)
    docs = loader.load()
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", api_key=api_key)
    
    return FAISS.from_documents(split_docs, embeddings)

# ==============================================================================
# 2. Autonomous Trade Agent Tools
# ==============================================================================
@tool
def search_trade_regulations(query: str) -> str:
    """
    Searches loaded official trade PDFs regarding Ley Aduanera, TIGIE tariffs, 
    Anexo 22, USMCA/T-MEC rules of origin, and NOM safety/labeling regulations.
    """
    vectorstore = initialize_trade_vector_db()
    if not vectorstore:
        return "No trade regulation documents are currently loaded in memory."
    
    results = vectorstore.similarity_search(query, k=4)
    return "\n\n".join([f"[Source: {Path(d.metadata.get('source', 'Doc')).name}]\n{d.page_content}" for d in results])

@tool
def calculate_mexican_import_duties(
    invoice_value_usd: float,
    freight_usd: float,
    insurance_usd: float,
    exchange_rate_mxn: float,
    ige_duty_rate: float,
    has_usmca_certificate: bool,
    apply_border_vat: bool = False
) -> dict:
    """
    Calculates the complete Mexican Pedimento tax structure in MXN:
    CIF Value, Ad-Valorem Duty (IGE), Customs Processing Fee (DTA), 
    Prevalidation (PRV), Value Added Tax (IVA 16% or 8%), and Total Payable Taxes.
    """
    cif_usd = invoice_value_usd + freight_usd + insurance_usd
    cif_mxn = cif_usd * exchange_rate_mxn
    
    # 1. Impuesto General de Importación (IGE) - 0% under USMCA preferential origin
    effective_ige_rate = 0.0 if has_usmca_certificate else ige_duty_rate
    ige_amount = cif_mxn * effective_ige_rate
    
    # 2. Derecho de Trámite Aduanero (DTA) - Fixed quota under USMCA Art. 5.11 vs 0.8%
    dta_amount = 410.0 if has_usmca_certificate else max(410.0, cif_mxn * 0.008)
    
    # 3. Prevalidación (PRV) - Fixed fee (~310 MXN average)
    prv_amount = 310.0
    
    # 4. Impuesto al Valor Agregado (IVA)
    vat_rate = 0.08 if apply_border_vat else 0.16
    taxable_base_iva = cif_mxn + ige_amount + dta_amount
    iva_amount = taxable_base_iva * vat_rate
    
    total_taxes = ige_amount + dta_amount + prv_amount + iva_amount
    
    return {
        "cif_value_usd": round(cif_usd, 2),
        "cif_value_mxn": round(cif_mxn, 2),
        "ige_duty_mxn": round(ige_amount, 2),
        "dta_fee_mxn": round(dta_amount, 2),
        "prv_fee_mxn": round(prv_amount, 2),
        "iva_vat_mxn": round(iva_amount, 2),
        "total_payable_taxes_mxn": round(total_taxes, 2)
    }

@tool
def generate_customs_compliance_report(
    title: str, 
    executive_summary: str, 
    tax_breakdown: str, 
    document_checklist: str
) -> str:
    """
    Generates a formal PDF trade compliance dictamen and clearance checklist.
    Call this tool whenever the user asks for a report, PDF, or official assessment.
    """
    pdf_filename = "USMCA_Customs_Compliance_Report.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#0F172A'))
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#1E40AF'))
    body_style = styles['BodyText']
    body_style.fontSize = 9
    
    story = [
        Paragraph(f"<b>{title}</b>", title_style),
        Paragraph("<b>Dictamen Técnico de Comercio Exterior T-MEC</b>", styles['Normal']),
        Spacer(1, 10),
        Paragraph("1. Resumen Ejecutivo y Marco Jurídico", heading_style),
        Paragraph(executive_summary.replace('\n', '<br/>'), body_style),
        Spacer(1, 8),
        Paragraph("2. Estimación de Contribuciones (Pedimento)", heading_style),
        Paragraph(tax_breakdown.replace('\n', '<br/>'), body_style),
        Spacer(1, 8),
        Paragraph("3. Lista de Verificación Documental (VUCEM)", heading_style),
        Paragraph(document_checklist.replace('\n', '<br/>'), body_style)
    ]
    
    doc.build(story)
    st.session_state['latest_pdf_generated'] = pdf_filename
    return f"Reporte PDF generado exitosamente: {pdf_filename}"

tools = [search_trade_regulations, calculate_mexican_import_duties, generate_customs_compliance_report]

# ==============================================================================
# 3. ReAct Agent Core Setup
# ==============================================================================
@st.cache_resource
def get_trade_agent():
    api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
    if not api_key:
        st.error("NVIDIA_API_KEY not configured.")
        st.stop()
        
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.1,
        api_key=api_key
    )
    
    system_prompt = (
        "You are an expert Autonomous Customs & Trade Agent specializing in Mexican Customs Law (Ley Aduanera), "
        "TIGIE tariff schedules, VUCEM clearance processes, and USMCA/T-MEC preferential rules of origin.\n"
        "OPERATIONAL RULES:\n"
        "1. LANGUAGE: Match the user's language (English or Spanish) in all final responses.\n"
        "2. TOOLS: Execute `search_trade_regulations` for legal validation, `calculate_mexican_import_duties` for taxes, "
        "and `generate_customs_compliance_report` whenever a PDF output is requested.\n"
        "3. FINAL OUTPUT: Always conclude tool executions with a detailed summary in the chat response."
    )
    
    return create_react_agent(llm, tools, prompt=system_prompt)

# ==============================================================================
# 4. Streamlit Interactive Application
# ==============================================================================
st.set_page_config(layout="wide", page_title="USMCA Customs Agent", page_icon="🛃")

st.title("🛃 Agente Autónomo de Comercio Exterior y T-MEC")
st.caption("Asistente AI de Despacho Aduanero, Cálculo de Contribuciones y Cumplimiento Regulatorio")

with st.sidebar:
    st.markdown("### 👤 Autor")
    st.markdown("**Dr. Robert Hernández Martínez**")
    
    if st.button("🔄 Indexar Leyes y Reglamentos Trade PDF"):
        with st.spinner("Indexando reglamentos en FAISS..."):
            v_db = initialize_trade_vector_db()
            if v_db:
                st.success("Base de datos de comercio exterior lista.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Download button for generated PDF
if 'latest_pdf_generated' in st.session_state and os.path.exists(st.session_state['latest_pdf_generated']):
    with open(st.session_state['latest_pdf_generated'], "rb") as file:
        st.download_button(
            label="📄 Descargar Dictamen Técnico (PDF)",
            data=file,
            file_name="USMCA_Compliance_Report.pdf",
            mime="application/pdf"
        )

if prompt := st.chat_input("Describa el embarque, consulte reglas de origen T-MEC o solicite un cálculo de pedimento..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            agent = get_trade_agent()
            with st.status("Evaluando normatividad y ejecutando cálculos aduanales...", expanded=True) as status:
                response = agent.invoke({"messages": [("user", prompt)]})
                status.update(label="Análisis aduanal completado", state="complete", expanded=False)

            # Robust response extraction
            ai_texts = [
                msg.content for msg in response["messages"] 
                if msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None)
            ]
            final_answer = "\n\n".join(ai_texts) if ai_texts else "Análisis completado."

            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            st.error(f"Error de ejecución del agente: {str(e)}")