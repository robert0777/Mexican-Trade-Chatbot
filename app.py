import streamlit as st
import os
import re
import time
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

# PDF Generation Imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# LangChain, LangGraph & Model Imports
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# ==============================================================================
# 1. FAISS VectorStore & RAG Engine
# ==============================================================================
@st.cache_resource(show_spinner=False)
def initialize_vector_db():
    """Builds or loads a FAISS vector database for semantic retrieval of trade docs."""
    pdf_dir = Path("./pdf_files_comercio_exterior")
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        return None

    loader = PyPDFDirectoryLoader(path=str(pdf_dir), silent_errors=True)
    docs = loader.load()
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(docs)
    
    # Using NVIDIA Embedding Endpoint for high-accuracy semantic search
    api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", api_key=api_key)
    
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# ==============================================================================
# 2. Advanced Autonomous Agentic Tools
# ==============================================================================
@tool
def search_usmca_trade_regulations(query: str) -> str:
    """
    Searches loaded legal trade documents for USMCA/T-MEC rules of origin, Mexican customs law (Ley Aduanera), 
    Anexo 22 fillable fields, TIGIE tariff classifications, and NOM safety compliance rules.
    """
    vectorstore = initialize_vector_db()
    if not vectorstore:
        return "No trade documents found in the database. Please verify PDF directory."
    
    results = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([f"[Source: {Path(doc.metadata.get('source', 'Doc')).name}]\n{doc.page_content}" for doc in results])
    return context

@tool
def calculate_mexican_customs_duties(
    invoice_value_usd: float,
    freight_usd: float,
    insurance_usd: float,
    exchange_rate_mxn: float,
    ige_duty_rate: float,
    has_usmca_certificate: bool,
    apply_border_vat: bool = False
) -> dict:
    """
    Calculates full pedimento duty structure into Mexican Pesos (MXN):
    CIF Base Value, Ad-Valorem Duty (IGE), Customs Processing Fee (DTA), Prevalidation (PRV), 
    Value Added Tax (IVA 16% or 8% border rate), and Total Payable Taxes.
    """
    # Convert to MXN
    cif_usd = invoice_value_usd + freight_usd + insurance_usd
    cif_mxn = cif_usd * exchange_rate_mxn
    
    # 1. Impuesto General de Importación (IGE)
    effective_ige_rate = 0.0 if has_usmca_certificate else ige_duty_rate
    ige_amount = cif_mxn * effective_ige_rate
    
    # 2. Derecho de Trámite Aduanero (DTA) - Art. 49 LFF
    # Preferential fixed rate under USMCA vs standard 8 per mille (0.008)
    dta_amount = 410.0 if has_usmca_certificate else max(410.0, cif_mxn * 0.008)
    
    # 3. Prevalidación (PRV) - Fixed administrative fee (~$310 MXN base average)
    prv_amount = 310.0
    
    # 4. Impuesto al Valor Agregado (IVA)
    vat_rate = 0.08 if apply_border_vat else 0.16
    taxable_base_iva = cif_mxn + ige_amount + dta_amount
    iva_amount = taxable_base_iva * vat_rate
    
    total_duties_mxn = ige_amount + dta_amount + prv_amount + iva_amount
    
    return {
        "cif_value_usd": round(cif_usd, 2),
        "cif_value_mxn": round(cif_mxn, 2),
        "ige_duty_mxn": round(ige_amount, 2),
        "dta_fee_mxn": round(dta_amount, 2),
        "prv_fee_mxn": round(prv_amount, 2),
        "iva_vat_mxn": round(iva_amount, 2),
        "total_payable_taxes_mxn": round(total_duties_mxn, 2)
    }

@tool
def generate_usmca_pdf_report(title: str, executive_summary: str, cost_breakdown: str, document_checklist: str) -> str:
    """
    Generates an official downloadable USMCA trade compliance dictamen and checklist PDF.
    Always trigger this tool when a user explicitly requests a report, PDF, or formal assessment.
    """
    pdf_filename = "USMCA_Customs_Compliance_Report.pdf"
    pdf_path = Path(pdf_filename)
    
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1E40AF'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = styles['BodyText']
    body_style.fontSize = 9
    body_style.spaceAfter = 6

    story = []
    lang = st.session_state.get('lang', 'English')
    
    sub_title = "Dictamen de Cumplimiento de Comercio Exterior T-MEC" if lang == "Español" else "USMCA / T-MEC Trade Compliance Technical Assessment"
    sec1 = "1. Resumen Ejecutivo y Base Legal" if lang == "Español" else "1. Executive Summary & Legal Basis"
    sec2 = "2. Estimación de Contribuciones y Gravámenes (MXN)" if lang == "Español" else "2. Estimated Duty & Tax Assessment (MXN)"
    sec3 = "3. Expediente Digital y VUCEM Checklist" if lang == "Español" else "3. Digital Audit & VUCEM Document Checklist"

    story.append(Paragraph(f"<b>{title}</b>", title_style))
    story.append(Paragraph(f"<b>{sub_title}</b>", styles['Normal']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(sec1, heading_style))
    story.append(Paragraph(executive_summary.replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(sec2, heading_style))
    story.append(Paragraph(cost_breakdown.replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(sec3, heading_style))
    story.append(Paragraph(document_checklist.replace('\n', '<br/>'), body_style))
    
    doc.build(story)
    
    st.session_state['latest_pdf_generated'] = str(pdf_path)
    return f"PDF report successfully generated: {pdf_filename}"

tools = [search_usmca_trade_regulations, calculate_mexican_customs_duties, generate_usmca_pdf_report]

# ==============================================================================
# 3. Agent Core setup
# ==============================================================================
@st.cache_resource
def get_agent_executor():
    api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
    if not api_key:
        st.error("NVIDIA_API_KEY not found in environment or secrets.")
        st.stop()
        
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.1,
        api_key=api_key
    )
    
    system_prompt = (
        "You are an expert Autonomous Customs & Trade Agent specializing in Mexican Customs Law (Ley Aduanera), "
        "TIGIE, VUCEM operations, and USMCA/T-MEC preferential rules of origin.\n"
        "RULES:\n"
        "1. LANGUAGE MATCHING: Match the language of the user query exactly (English or Spanish).\n"
        "2. ACCURACY: Always utilize `calculate_mexican_customs_duties` for tax estimates and `search_usmca_trade_regulations` for legal citations.\n"
        "3. FORMAL PDF: Call `generate_usmca_pdf_report` whenever requested to issue formal assessments."
    )
    
    return create_react_agent(llm, tools, prompt=system_prompt)

# ==============================================================================
# 4. Streamlit UI Interface
# ==============================================================================
st.set_page_config(layout="wide", page_title="USMCA Customs Agent", page_icon="🛃")

with st.sidebar:
    st.title("🛃 Operational Panel")
    selected_lang = st.radio("Language / Idioma", ["English", "Español"], index=0, key="lang_radio")
    st.session_state.lang = selected_lang
    
    st.markdown("---")
    st.markdown("**Developer:** Dr. Robert Hernández Martínez")
    st.markdown("**Affiliation:** UNAM / Trade AI Systems")
    st.markdown("---")
    
    if st.button("🔄 Index Custom Regulations"):
        with st.spinner("Indexing vector database from ./pdf_files_comercio_exterior..."):
            v_db = initialize_vector_db()
            if v_db:
                st.success("Vector DB successfully initialized!")
            else:
                st.warning("No PDFs found to index.")

# Main Layout
st.title("📦 USMCA / T-MEC Autonomous Customs Desk")
st.caption("AI-Powered Compliance, Tariff Calculation, and Audit Automation for Mexican Customs Brokers and Importers")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if 'latest_pdf_generated' in st.session_state and os.path.exists(st.session_state['latest_pdf_generated']):
    pdf_path = st.session_state['latest_pdf_generated']
    with open(pdf_path, "rb") as file:
        st.download_button(
            label="📄 Download Technical Dictamen (PDF)",
            data=file,
            file_name="USMCA_Compliance_Report.pdf",
            mime="application/pdf",
            key="pdf_download_top"
        )

if prompt := st.chat_input("Enter shipment details, ask about rules of origin, or request a pedimento calculation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            agent_executor = get_agent_executor()
            with st.status("Evaluating trade regulation compliance...", expanded=True) as status:
                response = agent_executor.invoke({"messages": [("user", prompt)]})
                status.update(label="Complete", state="complete", expanded=False)

            ai_texts = [msg.content for msg in response["messages"] if msg.type == "ai" and msg.content]
            final_answer = "\n\n".join(ai_texts) if ai_texts else "Analysis complete."

            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            st.error(f"Error executing agent workflow: {str(e)}")