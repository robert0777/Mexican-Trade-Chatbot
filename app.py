import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# LangChain, LangGraph & Model Imports
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

INDEX_PATH = "./faiss_index_trade"
PDF_DIR = "./pdf_files_comercio_exterior"

# ==============================================================================
# 1. High-Speed Local FAISS Vector Database Engine
# ==============================================================================
@st.cache_resource(show_spinner=False)
def initialize_trade_vector_db(force_reindex: bool = False):
    """Builds or loads a FAISS vector store from disk with truncation protection."""
    api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
    
    # FIX 1: Set truncate="END" so inputs over 512 tokens are truncated instead of throwing 400 Bad Request
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5", 
        api_key=api_key,
        truncate="END"
    )
    
    # 1. Load existing local index if available
    if Path(INDEX_PATH).exists() and not force_reindex:
        try:
            return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            pass # Rebuild index if loading fails
            
    # 2. Build index if missing or requested
    pdf_dir = Path(PDF_DIR)
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        return None

    loader = PyPDFDirectoryLoader(path=str(pdf_dir), silent_errors=True)
    docs = loader.load()
    if not docs:
        return None

    # FIX 2: Reduced chunk_size from 600 to 400 characters to stay comfortably under 512 tokens
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore

# ==============================================================================
# 2. Autonomous Trade Agent Tools
# ==============================================================================
@tool
def search_trade_regulations(query: str) -> str:
    """
    Searches loaded trade PDFs (HTS 2026, LIGIE/TIGIE, Ley Aduanera, Anexo 22, 
    USMCA rules of origin, MOA, CFF, and Ley del IVA) for legal grounding.
    """
    vectorstore = initialize_trade_vector_db()
    if not vectorstore:
        return "No trade regulation PDFs found in memory."
    
    results = vectorstore.similarity_search(query, k=2)
    return "\n\n".join([f"[Source: {Path(d.metadata.get('source', 'Doc')).name}]\n{d.page_content}" for d in results])

@tool
def calculate_mexican_import_duties(
    invoice_value_usd: float,
    freight_usd: float = 0.0,
    insurance_usd: float = 0.0,
    exchange_rate_mxn: float = 20.0,
    ige_duty_rate: float = 0.0,
    has_usmca_certificate: bool = True,
    apply_border_vat: bool = False
) -> dict:
    """Calculates Mexican customs duties (Pedimento structure) in MXN for any imported good."""
    cif_usd = invoice_value_usd + freight_usd + insurance_usd
    cif_mxn = cif_usd * exchange_rate_mxn
    
    effective_ige = 0.0 if has_usmca_certificate else ige_duty_rate
    ige_amount = cif_mxn * effective_ige
    dta_amount = 410.0 if has_usmca_certificate else max(410.0, cif_mxn * 0.008)
    prv_amount = 310.0
    
    vat_rate = 0.08 if apply_border_vat else 0.16
    taxable_base_iva = cif_mxn + ige_amount + dta_amount
    iva_amount = taxable_base_iva * vat_rate
    
    total_taxes_mxn = ige_amount + dta_amount + prv_amount + iva_amount
    
    return {
        "cif_value_usd": round(cif_usd, 2),
        "cif_value_mxn": round(cif_mxn, 2),
        "ige_duty_mxn": round(ige_amount, 2),
        "dta_fee_mxn": round(dta_amount, 2),
        "prv_fee_mxn": round(prv_amount, 2),
        "iva_vat_mxn": round(iva_amount, 2),
        "total_payable_taxes_mxn": round(total_taxes_mxn, 2)
    }

@tool
def calculate_us_import_duties(
    invoice_value_usd: float,
    freight_usd: float = 0.0,
    insurance_usd: float = 0.0,
    hts_code: str = "0000.00.00",
    standard_mfn_rate: float = 0.0,
    has_usmca_certificate: bool = True
) -> dict:
    """Calculates US Customs (CBP) entry duties and fees for any product entering the USA."""
    entered_value = invoice_value_usd + freight_usd + insurance_usd
    duty_rate = 0.0 if has_usmca_certificate else standard_mfn_rate  
    duty_amount = entered_value * duty_rate
    
    mpf_fee = 0.0 if has_usmca_certificate else max(31.67, min(614.35, entered_value * 0.003464))
    total_duties = duty_amount + mpf_fee
    
    return {
        "entered_value_usd": round(entered_value, 2),
        "hts_code": hts_code,
        "usmca_duty_rate": "0.0%" if has_usmca_certificate else f"{duty_rate*100}%",
        "customs_duty_usd": round(duty_amount, 2),
        "usmca_mpf_exemption": has_usmca_certificate,
        "mpf_fee_usd": round(mpf_fee, 2),
        "total_us_duties_usd": round(total_duties, 2)
    }

tools = [search_trade_regulations, calculate_mexican_import_duties, calculate_us_import_duties]

# ==============================================================================
# 3. Agent Configuration
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
        max_tokens=2048,
        timeout=180,
        api_key=api_key
    )
    
    system_prompt = (
        "You are an expert Universal Autonomous Customs & Trade Agent.\n\n"
        "OPERATIONAL INSTRUCTIONS:\n"
        "1. LANGUAGE: Respond strictly in the language used by the user (English or Spanish).\n"
        "2. DIRECTIONAL LOGIC:\n"
        "   - For exports to the US: Use `calculate_us_import_duties`.\n"
        "   - For imports into Mexico: Use `calculate_mexican_import_duties`.\n"
        "3. KNOWLEDGE LOOKUP: Use `search_trade_regulations` to verify specific HTS/TIGIE codes or permits.\n"
        "4. CHAT OUTPUT: Present clear Markdown summaries with duty breakdowns, HTS codes, and required documents."
    )
    
    return create_react_agent(llm, tools, prompt=system_prompt)

# ==============================================================================
# 4. Streamlit User Interface
# ==============================================================================
st.set_page_config(layout="wide", page_title="Universal USMCA Customs Desk", page_icon="🛃")

with st.sidebar:
    st.title("🛃 Operational Panel")
    selected_lang = st.radio("Interface Language / Idioma", ["English", "Español"], index=0, key="lang_radio")
    st.session_state.lang = selected_lang
    st.markdown("---")
    st.markdown("**Author:** Dr. Robert Hernández Martínez")
    st.markdown("---")
    
    if st.button("🔄 Re-Index Regulations"):
        with st.spinner("Building local FAISS disk index with token protections..."):
            v_db = initialize_trade_vector_db(force_reindex=True)
            if v_db:
                st.success("FAISS index saved locally to disk!")
            else:
                st.warning("No PDFs found in ./pdf_files_comercio_exterior")

st.title("📦 Universal USMCA / T-MEC Autonomous Trade Desk")
st.caption("High-Speed AI Agent for Duty Calculation, Tariff Classification & Regulatory Compliance")

# Auto-initialize FAISS from disk on launch
initialize_trade_vector_db()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Interactive Chat Loop with Live Streaming
if prompt := st.chat_input("Ask about any product import/export, calculate duties, or verify USMCA rules of origin..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        agent = get_trade_agent()
        response_placeholder = st.empty()
        
        try:
            with st.status("Evaluating trade regulations...", expanded=True) as status:
                full_response = ""
                for event in agent.stream({"messages": [("user", prompt)]}, stream_mode="values"):
                    messages = event.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                st.write(f"🛠️ **Executing Tool:** `{tc['name']}`")
                        elif last_msg.type == "ai" and last_msg.content and not getattr(last_msg, "tool_calls", None):
                            if isinstance(last_msg.content, str):
                                full_response = last_msg.content
                                response_placeholder.markdown(full_response)
                
                status.update(label="Analysis complete", state="complete", expanded=False)

            if not full_response:
                full_response = "USMCA trade analysis completed."
                response_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")