import streamlit as st
import os
import re
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

# ==============================================================================
# 1. FAISS Vector Database Engine (RAG)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def initialize_trade_vector_db():
    """Builds or loads a FAISS vector store for trade regulations and USMCA laws."""
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
# 2. Autonomous Trade Agent Tools (Direct UI Calculations)
# ==============================================================================
@tool
def search_trade_regulations(query: str) -> str:
    """
    Searches loaded trade PDFs regarding Ley Aduanera, TIGIE tariffs, HTS classifications,
    Anexo 22, USMCA/T-MEC rules of origin, and agricultural/NOM compliance standards.
    """
    vectorstore = initialize_trade_vector_db()
    if not vectorstore:
        return "No local trade regulation PDFs loaded in memory."
    
    results = vectorstore.similarity_search(query, k=4)
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
    """
    Calculates Mexican customs duties (Pedimento structure) in MXN:
    CIF Base, Ad-Valorem Duty (IGE), Customs Processing Fee (DTA), 
    Prevalidation (PRV), Value Added Tax (IVA 16% or 8%), and Total Payable Taxes.
    """
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
    hts_code: str = "0804.40.00",
    has_usmca_certificate: bool = True
) -> dict:
    """
    Calculates US Customs (CBP) entry duties and fees for exports entering the USA.
    Handles USMCA preferential rate (0% for originating goods) and Merchandise Processing Fee (MPF) exemption.
    """
    entered_value = invoice_value_usd + freight_usd + insurance_usd
    duty_rate = 0.0 if has_usmca_certificate else 0.112  
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

tools = [
    search_trade_regulations, 
    calculate_mexican_import_duties, 
    calculate_us_import_duties
]

# ==============================================================================
# 3. ReAct Agent Core Setup
# ==============================================================================
@st.cache_resource
def get_trade_agent():
    api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
    if not api_key:
        st.error("NVIDIA_API_KEY not configured. Please add it to environment or st.secrets.")
        st.stop()
        
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.1,
        max_tokens=2048,
        timeout=180,  # Extended socket timeout to eliminate HTTPS Read Timeout errors
        api_key=api_key
    )
    
    system_prompt = (
        "You are an expert Autonomous Customs & Trade Agent specializing in USMCA/T-MEC regulations, "
        "Mexican Customs Law (Ley Aduanera), US Customs (CBP) entries, and agricultural export rules.\n\n"
        "OPERATIONAL RULES:\n"
        "1. LANGUAGE: Match the user's input language (English or Spanish) in all responses.\n"
        "2. CROSS-BORDER DIRECTION:\n"
        "   - Exports from Mexico to the US (e.g., Texas): Use `calculate_us_import_duties` to evaluate US CBP entry rules.\n"
        "   - Imports into Mexico: Use `calculate_mexican_import_duties` for Pedimento tax calculation.\n"
        "3. TOOL CALLING: Execute `search_trade_regulations` for legal validation.\n"
        "4. UI PRESENTATION: Present final calculations, duty breakdowns, HTS codes, and required compliance documents (USDA/SENASICA/FDA) directly in clear, beautifully formatted Markdown in the chat."
    )
    
    return create_react_agent(llm, tools, prompt=system_prompt)

# ==============================================================================
# 4. Streamlit Interactive Interface
# ==============================================================================
st.set_page_config(layout="wide", page_title="USMCA Customs Agent", page_icon="🛃")

with st.sidebar:
    st.title("🛃 Operational Panel")
    selected_lang = st.radio("Interface Language / Idioma", ["English", "Español"], index=0, key="lang_radio")
    st.session_state.lang = selected_lang
    
    st.markdown("---")
    st.markdown("**Author:** Dr. Robert Hernández Martínez")
    st.markdown("---")
    
    if st.button("🔄 Index Custom Regulations"):
        with st.spinner("Indexing vector database from ./pdf_files_comercio_exterior..."):
            v_db = initialize_trade_vector_db()
            if v_db:
                st.success("Vector DB successfully initialized!")
            else:
                st.warning("No PDFs found to index in ./pdf_files_comercio_exterior.")

st.title("📦 USMCA / T-MEC Autonomous Customs Desk")
st.caption("AI-Powered Compliance Engine and Cross-Border Duty Calculator")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input Loop
if prompt := st.chat_input("Enter shipment details, calculate cross-border duties, or inquire about rules of origin..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            agent = get_trade_agent()
            
            with st.status("Evaluating trade regulations and executing agent tools...", expanded=True) as status:
                start_time = time.time()
                response = agent.invoke({"messages": [("user", prompt)]})
                
                for message in response["messages"]:
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            st.write(f"🛠️ **Executing Tool:** `{tool_call['name']}`")
                            st.json(tool_call['args'])
                    elif message.type == "tool":
                        st.write("👁️ **Tool Observation:**")
                        st.caption(str(message.content)[:300] + "...")

                elapsed = time.time() - start_time
                status.update(label=f"Done in {elapsed:.2f}s", state="complete", expanded=False)

            # Extract narrative answer from the response
            final_answer = ""
            for msg in reversed(response["messages"]):
                if msg.type == "ai":
                    if isinstance(msg.content, str) and msg.content.strip():
                        if not getattr(msg, "tool_calls", None):
                            final_answer = msg.content.strip()
                            break
                    elif isinstance(msg.content, list):
                        text_parts = [b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text"]
                        if text_parts:
                            final_answer = "\n".join(text_parts).strip()
                            break

            if not final_answer:
                tool_outputs = [str(m.content) for m in response["messages"] if m.type == "tool"]
                final_answer = "### 📝 Execution Summary:\n\n" + "\n\n".join(tool_outputs) if tool_outputs else "USMCA trade assessment complete."

            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")