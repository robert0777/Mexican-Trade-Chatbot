import streamlit as st
import os
import re
import time
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

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
# 2. Autonomous Agentic Tools
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

tools = [search_usmca_trade_regulations, calculate_usmca_import_duties]

# ==============================================================================
# 3. Agent Executor Initialization
# ==============================================================================
@st.cache_resource
def get_agent_executor():
    api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
    if not api_key:
        st.error("NVIDIA_API_KEY not configured.")
        st.stop()
        
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.1,        
        api_key=api_key,
        timeout=180,            
        max_retries=3  
    )
    
    system_prompt = (
        "You are an expert Autonomous Trade Agent focusing strictly on USMCA / T-MEC trade.\n"
        "1. LANGUAGE: Respond strictly in the language used by the user (English or Spanish).\n"
        "2. LEGAL CONTEXT: Explicitly cite applicable legal frameworks (e.g., Ley Aduanera, TIGIE, USMCA Chapter 5 Rules of Origin, NOMs, USDA/APHIS, SENASICA).\n"
        "3. CALCULATIONS: Always invoke `calculate_usmca_import_duties` when financial values are available. If freight/insurance are not given, assume reasonable defaults or $0 and note it.\n"
        "4. INPUT FORMAT TEMPLATE: Provide a clear structured data input template when asking the user for missing calculation inputs.\n"
        "5. RECOMMENDATIONS: End every response with clear, step-by-step recommendations on how to proceed with the export/import procedure."
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

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.8rem; color: #6B7280; text-align: center;">
            © 2026 USMCA Agent AI
        </div>
    """, unsafe_allow_html=True)

# Localization Setup
if st.session_state.lang == "English":
    title_text = "📦 USMCA / T-MEC Autonomous Trade Agent"
    desc_text = """
    > **Specialized USMCA (US-Mexico-Canada) Compliance & Calculations Assistant**  
    > 1. **Legal Framework**: Contextualized rules (*Ley Aduanera, TIGIE, USMCA Chapter 5, NOMs, USDA/APHIS*).  
    > 2. **Duty Calculations**: Automatic landed-cost estimation (*CIF, IGE, DTA, VAT*).  
    > 3. **Actionable Recommendations**: Step-by-step guidance for trade operations.
    """
    input_placeholder = "Describe your USMCA shipment..."
    status_label = "🧠 Agent is analyzing query and invoking USMCA tools..."
    tool_exec_label = "🛠️ **Executing Tool:**"
    tool_obs_label = "👁️ **Tool Observation:**"
    result_header = "### 📝 Compliance Assessment & Action Plan:"
    template_header = "💡 **Sample Input Template for Tax Calculations**"
    template_code = """Invoice Value: $50,000 USD
Freight: $2,500 USD
Insurance: $500 USD
Tariff Rate (IGE): 0%
Has USMCA Certificate of Origin: Yes"""
else:
    title_text = "📦 Agente Autónomo de Comercio Exterior T-MEC"
    desc_text = """
    > **Asistente Especializado en Cumplimiento y Cálculos T-MEC (México-EUA-Canadá)**  
    > 1. **Marco Legal**: Contexto regulatorio (*Ley Aduanera, TIGIE, Capítulo 5 T-MEC, NOMs, SENASICA*).  
    > 2. **Cálculo de Impuestos**: Estimación de costos CIF, IGE, DTA e IVA.  
    > 3. **Recomendaciones**: Pasos concretos para proceder con su trámite aduanal.
    """
    input_placeholder = "Describa su embarque T-MEC..."
    status_label = "🧠 El Agente está evaluando la consulta y ejecutando herramientas..."
    tool_exec_label = "🛠️ **Ejecutando Herramienta:**"
    tool_obs_label = "👁️ **Observación de Herramienta:**"
    result_header = "### 📝 Dictamen y Plan de Acción:"
    template_header = "💡 **Plantilla de Ejemplo para Cálculo de Impuestos**"
    template_code = """Valor Factura: $50,000 USD
Flete: $2,500 USD
Seguro: $500 USD
Tasa Arancelaria (IGE): 0%
Cuenta con Certificado de Origen T-MEC: Sí"""

st.title(title_text)
st.markdown(desc_text)

# Interactive Expander: Sample Format Template
with st.expander(template_header):
    st.code(template_code, language="yaml")

# Initialize Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Message History
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
                        safe_content = str(message.content).replace("<", "&lt;").replace(">", "&gt;")
                        st.caption(safe_content[:300] + "...")
                        
                elapsed = time.time() - start_time
                status.update(label=f"Done in {elapsed:.2f}s", state="complete", expanded=False)

            # Robust Response Extraction: Collect text across all AI turns
            ai_texts = []
            for msg in response["messages"]:
                if msg.type == "ai":
                    if isinstance(msg.content, str) and msg.content.strip():
                        clean_text = msg.content.strip().replace("<TOOL_CALLS>", "").replace("</TOOL_CALLS>", "")
                        if clean_text:
                            ai_texts.append(clean_text)
                    elif isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                txt = block.get("text", "").replace("<TOOL_CALLS>", "").replace("</TOOL_CALLS>", "")
                                if txt:
                                    ai_texts.append(txt)

            # Join non-empty AI text blocks
            final_answer = "\n\n".join(ai_texts) if ai_texts else "Analysis complete."

            st.markdown(result_header)
            st.markdown(final_answer)

            # Interactive Quick Copy Box
            st.markdown("---")
            st.caption("📋 **Quick Copy Response Content**")
            st.code(final_answer, language="markdown")

            # Append message to history
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")