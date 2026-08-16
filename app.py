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
    """Selects the top matching text chunks based on word overlap relevance."""
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
# 2. Autonomous Agentic Tools (Bilingual Capabilities)
# ==============================================================================
@tool
def buscar_regulaciones_aduaneras(consulta: str) -> str:
    """
    Search in loaded PDF documents for Mexican customs regulations, TIGIE tariffs, NOMs, and trade agreements (USMCA/T-MEC, EU-Mexico/TLCUEM).
    Busca en los documentos PDF cargados regulaciones arancelarias, TIGIE, NOMs y tratados (T-MEC, TLCUEM).
    """
    if "documents" not in st.session_state or not st.session_state.documents:
        return "No local PDF documents loaded in memory. / No hay documentos PDF cargados."
    
    chunks = select_relevant_chunks(consulta, st.session_state.documents)
    if not chunks:
        return "No relevant text chunks found in local docs. / No se encontraron fragmentos relevantes."
        
    context = "\n\n".join([f"[Doc: {Path(c.metadata['source']).name}]\n{c.page_content}" for c in chunks])
    return context

@tool
def calcular_impuestos_importacion(valor_factura: float, flete: float, seguro: float, tasa_ige: float, tiene_certificado_origen: bool) -> dict:
    """
    Calculates CIF base value, ad-valorem duty (IGE), customs handling fee (DTA), Value Added Tax (IVA 16%), and total landed import cost in Mexico.
    Calcula la base gravable CIF, el arancel IGE, el DTA, el IVA (16%) y el costo total estimado de importación en México.
    """
    cif = valor_factura + flete + seguro
    effective_ige = 0.0 if tiene_certificado_origen else tasa_ige
    ige_amount = cif * effective_ige
    dta_amount = 410.0 if tiene_certificado_origen else (cif * 0.008)
    iva_amount = (cif + ige_amount + dta_amount) * 0.16
    total_landed = cif + ige_amount + dta_amount + iva_amount
    
    return {
        "valor_cif_cif_value": round(cif, 2),
        "monto_ige_duty": round(ige_amount, 2),
        "monto_dta_customs_fee": round(dta_amount, 2),
        "monto_iva_vat": round(iva_amount, 2),
        "costo_total_estimado_total_landed_cost": round(total_landed, 2)
    }

@tool
def exportar_dictamen_pdf(titulo: str, resumen_ejecutivo: str, desglose_costos: str, requisitos_documentales: str) -> str:
    """
    Generates a downloadable official PDF compliance report and document checklist in English or Spanish.
    Genera un reporte oficial descargable en formato PDF con el dictamen de comercio exterior y la lista de verificación.
    """
    pdf_filename = "Customs_Compliance_Report.pdf"
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
    lang = st.session_state.get('lang', 'Español')
    
    # Subtitle adaptation
    sub_title = "Agente Inteligente de Comercio Exterior y Aduanas" if lang == "Español" else "Intelligent Customs & Foreign Trade Agent"
    sec1 = "1. Resumen Ejecutivo y Dictamen" if lang == "Español" else "1. Executive Summary & Assessment"
    sec2 = "2. Desglose Estimado de Impuestos y Gastos (CIF/IVA)" if lang == "Español" else "2. Estimated Duty & Tax Breakdown (CIF/VAT)"
    sec3 = "3. Lista de Verificación Documental para Pedimento" if lang == "Español" else "3. Customs Clearance Document Checklist"

    # Title & Header
    story.append(Paragraph(f"<b>{titulo}</b>", title_style))
    story.append(Paragraph(f"<b>{sub_title}</b>", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Sections
    story.append(Paragraph(sec1, heading_style))
    story.append(Paragraph(resumen_ejecutivo.replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(sec2, heading_style))
    story.append(Paragraph(desglose_costos.replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(sec3, heading_style))
    story.append(Paragraph(requisitos_documentales.replace('\n', '<br/>'), body_style))
    
    doc.build(story)
    
    st.session_state['latest_pdf_generated'] = str(pdf_path)
    return f"PDF report successfully generated: {pdf_filename}"

tools = [buscar_regulaciones_aduaneras, calcular_impuestos_importacion, exportar_dictamen_pdf]

# ==============================================================================
# 3. Agent Executor Initialization (With Bilingual Prompt Instruction)
# ==============================================================================
@st.cache_resource
def get_agent_executor():
    api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
    if not api_key:
        st.error("NVIDIA_API_KEY not configured. Add it to .env or st.secrets.")
        st.stop()
        
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.2,
        api_key=api_key,
        timeout=180,  
        max_retries=3  
    )
    
    # System prompt enforcing language auto-detection and execution guidance
    system_prompt = (
        "You are an expert Autonomous Trade & Customs Agent for Mexican international trade (USMCA, EU-Mexico).\n"
        "1. LANGUAGE: Automatically detect the user's language (English or Spanish) and respond fully in that language.\n"
        "2. REASONING: Use tools autonomously to search regulations, compute import duties, or export PDF reports.\n"
        "3. PRECISION: Provide explicit references to tariff rules (TIGIE), NOM safety standards, or certificate of origin benefits."
    )
    
    return create_react_agent(llm, tools, prompt=system_prompt)

# ==============================================================================
# 4. Streamlit User Interface & Localization
# ==============================================================================
st.set_page_config(layout="wide", page_title="Customs Agent AI / Agente de Aduanas", page_icon="📦")

# Sidebar - Language Selection & Settings
with st.sidebar:
    st.markdown("### 🌐 Language / Idioma")
    selected_lang = st.radio("Select Language / Seleccione Idioma", ["Español", "English"], index=0, key="lang_radio")
    st.session_state.lang = selected_lang
    
    st.markdown("---")
    st.markdown("### ⚙️ " + ("Configuración" if st.session_state.lang == "Español" else "Settings"))
    st.markdown("**Author / Autor:** Dr. Robert Hernández Martínez")
    st.markdown("---")
    
    btn_label = "🔄 Cargar / Actualizar PDFs Locales" if st.session_state.lang == "Español" else "🔄 Load / Refresh Local PDFs"
    if st.button(btn_label):
        with st.spinner("Processing ./pdf_files_comercio_exterior..."):
            load_documents()
            num_chunks = len(st.session_state.get('documents', []))
            if num_chunks > 0:
                msg = f"Cargados {num_chunks} fragmentos de documentos." if st.session_state.lang == "Español" else f"Loaded {num_chunks} document chunks."
                st.success(msg)
            else:
                msg = "No se encontraron PDFs en ./pdf_files_comercio_exterior" if st.session_state.lang == "Español" else "No PDFs found in ./pdf_files_comercio_exterior"
                st.warning(msg)

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.8rem; color: #6B7280; text-align: center;">
            © 2026 Customs Agent AI MX
        </div>
    """, unsafe_allow_html=True)

# UI Text Dictionaries
if st.session_state.lang == "Español":
    title_text = "📦 Agente Autónomo de Comercio Exterior y Aduanas"
    desc_text = """
    > **¿Qué hace diferente a este Agente de un Chatbot tradicional?**  
    > 1. **Razonamiento y Planificación**: Analiza objetivos y decide qué pasos ejecutar de forma autónoma en **Inglés o Español**.  
    > 2. **Ejecución de Herramientas**: Consulta leyes/tratados en PDF, calcula impuestos (*CIF, IGE, DTA, IVA*) y **genera reportes PDF**.  
    > 3. **Visibilidad en Tiempo Real**: Muestra el razonamiento, observaciones y llamada a herramientas durante el proceso.
    """
    input_placeholder = "Describa su embarque o solicite un dictamen en PDF (Español o Inglés)..."
    status_label = "🧠 El Agente está evaluando la consulta y ejecutando herramientas..."
    tool_exec_label = "🛠️ **Ejecutando Herramienta:**"
    tool_obs_label = "👁️ **Observación de Herramienta:**"
    result_header = "### 📝 Dictamen y Plan de Acción:"
    download_btn_label = "📄 Descargar Dictamen Oficial en PDF"
else:
    title_text = "📦 Autonomous Customs & Foreign Trade Agent"
    desc_text = """
    > **What sets this Agent apart from a standard Chatbot?**  
    > 1. **Reasoning & Planning**: Autonomously breaks down goals and plans execution steps in **English or Spanish**.  
    > 2. **Tool Execution**: Searches PDF legal texts, computes landed duty/taxes (*CIF, IGE, DTA, VAT*), and **generates PDF compliance reports**.  
    > 3. **Real-time Transparency**: Displays tool invocations, inputs, and intermediate observations as it works.
    """
    input_placeholder = "Describe your shipment or request a PDF compliance report (English or Spanish)..."
    status_label = "🧠 Agent is analyzing query and invoking tools..."
    tool_exec_label = "🛠️ **Executing Tool:**"
    tool_obs_label = "👁️ **Tool Observation:**"
    result_header = "### 📝 Compliance Assessment & Action Plan:"
    download_btn_label = "📄 Download Official PDF Report"

# Header Render
st.title(title_text)
st.markdown(desc_text)

# Initialize Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
                status.update(label=f"Done in {elapsed:.2f}s / Completado en {elapsed:.2f}s", state="complete", expanded=False)

            # Render Final Agent Response
            final_answer = response["messages"][-1].content
            st.markdown(result_header)
            st.markdown(final_answer)
            
            # Display PDF Download Button if generated
            if 'latest_pdf_generated' in st.session_state and os.path.exists(st.session_state['latest_pdf_generated']):
                pdf_path = st.session_state['latest_pdf_generated']
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label=download_btn_label,
                        data=file,
                        file_name="Customs_Compliance_Report.pdf",
                        mime="application/pdf"
                    )

            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            st.error(f"Execution Error / Error de Ejecución: {str(e)}")