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

def normalize_spanish_text(text: str) -> str:
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
                chunk_size=500, 
                chunk_overlap=50, 
                length_function=count_tokens
            )
            for doc in docs:
                doc.page_content = normalize_spanish_text(doc.page_content)
            st.session_state.documents = text_splitter.split_documents(docs)
        else:
            st.session_state.documents = []

def select_relevant_chunks(question: str, chunks: list, max_total_tokens: int = 4000) -> list:
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
# 2. Autonomous Agentic Tools
# ==============================================================================
@tool
def buscar_regulaciones_aduaneras(consulta: str) -> str:
    """Busca en los documentos PDF cargados regulaciones arancelarias, TIGIE, NOMs y tratados (T-MEC, TLCUEM)."""
    if "documents" not in st.session_state or not st.session_state.documents:
        return "No hay documentos PDF cargados en la base de datos local."
    
    chunks = select_relevant_chunks(consulta, st.session_state.documents)
    if not chunks:
        return "No se encontraron fragmentos relevantes en la base de datos de documentos."
        
    context = "\n\n".join([f"[Doc: {Path(c.metadata['source']).name}]\n{c.page_content}" for c in chunks])
    return context

@tool
def calcular_impuestos_importacion(valor_factura: float, flete: float, seguro: float, tasa_ige: float, tiene_certificado_origen: bool) -> dict:
    """Calcula la base gravable CIF, el arancel IGE, el DTA, el IVA (16%) y el costo total estimado de importación en México."""
    cif = valor_factura + flete + seguro
    effective_ige = 0.0 if tiene_certificado_origen else tasa_ige
    ige_amount = cif * effective_ige
    dta_amount = 410.0 if tiene_certificado_origen else (cif * 0.008)
    iva_amount = (cif + ige_amount + dta_amount) * 0.16
    total_landed = cif + ige_amount + dta_amount + iva_amount
    
    return {
        "valor_cif": round(cif, 2),
        "monto_ige": round(ige_amount, 2),
        "monto_dta": round(dta_amount, 2),
        "monto_iva": round(iva_amount, 2),
        "costo_total_estimado": round(total_landed, 2)
    }

@tool
def exportar_dictamen_pdf(titulo: str, resumen_ejecutivo: str, desglose_costos: str, requisitos_documentales: str) -> str:
    """Genera un reporte oficial descargable en formato PDF con el dictamen de comercio exterior y la lista de verificación."""
    pdf_filename = "Dictamen_Comercio_Exterior.pdf"
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
    
    # Title & Header
    story.append(Paragraph(f"<b>{titulo}</b>", title_style))
    story.append(Paragraph("<b>Agente Inteligente de Comercio Exterior y Aduanas</b>", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Resumen
    story.append(Paragraph("1. Resumen Ejecutivo y Dictamen", heading_style))
    story.append(Paragraph(resumen_ejecutivo.replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 10))
    
    # Costos
    story.append(Paragraph("2. Desglose Estimado de Impuestos y Gastos (CIF/IVA)", heading_style))
    story.append(Paragraph(desglose_costos.replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 10))
    
    # Requisitos Documentales
    story.append(Paragraph("3. Lista de Verificación Documental para Pedimento", heading_style))
    story.append(Paragraph(requisitos_documentales.replace('\n', '<br/>'), body_style))
    
    doc.build(story)
    
    # Flag in session state to trigger downloadable widget
    st.session_state['latest_pdf_generated'] = str(pdf_path)
    return f"Reporte PDF generado exitosamente con el nombre {pdf_filename}."

tools = [buscar_regulaciones_aduaneras, calcular_impuestos_importacion, exportar_dictamen_pdf]

# ==============================================================================
# 3. Agent Executor Initialization
# ==============================================================================
@st.cache_resource
def get_agent_executor():
    api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
    if not api_key:
        st.error("NVIDIA_API_KEY no configurada. Agréguela al archivo .env o a st.secrets.")
        st.stop()
        
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature=0.2,
        api_key=api_key
    )
    return create_react_agent(llm, tools)

# ==============================================================================
# 4. Streamlit User Interface
# ==============================================================================
st.set_page_config(layout="wide", page_title="Agente de Comercio Exterior", page_icon="📦")

# Header & Callout
st.title("📦 Agente Autónomo de Comercio Exterior y Aduanas")
st.markdown("""
> **¿Qué hace diferente a este Agente de un Chatbot tradicional?**  
> 1. **Razonamiento y Planificación**: Analiza objetivos y decide qué pasos ejecutar de forma autónoma.  
> 2. **Ejecución de Herramientas**: Consulta leyes/tratados en PDF, calcula impuestos (*CIF, IGE, DTA, IVA*) y **genera reportes PDF**.  
> 3. **Visibilidad en Tiempo Real**: Muestra el razonamiento, las observaciones y la llamada a herramientas durante el proceso.
""")

# Sidebar Configuration & Document Processing
with st.sidebar:
    st.markdown("### ⚙️ Configuración del Agente")
    st.markdown("**Autor:** Dr. Robert Hernández Martínez")
    st.markdown("---")
    
    if st.button("🔄 Cargar / Actualizar PDFs Locales"):
        with st.spinner("Cargando documentos de ./pdf_files_comercio_exterior..."):
            load_documents()
            num_chunks = len(st.session_state.get('documents', []))
            if num_chunks > 0:
                st.success(f"Cargados {num_chunks} fragmentos de documentos.")
            else:
                st.warning("No se encontraron archivos PDF en la carpeta ./pdf_files_comercio_exterior")

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.8rem; color: #6B7280; text-align: center;">
            © 2026 Agente de Comercio Exterior MX
        </div>
    """, unsafe_allow_html=True)

# Initialize Session Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Loop
if prompt := st.chat_input("Describa su embarque o solicite un dictamen en PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            agent_executor = get_agent_executor()
            
            with st.status("🧠 El Agente está evaluando la consulta y ejecutando herramientas...", expanded=True) as status:
                start_time = time.time()
                response = agent_executor.invoke({"messages": [("user", prompt)]})
                
                # Render intermediate tool execution steps
                for message in response["messages"]:
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            st.write(f"🛠️ **Ejecutando Herramienta:** `{tool_call['name']}`")
                            st.json(tool_call['args'])
                    elif message.type == "tool":
                        st.write("👁️ **Observación de Herramienta:**")
                        st.caption(str(message.content)[:300] + "...")
                        
                elapsed = time.time() - start_time
                status.update(label=f"Análisis completado en {elapsed:.2f} segundos", state="complete", expanded=False)

            # Render Final Agent Response
            final_answer = response["messages"][-1].content
            st.markdown("### 📝 Dictamen y Plan de Acción:")
            st.markdown(final_answer)
            
            # Display PDF Download Button if generated
            if 'latest_pdf_generated' in st.session_state and os.path.exists(st.session_state['latest_pdf_generated']):
                pdf_path = st.session_state['latest_pdf_generated']
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="📄 Descargar Dictamen Oficial en PDF",
                        data=file,
                        file_name="Dictamen_Comercio_Exterior.pdf",
                        mime="application/pdf"
                    )

            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            st.error(f"Error durante la ejecución del Agente: {str(e)}")